# ==============================================================================
# Imports
# ==============================================================================

# This is needed so that python finds the utils
import sys
sys.path.append("/home/gf332/miracle-compession/code")
sys.path.append("/homes/gf332/miracle-compession/code")

from imageio import imwrite

import argparse
import os, glob
import json
from tqdm import tqdm

# Needed for compression as the common source of randomness
from sobol_seq import i4_sobol_generate
from scipy.stats import norm

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tf.enable_eager_execution()

tfe = tf.contrib.eager
tfs = tf.contrib.summary
tfs_logger = tfs.record_summaries_every_n_global_steps

from architectures import ClicCNN, ClicLadderCNN, ClicLadderCNN2, ClicHyperVAECNN
from new_architectures import ClicNewLadderCNN
from utils import is_valid_file, setup_eager_checkpoints_and_restore
from load_data import load_and_process_image, create_random_crops, download_process_and_load_data

from compression import coded_sample, decode_sample

# ==============================================================================
# Predefined Stuff
# ==============================================================================

models = {
    "cnn": ClicCNN,
    "hyper_cnn": ClicHyperVAECNN,
    "ladder_cnn": ClicLadderCNN,
    "ladder_cnn2": ClicLadderCNN2,
    "new_ladder": ClicNewLadderCNN
}


optimizers = {
    "sgd": tf.train.GradientDescentOptimizer,
    "momentum": lambda lr:
                    tf.train.MomentumOptimizer(learning_rate=lr,
                                               momentum=0.9,
                                               use_nesterov=False),
    "adam": tf.train.AdamOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer
}

# ==============================================================================
# Auxiliary Functions
# ==============================================================================
def clic_input_fn(dataset, image_size=(256, 256), buffer_size=1000, batch_size=8):
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(lambda i: clic_parse_fn(i, image_size=image_size))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset

def clic_parse_fn(image, image_size=(316, 316)):
    return tf.image.random_crop(image, size=image_size + (3,))

def create_model(model_key, config):
    model = models[model_key]
    
    if model_key == "cnn":
        vae = model(prior=config["prior"],
                    likelihood=config["likelihood"],
                    padding="SAME_MIRRORED")

    elif model_key in ["hyper_cnn", "ladder_cnn", "ladder_cnn2"]:
        vae = model(latent_dist=config["prior"],
                    likelihood=config["likelihood"],
                    first_level_channels=config["first_level_channels"],
                    second_level_channels=config["second_level_channels"],
                    first_level_layers=config["first_level_layers"],
                    padding_first_level="SAME_MIRRORED",
                    padding_second_level="SAME_MIRRORED")

    elif model_key == "new_ladder":
        vae = model(latent_dist=config["prior"],
                    likelihood=config["likelihood"],
                    first_level_latents=config["first_level_latents"],
                    second_level_latents=config["second_level_latents"])
    else:
        raise Exception("Model: {} is not defined!".format(model_key))

    # Connect the model computational graph by executing a forward-pass
    vae(tf.zeros((1, 256, 256, 3)))
        
    return vae

def run(config_path=None,
        model_key="cnn",
        is_training=True,
        build_ac_dict=False,
        model_dir="/tmp/clic_test"):

    # ==========================================================================
    # Configuration
    # ==========================================================================

    config = {
        "training_set_size": 93085,
        "pixels_per_training_image": 256 * 256 * 3,

        # When using VALID for the hierarchical VAEs, this will give the correct
        # latent size
        "image_size": [256, 256],

        "batch_size": 8,
        "num_epochs": 20,
        
        "first_level_latents": 64,
        "second_level_latents": 192,

        "first_level_channels": 192,
        "second_level_channels": 128,
        "first_level_layers": 4,

        "loss": "nll_perceptual_kl",
        "likelihood": "laplace",
        "prior": "gaussian",

        # % of the number of batches when the coefficient is capped out
        # (i.e. for 1., the coef caps after the first epoch exactly)
        "warmup": 4.,
        "beta": 0.1,
        "gamma": 0.,
        "learning_rate": 3e-5,
        "optimizer": "adam",

        "log_freq": 50,
        "checkpoint_name": "_ckpt",
    }

    if config_path is not None:
        config = json.load(config_path)

    num_batches = config["training_set_size"] // config["batch_size"]


    print("Configuration:")
    print(json.dumps(config, indent=4, sort_keys=True))
    print("Num batches: {}".format(num_batches))

    # ==========================================================================
    # Load dataset
    # ==========================================================================

    train_dataset = download_process_and_load_data()
    
    train_dataset = clic_input_fn(train_dataset,
                                  batch_size=config["batch_size"],
                                  image_size=tuple(config["image_size"]))

    # ==========================================================================
    # Create VAE model
    # ==========================================================================

    g = tf.get_default_graph()
    
    # TODO: This is a temporary hack to log the graph in eager mode
    with g.as_default():
        
        vae = create_model(model_key, config)
        
        # Connect the model computational graph by executing a forward-pass
        vae(tf.zeros((1, 256, 256, 3)))

        del vae
    
    # Create actual model here
    vae = create_model(model_key, config)

    optimizer = optimizers[config["optimizer"]](config["learning_rate"])

    # ==========================================================================
    # Define Checkpoints
    # ==========================================================================

    global_step = tf.train.get_or_create_global_step()

    trainable_vars = vae.get_all_variables() + (global_step,)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")

    checkpoint, ckpt_prefix = setup_eager_checkpoints_and_restore(
        variables=trainable_vars,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=config["checkpoint_name"])

    # ==========================================================================
    # Tensorboard stuff
    # ==========================================================================

    logdir = os.path.join(model_dir, "log")
    writer = tfs.create_file_writer(logdir)
    writer.set_as_default()
    
    # Record the graph structure of the architecture
    tfs.graph(g)
    tfs.flush(writer)

    # ==========================================================================
    # Train VAE
    # ==========================================================================

    beta = config["beta"]

    # Combination coefficient for the mixture losses
    gamma = config["gamma"]

    if is_training:

        for epoch in range(1, config["num_epochs"] + 1):

            with tqdm(total=num_batches) as pbar:
                for batch in train_dataset:

                    # Increment global step
                    global_step.assign_add(1)

                    with tf.GradientTape() as tape, tfs_logger(config["log_freq"]):

                        B = batch.shape.as_list()[0]

                        # Predict the means of the pixels
                        output = vae(batch)
                        
                        warmup_coef = tf.minimum(1., global_step.numpy() / (config["warmup"] * num_batches))

                        log_prob = vae.log_prob
                        kl_divs = vae.kl_divergence
                        
                        total_kl = sum([tf.reduce_sum(kls) for kls in kl_divs])

                        output = tf.cast(output, tf.float32)
                        output = tf.clip_by_value(output, 0., 1.)

                        ms_ssim = 0. #tf.image.ssim_multiscale(batch, output, 1.)
                        
                        # This correction is necessary, so that the ms-ssim value is on the order
                        # of the KL and the log probability
                        ms_ssim_loss = tf.reduce_sum(1 - ms_ssim) * config["pixels_per_training_image"]

                        if config["loss"] == "nll_perceptual_kl":
                            loss = ((1 - gamma) * -log_prob + gamma * ms_ssim_loss + warmup_coef * beta * total_kl) / B

                        else:
                            raise Exception("Loss {} not available!".format(config["loss"]))


                        # Add tensorboard summaries
                        tfs.scalar("Loss", loss)
                        tfs.scalar("Log-Probability", log_prob / B)
                        tfs.scalar("KL", total_kl / B)
                        tfs.scalar("Beta-KL", beta * total_kl / B)
                        #tfs.scalar("Average MS-SSIM", tf.reduce_sum(ms_ssim) / B)
                        #tfs.scalar("MS-SSIM Loss", ms_ssim_loss)
                        tfs.scalar("Warmup-Coeff", warmup_coef)
                        tfs.scalar("Average PSNR", tf.reduce_sum(tf.image.psnr(batch, output, max_val=1.0)) / B)
                        tfs.image("Reconstruction", output)
                        tfs.image("Original", batch)
                        
                        for i, level_kl_divs in enumerate(kl_divs): 
                            tfs.scalar("Max-KL-on-Level-{}".format(i + 1), tf.reduce_max(level_kl_divs))

                    # Backprop
                    grads = tape.gradient(loss, vae.get_all_variables())
                    optimizer.apply_gradients(zip(grads, vae.get_all_variables()))

                    # Update the progress bar
                    pbar.update(1)
                    pbar.set_description("Epoch {}, Loss: {:.2f}, KL: {:.2f}, Log Prob: {:.4f}".format(epoch, loss, total_kl, log_prob))

            checkpoint.save(ckpt_prefix)
            
        print("Training finished!")

    else:
        print("Skipping training!")
        
       
    if build_ac_dict:
        
        miracle_bits = 8
        num_draws = 2**8
        
        train_image_dir = "/scratch/gf332/datasets/miracle_image_compression/train"
        train_image_paths = glob.glob(train_image_dir + "/*.png")

        probability_mass = [0] * num_draws

        for im_idx, im_path in enumerate(train_image_paths):

            print("Processing image {}/{}, {}".format(im_idx + 1, len(train_image_paths), im_path))

            image = load_and_process_image(im_path)[None, ...]

            reconstruction = vae(image)

            for i, (p, q) in enumerate(zip(vae.latent_priors, vae.latent_posteriors)):

                # Code layers
                print("Coding layer {}".format(i + 1))
                coded_samps = coded_sample(proposal=vae.latent_priors[1], 
                                           target=vae.latent_posteriors[1], 
                                           seed=im_idx, 
                                           n_points=30, 
                                           miracle_bits=miracle_bits)

                # Count up latent codes
                unique, _, counts = tf.unique_with_counts(coded_samps)

                unique = unique.numpy()
                counts = counts.numpy()

                for i, idx in enumerate(unique[unique < num_draws]):
                    probability_mass[idx] += counts[i]

            print("---------------------------------------")

            np.save("probability_mass.npy", np.array(probability_mass, np.int64))
        
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experimental models for CLIC')

    parser.add_argument('--config', type=open, default=None,
                    help='Path to the config JSON file.')
    parser.add_argument('--model', choices=list(models.keys()), default='cnn',
                    help='The model to train.')
    parser.add_argument('--no_training',
                        action="store_false",
                        dest="is_training",
                        default=True,
                        help='Should we just evaluate?')
    parser.add_argument('--build_ac_dict',
                        action="store_true",
                        dest="build_ac_dict",
                        default=False,
                        help='Should we build the Arithmetic Coding dictionary?')
    parser.add_argument('--model_dir',
                        type=lambda x: is_valid_file(parser, x),
                        default='/tmp/miracle_compress_clic',
                        help='The model directory.')

    args = parser.parse_args()
    
    run(config_path=args.config,
        model_key=args.model,
        is_training=args.is_training,
        build_ac_dict=args.build_ac_dict,
        model_dir=args.model_dir)
