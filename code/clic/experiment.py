# ==============================================================================
# Imports
# ==============================================================================

# This is needed so that python finds the utils
import sys
sys.path.append("/home/gf332/miracle-compession/code")

from imageio import imwrite

import argparse
import os, glob
import json
from tqdm import tqdm

# Needed for compression as the common source of randomness
from sobol_seq import i4_sobol_generate
from scipy.stats import norm

import tensorflow as tf
import tensorflow_probability as tfp
tf.enable_eager_execution()

tfe = tf.contrib.eager
tfs = tf.contrib.summary
tfs_logger = tfs.record_summaries_every_n_global_steps

from architectures import ClicCNN
from utils import is_valid_file, setup_eager_checkpoints_and_restore
from load_data import load_and_process_image, create_random_crops, download_process_and_load_data

# ==============================================================================
# Predefined Stuff
# ==============================================================================

models = {
    "cnn": ClicCNN
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
def clic_input_fn(dataset, buffer_size=1000, batch_size=8):
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)
    
    return dataset

def run(config_path=None,
        model_key="cnn",
        is_training=True,
        model_dir="/tmp/clic_test"):

    # ==========================================================================
    # Configuration
    # ==========================================================================

    config = {
        "training_set_size": 93085,

        "batch_size": 8,
        "num_epochs": 20,

        "loss": "neg_elbo",
        "beta": 0.1,
        "learning_rate": 3e-5,
        "optimizer": "adam",

        "log_freq": 200,
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

    train_dataset, valid_dataset = download_process_and_load_data()
    
    train_dataset = clic_input_fn(train_dataset,
                                  batch_size=config["batch_size"])

    # ==========================================================================
    # Create VAE model
    # ==========================================================================

    model = models[model_key]

    vae = model(prior="laplace")

    # Connect the model computational graph by executing a forward-pass
    vae(tf.zeros((1, 256, 256, 3)))

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

    # ==========================================================================
    # Train VAE
    # ==========================================================================

    beta = config["beta"]

    if is_training:

        for epoch in range(1, config["num_epochs"] + 1):

            with tqdm(total=num_batches) as pbar:
                for batch in train_dataset:

                    # Increment global step
                    global_step.assign_add(1)

                    with tf.GradientTape() as tape, tfs_logger(config["log_freq"]):

                        # Predict the means of the pixels
                        output = vae(batch)

                        log_prob = vae.log_prob
                        kl_div = vae.kl_divergence

                        if config["loss"] == "neg_elbo":
                            # Cross-entropy / MSE loss (depends on )
                            B = batch.shape.as_list()[0]
                            loss = (-log_prob + beta * kl_div) / B

                        elif config["loss"] == "psnr_kl":
                            # PSNR: the Gaussian negative log prob is the MSE
                            psnr = 20 * tf.log(config["max_pixel_value"]) - 10 * tf.log(-log_prob)

                            loss = -psnr + beta * kl_div

                        else:
                            raise Exception("Loss {} not available!".format(config["loss"]))

                        output = tf.cast(output, tf.float32)

                        # Add tensorboard summaries
                        tfs.scalar("Loss", loss)
                        tfs.scalar("Log-Probability", log_prob / B)
                        tfs.scalar("KL", kl_div / B)
                        tfs.scalar("Beta-KL", beta * kl_div / B)
                        tfs.image("Reconstruction", output)

                    # Backprop
                    grads = tape.gradient(loss, vae.get_all_variables())
                    optimizer.apply_gradients(zip(grads, vae.get_all_variables()))

                    # Update the progress bar
                    pbar.update(1)
                    pbar.set_description("Epoch {}, Loss: {:.2f}, KL: {:.2f}, Log Prob: {:.4f}".format(epoch, loss, kl_div, log_prob))

            checkpoint.save(ckpt_prefix)

    else:
        print("Skipping training!")


    # ==========================================================================
    # Compress images
    # ==========================================================================


    print(vae.encode(tf.convert_to_tensor(test_data[:1, ...] / 255., dtype=tf.float32)))


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
    parser.add_argument('--model_dir',
                        type=lambda x: is_valid_file(parser, x),
                        default='/tmp/miracle_compress_clic',
                        help='The model directory.')

    args = parser.parse_args()
    
    run(config_path=args.config,
        model_key=args.model,
        is_training=args.is_training,
        model_dir=args.model_dir)
