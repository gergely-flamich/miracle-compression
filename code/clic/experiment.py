# ==============================================================================
# Imports
# ==============================================================================

# This is needed so that python finds the utils
import sys
sys.path.append("/home/gf332/miracle-compession/code")
sys.path.append("/homes/gf332/miracle-compession/code")
sys.path.append("/homes/gf332/miracle-compession/code/compression")

import argparse
import os, glob
import json
from tqdm import tqdm

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tf.enable_eager_execution()

tfe = tf.contrib.eager
tfs = tf.contrib.summary
tfs_logger = tfs.record_summaries_every_n_global_steps

from utils import is_valid_file, setup_eager_checkpoints_and_restore
from load_data import load_and_process_image, create_random_crops, download_process_and_load_data
from compression import coded_sample, decode_sample
from pipeline import clic_input_fn, create_model, optimizers, models


def run(config_path,
        model_key="ladder",
        train_stage=0, # 0 - train first stage, 1 - train second stage, 2 - trained
        is_training=True,
        build_ac_dict=False,
        model_dir="/tmp/clic_test"):
    
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
        
        vae = create_model(model_key, config, train_stage=train_stage)

        del vae
    
    # Create actual model here
    vae = create_model(model_key, config)
    
    global_step = tf.train.get_or_create_global_step()
    
    learning_rate = lambda: tf.constant(config["learning_rate"])
#     learning_rate = tf.compat.v1.train.exponential_decay(config["learning_rate"],
#                                                          global_step,
#                                                          config["learning_rate_decay_step"], 
#                                                          0.96, 
#                                                          staircase=False)

    optimizer = optimizers[config["optimizer"]](learning_rate)

    # ==========================================================================
    # Define Checkpoints
    # ==========================================================================

    if isinstance(vae, tuple):
        trainable_vars = vae[0].get_all_variables() + vae[1].get_all_variables() + (global_step,)
        
    else:
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
    # Define training steps
    # ==========================================================================
    
    beta = config["beta"]

    # Combination coefficient for the mixture losses
    gamma = config["gamma"]
    

    def ladder_train_step(batch):
        
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

            loss = ((1 - gamma) * -log_prob + gamma * ms_ssim_loss + warmup_coef * beta * total_kl) / B

            # Add tensorboard summaries
            tfs.scalar("Learning-Rate", learning_rate())
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
            
            if vae.learn_log_gamma:
                tfs.scalar("Gamma", tf.exp(vae.log_gamma))

            for i, level_kl_divs in enumerate(kl_divs): 
                tfs.scalar("Max-KL-on-Level-{}".format(i + 1), tf.reduce_max(level_kl_divs))

        # Backprop
        grads = tape.gradient(loss, vae.get_all_variables())
        optimizer.apply_gradients(zip(grads, vae.get_all_variables()))
        
        return loss, total_kl, log_prob
        
    # ==========================================================================
    
    def two_stage_train_step(batch):      
        
        vae_man, vae_mes = vae
        
        with tf.GradientTape() as tape, tfs_logger(config["log_freq"]):

            B = batch.shape.as_list()[0]
            warmup_coef = tf.minimum(1., global_step.numpy() / (config["warmup"] * num_batches))

            if train_stage == 0:
                output = vae_man(batch)

                log_prob = vae_man.log_prob
                kl_divs = vae_man.kl_divergence

                total_kl = tf.reduce_sum(kl_divs)
                
                output = tf.cast(output, tf.float32)
                
            elif train_stage == 1:
                z = vae_man.encode(batch)
                z_ = vae_mes(z)
                
                log_prob = vae_mes.log_prob
                kl_divs = vae_mes.kl_divergence
                total_kl = tf.reduce_sum(kl_divs)
                
                if config["use_reconstruction_loss"]:
                    output = vae_man.decode(z_)
                    log_prob = vae_man.likelihood.log_prob(batch)
            
            loss = ( -log_prob + warmup_coef * beta * total_kl) / B

            # Add tensorboard summaries
            sum_num = train_stage + 1
            
            tfs.scalar("Loss-{}".format(sum_num), loss)
            tfs.scalar("Log-Probability-{}".format(sum_num), log_prob / B)
            tfs.scalar("KL-{}".format(sum_num), total_kl / B)
            tfs.scalar("Beta-KL-{}".format(sum_num), beta * total_kl / B)
            tfs.scalar("Max-KL-{}".format(sum_num), tf.reduce_max(kl_divs))

            tfs.scalar("Warmup-Coeff-{}".format(sum_num), warmup_coef)
            
            
            tfs.scalar("Gamma-{}".format(sum_num), tf.exp(vae[train_stage].log_gamma))
            
            if train_stage == 0 or config["use_reconstruction_loss"]:
                average_psnr = tf.reduce_sum(tf.image.psnr(batch, output, max_val=1.0)) / B
                tfs.scalar("Average PSNR-{}".format(sum_num), average_psnr)
                
                tfs.image("Reconstruction-{}".format(sum_num), output)
                tfs.image("Original-{}".format(sum_num), batch)
                
            
        # Backprop
        if train_stage == 0:
            grads = tape.gradient(loss, vae_man.get_all_variables())
            optimizer.apply_gradients(zip(grads, vae_man.get_all_variables()))
            
        elif train_stage == 1:
            grads = tape.gradient(loss, vae_mes.get_all_variables())
            optimizer.apply_gradients(zip(grads, vae_mes.get_all_variables()))
        
        return loss, total_kl, log_prob
    
    # ==========================================================================
    # Train VAE
    # ==========================================================================

    if is_training:

        for epoch in range(1, config["num_epochs"] + 1):

            with tqdm(total=num_batches) as pbar:
                for batch in train_dataset:

                    # Increment global step
                    global_step.assign_add(1)

                    if model_key in ["ladder"]:
                        loss, total_kl, log_prob = ladder_train_step(batch)
                        
                    elif model_key in ["two_stage"]:
                        loss, total_kl, log_prob = two_stage_train_step(batch)
                    

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
                coded_samps = coded_sample(proposal=vae.latent_priors[i], 
                                           target=vae.latent_posteriors[i], 
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

    parser.add_argument('--config', 
                        type=open,
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
                                       
    parser.add_argument('--train_stage',
                        type=int,
                        default=0)                                   

    args = parser.parse_args()

    run(config_path=args.config,
        model_key=args.model,
        is_training=args.is_training,
        build_ac_dict=args.build_ac_dict,
        model_dir=args.model_dir,
        train_stage=args.train_stage)
