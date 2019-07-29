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

# Models
from ladder_network import ClicNewLadderCNN
from two_stage_vae import ClicTwoStageVAE, ClicTwoStageVAE_Measure, ClicTwoStageVAE_Manifold

# ==============================================================================
# Predefined Stuff
# ==============================================================================

models = {
    "ladder": ClicNewLadderCNN,
    "two_stage": ClicTwoStageVAE
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
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def clic_parse_fn(image, image_size=(316, 316)):
    return tf.image.random_crop(image, size=image_size + (3,))

def create_model(model_key, config, train_stage=0):
    model = models[model_key]
   
    if model_key == "two_stage":
        
        vae_man = ClicTwoStageVAE_Manifold(latent_dist=config["latent_dist"],
                                           likelihood=config["likelihood"],
                                           latent_filters=config["latent_dims"],
                                           num_layers=config["first_level_layers"])
        
        
        vae_mes = ClicTwoStageVAE_Measure(latent_dist=config["latent_dist"],
                                           likelihood=config["likelihood"],
                                           latent_filters=config["latent_dims"],
                                           residual=config["residual"],
                                           num_layers=config["second_level_layers"])

        
        # Connect the model computational graph by executing a forward-pass
        vae_man(tf.zeros((1, 256, 256, 3)))
        
        z = vae_man.encode(tf.zeros((1, 256, 256, 3)))
        
        z_ = vae_mes(z)
        
        vae_man.decode(z_)
        
        vae = (vae_man, vae_mes)

    elif model_key == "ladder":
        vae = model(first_level_latent_dist=config["first_level_latent_dist"],
                    second_level_latent_dist=config["second_level_latent_dist"],
                    likelihood=config["likelihood"],
                    heteroscedastic=config["heteroscedastic"],
                    average_gamma=config["average_gamma"],
                    
                    first_level_latents=config["first_level_latents"],
                    second_level_latents=config["second_level_latents"],
                    
                    first_level_residual=config["first_level_residual"],
                    second_level_residual=config["second_level_residual"],
                    
                    first_level_channels=config["first_level_channels"],
                    second_level_channels=config["second_level_channels"],
                    
                    kernel_shape=tuple(config["kernel_shape"]),
                    padding=config["padding"],
                    
                    learn_log_gamma=config["learn_log_gamma"])
        
        # Connect the model computational graph by executing a forward-pass
        vae(tf.zeros((1, 256, 256, 3)))
    else:
        raise Exception("Model: {} is not defined!".format(model_key))

    
        
    return vae