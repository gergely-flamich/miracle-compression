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
from two_stage_vae import ClicTwoStageVAE

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
    dataset = dataset.prefetch(1)

    return dataset

def clic_parse_fn(image, image_size=(316, 316)):
    return tf.image.random_crop(image, size=image_size + (3,))

def create_model(model_key, config, train_stage=0):
    model = models[model_key]
   
    if model_key == "two_stage":
        
        vae = model(latent_dist=config["prior"],
                     likelihood=config["likelihood"],
                     latent_filters=config["two_stage_latent_dims"],
                     first_level_layers=config["two_stage_first_level_layers"],
                     second_level_layers=config["two_stage_second_level_layers"])
        
        vae.train_stage = train_stage

    elif model_key == "ladder":
        vae = model(latent_dist=config["prior"],
                    likelihood=config["likelihood"],
                    first_level_latents=config["first_level_latents"],
                    second_level_latents=config["second_level_latents"])
    else:
        raise Exception("Model: {} is not defined!".format(model_key))

    # Connect the model computational graph by executing a forward-pass
    vae(tf.zeros((1, 256, 256, 3)))
        
    return vae