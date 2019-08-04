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
from pipeline import clic_input_fn, create_model, optimizers, models

from kodak import compress_kodak

def run(config_path,
        kodak_path,
        rec_path,
        model_key,
        model_dir):
    
    config = json.load(config_path)

    # ==========================================================================
    # Create VAE model
    # ==========================================================================

    g = tf.get_default_graph()
    
    # TODO: This is a temporary hack to log the graph in eager mode
    with g.as_default():
        
        vae = create_model(model_key, config, train_stage=2)

        del vae
    
    # Create actual model here
    vae = create_model(model_key, config)
    
    global_step = tf.train.get_or_create_global_step()
    
    # ==========================================================================
    # Reload the model
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

    kodak_dataset_path = kodak_path
    reconstruction_root = rec_path

    rim = compress_kodak(vae=vae,
                         kodak_dataset_path=kodak_dataset_path,
                         reconstruction_root=reconstruction_root,
                         reconstruction_subdir=os.path.basename(model_dir), 
                         backfitting_steps_level_1=0,
                         use_log_prob=True,
                         theoretical=None,
                         verbose=False)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Experimental models for CLIC')

    parser.add_argument('--config', 
                        type=open,
                        help='Path to the config JSON file.')
    parser.add_argument('--model', choices=list(models.keys()), default='cnn',
                    help='The model to train.')  
    
    parser.add_argument('--kodak_path', type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('--rec_path', type=lambda x: is_valid_file(parser, x))
    
    parser.add_argument('--model_dir',
                        type=lambda x: is_valid_file(parser, x),
                        help='The model directory.')
                            

    args = parser.parse_args()

    run(config_path=args.config,
        model_key=args.model,
        model_dir=args.model_dir,
        kodak_path=args.kodak_path,
        rec_path=args.rec_path)
