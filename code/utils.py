import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import os, tempfile


# ==============================================================================
# Miscellaneous helper functions
# ==============================================================================

def is_valid_file(parser, arg):
    """
    Taken from
    https://stackoverflow.com/questions/11540854/file-as-command-line-argument-for-argparse-error-message-if-argument-is-not-va
    and
    https://stackoverflow.com/questions/9532499/check-whether-a-path-is-valid-in-python-without-creating-a-file-at-the-paths-ta
    """
    arg = str(arg)
    if os.path.exists(arg):
        return arg

    dirname = os.path.dirname(arg) or os.getcwd()
    try:
        with tempfile.TemporaryFile(dir=dirname): pass
        return arg
    except Exception:
        parser.error("A file at the given path cannot be created: " % arg)


# ==============================================================================
# Tensorflow helper functions
# ==============================================================================

def setup_eager_checkpoints_and_restore(variables, checkpoint_dir, checkpoint_name="_ckpt"):
    """
    Convenience function to set up TF eager checkpoints for the given variables.

    variables - iterable of TF Variables that we want to checkpoint
    checkpoint_dir - string containing the checkpoint path
    """

    ckpt_prefix = os.path.join(checkpoint_dir, checkpoint_name)

    checkpoint = tf.train.Checkpoint(**{v.name: v for v in variables})

    latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint_path is None:
        print("No checkpoint found!")
    else:
        print("Checkpoint found at {}, restoring...".format(latest_checkpoint_path))
        checkpoint.restore(latest_checkpoint_path).assert_consumed()
        print("Model restored!")

    return checkpoint, ckpt_prefix


# ==============================================================================
# Dataset loading functions
# ==============================================================================

def process_image(image, normalize=True):
    """
    image - raw representation of an image
    normalize - will adjust all pixels of the image to lie between 0 and 1
    for every channel.
    """

    img_tensor = tf.image.decode_image(image)
    img_tensor = tf.cast(img_tensor, tf.float32)

    if normalize:
        img_tensor /= 255.

    return img_tensor


def load_and_process_image(image_path):

    img_raw = tf.read_file(image_path)
    return process_image(img_raw)
