# ==============================================================================
# Imports
# ==============================================================================
import argparse
import os
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

from mnist_vae import MnistVAE
from utils import is_valid_file, setup_eager_checkpoints_and_restore

# ==============================================================================
# Predefined Stuff
# ==============================================================================

optimizers = {
    "sgd": tf.train.GradientDescentOptimizer,
    "momentum": lambda lr:
                    tf.train.MomentumOptimizer(learning_rate=lr,
                                               momentum=0.9,
                                               use_nesterov=True),
    "adam": tf.train.AdamOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer
}

# ==============================================================================
# Auxiliary Functions
# ==============================================================================
def mnist_input_fn(data, batch_size=128, shuffle_samples=5000):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(shuffle_samples)
    dataset = dataset.map(mnist_binary_parse_fn)
    dataset = dataset.batch(batch_size)

    return dataset

def mnist_binary_parse_fn(data, normalizing_const=255.):
    return tf.cast(data, tf.float32) / normalizing_const

# ==============================================================================
# Main Function
# ==============================================================================
def run(args):
    """
    args - dict provided by argparse
    """

    # ==========================================================================
    # Configuration
    # ==========================================================================

    config = {
        "training_set_size": 60000,
        "max_pixel_value": 1.,

        "num_latents": 10,
        "hidden_units": 100,
        "data_likelihood": "bernoulli",

        "batch_size": 128,
        "num_epochs": 20,

        "loss": "neg_elbo",
        "beta": 10.,
        "learning_rate": 1e-3,
        "optimizer": "adam",

        "log_freq": 250,
        "checkpoint_name": "_ckpt",
    }

    num_batches = config["training_set_size"] // config["batch_size"]


    print("Configuration:")
    print(json.dumps(config, indent=4, sort_keys=True))
    print("Num batches: {}".format(num_batches))

    # ==========================================================================
    # Load dataset
    # ==========================================================================

    ((train_data, _),
    (test_data, _)) = tf.keras.datasets.mnist.load_data()

    train_dataset = mnist_input_fn(train_data[:num_batches * config["batch_size"]],
                                   batch_size=config["batch_size"])

    # ==========================================================================
    # Create VAE model
    # ==========================================================================

    vae = MnistVAE(hidden_units=config["hidden_units"],
                   num_latents=config["num_latents"],
                   data_likelihood=config["data_likelihood"])

    # Connect the model computational graph by executing a forward-pass
    vae(tf.zeros((1, 28, 28)))

    optimizer = optimizers[config["optimizer"]](config["learning_rate"])

    # ==========================================================================
    # Define Checkpoints
    # ==========================================================================

    global_step = tf.train.get_or_create_global_step()

    trainable_vars = vae.get_all_variables() + (global_step,)
    checkpoint_dir = os.path.join(args.model_dir, "checkpoints")

    checkpoint, ckpt_prefix = setup_eager_checkpoints_and_restore(
        variables=trainable_vars,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=config["checkpoint_name"])

    # ==========================================================================
    # Tensorboard stuff
    # ==========================================================================

    logdir = os.path.join(args.model_dir, "log")
    writer = tfs.create_file_writer(logdir)
    writer.set_as_default()

    # ==========================================================================
    # Train VAE
    # ==========================================================================

    beta = config["beta"]

    if args.is_training:

        for epoch in range(1, config["num_epochs"] + 1):

            print("Epoch {}".format(epoch))

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
                            loss = -log_prob + beta * kl_div

                        elif config["loss"] == "psnr_kl":
                            # PSNR: the Gaussian negative log prob is the MSE
                            psnr = 20 * tf.log(config["max_pixel_value"]) - 10 * tf.log(-log_prob)

                            loss = -psnr + beta * kl_div

                        else:
                            raise Exception("Loss {} not available!".format(config["loss"]))

                        output = tf.cast(tf.expand_dims(output, axis=-1), tf.float32)

                        # Add tensorboard summaries
                        tfs.scalar("loss", loss)
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


    print(vae.encode(tf.convert_to_tensor(test_data[:1, ...] / 255.)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Bayes By Backprop models')

    parser.add_argument('--no_training',
                        action="store_false",
                        dest="is_training",
                        default=True,
                        help='Should we just evaluate?')
    parser.add_argument('--model_dir',
                        type=lambda x: is_valid_file(parser, x),
                        default='/tmp/miracle_compress_mnist',
                        help='The model directory.')

    args = parser.parse_args()
    run(args)
