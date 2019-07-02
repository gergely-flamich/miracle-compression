"""
This file contains stuff related to the compression part of the project
"""
# ==============================================================================
# Imports
# ==============================================================================

# This is needed so that python finds the utils
import sys
sys.path.append("/home/gf332/miracle-compession/code")
sys.path.append("/home/gf332/miracle-compession/code/compression")
sys.path.append("/Users/gergelyflamich/Documents/Work/MLMI/miracle-compession/code")
sys.path.append("/Users/gergelyflamich/Documents/Work/MLMI/miracle-compession/code/compression")

import numpy as np

import os, glob
from tqdm import tqdm as tqdm

import tensorflow as tf
tfq = tf.quantization

import tensorflow_probability as tfp
tfd = tfp.distributions

def rejection_sample(p,
                     q,
                     num_draws,
                     seed,
                     n_points=20):

    
    tf.random.set_random_seed(seed)
    
    latent_shape = q.loc.shape.as_list()
    num_latents = np.prod(latent_shape)
    
    # Create flattened versions of the distributions
    p_loc = tf.reshape(p.loc, [-1])
    p_scale = tf.reshape(p.scale, [-1])
    
    p = tfd.Normal(loc=p_loc, scale=p_scale)
    
    q_loc = tf.reshape(q.loc, [-1])
    q_scale = tf.reshape(q.scale, [-1])
    
    q = tfd.Normal(loc=q_loc, scale=q_scale)
    
    # Set up bins for a single distribution
    p_mass = tf.concat(([0.], [1. / (n_points - 2)] * (n_points - 2), [0.]), axis=0)
    quantiles = np.linspace(0., 1., n_points + 1)
    
    # Set up bins for each latent by repeating it
    # Note: sample space is first index, number of latents is second
    p_mass = np.repeat(p_mass[None, :], num_latents, axis=0).T
    quantiles = np.repeat(quantiles[None, :], num_latents, axis=0)
    
    # Density of Q at edges of the bins
    open_sections = q.quantile(quantiles[:, 1:-1].T)
    
    open_cdf = p.cdf(open_sections)
    cdfs = tf.concat((tf.zeros((1, num_latents)), open_cdf, tf.ones((1, num_latents))), axis=0)
    probs = cdfs[1:, :] - cdfs[:-1, :]
    
    # Need these for the parallelisation
    latent_indices = tf.convert_to_tensor(np.arange(num_latents))
    
    indices_ = np.repeat(np.arange(n_points, dtype=np.float32)[None, :], num_latents, axis=0).T
    indices_ = tf.convert_to_tensor(indices_)
    indices = indices_[:-1, :]
    
    last_indices = indices_[-1:, :]
    
    infinities = np.inf * tf.ones_like(indices)
    
    # --------------------------------------------------------------
    # Perform Rejection Sampling
    # --------------------------------------------------------------

    # Initialise p_i for each latent
    p_i = tf.zeros((n_points, num_latents))
    
    # Accepted is now a mask
    accepted = tf.zeros((num_latents,), dtype=tf.bool)
    
    # Initialise all samples to infinity (as an invalid value)
    sample = np.inf * tf.ones((num_latents))
    sample_indices = np.inf * tf.ones((num_latents))

    for j in tqdm(range(num_draws)):
        
        # Sum along the sample space dimension
        # to get total probability mass for each dim
        p_star = tf.reduce_sum(p_i, axis=0)
        
        # Calculate the alpha_is
        a = p_mass - p_i
        b = (1 - p_star) * probs

        alpha_i = tf.where(a < b, a, b)
        
        # Draw a sample from the proposal distribution
        new_sample = p.sample()

        # Find which bucket the sample falls into
        bucket_index_matrix = tf.concat([tf.where(new_sample < open_sections, indices, infinities), last_indices],
                                        axis=0)

        buckets = tf.argmin(bucket_index_matrix, axis=0)
        buckets = tf.transpose(tf.stack((buckets, latent_indices)))

        # Get the appropriate alphas and Qs
        alphas_i = tf.gather_nd(alpha_i, buckets)
        probs_i = tf.gather_nd(probs, buckets)

        # Calculate the betas
        betas = alphas_i / ((1 - p_star) * probs_i)

        accept_samples = tf.random.uniform((num_latents,))
        
        # Get new accept flags
        new_accepted = accept_samples < betas
        
        # Update the sample and index matrices which don't have their values set yet
        # and are accepted in the current round
        update_indices = tf.logical_and(new_accepted, tf.logical_not(accepted))
        
        sample = tf.where(update_indices, new_sample, sample)
        sample_indices = tf.where(update_indices, j * tf.ones_like(sample_indices), sample_indices)
        #print(sample)
        
        accepted = tf.math.logical_or(new_accepted, accepted)
        #print(accepted)
        
        p_i = p_i + alpha_i

    return sample, sample_indices, accepted



def decode_rejection_sample(p, seed, index_matrix, num_draws):
    tf.random.set_random_seed(seed)
    
    latent_shape = p.loc.shape.as_list()
    num_latents = np.prod(latent_shape)
    
    # Create flattened versions of the distributions
    p_loc = tf.reshape(p.loc, [-1])
    p_scale = tf.reshape(p.scale, [-1])
    
    p = tfd.Normal(loc=p_loc, scale=p_scale)
    
    # Initialise all samples to infinity (as an invalid value)
    sample = np.inf * tf.ones((num_latents))
    
    for j in tqdm(range(num_draws)):
        
        new_sample = p.sample()
        
        # Set the appropriate samples
        sample = tf.where(index_matrix == j, new_sample, sample)
        
        # We have to do this so that the decoder's prior samples' 
        # random generator state is in sync with the encoder
        tf.random.uniform((num_latents,))
        
    return sample



def coded_sample(proposal, target, seed, n_points=30, miracle_bits=8, outlier_mode="quantize"):
    
    num_draws = 2**miracle_bits
    
    samples, sample_indices, accepted = rejection_sample(p=proposal,
                                                           q=target,
                                                           n_points=30,
                                                           num_draws=num_draws,
                                                           seed=seed)
    
    # How should we deal with latent dimensions that weren't accepted?
    
    # quantize: just sample the posterior and reduce its precision
    if outlier_mode == "quantize":
        
        # Flatten sample
        outlier_samples = tf.reshape(target.sample(), [-1])
        
        # Halve precision
        outlier_samples = tfq.quantize(outlier_samples, -30, 30, tf.quint16).output
        
        # Cast to float so we can combine with sample index matrix
        outlier_samples = tf.cast(outlier_samples, tf.float32)

    # TODO: Do Miracle-style importance sampling
    elif outlier_mode == "importance_sample":
        pass
    
    # Combine sample indices with outliers
    coded_samples = tf.where(accepted, sample_indices, outlier_samples)
    coded_samples = tf.cast(coded_samples, tf.int32)
    
    return coded_samples



def decode_sample(coded_sample, proposal, seed, n_points=30, miracle_bits=8, outlier_mode="quantize"):
    
    num_draws = 2**miracle_bits
    
    decoded_sample = decode_rejection_sample(p=proposal,
                                             seed=seed,
                                             index_matrix=coded_sample.numpy(),
                                             num_draws=num_draws)
    
    # quantize: just sample the posterior and reduce its precision
    if outlier_mode == "quantize":
        
        decoded_outliers = tfq.dequantize(tf.cast(coded_sample, tf.quint16), -30, 30)
        
    # TODO: Do Miracle-style importance sampling
    elif outlier_mode == "importance_sample":
        pass
    
    decoded_sample = tf.where(coded_sample < num_draws, decoded_sample, decoded_outliers)
    
    return decoded_sample