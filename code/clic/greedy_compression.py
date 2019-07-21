# ==============================================================================
# Imports
# ==============================================================================

# This is needed so that python finds the utils
import sys
sys.path.append("/home/gf332/miracle-compession/code")
sys.path.append("/home/gf332/miracle-compession/code/compression")
sys.path.append("/homes/gf332/miracle-compession/code")
sys.path.append("/homes/gf332/miracle-compession/code/compression")
sys.path.append("/Users/gergelyflamich/Documents/Work/MLMI/miracle-compession/code")
sys.path.append("/Users/gergelyflamich/Documents/Work/MLMI/miracle-compession/code/compression")

import numpy as np

import os, glob
from tqdm import tqdm as tqdm

import tensorflow as tf
tfq = tf.quantization

import tensorflow_probability as tfp
tfd = tfp.distributions

from binary_io import to_bit_string, from_bit_string

def code_greedy_sample(target, proposal, n_bits_per_step, n_steps, seed, rho=1.):
    
    # Make sure the distributions have the correct type
    if target.dtype is not tf.float32:
        raise Exception("Target datatype must be float32!")
        
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
        
    dim = len(proposal.loc)
    
    n_samples = int(2**n_bits_per_step)
    
    #print("Taking {} samples per step".format(n_samples))

    best_sample = tf.zeros((1, dim), dtype=tf.float32)
    sample_index = []
    
    # The scale divisor needs to be square rooted because
    # we are dealing with standard deviations and not variances
    scale_divisor = np.sqrt(n_steps)
    
    proposal_shard = tfd.Normal(loc=proposal.loc / n_steps,
                                scale=rho * proposal.scale / scale_divisor)

    for i in range(n_steps):

        # Set new seed
        tf.random.set_random_seed(1000 * seed + i)
        
        samples = tf.tile(best_sample, [n_samples, 1]) + proposal_shard.sample(n_samples)

        log_probs = tf.reduce_sum(target.log_prob(samples), axis=1)

        index = tf.argmax(log_probs)

        best_sample = samples[index:index + 1, :]

        sample_index.append(to_bit_string(index.numpy(), n_bits_per_step))

    sample_index = ''.join(sample_index)
    
    return best_sample, sample_index


def decode_greedy_sample(sample_index, proposal, n_bits_per_step, n_steps, seed, rho=1.):
    
    # Make sure the distributions have the correct type
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
        
    dim = len(proposal.loc)
    
    indices = [from_bit_string(sample_index[i:i + n_bits_per_step]) 
               for i in range(0, n_bits_per_step * n_steps, n_bits_per_step)]
        
    # The scale divisor needs to be square rooted because
    # we are dealing with standard deviations and not variances
    scale_divisor = np.sqrt(n_steps)    
    
    proposal_shard = tfd.Normal(loc=proposal.loc / n_steps,
                                scale=rho * proposal.scale / scale_divisor)    
    
    n_samples = int(2**n_bits_per_step)
    
    sample = tf.zeros((1, dim), dtype=tf.float32)
        
    for i in range(n_steps):
        
        # Set new seed
        tf.random.set_random_seed(1000 * seed + i)
        
        samples = tf.tile(sample, [n_samples, 1]) + proposal_shard.sample(n_samples)

        index = indices[i]

        sample = samples[index:index + 1, :]
    
    return sample


def code_grouped_greedy_sample(target, 
                               proposal, 
                               n_bits_per_step, 
                               n_steps, 
                               seed, 
                               max_group_size_bits=12, # group size limited to 2^max_group_size_bits
                               adaptive=True,
                               rho=1.):
    
    # Make sure the distributions have the correct type
    if target.dtype is not tf.float32:
        raise Exception("Target datatype must be float32!")
        
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
    
    n_bits_per_group = n_bits_per_step * n_steps
    
    num_dimensions = np.prod(proposal.loc.shape.as_list())
    
    # ====================================================================== 
    # Preprocessing step: determine groups for sampling
    # ====================================================================== 
    if adaptive:
        
        group_start_indices = [0]
        group_kls = []
        
        kl_divs = tf.reshape(tfd.kl_divergence(target, proposal), [-1]).numpy()
        total_kl_bits = np.sum(kl_divs) / np.log(2)
        
        print("Total KL to split up: {:.2f} bits, "
              "maximum bits per group: {}, "
              "estimated number of groups: {},"
              "coding {} dimensions".format(total_kl_bits, 
                                            n_bits_per_group,
                                            total_kl_bits // n_bits_per_group + 1,
                                            num_dimensions
                                            ))
        
        current_group_size = 0
        current_group_kl = 0
        idx = -1
        
        while idx < num_dimensions - 1:
            
            idx = idx + 1
            current_group_size = current_group_size + 1
            current_group_kl = current_group_kl + kl_divs[idx]
            
            num_group_samps = np.ceil(np.exp(current_group_kl))
            
            if not ( current_group_size < 2**max_group_size_bits and 
                     num_group_samps < 2**n_bits_per_group ):
                
                group_start_indices.append(idx)
                group_kls.append((current_group_kl - kl_divs[idx]) / np.log(2))
                
                current_group_size = 0
                current_group_kl = 0
                
                idx = idx - 1
                
            
            if idx == len(kl_divs) - 1:
                group_kls.append(current_group_kl / np.log(2))
    
    else:
        group_size = int(2**max_group_size_bits)
        
        group_start_indices = list(range(0, num_dimensions, group_size))
        
    # ====================================================================== 
    # Sample each group
    # ====================================================================== 
    
    results = []
    
    group_start_indices += [num_dimensions]
    
    # rescale proposal by the proposal
    p_loc = tf.reshape(tf.zeros_like(proposal.loc), [-1])
    p_scale = tf.reshape(tf.ones_like(proposal.scale), [-1])
    
    # rescale target by the proposal
    t_loc = (tf.reshape(target.loc, [-1]) - p_loc) / p_scale
    t_scale = tf.reshape(target.scale, [-1]) / p_scale
    
    for i in tqdm(range(len(group_start_indices) - 1)):
        
        start_idx = group_start_indices[i]
        end_idx = group_start_indices[i + 1]
        
        results.append(code_greedy_sample(
            target=tfd.Normal(loc=t_loc[start_idx:end_idx],
                              scale=t_scale[start_idx:end_idx]), 
            
            proposal=tfd.Normal(loc=p_loc[start_idx:end_idx],
                                scale=p_scale[start_idx:end_idx]), 
            
            n_bits_per_step=n_bits_per_step, 
            n_steps=n_steps, 
            seed=seed,
            rho=rho))
        
    samples, codes = zip(*results)
    
    bitcode = ''.join(codes)
    sample = tf.concat(samples, axis=1)
    
    return sample, bitcode, group_start_indices
        

def decode_grouped_greedy_sample(bitcode, 
                                 group_start_indices,
                                 proposal, 
                                 n_bits_per_step, 
                                 n_steps, 
                                 seed, 
                                 max_group_size_bits=12, # group size limited to 2^max_group_size_bits
                                 adaptive=True,
                                 rho=1.):
    
    # Make sure the distributions have the correct type
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
    
    n_bits_per_group = n_bits_per_step * n_steps
    
    num_dimensions = np.prod(proposal.loc.shape.as_list())
    
    # ====================================================================== 
    # Decode each group
    # ====================================================================== 
                
    samples = []
    
    group_start_indices += [num_dimensions]
    
    p_loc = tf.reshape(tf.zeros_like(proposal.loc), [-1])
    p_scale = tf.reshape(tf.ones_like(proposal.scale), [-1])
    
    for i in tqdm(range(len(group_start_indices) - 1)):
        
        samples.append(decode_greedy_sample(
            sample_index=bitcode[n_bits_per_group * i: n_bits_per_group * (i + 1)],
            
            proposal=tfd.Normal(loc=p_loc[group_start_indices[i]:group_start_indices[i + 1]],
                                scale=p_scale[group_start_indices[i]:group_start_indices[i + 1]]), 
            
            n_bits_per_step=n_bits_per_step, 
            n_steps=n_steps, 
            seed=seed,
            rho=rho))
        
    sample = tf.concat(samples, axis=1)
    
    # Rescale the sample
    sample = tf.reshape(proposal.scale, [-1]) * sample + tf.reshape(proposal.loc, [-1])
    
    return sample