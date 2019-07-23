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

def code_greedy_sample(target, 
                       proposal, 
                       n_bits_per_step, 
                       n_steps, 
                       seed, 
                       rho=1., 
                       backfitting_steps=0,
                       use_log_prob=False):
    
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
        samples = proposal_shard.sample(n_samples)

        test_samples = tf.tile(best_sample, [n_samples, 1]) + samples

        log_probs = tf.reduce_sum(target.log_prob(test_samples), axis=1)

        index = tf.argmax(log_probs)

        best_sample = test_samples[index:index + 1, :]

        sample_index.append(index.numpy())
    
    # ----------------------------------------------------------------------
    # Perform backfitting
    # ----------------------------------------------------------------------
    
    for b in range(backfitting_steps):
        
        # Single backfitting step
        for i in range(n_steps):

            # Set seed to regenerate the previously generated samples here
            tf.random.set_random_seed(1000 * seed + i)
            samples = proposal_shard.sample(n_samples)
            
            idx = sample_index[i]
            # Undo the addition of the current sample
            best_sample = best_sample - samples[idx : idx + 1, :]
            
            # Generate candidate samples
            test_samples = tf.tile(best_sample, [n_samples, 1]) + samples

            if use_log_prob:
                test_scores = tf.reduce_sum(target.log_prob(test_samples), axis=1)
            else:
                test_scores = tf.reduce_sum(-((test_samples - target.loc)**2 / target.scale**2)**4,
                                           axis=1)

            index = tf.argmax(test_scores)

            best_sample = test_samples[index:index + 1, :]

            sample_index[i] = index.numpy()
    
    
    sample_index = list(map(lambda x: to_bit_string(x, n_bits_per_step), sample_index))
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
                               n_steps, 
                               n_bits_per_step,
                               seed,
                               max_group_size_bits=12,
                               adaptive=True,
                               backfitting_steps=0,
                               use_log_prob=False,
                               rho=1.):
    
    # Make sure the distributions have the correct type
    if target.dtype is not tf.float32:
        raise Exception("Target datatype must be float32!")
        
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
    
    n_bits_per_group = n_bits_per_step * n_steps
    
    num_dimensions = np.prod(proposal.loc.shape.as_list())
    
    # rescale proposal by the proposal
    p_loc = tf.reshape(tf.zeros_like(proposal.loc), [-1])
    p_scale = tf.reshape(tf.ones_like(proposal.scale), [-1])
    
    # rescale target by the proposal
    t_loc = tf.reshape((target.loc - proposal.loc) / proposal.scale, [-1])
    t_scale = tf.reshape(target.scale / proposal.scale, [-1])
    
    kl_divs = tf.reshape(tfd.kl_divergence(target, proposal), [-1]).numpy()
        
    # ====================================================================== 
    # Preprocessing step: determine groups for sampling
    # ====================================================================== 

    group_start_indices = [0]
    group_kls = []

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
    prev_idx = -2
    twice_same = False

    while idx < num_dimensions - 1:

        if twice_same and idx == prev_idx:

            print("oh no: {}".format(idx))
            return

        twice_same = idx == prev_idx
        prev_idx = idx

        idx = idx + 1
        current_group_size = current_group_size + 1
        current_group_kl = current_group_kl + kl_divs[idx]

        #num_group_samps = np.ceil(np.exp(current_group_kl))

        if not ( np.log(current_group_size) / np.log(2) < max_group_size_bits and 
                 current_group_kl < n_bits_per_group * np.log(2) - 1):

            group_start_indices.append(idx)
            group_kls.append((current_group_kl - kl_divs[idx]) / np.log(2))

            current_group_size = 0
            current_group_kl = 0

            idx = idx - 1


        if idx == len(kl_divs) - 1:
            group_kls.append(current_group_kl / np.log(2))

    # ====================================================================== 
    # Sample each group
    # ====================================================================== 
    
    results = []
    
    group_start_indices += [num_dimensions] 
    
    for i in tqdm(range(len(group_start_indices) - 1)):
        
        start_idx = group_start_indices[i]
        end_idx = group_start_indices[i + 1]
        
        result = code_greedy_sample(
            target=tfd.Normal(loc=t_loc[start_idx:end_idx],
                              scale=t_scale[start_idx:end_idx]), 

            proposal=tfd.Normal(loc=p_loc[start_idx:end_idx],
                                scale=p_scale[start_idx:end_idx]), 

            n_bits_per_step=n_bits_per_step, 
            n_steps=n_steps, 
            seed=i + seed,
            backfitting_steps=backfitting_steps,
            use_log_prob=use_log_prob,
            rho=rho)
        
        results.append(result)
        
    samples, codes = zip(*results)
    
    bitcode = ''.join(codes)
    sample = tf.concat(samples, axis=1)
    
    # Rescale the sample
    sample = tf.reshape(proposal.scale, [-1]) * sample + tf.reshape(proposal.loc, [-1])
    
    return sample, bitcode, group_start_indices
        
def decode_grouped_greedy_sample(bitcode, 
                                 group_start_indices,
                                 proposal, 
                                 n_bits_per_step, 
                                 n_steps, 
                                 seed,
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
            seed=i + seed,
            rho=rho))
        
    sample = tf.concat(samples, axis=1)
    
    # Rescale the sample
    sample = tf.reshape(proposal.scale, [-1]) * sample + tf.reshape(proposal.loc, [-1])
    
    return sample


# ==============================================================================================
# ==============================================================================================
# ==============================================================================================
#
# Importance Sampling
#
# ==============================================================================================
# ==============================================================================================
# ==============================================================================================

def code_importance_sample(target,
                           proposal,
                           n_coding_bits,
                           seed):
    
    # Make sure the distributions have the correct type
    if target.dtype is not tf.float32:
        raise Exception("Target datatype must be float32!")
        
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
        
    dim = len(proposal.loc)
    
    #print("Taking {} samples per step".format(n_samples))

    best_sample = tf.zeros((1, dim), dtype=tf.float32)
    sample_index = []
    
    kls = tfd.kl_divergence(target, proposal)
    total_kl = tf.reduce_sum(kls)
    
    num_samples = tf.cast(tf.ceil(tf.exp(total_kl)), tf.int32)
    
    # Set new seed
    tf.random.set_random_seed(seed)
    samples = proposal.sample(num_samples)

    importance_weights = tf.reduce_sum(target.log_prob(samples) - proposal.log_prob(samples), axis=1)

    index = tf.argmax(importance_weights)

    best_sample = samples[index:index + 1, :]
    #print(index, seed)
    
    if np.log(index + 1) / np.log(2) > n_coding_bits:
        raise Exception("Not enough bits to code importance sample!")
    
    bitcode = to_bit_string(index.numpy(), n_coding_bits)

    return best_sample, bitcode

def decode_importance_sample(sample_index, proposal, seed):
    
    # Make sure the distributions have the correct type
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
        
    dim = len(proposal.loc)
    
    index = from_bit_string(sample_index)
    
    #print(index, seed)
    tf.random.set_random_seed(seed)
    samples = proposal.sample(index + 1)
    
    return samples[index:, ...]

def code_grouped_importance_sample(target, 
                                   proposal, 
                                   seed,
                                   n_bits_per_group,
                                   max_group_size_bits=4,
                                   dim_kl_bit_limit=12):
    
    # Make sure the distributions have the correct type
    if target.dtype is not tf.float32:
        raise Exception("Target datatype must be float32!")
        
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
    
    num_dimensions = np.prod(proposal.loc.shape.as_list())
    
    # rescale proposal by the proposal
    p_loc = tf.reshape(tf.zeros_like(proposal.loc), [-1])
    p_scale = tf.reshape(tf.ones_like(proposal.scale), [-1])
    
    # rescale target by the proposal
    t_loc = tf.reshape((target.loc - proposal.loc) / proposal.scale, [-1])
    t_scale = tf.reshape(target.scale / proposal.scale, [-1])
    
    # If we're going to do importance sampling, separate out dimensions with large KL,
    # we'll deal with them separately.
    kl_bits = tf.reshape(tfd.kl_divergence(target, proposal), [-1]) / np.log(2)

    t_loc = tf.where(kl_bits <= dim_kl_bit_limit, t_loc, p_loc)
    t_scale = tf.where(kl_bits <= dim_kl_bit_limit, t_scale, p_scale)

    # We'll send the quantized samples for dimensions with high KL
    outlier_indices = tf.where(kl_bits > dim_kl_bit_limit)

    target_samples = tf.reshape(target.sample(), [-1])

    # Select only the bits of the sample that are relevant
    outlier_samples = tf.gather_nd(target_samples, outlier_indices)

    # Halve precision
    outlier_samples = tfq.quantize(outlier_samples, -30, 30, tf.quint16).output

    outlier_extras = (outlier_indices, outlier_samples)

    kl_divs = tf.reshape(
        tfd.kl_divergence(tfd.Normal(loc=t_loc, scale=t_scale), 
                          tfd.Normal(loc=p_loc, scale=p_scale)), [-1]).numpy()

    group_start_indices = [0]
    group_kls = []

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
    prev_idx = -2
    twice_same = False

    while idx < num_dimensions - 1:

        if twice_same and idx == prev_idx:

            print("oh no: {}".format(idx))
            return

        twice_same = idx == prev_idx
        prev_idx = idx

        idx = idx + 1
        current_group_size = current_group_size + 1
        current_group_kl = current_group_kl + kl_divs[idx]

        #num_group_samps = np.ceil(np.exp(current_group_kl))

        if not ( np.log(current_group_size) / np.log(2) < max_group_size_bits and 
                 current_group_kl < n_bits_per_group * np.log(2) - 1):

            group_start_indices.append(idx)
            group_kls.append((current_group_kl - kl_divs[idx]) / np.log(2))

            current_group_size = 0
            current_group_kl = 0

            idx = idx - 1


        if idx == len(kl_divs) - 1:
            group_kls.append(current_group_kl / np.log(2))
        
    print(np.max(group_kls))
    
    # ====================================================================== 
    # Sample each group
    # ====================================================================== 
    
    results = []
    
    group_start_indices += [num_dimensions] 
    
    for i in tqdm(range(len(group_start_indices) - 1)):
        
        start_idx = group_start_indices[i]
        end_idx = group_start_indices[i + 1]
        
        result = code_importance_sample(
            target=tfd.Normal(loc=t_loc[start_idx:end_idx],
                              scale=t_scale[start_idx:end_idx]), 

            proposal=tfd.Normal(loc=p_loc[start_idx:end_idx],
                                scale=p_scale[start_idx:end_idx]), 

            n_coding_bits=n_bits_per_group,
            seed=i + seed)
        
        results.append(result)
        
    samples, codes = zip(*results)
    
    bitcode = ''.join(codes)
    sample = tf.concat(samples, axis=1)
    
    # Rescale the sample
    sample = tf.reshape(proposal.scale, [-1]) * sample + tf.reshape(proposal.loc, [-1])
    
    sample = tf.where(kl_bits <= dim_kl_bit_limit, tf.squeeze(sample), target_samples)
    
    return sample, bitcode, group_start_indices, outlier_extras

def decode_grouped_importance_sample(bitcode, 
                                     group_start_indices,
                                     proposal, 
                                     n_bits_per_group,
                                     seed,
                                     outlier_indices,
                                     outlier_samples):
    
    # Make sure the distributions have the correct type
    if proposal.dtype is not tf.float32:
        raise Exception("Proposal datatype must be float32!")
    
    num_dimensions = np.prod(proposal.loc.shape.as_list())
    
    # ====================================================================== 
    # Decode each group
    # ====================================================================== 
                
    samples = []
    
    group_start_indices += [num_dimensions]
    
    p_loc = tf.reshape(tf.zeros_like(proposal.loc), [-1])
    p_scale = tf.reshape(tf.ones_like(proposal.scale), [-1])

    for i in tqdm(range(len(group_start_indices) - 1)):
        
        samples.append(decode_importance_sample(
            sample_index=bitcode[n_bits_per_group * i: n_bits_per_group * (i + 1)],
            
            proposal=tfd.Normal(loc=p_loc[group_start_indices[i]:group_start_indices[i + 1]],
                                scale=p_scale[group_start_indices[i]:group_start_indices[i + 1]]), 
            seed=i + seed))
        
    sample = tf.concat(samples, axis=1)
    
    # Rescale the sample
    sample = tf.reshape(proposal.scale, [-1]) * sample + tf.reshape(proposal.loc, [-1])
    sample = tf.squeeze(sample)
    
    # Dequantize outliers
    outlier_samples = tfq.dequantize(tf.cast(outlier_samples, tf.quint16), -30, 30)
    
    # Add back the quantized outliers
    updates = tf.scatter_nd(tf.reshape(outlier_indices, [-1, 1]), 
                            tf.reshape(outlier_samples, [-1]), 
                            sample.shape.as_list())
                            
    sample = tf.where(tf.equal(updates, 0), sample, updates)
    
    return sample