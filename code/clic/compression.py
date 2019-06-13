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

# Needed for compression as the common source of randomness
from sobol_seq import i4_sobol_generate
from scipy.stats import norm

from sampling import IntervalTree, a_star_sample_codable, normal_normal_log_diff, normal_normal_region_bound


# Define constants
num_samples = 2**15


p_mu_2 = np.load("p_mu.npy")
p_sigma_2 = np.load("p_sigma.npy")
q_mu_2 = np.load("q_mu.npy")
q_sigma_2 = np.load("q_sigma.npy")

def coded_sample(prior_loc, prior_scale, post_loc, post_scale, samp_tree):

    # Define functions to be used during the sampling procedure
    prop_log_mass = lambda a, b: np.log(norm.cdf(b, loc=prior_loc, scale=prior_scale) - \
                                        norm.cdf(a, loc=prior_loc, scale=prior_scale))

    log_diff = lambda x: normal_normal_log_diff(x, prior_loc, prior_scale, post_loc, post_scale)

    region_bound = lambda a, b: normal_normal_region_bound(a,
                                                           b,
                                                           prior_loc,
                                                           prior_scale,
                                                           post_loc,
                                                           post_scale)

    prop_cdf = lambda x: norm.cdf(x, loc=prior_loc, scale=prior_scale)
    prop_inv_cdf = lambda x: norm.ppf(x, loc=prior_loc, scale=prior_scale)

    # Draw the sample
    return a_star_sample_codable(prop_log_mass=prop_log_mass,
                                 log_diff=log_diff,
                                 samp_tree=samp_tree,
                                 prop_cdf=prop_cdf,
                                 prop_inv_cdf=prop_inv_cdf,
                                 region_bound=region_bound,
                                 eps=1e-2)




samp_tree = IntervalTree(num_nodes=num_samples)

bad_latents = []
samples = []

np.random.seed(10)

try:
    for i in tqdm(range(len(p_mu_2))):

        if np.abs(p_mu_2[i] - q_mu_2[i]) / p_sigma_2[i] > 1.:
            bad_latents.append(i)
        else:
            samp = coded_sample(p_mu_2[i], p_sigma_2[i], q_mu_2[i], q_sigma_2[i], samp_tree)

            samples.append(samp)

except:
    print("Failed at {}".format(i))
