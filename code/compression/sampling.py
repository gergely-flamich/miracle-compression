"""
This file implements sampling methods for Miracle.

Currently implemented:
  - A* sampling
  - Rejection Sampling (needs work)
"""

# ==============================================================================
# Imports
# ==============================================================================

import numpy as np
from scipy.stats import truncnorm, norm

from sobol_seq import i4_sobol_generate

np.seterr(all="raise", under="warn")

import line_profiler

# Priority Queue
import heapq as hq

from data_structures import IntervalTree

# ==============================================================================
# Helper functions
# ==============================================================================

class SamplingError(Exception):
    pass

# ------------------------------------------------------------------------------
# Helper functions for A* sampling
# ------------------------------------------------------------------------------

def normal_normal_log_diff(x, mu_prop, sigma_prop, mu_target, sigma_target):
    return norm.logpdf(x, mu_target, sigma_target) - norm.logpdf(x, mu_prop, sigma_prop)

def normal_normal_region_bound(a, b, mu_prop, sigma_prop, mu_target, sigma_target):

    # The log difference is technically not boundable in this case, but
    # in reality, we will never tend to infinity with the sampling points
#     if sigma_prop < sigma_target:
#         raise SamplingError("Log difference is not boundable!")

    # o(x) is convex
    if sigma_prop < sigma_target:
        
        a = a if a > -np.inf else mu_prop - 3 * sigma_prop
        b = b if b < np.inf else mu_prop + 3 * sigma_prop
        
        a_bound = normal_normal_log_diff(a, mu_prop, sigma_prop, mu_target, sigma_target)
        b_bound = normal_normal_log_diff(b, mu_prop, sigma_prop, mu_target, sigma_target)

        return np.maximum(a_bound, b_bound)
    # o(x) is concave
    else:
        if a < mu_target < b:
            # The log difference attains its maximum here
            max_x = (mu_prop * sigma_target**2 - mu_target * sigma_prop**2) / (sigma_target**2 - sigma_prop**2)

            return normal_normal_log_diff(max_x, mu_prop, sigma_prop, mu_target, sigma_target)
        else:
            # Both bounds can never be none, because the above branch will always get executed in the case
            # a = -inf, b = inf since the mean will always be in (-inf, inf)
            a_bound = None if a == -np.inf else normal_normal_log_diff(a, mu_prop, sigma_prop, mu_target, sigma_target)
            b_bound = None if b == np.inf else normal_normal_log_diff(b, mu_prop, sigma_prop, mu_target, sigma_target)

            if a_bound is None:
                return b_bound
            elif b_bound is None:
                return a_bound
            else:
                return np.maximum(a_bound, b_bound)

# ------------------------------------------------------------------------------
# Gumbel related stuff
# ------------------------------------------------------------------------------

def gumbel_pdf(x, loc=0.):

    z = -(x - loc)
    return np.exp(z - np.exp(z))

def gumbel_cdf(x, loc=0.):

    z = -(x - loc)
    return np.exp(-np.exp(z))

def gumbel_inv_cdf(x, loc=0.):
    return -np.log(-np.log(x)) + loc

def gumbel_sample(loc=0., size=None):

    u = np.random.uniform(size=size)
    return gumbel_inv_cdf(u, loc=loc)

# ------------------------------------------------------------------------------
# Truncated Gumbel related stuff
# ------------------------------------------------------------------------------

def trunc_gumbel_pdf(x, trunc, loc=0.):

    z = -(np.minimum(x, trunc) - loc)

    return (x < trunc) * np.exp(z - np.exp(z) + np.exp(-trunc + loc))

def trunc_gumbel_cdf(x, trunc, loc=0.):

    z = -(np.minimum(x, trunc) - loc)

    return np.exp(-np.exp(z) + np.exp(-trunc + loc))

def trunc_gumbel_inv_cdf(x, trunc, loc=0.):

    return -np.log(np.exp(-trunc + loc) - np.log(x)) + loc

def trunc_gumbel_sample(trunc, loc=0., size=None):

    u = np.random.uniform(size=size)
    return trunc_gumbel_inv_cdf(u, trunc=trunc, loc=loc)


# ==============================================================================
# A* sampling (Cannot be coded)
# ==============================================================================

def a_star_sample(prop_trunc_samp, prop_log_mass, log_diff, region_bound, seed, code_path=None):
    """
    prop_log_mass - function taking 2 arguments a, b and calculates \log\int_a^b i(x) dx

    prop_trunc_samp - function taking 2 arguments a, b and samples from the truncated Gibbs
                      distribuiton of i(x), i.e. it samples X ~ exp(i(x))/Z where
                      x \in [a, b] and Z = \int_a^b exp(i(x)) dx

    log_diff - function taking 1 argument, is o(x) in the paper

    region_bound - function taking 2 arguments a, b; is M(B) in the paper
    """

    np.random.seed(seed)

    # Initialisation
    lower_bound = -np.inf
    samp = None
    k = 0

    queue = []

    G = []
    X = []
    B = []
    M = []

    # First split:
    # generate maximum and maximum location
    b_1 = (-np.inf, np.inf)

    g_1 = gumbel_sample(loc=prop_log_mass(*b_1))
    x_1 = prop_trunc_samp(*b_1)

    m_1 = region_bound(*b_1)

    # If we are decoding and no directions are provided,
    # it means we should return the root node
    if code_path is not None and len(code_path) == 0:
        return x_1

    # Store G_1, X_1, B_1, M_1
    G.append(g_1)
    X.append(x_1)
    B.append(b_1)
    M.append(m_1)

    # The heapq implementation of the heap is a min heap not a max heap!
    hq.heappush(queue, (-(g_1 + m_1), 0, ""))

    # Run A* search
    # Note: since we are using the negative of the upper bounds
    # we have to negate it again at this check
    while len(queue) > 0 and lower_bound < -min(queue)[0]:

        # Get the bound with the highest priority
        _, p, path = hq.heappop(queue)

        # Calculate new proposed lower bound based on G_p
        lower_bound_p = G[p] + log_diff(X[p])

        # Check if the lower bound can be raised
        if lower_bound < lower_bound_p:

            lower_bound = lower_bound_p
            samp = X[p]
            samp_path = path

        # Partition the space: split the current interval by X_p
        L = (B[p][0], X[p])
        R = (X[p], B[p][1])

        # Go down the heap / partitions
        for C, direction in zip([L, R], ['0', '1']):

            # TODO: check if this is a sufficiently good empty set condition
            if not C[0] == C[1]:

                k += 1

                b_k = C
                g_k = trunc_gumbel_sample(loc=prop_log_mass(*C),
                                          trunc=G[p])
                x_k = prop_trunc_samp(*C)

                # If the path to the sample matches, then we are done and the
                # current sample is the one we wanted
                if code_path is not None and code_path == path + direction:
                    return x_k

                # Store B_k, G_k, X_k
                B.append(b_k)
                G.append(g_k)
                X.append(x_k)

                # Check if there is a point in continuing the search along this path
                if lower_bound < g_k + M[p]:
                    m_k = region_bound(*b_k)
                    M.append(m_k)

                    if lower_bound < g_k + m_k:

                        hq.heappush(queue, (-(g_k + m_k), k, path + direction))
                else:
                    # We push a non-informative bound here, so that the length of M
                    # is the same as the rest
                    M.append(0)

    return lower_bound, samp, samp_path

# ==============================================================================
# A* sampling can be coded
# ==============================================================================

def a_star_sample_codable(prop_log_mass,
                          log_diff,
                          samp_tree,
                          region_bound,
                          prop_cdf,
                          prop_inv_cdf,
                          eps=1e-4):
    """
    prop_log_mass - function taking 2 arguments a, b and calculates \log\int_a^b i(x) dx

    log_diff - function taking 1 argument, is o(x) in the paper

    region_bound - function taking 2 arguments a, b; is M(B) in the paper
    """

    if type(samp_tree) != IntervalTree:
        raise SamplingError("samp_tree must be an IntervalTree!")

    # Initialisation
    lower_bound = -np.inf
    samp = None
    k = 0

    queue = []

    samp_idx = 0

    G = []
    X = []
    B = []
    M = []

    # First split:
    # generate maximum and maximum location
    b_1 = (-np.inf, np.inf)

    g_1 = gumbel_sample(loc=prop_log_mass(*b_1))

    x_1, idx = samp_tree.between(low=b_1[0],
                                 high=b_1[1],
                                 cdf=prop_cdf,
                                 inv_cdf=prop_inv_cdf)

    m_1 = region_bound(*b_1)

    # Store G_1, X_1, B_1, M_1
    G.append(g_1)
    X.append(x_1)
    B.append(b_1)
    M.append(m_1)

    # The heapq implementation of the heap is a min heap not a max heap!
    hq.heappush(queue, (-(g_1 + m_1), 0, idx))

    # Run A* search
    # Note: since we are using the negative of the upper bounds
    # we have to negate it again at this check
    while len(queue) > 0 and lower_bound < -min(queue)[0]:

        # Get the bound with the highest priority
        _, p, p_idx = hq.heappop(queue)

        # Calculate new proposed lower bound based on G_p
        lower_bound_p = G[p] + log_diff(X[p])

        # Check if the lower bound can be raised
        if lower_bound < lower_bound_p:

            lower_bound = lower_bound_p

            idx = p_idx
            samp = X[p]

        # Partition the space: split the current interval by X_p
        L = (B[p][0], X[p])
        R = (X[p], B[p][1])

        # Go down the heap / partitions
        for C, direction in zip([L, R], ['0', '1']):

            # TODO: check if this is a sufficiently good empty set condition
            if not np.abs(C[0] - C[1]) < eps:

                k += 1

                b_k = C

                g_k = trunc_gumbel_sample(loc=prop_log_mass(*C),
                                          trunc=G[p])

                x_k, k_idx = samp_tree.between(low=b_k[0],
                                               high=b_k[1],
                                               cdf=prop_cdf,
                                               inv_cdf=prop_inv_cdf)

                # Store B_k, G_k, X_k
                B.append(b_k)
                G.append(g_k)
                X.append(x_k)

                # Check if there is a point in continuing the search along this path
                if lower_bound < g_k + M[p]:
                    m_k = region_bound(*b_k)
                    M.append(m_k)

                    if lower_bound < g_k + m_k:

                        hq.heappush(queue, (-(g_k + m_k), k, k_idx))
                else:
                    # We push a non-informative bound here, so that the length of M
                    # is the same as the rest
                    M.append(0)

    return lower_bound, samp, idx
