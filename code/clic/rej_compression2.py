import argparse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt

from tqdm import tqdm

tf.enable_eager_execution()

@profile
def rejection_sample(p_loc,
                     p_scale,
                     q_loc,
                     q_scale,
                     num_latents,
                     num_samples,
                     show_plot,
                     n_points=20):

    p = tfd.Normal(loc=p_loc * tf.ones(num_latents), scale=p_scale * tf.ones(num_latents))
    q = tfd.Normal(loc=q_loc * tf.ones(num_latents), scale=q_scale * tf.ones(num_latents))

    plot_points = np.linspace(-10., 10., 200)


    p_mass = tf.concat(([0.], [1. / (n_points - 2)] * (n_points - 2), [0.]), axis=0)
    quantiles = np.linspace(0., 1., n_points + 1)
    open_sections = q.quantile(quantiles[1:-1])

    open_cdf = p.cdf(open_sections)
    cdfs = tf.concat(([0.], open_cdf, [1.]), axis=0)
    probs = cdfs[1:] - cdfs[:-1]

    rejection_samples = []
    for i in tqdm(range(num_samples)):

        p_i = tf.zeros((n_points))

        accepted = False
        j = 0

        while not accepted:
            p_star = tf.reduce_sum(p_i)

            alpha_i = tf.where(p_mass - p_i < (1 - p_star) * probs, p_mass - p_i, (1 - p_star) * probs)
            sample = p.sample()

            bucket = tf.concat((tf.reshape(tf.where(sample < open_sections),[-1]), [n_points-1]), axis=0)[0]

            beta = (alpha_i[bucket]) / ((1 - p_star) * probs[bucket])
            accepted = (tf.random.uniform(()) < beta)

            p_i += alpha_i
            j += 1

        #print(j)
        rejection_samples.append(sample)

    if show_plot:
        plt.hist(rejection_samples, range=(-3., 8.), normed=True, bins=100)
        plt.plot(plot_points, q.prob(plot_points), 'r')
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayes By Backprop models')

    parser.add_argument('--p_loc', type=float, default=0)
    parser.add_argument('--p_scale', type=float, default=1)
    parser.add_argument('--q_loc', type=float, default=0)
    parser.add_argument('--q_scale', type=float, default=1)
    parser.add_argument('--show_plot', action='store_true', default=False)
    parser.add_argument('--num_samples', type=int, default=100)

    args = parser.parse_args()

    rejection_sample(p_loc=args.p_loc,
                     p_scale=args.p_scale,
                     q_loc=args.q_loc,
                     q_scale=args.q_scale,
                     show_plot=args.show_plot,
                     num_samples=args.num_samples)
