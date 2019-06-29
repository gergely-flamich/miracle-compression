import argparse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import matplotlib.pyplot as plt

from tqdm import tqdm

@profile
def rejection_sample(p_loc,
                     p_scale,
                     q_loc,
                     q_scale,
                     num_samples):

    p = tfd.Normal(loc=p_loc, scale=p_scale)
    q = tfd.Normal(loc=q_loc, scale=q_scale)

    plot_points = np.linspace(-3., 8., 200)

    sess = tf.Session()

    n_points = 20
    p_mass = tf.concat(([0.], [1. / (n_points-2) for _ in range(n_points-2)], [0.]), axis=0)
    quantiles = np.linspace(0., 1., n_points+1)
    open_sections = q.quantile(quantiles[1:-1])

    p_i = tf.Variable(tf.zeros((n_points)))
    sess.run(tf.initialize_all_variables())
    p_star = tf.reduce_sum(p_i)
    open_cdf = p.cdf(open_sections)
    cdfs = tf.concat(([0.], open_cdf, [1.]), axis=0)
    probs = cdfs[1:] - cdfs[:-1]
    alpha_i = tf.where(p_mass - p_i < (1 - p_star) * probs, p_mass - p_i, (1 - p_star) * probs)
    sample = p.sample()
    bucket = tf.concat((tf.reshape(tf.where(sample < open_sections),[-1]), [n_points-1]), axis=0)[0]
    beta = (alpha_i[bucket]) / ((1 - p_star) * probs[bucket])
    accept = (tf.random.uniform(()) < beta)
    update_op = [p_i.assign(p_i + alpha_i)]

    rejection_samples = []
    for i in tqdm(range(num_samples)):
        sess.run(tf.initialize_all_variables())
        accepted = False
        s = 0.
        j = 0
        while not accepted:
            accepted, s = sess.run([accept, sample])
            sess.run(update_op)
            j += 1
        rejection_samples.append(s)


    plt.hist(rejection_samples, range=(-3., 8.), normed=True, bins=100)
    plt.plot(plot_points, sess.run(q.prob(plot_points)), 'r')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bayes By Backprop models')

    parser.add_argument('--p_loc', type=float, default=0)
    parser.add_argument('--p_scale', type=float, default=1)
    parser.add_argument('--q_loc', type=float, default=0)
    parser.add_argument('--q_scale', type=float, default=1)
    parser.add_argument('--num_samples', type=float, default=100)

    args = parser.parse_args()

    rejection_sample(p_loc=args.p_loc,
                     p_scale=args.p_scale,
                     q_loc=args.q_loc,
                     q_scale=args.q_scale,
                     num_samples=args.num_samples)
