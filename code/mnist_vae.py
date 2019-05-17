import tensorflow as tf
from sonnet import AbstractModule, Linear, BatchFlatten, BatchReshape, reuse_variables
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt


class MnistVAE(AbstractModule):

    _allowed_likelihoods = set(["bernoulli", "gaussian"])

    def __init__(self,
                 hidden_units=100,
                 num_latents=2,
                 data_likelihood="bernoulli",
                 name="mnist_vae"):

        # Initialise the superclass
        super(MnistVAE, self).__init__(name=name)

        self._hidden_units = hidden_units
        self._num_latents = num_latents

        if data_likelihood not in self._allowed_likelihoods:
            raise tf.errors.InvalidArgumentError("data_likelihood must be one of {}"
                                                 .format(self._allowed_likelihoods))

        self._data_likelihood=data_likelihood

        self._latent_prior = tfd.Normal(
            loc=tf.zeros([num_latents]),
            scale=tf.ones([num_latents])
        )


    @property
    def log_prob(self):
        self._ensure_is_connected()

        return tf.reduce_sum(self._log_prob)

    @property
    def kl_divergence(self):
        """
        Calculates the KL divergence between the current variational posterior and the prior:
        KL[ q(z | theta) || p(z) ]
        """

        self._ensure_is_connected()

        return tf.reduce_sum(
            tfd.kl_divergence(self._q, self._latent_prior))

    @reuse_variables
    def encode(self, inputs):
        """
        The encoder will predict the variational
        posterior q(z | x) = N(z | mu(x), sigma(x)).

        This will be done by using a two-headed network

        Note: reuse_variables is required so that when we call
        encode on its own, it uses the trained weights
        """

        # Turn the images into vectors
        flatten = BatchFlatten()
        flattened = flatten(inputs)

        # First fully connected layer
        linear1 = Linear(output_size=self._hidden_units,
                         name="encoder_linear1")

        pre_activations = linear1(flattened)
        activations = tf.nn.relu(pre_activations)

        # Second fully connected layer
        linear2 = Linear(output_size=self._hidden_units,
                         name="encoder_linear2")

        pre_activations = linear2(activations)
        activations = tf.nn.relu(pre_activations)

        # Mean-head
        linear_mu = Linear(output_size=self._num_latents,
                           name="encoder_linear_mu")
        mu = linear_mu(activations)

        # Variance-head
        linear_sigma = Linear(output_size=self._num_latents,
                              name="encoder_linear_sigma")
        sigma = tf.nn.softplus(linear_sigma(activations))

        return mu, sigma


    @reuse_variables
    def decode(self, latents):
        """
        Note: reuse_variables is required so that when we call
        encode on its own, it uses the trained weights

        """

        # First fully connected layer
        linear1 = Linear(output_size=self._hidden_units,
                         name="decoder_linear1")

        pre_activations = linear1(latents)
        activations = tf.nn.relu(pre_activations)

        # Second fully connected layer
        linear2 = Linear(output_size=self._hidden_units,
                         name="decoder_linear2")

        pre_activations = linear2(activations)
        activations = tf.nn.relu(pre_activations)

        # Predict the means of a Bernoulli
        linear_out = Linear(output_size=28 * 28,
                            name="decoder_out")

        logits = linear_out(activations)

        reshaper = BatchReshape(shape=(28, 28))
        logits = reshaper(logits)

        return logits


    def _build(self, inputs):
        """
        Build standard VAE:
        1. Encode input -> latent mu, sigma
        2. Sample z ~ N(z | mu, sigma)
        3. Decode z -> output Bernoulli means
        4. Sample o ~ Bernoulli(o | z)
        """

        # Get the means and variances of variational posteriors
        q_mu, q_sigma = self.encode(inputs)

        q = tfd.Normal(loc=q_mu, scale=q_sigma)

        latents = q.sample()

        # Needed to calculate KL term
        self._q = q

        # Get Bernoulli likelihood means
        p_logits = self.decode(latents)

        if self._data_likelihood == "bernoulli":
            p = tfd.Bernoulli(logits=p_logits)

        else:
            p = tfd.Normal(loc=p_logits, scale=tf.ones_like(p_logits))

        self._log_prob = p.log_prob(inputs)

        return p.sample()
