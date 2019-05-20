import numpy as np

import tensorflow as tf
from sonnet import AbstractModule, Linear, BatchFlatten, BatchReshape, reuse_variables, \
    Conv2D, BatchNorm
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

        # Needed for batch norm
        self._is_training = True

    def training_finished(self):
        """
        Required so that batch norm can be turned on/ off appropriately
        """
        self._ensure_is_connected()

        self._is_training = False

    @reuse_variables
    def encode(self, inputs):
        raise NotImplementedError

    @reuse_variables
    def decode(self, latents):
        raise NotImplementedError

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
            p_logits = tf.nn.sigmoid(p_logits)

            p = tfd.Normal(loc=p_logits, scale=tf.ones_like(p_logits))

        self._log_prob = p.log_prob(inputs)

        return p_logits


# ==============================================================================

class MnistFC_VAE(MnistVAE):

    def __init__(self,
                 hidden_units=100,
                 num_latents=2,
                 data_likelihood="bernoulli",
                 name="mnist_fc_vae"):

        # Initialise the superclass
        super(MnistFC_VAE, self).__init__(name=name,
                                          hidden_units=hidden_units,
                                          num_latents=num_latents,
                                          data_likelihood=data_likelihood)


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

        bn1 = BatchNorm(update_ops_collection=None)
        activations = bn1(activations, is_training=self._is_training)

        # Second fully connected layer
        linear2 = Linear(output_size=self._hidden_units,
                         name="encoder_linear2")

        pre_activations = linear2(activations)
        activations = tf.nn.relu(pre_activations)

        bn2 = BatchNorm(update_ops_collection=None)
        activations = bn2(activations, is_training=self._is_training)

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

        bn1 = BatchNorm(update_ops_collection=None)
        activations = bn1(activations, is_training=self._is_training)


        # Second fully connected layer
        linear2 = Linear(output_size=self._hidden_units,
                         name="decoder_linear2")

        pre_activations = linear2(activations)
        activations = tf.nn.relu(pre_activations)

        bn2 = BatchNorm(update_ops_collection=None)
        activations = bn2(activations, is_training=self._is_training)

        # Predict the means of a Bernoulli
        linear_out = Linear(output_size=28 * 28,
                            name="decoder_out")

        logits = linear_out(activations)

        reshaper = BatchReshape(shape=(28, 28))
        logits = reshaper(logits)

        return logits


# ==============================================================================


class MnistFC_CNN_VAE(MnistVAE):

    def __init__(self,
                 hidden_units=100,
                 num_latents=2,
                 data_likelihood="bernoulli",
                 name="mnist_fc_vae"):

        # Initialise the superclass
        super(MnistFC_CNN_VAE, self).__init__(name=name,
                                       hidden_units=hidden_units,
                                       num_latents=num_latents,
                                       data_likelihood=data_likelihood)


    @reuse_variables
    def encode(self, inputs):
        """
        The encoder will predict the variational
        posterior q(z | x) = N(z | mu(x), sigma(x)).

        This will be done by using a two-headed network

        Note: reuse_variables is required so that when we call
        encode on its own, it uses the trained weights
        """

        # Go from N x 28 x 28 x 1 -> N x 14 x 14 x 16
        self.conv1 = Conv2D(output_channels=16,
                            kernel_shape=(5, 5),
                            stride=2,
                            name="encoder_conv1")

        activations = tf.nn.relu(self.conv1(inputs[..., tf.newaxis]))

        bn1 = BatchNorm(update_ops_collection=None,
                        name="encoder_bn1")
        activations = bn1(activations, is_training=self._is_training)

        # Go from N x 14 x 14 x 16 -> N x 7 x 7 x 32
        self.conv2 = Conv2D(output_channels=32,
                            kernel_shape=(3, 3),
                            stride=2,
                            name="encoder_conv2")


        activations = tf.nn.relu(self.conv2(activations))

        bn2 = BatchNorm(update_ops_collection=None,
                        name="encoder_bn2")
        activations = bn2(activations, is_training=self._is_training)

        # Turn the convolved images into vectors
        flatten = BatchFlatten()
        flattened = flatten(activations)

        # First fully connected layer
        linear1 = Linear(output_size=self._hidden_units,
                         name="encoder_linear1")

        pre_activations = linear1(flattened)
        activations = tf.nn.relu(pre_activations)

        bn3 = BatchNorm(update_ops_collection=None,
                        name="encoder_bn3")
        activations = bn3(activations, is_training=self._is_training)

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

        bn1 = BatchNorm(update_ops_collection=None,
                        name="decoder_bn1")
        activations = bn1(activations, is_training=self._is_training)

        # Predict the means of the data likelihood
        linear2 = Linear(output_size=7 * 7 * 32,
                         name="decoder_linear2")

        activations = tf.nn.relu(linear2(activations))

        bn2 = BatchNorm(update_ops_collection=None,
                        name="decoder_bn1")
        activations = bn2(activations, is_training=self._is_training)

        reshaper = BatchReshape(shape=(7, 7, 32))
        activations = reshaper(activations)

        # Go from N x 7 x 7 x 32 -> N x 14 x 14 x 16
        deconv1 = self.conv2.transpose()

        activations = tf.nn.relu(deconv1(activations))

        bn3 = BatchNorm(update_ops_collection=None,
                        name="decoder_bn3")
        activations = bn3(activations, is_training=self._is_training)

        # Go from N x 14 x 14 x 16 -> N x 28 x 28 x 1
        deconv2 = self.conv1.transpose()

        logits = tf.squeeze(deconv2(activations))

        return logits


# ==============================================================================

class MnistCNN_VAE(MnistVAE):

    def __init__(self,
                 hidden_units=100,
                 num_latents=2,
                 data_likelihood="bernoulli",
                 name="mnist_fc_vae"):

        # Initialise the superclass
        super(MnistCNN_VAE, self).__init__(name=name,
                                           hidden_units=hidden_units,
                                           num_latents=num_latents,
                                           data_likelihood=data_likelihood)

    @reuse_variables
    def encode(self, inputs):
        """
        The encoder will predict the variational
        posterior q(z | x) = N(z | mu(x), sigma(x)).

        This will be done by using a two-headed network

        Note: reuse_variables is required so that when we call
        encode on its own, it uses the trained weights
        """

        # Go from N x 28 x 28 x 1 -> N x 14 x 14 x 16
        self.conv1 = Conv2D(output_channels=16,
                            kernel_shape=(5, 5),
                            stride=2,
                            name="encoder_conv1")

        activations = tf.contrib.layers.gdn(self.conv1(inputs[..., tf.newaxis]),
                                            name="encoder_gdn1")

        # Go from N x 14 x 14 x 16 -> N x 7 x 7 x 32
        self.conv2 = Conv2D(output_channels=32,
                            kernel_shape=(3, 3),
                            stride=2,
                            name="encoder_conv2")

        activations = tf.contrib.layers.gdn(self.conv2(activations),
                                            name="encoder_gdn2")

        # Go from N x 7 x 7 x 32 -> N x 1 x 1 x num_latents
        self.conv_mu = Conv2D(output_channels=self._num_latents,
                              kernel_shape=(7, 7),
                              padding="VALID",
                              name="encoder_conv_mu")

        mu = self.conv_mu(activations)

        # Variance-head
        conv_sigma = Conv2D(output_channels=self._num_latents,
                            kernel_shape=(7, 7),
                            padding="VALID",
                            name="encoder_conv_var")

        sigma = tf.nn.softplus(conv_sigma(activations))

        return mu, sigma


    @reuse_variables
    def decode(self, latents):
        """
        Note: reuse_variables is required so that when we call
        encode on its own, it uses the trained weights

        """

        deconv1 = self.conv_mu.transpose()
        activations = tf.contrib.layers.gdn(deconv1(latents),
                                            inverse=True,
                                            name="decoder_gdn1")

        deconv2 = self.conv2.transpose()
        activations = tf.contrib.layers.gdn(deconv2(activations),
                                            inverse=True,
                                            name="decoder_gdn2")

        deconv3 = self.conv1.transpose()
        activations = tf.contrib.layers.gdn(deconv3(activations),
                                            inverse=True,
                                            name="decoder_gdn3")

        logits = tf.squeeze(activations)

        return logits


