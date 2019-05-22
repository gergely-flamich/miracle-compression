import tensorflow as tf
from sonnet import AbstractModule, Linear, BatchFlatten, BatchReshape, reuse_variables, \
    Conv2D, BatchNorm
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.pyplot as plt


class ClicVAE(AbstractModule):

    def __init__(self,
                 name="clic_vae"):

        # Initialise the superclass
        super(ClicVAE, self).__init__(name=name)


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

        p = tfd.Normal(loc=p_logits, scale=tf.ones_like(p_logits))

        self._log_prob = p.log_prob(inputs)

        return p_logits


# ==============================================================================


class ClicCNN(ClicVAE):

    def __init__(self,
                 top_conv_channels=128,
                 bottom_conv_channels=192,
                 name="clic_cnn_vae"):

        # Initialise the superclass
        super(ClicCNN, self).__init__(name=name)

        self._top_conv_channels = top_conv_channels
        self._bottom_conv_channels = bottom_conv_channels

    @reuse_variables
    def encode(self, inputs):
        """
        The encoder will predict the variational
        posterior q(z | x) = N(z | mu(x), sigma(x)).

        This will be done by using a two-headed network

        Note: reuse_variables is required so that when we call
        encode on its own, it uses the trained weights
        """

        # First convolution layer
        self.conv1 = Conv2D(output_channels=self._top_conv_channels,
                            kernel_shape=(5, 5),
                            stride=2,
                            padding="VALID",
                            name="encoder_conv1")

        activations = tf.contrib.layers.gdn(self.conv1(inputs[..., tf.newaxis]),
                                            name="encoder_gdn1")

        # Second convolution layer
        self.conv2 = Conv2D(output_channels=self._top_conv_channels,
                            kernel_shape=(5, 5),
                            stride=2,
                            padding="VALID",
                            name="encoder_conv2")

        activations = tf.contrib.layers.gdn(self.conv2(activations),
                                            name="encoder_gdn2")

        # Third convolution layer
        self.conv3 = Conv2D(output_channels=self._top_conv_channels,
                            kernel_shape=(5, 5),
                            stride=2,
                            padding="VALID",
                            name="encoder_conv3")

        activations = tf.contrib.layers.gdn(self.conv3(activations),
                                            name="encoder_gdn3")

        # Latent convolution layer
        self.conv_mu = Conv2D(output_channels=self._bottom_conv_channels,
                              kernel_shape=(5, 5),
                              padding="VALID",
                              name="encoder_conv_mu")

        mu = self.conv_mu(activations)

        # Variance-head
        conv_sigma = Conv2D(output_channels=self._bottom_conv_channels,
                            kernel_shape=(5, 5),
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

        deconv2 = self.conv3.transpose()
        activations = tf.contrib.layers.gdn(deconv2(activations),
                                            inverse=True,
                                            name="decoder_gdn2")

        deconv3 = self.conv2.transpose()
        activations = tf.contrib.layers.gdn(deconv3(activations),
                                            inverse=True,
                                            name="decoder_gdn3")

        deconv4 = self.conv3.transpose()
        activations = tf.contrib.layers.gdn(deconv4(activations),
                                            inverse=True,
                                            name="decoder_gdn4")


        logits = tf.squeeze(activations)

        return logits


