import tensorflow as tf
from sonnet import AbstractModule, Linear, BatchFlatten, BatchReshape, reuse_variables, \
    Conv2D, BatchNorm
import tensorflow_probability as tfp
tfd = tfp.distributions

from miracle_modules import ConvDS

from utils import InvalidArgumentError

# ==============================================================================
# ==============================================================================
#
# Superclasses
#
# ==============================================================================
# ==============================================================================

class ClicVAE(AbstractModule):

    _allowed_priors = ["gaussian", "laplace"]
    _allowed_likelihoods = ["gaussian", "laplace"]
    _allowed_paddings = ["SAME", "VALID", "SAME_MIRRORED"]

    def __init__(self,
                 prior="gaussian",
                 likelihood="gaussian",
                 padding="SAME",
                 name="clic_vae"):

        # Initialise the superclass
        super(ClicVAE, self).__init__(name=name)

        if prior not in self._allowed_priors:
            raise tf.errors.InvalidArgumentError("prior must be one of {}"
                                                 .format(self._allowed_priors))
        if likelihood not in self._allowed_likelihoods:
            raise tf.errors.InvalidArgumentError("likelihood must be one of {}"
                                                 .format(self._allowed_likelihoods))
        if padding not in self._allowed_paddings:
            raise tf.errors.InvalidArgumentError("padding must be one of {}"
                                                 .format(self._allowed_paddings))


        self._prior_dist = prior
        self._likelihood = likelihood
        self._padding = padding


    def required_input_size(self, shape, for_padding="VALID"):
        raise NotImplementedError

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
        4. Sample o ~ Normal(o | z)
        """

        # Get the means and variances of variational posteriors
        q_mu, q_sigma = self.encode(inputs)

        if self._prior_dist == "gaussian":
            q = tfd.Normal(loc=q_mu, scale=q_sigma)

        elif self._prior_dist == "laplace":
            q = tfd.Laplace(loc=q_mu, scale=q_sigma)

        latents = q.sample()

        if self._prior_dist == "gaussian":
            self._latent_prior = tfd.Normal(loc=tf.zeros_like(latents), scale=tf.ones_like(latents))

        elif self._prior_dist == "laplace":
            self._latent_prior = tfd.Laplace(loc=tf.zeros_like(latents), scale=tf.ones_like(latents))


        # Needed to calculate KL term
        self._q = q

        # Get Bernoulli likelihood means
        p_logits = self.decode(latents)

        if self._likelihood == "gaussian":
            p = tfd.Normal(loc=p_logits, scale=tf.ones_like(p_logits))
        else:
            p = tfd.Laplace(loc=p_logits, scale=tf.ones_like(p_logits))

        self._log_prob = p.log_prob(inputs)

        return p_logits


# ==============================================================================

class ClicHierarchicalVAE(AbstractModule):

    _allowed_latent_dists = {
        "gaussian": tfd.Normal,
        "laplace": tfd.Laplace
    }

    _allowed_likelihoods = {
        "gaussian": tfd.Normal,
        "laplace": tfd.Laplace
    }

    _allowed_paddings = ["SAME", "VALID", "SAME_MIRRORED"]

    _latent_priors = []
    _latent_posteriors = []

    def __init__(self,
                 num_levels,
                 latent_dist="gaussian",
                 likelihood="gaussian",
                 standardized=False,
                 padding_first_level="SAME",
                 padding_second_level="SAME",
                 name="hierarchical_vae"):

        super(ClicHierarchicalVAE, self).__init__(name=name)

        self._num_levels = num_levels

        if latent_dist not in self._allowed_latent_dists:
            raise tf.errors.InvalidArgumentError("latent_dist must be one of {}"
                                                 .format(self._allowed_latent_dists))

        self._latent_dist = self._allowed_latent_dists[latent_dist]

        if likelihood not in self._allowed_likelihoods:
            raise tf.errors.InvalidArgumentError("likelihood must be one of {}"
                                                 .format(self._allowed_likelihoods))

        self._likelihood_dist = self._allowed_likelihoods[likelihood]

        if padding_first_level not in self._allowed_paddings:
            raise tf.errors.InvalidArgumentError("padding_first_level must be one of {}"
                                                 .format(self._allowed_paddings))
        self._padding_first_level = padding_first_level

        if padding_second_level not in self._allowed_paddings:
            raise tf.errors.InvalidArgumentError("padding_second_level must be one of {}"
                                                 .format(self._allowed_paddings))
        self._padding_second_level = padding_second_level


        self._standardized = standardized

    @reuse_variables
    def encode(self, inputs):
        raise NotImplementedError

    @reuse_variables
    def decode(self, latents):
        raise NotImplementedError


    @property
    def kl_divergence(self):
        self._ensure_is_connected()

        if (len(self._latent_posteriors) != self._num_levels or
            len(self._latent_priors) != self._num_levels):

            raise Exception("Need a full pass through the VAE to calculate KL!")

        klds = [tfd.kl_divergence(posterior, prior)
                for posterior, prior in zip(self._latent_posteriors, self._latent_priors)]

        return klds

    @property
    def log_prob(self):
        self._ensure_is_connected()

        if (len(self._latent_posteriors) != self._num_levels or
            len(self._latent_priors) != self._num_levels):

            raise Exception("Need a full pass through the VAE to calculate log probability!")

        return tf.reduce_sum(self._log_prob)


    def _build(self, inputs):

        latents = self.encode(inputs,
                              level=self._num_levels)

        decoded_loc, decoded_scale = self.decode(latents,
                                                 decode_level=self._num_levels)

        likelihood_variance = decoded_scale if not self._standardized else tf.ones_like(decoded_scale)

        self._likelihood = self._likelihood_dist(loc=decoded_loc,
                                                 scale=likelihood_variance)

        self._log_prob = self._likelihood.log_prob(inputs)

        return decoded_loc


# ==============================================================================
# ==============================================================================
#
# Experimental architectures
#
# ==============================================================================
# ==============================================================================


class ClicCNN(ClicVAE):

    def __init__(self,
                 top_conv_channels=128,
                 bottom_conv_channels=192,
                 prior="gaussian",
                 likelihood="gaussian",
                 padding="SAME",
                 name="clic_cnn_vae"):

        # Initialise the superclass
        super(ClicCNN, self).__init__(prior=prior,
                                      likelihood=likelihood,
                                      padding=padding,
                                      name=name)

        self._top_conv_channels = top_conv_channels
        self._bottom_conv_channels = bottom_conv_channels


    def required_input_size(self, shape, for_padding="VALID"):

            shape = self.conv_mu.required_input_size(shape, for_padding=for_padding)
            shape = self.conv_ds3.required_input_size(shape, for_padding=for_padding)
            shape = self.conv_ds2.required_input_size(shape, for_padding=for_padding)
            shape = self.conv_ds1.required_input_size(shape, for_padding=for_padding)

            return shape


    @reuse_variables
    def encode(self, inputs):
        """
        The encoder will predict the variational
        posterior q(z | x) = N(z | mu(x), sigma(x)).

        This will be done by using a two-headed network

        Note: reuse_variables is required so that when we call
        encode on its own, it uses the trained weights
        """

        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # First convolution layer
        self.conv_ds1 = ConvDS(output_channels=self._top_conv_channels,
                               kernel_shape=(5, 5),
                               padding=self._padding,
                               downsampling_rate=2,
                               use_gdn=True,
                               name="encoder_conv_ds1")


        # Second convolution layer
        self.conv_ds2 = ConvDS(output_channels=self._top_conv_channels,
                               kernel_shape=(5, 5),
                               padding=self._padding,
                               downsampling_rate=2,
                               use_gdn=True,
                               name="encoder_conv_ds2")

        # Third convolution layer
        self.conv_ds3 = ConvDS(output_channels=self._top_conv_channels,
                               kernel_shape=(5, 5),
                               padding=self._padding,
                               downsampling_rate=2,
                               use_gdn=True,
                               name="encoder_conv_ds3")

        # Latent distribution moment predictiors
        # Mean
        self.conv_mu = ConvDS(output_channels=self._bottom_conv_channels,
                               kernel_shape=(5, 5),
                               padding=self._padding,
                               downsampling_rate=2,
                               use_gdn=False,
                               name="encoder_conv_mu")

        # Covariance
        conv_sigma = ConvDS(output_channels=self._bottom_conv_channels,
                               kernel_shape=(5, 5),
                               padding=self._padding,
                               downsampling_rate=2,
                               use_gdn=False,
                               name="encoder_conv_sigma")

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        activations = self.conv_ds3(self.conv_ds2(self.conv_ds1(inputs)))
        mu = self.conv_mu(activations)
        sigma = tf.nn.softplus(conv_sigma(activations))

        return mu, sigma


    @reuse_variables
    def decode(self, latents):
        """
        Note: reuse_variables is required so that when we call
        encode on its own, it uses the trained weights

        """

        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        deconv = self.conv_mu.transpose()

        deconv_us1 = self.conv_ds3.transpose()

        deconv_us2 = self.conv_ds2.transpose()

        deconv_us3 = self.conv_ds1.transpose()

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        activations = deconv_us3(deconv_us2(deconv_us1(deconv(latents))))

        logits = tf.squeeze(activations)

        return logits


# ==============================================================================

class ClicLadderCNN(ClicHierarchicalVAE):

    def __init__(self,
                 latent_dist="gaussian",
                 likelihood="gaussian",
                 first_level_channels=192,
                 second_level_channels=128,
                 first_level_layers=4,
                 padding_first_level="SAME",
                 padding_second_level="SAME",
                 name="clic_ladder_cnn"):

        super(ClicLadderCNN, self).__init__(latent_dist=latent_dist,
                                                likelihood=likelihood,
                                                standardized=True,
                                                num_levels=2,
                                                padding_first_level=padding_first_level,
                                                padding_second_level=padding_second_level,
                                                name=name)

        self._first_level_channels = first_level_channels
        self._second_level_channels = second_level_channels

        self._first_level_layers = first_level_layers


    @reuse_variables
    def encode(self, inputs, level=1, eps=1e-5):
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # First level

        self._first_level = [
            ConvDS(output_channels=self._first_level_channels,
                   kernel_shape=(5,  5),
                   num_convolutions=1,
                   padding=self._padding_first_level,
                   downsampling_rate=2,
                   use_gdn=True,
                   name="encoder_level_1_conv_ds{}".format(idx))
            for idx in range(1, self._first_level_layers)
        ]

        self._first_level_loc_head = ConvDS(output_channels=self._first_level_channels,
                                            kernel_shape=(5,  5),
                                            num_convolutions=1,
                                            padding=self._padding_first_level,
                                            downsampling_rate=2,
                                            use_gdn=False,
                                            name="encoder_level_1_conv_loc")

        first_level_scale_head = ConvDS(output_channels=self._first_level_channels,
                                        kernel_shape=(5,  5),
                                        num_convolutions=1,
                                        padding=self._padding_first_level,
                                        downsampling_rate=2,
                                        use_gdn=False,
                                        name="encoder_level_1_conv_scale")

        # Second level

        self._second_level = [
            ConvDS(output_channels=self._second_level_channels,
                   kernel_shape=(3,  3),
                   num_convolutions=1,
                   padding=self._padding_second_level,
                   downsampling_rate=1,
                   use_gdn=False,
                   activation="leaky_relu",
                   name="encoder_level_2_conv_ds1"),
            ConvDS(output_channels=self._second_level_channels,
                   kernel_shape=(5,  5),
                   num_convolutions=1,
                   padding=self._padding_second_level,
                   downsampling_rate=2,
                   use_gdn=False,
                   activation="leaky_relu",
                   name="encoder_level_2_conv_ds2")
        ]

        self._second_level_loc_head = ConvDS(output_channels=self._second_level_channels,
                                       kernel_shape=(5,  5),
                                       num_convolutions=1,
                                       padding=self._padding_second_level,
                                       downsampling_rate=2,
                                       use_gdn=False,
                                       activation="none",
                                       name="encoder_level_2_conv_loc")

        second_level_scale_head = ConvDS(output_channels=self._second_level_channels,
                                         kernel_shape=(5,  5),
                                         num_convolutions=1,
                                         padding=self._padding_second_level,
                                         downsampling_rate=2,
                                         use_gdn=False,
                                         activation="none",
                                         name="encoder_level_2_conv_scale")

        # The top-down level from the second to the first level
        self._topdown_level = [
            self._second_level_loc_head.transpose(),
        ]

        # Iterate through in reverse
        for level in self._second_level[:0:-1]:
            self._topdown_level.append(level.transpose())

        topdown_loc_head = self._second_level[0].transpose(name="topdown_loc_head")
        topdown_scale_head = self._second_level[0].transpose(name="topdown_scale_head")


        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        activations = inputs

        for layer in self._first_level:
            activations = layer(activations)

        # First stochastic level statistics
        first_level_loc = self._first_level_loc_head(activations)
        first_level_scale = tf.nn.softplus(first_level_scale_head(activations))

        # This is a probabilistic ladder network with this connection
        activations = first_level_loc

        for layer in self._second_level:
            activations = layer(activations)

        # Second stochastic level statistics
        second_level_loc = self._second_level_loc_head(activations)
        second_level_scale = tf.nn.softplus(second_level_scale_head(activations))

        # Top distribution
        second_level_posterior = self._latent_dist(loc=second_level_loc,
                                                   scale=second_level_scale)

        activations = second_level_posterior.sample()

        latents = (activations,)

        for layer in self._topdown_level:
            activations = layer(activations)

        # Topdown statistics
        topdown_loc = topdown_loc_head(activations)
        topdown_scale = tf.nn.softplus(topdown_scale_head(activations))

        # Combined first level statistics
        topdown_scale_sq_inv= 1. / (tf.pow(topdown_scale, 2) + eps)
        first_level_scale_sq_inv= 1. / (tf.pow(first_level_scale, 2) + eps)

        combined_var = 1. / (topdown_scale_sq_inv + first_level_scale_sq_inv)
        combined_scale = tf.sqrt(combined_var)

        combined_loc = (topdown_loc * first_level_scale_sq_inv + first_level_loc * topdown_scale_sq_inv) * combined_var

        # First level distribution
        first_level_posterior = self._latent_dist(loc=combined_loc,
                                                  scale=combined_scale)

        activations = first_level_posterior.sample()

        latents = latents + (activations,)

        self._latent_posteriors = [first_level_posterior, second_level_posterior]

        return latents


    @reuse_variables
    def decode(self, latents, decode_level=1):
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # Go from top to bottom
        # Second level
        decoder_second_level = [
            self._second_level_loc_head.transpose(),
        ]

        # Iterate through in reverse
        for level in self._second_level[:0:-1]:
            decoder_second_level.append(level.transpose())

        first_loc_head = self._second_level[0].transpose(name="decoder_loc_head")
        first_scale_head = self._second_level[0].transpose(name="decoder_scale_head")

        # First level
        decoder_first_level = [
            self._first_level_loc_head.transpose(),
        ]

        # Iterate through in reverse
        for level in self._first_level[::-1]:
            decoder_first_level.append(level.transpose())

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        if len(latents) != decode_level:
            raise InvalidArgumentError("Length of latents ({}) has to equal to level number {}".format(len(latents), decode_level))

        if decode_level == 2:
            second_layer_prior = self._latent_dist(loc=tf.zeros_like(latents[0]),
                                                scale=tf.ones_like(latents[0]))

            activations = latents[0]

            for layer in decoder_second_level:
                activations = layer(activations)

            first_loc = first_loc_head(activations)
            first_scale = tf.nn.softplus(first_scale_head(activations))

            # First layer prior
            first_layer_prior = self._latent_dist(loc=first_loc,
                                                scale=first_scale)

            self._latent_priors = [first_layer_prior, second_layer_prior]

            activations = latents[1]

        elif decode_level == 1:

            activations = latents[0]

        for layer in decoder_first_level:
            activations = layer(activations)


        return activations, tf.ones_like(activations)


# ==============================================================================

class ClicHyperVAECNN(ClicHierarchicalVAE):

    def __init__(self,
                 latent_dist="gaussian",
                 likelihood="gaussian",
                 first_level_channels=192,
                 second_level_channels=128,
                 first_level_layers=4,
                 padding_first_level="SAME",
                 padding_second_level="SAME",
                 name="clic_hyper_vae"):

        super(ClicHyperVAECNN, self).__init__(latent_dist=latent_dist,
                                              likelihood=likelihood,
                                              standardized=True,
                                              num_levels=2,
                                              padding_first_level=padding_first_level,
                                              padding_second_level=padding_second_level,
                                              name=name)

        self._first_level_channels = first_level_channels
        self._second_level_channels = second_level_channels

        self._first_level_layers = first_level_layers


    @reuse_variables
    def encode(self, inputs, level=1, eps=1e-5):
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # First level

        self._first_level = [
            ConvDS(output_channels=self._first_level_channels,
                   kernel_shape=(5,  5),
                   num_convolutions=1,
                   padding=self._padding_first_level,
                   downsampling_rate=2,
                   use_gdn=True,
                   name="encoder_level_1_conv_ds{}".format(idx))
            for idx in range(1, self._first_level_layers)
        ]

        self._first_level_loc_head = ConvDS(output_channels=self._first_level_channels,
                                      kernel_shape=(5,  5),
                                      num_convolutions=1,
                                      padding=self._padding_first_level,
                                      downsampling_rate=2,
                                      use_gdn=False,
                                      name="encoder_level_1_conv_loc")

        first_level_scale_head = ConvDS(output_channels=self._first_level_channels,
                                        kernel_shape=(5,  5),
                                        num_convolutions=1,
                                        padding=self._padding_first_level,
                                        downsampling_rate=2,
                                        use_gdn=False,
                                        name="encoder_level_1_conv_scale")

        # Second level

        self._second_level = [
            ConvDS(output_channels=self._second_level_channels,
                   kernel_shape=(3,  3),
                   num_convolutions=1,
                   padding=self._padding_second_level,
                   downsampling_rate=1,
                   use_gdn=False,
                   activation="leaky_relu",
                   name="encoder_level_2_conv_ds1"),
            ConvDS(output_channels=self._second_level_channels,
                   kernel_shape=(5,  5),
                   num_convolutions=1,
                   padding=self._padding_second_level,
                   downsampling_rate=2,
                   use_gdn=False,
                   activation="leaky_relu",
                   name="encoder_level_2_conv_ds2")
        ]

        self._second_level_loc_head = ConvDS(output_channels=self._second_level_channels,
                                       kernel_shape=(5,  5),
                                       num_convolutions=1,
                                       padding=self._padding_second_level,
                                       downsampling_rate=2,
                                       use_gdn=False,
                                       activation="none",
                                       name="encoder_level_2_conv_loc")

        second_level_scale_head = ConvDS(output_channels=self._second_level_channels,
                                         kernel_shape=(5,  5),
                                         num_convolutions=1,
                                         padding=self._padding_second_level,
                                         downsampling_rate=2,
                                         use_gdn=False,
                                         activation="none",
                                         name="encoder_level_2_conv_scale")


        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        activations = inputs

        for layer in self._first_level:
            activations = layer(activations)

        # First stochastic level statistics
        first_level_loc = self._first_level_loc_head(activations)
        first_level_scale = tf.nn.softplus(first_level_scale_head(activations))

        first_level_posterior = self._latent_dist(loc=first_level_loc, scale=first_level_scale)

        # This is a probabilistic ladder network with this connection
        activations = first_level_posterior.sample()

        latents = (activations,)

        for layer in self._second_level:
            activations = layer(activations)

        # Second stochastic level statistics
        second_level_loc = self._second_level_loc_head(activations)
        second_level_scale = tf.nn.softplus(second_level_scale_head(activations))

        # Top distribution
        second_level_posterior = self._latent_dist(loc=second_level_loc,
                                                   scale=second_level_scale)

        activations = second_level_posterior.sample()

        latents = (activations,) + latents

        self._latent_posteriors = [first_level_posterior, second_level_posterior]

        return latents


    @reuse_variables
    def decode(self, latents, decode_level=1):
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # Go from top to bottom
        # Second level
        decoder_second_level = [
            self._second_level_loc_head.transpose(),
        ]

        # Iterate through in reverse
        for level in self._second_level[:0:-1]:
            decoder_second_level.append(level.transpose())

        first_loc_head = self._second_level[0].transpose(name="decoder_loc_head")
        first_scale_head = self._second_level[0].transpose(name="decoder_scale_head")

        # First level
        decoder_first_level = [
            self._first_level_loc_head.transpose(),
        ]

        # Iterate through in reverse
        for level in self._first_level[::-1]:
            decoder_first_level.append(level.transpose())

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        if len(latents) != decode_level:
            raise InvalidArgumentError("Length of latents ({}) has to equal to level number {}".format(len(latents), decode_level))

        if decode_level == 2:
            second_layer_prior = self._latent_dist(loc=tf.zeros_like(latents[0]),
                                                scale=tf.ones_like(latents[0]))

            activations = latents[0]

            for layer in decoder_second_level:
                activations = layer(activations)

            first_loc = first_loc_head(activations)
            first_scale = tf.nn.softplus(first_scale_head(activations))

            # First layer prior
            first_layer_prior = self._latent_dist(loc=first_loc,
                                                scale=first_scale)

            self._latent_priors = [first_layer_prior, second_layer_prior]

            activations = latents[1]

        elif decode_level == 1:

            activations = latents[0]

        for layer in decoder_first_level:
            activations = layer(activations)


        return activations, tf.ones_like(activations)

    # ==============================================================================

class ClicLadderCNN2(ClicHierarchicalVAE):

    def __init__(self,
                 latent_dist="gaussian",
                 likelihood="gaussian",
                 first_level_channels=192,
                 second_level_channels=128,
                 first_level_layers=4,
                 padding_first_level="SAME",
                 padding_second_level="SAME",
                 name="clic_ladder_cnn2"):

        super(ClicLadderCNN2, self).__init__(latent_dist=latent_dist,
                                                likelihood=likelihood,
                                                standardized=True,
                                                num_levels=2,
                                                padding_first_level=padding_first_level,
                                                padding_second_level=padding_second_level,
                                                name=name)

        self._first_level_channels = first_level_channels
        self._second_level_channels = second_level_channels

        self._first_level_layers = first_level_layers


    @reuse_variables
    def encode(self, inputs, level=1, eps=1e-6):
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # First level

        self._first_level = [
            ConvDS(output_channels=self._first_level_channels,
                   kernel_shape=(5,  5),
                   num_convolutions=1,
                   padding=self._padding_first_level,
                   downsampling_rate=2,
                   use_gdn=True,
                   name="encoder_level_1_conv_ds{}".format(idx))
            for idx in range(1, self._first_level_layers)
        ]

        self._first_level_loc_head = ConvDS(output_channels=self._first_level_channels,
                                      kernel_shape=(5,  5),
                                      num_convolutions=1,
                                      padding=self._padding_first_level,
                                      downsampling_rate=2,
                                      use_gdn=False,
                                      name="encoder_level_1_conv_loc")

        first_level_scale_head = ConvDS(output_channels=self._first_level_channels,
                                        kernel_shape=(5,  5),
                                        num_convolutions=1,
                                        padding=self._padding_first_level,
                                        downsampling_rate=2,
                                        use_gdn=False,
                                        name="encoder_level_1_conv_scale")

        # Second level

        self._second_level = [
            ConvDS(output_channels=self._second_level_channels,
                   kernel_shape=(3,  3),
                   num_convolutions=1,
                   padding=self._padding_second_level,
                   downsampling_rate=1,
                   use_gdn=False,
                   activation="leaky_relu",
                   name="encoder_level_2_conv_ds1"),
            ConvDS(output_channels=self._second_level_channels,
                   kernel_shape=(5,  5),
                   num_convolutions=1,
                   padding=self._padding_second_level,
                   downsampling_rate=2,
                   use_gdn=False,
                   activation="leaky_relu",
                   name="encoder_level_2_conv_ds2")
        ]

        self._second_level_loc_head = ConvDS(output_channels=self._second_level_channels,
                                       kernel_shape=(5,  5),
                                       num_convolutions=1,
                                       padding=self._padding_second_level,
                                       downsampling_rate=2,
                                       use_gdn=False,
                                       activation="none",
                                       name="encoder_level_2_conv_loc")

        second_level_scale_head = ConvDS(output_channels=self._second_level_channels,
                                         kernel_shape=(5,  5),
                                         num_convolutions=1,
                                         padding=self._padding_second_level,
                                         downsampling_rate=2,
                                         use_gdn=False,
                                         activation="none",
                                         name="encoder_level_2_conv_scale")

        # The top-down level from the second to the first level
        self._topdown_level = [
            self._second_level_loc_head.transpose(),
        ]

        # Iterate through in reverse
        for level in self._second_level[:0:-1]:
            self._topdown_level.append(level.transpose())

        self._topdown_loc_head = self._second_level[0].transpose(name="topdown_loc_head")
        self._topdown_scale_head = self._second_level[0].transpose(name="topdown_scale_head")


        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        activations = inputs

        for layer in self._first_level:
            activations = layer(activations)

        # First stochastic level statistics
        first_level_loc = self._first_level_loc_head(activations)
        first_level_precision = tf.nn.softplus(first_level_scale_head(activations))

        # This is a probabilistic ladder network with this connection
        activations = first_level_loc

        for layer in self._second_level:
            activations = layer(activations)

        # Second stochastic level statistics
        second_level_loc = self._second_level_loc_head(activations)

        # The sigmoid activation enforces that the posterior variance is always less than 
        # the prior variance (which is 1)
        second_level_scale = tf.nn.sigmoid(second_level_scale_head(activations))

        # Top distribution
        second_level_posterior = self._latent_dist(loc=second_level_loc,
                                                   scale=second_level_scale)

        activations = second_level_posterior.sample()

        latents = (activations,)

        for layer in self._topdown_level:
            activations = layer(activations)

        # Topdown statistics
        topdown_loc = self._topdown_loc_head(activations)
        topdown_scale = tf.nn.softplus(self._topdown_scale_head(activations))

        # Combined first level statistics
        topdown_precision = 1. / (tf.pow(topdown_scale, 2) + eps)

        combined_var = 1. / (topdown_precision + first_level_precision)
        combined_scale = tf.sqrt(combined_var)

        combined_loc = (topdown_loc * first_level_precision + \
                        first_level_loc * topdown_precision) / combined_var

        # First level distribution
        first_level_posterior = self._latent_dist(loc=combined_loc,
                                                  scale=combined_scale)

        activations = first_level_posterior.sample()

        latents = latents + (activations,)

        self._latent_posteriors = [first_level_posterior, second_level_posterior]

        return latents


    @reuse_variables
    def decode(self, latents, decode_level=1):
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # Go from top to bottom
        # Second level
        decoder_second_level = self._topdown_level

        first_loc_head = self._topdown_loc_head
        first_scale_head = self._topdown_scale_head

        # First level
        decoder_first_level = [
            self._first_level_loc_head.transpose(),
        ]

        # Iterate through in reverse
        for level in self._first_level[::-1]:
            decoder_first_level.append(level.transpose())

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        # if len(latents) != decode_level:
        #     raise InvalidArgumentError("Length of latents ({}) has to equal to level number {}".format(len(latents), decode_level))

        if decode_level == 2:
            second_layer_prior = self._latent_dist(loc=tf.zeros_like(latents[0]),
                                                   scale=tf.ones_like(latents[0]))

            activations = latents[0]

            for layer in decoder_second_level:
                activations = layer(activations)

            first_loc = first_loc_head(activations)
            first_scale = tf.nn.softplus(first_scale_head(activations))

            # First layer prior
            first_layer_prior = self._latent_dist(loc=first_loc,
                                                  scale=first_scale)

            self._latent_priors = [first_layer_prior, second_layer_prior]

            activations = latents[1]

            for layer in decoder_first_level:
                activations = layer(activations)

            return activations, tf.ones_like(activations)

        elif decode_level == "second":
            second_layer_prior = self._latent_dist(loc=tf.zeros_like(latents[0]),
                                                scale=tf.ones_like(latents[0]))

            activations = latents

            for layer in decoder_second_level:
                activations = layer(activations)

            first_loc = first_loc_head(activations)
            first_scale = tf.nn.softplus(first_scale_head(activations))

            # First layer prior
            first_layer_prior = self._latent_dist(loc=first_loc,
                                                scale=first_scale)

            self._latent_priors = [first_layer_prior, second_layer_prior]

            return first_loc, first_scale

        elif decode_level == "first":

            activations = latents

            for layer in decoder_first_level:
                activations = layer(activations)

            return activations, tf.ones_like(activations)

        return None

# ==============================================================================
