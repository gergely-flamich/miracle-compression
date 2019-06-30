import tensorflow as tf
import sonnet as snt
import tensorflow_probability as tfp
tfd = tfp.distributions

from miracle_modules import ConvDS

from utils import InvalidArgumentError

class ClicNewLadderCNN(snt.AbstractModule):

    _allowed_latent_dists = {
        "gaussian": tfd.Normal,
        "laplace": tfd.Laplace
    }

    _allowed_likelihoods = {
        "gaussian": tfd.Normal,
        "laplace": tfd.Laplace
    }

    def __init__(self,
                 latent_dist="gaussian",
                 likelihood="gaussian",
                 first_level_latents=192,
                 second_level_latents=192,
                 first_level_layers=4,
                 name="new_clic_ladder_cnn"):

        # Call superclass
        super(ClicNewLadderCNN, self).__init__(name=name)


        # Error checking
        if latent_dist not in self._allowed_latent_dists:
            raise tf.errors.InvalidArgumentError("latent_dist must be one of {}"
                                                 .format(self._allowed_latent_dists))
        if likelihood not in self._allowed_likelihoods:
            raise tf.errors.InvalidArgumentError("likelihood must be one of {}"
                                                 .format(self._allowed_likelihoods))


        # Set variables
        self.latent_dist = self._allowed_latent_dists[latent_dist]
        self.likelihood_dist = self._allowed_likelihoods[likelihood]

        self.first_level_layers = first_level_layers
        self.first_level_latents = first_level_latents
        
        self.second_level_latents = second_level_latents

    @property
    def log_prob(self):
        self._ensure_is_connected()

        return tf.reduce_sum(self._log_prob)

    @property
    def kl_divergence(self):
        self._ensure_is_connected()

        return [tfd.kl_divergence(posterior, prior) for posterior, prior in
                zip(self.latent_posteriors, self.latent_priors)]

    @snt.reuse_variables
    def encode(self, inputs, eps=1e-12):
        # ----------------------------------------------------------------------
        # Define constants
        # ----------------------------------------------------------------------

        kernel_shape = (5, 5)
        first_level_channels = 192
        second_level_channels = 128
        padding = "SAME_MIRRORED"

        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # First level
        self.first_level = [
            ConvDS(output_channels=first_level_channels,
                   kernel_shape=kernel_shape,
                   num_convolutions=1,
                   padding=padding,
                   downsampling_rate=2,
                   use_gdn=True,
                   name="encoder_level_1_conv_ds{}".format(idx))
            for idx in range(1, self.first_level_layers)
        ]

        self.first_level_loc = ConvDS(output_channels=self.first_level_latents,
                                      kernel_shape=kernel_shape,
                                      num_convolutions=1,
                                      downsampling_rate=2,
                                      padding=padding,
                                      use_gdn=False,
                                      name="encoder_level_1_loc")

        self.first_level_log_scale = ConvDS(output_channels=self.first_level_latents,
                                        kernel_shape=kernel_shape,
                                        num_convolutions=1,
                                        downsampling_rate=2,
                                        padding=padding,
                                        use_gdn=False,
                                        name="encoder_level_1_scale")

        # Second Level

        self.second_level = [
            ConvDS(output_channels=second_level_channels,
                   kernel_shape=kernel_shape,
                   num_convolutions=1,
                   downsampling_rate=1,
                   padding=padding,
                   use_gdn=False,
                   activation="leaky_relu"),
            ConvDS(output_channels=second_level_channels,
                   kernel_shape=kernel_shape,
                   num_convolutions=1,
                   downsampling_rate=1,
                   padding=padding,
                   use_gdn=False,
                   activation="leaky_relu")
        ]

        self.second_level_loc = ConvDS(output_channels=self.second_level_latents,
                                       kernel_shape=kernel_shape,
                                       num_convolutions=1,
                                       padding=padding,
                                       downsampling_rate=2,
                                       use_gdn=False,
                                       name="encoder_level_2_loc")

        self.second_level_log_scale = ConvDS(output_channels=self.second_level_latents,
                                         kernel_shape=kernel_shape,
                                         num_convolutions=1,
                                         padding=padding,
                                         downsampling_rate=2,
                                         use_gdn=False,
                                         name="encoder_level_2_loc")

        # Second to first level

        self.reverse_level = [self.second_level_loc.transpose()]

        # Iterate through the second level backwards
        for layer in self.second_level[:0:-1]:
            self.reverse_level.append(layer.transpose())


        self.reverse_level_loc = self.second_level[0].transpose(name="reverse_level_loc")
        self.reverse_level_log_scale = self.second_level[0].transpose(name="reverse_level_scale")

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        # First level
        activations = inputs

        for layer in self.first_level:
            activations = layer(activations)


        # Get first level statistics
        first_level_loc = self.first_level_loc(activations)
        first_level_log_scale = self.first_level_log_scale(activations)
        first_level_scale = tf.math.exp(first_level_log_scale)

        # Pass on the mean as the activations
        activations = first_level_loc

        # Second level
        for layer in self.second_level:
            activations = layer(activations)


        # Get second level statistics
        second_level_loc = self.second_level_loc(activations)
        second_level_log_scale = self.second_level_log_scale(activations)
        second_level_scale = tf.nn.sigmoid(second_level_log_scale)

        # Create latent posterior distribution
        self.second_level_posterior = tfd.Normal(loc=second_level_loc,
                                                 scale=second_level_scale)

        # Get z_2 ~ N(mu_2, sigma_2)
        second_level_latents = self.second_level_posterior.sample()
        activations = second_level_latents

        # Reverse level
        for layer in self.reverse_level:
            activations = layer(activations)


        # Get reverse level statistics
        reverse_level_loc = self.reverse_level_loc(activations)
        reverse_level_log_scale = self.reverse_level_log_scale(activations)
        reverse_level_scale = tf.math.exp(reverse_level_log_scale)

        # Combine statistics on the first level
        first_level_var = tf.square(first_level_scale)
        reverse_level_var = tf.square(reverse_level_scale)

        first_level_prec = 1. / (first_level_var + eps)
        reverse_level_prec = 1. / (reverse_level_var + eps)

        # Combined variance
        combined_var = 1. / (first_level_prec + reverse_level_prec)
        combined_scale = tf.sqrt(combined_var)

        # Combined location
        combined_loc = first_level_loc * reverse_level_prec
        combined_loc += reverse_level_loc * first_level_prec
        combined_loc *= combined_var

        # Create latent posterior distribution
        self.first_level_posterior = tfd.Normal(loc=combined_loc,
                                                scale=combined_scale)

        # Get z_1 ~ N(mu_1, sigma_1)
        first_level_latents = self.first_level_posterior.sample()

        latents = (second_level_latents, first_level_latents)
        self.latent_posteriors = (self.first_level_posterior, self.second_level_posterior)

        return latents


    @snt.reuse_variables
    def decode(self, latents):
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # Second level
        second_level = self.reverse_level

        second_level_loc = self.reverse_level_loc
        second_level_log_scale = self.reverse_level_log_scale

        # First level
        first_level = [self.first_level_loc.transpose()]

        for layer in self.first_level[::-1]:
            first_level.append(layer.transpose())

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        second_level_latents, first_level_latents = latents

        # Create second level prior
        self.second_level_prior = tfd.Normal(loc=tf.zeros_like(second_level_latents),
                                             scale=tf.ones_like(second_level_latents))

        # Apply second level
        activations = second_level_latents

        for layer in second_level:
            activations = layer(activations)

        # Get first level statistics
        first_level_loc = second_level_loc(activations)
        first_level_log_scale = second_level_log_scale(activations)
        first_level_scale = tf.math.exp(first_level_log_scale)

        # Create first level prior
        self.first_level_prior = tfd.Normal(loc=first_level_loc,
                                            scale=first_level_scale)

        # Apply first level
        activations = first_level_latents

        for layer in first_level:
            activations = layer(activations)

        self.latent_priors = (self.first_level_prior, self.second_level_prior)

        return activations


    def _build(self, inputs):

        latents = self.encode(inputs)
        reconstruction = self.decode(latents)
        
        reconstruction = tf.nn.sigmoid(reconstruction)

        self.likelihood = self.likelihood_dist(loc=reconstruction,
                                               scale=tf.ones_like(reconstruction))

        self._log_prob = self.likelihood.log_prob(inputs)

        return reconstruction
