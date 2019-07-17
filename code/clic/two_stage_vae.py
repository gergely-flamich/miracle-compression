import numpy as np

import tensorflow as tf
tfq = tf.quantization

import sonnet as snt

import tensorflow_probability as tfp
tfd = tfp.distributions

from miracle_modules import ConvDS
from compression import coded_sample, decode_sample
from coding import ArithmeticCoder, write_bin_code, read_bin_code

from utils import InvalidArgumentError


# ==============================================================================
# ==============================================================================
# 2-Stage VAE
# ==============================================================================
# ==============================================================================


class ClicTwoStageVAE(snt.AbstractModule):

    _allowed_latent_dists = {
        "gaussian": tfd.Normal
    }

    _allowed_likelihoods = {
        "gaussian": tfd.Normal,
        "laplace": tfd.Laplace
    }

    def __init__(self,
                 latent_dist="gaussian",
                 likelihood="gaussian",
                 latent_filters=192,
                 first_level_layers=4,
                 second_level_layers=4,
                 name="clic_two_stage_vae"):

        # Call superclass
        super(ClicTwoStageVAE, self).__init__(name=name)


        # Error checking
        if latent_dist not in self._allowed_latent_dists:
            raise InvalidArgumentError("latent_dist must be one of {}, was {}"
                                                 .format(self._allowed_latent_dists, latent_dist))
        if likelihood not in self._allowed_likelihoods:
            raise InvalidArgumentError("likelihood must be one of {}, was {}"
                                                 .format(self._allowed_likelihoods, likelihood))


        # Set variables
        self.latent_dist = latent_dist
        self.likelihood_dist = likelihood

        self.first_level_layers = first_level_layers
        self.second_level_layers = second_level_layers
        
        self.latent_filters = latent_filters
        
        self.train_stage = 0
        
        self.first_run = True

    @property
    def log_gamma(self):
        self._ensure_is_connected()

        return self.manifold_vae.log_gamma if self.train_stage == 0 else self.measure_vae.log_gamma
        
    @property
    def log_prob(self):
        self._ensure_is_connected()

        return self.manifold_vae.log_prob if self.train_stage == 0 else self.measure_vae.log_prob

    @property
    def kl_divergence(self):
        self._ensure_is_connected()

        return self.manifold_vae.kl_divergence if self.train_stage == 0 else self.measure_vae.kl_divergence

    
    def get_all_training_variables(self):
        self._ensure_is_connected()
        
        return self.manifold_vae.get_all_variables() if self.train_stage == 0 else self.measure_vae.get_all_variables()
        
    
    @snt.reuse_variables
    def encode(self, inputs):
       
        latents = self.manifold_vae.encode(inputs)
        
        if self.train_stage > 0 or self.first_run:    
            self.first_latents = latents
            latents = self.measure_vae.encode(latents)
            
        return latents
    

    @snt.reuse_variables
    def decode(self, latents):
        
        reconstruction = latents
        
        if self.train_stage > 0 or self.first_run:    
            reconstruction = self.measure_vae.decode(reconstruction)
            
            if self.train_stage == 1:
                return reconstruction
                
        reconstruction = self.manifold_vae.decode(reconstruction)
   
        return reconstruction


    def _build(self, inputs):

        self.manifold_vae = ClicTwoStageVAE_Manifold(latent_dist=self.latent_dist,
                                                 likelihood=self.likelihood_dist,
                                                 latent_filters=self.latent_filters,
                                                 num_layers=self.first_level_layers)
        
        self.measure_vae = ClicTwoStageVAE_Measure(latent_dist=self.latent_dist,
                                                 likelihood=self.likelihood_dist,
                                                 latent_filters=self.latent_filters,
                                                 num_layers=self.second_level_layers)
            
        
        latents = self.encode(inputs)
        reconstruction = self.decode(latents)
        
        self.manifold_vae._log_prob = self.manifold_vae.likelihood.log_prob(inputs)
        
        if self.train_stage > 0:
            self.measure_vae._log_prob = self.measure_vae.likelihood.log_prob(self.first_latents)
        
        if self.first_run:
            self.first_run = False

        return reconstruction

# ==============================================================================
# ==============================================================================
# Components of the two stage VAE: manifold and measure VAEs
# ==============================================================================
# ==============================================================================

# ==============================================================================
# First stage
# ==============================================================================

class ClicTwoStageVAE_Manifold(snt.AbstractModule):
    _allowed_latent_dists = {
        "gaussian": tfd.Normal
    }

    _allowed_likelihoods = {
        "gaussian": tfd.Normal,
        "laplace": tfd.Laplace
    }

    def __init__(self,
                 latent_dist="gaussian",
                 likelihood="gaussian",
                 latent_filters=192,
                 num_layers=4,
                 name="clic_two_stage_vae_manifold"):

        # Call superclass
        super(ClicTwoStageVAE_Manifold, self).__init__(name=name)


        # Error checking
        if latent_dist not in self._allowed_latent_dists:
            raise InvalidArgumentError("latent_dist must be one of {}"
                                                 .format(self._allowed_latent_dists))
        if likelihood not in self._allowed_likelihoods:
            raise InvalidArgumentError("likelihood must be one of {}"
                                                 .format(self._allowed_likelihoods))


        # Set variables
        self.latent_dist = self._allowed_latent_dists[latent_dist]
        self.likelihood_dist = self._allowed_likelihoods[likelihood]

        self.num_layers = num_layers        
        self.latent_filters = latent_filters

    @property
    def log_prob(self):
        self._ensure_is_connected()

        return tf.reduce_sum(self._log_prob)

    @property
    def kl_divergence(self):
        self._ensure_is_connected()

        return tfd.kl_divergence(self.posterior, self.prior)

    @snt.reuse_variables
    def encode(self, inputs):
        # ----------------------------------------------------------------------
        # Define constants
        # ----------------------------------------------------------------------

        kernel_shape = (5, 5)
        channels = 192
        padding = "SAME_MIRRORED"

        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # First level
        self.layers = [
            ConvDS(output_channels=channels,
                   kernel_shape=kernel_shape,
                   num_convolutions=1,
                   padding=padding,
                   downsampling_rate=2,
                   use_gdn=True,
                   name="encoder_conv_ds_{}".format(idx))
            for idx in range(1, self.num_layers)
        ]

        self.encoder_loc_head = ConvDS(output_channels=self.latent_filters,
                                      kernel_shape=kernel_shape,
                                      num_convolutions=1,
                                      downsampling_rate=2,
                                      padding=padding,
                                      use_gdn=False,
                                      name="encoder_loc")

        self.encoder_log_scale_head = ConvDS(output_channels=self.latent_filters,
                                        kernel_shape=kernel_shape,
                                        num_convolutions=1,
                                        downsampling_rate=2,
                                        padding=padding,
                                        use_gdn=False,
                                        name="encoder_log_scale")

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        # First level
        activations = inputs

        for layer in self.layers:
            activations = layer(activations)


        # Get first level statistics
        loc = self.encoder_loc_head(activations)
        log_scale = self.encoder_log_scale_head(activations)
        scale = tf.math.exp(log_scale)

        # Create latent posterior distribution
        self.posterior = tfd.Normal(loc=loc,
                                    scale=scale)
        
        latents = self.posterior.sample()
        
        return latents


    @snt.reuse_variables
    def decode(self, latents):
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        layers = [self.encoder_loc_head.transpose()]

        for layer in self.layers[::-1]:
            layers.append(layer.transpose())

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        # Create second level prior
        self.prior = tfd.Normal(loc=tf.zeros_like(latents),
                                scale=tf.ones_like(latents))
        
        activations = latents

        for layer in layers:
            activations = layer(activations)
            
        reconstruction = tf.nn.sigmoid(activations)
            
        self.log_gamma = tf.get_variable("gamma_x", dtype=tf.float32, initializer=0.)
        gamma = tf.exp(self.log_gamma)
        self.likelihood = self.likelihood_dist(loc=reconstruction,
                                               scale=gamma)

        return reconstruction


    def _build(self, inputs):

        latents = self.encode(inputs)  
        reconstruction = self.decode(latents)
        
        self._log_prob = self.likelihood.log_prob(inputs)

        return reconstruction
    
    
# ==============================================================================
# Second stage
# ==============================================================================

class ClicTwoStageVAE_Measure(snt.AbstractModule):
    
    _allowed_latent_dists = {
        "gaussian": tfd.Normal
    }

    _allowed_likelihoods = {
        "gaussian": tfd.Normal,
        "laplace": tfd.Laplace
    }

    def __init__(self,
                 latent_dist="gaussian",
                 likelihood="gaussian",
                 latent_filters=192,
                 num_layers=4,
                 residual=True,
                 name="clic_two_stage_vae_measure"):

        # Call superclass
        super(ClicTwoStageVAE_Measure, self).__init__(name=name)


        # Error checking
        if latent_dist not in self._allowed_latent_dists:
            raise InvalidArgumentError("latent_dist must be one of {}"
                                                 .format(self._allowed_latent_dists))
        if likelihood not in self._allowed_likelihoods:
            raise InvalidArgumentError("likelihood must be one of {}"
                                                 .format(self._allowed_likelihoods))


        # Set variables
        self.latent_dist = self._allowed_latent_dists[latent_dist]
        self.likelihood_dist = self._allowed_likelihoods[likelihood]

        self.num_layers = num_layers        
        self.latent_filters = latent_filters
        
        # Should we have residual connections?
        self.residual = residual
        
        # ----------------------------------------------------------------------
        # Define constants
        # ----------------------------------------------------------------------

        self.kernel_shape = (5, 5)
        self.channels = 192
        self.padding = "SAME_MIRRORED"

    @property
    def log_prob(self):
        self._ensure_is_connected()

        return tf.reduce_sum(self._log_prob)

    @property
    def kl_divergence(self):
        self._ensure_is_connected()

        return tfd.kl_divergence(self.posterior, self.prior)

    @snt.reuse_variables
    def encode(self, inputs):

        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        # First level
        self.layers = [
            ConvDS(output_channels=self.channels,
                   kernel_shape=self.kernel_shape,
                   num_convolutions=1,
                   padding=self.padding,
                   downsampling_rate=1,
                   use_gdn=False,
                   activation="leaky_relu",
                   name="encoder_conv_ds_{}".format(idx))
            for idx in range(1, self.num_layers)
        ]

        self.encoder_loc_head = ConvDS(output_channels=self.latent_filters,
                                      kernel_shape=self.kernel_shape,
                                      num_convolutions=1,
                                      downsampling_rate=1,
                                      padding=self.padding,
                                      use_gdn=False,
                                      name="encoder_loc")

        self.encoder_log_scale_head = ConvDS(output_channels=self.latent_filters,
                                        kernel_shape=self.kernel_shape,
                                        num_convolutions=1,
                                        downsampling_rate=1,
                                        padding=self.padding,
                                        use_gdn=False,
                                        name="encoder_log_scale")

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        # First level
        activations = inputs

        for layer in self.layers:
            activations = layer(activations)

        
        if self.residual:
            # concatenate along channels
            activations = tf.concat([activations, inputs], axis=-1)

        # Get first level statistics
        loc = self.encoder_loc_head(activations)
        log_scale = self.encoder_log_scale_head(activations)
        scale = tf.math.exp(log_scale)

        # Create latent posterior distribution
        self.posterior = tfd.Normal(loc=loc,
                                    scale=scale)
        
        latents = self.posterior.sample()

        return latents


    @snt.reuse_variables
    def decode(self, latents):
        # ----------------------------------------------------------------------
        # Define layers
        # ----------------------------------------------------------------------

        layers = [ConvDS(output_channels=self.channels,
                         kernel_shape=self.kernel_shape,
                         num_convolutions=1,
                         padding=self.padding,
                         downsampling_rate=1,
                         use_gdn=False,
                         activation="leaky_relu",
                         name="decoder_conv_ds")]

        for layer in self.layers[::-1]:
            layers.append(layer.transpose())
           
            
        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        # Create second level prior
        self.prior = tfd.Normal(loc=tf.zeros_like(latents),
                                scale=tf.ones_like(latents))
        
        activations = latents

        for layer in layers[:-1]:
            activations = layer(activations)
            
        if self.residual:
            # concatenate along channels
            activations = tf.concat([activations, latents], axis=-1)

        # Final layer
        activations = layers[-1](activations)
        
        reconstruction = tf.nn.sigmoid(activations)
            
        self.log_gamma = tf.get_variable("gamma_z", dtype=tf.float32, initializer=0.)
        gamma = tf.exp(self.log_gamma)
        self.likelihood = self.likelihood_dist(loc=reconstruction,
                                               scale=gamma)
        return reconstruction


    def _build(self, inputs):

        latents = self.encode(inputs)

        reconstruction = self.decode(latents)
        
        self._log_prob = self.likelihood.log_prob(inputs)
        
        return reconstruction