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
        
        self.train_first = True
        self.train_second = False
        
        self.first_run = True

    @property
    def log_prob(self):
        self._ensure_is_connected()

        return self.manifold_vae.log_prob if self.train_first else self.measure_vae.log_prob

    @property
    def kl_divergence(self):
        self._ensure_is_connected()

        return self.manifold_vae.kl_divergence if self.train_first else self.measure_vae.kl_divergence

    
    def get_all_training_variables(self):
        self._ensure_is_connected()
        
        return self.manifold_vae.get_all_variables() if self.train_first else self.measure_vae.get_all_variables()
        
    
    @snt.reuse_variables
    def encode(self, inputs, use_second=False):
       
        latents = self.manifold_vae.encode(inputs)
        
        if use_second:    
            latents = self.measure_vae.encode(latents)
            
        return latents
    

    @snt.reuse_variables
    def decode(self, latents, use_second=False):
        
        if use_second:    
            reconstruction = self.measure_vae.decode(latents)
            
            if self.train_second:
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
            
        
        latents = self.encode(inputs,
                              use_second=self.first_run or not self.train_first)
        reconstruction = self.decode(latents,
                                     use_second=self.first_run or not self.train_first)

        if self.first_run:
            self.first_run = False

        return reconstruction
    
    
    # =========================================================================================
    # Compression
    # =========================================================================================
    
    def expand_prob_mass(self, 
                         probability_mass, 
                         gamma, 
                         miracle_bits,
                         outlier_mode="quantize",
                         verbose=False):
        
        if outlier_mode == "quantize":
            
            # Create a probability mass for 16-bit symbols
            P = gamma * np.ones(2**16)

            P[1:2**miracle_bits + 1] = probability_mass

            if verbose: 
                miracle_mass = np.sum(probability_mass)
                outlier_mass = (gamma * 2**16) - 2**miracle_bits

                print("Outlier / Miracle Mass Ratio: {:.4f}".format(outlier_mass / miracle_mass))
                
        elif outlier_mode == "importance_sample":
            
            P = np.ones(1 + 2**miracle_bits)
            P[1:2**miracle_bits + 1] = probability_mass
        
        return P

    def code_image(self, 
                   image, 
                   seed, 
                   miracle_bits, 
                   probability_mass, 
                   comp_file_path, 
                   n_points=30, 
                   gamma=100, 
                   precision=32,
                   outlier_mode="quantize",
                   verbose=False):
        
        # -------------------------------------------------------------------------------------
        # Step 1: Set the latent distributions for the image
        # -------------------------------------------------------------------------------------
        
        # Calculate the posteriors
        latents = self.encode(image)
        
        # Calculate the priors
        self.decode(latents)
        
        # -------------------------------------------------------------------------------------
        # Step 2: Create a coded sample of the latent space
        # -------------------------------------------------------------------------------------
    
        # Code first level
        coded_first_level = coded_sample(proposal=self.latent_priors[0], 
                                         target=self.latent_posteriors[0], 
                                         seed=seed, 
                                         n_points=n_points, 
                                         miracle_bits=miracle_bits,
                                         outlier_mode=outlier_mode)
        # Code second level
        coded_second_level = coded_sample(proposal=self.latent_priors[1], 
                                          target=self.latent_posteriors[1], 
                                          seed=seed, 
                                          n_points=n_points, 
                                          miracle_bits=miracle_bits,
                                          outlier_mode=outlier_mode)
        
        first_level_shape = self.latent_priors[0].loc.shape.as_list()
        second_level_shape = self.latent_priors[1].loc.shape.as_list()
        
        # The -1 at the end will turn into a 0 (EOF) on the next line
        coded_latents = tf.concat([coded_first_level, coded_second_level, [-1]], axis=0).numpy()
        
        # -------------------------------------------------------------------------------------
        # Step 3: Arithmetic code the coded samples
        # -------------------------------------------------------------------------------------
        
        # Shift the code symbols forward by one, since 0 is a special end of file symbol
        coded_latents = coded_latents + 1
        
        # Create a probability mass for 16-bit symbols
        probability_mass = self.expand_prob_mass(probability_mass, 
                                                 gamma, 
                                                 miracle_bits, 
                                                 outlier_mode,
                                                 verbose)
        
        # Create coder
        coder = ArithmeticCoder(probability_mass, precision=precision)
        
        bitcode = coder.encode(coded_latents)
        
        # Log code length and expected code length
        if verbose:
            total_mass = np.sum(probability_mass)
            log_prob_mass = np.log(probability_mass)
            log_total_mass = np.log(total_mass)
            
            code_log_prob = 0
            
            for i in range(len(coded_latents)):
                code_log_prob += log_prob_mass[coded_latents[i]]
                
            # Normalize
            code_log_prob -= log_total_mass * len(coded_latents)
            
            print("Expected code length: {:.2f} bits".format(-code_log_prob))
            print("Actual code length: {} bits".format(len(bitcode))) 

        # -------------------------------------------------------------------------------------
        # Step 4: Write the compressed file
        # -------------------------------------------------------------------------------------
        
        extras = [seed, gamma] + first_level_shape[1:3] + second_level_shape[1:3]
    
        write_bin_code(''.join(bitcode), 
                       comp_file_path, 
                       extras=extras)
        
    
    def decode_image(self, 
                     comp_file_path, 
                     probability_mass, 
                     miracle_bits, 
                     n_points=30, 
                     precision=32,
                     outlier_mode="quantize",
                     verbose=False):
        
        # -------------------------------------------------------------------------------------
        # Step 1: Read the compressed file
        # -------------------------------------------------------------------------------------
        
        # the extras are: seed, gamma and W x H of the two latent levels
        code, extras = read_bin_code(comp_file_path, num_extras=6)
        
        print(extras)
        
        seed = extras[0]
        gamma = extras[1]
        
        # Get shape information back
        first_level_shape = [1] + extras[2:4] + [self.first_level_latents]
        second_level_shape = [1] + extras[4:] + [self.second_level_latents]
        
        # Total number of latents on levels
        num_first_level = np.prod(first_level_shape)
        num_second_level = np.prod(second_level_shape)
        
        # -------------------------------------------------------------------------------------
        # Step 2: Decode the arithmetic code
        # -------------------------------------------------------------------------------------
        
        # Create a probability mass for 16-bit symbols
        probability_mass = self.expand_prob_mass(probability_mass,
                                                 gamma, 
                                                 miracle_bits, 
                                                 outlier_mode,
                                                 verbose)
        
        decoder = ArithmeticCoder(probability_mass, precision=precision)
    
        decompressed = decoder.decode_fast(code, verbose=verbose)
        
        # -------------------------------------------------------------------------------------
        # Step 3: Decode the samples using MIRACLE
        # -------------------------------------------------------------------------------------
        
        # Decode second level
        proposal = tfd.Normal(loc=tf.zeros(second_level_shape),
                              scale=tf.ones(second_level_shape))
        
        # Remember to shift the codes back by one, since we shifted them forward during encoding
        # Note: the second level needs to have the EOF 0 cut off from the end
        coded_first_level = tf.convert_to_tensor(decompressed[:num_first_level]) - 1
        coded_second_level = tf.convert_to_tensor(decompressed[num_first_level:-1]) - 1
        
        decoded_second_level = decode_sample(coded_sample=coded_second_level,
                                             proposal=proposal, 
                                             seed=seed, 
                                             n_points=n_points, 
                                             miracle_bits=miracle_bits, 
                                             outlier_mode=outlier_mode)
        
        decoded_second_level = tf.reshape(decoded_second_level, second_level_shape)
        
        # Now we can calculate the the first level priors
        self.decode((decoded_second_level,
                     tf.zeros(first_level_shape)))
        
        # Decode first level
        
        decoded_first_level = decode_sample(coded_sample=coded_first_level,
                                            proposal=self.latent_priors[0], 
                                            seed=seed, 
                                            n_points=n_points, 
                                            miracle_bits=miracle_bits, 
                                            outlier_mode=outlier_mode)
        
        decoded_first_level = tf.reshape(decoded_first_level, first_level_shape)
        
        
        # -------------------------------------------------------------------------------------
        # Step 4: Reconstruct the image with the VAE
        # -------------------------------------------------------------------------------------
        
        reconstruction = self.decode((decoded_second_level,
                                      decoded_first_level))
        
        return tf.squeeze(reconstruction)
    
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
    def encode(self, inputs, eps=1e-12):
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

        return tf.nn.sigmoid(activations)


    def _build(self, inputs):

        latents = self.encode(inputs)

        reconstruction = self.decode(latents)

        self.likelihood = self.likelihood_dist(loc=reconstruction,
                                               scale=tf.get_variable("gamma_x", 0.))

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

    @property
    def log_prob(self):
        self._ensure_is_connected()

        return tf.reduce_sum(self._log_prob)

    @property
    def kl_divergence(self):
        self._ensure_is_connected()

        return tfd.kl_divergence(self.posterior, self.prior)

    @snt.reuse_variables
    def encode(self, inputs, eps=1e-12):
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
                   downsampling_rate=1,
                   use_gdn=False,
                   activation="leaky_relu",
                   name="encoder_conv_ds_{}".format(idx))
            for idx in range(1, self.num_layers)
        ]

        self.encoder_loc_head = ConvDS(output_channels=self.latent_filters,
                                      kernel_shape=kernel_shape,
                                      num_convolutions=1,
                                      downsampling_rate=1,
                                      padding=padding,
                                      use_gdn=False,
                                      name="encoder_loc")

        self.encoder_log_scale_head = ConvDS(output_channels=self.latent_filters,
                                        kernel_shape=kernel_shape,
                                        num_convolutions=1,
                                        downsampling_rate=1,
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

        return tf.nn.sigmoid(activations)


    def _build(self, inputs):

        latents = self.encode(inputs)

        reconstruction = self.decode(latents)

        self.likelihood = self.likelihood_dist(loc=reconstruction,
                                               scale=tf.get_variable("gamma_y", 0.))

        self._log_prob = self.likelihood.log_prob(inputs)

        return reconstruction