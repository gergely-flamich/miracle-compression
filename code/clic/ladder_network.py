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
# Ladder network
# ==============================================================================
# ==============================================================================

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
                 first_level_residual=False,
                 second_level_residual=False,
                 first_level_channels=192,
                 second_level_channels=128,
                 kernel_shape=(5, 5),
                 padding="SAME_MIRRORED",
                 learn_log_gamma=False,
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

        self.first_level_latents = first_level_latents
        self.second_level_latents = second_level_latents
        
        self.first_level_residual = first_level_residual
        self.second_level_residual = second_level_residual
        
        self.first_level_channels = first_level_channels
        self.second_level_channels = second_level_channels

        self.learn_log_gamma = learn_log_gamma
        
        self.log_gamma = 0
        self.first_level_layers = 4
        
        self.padding = padding
        self.kernel_shape = kernel_shape
     
        
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
        # Define layers
        # ----------------------------------------------------------------------

        # First level
        self.first_level = [
            ConvDS(output_channels=self.first_level_channels,
                   kernel_shape=self.kernel_shape,
                   num_convolutions=1,
                   padding=self.padding,
                   downsampling_rate=2,
                   use_gdn=True,
                   name="encoder_level_1_conv_ds{}".format(idx))
            for idx in range(1, self.first_level_layers)
        ]

        self.first_level_loc = ConvDS(output_channels=self.first_level_latents,
                                      kernel_shape=self.kernel_shape,
                                      num_convolutions=1,
                                      downsampling_rate=2,
                                      padding=self.padding,
                                      use_gdn=False,
                                      name="encoder_level_1_loc")

        self.first_level_log_scale = ConvDS(output_channels=self.first_level_latents,
                                        kernel_shape=self.kernel_shape,
                                        num_convolutions=1,
                                        downsampling_rate=2,
                                        padding=self.padding,
                                        use_gdn=False,
                                        name="encoder_level_1_scale")
        
        # ----------------------------------------------------------------
        # First level residual connections
        # ----------------------------------------------------------------
        
        if self.first_level_residual:
            
            self.first_level_res1 = ConvDS(output_channels=self.first_level_channels,
                                           kernel_shape=(7, 7),
                                           num_convolutions=1,
                                           padding=self.padding,
                                           downsampling_rate=4,
                                           use_gdn=False,
                                           name="encoder_level_1_res_1".format(idx))
            
            self.first_level_res2 = ConvDS(output_channels=self.first_level_channels,
                                           kernel_shape=(5, 5),
                                           num_convolutions=1,
                                           padding=self.padding,
                                           downsampling_rate=2,
                                           use_gdn=False,
                                           name="encoder_level_1_res_2".format(idx))

        # Second Level

        self.second_level = [
            ConvDS(output_channels=self.second_level_channels,
                   kernel_shape=self.kernel_shape,
                   num_convolutions=1,
                   downsampling_rate=1,
                   padding=self.padding,
                   use_gdn=False,
                   activation="leaky_relu"),
            ConvDS(output_channels=self.second_level_channels,
                   kernel_shape=self.kernel_shape,
                   num_convolutions=1,
                   downsampling_rate=1,
                   padding=self.padding,
                   use_gdn=False,
                   activation="leaky_relu")
        ]

        self.second_level_loc = ConvDS(output_channels=self.second_level_latents,
                                       kernel_shape=self.kernel_shape,
                                       num_convolutions=1,
                                       padding=self.padding,
                                       downsampling_rate=2,
                                       use_gdn=False,
                                       name="encoder_level_2_loc")

        self.second_level_log_scale = ConvDS(output_channels=self.second_level_latents,
                                         kernel_shape=self.kernel_shape,
                                         num_convolutions=1,
                                         padding=self.padding,
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

        if not self.first_level_residual:
            for layer in self.first_level:
                activations = layer(activations)
                
        else:
            activations = self.first_level[0](activations)
            
            # First residual activations
            res1 = self.first_level_res1(activations)
            
            activations = self.first_level[1](activations)
            
            # Second residual activations
            res2 = self.first_level_res2(activations)
            
            activations = self.first_level[2](activations)
            
            # Residual connections along the channels
            activations = tf.concat([activations, res1, res2], axis=-1)


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

        return tf.nn.sigmoid(activations)


    def _build(self, inputs):

        latents = self.encode(inputs)
        reconstruction = self.decode(latents)
        
        self.log_gamma = tf.get_variable("gamma", dtype=tf.float32, initializer=0.) if self.learn_log_gamma else 0.
        gamma = tf.exp(self.log_gamma)

        self.likelihood = self.likelihood_dist(loc=reconstruction,
                                               scale=gamma)

        self._log_prob = self.likelihood.log_prob(inputs)

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
    
# ==============================================================================================

    def code_image_greedy(self, 
                          image, 
                          seed, 
                          comp_file_path,
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
        
        
    
    def decode_image_greedy(self,
                            comp_file_path,
                            verbose=False):
        pass
                            