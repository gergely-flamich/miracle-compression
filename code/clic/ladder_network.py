import os

import numpy as np

import tensorflow as tf
tfq = tf.quantization

import sonnet as snt

import tensorflow_probability as tfp
tfd = tfp.distributions

from miracle_modules import ConvDS
from compression import coded_sample, decode_sample
from coding import ArithmeticCoder
from binary_io import write_bin_code, read_bin_code
from greedy_compression import code_grouped_greedy_sample, decode_grouped_greedy_sample
from greedy_compression import code_grouped_importance_sample, decode_grouped_importance_sample

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
                 first_level_latent_dist="gaussian",
                 second_level_latent_dist="gaussian",
                 likelihood="gaussian",
                 heteroscedastic=False,
                 average_gamma=False,
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
        if first_level_latent_dist not in self._allowed_latent_dists:
            raise tf.errors.InvalidArgumentError("latent_dist must be one of {}"
                                                 .format(self._allowed_latent_dists))
        if second_level_latent_dist not in self._allowed_latent_dists:
            raise tf.errors.InvalidArgumentError("latent_dist must be one of {}"
                                                 .format(self._allowed_latent_dists))
        if likelihood not in self._allowed_likelihoods:
            raise tf.errors.InvalidArgumentError("likelihood must be one of {}"
                                                 .format(self._allowed_likelihoods))


        # Set variables
        self.first_level_latent_dist = self._allowed_latent_dists[first_level_latent_dist]
        self.second_level_latent_dist = self._allowed_latent_dists[second_level_latent_dist]
        self.likelihood_dist = self._allowed_likelihoods[likelihood]
        
        # Should we predict the variance too?
        self.heteroscedastic = heteroscedastic
        self.average_gamma = average_gamma
        
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
                                           name="encoder_level_1_res_1")
            
            self.first_level_res2 = ConvDS(output_channels=self.first_level_channels,
                                           kernel_shape=(5, 5),
                                           num_convolutions=1,
                                           padding=self.padding,
                                           downsampling_rate=2,
                                           use_gdn=False,
                                           name="encoder_level_1_res_2")

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
        self.second_level_posterior = self.second_level_latent_dist(loc=second_level_loc,
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
        self.first_level_posterior = self.first_level_latent_dist(loc=combined_loc,
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
        first_level = [self.first_level_loc.transpose(force_output_channels=self.first_level_channels)]

        for layer in self.first_level[::-1]:
            first_level.append(layer.transpose())
            
        if self.heteroscedastic:
            likelihood_log_scale_head = self.first_level[0].transpose()

        # ----------------------------------------------------------------------
        # Apply layers
        # ----------------------------------------------------------------------

        second_level_latents, first_level_latents = latents

        # Create second level prior
        self.second_level_prior = self.second_level_latent_dist(loc=tf.zeros_like(second_level_latents),
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
        self.first_level_prior = self.first_level_latent_dist(loc=first_level_loc,
                                                              scale=first_level_scale)

        # Apply first level
        activations = first_level_latents

        for layer in first_level[:-1]:
            activations = layer(activations)

        self.latent_priors = (self.first_level_prior, self.second_level_prior)

        reconstruction = first_level[-1](activations)
        reconstruction = tf.nn.sigmoid(reconstruction)
        
        if self.learn_log_gamma:
            self.log_gamma = tf.get_variable("gamma", dtype=tf.float32, initializer=0.)
        elif self.heteroscedastic:
            self.log_gamma = likelihood_log_scale_head(activations)
        else:
            self.log_gamma = 0.
        
        gamma = tf.exp(self.log_gamma)
        
        if self.average_gamma:
            gamma = tf.reduce_mean(gamma)

        self.likelihood = self.likelihood_dist(loc=reconstruction,
                                               scale=gamma)
        return reconstruction


    def _build(self, inputs):

        latents = self.encode(inputs)
        reconstruction = self.decode(latents)
        
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
    
        # Code second level
        coded_second_level, second_level_samp = coded_sample(proposal=self.latent_priors[1], 
                                                              target=self.latent_posteriors[1], 
                                                              seed=seed, 
                                                              n_points=n_points, 
                                                              miracle_bits=miracle_bits,
                                                              outlier_mode=outlier_mode)
        
        second_level_samp = tf.reshape(second_level_samp, self.latent_priors[1].loc.shape.as_list())
        
        # Set first level prior correctly
        latents = (second_level_samp, latents[1])
        self.decode(latents)
        
        # Code first level
        coded_first_level, _ = coded_sample(proposal=self.latent_priors[0], 
                                             target=self.latent_posteriors[0], 
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
        code, extras, _ = read_bin_code(comp_file_path, num_extras=6)
        
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
                          n_steps,
                          n_bits_per_step,
                          comp_file_path,
                          backfitting_steps_level_1=0,
                          backfitting_steps_level_2=0,
                          use_log_prob=False,
                          rho=1.,
                          use_importance_sampling=True,
                          first_level_max_group_size_bits=12,
                          second_level_n_bits_per_group=20,
                          second_level_max_group_size_bits=4,
                          second_level_dim_kl_bit_limit=12,
                          outlier_index_bytes=3,
                          outlier_sample_bytes=2,
                          verbose=False):
        
        # -------------------------------------------------------------------------------------
        # Step 1: Set the latent distributions for the image
        # -------------------------------------------------------------------------------------
        
        # Calculate the posteriors
        latents = self.encode(image)
        
        # Calculate the priors
        self.decode(latents)
        
        image_shape = image.shape.as_list()
        first_level_shape = self.latent_priors[0].loc.shape.as_list()
        second_level_shape = self.latent_priors[1].loc.shape.as_list()
        
        # -------------------------------------------------------------------------------------
        # Step 2: Create a coded sample of the latent space
        # -------------------------------------------------------------------------------------
        
        if verbose: print("Coding second level")
            
        if use_importance_sampling:
            
            sample2, code2, group_indices2, outlier_extras2 = code_grouped_importance_sample(target=self.latent_posteriors[1], 
                                                                         proposal=self.latent_priors[1], 
                                                                         n_bits_per_group=second_level_n_bits_per_group, 
                                                                         seed=seed, 
                                                                         max_group_size_bits=second_level_max_group_size_bits,
                                                                         dim_kl_bit_limit=second_level_dim_kl_bit_limit)
            
            outlier_extras2 = list(map(lambda x: tf.reshape(x, [-1]).numpy(), outlier_extras2))
            
        else:
            sample2, code2, group_indices2 = code_grouped_greedy_sample(target=self.latent_posteriors[1], 
                                                                        proposal=self.latent_priors[1], 
                                                                        n_bits_per_step=n_bits_per_step, 
                                                                        n_steps=n_steps, 
                                                                        seed=seed, 
                                                                        max_group_size_bits=second_level_max_group_size_bits,
                                                                        adaptive=True,
                                                                        backfitting_steps=backfitting_steps_level_2,
                                                                        use_log_prob=use_log_prob,
                                                                        rho=rho)
            
        # We will encode the group differences as this will cost us less
        group_differences2 = [0]
        
        for i in range(1, len(group_indices2)):
            group_differences2.append(group_indices2[i] - group_indices2[i - 1])
        
        # We need to adjust the priors to the second stage sample
        latents = (tf.reshape(sample2, second_level_shape), latents[1])
        
        # Calculate the priors
        self.decode(latents)
        
        if verbose: print("Coding first level")
            
        sample1, code1, group_indices1 = code_grouped_greedy_sample(target=self.latent_posteriors[0], 
                                                                    proposal=self.latent_priors[0], 
                                                                    n_bits_per_step=n_bits_per_step, 
                                                                    n_steps=n_steps, 
                                                                    seed=seed, 
                                                                    max_group_size_bits=first_level_max_group_size_bits,
                                                                    backfitting_steps=backfitting_steps_level_1,
                                                                    use_log_prob=use_log_prob,
                                                                    adaptive=True)
        
        # We will encode the group differences as this will cost us less
        group_differences1 = [0]
        
        for i in range(1, len(group_indices1)):
            group_differences1.append(group_indices1[i] - group_indices1[i - 1])
        
        bitcode = code1 + code2
        # -------------------------------------------------------------------------------------
        # Step 3: Write the compressed file
        # -------------------------------------------------------------------------------------
        
        extras = [seed, n_steps, n_bits_per_step] + first_level_shape[1:3] + second_level_shape[1:3]
        
        var_length_extras = [group_differences1, group_differences2]
        var_length_bits = [first_level_max_group_size_bits,  
                           second_level_max_group_size_bits]
        
        if use_importance_sampling:
            
            var_length_extras += outlier_extras2
            var_length_bits += [ outlier_index_bytes * 8, outlier_sample_bytes * 8 ]
    
        write_bin_code(bitcode, 
                       comp_file_path, 
                       extras=extras,
                       var_length_extras=var_length_extras,
                       var_length_bits=var_length_bits)
        
        # -------------------------------------------------------------------------------------
        # Step 4: Some logging information
        # -------------------------------------------------------------------------------------
        
        total_kls = [tf.reduce_sum(x) for x in self.kl_divergence]
        total_kl = sum(total_kls)

        theoretical_byte_size = (total_kl + 2 * np.log(total_kl + 1)) / np.log(2) / 8
        extra_byte_size = len(group_indices1) * var_length_bits[0] // 8 + \
                          len(group_indices2) * var_length_bits[1] // 8 + 7 * 2
        actual_byte_size = os.path.getsize(comp_file_path)

        actual_no_extra = actual_byte_size - extra_byte_size
        
        first_level_theoretical = (total_kls[0] + 2 * np.log(total_kls[0] + 1)) / np.log(2) / 8
        first_level_actual_no_extra = len(code1) / 8
        first_level_extra = len(group_indices1) * var_length_bits[0] // 8

        sample1_reshaped = tf.reshape(sample1, first_level_shape)
        first_level_avg_log_lik = tf.reduce_mean(self.latent_posteriors[0].log_prob(sample1_reshaped))
        first_level_sample_avg = tf.reduce_mean(self.latent_posteriors[0].log_prob(self.latent_posteriors[0].sample()))
        
        second_level_theoretical = (total_kls[1] + 2 * np.log(total_kls[1] + 1)) / np.log(2) / 8
        second_level_actual_no_extra = len(code2) / 8
        second_level_extra = len(group_indices2) * var_length_bits[1] // 8 + 1
        
        second_bpp = (second_level_actual_no_extra + second_level_extra) * 8 / (image_shape[1] * image_shape[2]) 

        sample2_reshaped = tf.reshape(sample2, second_level_shape)
        second_level_avg_log_lik = tf.reduce_mean(self.latent_posteriors[1].log_prob(sample2_reshaped))
        second_level_sample_avg = tf.reduce_mean(self.latent_posteriors[1].log_prob(self.latent_posteriors[1].sample()))
        
        bpp = 8 * actual_byte_size / (image_shape[1] * image_shape[2]) 
        
        summaries = {
            "image_shape": image_shape,
            "theoretical_byte_size": theoretical_byte_size,
            "actual_byte_size": actual_byte_size,
            "extra_byte_size": extra_byte_size,
            "actual_no_extra": actual_no_extra,
            "second_bpp": second_bpp,
            "bpp": bpp
        }
        
        if verbose:

            print("Image dimensions: {}".format(image_shape))
            print("Theoretical size: {:.2f} bytes".format(theoretical_byte_size))
            print("Actual size: {:.2f} bytes".format(actual_byte_size))
            print("Extra information size: {:.2f} bytes {:.2f}% of actual size".format(extra_byte_size, 
                                                                                       100 * extra_byte_size / actual_byte_size))
            print("Actual size without extras: {:.2f} bytes".format(actual_no_extra))
            print("Efficiency: {:.3f}".format(actual_byte_size / theoretical_byte_size))
            print("")
            
            print("First level theoretical size: {:.2f} bytes".format(first_level_theoretical))
            print("First level actual (no extras) size: {:.2f} bytes".format(first_level_actual_no_extra))
            print("First level extras size: {:.2f} bytes".format(first_level_extra))
            print("First level Efficiency: {:.3f}".format(
                (first_level_actual_no_extra + first_level_extra) / first_level_theoretical))
            
            print("First level # of groups: {}".format(len(group_indices1)))
            print("First level greedy sample average log likelihood: {:.4f}".format(first_level_avg_log_lik))
            print("First level average sample log likelihood on level 1: {:.4f}".format(first_level_sample_avg))
            print("")
           
            print("Second level theoretical size: {:.2f} bytes".format(second_level_theoretical))
            print("Second level actual (no extras) size: {:.2f} bytes".format(second_level_actual_no_extra))
            print("Second level extras size: {:.2f} bytes".format(second_level_extra))

            if use_importance_sampling:
                print("{} outliers were not compressed (higher than {} bits of KL)".format(len(outlier_extras2[0]),
                                                                                           second_level_dim_kl_bit_limit))
            print("Second level Efficiency: {:.3f}".format(
                (second_level_actual_no_extra + second_level_extra) / second_level_theoretical))
            print("Second level's contribution to bpp: {:.4f}".format(second_bpp))
            print("Second level # of groups: {}".format(len(group_indices2)))
            print("Second level greedy sample average log likelihood: {:.4f}".format(second_level_avg_log_lik))
            print("Second level average sample log likelihood on level 1: {:.4f}".format(second_level_sample_avg))
            print("")
            
            print("{:.4f} bits / pixel".format( bpp ))
        
        return (sample2, sample1), summaries
        
    
    def decode_image_greedy(self,
                            comp_file_path,
                            use_importance_sampling=True,
                            rho=1.,
                            verbose=False):
        
        # -------------------------------------------------------------------------------------
        # Step 1: Read the compressed file
        # -------------------------------------------------------------------------------------
        
        # the extras are: seed, n_steps, n_bits_per_step and W x H of the two latent levels
        # var length extras are the two lists of group indices
        num_var_length_extras = 2
        
        if use_importance_sampling:
            num_var_length_extras += 2
        
        code, extras, var_length_extras = read_bin_code(comp_file_path, 
                                                        num_extras=7, 
                                                        num_var_length_extras=num_var_length_extras)
        
        seed = extras[0]
        
        n_steps = extras[1]
        n_bits_per_step = extras[2]
        
        # Get shape information back
        first_level_shape = [1] + extras[3:5] + [self.first_level_latents]
        second_level_shape = [1] + extras[5:] + [self.second_level_latents]
        
        # Total number of latents on levels
        num_first_level = np.prod(first_level_shape)
        num_second_level = np.prod(second_level_shape)
        
        first_code_length = n_steps * n_bits_per_step * (len(var_length_extras[0]) - 1)
        second_code_length = n_steps * n_bits_per_step * (len(var_length_extras[1]) - 1)
        
        code1 = code[:first_code_length]
        code2 = code[first_code_length:first_code_length + second_code_length]
        
        # -------------------------------------------------------------------------------------
        # Step 2: Decode the samples
        # -------------------------------------------------------------------------------------
        
        # Decode second level
        proposal = tfd.Normal(loc=tf.zeros(second_level_shape),
                              scale=tf.ones(second_level_shape))
        
        
        # Get group indices back
        group_differences2 = var_length_extras[1]
        
        group_indices2 = [0]
        
        for i in range(1, len(group_differences2)):
            group_indices2.append(group_indices2[i - 1] + group_differences2[i])
        
        print("Decoding second level")
        if use_importance_sampling:
            decoded_second_level = decode_grouped_importance_sample(bitcode=code2, 
                                                                    group_start_indices=group_indices2[:-1],
                                                                    proposal=proposal, 
                                                                    n_bits_per_group=20,
                                                                    seed=seed,
                                                                    outlier_indices=var_length_extras[2],
                                                                    outlier_samples=var_length_extras[3])
        
        else:
            decoded_second_level = decode_grouped_greedy_sample(bitcode=code2, 
                                                                group_start_indices=var_length_extras[1],
                                                                proposal=proposal, 
                                                                n_bits_per_step=n_bits_per_step, 
                                                                n_steps=n_steps, 
                                                                seed=seed,
                                                                rho=rho,
                                                                adaptive=True)
        
        decoded_second_level = tf.reshape(decoded_second_level, second_level_shape)
        
        # Now we can calculate the the first level priors
        self.decode((decoded_second_level,
                     tf.zeros(first_level_shape)))
        
        # Get group indices back
        group_differences1 = var_length_extras[0]
        
        group_indices1 = [0]
        
        for i in range(1, len(group_differences1)):
            group_indices1.append(group_indices1[i - 1] + group_differences1[i])
        
        # Decode first level
        print("Decoding first level")
        decoded_first_level = decode_grouped_greedy_sample(bitcode=code1, 
                                                            group_start_indices=group_indices1,
                                                            proposal=self.latent_priors[0], 
                                                            n_bits_per_step=n_bits_per_step, 
                                                            n_steps=n_steps, 
                                                            seed=seed,
                                                            rho=rho,
                                                            adaptive=True)
        
        decoded_first_level = tf.reshape(decoded_first_level, first_level_shape)
        
        
        # -------------------------------------------------------------------------------------
        # Step 4: Reconstruct the image with the VAE
        # -------------------------------------------------------------------------------------
        
        reconstruction = self.decode((decoded_second_level,
                                      decoded_first_level))
        
        return tf.squeeze(reconstruction)
        