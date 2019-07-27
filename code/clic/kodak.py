import requests, os, glob

from imageio import imwrite

import tensorflow as tf
import numpy as np

from load_data import load_and_process_image

from greedy_compression import code_grouped_greedy_sample, decode_grouped_greedy_sample
from greedy_compression import code_grouped_importance_sample, decode_grouped_importance_sample
                
transpose_list = lambda l: list(map(list, zip(*l)))


def download_kodak_dataset(download_path, 
                           kodak_url_format = "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"):
    
    num_kodak_images = 24
    kodak_file_format = download_path  + "/kodim{:02d}.png"
    
    if not os.path.exists(download_path):
        print("Creating download directory " + download_path)
        
        os.makedirs(download_path)
    
    for i in range(1, num_kodak_images + 1):
        
        if os.path.exists(kodak_file_format.format(i)):
            continue
        
        print("Downloading image " + kodak_url_format.format(i))
        
        with open(kodak_file_format.format(i), 'wb') as handle:
            response = requests.get(kodak_url_format.format(i), stream=True)

            if not response.ok:
                print(response)

            for block in response.iter_content(1024):
                if not block:
                    break

                handle.write(block)

                
                
def create_kodak_dataset(kodak_dataset_path="../../data/kodak/",
                         n_kodak_images=24,
                         kodak_im_format="/kodim{:02d}.png"):
    
    download_kodak_dataset(kodak_dataset_path)
    
    kodak_im_paths = [kodak_dataset_path + kodak_im_format.format(i) for i in range(1, n_kodak_images + 1)]

    kodak_paths_ds = tf.data.Dataset.from_tensor_slices(kodak_im_paths)
    
    kodak_ds = kodak_paths_ds.map(load_and_process_image)
    
    return kodak_ds
    

    
def create_kodak_comparison_dataset(comparison_dataset_path,
                                    comparison_dataset_subdir,
                                    n_kodak_images=24,
                                    kodak_im_format="/kodim{:02d}.png"):
    
    comp_ds_dirs = glob.glob(comparison_dataset_path + "/" + comparison_dataset_subdir + "*")
    comp_ds_dirs.sort()
    
    comp_ds_folders = [os.path.basename(p) for p in comp_ds_dirs]
    
    comp_ds_im_paths = [[comp_ds_dir + kodak_im_format.format(i) for i in range(1, n_kodak_images + 1)]
                        for comp_ds_dir in comp_ds_dirs]
    
    comp_ds_paths_ds = tf.data.Dataset.from_tensor_slices(transpose_list(comp_ds_im_paths))
    
    comp_ds = comp_ds_paths_ds.map(lambda x: tf.map_fn(load_and_process_image, x, dtype=tf.float32))
    
    return comp_ds
                

    
def compress_kodak(kodak_dataset_path,
                   reconstruction_subdir,
                   vae,
                   backfitting_steps_level_1=0,
                   use_log_prob=True,
                   reconstruction_root="../../data/kodak_miracle/",
                   n_kodak_images=24,
                   kodak_im_format="kodim{:02d}.png",
                   comp_file_format="/kodim{:02d}.miracle",
                   theoretical=None,
                   verbose=False):
    
    reconstruction_path = reconstruction_root + "/" + reconstruction_subdir
    
    if not os.path.exists(reconstruction_path):
        print("Creating reconstruction directory " + reconstruction_path)
        
        os.makedirs(reconstruction_path)
    
    # Create lists of paths for every image in the dataset
    kodak_im_paths = [kodak_dataset_path + "/" + kodak_im_format.format(i) 
                      for i in range(1, n_kodak_images + 1)]
    
    reconstruction_im_paths = [reconstruction_path + "/" + kodak_im_format.format(i) 
                               for i in range(1, n_kodak_images + 1)]
    
    comp_file_paths = [reconstruction_path + "/" + comp_file_format.format(i) 
                       for i in range(1, n_kodak_images + 1)]
    
    # Load in the Kodak dataset
    kodak_paths_ds = tf.data.Dataset.from_tensor_slices(kodak_im_paths)
    kodak_ds = kodak_paths_ds.map(load_and_process_image)
    
    n_bits_per_step = 14
    n_steps = 30
    seed = 1
    rho = 1.
    first_level_max_group_size_bits=12
    second_level_max_group_size_bits=4

    for i, kodak_im in enumerate(kodak_ds):
        
        kodak_im = kodak_im[tf.newaxis, ...]
        
        if theoretical is not None:
            
            # Everything is sampled from the true posterior
            if theoretical == "full":
                reconstruction = vae(kodak_im)
                
            # The first level is sampled using the coded method, 
            # the second level is sampled from the true posterior
            elif theoretical == "coded_first":
            
                # Set priors and posteriors
                vae(kodak_im)
                
                sample1, _, _ = code_grouped_greedy_sample(target=vae.latent_posteriors[0], 
                                                            proposal=vae.latent_priors[0], 
                                                            n_bits_per_step=n_bits_per_step, 
                                                            n_steps=n_steps, 
                                                            seed=seed, 
                                                            max_group_size_bits=first_level_max_group_size_bits,
                                                            backfitting_steps=backfitting_steps_level_1,
                                                            use_log_prob=use_log_prob,
                                                            adaptive=True)
                
                reconstruction = vae.decode((tf.zeros_like(vae.latent_priors[1].loc),
                                             tf.reshape(sample1, vae.latent_priors[0].loc.shape.as_list())
                                            ))

            else:
                raise Exception("unrecognised theoretical setup: " + theoretical)
                
            reconstruction = tf.cast(tf.squeeze(255 * reconstruction), tf.uint8).numpy()
            
            print("Writing " + reconstruction_im_paths[i])
            imwrite(reconstruction_im_paths[i], reconstruction)
            
            total_kl = sum([tf.reduce_sum(x) for x in vae.kl_divergence])
            theoretical_byte_size = (total_kl + 2 * np.log(total_kl + 1)) / np.log(2)
            
            image_shape = kodak_im.shape.as_list()
            
            bpp = theoretical_byte_size / (image_shape[1] * image_shape[2]) 
            
            with open(reconstruction_path + "/bpp.csv", "a") as log_csv:
                log_csv.write(kodak_im_format.format(i + 1) + ",{:.4f}\n".format(bpp))
                
                
                
        else:
            if os.path.exists(comp_file_paths[i]):
                print(comp_file_paths[i] + " already exists, skipping coding.")

            else:
                _, summaries = vae.code_image_greedy(image=kodak_im,
                                                    seed=seed,
                                                    rho=rho,
                                                    n_steps=n_steps,
                                                    n_bits_per_step=n_bits_per_step,
                                                    comp_file_path=comp_file_paths[i],
                                                    backfitting_steps_level_1=backfitting_steps_level_1,
                                                    backfitting_steps_level_2=0,
                                                    use_log_prob=use_log_prob,
                                                    first_level_max_group_size_bits=first_level_max_group_size_bits,
                                                    second_level_n_bits_per_group=20,
                                                    second_level_max_group_size_bits=second_level_max_group_size_bits,
                                                    second_level_dim_kl_bit_limit=12,
                                                    verbose=verbose)

                with open(reconstruction_path + "/bpp.csv", "a") as log_csv:
                    efficiency = summaries["actual_byte_size"] / summaries["theoretical_byte_size"]
                    log_csv.write(kodak_im_format.format(i + 1) + ",{:.4f},{:.4f}\n".format(summaries["bpp"],
                                                                                            efficiency))

            if os.path.exists(reconstruction_im_paths[i]):
                print(reconstruction_im_paths[i] + " already exists, skipping reconstruction.")

            else:
                reconstruction = vae.decode_image_greedy(comp_file_path=comp_file_paths[i],
                                                         verbose=verbose,
                                                         rho=rho)
                
                print("Writing " + reconstruction_im_paths[i])
                
                reconstruction = tf.cast(tf.squeeze(255 * reconstruction), tf.uint8).numpy()
                imwrite(reconstruction_im_paths[i], reconstruction)