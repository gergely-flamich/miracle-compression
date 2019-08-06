import requests, os, glob, time, json

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
                                    comparison_dataset_subdir="*",
                                    n_kodak_images=24,
                                    kodak_im_format="/kodim{:02d}.png"):
    
    comp_ds_dirs = glob.glob(comparison_dataset_path + "/" + comparison_dataset_subdir + "*")
    comp_ds_dirs = list(filter(os.path.isdir, comp_ds_dirs))
    comp_ds_dirs.sort()
    
    comp_ds_folders = [os.path.basename(p) for p in comp_ds_dirs]
    
    comp_ds_im_paths = [[comp_ds_dir + kodak_im_format.format(i) for i in range(1, n_kodak_images + 1)]
                        for comp_ds_dir in comp_ds_dirs]
    
    comp_ds_paths_ds = tf.data.Dataset.from_tensor_slices(transpose_list(comp_ds_im_paths))
    
    comp_ds = comp_ds_paths_ds.map(lambda x: tf.map_fn(load_and_process_image, x, dtype=tf.float32))
    
    return comp_ds, comp_ds_paths_ds
                

    
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
    
    
    if theoretical is not None:
        reconstruction_subdir = "theoretical_" + reconstruction_subdir
    
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
        
        kodak_im_name = kodak_im_format.format(i + 1)
        stats_path = reconstruction_root + "/stats.json"
        
        if theoretical is not None:
            
            # Everything is sampled from the true posterior
            if theoretical == "full":
                reconstruction = vae(kodak_im)
                
                encoding_time = -1
                decoding_time = -1
                
            # The first level is sampled using the coded method, 
            # the second level is sampled from the true posterior
            elif theoretical == "coded_first":
            
                # Set priors and posteriors
                vae(kodak_im)
                
                start_time = time.time()
                
                sample1, _, _ = code_grouped_greedy_sample(target=vae.latent_posteriors[0], 
                                                            proposal=vae.latent_priors[0], 
                                                            n_bits_per_step=n_bits_per_step, 
                                                            n_steps=n_steps, 
                                                            seed=seed, 
                                                            max_group_size_bits=first_level_max_group_size_bits,
                                                            backfitting_steps=backfitting_steps_level_1,
                                                            use_log_prob=use_log_prob,
                                                            adaptive=True)
                
                encoding_time = time.time() - start_time
                
                start_time = time.time()
                
                reconstruction = vae.decode((tf.zeros_like(vae.latent_priors[1].loc),
                                             tf.reshape(sample1, vae.latent_priors[0].loc.shape.as_list())
                                            ))
                
                decoding_time = time.time() - start_time

            else:
                raise Exception("unrecognised theoretical setup: " + theoretical)
                
            
            
            
            total_kl = sum([tf.reduce_sum(x) for x in vae.kl_divergence])
            theoretical_byte_size = (total_kl + 2 * np.log(total_kl + 1)) / np.log(2)
            
            image_shape = kodak_im.shape.as_list()
            
            bpp = theoretical_byte_size / (image_shape[1] * image_shape[2]) 

            summaries = {"bpp": float(bpp.numpy()),
                         "encoding_time": encoding_time,
                         "decoding_time": decoding_time,
                         "total_time": encoding_time + decoding_time}
            
        # Non-theoretical reconstruction
        else:
            if os.path.exists(comp_file_paths[i]):
                print(comp_file_paths[i] + " already exists, skipping coding.")

            else:
                
                start_time = time.time()
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
                
                encoding_time = time.time() - start_time
                
            if os.path.exists(reconstruction_im_paths[i]):
                print(reconstruction_im_paths[i] + " already exists, skipping reconstruction.")

            else:
                start_time = time.time()
                reconstruction = vae.decode_image_greedy(comp_file_path=comp_file_paths[i],
                                                         verbose=verbose,
                                                         rho=rho)
                decoding_time = time.time() - start_time
                print("Writing " + reconstruction_im_paths[i])
          
        if theoretical or not os.path.exists(reconstruction_im_paths[i]):
            ms_ssim = tf.image.ssim_multiscale(kodak_im, reconstruction, max_val=1.0)
            psnr = tf.image.psnr(kodak_im, reconstruction, max_val=1.0)    

            if not os.path.exists(reconstruction_im_paths[i]):
                reconstruction = tf.cast(tf.squeeze(255 * reconstruction), tf.uint8).numpy()
                imwrite(reconstruction_im_paths[i], reconstruction)

            summaries["encoding_time"] = encoding_time,
            summaries["decoding_time"] = decoding_time,
            summaries["total_time"] = encoding_time + decoding_time
            summaries["ms_ssim"] = float(ms_ssim.numpy())
            summaries["psnr"] = float(psnr.numpy())

            print(summaries)

            if os.path.exists(stats_path):
                with open(stats_path, "r") as stats_fp:
                    stats = json.load(stats_fp)
            else:
                stats = {}

            if kodak_im_name not in stats:
                stats[kodak_im_name] = {}

            with open(stats_path, "w") as stats_fp:
                stats[kodak_im_name][reconstruction_subdir] = summaries

                json.dump(stats, stats_fp)
            
            
def baseline_compress_kodak(kodak_dataset_path,
                           reconstruction_root="../../data/kodak_miracle/",
                           n_kodak_images=24,
                           kodak_im_format="kodim{:02d}",
                           jpeg_quality_steps=7,
                           verbose=False):
  
    # Create lists of paths for every image in the dataset
    kodak_im_paths = [kodak_dataset_path + "/" + kodak_im_format.format(i) + ".png"
                      for i in range(1, n_kodak_images + 1)]

    # Load in the Kodak dataset
    kodak_paths_ds = tf.data.Dataset.from_tensor_slices(kodak_im_paths)
    kodak_ds = kodak_paths_ds.map(load_and_process_image)

    jpeg_max_quality = 95
    jpeg_min_quality = 1

    # Create JPEG baseline
    for j in range(jpeg_min_quality, jpeg_max_quality, jpeg_quality_steps):

        print("JPEG quality setting: {}".format(j))
        
        reconstruction_path = reconstruction_root + "/jpeg_{}".format(j)
        
        if not os.path.exists(reconstruction_path):
            print("Creating reconstruction directory " + reconstruction_path)

            os.makedirs(reconstruction_path)

        reconstruction_im_paths = [reconstruction_path + "/" + kodak_im_format.format(i) + ".jpg"
                                   for i in range(1, n_kodak_images + 1)]
            
        for i, kodak_im in tqdm(enumerate(kodak_ds)):
            imsave(reconstruction_im_paths[i], quality=j)
            
        # TODO: Create JPEG2000 baseline

        # TODO: Create WebP baseline
    