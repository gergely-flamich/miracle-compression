import os, glob
from tqdm import tqdm

from imageio import imwrite

import tensorflow as tf
tf.enable_eager_execution()

COMPRESSION_SUBDIR = 'datasets/miracle_image_compression'

TRAIN_DATASET_URL = 'https://data.vision.ee.ethz.ch/cvl/clic/professional_train.zip'
TRAIN_DATASET_ARCHIVE = 'train.zip'

VALIDATION_DATASET_URL = 'https://data.vision.ee.ethz.ch/cvl/clic/professional_valid.zip'
VALIDATION_DATASET_ARCHIVE = 'valid.zip'


def process_image(image, normalize=True):
    """
    normalize will adjust all pixels of the image to lie between 0 and 1
    for every colour dimension.
    """

    img_tensor = tf.image.decode_image(image)

    if normalize:
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor /= 255.

    return img_tensor


def load_and_process_image(image_path, normalize=True):

    img_raw = tf.read_file(image_path)
    return process_image(img_raw, normalize=normalize)


def create_random_crops(image, crop_coef=5, crop_size=256):

    w = image.shape[0]
    h = image.shape[1]

    num_crops = crop_coef * (w // crop_size) * (h // crop_size)

    crops = []

    for i in range(num_crops):
        crops.append(tf.image.random_crop(image,
                                          size=(crop_size, crop_size, 3)))

    return crops


def download_process_and_load_data():

    # Download stuff using Keras' utilities
    train_path = tf.keras.utils.get_file(fname=TRAIN_DATASET_ARCHIVE,
                                         origin=TRAIN_DATASET_URL,
                                         cache_subdir=COMPRESSION_SUBDIR,
                                         extract=True)

    valid_path = tf.keras.utils.get_file(fname=VALIDATION_DATASET_ARCHIVE,
                                         origin=VALIDATION_DATASET_URL,
                                         cache_subdir=COMPRESSION_SUBDIR,
                                         extract=True)

    # Get the images
    train_path, _ = os.path.splitext(train_path)
    valid_path, _ = os.path.splitext(valid_path)

    train_image_paths = glob.glob(train_path + "/*.png")
    valid_image_paths = glob.glob(valid_path + "/*.png")

    train_paths_dataset = tf.data.Dataset.from_tensor_slices(train_image_paths)
    train_image_dataset = train_paths_dataset.map(lambda im: load_and_process_image(im, normalize=False),
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

    valid_paths_dataset = tf.data.Dataset.from_tensor_slices(valid_image_paths)
    valid_image_dataset = valid_paths_dataset.map(lambda im: load_and_process_image(im, normalize=False),
                                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)


    # Process the images
    train_processed_path = train_path + "_processed/"
    valid_processed_path = valid_path + "_processed/"

    print("Processing Training Data!")
    if not os.path.exists(train_processed_path):
        os.mkdir(train_processed_path)


        for idx, image in enumerate(train_image_dataset):

            print("Processing image {} out of {}".format(idx, len(train_image_paths)))

            crops = create_random_crops(image)

            for i, crop in tqdm(enumerate(crops), total=len(crops)):
                
                imwrite(train_processed_path + "{}_{}.png".format(idx, i), crop.numpy())

            del crops

    else:
        print("Data already processed!")


    print("Processing Validation Data!")
    if not os.path.exists(valid_processed_path):
        os.mkdir(valid_processed_path)


        for idx, image in enumerate(valid_image_dataset):

            print("Processing image {} out of {}".format(idx, len(valid_image_paths)))

            crops = create_random_crops(image)

            for i, crop in tqdm(enumerate(crops), total=len(crops)):
                imwrite(valid_processed_path + "{}_{}.png".format(idx, i), crop.numpy())

            del crops

    else:
        print("Data already processed!")


    # Load the data as datasets
    train_image_paths = glob.glob(train_processed_path + "/*.png")
    valid_image_paths = glob.glob(valid_processed_path + "/*.png")

    train_paths_dataset = tf.data.Dataset.from_tensor_slices(train_image_paths)
    train_image_dataset = train_paths_dataset.map(lambda im: load_and_process_image(im, normalize=True),
                                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

    valid_paths_dataset = tf.data.Dataset.from_tensor_slices(valid_image_paths)
    valid_image_dataset = valid_paths_dataset.map(lambda im: load_and_process_image(im, normalize=True),
                                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_image_dataset, valid_image_dataset