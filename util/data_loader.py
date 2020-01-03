'''
Student Name: YiFan Wang
Student ID:300304266
'''
from tensorflow.keras.preprocessing.image import img_to_array

# You need to install "imutils" lib by the following command:
#               pip install imutils
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import argparse
import numpy as np
import random
import tensorflow as tf

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--train_data_dir", default = "./Train_data",
                      help = "path to train_data_dir")

    args.add_argument("--train_data_imputation_dir", default="./Train_data_imputation",
                      help="path to train_data_dir")

    args = vars(args.parse_args())
    return args

def img_preprocessing(image, crop):
    height, width, channels = image.shape

    if crop is not None:
        height_cropped, width_cropped = int(height * crop), int(width * crop)
        image = image[height_cropped:-height_cropped, width_cropped:-width_cropped]

    return image


def load_images(test_data_dir, image_size = (300, 300)):
    """
    Load images from local directory
    :return: the image list (encoded as an array)
    """
    # loop over the input images
    images_data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(test_data_dir)))

    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)

        processed_img = img_preprocessing(image,crop=0.2)

        image = cv2.resize(processed_img, image_size)
        image = img_to_array(image)
        images_data.append(image)

        # extract the class label from the image path and update the
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)

    return images_data, sorted(labels)


def convert_img_to_array(images, labels):
    # Convert to numpy and do constant normalize
    X_test = np.array(images, dtype="float")
    y_test = np.array(labels)

    #Binarize the labels
    # convert class vectors to binary class matrices
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)

    return X_test, np.array(y_test)


def load_cnn_train_data(with_imputation):
    # Parse the arguments
    args = parse_args()

    # Train Data folder
    train_data_dir = args["train_data_dir"]
    train_data_imputation_dir = args["train_data_imputation_dir"]

    # Image size, please define according to your settings when training your model.
    image_size = (64, 64)

    if with_imputation:
        images1, labels1 = load_images(train_data_dir, image_size)

        # imputation images
        images2, labels2 = load_images(train_data_imputation_dir, image_size)
        data, labels = convert_img_to_array(images1+images2, labels1+labels2)

    else:
        # Load images
        images, labels = load_images(train_data_dir, image_size)
        data, labels = convert_img_to_array(images, labels)
    return data, labels


def load_mlp_train_data():
    # Parse the arguments
    args = parse_args()

    # Train folder
    test_data_dir = args["train_data_dir"]

    # Image size, please define according to your settings when training your model.
    image_size = (32, 32)

    # Load images
    images, labels = load_images(test_data_dir, image_size)

    y_test = np.array(labels)

    x_test = []
    for image in images:
        merge_channel_mean = np.mean(image, axis = 2)
        flat_image = np.concatenate(merge_channel_mean)
        x_test.append(flat_image)

    return np.array(x_test),y_test


