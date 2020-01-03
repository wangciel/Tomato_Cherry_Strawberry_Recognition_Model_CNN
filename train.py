#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2019 Created by Yiming Peng and Bing Xue
"""

'''
Student Name: YiFan Wang
Student ID:300304266
'''
from __future__ import print_function
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
import random

from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from sklearn.model_selection import KFold
from util.cnn_model_maker import make_model
from util.data_loader import load_cnn_train_data


# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

# Hyper parameters
batch_size = 45
num_classes = 3
epochs = 8

# input image dimensions
img_rows, img_cols = 64, 64

data,labels = load_cnn_train_data(with_imputation=True)

def format_image_data(data):
    if K.image_data_format() == 'channels_first':
        data = data.reshape(data.shape[0], 3, img_rows, img_cols)
        input_shape = (3, img_rows, img_cols)
    else:
        data = data.reshape(data.shape[0], img_rows, img_cols,3)
        input_shape = (img_rows, img_cols,3)

    data = data.astype('float32')
    data /= 255
    print('x_train shape:', data.shape)
    print(data.shape[0], 'train samples, test samples')

    return data,input_shape

data, input_shape = format_image_data(data)

def construct_model():
    model = make_model(input_shape)
    model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['categorical_accuracy'])

    return model


def train_model(model,x_train, y_train, x_test, y_test):

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    return model, history


def save_model(model, dir):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """

    model.save(dir)
    print("Model Saved Successfully.")


def plot_learning_curve(values, xlabel, ylabel, save_dir, title):

    plot = plt.figure()
    for value in values:
        plt.plot(value)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(['train', 'test'], loc='lower right')
    plot.savefig(save_dir)


if __name__ == '__main__':
    model = construct_model()

    accuracy, val_accuracy  = [],[]
    loss, val_loss  = [],[]
    test_loss,test_accuracy = [],[]

    kf = KFold(n_splits=4,shuffle=True)
    start_time = time.time()

    for train_index, test_index in kf.split(data):
        x_train, y_train, x_test, y_test = data[train_index], labels[train_index], data[test_index], labels[test_index]
        model,history = train_model(model, x_train, y_train, x_test, y_test)

        accuracy.append(history.history['categorical_accuracy'])
        val_accuracy.append(history.history['val_categorical_accuracy'])
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])

        score = model.evaluate(x_test, y_test, verbose=0)
        test_loss.append(score[0])
        test_accuracy.append(score[1])

    training_time = time.time() - start_time
    print("--- training time %s seconds ---" % (time.time() - start_time))


    save_model(model, dir="model/model.h5")
    plot_learning_curve([np.mean(accuracy, axis=0), np.mean(val_accuracy, axis=0)], 'epoch', 'accuracy',
                        'plots/acc_of_CNN.png', 'acc_of_CNN')
    plot_learning_curve([np.mean(loss, axis=0), np.mean(val_loss, axis=0)], 'epoch', 'loss',
                        'plots/loss_of_CNN.png', 'loss_of_CNN')

    print('Test loss:', np.mean(test_loss))
    print('Test accuracy:', np.mean(test_accuracy))

