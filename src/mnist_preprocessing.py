import keras
from keras.datasets import mnist
from keras import backend as K

import numpy as np

def preprocess_images(images):
    xs = images.copy()
    img_rows, img_cols = 28, 28   #TODO make this flexible 
    # reshape to inputs to correct shape
    if K.image_data_format() == 'channels_first':
        xs = xs.reshape(xs.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        xs = xs.reshape(xs.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    xs = xs.astype('float32')
    xs /= 255

    return xs
    

def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)
    return ((x_train, x_test), (y_train, y_test))

