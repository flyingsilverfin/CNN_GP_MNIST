import keras
from keras.datasets import mnist

def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return ((x_train, x_test), (y_train, y_test))

