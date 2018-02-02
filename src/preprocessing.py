from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
import numpy as np

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
# 60k train, 10k test
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# pre-trained MNIST model
full_model = keras.models.load_model('mnist_cnn.h5')
plot_model(full_model, to_file='mnist_model.png')

feature_extractor = K.function([full_model.layers[0].input, K.learning_phase()],
                               [full_model.layers[6].output])

features_train = np.empty((x_train.shape[0], 128))
cnn_output_train = np.empty(y_train.shape)
features_test = np.empty((x_test.shape[0], 128))
cnn_output_test = np.empty(y_test.shape)


print("Computing on train set")
# training data
# iterate over batches
for i in range(0, x_train.shape[0] - batch_size, batch_size):
    if (i%1000 < batch_size):
        print("Running CNN on on training sample ", i)
    batch = x_train[i : i + batch_size]
    features = feature_extractor([batch, 0])[0]
    features_train[i: i + batch_size] = features
    softmax_outputs = full_model.predict_on_batch(batch)
    cnn_output_train[i : i+batch_size] = softmax_outputs
# handle last incomplete batch
features_train[i:] = feature_extractor([x_train[i:], 0])[0]
cnn_output_train[i:] = full_model.predict_on_batch(x_train[i:])

print("Computing on test set")

# testing data
for i in range(0, x_test.shape[0] - batch_size, batch_size):   
    if (i%1000 < batch_size):
        print("Running CNN on on test sample ", i)
    batch = x_test[i : i + batch_size]
    features = feature_extractor([batch, 0])[0]
    features_test[i: i + batch_size] = features
    softmax_outputs = full_model.predict_on_batch(batch)
    cnn_output_test[i : i+batch_size] = softmax_outputs
# handle last incomplete batch
features_test[i:] = feature_extractor([x_test[i:], 0])[0]
cnn_output_test[i:] = full_model.predict_on_batch(x_test[i:])

# write out to file
np.savetxt("mnist_train_features.csv", features_train, delimiter=",")
np.savetxt("mnist_train_cnn_output.csv", cnn_output_train, delimiter=",")
np.savetxt("mnist_test_features.csv", features_test, delimiter=",")
np.savetxt("mnist_test_cnn_output.csv", cnn_output_test, delimiter=",")

