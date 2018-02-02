import numpy as np                      # MATLAB-style matrix and vector manipulation

from keras.datasets import mnist
import keras


def get_mnist_classes():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (y_train, y_test)




"""
So SciPy runs out of memory very quickly even when cutting down to 12k examples for 0's and 1's and only 10 features instead of 128...
"""
class Scipy_GP(object):
    def __init__(self, xs, ys):
        import sklearn.gaussian_process as gp   # Gaussian process modeling


        self.gp = gp

        # limit data to speed up development
        self.limit_classes = 2
        self.limit_features = 10
        self.xs, self.ys = self.filter(xs, ys)


    def filter(self, xs, ys):
        print("Filtering xs and ys to ", self.limit_features, " features and ", self.limit_classes, " classes")
        # note ys are expected to be 1-hot encoded

        # TODO rewrite this numpy-esque
        filtered_xs = [[] for i in range(self.limit_classes)]
        filtered_ys = [[] for i in range(self.limit_classes)]

        print(filtered_ys)

        for (x,y) in zip(xs, ys):
            for i in range(self.limit_classes):
                one_hot_class = keras.utils.to_categorical(i, ys.shape[-1])
                #print(one_hot_class, y)
                if np.array_equal(y, one_hot_class):
                    filtered_xs[i].append(x[:self.limit_features])
                    filtered_ys[i].append(y[:self.limit_classes])
                    break


        filtered_xs = [np.array(x) for x in  filtered_xs]
        filtered_ys = [np.array(x) for x in  filtered_ys]

        # collapse the per-class matrices into one big matrix
        return (np.vstack(filtered_xs), np.vstack(filtered_ys))

    def train(self):
        print("Beginning training")
        nu = 1.0
        sigma = 1.0
        ls = [np.exp(-1) for i in range(self.ys.shape[-1])]
        kernel = nu**2 * self.gp.kernels.RBF(length_scale=ls) + self.gp.kernels.WhiteKernel(noise_level=sigma)
        model = self.gp.GaussianProcessRegressor(kernel=kernel) 
        model.fit(self.xs, self.ys)






if __name__ == "__main__":
    xs_train = np.genfromtxt("../data/mnist_train_features.csv", delimiter=",")
    #nn_train = np.genfromtxt("../data/mnist_train_cnn_output.csv", delimiter=",")
    xs_test = np.genfromtxt("../data/mnist_test_features.csv", delimiter=",")
    #nn_test = np.genfromtxt("../data/mnist_test_cnn_output.csv",  delimiter=",")

    ys_train, ys_test = get_mnist_classes()

    gp = Scipy_GP(xs_train, ys_train)
    gp.train()
