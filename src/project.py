import abc

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np                      # MATLAB-style matrix and vector manipulation
from keras.datasets import mnist
import keras


def get_mnist_classes():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (y_train, y_test)



# centralize filtering, configurations to comming superclass
# Specialize imports 
# and implementation
class GP(object):
    def __init__(self, xs_train, ys_train, limit_classes = 2, limit_features = 10):
        self.limit_classes = limit_classes
        self.limit_features = limit_features
        self.xs_train, self.ys_train = self._filter(xs_train, ys_train)

    def _filter(self, xs, ys):
        print("Filtering xs and ys to ", self.limit_features, " features and ", self.limit_classes, " classes")
        # note ys are expected to be 1-hot encoded

        # TODO rewrite this more numpy-esque
        filtered_xs = [[] for i in range(self.limit_classes)]
        filtered_ys = [[] for i in range(self.limit_classes)]

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

    @abc.abstractmethod
    def train(self):
        print("Abstract train method")
        pass

    @abc.abstractmethod
    def predict(self):
        print("Abstract predict method")
        pass

    def plot(self, xs, ys, mean, var, training = None):
        if ys.shape[0] > 1:
            print("Cannot plot >1 dimensional predictions")
            return
        # 1 dimensional input
        if xs.shape[0] == 1:
            if training is not None:
                plt.plot(training[0], training[1], 'kx', mew=2)
            line, = plt.plot(xs, mean, lw=2)
            _ = plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color=line.get_color(), alpha=0.2)
        # 2 dimensional input
        elif xs.shape[1] == 2:
            print("Grid plotting to be implemented")
            # TODO
            pass
        else:
            print("Cannot plot > 2 dimensional x's")
            return

"""
So SciPy runs out of memory very quickly even when cutting down to 12k examples for 0's and 1's and only 10 features instead of 128...
"""
class Scipy_GP(GP):
    def __init__(self, xs_train, ys_train, limit_classes = 2, limit_features = 10):
        super(Scipy_GP, self).__init__(xs_train, ys_train, limit_classes, limit_features)

        import sklearn.gaussian_process as gp   # Use sklearn's GP
        self.gp = gp


    def train(self):
        print("Beginning training using SciPy GP")
        nu = 1.0
        sigma = 1.0
        ls = [np.exp(-1) for i in range(self.ys_train.shape[-1])]
        kernel = nu**2 * self.gp.kernels.RBF(length_scale=ls) + self.gp.kernels.WhiteKernel(noise_level=sigma)
        model = self.gp.GaussianProcessRegressor(kernel=kernel) 
        model.fit(self.xs_train, self.ys_train)

    def predict(self, xs):
        print("No point implementing scipy GP predict since can't train it!")
        pass


class GPFlow_GP(GP):
    def __init__(self, xs_train, ys_train, limit_classes = 2, limit_features = 2):
        super(GPFlow_GP, self).__init__(xs_train, ys_train, limit_classes, limit_features)
        
        import gpflow as gpflow
        self.gpflow = gpflow


# borrow one of these that supports batching
# m3 = gpflow.models.SVGP(X, Y, gpflow.kernels.RBF(1),
#                       likelihood=gpflow.likelihoods.Gaussian(),
#                       Z=X.copy(), q_diag=False)
# m3.feature.set_trainable(False)
# m4 = gpflow.models.SVGP(X, Y, gpflow.kernels.RBF(1),
#                       likelihood=gpflow.likelihoods.Gaussian(),
#                       Z=X.copy(), q_diag=False, whiten=True)
# m4.feature.set_trainable(False)
# m5 = gpflow.models.SGPR(X, Y, gpflow.kernels.RBF(1), Z=X.copy())
# m5.feature.set_trainable(False)
# m6 = gpflow.models.GPRFITC(X, Y, gpflow.kernels.RBF(1), Z=X.copy())
# m6.feature.set_trainable(False)
# models = [m1, m2, m3, m4, m5, m6]


    def do_configure(self):
        print("Configuring GPFlow regression")
        gpflow = self.gpflow
        self.kernel = gpflow.kernels.RBF(input_dim=self.limit_features)
        self.kernel += gpflow.kernels.White(input_dim=self.limit_features)
        # self.model = gpflow.models.GPR(self.xs_train, self.ys_train, kern=self.kernel)
        
        # SVGP has batching capabilities
        # and is a regressor
        self.model = gpflow.models.SVGP(self.xs_train, self.ys_train, 
                                         self.kernel, likelihood=gpflow.likelihoods.Gaussian(),
                                         Z=self.xs_train.copy(),
                                         minibatch_size=100)
        self.model.X.minibatch_size=100
        self.model.Y.minibatch_size=100

        self.optimiser = gpflow.train.ScipyOptimizer()

    def print_configuration(self):
        print("----- Kernel -----")
        print(self.kernel.as_pandas_table())

        print("----- Optimiser -----")
        print(self.optimiser)



    def train(self):
        print("Beginning training")
        self.optimiser.minimize(self.model)
        print("Finished training")

    def predict(self, xs, plot=True):


        self.mean, self.var = self.model.predict_y(xs)
    
        if plot:
            #superclass impl
            self.plot(xs, self.mean, self.var, (self.xs_train, self.ys_train))
    

    



if __name__ == "__main__":
    xs_train = np.genfromtxt("../data/mnist_train_features.csv", delimiter=",")
    #nn_train = np.genfromtxt("../data/mnist_train_cnn_output.csv", delimiter=",")
    xs_test = np.genfromtxt("../data/mnist_test_features.csv", delimiter=",")
    #nn_test = np.genfromtxt("../data/mnist_test_cnn_output.csv",  delimiter=",")

    ys_train, ys_test = get_mnist_classes()

    gp = GPFlow_GP(xs_train, ys_train)
    gp.do_configure()
    gp.print_configuration()
    gp.train()
    gp.predict(gp._filter(xs_test, ys_test)[0])
