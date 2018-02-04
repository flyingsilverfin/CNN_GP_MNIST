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

    def gridParams(self, mins = [-1.0, 1.0], maxs = [-1.0, 1.0], n = 50):
        nGrid = n
        xspaced = np.linspace(mins[0], maxs[0], nGrid)
        yspaced = np.linspace(mins[1], maxs[1], nGrid)
        xx, yy = np.meshgrid(xspaced, yspaced)
        Xplot = np.vstack((xx.flatten(),yy.flatten())).T
        return mins, maxs, xx, yy, Xplot

    # def plot2D(self, plotter, gridparams, ys):
    #     assert ys.shape[1] == 2, "Can only handle 2 classes right now"
    #     col1 = '#0172B2'
    #     col2 = '#CC6600'
    #     mins, maxs, xx, yy, Xplot = gridparams
    #     p = ys #m.predict_y(Xplot)[0]
    #     plotter.plot(self.xs_train[:,0][np.argmax(self.ys_train, axis=0)], 
    #                  self.xs_train[:,1][np.argmax(self.ys_train, axis=0)], 'o', color=col1, mew=0, alpha=0.5)
    #     plotter.plot(self.xs_train[:,0][np.argmin(self.ys_train, axis=0)], 
    #                  self.xs_train[:,1][np.argmin(self.ys_train, axis=0)], 'o', color=col2, mew=0, alpha=0.5)
    #     #if hasattr(m, 'feat') and hasattr(m.feat, 'Z'):
    #     #    Z = m.feature.Z.read_value()
    #     #    ax.plot(Z[:,0], Z[:,1], 'ko', mew=0, ms=4)
    #     #    ax.set_title('m={}'.format(Z.shape[0]))
    #     #else:
    #     #    ax.set_title('full')
    #     #plotter.set_title('Contours')
    #     plotter.contour(xx, yy, p.reshape(*xx.shape), [0.5], colors='k', linewidths=1.8, zorder=100)

    def plot(self, xs, ys, mean, var, training = None):
        # 1 dimensional input
        if xs.shape[0] == 1:
            if ys.shape[0] > 1:
                print("Cannot plot >1 dimensional predictions")
                return
            if training is not None:
                plt.plot(training[0], training[1], 'kx', mew=2)
            line, = plt.plot(xs, mean, lw=2)
            _ = plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color=line.get_color(), alpha=0.2)
        # 2 dimensional input
        elif xs.shape[1] == 2:
           print("Grid plotting currently not supported")
           return
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


class GPFlow_GP_Regression(GPFlow_GP):
    def __init__(self, xs_train, ys_train, limit_classes = 2, limit_features = 2):
        super(GPFlow_GP_Regression, self).__init__(xs_train, ys_train, limit_classes, limit_features)

    def do_configure(self):
        print("Configuring GPFlow regression")
        gpflow = self.gpflow
        self.kernel = gpflow.kernels.RBF(input_dim=self.limit_features)
        self.kernel += gpflow.kernels.White(input_dim=self.limit_features)
        # self.model = gpflow.models.GPR(self.xs_train, self.ys_train, kern=self.kernel)

        # SVGP has batching capabilities
        # and is a regressor

        ys = np.argmax(self.ys_train, axis = 0)
        self.model = gpflow.models.SVGP(self.xs_train, self.ys_train, 
                                        self.kernel, 
                                        likelihood=gpflow.likelihoods.Gaussian(),
                                        Z=self.xs_train[::50],
                                        num_latent = 10,
                                        minibatch_size=100)
        self.model.feature.trainable = False

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

        print(self.mean)
        print(self.var)

class GPFlow_GP_Classification(GPFlow_GP):
    def __init__(self, xs_train, ys_train, limit_classes = 2, limit_features = 2):
        super(GPFlow_GP_Classification, self).__init__(xs_train, ys_train, limit_classes, limit_features)
        plt.figure(figsize=(12,6))
        plt.plot(xs_train, np.argmax(ys_train, axis=1), '.')

    def do_configure(self):
        print("Configuring GPFlow Classification")
        gpflow = self.gpflow
        self.kernel = gpflow.kernels.RBF(input_dim=self.limit_features)
        self.kernel += gpflow.kernels.White(input_dim=self.limit_features, variance=0.01)

        ys = np.argmax(self.ys_train, axis = 1) # convert back from one-hot to integer labeled classes
        self.model = gpflow.models.SVGP(self.xs_train, ys, 
                                        self.kernel, 
                                        likelihood=gpflow.likelihoods.MultiClass(self.limit_classes),
                                        Z=self.xs_train[::50].copy(),
                                        num_latent = self.limit_classes,
					minibatch_size = 10000,
                                        whiten=True,
                                        q_diag=True)
            
        self.model.kern.white.variance.trainable = False
        self.model.feature.trainable = False
        self.model.as_pandas_table()

        print(self.model.kern.rbf)

        self.optimiser = gpflow.train.ScipyOptimizer()

    def print_configuration(self):
        print("----- Kernel -----")
        print(self.model.kern.as_pandas_table())

        print("----- Optimiser -----")
        print(self.optimiser)

    def plot_multiclass_model(self, m):
        f = plt.figure(figsize=(12,6))
        a1 = f.add_axes([0.05, 0.05, 0.9, 0.6])
        a2 = f.add_axes([0.05, 0.7, 0.9, 0.1])
        a3 = f.add_axes([0.05, 0.85, 0.9, 0.1])


        xx = np.linspace(m.X.read_value().min(), m.X.read_value().max(), 200).reshape(-1,1)
        mu, var = m.predict_f(xx)
        mu, var = mu.copy(), var.copy()
        p, _ = m.predict_y(xx)

        a3.set_xticks([])
        a3.set_yticks([])


        a3.set_xticks([])
        a3.set_yticks([])

        #for i in range(m.likelihood.num_classes):
        for i in range(mu.shape[1]):
            x = m.X.read_value()[m.Y.read_value().flatten()==i]
            points, = a3.plot(x, x*0, '.')
            color=points.get_color()
            a1.plot(xx, mu[:,i], color=color, lw=2)
            a1.plot(xx, mu[:,i] + 2*np.sqrt(var[:,i]), '--', color=color)
            a1.plot(xx, mu[:,i] - 2*np.sqrt(var[:,i]), '--', color=color)
            a2.plot(xx, p[:,i], '-', color=color, lw=2)

        a2.set_ylim(-0.1, 1.1)
        a2.set_yticks([0, 1])
        a2.set_xticks([])
        plt.show(block=False)


    def train(self):
        print("Beginning training")
        self.optimiser.minimize(self.model)
        print("Finished training")

    def predict(self, xs, plot=True):
        self.mean, self.var = self.model.predict_y(xs)
    
        if plot:
            #superclass impl
            self.plot_multiclass_model(self.model)
    
        print(self.mean)
        print(self.var)


if __name__ == "__main__":
    xs_train = np.genfromtxt("data/mnist_train_features.csv", delimiter=",")
    #nn_train = np.genfromtxt("../mnist_train_cnn_output.csv", delimiter=",")
    xs_test = np.genfromtxt("data/mnist_test_features.csv", delimiter=",")
    #nn_test = np.genfromtxt("../mnist_test_cnn_output.csv",  delimiter=",")

    ys_train, ys_test = get_mnist_classes()

    gp = GPFlow_GP_Classification(xs_train, ys_train, limit_features = 1, limit_classes = 10)
    gp.do_configure()
    gp.print_configuration()
    gp.train()
    gp.print_configuration()
    gp.predict(gp._filter(xs_test, ys_test)[0])
    input()
