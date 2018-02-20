import os
import gpflow
import numpy as np
#import matplotlib.pyplot as plt
#plt.style.use('ggplot')

#import matplotlib as mpl
#mpl.rcParams['pgf.rcfonts'] = False

from mnist_preprocessing import get_mnist_data


class GP_MNIST_SVGP(object):
    def __init__(self, 
                 nb_feats=128,
                 nb_classes=10, 
                 xs_train=None, 
                 ys_train=None,  
                 kernel=gpflow.kernels.Matern12, 
                 whitevar=0.1,
                 latent_split=20,
                 whiten=True,
                 q_diag=True,
                 minibatch=8000,
                 whitevar_trainable=False,
                 feat_trainable=True,
                 name="GP classifier",
                 save_dir='../models/gp/',
                 retrain=False):
        self.name = name

        if xs_train is None and ys_train is None:
            # hardcoded default
            xs_train = np.genfromtxt("../data/mnist_train_features.csv", delimiter=",")
            ((_, _), (ys_train, _)) = get_mnist_data()


        # don't need 1-hot encoding for GP
        if nb_classes == ys_train.shape[-1]:
            ys_train = np.argmax(ys_train, axis=1)
        
        self.model = gpflow.models.SVGP(
            xs_train, ys_train,
            kern=kernel(input_dim=nb_feats) + gpflow.kernels.White(input_dim=nb_feats, variance=whitevar),
            likelihood=gpflow.likelihoods.MultiClass(nb_classes),
            Z=xs_train[::latent_split].copy(), 
            num_latent=nb_classes, 
            whiten=whiten, 
            q_diag=q_diag,
            minibatch_size=minibatch)

        self.model.kern.white.variance.trainable = whitevar_trainable
        self.model.feature.trainable = feat_trainable

        print("Initialized", name)

        save_name = self.name + "_model.npy"
        save_path = save_dir + save_name
        if retrain or save_name not in os.listdir(save_dir):
            print("Training GP from scratch")
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(self.model)
            print("Saving trained model to", save_path)
            params = self.model.read_trainables()
            np.save(save_path, params)
        else:
            print("Loading GP model from file", save_path)
            params = np.load(save_path).item()
            self.model.assign(params)

    """
    params:
        xs: numpy array of datas
    returns:
        (mu, var)
        mu: shape = (inputs, nb_classes) of class probabilities per input
        var: prediction variance per class per input
    """
    def predict_batch(self, xs):
        return self.model.predict_y(xs)
        
