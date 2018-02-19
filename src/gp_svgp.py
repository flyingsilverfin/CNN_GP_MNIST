import os
import gpflow
import numpy as np
#import matplotlib.pyplot as plt
#plt.style.use('ggplot')

#import matplotlib as mpl
#mpl.rcParams['pgf.rcfonts'] = False

from mnist_preprocessing import get_mnist_data

os.environ["CUDA_VISIBLE_DEVICES"] = ''


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
                 name="GP classifier"
                 ):
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

        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(self.model)


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
        