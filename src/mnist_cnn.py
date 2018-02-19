import keras
from keras.datasets import mnist
from keras import backend as K
import tensorflow as tf

import numpy as np

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

# loads a pretrained keras model
# can be used for prediction, feature extraction, adverserial attacks etc

# TODO remove hard coded images sizes

class CNN_MNIST(object):
    def __init__(self, model_path='../models/mnist_cnn.h5', isolate=False):
        self.isolate = isolate
        self.model_path = model_path
        if not isolate:
            # instantiate long-lived model
            self.model = keras.models.load_model(model_path)

    def predict_img_batch(self, images):
        if self.isolate:
            with tf.Graph().as_default():
                with tf.Session().as_default() as sess:
                    K.set_session(sess)
                    model = keras.models.load_model(self.model_path)
                    cnn_probs = model.predict_on_batch(images)
                    return cnn_probs
                    #cnn_predicted_classes = np.argmax(cnn_test_probs, axis=1)
                    #print("Num CNN incorrect:", np.count_nonzero(cnn_predicted_classes != correct_classes))
        else:
            return self.model.predict_on_batch(images)

    def extract_features(self, images, layer=6):
        if self.isolate:
            with tf.Graph().as_default():
                with tf.Session().as_default() as sess:
                    K.set_session(sess)
                    K.set_learning_phase(0) #set learning phase to TEST

                    #make a fresh copy to not modify other graphs
                    model = keras.models.load_model(self.model_path)
                    feature_extractor = K.function([model.layers[0].input, K.learning_phase()],
                                                [model.layers[layer].output])
                    
                    return feature_extractor([images])[0]
        else:
            feature_extractor = K.function([self.model.layers[0].input, K.learning_phase()],
                                           [self.model.layers[layer].output])
            return feature_extractor([images])[0]


    """
    params:
        examples: inputs to the CNN which will be distorted
        epsilon: amount of distortion to apply

    returns (predictions, perturbed_examples)
        predictions: N-Class predictions for each example after perturbations
        perturbed_examples: the examples that have been perturbed (same format as input examples)
    """
    def fgsm_attack(self, examples, epsilon=0.2):
        with tf.Graph().as_default() as graph:
            with tf.Session().as_default() as sess:
                K.set_learning_phase(0) #set learning phase

                #make a fresh copy to not modify other graph
                model = keras.models.load_model(self.model_path)

                x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
                y = tf.placeholder(tf.float32, shape=(None, 10))

                # Initialize the Fast Gradient Sign Method (FGSM) attack object and graph
                wrap = KerasModelWrapper(model)
                fgsm = FastGradientMethod(wrap, sess=sess)
                fgsm_params = {'eps': epsilon,
                               'clip_min': 0.,
                               'clip_max': 1.}
                adv_x = fgsm.generate(x, **fgsm_params)
                # Consider the attack to be constant
                adv_x = tf.stop_gradient(adv_x)
                preds_adv = model(adv_x)

                adv_predictions = sess.run(preds_adv, {x: examples})
                perturbed = sess.run(adv_x, {x: examples})
                
                return (adv_predictions, perturbed)

    
    