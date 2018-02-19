from sklearn import svm


class SVM_MNIST(object):
    def __init__(self, xs_train, ys_train, nb_classes):

        self.model = svm.SVC(decision_function_shape='ovo', probability=True)

        if nb_classes == ys_train.shape[-1]:
            ys_train = np.argmax(ys_train_full, axis=1) # no need for 1-hot encoding
        
        self.model.fit(xs_train, ys_train)

    def predict_batch(self, features_batch):
        return self.model.predict(adv_fsgm_features)
