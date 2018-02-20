from sklearn import svm
import pickle
import os

class SVM_MNIST(object):
    def __init__(self, xs_train, ys_train, nb_classes, name="SVM classifier", save_dir='../models/svm/', retrain=False):
        self.name = name

        save_name = 'svm_' + self.name + '_model.pkl'
        save_path = save_dir + save_name

        if retrain or save_name not in os.listdir(save_dir):
            print("Training SVM")
            self.model = svm.SVC(decision_function_shape='ovo', probability=True)
            if nb_classes == ys_train.shape[-1]:
                ys_train = np.argmax(ys_train_full, axis=1) # no need for 1-hot encoding
            self.model.fit(xs_train, ys_train)
            print("Finished Training SVM")
            print("Saving pickle to", save_path)
            with open(save_path, 'wb') as f:
                pickle.dump(self.model, f)    

        else:
            print("Loading SVM from", save_path)
            with open(save_path, 'rb') as f:
                self.model = pickle.load(f)

    def predict_batch(self, features_batch):
        return self.model.predict(features_batch)
