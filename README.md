# Combining CNNs and Gaussian Processes

# Running

## Requirements

* Keras + Tensorflow
* h5py, pydot
* GPFlow (submodule included)
  * Install with `pip install .` in `deps/GPflow`

## Preprocessing

The first thing to do is to take the pre-trained MNIST CNN 
And run all our training/test images through it, using the last layer before the softmax classifier
as a 128-element feature vector representing the image

The resulting features, and also CNN outputs are saved under data/
