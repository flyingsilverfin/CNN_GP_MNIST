# Combining CNNs and Gaussian Processes

# Running

## Requirements

* Python 3.5+ 
  * On Ubuntu 14.04 use http://ppa.launchpad.net/jonathonf/ PPA for the python 3.6

* Keras + Tensorflow
* tkinter-3 (for matplotlib)
* `pip install h5py pydot pytest matplotlib python3-tk`
* GPFlow (submodule included)
  * Install with `pip install .` in `deps/GPflow`

## Preprocessing

The first thing to do is to take the pre-trained MNIST CNN 
And run all our training/test images through it, using the last layer before the softmax classifier
as a 128-element feature vector representing the image

The resulting features, and also CNN outputs are saved under data/
