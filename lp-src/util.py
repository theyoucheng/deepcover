
import cPickle
import gzip
import matplotlib.pyplot as plt

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams
from theano.tensor.signal import pool


#### Load the MNIST data
def mnist_load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

def display(x, prediction, label, save=True, title=''):
    x_train = x.reshape(1,784)
    plt.title('Prediction: ' + str(prediction)) 
    plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    if not save:
      plt.title('Input image') 
      plt.show()
      return

    if title=='':
      plt.savefig('tmp.png', dpi=100)
    else:
      plt.savefig(title+'.png', dpi=100)

