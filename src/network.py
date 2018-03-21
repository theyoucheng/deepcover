
from __future__ import division, print_function, absolute_import

import tensorflow as tf


'''
  In DeepCover, a neural network is characterised by parameters of each its layer.
  At each layer, let us say k, we have 
  * wk: the weight variable
  * bk: the bias variable
  * mpsk (optional): the maxpool size
  * others: for now, we fix the strides for filters (i.e., 1) and maxpool (i.e., mpsk)
            and the padding is set as 'VALID'
  Example,
  params={
    'w1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    'w2': tf.Variable(tf.random_normal([5, 5, 32, 64])),    
    'b1': tf.Variable(tf.random_normal([32])),
    'b2': tf.Variable(tf.random_normal([64])),
    'mps1': tf.constant(2)
    ...
  }
'''
