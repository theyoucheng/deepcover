
import numpy as np
#from intervalt import *
import sys

def __MAX__(x, y):
  if x>y: return x
  else: return y

class NNett:
  def __init__(self, _weights, _biases):
    self.weights=_weights
    self.biases=_biases
    self.weights.append([]) # output layer has empty weight vector
    self.biases=[[]]+self.biases # input layer has empty bias vector

  def eval(self, X):
    # act[i][j] is the activation value (after ReLU) of the j-th neuron at the i-th layer
    act=[]
    act.append(X) # X is the input vector to be evaluated

    N=len(self.weights) # N is the #layers

    for i in range(1, N):
      act.append([])
      M=len(self.weights[i-1][0]) # M is the #neurons at layer (i+1)
      # to compute the activation value for each neuron at layer i
      for j in range(0, M):
        val=0 # the activation value is the weighted sum of input from previous layer, plus the bias
        for k in range(0, len(self.weights[i-1])):
          val+=__MAX__(act[i-1][k],0) * self.weights[i-1][k][j]
        val+=self.biases[i][j]
        #if i<N-1 and val<=0: # ReLU
        #  val=0
        act[i].append(val)

    label=np.argmax(act[N-1])
    return label, act

  ## X are interval inputs
  #def intervalt_eval(self, X):
  #  # act[i][j] is the activation value (after ReLU) of the j-th neuron at the i-th layer
  #  act=[]
  #  act.append(X) # X is the input vector to be evaluated

  #  N=len(self.weights) # N is the #layers

  #  for i in range(1, N):
  #    act.append([])
  #    M=len(self.weights[i-1][0]) # M is the #neurons at layer (i+1)
  #    # to compute the activation value for each neuron at layer i
  #    for j in range(0, M):
  #      val=intervalt(0) # the activation value is the weighted sum of input from previous layer, plus the bias
  #      for k in range(0, len(self.weights[i-1])):
  #        #val+=act[i-1][k] * self.weights[i-1][k][j]
  #        val=intervalt_add(val, intervalt_times_const(act[i-1][k], self.weights[i-1][k][j]))
  #      #val+=self.biases[i][j]
  #      val=intervalt_add_const(val, self.biases[i][j])
  #      if i<N-1: # ReLU
  #        val=intervalt_relu(val)
  #      act[i].append(val)

  #  #label=np.argmax(act[N-1])
  #  #return label, act
  #  return act


#### convolutional neural networks (cnn)
#
#class Layert:
#  def __init__(self, _w, _b, _is_conv=False, _mp_size=0):
#    self.w=_w
#    self.b=_b
#    self.is_conv=_is_conv
#    self.mp_size_x=_mp_size
#    self.mp_size_y=_mp_size
#
#class CNNett:
#  def __init__(self, _hidden_layers):
#    self.hidden_layers=_hidden_layers ## hidden layers
#
#  def eval(self, X):
#    ### X shall be an array ==> 28x28
#    #X=np.reshape(X, (28, 28)) 
#    X=X.reshape(28, 28)
#
#    N=len(self.hidden_layers)+1
#
#    ## the final activation vector
#    ## act shall be a vector of 'arrays'
#    act=[] 
#
#    act.append(np.array([X])) ## input layer
#    index=0
#
#    ## to propagate through each hidden layer
#    for layer in self.hidden_layers:
#      print 'We are at layer {0}'.format(index+1)
#      if layer.is_conv: ## is convolutional layer
#        nf=len(layer.w) ## number of filters
#        print '** number of filters: {0}'.format(nf)
#        conv_act=[] ## conv_act contains these filtered activations
#        ## to apply each filter
#        for i in range(0, nf):
#          _nf=len(act[index]) ## number of filter from the preceding layer
#          #print '**** number of preceding filters: {0}'.format(_nf)
#          #acts_for_mp=[]
#          ## there may be multiple filtered pieces from last layer
#          nr=act[index][0].shape[0] # number of rows
#          nc=act[index][0].shape[1] # number of columns
#          nfr=layer.w[i][0].shape[0] # number of filter rows
#          nfc=layer.w[i][0].shape[1] # number of filter columns
#          f_act=np.zeros((nr-nfr+1,nc-nfc+1))
#          for J in range(0, f_act.shape[0]):
#            for K in range(0, f_act.shape[1]):
#              
#              for j in range(0, _nf):
#                ## act[index][j] is the input
#                a=act[index][j]
#            
#                for l in range(0, nfr):
#                  for m in range(0, nfc):
#                    f_act[J][K]+=layer.w[i][j][l][m]*a[J+l][K+m]
#              f_act[J][K]+=layer.b[i]
#              if f_act[J][K]<0: f_act[J][K]=0
#              
#          #########
#          #acts_for_mp.append(np.array(f_act))
#
#          ### max-pool  
#          nr=f_act.shape[0]
#          nc=f_act.shape[1]
#          #### shape after max-pooling
#          p_act=np.zeros((nr/layer.mp_size_x, nc/layer.mp_size_y))
#          for I in range(0, p_act.shape[0]):
#            for J in range(0, p_act.shape[1]):
#              ##########
#                for ii in range(layer.mp_size_x*I, layer.mp_size_x*(I+1)):
#                  for jj in range(layer.mp_size_y*J, layer.mp_size_y*(J+1)):
#                    if f_act[ii][jj]> p_act[I][J]: p_act[I][J]=f_act[ii][jj]
#          conv_act.append(np.array(p_act))
#        #conv_act=np.array(conv_act) ## ==> array
#        act.append(np.array(conv_act))
#      else: ## fully connected layer
#        a=act[index] # the preceeding layer
#        print '*** shape: {0}'.format(a.shape)
#        print '*** w shape: {0}'.format(layer.w.shape)
#        nr=layer.w.shape[0]
#        nc=layer.w.shape[1]
#        ### reshape
#        aa=a.reshape(1, nr)
#
#        this_act=np.zeros((1,nc))
#        for I in range(0, nc):
#          for II in range(0, nr):
#            this_act[0][I]+=aa[0][II]*layer.w[II][I]
#          this_act[0][I]+=layer.b[I]
#          if index < N-2 and this_act[0][I]<0: this_act[0][I]=0
#        act.append(np.array(this_act))
#      ### next layer
#      index+=1
# 
#    label=np.argmax(act[index][0])
#    print act[index][0]
#    print 'label is {0}'.format(label)
#    return label, act
