import numpy as np
import sys

class Layert:
  def __init__(self, _w, _b, _is_conv=False, _mp_size=0):
    self.w=_w
    self.b=_b
    self.is_conv=_is_conv
    self.mp_size_x=_mp_size
    self.mp_size_y=_mp_size

class CNNett:
  def __init__(self, _hidden_layers):
    self.hidden_layers=_hidden_layers ## hidden layers

  def eval(self, X):
    ### X shall be an array ==> 28x28
    #X=np.reshape(X, (28, 28)) 
    X=np.array(X)
    X=X.reshape(28, 28)

    N=len(self.hidden_layers)+1

    ## the final activation vector
    ## act shall be a vector of 'arrays'
    act=[] 
    act2=[] 

    act.append(np.array([X])) ## input layer
    act2.append(np.array([X])) ## input layer
    index=0

    ## to propagate through each hidden layer
    for layer in self.hidden_layers:
      #print 'We are at layer {0}'.format(index+1)
      if layer.is_conv: ## is convolutional layer
        nf=len(layer.w) ## number of filters
        #print '** number of filters: {0}'.format(nf)
        conv_act=[] ## conv_act contains these filtered activations
        conv_act2=[] ## conv_act contains these filtered activations
        _nf=len(act[index]) ## number of filter from the preceding layer
        #print '**** number of preceding filters: {0}'.format(_nf)
        ## to apply each filter
        for i in range(0, nf):
          #acts_for_mp=[]
          ## there may be multiple filtered pieces from last layer
          nr=act[index][0].shape[0] # number of rows
          nc=act[index][0].shape[1] # number of columns
          nfr=layer.w[i][0].shape[0] # number of filter rows
          nfc=layer.w[i][0].shape[1] # number of filter columns
          f_act=np.zeros((nr-nfr+1,nc-nfc+1))
          #print 'fact shape: {0}'.format(f_act.shape)
          #for J in range(0, f_act.shape[0]):
          #  for K in range(0, f_act.shape[1]):
          #    
          #    for j in range(0, _nf):
          #      ## act[index][j] is the input
          #      a=act[index][j]
          #  
          #      for l in range(0, nfr):
          #        for m in range(0, nfc):
          #          f_act[J][K]+=layer.w[i][j][l][m]*a[J+l][K+m]
          #    f_act[J][K]+=layer.b[i]
          #    if f_act[J][K]<0: f_act[J][K]=0
          for J in range(0, f_act.shape[0]):
            for K in range(0, f_act.shape[1]):
              
              for j in range(0, _nf):
                ## act[index][j] is the input
                a=act[index][j]

                #print a
                #print '==========='
                #print layer.w[i][j]
            
                for l in range(0, nfr):
                  for m in range(0, nfc):
                    f_act[J][K]+=layer.w[i][j][m][l]*a[J+nfr-m-1][K+nfc-l-1]
              f_act[J][K]+=layer.b[i]
              if f_act[J][K]<0: f_act[J][K]=0
              
          ### max-pool  
          nr=f_act.shape[0]
          nc=f_act.shape[1]
          #### shape after max-pooling
          #p_act=f_act
          p_act=np.zeros((nr/layer.mp_size_x, nc/layer.mp_size_y))
          #print 'pact shape: {0}'.format(p_act.shape)
          for I in range(0, p_act.shape[0]):
            for J in range(0, p_act.shape[1]):
              ##########
                for ii in range(layer.mp_size_x*I, layer.mp_size_x*(I+1)):
                  for jj in range(layer.mp_size_y*J, layer.mp_size_y*(J+1)):
                    if f_act[ii][jj]> p_act[I][J]: p_act[I][J]=f_act[ii][jj]
          ##print p_act 
          ##sys.exit(0)
          conv_act.append(np.array(p_act))
          conv_act2.append(np.array(f_act))
        #conv_act=np.array(conv_act) ## ==> array
        act.append(np.array(conv_act))
        act2.append(np.array(conv_act2))
        #if index==0:
        #  print act[1].shape
        #  print act2[1].shape
        #  sys.exit(0)
      else: ## fully connected layer
        #a=act[index] # the preceeding layer
        nr=layer.w.shape[0]
        nc=layer.w.shape[1]
        ### reshape
        aa=act[index].reshape(1, nr)
        #print '*** shape: {0}'.format(aa.shape)
        #print '*** w shape: {0}'.format(layer.w.shape)

        this_act=np.zeros((1,nc))
        for I in range(0, nc):
          for II in range(0, nr):
            this_act[0][I]+=aa[0][II]*layer.w[II][I]
          this_act[0][I]+=layer.b[I]
          if index < N-2 and this_act[0][I]<0: this_act[0][I]=0
        act.append(np.array([this_act]))
        act2.append(np.array([this_act]))

      ### next layer
      index+=1
 
    label=np.argmax(act[index][0])
    #print act[index][0]
    #print 'label is {0}'.format(label)
    return label, act, act2
