import sys
sys.path.insert(0, '../../src/')
import random
import numpy as np
import json
import os
from datetime import datetime

from cnnett import Layert
from cnnett import CNNett
from conv_lp import *
from util import *

def ssc_pair_conv(cnnet, current_layer, current_filter, current_I, current_J, prior_filter, prior_I, prior_J, test_data, di, prior_fs):

  index=-1
  tot=len(test_data[0].eval())

  ordering=list(range(tot))
  np.random.shuffle(ordering)

  cex=False

  while index<tot-1:

    index+=1

    X=test_data[0][ordering[index]].eval()
    label=test_data[1][ordering[index]].eval()


    label_, act, act2=cnnet.eval(list(X))

    feasible, new_x, d=conv_ss(current_layer-1, prior_filter, prior_I, prior_J, current_filter, current_I, current_J, cnnet, X, act2, act, prior_fs)

    if feasible:
      label__, act, act2=cnnet.eval(list(new_x))
      if label_!=label__:
        if label_==label or label__==label:
          cex=True

      return True, index, cex, d, label, label_, label__

    if index>=40: break ## 

  return False, index, cex, -1, -1, -1, -1

def main():
  outs="ss-pairs"+str(datetime.now()).replace(' ','-')+'/'
  os.system('mkdir -p {0}'.format(outs))
  training_data, validation_data, test_data = mnist_load_data_shared(filename="../data/mnist.pkl.gz")

  fname='cnn1'

  ws=np.load('cnns/cnn1-weights-mnist.npy')
  bs=np.load('cnns/cnn1-biases-mnist.npy')
  
  layer1=Layert(ws[0], bs[0], True, 2)
  layer2=Layert(ws[1], bs[1], True, 2)
  layer3=Layert(ws[2], bs[2])
  layer4=Layert(ws[3], bs[3])
  
  cnnet=CNNett([layer1, layer2, layer3, layer4])


  outs_=outs+fname+"/" 
  if not os.path.exists(outs_):
    os.system('mkdir -p {0}'.format(outs_))

  s='Neural net tested: {0}\n'.format(fname) 
  fres=fname+'-results.txt'
  f=open(outs_+fres, "a")
  f.write(s)
  f.close()

  ## to simplify things, let's have compute an act here
  X=test_data[0][0].eval()
  ## act2 is before max-pooling
  _, act, act2=cnnet.eval(list(X))

  ncex=0
  covered=0
  not_covered=0

  N=len(act)
  for current_layer in range(2,N): 
    for current_filter in range(0, len(act2[current_layer])):
      a=act2[current_layer][current_filter]
      for current_I in range(0, len(a)):
        for current_J in range(0, len(a[current_I])):
          ##### To test (current_layer, current_filter, current_I, current_J)
          prior_mps=set() ## these at prior mp layer that affect current_I,current_J
          nfr=cnnet.hidden_layers[current_layer-1].w[current_filter][0].shape[0] # number of filter rows
          nfc=cnnet.hidden_layers[current_layer-1].w[current_filter][0].shape[1] # number of filter columns
          for l in range(0, nfr):
            for m in range(0, nfc):
              prior_mps.add((current_I+nfr-m-1, current_J+nfc-l-1))

          prior_fs=set() ### these at prior filters that affect the current_I,current_J
          for x in prior_mps:
            for ii in range(cnnet.hidden_layers[current_layer-2].mp_size_x*x[0], cnnet.hidden_layers[current_layer-2].mp_size_x*(x[0]+1)):
              for jj in range(cnnet.hidden_layers[current_layer-2].mp_size_y*x[1], cnnet.hidden_layers[current_layer-2].mp_size_y*(x[1]+1)):
                prior_fs.add((ii,jj))

          for prior_filter in range(0, len(act2[current_layer-1])):
            #
            b=act2[current_layer-1][prior_filter]
            for prior_I in range(0, len(b)):
              for prior_J in range(0, len(b[prior_I])): ###(prior_layer, prior_filter)
                if not ((prior_I, prior_J) in prior_fs):
                  continue
                found, tested, cex, d, label, label_, label__=ssc_pair_conv(cnnet, current_layer, current_filter, current_I, current_J, prior_filter, prior_I, prior_J, test_data, outs_, prior_fs)
                if found: covered+=1
                else: not_covered+=1
                if cex: ncex+=1
                s='{0}-{1}-{2}-{3}: {4}-{5}-{6}-{7}, '.format(current_layer-1, prior_filter, prior_I, prior_J, current_layer, current_filter, current_I, current_J)
                s+='{0}, tested images: {1}, cex={9}, ncex={2}, covered={3}, not_covered={4}, d={5}, {6}:{7}-{8}\n'.format(found, tested, ncex, covered, not_covered, d, label, label_, label__, cex)
                f=open(outs_+fres, "a")
                f.write(s)
                f.close()
  f=open('./results-ssc.txt', "a")
  tot_pairs=covered+not_covered;
  s='{0}: ssc-coverage: {1}, CEX\%={2}, #CEX={3}, tot_pairs={4}, covered={5}, not-covered={6}\n'.format(fname, 1.0*covered/tot_pairs, 1.0*ncex/tot_pairs, ncex, tot_pairs, covered, not_covered)
  f.write(s)


if __name__=="__main__":
  main()
