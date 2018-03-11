import sys
sys.path.insert(0, '../../../src/')
import random
import numpy as np
import json
import os
from datetime import datetime

from util import *
from nnett import *
from lp import *

def svc_pair(nnet, I, J, K, test_data, di):

  index=-1
  tot=len(test_data[0].eval())

  ordering=list(range(tot))
  np.random.shuffle(ordering)

  ade=False

  while index<tot-1:

    index+=1

    X=test_data[0][ordering[index]].eval()
    label=test_data[1][ordering[index]].eval()


    label_, act=nnet.eval(list(X)) ## original label by DNN

    ### the LP rountine
    sfactor=1.0
    feasible, new_x, d=rp_svc(I, J, K, nnet, X, act, 1+sfactor)

    if feasible:
      label__, act=nnet.eval(list(new_x)) ## the next label by DNN

      if label_!=label__:
        if label__==label or label_==label: ade=True

      return True, ade, index, 0, d, label, label_, label__

    if index>=40: break ## 

  return False, False, index, -1, -1, -1, -1, -1

def main():

  di='../../random-nn/'

  outs="./svc-pairs"+str(datetime.now()).replace(' ','-')+'/'
  os.system('mkdir -p {0}'.format(outs))
  training_data, validation_data, test_data = mnist_load_data_shared(filename="../../data/mnist.pkl.gz")
  nnindex=-1
  with open(di+'README.txt') as f:
    lines = f.readlines()
    for line in lines:

      nnindex+=1
      if nnindex<1: continue

      fname=line.split()[0]
      with open(di+'w_'+fname, "r") as infile:
        weights=json.load(infile)
      with open(di+'b_'+fname, "r") as infile:
        biases=json.load(infile)

      nnet=NNett(weights, biases)
      N=len(nnet.weights)

      s='Neural net tested: {0}\n'.format(fname) 
      fres=fname+'-results.txt'
      f=open(outs+fres, "a")
      f.write(s)
      f.close()

      covered=0
      not_covered=0
      i_begin=1
      j_begin=0
      k_begin=0
      nade=0 # number of adversarial examples
      for I in range(i_begin, N-1): ## iterate each hidden layer
        M=len(nnet.weights[I-1][0])
        for J in range(j_begin, M):
          for K in range(k_begin, len(nnet.weights[I][0])):
            found, is_ade, tested, ncex_, d_, label, label_, label__=svc_pair(nnet, I, J, K, test_data, outs)
            if found: covered+=1
            else: not_covered+=1

            if is_ade: nade+=1

            s='I-J-K: {0}-{1}-{2}, '.format(I, J, K)
            s+='{0}, tested images: {1}, nade={2}, d={3}, covered={4}, not_covered={5}, ncex={6}, labels: {7}:{8}-{9}\n'.format(found, tested, nade, d_, covered, not_covered, nade, label, label_, label__)
            f=open(outs+fres, "a")
            f.write(s)
            f.close()
      f=open('./results-svc.txt', "a")
      s='{0}: svc-coverage: {1}, nade={2}, covered={3}, not-covered={4}, CEX={5}\n'.format(fname, 1.0*covered/(covered+not_covered), nade, covered, not_covered, nade)
      f.write(s)


if __name__=="__main__":
  main()
