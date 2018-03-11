
import sys
sys.path.insert(0, '../../../src/')
import random
import numpy as np
import json
import os
import time
from datetime import datetime

from util import *
from nnett import *
from lp import *


def ssc_pair(nnet, I, J, K, test_data, di):

  index=-1
  tot=len(test_data[0].eval())

  ordering=list(range(tot))
  np.random.shuffle(ordering)

  cex=False

  while index<tot-1:

    index+=1

    X=test_data[0][ordering[index]].eval()
    label=test_data[1][ordering[index]].eval()


    label_, act=nnet.eval(list(X))

    times=[]

    start=time.time()
    feasible, new_x, d, s1, s2=rp_ssc(I, J, K, nnet, X, act)
    end=time.time()

    times.append(end-start)

    if feasible:
      label__, act=nnet.eval(list(new_x))
      if label==label_ or label==label__:
        if label_!=label__:
          cex=True

        for i in range(0, 99):
          start=time.time()
          feasible, new_x, d, s1, s2=rp_ssc(I, J, K, nnet, X, act)
          end=time.time()
          times.append(end-start)

        tot_time=0
        for t in times:
          tot_time+=t
        tot_time=1.0*tot_time/len(times)


        f=open(di+'results.txt'.format(label), "a")
        #s='index: {0}\n'.format(index)
        s='#vars: {0}, #constraints: {1}, #time: {2}\n'.format(s1, s2, tot_time) 
        f.write(s)
        f.close()

      return True, index, cex, d, label, label_, label__

    if index>=40: break ## 

  return False, index, cex, -1, -1, -1, -1

def main():
  di='../../random-nn/'
  training_data, validation_data, test_data = mnist_load_data_shared(filename="../../data/mnist.pkl.gz")
  nnindex=-1
  with open(di+'README.txt') as f:
    lines = f.readlines()
    for line in lines:

      nnindex+=1
      if nnindex<7: continue

      fname=line.split()[0]
      with open(di+'w_'+fname, "r") as infile:
        weights=json.load(infile)
      with open(di+'b_'+fname, "r") as infile:
        biases=json.load(infile)

      nnet=NNett(weights, biases)
      N=len(nnet.weights)

      s='Neural net tested: {0}\n'.format(fname) 
      f=open('./results.txt', "a")
      f.write(s)
      f.close()

      ncex=0
      covered=0
      not_covered=0
      i_begin=1
      j_begin=0
      k_begin=0
      flag=False
      for I in range(i_begin, N-1): ## iterate each hidden layer
        M=len(nnet.weights[I-1][0])
        f=open('./results.txt', "a")
        s='L{0}-{1}: '.format(I, I+1)
        f.write(s)
        for J in range(j_begin, M):
          for K in range(k_begin, len(nnet.weights[I][0])):
            flag=True
            found, tested, cex, d, label, label_, label__=ssc_pair(nnet, I, J, K, test_data, './')
            if found: covered+=1
            else:
              not_covered+=1
              flag=False
            if cex: ncex+=1
            #s='I-J-K: {0}-{1}-{2}, '.format(I, J, K)
            #s+='{0}, tested images: {1}, ncex={2}, covered={3}, not_covered={4}, d={5}, {6}:{7}-{8}\n'.format(found, tested, ncex, covered, not_covered, d, label, label_, label__)
            #f=open(outs+'results.txt', "a")
            #f.write(s)
            #f.close()
            if flag: break
          k_begin=0
          if flag: break
        j_begin=0
      #f=open(di+'results.txt', "a")
      #s='{0}: mcdc-coverage: {1}, CEX={2}, covered={3}, not-covered={4}\n'.format(fname, 1.0*covered/(covered+not_covered), ncex, covered, not_covered)
      #f.write(s)


if __name__=="__main__":
  main()
