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

    feasible, new_x, d, _, _=rp_ssc(I, J, K, nnet, X, act)

    if feasible:
      label__, act=nnet.eval(list(new_x))
      if label_!=label__:
        if label_==label or label__==label:
          cex=True

      return True, index, cex, d, label, label_, label__

    if index>=40: break ## 

  return False, index, cex, -1, -1, -1, -1

def main():
  kappa=10
  di='../../random-nn/'
  outs="./ssc-pairs"+str(datetime.now()).replace(' ','-')+'/'
  os.system('mkdir -p {0}'.format(outs))
  training_data, validation_data, test_data = mnist_load_data_shared(filename="../../data/mnist.pkl.gz")

  nnindex=-1
  with open(di+'README.txt') as f:
    lines = f.readlines()
    for line in lines:

      nnindex+=1

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

      ncex=0
      covered=0
      not_covered=0
      i_begin=2
      j_begin=0
      k_begin=0
      for I in range(i_begin, N): ## iterate each hidden layer
        for K in range(k_begin, len(nnet.weights[I-1][0])):
          ## to find the top-kappa weights to node K
          weights_to_k=[]
          for J in range(0, len(nnet.weights[I-1])):
            weights_to_k.append(abs(nnet.weights[I-1][J][K]))

          top_kappa=[]
          for ka in range(0, kappa):
            _, J=max( (v, i) for i, v in enumerate(weights_to_k) )
            top_kappa.append(J)
            weights_to_k.pop(J)

          for J in top_kappa: #range(j_begin, M):
            found, tested, cex, d, label, label_, label__=ssc_pair(nnet, I-1, J, K, test_data, outs)
            if found: covered+=1
            else: not_covered+=1
            if cex: ncex+=1
            s='I-J-K: {0}-{1}-{2}, '.format(I-1, J, K)
            s+='{0}, tested images: {1}, cex={9}, ncex={2}, covered={3}, not_covered={4}, d={5}, {6}:{7}-{8}\n'.format(found, tested, ncex, covered, not_covered, d, label, label_, label__, cex)
            f=open(outs+fres, "a")
            f.write(s)
            f.close()
          k_begin=0
        j_begin=0
      f=open(di+'results-ssc-kappa{0}.txt'.format(kappa), "a")
      tot_pairs=covered+not_covered;
      s='{0}: aac-coverage: {1}, CEX\%={2}, #CEX={3}, tot_pairs={4}, covered={5}, not-covered={6}\n'.format(fname, 1.0*covered/tot_pairs, 1.0*ncex/tot_pairs, ncex, tot_pairs, covered, not_covered)
      f.write(s)


if __name__=="__main__":
  main()
