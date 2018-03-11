
import sys
sys.path.insert(0, '../../src/')
import os
from datetime import datetime
import random
import numpy as np
import json

from util import *
from nnett import *

training_data, validation_data, test_data = mnist_load_data_shared()

## DNNs
di='../random-nn2/' 

with open(di+'README.txt') as f:
  lines = f.readlines()
  count=-1
  for line in lines:

    count+=1

    ## read each DNN
    fname=line.split()[0]
    print 'Neuron coverage attack: DNN {0} ... '.format(fname),
    with open(di+'w_'+fname, "r") as infile:
      weights=json.load(infile)
    with open(di+'b_'+fname, "r") as infile:
      biases=json.load(infile)

    nnet=NNett(weights, biases)

    # randomlize the test data
    tot=len(test_data[0].eval())
    ordering=list(range(tot))
    np.random.shuffle(ordering)

    covered=0
    tot=0

    X=test_data[0][ordering[0]].eval()
    _, act_=nnet.eval(X)
    act_b_map=[]
    for i in range(0, len(act_)):
      act_b_map.append([])
      for j in range(0, len(act_[i])):
        act_b_map[i].append(False)

    cex=0
    index=-1
    ## we select 25 images to attack neuron coverage
    while index < 24:
      index+=1
      X=test_data[0][ordering[index]].eval()
      X1=X[:]

      for i in range(0, len(X)):
        if X[i]==0:
          X[i]+=np.random.uniform(0.0,0.1)

      _, act=nnet.eval(X)
      for i in range(0, len(act)):
        for j in range(0, len(act[i])):
          if act[i][j]>0: act_b_map[i][j]=True

    for i in range(0, len(act)):
      for j in range(0, len(act[i])):
        tot+=1
        if act_b_map[i][j]: covered+=1

    print '{0} covered'.format(covered*1.0/tot)

    with open("./Result.txt", "a") as myfile:
      myfile.write('{0}, neuron coverage by 25 inputs: {1} \n'.format(fname, covered*1.0/tot))

