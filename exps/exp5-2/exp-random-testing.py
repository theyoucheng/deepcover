import sys
sys.path.insert(0, '../../src/')
import random
import numpy as np
import json
import os
from datetime import datetime

from util import *
from nnett import *

training_data, validation_data, test_data = mnist_load_data_shared()

def mutate(X, seed):
  res=[]
  epsilon=0.1
  for i in range(0, len(X)):
    seed+=1
    np.random.seed(seed+i)
    delta=np.random.uniform(-epsilon, epsilon)
    x=X[i]+delta
    if x>1: x=1
    if x<0: x=0
    res.append(x)
  return res

def main():

  di='../random-nn/'
  seed=1234
  
  count=0
  with open(di+'README.txt') as f:
    count+=1
    index=999 ## let's start from the 1000th image in test data
    lines = f.readlines()
    for line in lines:
  
      index+=1
  
      ## to read the DNN
      fname=line.split()[0]
      with open(di+'w_'+fname, "r") as infile:
        weights=json.load(infile)
      with open(di+'b_'+fname, "r") as infile:
        biases=json.load(infile)
  
      nnet=NNett(weights, biases)
  
      #os.system('mkdir -p {0}'.format(outs))
  
      X=test_data[0][index].eval()
      label=test_data[1][index].eval()
  
      label_, act=nnet.eval(X)

      f=open('./_DNN{0}-README.txt'.format(count), "w")
      s=''
      s+='Neural net tested: {0}\n'.format(fname) 
      s+='Input: MNIST test data index: {0}\n'.format(index) 
      s+='Input: MNIST test data label: {0}\n'.format(label) 
      f.write(s)
      f.close()
      print s,
    
  
      #### run nnet testing
      cex=0
      tests=0
      while tests<10*10000: #for x in f_data:
        x=mutate(X, seed)
        seed+=28*28
        label_, act_=nnet.eval(x)
        if label_!=label: ## adversarial found
          cex+=1
        tests+=1
        if tests%100==0:
          f=open('./_DNN{0}-README.txt'.format(count), "a")
          s='###### counterexamples: {0}, total tests {1}\n'.format(cex, tests)
          f.write(s)
          f.close()
          print s,
        if cex>100: break
        
  
      with open("./Result-random.txt", "a") as myfile:
        myfile.write('{0},  counterexamples {1} \n'.format(fname, cex))

if __name__=="__main__":
  main()
