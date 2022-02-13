
from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications.vgg16 import VGG16
from keras.applications import inception_v3, mobilenet, xception
from keras.models import load_model
import matplotlib.pyplot as plt
import csv

import argparse
import os
import numpy as np

from utils import *
from to_explain import *
from comp_explain import *

def main():
  parser=argparse.ArgumentParser(description='To explain neural network decisions' )
  parser.add_argument(
    '--model', dest='model', default='-1', help='the input neural network model (.h5)')
  parser.add_argument("--inputs", dest="inputs", default="-1",
                    help="the input test data directory", metavar="DIR")
  parser.add_argument("--outputs", dest="outputs", default="outs",
                    help="the outputput test data directory", metavar="DIR")
  parser.add_argument("--measures", dest="measures", default=['tarantula', 'zoltar', 'ochiai', 'wong-ii'],
                    help="the SFL measures (tarantula, zoltar, ochiai, wong-ii)", metavar="" , nargs='+')
  parser.add_argument("--measure", dest="measure", default="None",
                    help="the SFL measure", metavar="MEASURE")
  parser.add_argument("--mnist-dataset", dest="mnist", help="MNIST dataset", action="store_true")
  parser.add_argument("--normalized-input", dest="normalized", help="To normalize the input", action="store_true")
  parser.add_argument("--cifar10-dataset", dest="cifar10", help="CIFAR-10 dataset", action="store_true")
  parser.add_argument("--grayscale", dest="grayscale", help="MNIST dataset", action="store_true")
  parser.add_argument("--vgg16-model", dest='vgg16', help="vgg16 model", action="store_true")
  parser.add_argument("--inception-v3-model", dest='inception_v3', help="inception v3 model", action="store_true")
  parser.add_argument("--xception-model", dest='xception', help="Xception model", action="store_true")
  parser.add_argument("--mobilenet-model", dest='mobilenet', help="mobilenet model", action="store_true")
  parser.add_argument("--attack", dest='attack', help="to atatck", action="store_true")
  parser.add_argument("--text-only", dest='text_only', help="for efficiency", action="store_true")
  parser.add_argument("--input-rows", dest="img_rows", default="224",
                    help="input rows", metavar="INT")
  parser.add_argument("--input-cols", dest="img_cols", default="224",
                    help="input cols", metavar="INT")
  parser.add_argument("--input-channels", dest="img_channels", default="3",
                    help="input channels", metavar="INT")
  parser.add_argument("--x-verbosity", dest="x_verbosity", default="0",
                    help="the verbosity level of explanation output", metavar="INT")
  parser.add_argument("--top-classes", dest="top_classes", default="1",
                    help="check the top-xx classifications", metavar="INT")
  parser.add_argument("--adversarial-ub", dest="adv_ub", default="1.",
                    help="upper bound on the adversarial percentage (0, 1]", metavar="FLOAT")
  parser.add_argument("--adversarial-lb", dest="adv_lb", default="0.",
                    help="lower bound on the adversarial percentage (0, 1]", metavar="FLOAT")
  parser.add_argument("--masking-value", dest="adv_value", default="234",
                    help="masking value for input mutation", metavar="INT")
  parser.add_argument("--testgen-factor", dest="testgen_factor", default="0.2",
                    help="test generation factor (0, 1]", metavar="FLOAT")
  parser.add_argument("--testgen-size", dest="testgen_size", default="2000",
                    help="testgen size ", metavar="INT")
  parser.add_argument("--testgen-iterations", dest="testgen_iter", default="1",
                    help="to control the testgen iteration", metavar="INT")
  parser.add_argument("--causal", dest='causal', help="causal explanation", action="store_true")
  parser.add_argument("--wsol", dest='wsol_file', help="weakly supervised object localization", metavar="FILE")
  parser.add_argument("--occlusion", dest='occlusion_file', help="to load the occluded images", metavar="FILE")  
  parser.add_argument("--partition-style", dest='partition_style', default="straight", help="diagonal partitioning or rectangular")


  args=parser.parse_args()

  img_rows, img_cols, img_channels = int(args.img_rows), int(args.img_cols), int(args.img_channels)

  ## some common used datasets
  if args.mnist:
    img_rows, img_cols, img_channels = 28, 28, 1
  elif args.cifar10:
    img_rows, img_cols, img_channels = 32, 32, 3
  elif args.inception_v3 or args.xception:
    img_rows, img_cols, img_channels = 299, 299, 3

  ## to load the input DNN model
  if args.model!='-1':
    dnn=load_model(args.model)
  elif args.vgg16:
    print ('to load VGG16')
    dnn=VGG16()
    print ('done')
  elif args.mobilenet:
    dnn=mobilenet.MobileNet()
  elif args.inception_v3:
    dnn=inception_v3.InceptionV3()
  elif args.xception:
    dnn=xception.Xception()
  else:
    raise Exception ('A DNN model needs to be provided...')
    
   
  if args.partition_style == 'straight':
      straight_part = True
  elif args.partition_style == 'diagonal':
      straight_part = False
  else:
      raise Exception ('Only straight or diagonal partitions possible')

  ## to load the input data
  fnames=[]
  xs=[]
  if args.inputs!='-1':
    for path, subdirs, files in os.walk(args.inputs):
      for name in files:
        fname=(os.path.join(path, name))
        if fname.endswith('.jpg') or fname.endswith('.png') or fname.endswith('.JPEG'):
            if args.grayscale is True or args.mnist:
              x=image.load_img(fname, target_size=(img_rows, img_cols), color_mode = "grayscale")
              x=np.expand_dims(x,axis=2)
            else: 
              x=image.load_img(fname, target_size=(img_rows, img_cols))
            x=np.expand_dims(x,axis=0)
            xs.append(x)
            fnames.append(fname)
  else:
    raise Exception ('What do you want me to do?')
  xs=np.vstack(xs)
  xs = xs.reshape(xs.shape[0], img_rows, img_cols, img_channels)
  print ('\n[Total data loaded: {0}]'.format(len(xs)))

  eobj=explain_objectt(dnn, xs)
  eobj.outputs=args.outputs
  eobj.top_classes=int(args.top_classes)
  eobj.adv_ub=float(args.adv_ub)
  eobj.adv_lb=float(args.adv_lb)
  eobj.adv_value=float(args.adv_value)
  eobj.testgen_factor=float(args.testgen_factor)
  eobj.testgen_size=int(args.testgen_size)
  eobj.testgen_iter=int(args.testgen_iter)
  eobj.vgg16=args.vgg16
  eobj.mnist=args.mnist
  eobj.cifar10=args.cifar10
  eobj.inception_v3=args.inception_v3
  eobj.xception=args.xception
  eobj.mobilenet=args.mobilenet
  eobj.attack=args.attack
  eobj.text_only=args.text_only
  eobj.normalized=args.normalized
  eobj.x_verbosity=int(args.x_verbosity)
  eobj.fnames=fnames
  eobj.occlusion_file=args.occlusion_file
  eobj.straight_part = straight_part
  measures = []
  if not args.measure=='None':
      measures.append(args.measure)
  else: measures = args.measures
  eobj.measures=measures

  if not args.wsol_file is None:
      print (args.wsol_file)
      boxes={}
      with open(args.wsol_file, 'r') as csvfile:
        res=csv.reader(csvfile, delimiter=' ')
        for row in res:
          boxes[row[0]]=[int(row[1]), int(row[2]), int(row[3]), int(row[4])]
      eobj.boxes=boxes


  if args.causal:
      comp_explain(eobj)
  else: to_explain(eobj)

if __name__=="__main__":
  main()

