
import argparse
import sys

import keras
from keras.models import *
from keras.layers import * 
from keras import *


def main():
  parser=argparse.ArgumentParser(
          description='DeepCover: Uncover Bugs in Deep Learning' )

  parser.add_argument('model', action='store', nargs='+', help='The input neural network model (.h5)')

  parser.add_argument(
          '--cover', metavar='ss', action='store', help='The covering method: ss, sv, ds, dv', default='ss')

  args=parser.parse_args()

  model = load_model(args.model[0])

  if not (args.cover in ['ss', 'sv', 'ds', 'dv']):
    print ('Covering method cannot be recognized: ' + args.cover)
    sys.exit(0)

  print ('\n== WARNING == \n')
  print (
    'The input model:       ' + args.model[0] + '\n' +
    'The covering method:   ' + args.cover  + '\n'
    )
  print ('This keras compatible implementation of DeepCover testing is currently under deverlopment...\n')
  print ('=============\n')

if __name__=="__main__":
  main()
