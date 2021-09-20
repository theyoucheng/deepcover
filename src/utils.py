#import matplotlib.pyplot as plt
from keras import *
from keras import backend as K
import numpy as np
from PIL import Image
import copy
import sys, os
import cv2
import matplotlib
import matplotlib.pyplot as plt
from keras.preprocessing.image import save_img
from keras.applications import vgg16
from keras.applications import inception_v3, mobilenet, xception

class explain_objectt:
  def __init__(self, model, inputs):
    self.model=model
    self.inputs=inputs
    self.outputs=None
    self.top_classes=None
    self.adv_ub=None
    self.adv_lb=None
    self.adv_value=None
    self.testgen_factor=None
    self.testgen_size=None
    self.testgen_iter=None
    self.vgg16=None
    self.mnist=None
    self.cifar10=None
    self.inception_v3=None
    self.xception=None
    self.mobilenet=None
    self.attack=None
    self.text_only=None
    self.measures=None
    self.normalized=None
    self.fnames=[]


class sbfl_elementt:
  def __init__(self, x, y, xs, ys, model, adv_part=None):
    self.x=x
    self.y=y
    self.xs=xs
    self.ys=ys
    self.model=model
    self.adv_part=adv_part

# Yield successive n-sized 
# chunks from l. 
def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

def arr_to_str(inp):
  ret=inp[0]
  for i in range(1, len(inp)):
    ret+=' '
    ret+=inp[i]
  return ret

def sbfl_preprocess(eobj, chunk):
  x=chunk.copy()
  if eobj.vgg16 is True:
    x=vgg16.preprocess_input(x)
  elif eobj.inception_v3 is True:
    x=inception_v3.preprocess_input(x)
  elif eobj.xception is True:
    x=xception.preprocess_input(x)
  elif eobj.mobilenet is True:
    x=mobilenet.preprocess_input(x)
  elif eobj.normalized is True:
    x=x/255.
  elif eobj.mnist is True or eobj.cifar10 is True:
    x=x/255.
  return x

def save_an_image(im, title, di='./'):
  if not di.endswith('/'):
    di+='/'
  save_img((di+title+'.jpg'), im)

def top_plot(sbfl_element, ind, di, metric='', eobj=None, bg=128, online=False, online_mark=[255,0,255]):
  origin_data=sbfl_element.x
  sp=origin_data.shape

  try:
    #print ('mkdir -p {0}'.format(di))
    os.system('mkdir -p {0}'.format(di))
  except: pass

  save_an_image(origin_data, 'origin-{0}'.format(sbfl_element.y), di)

  im_flag=np.zeros(sp, dtype=bool)
  im_o=np.multiply(np.ones(sp), bg)
  count=0
  base=int((ind.size/sp[2])/100)
  pos=ind.size-1
  found_exp = False
  while pos>=0:
    ipos=np.unravel_index(ind[pos], sp)
    if not im_flag[ipos]:
      for k in range(0,sp[2]):
        im_o[ipos[0]][ipos[1]][k]=origin_data[ipos[0]][ipos[1]][k]
        im_flag[ipos[0]][ipos[1]][k]=True
      count+=1
      if count%base==0:
        save_an_image(im_o, '{1}-{0}'.format(int(count/base), metric), di)
        res=sbfl_element.model.predict(sbfl_preprocess(eobj, np.array([im_o])))
        y=np.argsort(res)[0][-eobj.top_classes:]
        #print (int(count/base), '>>>', y, sbfl_element.y, y==sbfl_element.y)
        if y==sbfl_element.y and not found_exp: 
          save_an_image(im_o, 'explanation-found-{1}-{0}'.format(int(count/base), metric), di)
          found_exp = True
    pos-=1

