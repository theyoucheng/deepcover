#import matplotlib.pyplot as plt
from keras import *
from keras import backend as K
import numpy as np
from PIL import Image
import copy
import sys, os
import cv2
import csv
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
    self.x_verbosity=None
    self.fnames=[]
    self.boxes=None
    self.occlusion_file=None
    self.min_exp=1.1
    self.straight_part=True

class sbfl_elementt:
  def __init__(self, x, y, xs, ys, model, fname=None, adv_part=None):
    self.x=x
    self.y=y
    self.xs=xs
    self.ys=ys
    self.model=model
    self.fname=fname
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

#returns distance between (x1, y1) and (x2,y2)
def distance(x1,y1,x2,y2):
    temp = (x2-x1)**2 + (y2-y1)**2
    if temp <= 0:
        return 0
    else:
        return math.sqrt(temp)

#returns angle from a to c around b counterclockwise
def get_angle(ax,ay,bx,by,cx,cy):
    angle = math.degrees(math.atan2(cy-by,cx-bx) - math.atan2(ay-by,ax-bx))
    return angle+360 if angle < 0 else angle
  
#using Bretschneider's formula
def get_area(x1, y1, x2, y2, x3, y3, x4, y4):
      a = distance(x1, y1, x2, y2)
      b = distance(x1, y1, x4, y4)
      c = distance(x4, y4, x3, y3)
      d = distance(x2, y2, x3, y3)
      theta1 = get_angle(x4, y4, x1, y1, x2, y2)
      theta2 = get_angle(x4, y4, x3, y3, x2, y2)
      s = (a+b+c+d)/2
      theta = theta1 + theta2
      midstep = ((s-a)*(s-b)*(s-c)*(s-d))-(a*b*c*d*(math.cos(theta/2)**2))
      if midstep <= 1:
          return 1
      else:
          return (math.sqrt(midstep))

#return point of intersection (as an integer) of two line segments l1 and l2, where l1 is 
# defined by distinct points (x1,y1) and (x2,y2) and l2 is defined by distinct points (x3,y3) and (x4,y4)
def get_intersection(x1,y1,x2,y2,x3,y3,x4,y4):
    denom = ((x1-x2)*(y3-y4))-((y1-y2)*(x3-x4))
    if denom == 0:
        return (0,0)
    return (int(((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/denom),int(((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/denom))

#used in in_shape
def in_shape_sign(x1, y1, x2, y2, x3, y3):
    return((x1-x3)*(y2-y3) - (x2-x3)*(y1-y3))

#returns True if a point (x,y) is in a shape
def in_shape(x, y, x1, y1, x2, y2, x3, y3, x4, y4):
    d1 = in_shape_sign(x, y, x1, y1, x2, y2)
    d2 = in_shape_sign(x, y, x2, y2, x3, y3)
    d3 = in_shape_sign(x, y, x3, y3, x4, y4)
    d4 = in_shape_sign(x, y, x4, y4, x1, y1)
    if (((d1 >= 0) and (d2 >=0) and (d3 >= 0) and (d4 >= 0)) or ((d1 <= 0) and (d2 <=0) and (d3 <= 0) and (d4 <= 0))):
        return True
    else:
        return False

#fills a (potentially non-rectangular) mao with a given score
def fill_map(heatMap, score, x1, y1, x2, y2, x3, y3, x4, y4):
    x_min = min(x1,x2,x3,x4)
    x_max = max(x1,x2,x3,x4)
    y_min = min(y1,y2,y3,y4)
    y_max = max(y1,y2,y3,y4)
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            if in_shape(x, y, x1, y1, x2, y2, x3, y3, x4, y4):
                heatMap[x,y,:] = score
    return heatMap

#copies the value of heatMap2 into heatMap1 over a given area
def copy_map(heatMap1, heatMap2, x1, y1, x2, y2, x3, y3, x4, y4):
    x_min = min(x1,x2,x3,x4)
    x_max = max(x1,x2,x3,x4)
    y_min = min(y1,y2,y3,y4)
    y_max = max(y1,y2,y3,y4)
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            if in_shape(x, y, x1, y1, x2, y2, x3, y3, x4, y4):
                heatMap1[x,y,:] = heatMap2[x,y,:]
    return heatMap1

#counts the number of changed pixels from the original to the masked image
#returned as a percentage
def mask_count(image, original):
    count = 0
    x, y = int(image.shape[0]), int(image.shape[1])
    for i in range(x):
        for j in range(y):
            if not(np.array_equal(image[i, j, :],original[i, j, :])):
                count += 1
    return round((count*100/(x*y)),2)

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

def top_plot(sbfl_element, ind, di, metric='', eobj=None, top_excluding=0., bg=128, online=False, online_mark=[255,0,255], timer=0, res_path = '', index = 0, iterations=0, version=False):
  if version:
      tag = 'irreg'
  else:
      tag = ''

  origin_data=sbfl_element.x
  sp=origin_data.shape

  try:
    #print ('mkdir -p {0}'.format(di))
    os.system('mkdir -p {0}'.format(di))
  except: pass

  save_an_image(origin_data, 'origin-{0}'.format(sbfl_element.y), di)

  ret=None

  im_flag=np.zeros(sp, dtype=bool)
  im_o=np.multiply(np.ones(sp), bg)
  count=0
  base=int((ind.size/sp[2])/100)
  pos=ind.size-1
  found_exp = False
  while pos>=0:
    ipos=np.unravel_index(ind[pos], sp)
    if not im_flag[ipos]:
      count+=1
      for k in range(0,sp[2]):
        im_flag[ipos[0]][ipos[1]][k]=True
      # to exclude top pixels for the explanation
      if (count/base)/100. <= top_excluding: 
          pos-=1
          continue
      for k in range(0,sp[2]):
        im_o[ipos[0]][ipos[1]][k]=origin_data[ipos[0]][ipos[1]][k]
        #im_flag[ipos[0]][ipos[1]][k]=True
      if count%base==0:
        save_an_image(im_o, '{1}-{0}'.format(int(count/base), metric), di)
        res=sbfl_element.model.predict(sbfl_preprocess(eobj, np.array([im_o])))
        y=np.argsort(res)[0][-eobj.top_classes:]
        #print (int(count/base), '>>>', y, sbfl_element.y, y==sbfl_element.y)
        if y==sbfl_element.y and not found_exp: 
          with open(os.path.join(res_path, "results"+tag+'.csv'), 'a', newline='\n') as csvfile:
              csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
              csvwriter.writerow([index, iterations, timer, mask_count(im_o, origin_data)])
              csvfile.write('\n')
              csvfile.close()
          save_an_image(im_o, 'explanation-found-{1}-{0}'.format(int(count/base), metric)+tag+str(index)+'-iter'+str(iterations), di)
          found_exp = True
          if not eobj.boxes is None: # wsol calculation
              vect=eobj.boxes[sbfl_element.fname.split('/')[-1]]
              ref_flag=np.zeros(sp, dtype=bool)
              ref_flag[vect[0]:vect[2], vect[1]:vect[3], :]=True

              union=np.logical_or(im_flag, ref_flag)
              inter=np.logical_and(im_flag, ref_flag)
              iou=np.count_nonzero(inter)*1./np.count_nonzero(union)
              ret=iou
          elif not eobj.occlusion_file is None: # occlusion calculation
                ref_flag=np.zeros(sp, dtype=bool)
                for i in range(0, sp[0]):
                    for j in range(0, sp[1]):
                        if origin_data[i][j][0] == 0 and origin_data[i][j][1] == 0 and origin_data[i][j][2] == 0:
                            ref_flag[i][j][:] = True

                union=np.logical_or(im_flag, ref_flag)
                inter=np.logical_and(im_flag, ref_flag)
                iou=np.count_nonzero(inter)*1./np.count_nonzero(union)
                intersection=np.count_nonzero(inter)*1./np.count_nonzero(ref_flag)
                ret=[(count/base)/100., intersection, iou]

          if eobj.x_verbosity>0: return ret

    pos-=1
  return ret

