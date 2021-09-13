
import numpy as np
from utils import *
from sbfl import *

def to_attack(eobj, ind, origin_data, bg_v, init_step, step_incr):

  v_type=type(bg_v)

  top_classes=eobj.top_classes
  sp=origin_data.shape

  x=origin_data
  model=eobj.model
  y=np.argsort(model.predict(sbfl_preprocess(eobj,np.array([x]))))[0][-top_classes:]
  
  latest_step=ind.size

  #im=np.ones(sp)
  #im=np.multiply(im, bg_v)
  im=x.copy()
  im_flag=np.zeros(im.shape, dtype=bool)

  pos=ind.size-1
  old_count=1
  count=1

  adv_v=0

  while pos>=0:

    ipos=np.unravel_index(ind[pos], sp)
    if not im_flag[ipos]:
      for k in range(0,sp[2]):
          if type(bg_v)==np.ndarray:
            im[ipos[0]][ipos[1]][k]=bg_v[ipos[0]][ipos[1]][k]
          else:
            im[ipos[0]][ipos[1]][k]=bg_v
          im_flag[ipos[0]][ipos[1]][k]=True
      count+=1

    pos-=1

    if count<init_step: continue ## to start from a partial image

    if count>5000: break

    if count-old_count>=step_incr:
      old_count=count
      
      adv_v=model.predict(sbfl_preprocess(eobj, np.array([im])))
      adv_y=np.argsort(adv_v)[0][-top_classes:]
      if len(np.intersect1d(y, adv_y))==0:
        #if np.sort(adv_v)[0][-top_classes:][0]>.5:
          return im, count, np.sort(adv_v)[0][-top_classes:]

  return x, x.size//sp[2], [None]

