import numpy as np
from utils import *

def spectra_sym_gen(eobj, x, y, adv_value=1, testgen_factor=.2, testgen_size=0):

  
  v_type=type(adv_value)
  model=eobj.model
  failing=[]
  passing=[]

  #inputs=[]
  sp=x.shape
  x_flag=np.zeros(sp, dtype=bool)
  portion=int(sp[0]*testgen_factor)
  incr=1/6*portion
  if portion<1: portion=1
  L0=np.array(np.arange(x.size))
  L0=np.reshape(L0, sp)
  
  while (not np.all(x_flag)) or len(passing)+len(failing)<testgen_size:
    #print ('####', len(passing), len(failing))
    t=x.copy()

    i0=np.random.randint(0,sp[0])
    i1=np.random.randint(0,sp[1])

    h=portion 
    region=L0[ np.max([i0-h,0]) : np.min([i0+h, sp[0]]), np.max([i1-h,0]):np.min([i1+h,sp[1]])].flatten()

    L=region #L0[0:portion]
    if v_type==np.ndarray:
      np.put(t, L, adv_value.take(L))
    else:
      np.put(t, L, adv_value)
    x_flag.flat[L]=True #np.put(x, L, True)
    new_y=np.argsort(model.predict(sbfl_preprocess(eobj, np.array([t]))))[0][-eobj.top_classes:]
    is_adv=(len(np.intersect1d(y, new_y))==0)

    if is_adv:
      failing.append(t)
      ## to find a passing
      ite=h #testgen_factor
      while ite>1: #ite>0.01:
        t2=x.copy()
        #ite=ite-1#ite//2 #ite=(ite+0)/2
        ite=int(ite-incr)
        if ite<1: break
        region=L0[ np.max([i0-ite,0]) : np.min([i0+ite, sp[0]]), np.max([i1-ite,0]):np.min([i1+ite,sp[1]])].flatten()

        L=region #L0[0:portion]
        if v_type==np.ndarray:
          np.put(t, L, adv_value.take(L))
        else:
          np.put(t, L, adv_value)
        x_flag.flat[L]=True #np.put(x, L, True)
        new_y=np.argsort(model.predict(sbfl_preprocess(eobj, np.array([t]))))[0][-eobj.top_classes:]
        #is_adv=(len(np.intersect1d(y, new_y))==0)
        #ite-=0.01
        #L2=L0[0:int(ite/testgen_factor*portion)]
        #if v_type==np.ndarray:
        #  np.put(t2, L2, adv_value.take(L2))
        #else:
        #  np.put(t2, L2, adv_value)
        #new_y=np.argsort(model.predict(sbfl_preprocess(eobj, np.array([t2]))))[0][-eobj.top_classes:]
        ##print (y, new_y)
        if (len(np.intersect1d(y, new_y))!=0):
          passing.append(t2)
          break

    else:
      passing.append(t)
      ## to find a failing
      ite=h #testgen_factor
      while ite<sp[0]/2: #0.99:
        t2=x.copy()
        #ite=ite+1#ite*2
        ite=int(ite+incr)
        if ite>sp[0]/2: break
        region=L0[ np.max([i0-ite,0]) : np.min([i0+ite, sp[0]]), np.max([i1-ite,0]):np.min([i1+ite,sp[1]])].flatten()

        L=region #L0[0:portion]
        if v_type==np.ndarray:
          np.put(t, L, adv_value.take(L))
        else:
          np.put(t, L, adv_value)
        x_flag.flat[L]=True #np.put(x, L, True)
        new_y=np.argsort(model.predict(sbfl_preprocess(eobj, np.array([t]))))[0][-eobj.top_classes:]
        #t2=x.copy()
        #ite=(ite+1)/2
        ##ite+=0.01
        #L2=L0[0:int(ite/testgen_factor*portion)]
        #if v_type==np.ndarray:
        #  np.put(t2, L2, adv_value.take(L2))
        #else:
        #  np.put(t2, L2, adv_value)
        #new_y=np.argsort(model.predict(sbfl_preprocess(eobj, np.array([t2]))))[0][-eobj.top_classes:]
        if (len(np.intersect1d(y, new_y))==0):
          failing.append(t2)
          x_flag.flat[L]=True
          break

  return np.array(passing), np.array(failing)

def spectra_gen(x, adv_value=1, testgen_factor=0.01, testgen_size=0):

  #print (adv_value, testgen_factor, testgen_size)
  v_type=type(adv_value)

  inputs=[]
  sp=x.shape
  x_flag=np.zeros(sp, dtype=bool)
  portion=int(x.size*testgen_factor) #int(x.size/sp[2]*testgen_factor)
  
  while (not np.all(x_flag)) or len(inputs)<testgen_size:
    t=x.copy()
    L=np.random.choice(x.size, portion)
    if v_type==np.ndarray:
      #t.flat[L]=adv_value.take(L) 
      np.put(t, L, adv_value.take(L))
    else:
      #t.flat[L]=adv_value 
      np.put(t, L, adv_value)
    x_flag.flat[L]=True #np.put(x, L, True)
    #for pos in L:
    #  ipos=np.unravel_index(pos,sp) 
    #  #if v_type==np.ndarray:
    #  #  t.flat[pos]=adv_value.flat[pos]
    #  #else: t.flat[pos]=adv_value
    #  #x_flag.flat[pos]=True #np.put(x, L, True)
    #  for j in range(0, sp[2]):
    #    if v_type==np.ndarray:
    #      t[ipos[0]][ipos[1]][j]=adv_value[ipos[0]][ipos[1]][j]
    #    else:
    #      t[ipos[0]][ipos[1]][j]=adv_value
    #    x_flag[ipos[0]][ipos[1]][j]=True
    inputs.append(t)
  return inputs
