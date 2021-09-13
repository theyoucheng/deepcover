import numpy as np
from utils import *

def to_rank(sbfl_element, metric='zoltar'):
  origin_data=sbfl_element.x
  sp=origin_data.shape
  ef=np.zeros(sp,dtype=float)
  nf=np.zeros(sp,dtype=float)
  ep=np.zeros(sp,dtype=float)
  np_=np.zeros(sp,dtype=float)

  xs=np.array(sbfl_element.xs)

  diffs=np.abs(xs-origin_data)
  #diffs=diffs - (1+0.05 * origin_data)
  #diffs[diffs>0]=0

  for i in range(0, len(diffs)):
    is_adv=(sbfl_element.y!=sbfl_element.ys[i])
    ds_i1=diffs[i].copy()
    ds_i1[ds_i1>0]=1
    ds_i2=diffs[i].copy()
    ds_i2[ds_i2>0]=-1
    ds_i2[ds_i2==0]=+1
    ds_i2[ds_i2==-1]=0
    if is_adv:
      ef=ef+ds_i1
      nf=nf+ds_i2
      #ef=ef+ds_i2
      #nf=nf+ds_i1
      #for index, _ in np.ndenumerate(diffs[i]):
      #  flag=diffs[i][index]>0
      #  if flag:
      #    ef[index]+=1
      #  else:
      #    nf[index]+=1
    else:
      ep=ep+ds_i1
      np_=np_+ds_i2
      #ep=ep+ds_i2
      #np_=np_+ds_i1
      #for index, _ in np.ndenumerate(diffs[i]):
      #  flag=diffs[i][index]>0
      #  if flag:
      #    ep[index]+=1
      #  else:
      #    np_[index]+=1

  ind=None
  spectrum=None
  if metric=='random':
    spectrum=np.random.rand(sp[0], sp[1], sp[2])
  elif metric=='zoltar':
    zoltar=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      if aef==0:
        zoltar[index]=0
      else:
        k=(10000.0*anf*aep)/aef
        zoltar[index]=(aef*1.0)/(aef+anf+aep+k)
    spectrum=zoltar
  elif metric=='wong-ii':
    wong=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      wong[index]=aef-aep
    spectrum=wong
  elif metric=='ochiai':
    ochiai=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      try:
        ochiai[index]=aef/np.sqrt((aef+anf)*(aef+aep))
      except: ochiai[index]=0
    spectrum=ochiai
  elif metric=='tarantula':
    tarantula=np.zeros(sp, dtype=float)
    for index, x in np.ndenumerate(origin_data):
      aef=ef[index]
      anf=nf[index]
      anp=np_[index]
      aep=ep[index]
      try: tarantula[index]=(aef/(aef+anf))/(aef/(aef+anf)+anp/(aep+anp))
      except: tarantula[index]=0
    spectrum=tarantula
  else:
    raise Exception('The measure is not supported: {0}'.format(metric))

  spectrum_flags=np.zeros(sp, dtype=bool)
  for iindex, _ in np.ndenumerate(spectrum):
    tot=0
    for j in range(0, (sp[2])):
      if not spectrum_flags[iindex[0]][iindex[1]][j]:
        tot+=spectrum[iindex[0]][iindex[1]][j]
    for j in range(0, (sp[2])):
      if not spectrum_flags[iindex[0]][iindex[1]][j]:
        spectrum_flags[iindex[0]][iindex[1]][j]=True
        spectrum[iindex[0]][iindex[1]][j]=tot

  # to smooth
  smooth = np.ones(spectrum.shape)
  sI = spectrum.shape[0]
  sJ = spectrum.shape[1]
  sd = (int)(sI*(10. / 224))
  for si in range(0, spectrum.shape[0]):
      for sj in range(0, spectrum.shape[1]):
          for sk in range(0, spectrum.shape[2]): 
              smooth[si][sj][sk] = np.mean(spectrum[np.max([0, si-sd]):np.min([sI, si+sd]), np.max([0,sj-sd]):np.min([sJ, sj+sd]), sk])
  spectrum = smooth

  ind=np.argsort(spectrum, axis=None)

  return ind, spectrum
