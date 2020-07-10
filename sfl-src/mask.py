import numpy as np

def find_mask(x, p=2/32):
  sp=x.shape
  h=int(sp[0]*p)
  if h<1: h=1
  tmp_x=x.copy()
  bg_x=x.copy()

  for iindex, _ in np.ndenumerate(x):
    i0=iindex[0]
    i1=iindex[1]
    region=tmp_x[ np.max([i0-h,0]) : np.min([i0+h, sp[0]]), np.max([i1-h,0]):np.min([i1+h,sp[1]])]
    v=np.min(region)
    for j in range(0, (sp[2])):
      #bg_x[i0][i1][j]=v
      bg_x[i0][i1][j]=np.mean(region[:,:,j])

  return bg_x
