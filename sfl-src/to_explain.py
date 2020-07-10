import numpy as np
from spectra_gen import *
from to_rank import *
from utils import *
from datetime import datetime
from mask import *

def to_explain(eobj):
  print ('to explain...')
  model=eobj.model
  ## to create output DI
  di=eobj.outputs
  try:
    os.system('mkdir -p {0}'.format(di))
    print ('mkdir -p {0}'.format(di))
  except: pass

  for i in range(0, len(eobj.inputs)):
    print ('## Input ', i)
    x=eobj.inputs[i]
    res=model.predict(sbfl_preprocess(eobj, np.array([x])))
    y=np.argsort(res)[0][-eobj.top_classes:]
    print (y, np.sort(res)[0][-eobj.top_classes:])
    ite=0
    reasonable_advs=False
    while ite<eobj.testgen_iter:
      print ('#### spectra gen: iteration', ite)
      ite+=1

      #mask=find_mask(x)
      #eobj.adv_value=mask
      #eobj.adv_value=234
      passing, failing=spectra_sym_gen(eobj, x, y[-1:], adv_value=eobj.adv_value, testgen_factor=eobj.testgen_factor, testgen_size=eobj.testgen_size)
      spectra=[]
      num_advs=len(failing)
      adv_xs=[]
      adv_ys=[]
      for e in passing:
        adv_xs.append(e)
        adv_ys.append(0)
      for e in failing:
        adv_xs.append(e)
        adv_ys.append(-1)
      tot=len(adv_xs)

      adv_part=num_advs*1./tot
      print ('###### adv_percentage:', adv_part, num_advs, tot)

      if adv_part<=eobj.adv_lb:
        print ('###### too few advs')
        continue
      elif adv_part>=eobj.adv_ub:
        print ('###### too many advs')
        continue
      else: 
        reasonable_advs=True
        break

    if not reasonable_advs:
      print ('###### failed to explain')
      continue

    ## to obtain the ranking for Input i
    selement=sbfl_elementt(x, 0, adv_xs, adv_ys, model)
    dii=di+'/{0}'.format(str(datetime.now()).replace(' ', '-'))
    dii=dii.replace(':', '-')
    os.system('mkdir -p {0}'.format(dii))
    for measure in eobj.measures:
      ranking_i=to_rank(selement, measure)
      selement.y = y
      diii=dii+'/{0}'.format(measure)
      os.system('mkdir -p {0}'.format(diii))
      np.savetxt(diii+'/ranking.txt', ranking_i, fmt='%s')
      if not eobj.text_only:
        top_plot(selement, ranking_i, diii, measure, eobj)
