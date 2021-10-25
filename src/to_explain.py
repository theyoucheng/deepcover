import time
import numpy as np
from spectra_gen import *
from to_rank import *
from utils import *
from datetime import datetime
from mask import *

def to_explain(eobj):
  print ('\n[To explain: SFL (Software Fault Localization) is used]')
  print ('  ### [Measures: {0}]'.format(eobj.measures))
  model=eobj.model
  ## to create output DI
  #print ('\n[Create output folder: {0}]'.format(eobj.outputs))
  di=eobj.outputs
  try:
    os.system('mkdir -p {0}'.format(di))
  except: pass

  if not eobj.boxes is None:
      f = open(di+"/wsol-results.txt", "a")
      f.write('input_name   x_method    intersection_with_groundtruth\n')
      f.close()

  for i in range(0, len(eobj.inputs)):
    x=eobj.inputs[i]
    res=model.predict(sbfl_preprocess(eobj, np.array([x])))
    y=np.argsort(res)[0][-eobj.top_classes:]

    print ('\n[Input {2}: {0} / Output Label (to Explain): {1}]'.format(eobj.fnames[i], y, i))

    ite=0
    reasonable_advs=False
    while ite<eobj.testgen_iter:
      print ('  #### [Start generating SFL spectra...]')
      start=time.time()
      ite+=1

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
      #print ('###### adv_percentage:', adv_part, num_advs, tot)
      end=time.time()
      print ('  #### [SFL spectra generation DONE: passing {0:.2f} / failing {1:.2f}, total {2}; time: {3:.0f} seconds]'.format(1-adv_part, adv_part, tot, end-start))

      if adv_part<=eobj.adv_lb:
        print ('  #### [too few failing tests: SFL explanation aborts]') 
        continue
      elif adv_part>=eobj.adv_ub:
        print ('  #### [too few many tests: SFL explanation aborts]') 
        continue
      else: 
        reasonable_advs=True
        break

    if not reasonable_advs:
      #print ('###### failed to explain')
      continue

    ## to obtain the ranking for Input i
    selement=sbfl_elementt(x, 0, adv_xs, adv_ys, model, eobj.fnames[i])
    dii=di+'/{0}'.format(str(datetime.now()).replace(' ', '-'))
    dii=dii.replace(':', '-')
    os.system('mkdir -p {0}'.format(dii))
    for measure in eobj.measures:
      print ('  #### [Measuring: {0} is used]'.format(measure))
      ranking_i, spectrum=to_rank(selement, measure)
      selement.y = y
      diii=dii+'/{0}'.format(measure)
      print ('  #### [Saving: {0}]'.format(diii))
      os.system('mkdir -p {0}'.format(diii))
      np.savetxt(diii+'/ranking.txt', ranking_i, fmt='%s')

      # to plot the heatmap
      spectrum = np.array((spectrum/spectrum.max())*255)
      gray_img = np.array(spectrum[:,:,0],dtype='uint8')
      #print (gray_img)
      heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
      if x.shape[2]==1:
          x3d = np.repeat(x[:, :, 0][:, :, np.newaxis], 3, axis=2)
      else: x3d = x
      fin = cv2.addWeighted(heatmap_img, 0.7, x3d, 0.3, 0)
      plt.rcParams["axes.grid"] = False
      plt.imshow(cv2.cvtColor(fin, cv2.COLOR_BGR2RGB))
      plt.savefig(diii+'/heatmap_{0}.png'.format(measure))

      # to plot the top ranked pixels
      if not eobj.text_only:
          ret=top_plot(selement, ranking_i, diii, measure, eobj)
          if not eobj.boxes is None:
              f = open(di+"/wsol-results.txt", "a")
              f.write('{0} {1} {2}\n'.format(eobj.fnames[i], measure, ret))
              f.close()

