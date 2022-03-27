import numpy as np
from spectra_gen import *
from to_rank import *
from utils import *
from datetime import datetime
from mask import *
from itertools import combinations
import math
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import cv2
import time

### todo: more subtle approach to distribute the total score
### todo: maybe I should put a delta=
class boxt:
  def __init__(self, x1, x2, y1, y2):
    self.x1, self.x2, self.y1, self.y2=x1, x2, y1, y2

  def area(self):
      return (self.x2-self.x1)*(self.y2-self.y1)

  # to check if current box includes param box
  def include(self, box):
      #print( self.x1, box.x1, self.x2, box.x2, self.y1, box.y1, self.y2, box.y2)
      #print( self.x1<=box.x1 and self.x2>=box.x2 and self.y1<=box.y1 and self.y2>=box.y2)
      return self.x1<=box.x1 and self.x2>=box.x2 and self.y1<=box.y1 and self.y2>=box.y2

# remove redundent box-es from a list of boxes
def clean_boxes(boxes):
    new_boxes = []
    for i in range(0, len(boxes)):
        min_flag = True # is min box
        for j in range(0, len(boxes)):
            if i==j: continue
            if boxes[i].include(boxes[j]):
                min_flag = False
                break
        if min_flag:
            new_boxes.append(boxes[i])
    return new_boxes

# disjoint box1 includes dbox2:
def dbox_includes(dbox1, dbox2):
    flag = True
    for box2 in dbox2:
        include = False
        for box1 in dbox1:
            if box1.include(box2):
                include = True
                break
        if not include:
            flag = False
            break

    return flag

# remove redundent disjoint box-es from a list of disjoint boxes
def clean_dboxes(dboxes):
    new_boxes = []
    for i in range(0, len(dboxes)):
        min_flag = True # is min d box
        for j in range(0, len(dboxes)):
            if i==j: continue
            if dbox_includes(dboxes[i], dboxes[j]):
                min_flag = False
                break
        if min_flag:
            new_boxes.append(dboxes[i])
    return new_boxes

class nodet:
  def __init__(self, heatMap, frags, x1, x2, y1, y2, inp, outp, totScore, mask_value, depth):
    self.heatMap=heatMap
    self.frags=2 #frags
    self.x1, self.x2, self.y1, self.y2=x1, x2, y1, y2
    self.inp=inp
    self.outp=outp
    self.totScore=totScore
    self.mask_value=mask_value
    self.fragSize_lb= 3 
    self.depth = depth

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

## Given the 'truthTable', to compute the i-th feature's score
def causal_search(i_feature, truthTable):
  all_true=[]
  for i in range(0, len(truthTable)-1):
    all_true.append(True)
  for i in range(0, len(truthTable)):
    i_rows=truthTable[i]
    for row in i_rows:
      if row[i_feature] or row[-1:][0]: continue
      if i==0:
        return row[:-1], all_true #1/(1.+1)
      ## to search these rows with i-1 False
      minus_i_rows=truthTable[i-1]
      for row2 in minus_i_rows:
        if not row2[i_feature] or not row2[-1:][0]: continue
        xor_res=np.logical_xor(row, row2)
        if np.count_nonzero(xor_res)>2: continue
        c=len(row)-1-np.count_nonzero(row)
        return row[:-1], row2[:-1]
  return None, None

# approximate_exps: Disjoint BOXes of EXPlanations
def compositional_causal_explain(node, eobj, dbox_exps):
  tmp = 1
  heatMap=np.zeros(node.heatMap.shape)
  frags=2 #node.frags
  x1, x2, y1, y2=node.x1, node.x2, node.y1, node.y2
  inp=node.inp
  outp=node.outp
  mask_value=node.mask_value

  length=x2-x1+1
  height=y2-y1+1

  final_boxes=None
  final_scores=None
  final_res_rows=None
  var_factor=-1
  max_factor=-1
  ave_factor=-1
  area_factor=10000

  #if node.depth>2 or node.totScore<=10 or length<node.fragSize_lb or height<node.fragSize_lb: ## end point
  if length<node.fragSize_lb or height<node.fragSize_lb: ## end point
      regionSize=heatMap[x1:x2,y1:y2,:].size #(x2-x1)*(y2-y1)*3
      heatMap[x1:x2,y1:y2,:]=node.totScore/regionSize
      return heatMap
  
  for s in range(0, 1): #step):
    boxes=[]
    xi = np.random.randint(x1+1,x2)
    yi = np.random.randint(y1+1,y2)
    box0=boxt(x1, xi, y1, yi)
    box1=boxt(x1, xi, yi, y2)
    box2=boxt(xi, x2, y1, yi)
    box3=boxt(xi, x2, yi, y2)
    boxes.append(box0)
    boxes.append(box1)
    boxes.append(box2)
    boxes.append(box3)
  
    # to build the truth table
    truthTable=[]
    n=4
    indices=np.arange(0, n)
    for r in range(1, n+1):
      comb_list = list(combinations(indices, r))
      rows_r=[]
      for comb in comb_list:
        early_stop = 0
        row=[]
        mutant=inp.copy()
        for index in indices:
          if index in comb:
            row.append(False)
            mutant[boxes[index].x1:boxes[index].x2, boxes[index].y1:boxes[index].y2, :]=mask_value
            early_stop = early_stop + 1
          else: row.append(True)
        
        #if early_stop<=2: continue

        res=eobj.model.predict(sbfl_preprocess(eobj, np.array([mutant])))
        #y_mutant=np.argsort(res)[0][-1:]
        y_mutant=get_prediction(res)#np.argsort(res)[0][-1:]
        if not (y_mutant[0] in outp):
          row.append(False)
        else:
          row.append(True)
          # an explanation can be found by using disjoint (out of n=4) superpixels
          # if r == 3 # if only considering one superpixel out of 4
          if r < n:
            dboxes = []
            mutant = np.ones(inp.shape) * mask_value
            for index in indices:
                if not index in comb:
                      mutant[boxes[index].x1:boxes[index].x2, boxes[index].y1:boxes[index].y2, :]=inp[boxes[index].x1:boxes[index].x2, boxes[index].y1:boxes[index].y2, :]
                      dboxes.append(boxes[index])
            res=eobj.model.predict(sbfl_preprocess(eobj, np.array([mutant])))
            y_mutant=get_prediction(res)
            if (y_mutant[0] in outp):
                dbox_exps.append(dboxes)

        rows_r.append(np.array(row))
      ##
      truthTable.append(np.array(rows_r))

    # to compute the scores
    scores=np.zeros((frags,frags))
    res_rows=[]
    for i_feature in range(0, n):
      row, row2=causal_search(i_feature, truthTable)
      res_rows.append(row2)
      uIndex=np.unravel_index(i_feature, (frags,frags))
      res=0
      if row is not None:
        res=1./(len(row)-np.count_nonzero(row)+1)
      scores[uIndex]=res

    std=np.std(scores)
    ave=np.mean(scores)
    max_=np.max(scores)
    if std>var_factor:
      var_factor=std
      ave_factor=ave
      final_boxes=boxes
      final_scores=scores
      final_res_rows=res_rows

  if final_scores is not None:
    norm_sum=final_scores.sum()
    if norm_sum>0:
      final_scores=(final_scores/norm_sum)*node.totScore
    else:
        regionSize=heatMap[x1:x2,y1:y2,:].size 
        heatMap[x1:x2,y1:y2,:]=node.totScore/regionSize
        return heatMap

  if final_scores is None: 
      regionSize=heatMap[x1:x2,y1:y2,:].size 
      heatMap[x1:x2,y1:y2,:]=node.totScore/regionSize
      return heatMap
  else:
      for i in range(0, len(final_boxes)):
          box=final_boxes[i]
          uIndex=np.unravel_index(i, (2,2))
          if final_scores[uIndex]<0.0001:
            heatMap[box.x1:box.x2,box.y1:box.y2,:]=0
            continue
          child_inp=inp.copy()
              
          child_node=nodet(heatMap, frags, box.x1, box.x2, box.y1, box.y2, child_inp, node.outp, final_scores[uIndex], node.mask_value, node.depth+1)
          child_heatMap=compositional_causal_explain(child_node, eobj, dbox_exps)
          heatMap[box.x1:box.x2,box.y1:box.y2,:]=child_heatMap[box.x1:box.x2,box.y1:box.y2,:]
                

  return heatMap


def comp_explain(eobj):
  print ('\n[To explain: Causal Explanation is used]')
  model=eobj.model # this is the model to explain
  ## to create output DI
  di=eobj.outputs # output dir
  try:
    os.system('mkdir -p {0}'.format(di))
    #print ('mkdir -p {0}'.format(di))
  except: pass

  if not eobj.occlusion_file is None:
      f = open(di+"/occlusion-results.txt", "a")
      f.write('input_name   x_method    [x size, intersection with occlu, iou with occlu]\n')
      f.close()

  landmark = False
  for index in range(0, len(eobj.inputs)):
    name=eobj.fnames[index]
    x=eobj.inputs[index]
    res=model.predict(sbfl_preprocess(eobj, np.array([x])))
    #y=np.argsort(res)[0][-eobj.top_classes:]
    y=get_prediction(res)#np.argsort(res)[0][-1:]
    #print ('## Input:', index, name)
    print ('\n[Input {2}: {0} / Output Label (to Explain): {1}]'.format(eobj.fnames[index], y, index))
    #print ('## Output:', y, np.sort(res)[0][-eobj.top_classes:])
    #print ('## Output:', np.argsort(res)[0][-5:])
    #print (x.shape)
    #continue

    dii=di+'/{1}-{0}'.format(str(datetime.now()).replace(' ', '-'), "causal")
    dii=dii.replace(':', '-')
    os.system('mkdir -p {0}'.format(dii))
    hmaps = []
    hmap = np.zeros(x.shape)
    iou_min = 1
    exp_min = 1
    intersection_min = 1
    dbox_exps = [[boxt(0, int(x.shape[0]), 0, int(x.shape[1]))]]
    for i in range(0,eobj.testgen_iter):
        print ('  #### [Iter {0}: Start Causal Refinement...]'.format(i))
        heatMap=np.zeros(x.shape) # initialise an all-zero heatmap
        frags=2 #3 ## 3x3 is the limit an exhuastive search can handle
        x1, x2, y1, y2=0, int(x.shape[0]), 0, int(x.shape[1])
        totScore = 10000.
        #heatMap=np.ones(x.shape) * (totScore / heatMap.size)
        node=nodet(heatMap, frags, x1, x2, y1, y2, x, y, totScore, mask_value=eobj.adv_value, depth=0)
        # to call the recursive 'explain' method
        start = time.time()
        res_heatMap=compositional_causal_explain(node, eobj, dbox_exps)
        dbox_exps = clean_dboxes(dbox_exps)
        print ("  #### dbox for exps: {0}".format(len(dbox_exps)))
        for bindex, dbox_exp in enumerate(dbox_exps):
            #print(bbox_exp.x1, bbox_exp.x2, bbox_exp.y1, bbox_exp.y2)
            mutant = np.ones(x.shape) * eobj.adv_value
            for bbox_exp in dbox_exp:
                mutant[bbox_exp.x1:bbox_exp.x2, bbox_exp.y1:bbox_exp.y2, :]=x[bbox_exp.x1:bbox_exp.x2, bbox_exp.y1:bbox_exp.y2, :]
            save_an_image(mutant, 'bbox{0}'.format(bindex), dii)
        end = time.time()
        print ('  #### [Causal Refinement Done... Time: {0:.0f} seconds]'.format(end-start))

        ## update res_heatMap
        res_heatMap = np.array((res_heatMap/res_heatMap.max()))
        hmaps.append(res_heatMap)
        if eobj.causal_min:
            # min
            heatmap_bk = copy.deepcopy(hmaps[0])
            hmaps[0] = np.minimum(hmaps[0], res_heatMap)
            res_heatMap = hmaps[0]
        elif eobj.causal_max:
            # max
            heatmap_bk = copy.deepcopy(hmaps[0])
            hmaps[0] = np.maximum(hmaps[0], res_heatMap)
            res_heatMap = hmaps[0]
        else:
            # average by default
            hmap = hmap + res_heatMap
            res_heatMap = hmap/len(hmaps)

        smooth = np.ones(res_heatMap.shape)
        sI = res_heatMap.shape[0]
        sJ = res_heatMap.shape[1]
        sd = 3 #5 2
        for si in range(0, res_heatMap.shape[0]):
            for sj in range(0, res_heatMap.shape[1]):
                for sk in range(0, 3):
                    smooth[si][sj][sk] = np.mean(res_heatMap[np.max([0, si-sd]):np.min([sI, si+sd]), np.max([0,sj-sd]):np.min([sJ, sj+sd]), sk])


        res_heatMap = smooth
        res_heatMap = np.array((res_heatMap/res_heatMap.max())*255)

        gray_img = np.array(res_heatMap[:,:,0],dtype='uint8')
        heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
        fin = cv2.addWeighted(heatmap_img, 0.7, x, 0.3, 0)
        plt.rcParams["axes.grid"] = False
        plt.imshow(cv2.cvtColor(fin, cv2.COLOR_BGR2RGB))

        hmap_name = (dii+'/heatmap_iter{0}.png'.format(i))
        plt.axis('off')
        plt.savefig(hmap_name, bbox_inches='tight', pad_inches=0)
        print ('  #### [Saved Heatmap: {0}]'.format(hmap_name))

        if not eobj.text_only:
          selement = sbfl_elementt(x, 0, None, None, model)
          selement.y = y
          ind = np.argsort(res_heatMap, axis=None)
          
          outs_dir = dii+'/iter{0}'.format(i)
          print ('  #### [Saving into {0}]'.format(outs_dir))
          ret = top_plot(selement, ind, outs_dir, "causal", eobj)

          # In addition, we draw explanations inside each bbox
          top_plot_exps(selement, ind, dii, "causal", eobj, dbox_exps)

          if not eobj.occlusion_file is None:
              f = open(di+"/occlusion-results.txt", "a")
              f.write('{0} {1} {2}\n'.format(eobj.fnames[i], 'causal', ret))
              f.close()

          print ('  #### [Done]')


