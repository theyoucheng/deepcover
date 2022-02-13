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

#shape of the box
#points are stored in anti-clockwise order, starting with the 'bottom-left' point
class boxt:
  def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4):
    self.x1, self.x2, self.x3, self.x4, self.y1, self.y2, self.y3, self.y4=x1, x2, x3, x4, y1, y2, y3, y4

  def area(self):
    return get_area(x1, y1, x2, y2, x3, y3, x4, y4)

#information about the current state of the focus area
class nodet:
  def __init__(self, heatMap, grayMap, frags, x1, x2, x3, x4, y1, y2, y3, y4, inp, outp, totScore, mask_value, depth, part):
    self.heatMap=heatMap
    self.graymap = grayMap
    self.frags=2 #frags
    self.x1, self.x2, self.x3, self.x4, self.y1, self.y2, self.y3, self.y4=x1, x2, x3, x4, y1, y2, y3, y4
    self.inp=inp
    self.outp=outp
    self.totScore=totScore
    self.mask_value=mask_value
    self.fragSize_lb= 3
    self.depth = depth
    self.part = part

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


def compositional_causal_explain(node, eobj):
  tmp = 1
  heatMap=np.zeros(node.heatMap.shape)
  frags=2 #node.frags
  x1, x2, x3, x4, y1, y2, y3, y4=node.x1, node.x2, node.x3, node.x4, node.y1, node.y2, node.y3, node.y4
  straight_part = eobj
  len_x, len_y = node.heatMap.shape[0], node.heatMap.shape[1]
  inp=node.inp
  outp=node.outp
  mask_value=node.mask_value
  rng = np.random.default_rng()
  straight_part = node.part

  final_boxes=None
  final_scores=None
  final_res_rows=None
  var_factor=-1
  max_factor=-1
  ave_factor=-1
  area_factor=10000

  if straight_part:
    length=x2-x1+1
    height=y3-y1+1
    #if node.depth>2 or node.totScore<=10 or length<node.fragSize_lb or height<node.fragSize_lb: ## end point
    if length<node.fragSize_lb or height<node.fragSize_lb or node.depth > 10 or node.totScore <= 10: ## end point
      regionSize=heatMap[x1:x2,y1:y3,:].size #(x2-x1)*(y2-y1)*3
      heatMap[x1:x2,y1:y2,:]=node.totScore/regionSize
      return heatMap
  else:
    #length of shortest side
    min_length = min(distance(x1, y1, x2, y2), distance(x1, y1, x4, y4), distance(x4, y4, x3, y3), distance(x2, y2, x3, y3))+1
    min_gap = min((x2-x1), (x3-x4), (y4-y1), (y3-y2))
    #if node.depth>2 or node.totScore<=10 or length<node.fragSize_lb or height<node.fragSize_lb: ## end point
    if min_length<node.fragSize_lb or min_gap <= 2 or node.depth > 20 or node.totScore <= 10: ## end point
      regionSize=get_area(x1, y1, x2, y2, x3, y3, x4, y4)*3 #area of shape *3
      heatMap = fill_map(heatMap.copy(), node.totScore/regionSize, x1, y1, x2, y2, x3, y3, x4, y4)
      return heatMap

  
  for s in range(0, 1): #step):
    boxes=[]

    if straight_part:
        xi = np.random.randint(x1+1,x2)
        yi = np.random.randint(y1+1,y3)
        #xi = rng.choice(len(rows), 1, p=rows)[0]
        #yi = rng.choice(len(cols), 1, p=cols)[0]
        box0=boxt(x1, y1, xi, y1, xi, yi, x1, yi)
        box1=boxt(x1, yi, xi, yi, xi, y3, x1, y3)
        box2=boxt(xi, y1, x2, y1, x2, yi, xi, yi)
        box3=boxt(xi, yi, x2, yi, x2, y3, xi, y3)
        boxes.append(box0)
        boxes.append(box1)
        boxes.append(box2)
        boxes.append(box3)

    else:
        found = True
        while found:
            xi = np.random.randint(x1+1,x2)
            #xi = int(rng.choice(len_x, 1, p=rowsi)[0])
            yi = (xi-x1)*((y2-y1)/(x2-x1)) + y1

            yj = np.random.randint(y1+1,y4)
            #yj = int(rng.choice(len_y, 1, p=colsj)[0])
            xj = (yj-y1)*((x4-x1)/(y4-y1)) + x1

            xk = np.random.randint(x4+1,x3)
            #xk = int(rng.choice(len_x, 1, p=rowsk)[0])
            yk = (xk-x4)*((y3-y4)/(x3-x4)) + y4

            yl = np.random.randint(y2+1,y3)
            #yl = int(rng.choice(len_y, 1, p=colsl)[0])
            xl = (yl-y2)*((x3-x2)/(y3-y2)) + x2

            (xmid, ymid) = get_intersection(xi,yi,xk,yk,xj,yj,xl,yl)
            if (xmid, ymid) != (0,0):
                found = False

        yi, xj, yk, xl = int(yi), int(xj), int(yk), int(xl)
        # stored anti-clockwise, from bottom left
        box0=boxt(x1,y1,xi,yi,xmid,ymid,xj,yj)
        box1=boxt(xi,yi,x2,y2,xl,yl,xmid,ymid)
        box2=boxt(xmid,ymid,xl,yl,x3,y3,xk,yk)
        box3=boxt(xj,yj,xmid,ymid,xk,yk,x4,y4)
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
            if straight_part: mutant[boxes[index].x1:boxes[index].x2, boxes[index].y1:boxes[index].y3, :]=mask_value
            else: mutant = fill_map(mutant.copy(), mask_value, boxes[index].x1, boxes[index].y1, boxes[index].x2, boxes[index].y2, boxes[index].x3, boxes[index].y3, boxes[index].x4, boxes[index].y4)
            early_stop = early_stop + 1
          else: row.append(True)
        
        #if early_stop<=2: continue

        res=eobj.model.predict(sbfl_preprocess(eobj, np.array([mutant])))
        y_mutant=np.argsort(res)[0][-1:]
        if not (y_mutant[0] in outp):
          row.append(False)
        else:
          row.append(True)
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
        regionSize=get_area(x1, y1, x2, y2, x3, y3, x4, y4)*3 #area of shape *3
        heatMap = fill_map(heatMap.copy(), node.totScore/regionSize, x1, y1, x2, y2, x3, y3, x4, y4)
        return heatMap

  if final_scores is None:
      regionSize=get_area(x1, y1, x2, y2, x3, y3, x4, y4)*3 #area of shape *3
      heatMap = fill_map(heatMap.copy(), node.totScore/regionSize, x1, y1, x2, y2, x3, y3, x4, y4)
      return heatMap
  else:
      if straight_part:
          for i in range(0, len(final_boxes)):
              box=final_boxes[i]
              uIndex=np.unravel_index(i, (2,2))
              if final_scores[uIndex]<0.0001:
                heatMap[box.x1:box.x2,box.y1:box.y3,:]=0
                continue
              child_inp=inp.copy()

              child_node=nodet(heatMap, node.graymap, frags, box.x1, box.x2, box.x1, box.x2, box.y1, box.y3, box.y1, box.y3, child_inp, node.outp, final_scores[uIndex], node.mask_value, node.depth+1, straight_part)
              child_heatMap=compositional_causal_explain(child_node, eobj)
              heatMap[box.x1:box.x2,box.y1:box.y3,:]=child_heatMap[box.x1:box.x2,box.y1:box.y3,:]
      else:
          for i in range(0, len(final_boxes)):
              box=final_boxes[i]
              uIndex=np.unravel_index(i, (2,2))
              if final_scores[uIndex]<0.0001:
                  heatMap = fill_map(heatMap.copy(), 0, box.x1, box.y1, box.x2, box.y2, box.x3, box.y3, box.x4, box.y4)
                  continue
              child_inp=inp.copy()
              child_node=nodet(heatMap, node.graymap, frags, box.x1, box.x2, box.x3, box.x4, box.y1, box.y2, box.y3, box.y4, child_inp, node.outp, final_scores[uIndex], node.mask_value, node.depth+1, straight_part)
              child_heatMap=compositional_causal_explain(child_node, eobj)
              heatMap = copy_map(heatMap.copy(), child_heatMap, box.x1, box.y1, box.x2, box.y2, box.x3, box.y3, box.x4, box.y4)

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
    y=np.argsort(res)[0][-eobj.top_classes:]
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
    for i in range(0,eobj.testgen_iter):
        print ('  #### [Iter {0}: Start Causal Refinement...]'.format(i))
        heatMap=np.zeros(x.shape) # initialise an all-zero heatmap
        frags=2 #3 ## 3x3 is the limit an exhuastive search can handle
        x1, x2, y1, y2=0, int(x.shape[0]), 0, int(x.shape[1])
        totScore = 10000.
        #heatMap=np.ones(x.shape) * (totScore / heatMap.size)
        node=nodet(heatMap, gray_img, frags, x1, x2, x2, x1, y1, y1, y2, y2, x, y, totScore, mask_value=eobj.adv_value, depth=0, part=eobj.straight_part)
        # to call the recursive 'explain' method
        start = time.time()
        res_heatMap=compositional_causal_explain(node, eobj)
        end = time.time()
        print ('  #### [Causal Refinement Done... Time: {0:.0f} seconds]'.format(end-start))

        hmaps.append(res_heatMap)
        hmap = hmap + res_heatMap

        ## update res_heatMap
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
          #ret = top_plot(selement, ind, outs_dir, "causal", eobj)
          for tc in range(0, 10): # to exclude top pixels from the explanation
            top_plot(selement, ind, outs_dir+'-excluding-{0}'.format(tc), "causal", eobj, tc/100.)

          if not eobj.occlusion_file is None:
              f = open(di+"/occlusion-results.txt", "a")
              f.write('{0} {1} {2}\n'.format(eobj.fnames[i], 'causal', ret))
              f.close()

          print ('  #### [Done]')


