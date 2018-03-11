
import cplex
import random
from util import *
from cnnett import *
import sys


# act2==>act
def conv_ss(prior_layer, prior_layer_filter, prior_I, prior_J, current_layer_filter, current_I, current_J, cnnet, X, act, act0, prior_fs):
  var_names0=['d']
  objective=[1]
  lower_bounds=[0.0]
  upper_bounds=[1.0]
  if True: #prior_layer==0:
    upper_bounds=[0.3]

  var_names=[] ## var_names are variables for neurons before max-pooling
  
  N=len(act) # #layers
  for i in range(0, N):
    M=len(act[i]) # #neurons at layer i
    var_names.append(np.empty(act[i].shape, dtype="S40"))
    for j in range(0, M):
      a=act[i][j]
      for k in range(0, len(a)):
        for l in range(0, len(a[k])):
          var_name='x_{0}_{1}_{2}_{3}'.format(i,j,k,l)
          objective.append(0)
          lower_bounds.append(-cplex.infinity)
          upper_bounds.append(cplex.infinity)
          var_names[i][j][k][l]=var_name
          var_names0.append(var_name)

 
  constraints=[]
  rhs=[]
  constraint_senses=[]
  constraint_names=[]

  for i in range(0, len(var_names[0])):
    a=var_names[0][i]
    for k in range(0, len(a)):
      for l in range(0, len(a[k])):
        v=a[k][l]
        # x<=x0+d
        constraints.append([['d', v], [-1, 1]])
        rhs.append(act[0][i][k][l])
        constraint_senses.append("L")
        constraint_names.append("x<=x"+str(i)+"+d")
        # x>=x0-d
        constraints.append([['d', v], [1, 1]])
        rhs.append(act[0][i][k][l])
        constraint_senses.append("G")
        constraint_names.append("x>=x"+str(i)+"-d")
        # x<=1
        constraints.append([[v], [1]])
        rhs.append(1.0)
        constraint_senses.append("L")
        constraint_names.append("x<=1")
        # x>=0
        constraints.append([[v], [1]])
        rhs.append(0.0)
        constraint_senses.append("G")
        constraint_names.append("x>=0")
        if (0==prior_layer and i==prior_layer_filter and prior_I==k and prior_J==l): 
          if act[0][i][k][l]==0:
            # x>=0.1
            constraints.append([[v], [1]])
            rhs.append(0.004)
            constraint_senses.append("G")
            constraint_names.append("x>=0")
          else: #x==0
            constraints.append([[v], [1]])
            rhs.append(0.0)
            constraint_senses.append("L")
            constraint_names.append("x<=0")


  index=0
  conv_acts=[np.copy(var_names[0])] 
  for layer in cnnet.hidden_layers:
    if layer.is_conv: 
      nf=len(layer.w) 
      _nf=len(act[index]) 
      conv_act=[]
      for i in range(0, nf):
        #nr=act0[index][0].shape[0] # number of rows
        #nc=act0[index][0].shape[1] # number of columns
        nfr=layer.w[i][0].shape[0] # number of filter rows
        nfc=layer.w[i][0].shape[1] # number of filter columns
        f_act=act[index+1][i] #np.zeros((nr-nfr+1,nc-nfc+1))

        if index+1==prior_layer+1 and i!=current_layer_filter: continue

        for J in range(0, f_act.shape[0]):
          if index+1==prior_layer+1 and i==current_layer_filter and current_I!=J: continue
          for K in range(0, f_act.shape[1]):
            if index+1==prior_layer+1 and i==current_layer_filter and current_I==J and current_J!=K: continue

            if index+1==prior_layer and not (J,K) in prior_fs: continue
  
            constraint=[[],[]] 
            constraint[0].append(var_names[index+1][i][J][K])
            constraint[1].append(-1)

            for j in range(0, _nf):
              #a=var_names[index][j]
              a=conv_acts[index][j]
  
              for l in range(0, nfr):
                for m in range(0, nfc):
                  #f_act[J][K]+=layer.w[i][j][m][l]*a[J+nfr-m-1][K+nfc-l-1]
                  # we assume the existence of max-pooling...
                  #print a
                  #print index
                  #print j
                  #print a.shape
                  constraint[0].append(a[J+nfr-m-1][K+nfc-l-1])
                  #if (index==prior_layer and i==prior_layer_filter and prior_I==J and prior_J==K):
                  #if (index==prior_layer and j==prior_layer_filter and prior_I==J+nfr-m-1 and prior_J==K+nfc-l-1):
                  #  if act[index][j][J+nfr-m-1][K+nfc-l-1]>0:
                  #    constraint[1].append(0)
                  #  else:
                  #    constraint[1].append(layer.w[i][j][m][l])
                  #else:
                  #if act[index][j][J+nfr-m-1][K+nfc-l-1]>0 or index==0:
                  constraint[1].append(layer.w[i][j][m][l])
                  #else:
                  #  constraint[1].append(0)
            constraints.append(constraint)
            rhs.append(-layer.b[i])
            constraint_senses.append("E")
            constraint_names.append("eq:"+"x_"+str(i)+"_"+str(j))
            ### ReLU
            _constraint=[[],[]]
            v=var_names[index+1][i][J][K]
            _constraint[0].append(v)
            _constraint[1].append(1)
            constraints.append(_constraint)
            rhs.append(0)              
            if (index+1==prior_layer and i==prior_layer_filter and prior_I==J and prior_J==K) or (index+1==prior_layer+1 and i==current_layer_filter and current_I==J and current_J==K):
            #if (index+1==prior_layer+1 and i==current_layer_filter and current_I==J and current_J==K):
              if act[index+1][i][J][K]>0:
                constraint_senses.append("L")
              else:
                constraint_senses.append("G")
            else:
              if act[index+1][i][J][K]>0:
                constraint_senses.append("G")
              else:
                constraint_senses.append("L")
            constraint_names.append("relu: "+v)

        ### max-pool  
        nr=f_act.shape[0]
        nc=f_act.shape[1]
        #### shape after max-pooling
        #p_act=np.zeros((nr/layer.mp_size_x, nc/layer.mp_size_y))
        p_act=np.empty((nr/layer.mp_size_x, nc/layer.mp_size_y), dtype="S40")
        #print '/////'
        #print p_act.shape
        #print '/////'
        #print f_act.shape
        #sys.exit(0)
        for pi in range(0, len(p_act)):
          for pj in range(0, len(p_act[pi])):
            v='L{0}F{1}-{2}-{3}'.format(index+1, i, pi, pj)
            p_act[pi][pj]=v
            var_names0.append(v)
            objective.append(0)
            lower_bounds.append(-cplex.infinity)
            upper_bounds.append(cplex.infinity)
        for Ii in range(0, p_act.shape[0]):
          for Ji in range(0, p_act.shape[1]):
            ##########
            II=layer.mp_size_x*Ii
            JJ=layer.mp_size_y*Ji
            for ii in range(layer.mp_size_x*Ii, layer.mp_size_x*(Ii+1)):
              for jj in range(layer.mp_size_y*Ji, layer.mp_size_y*(Ji+1)):
                #if act[index+1][i][ii][jj]> act[index+1][i][II][JJ]:
                #  #if index==prior_layer and i==prior_layer_filter and prior_I==ii and prior_J==jj:
                #  #  continue ### this one has been negated to 0 
                #  II=ii
                #  JJ=jj
                ##p_act[Ii][Ji]=var_names[ii][jj]
                _constraint=[[],[]]
                _constraint=[[p_act[Ii][Ji], var_names[index+1][i][ii][jj]],[1,-1]]
                constraints.append(_constraint)
                rhs.append(0)  
                constraint_senses.append("G")
                constraint_names.append("max-pooling")
            _constraint=[[],[]]
            _constraint=[[p_act[Ii][Ji]],[1]]
            constraints.append(_constraint)
            rhs.append(0)  
            constraint_senses.append("G")
            constraint_names.append("max-pooling")
        conv_act.append(np.array(p_act))
      conv_acts.append(np.array(conv_act))
    else:
      nr=layer.w.shape[0]
      nc=layer.w.shape[1]
      ### reshape
      #vs=var_names[index].reshape(1, nr)
      vs=conv_acts[index].reshape(1,nr)
      #print vs
      #sys.exit(0)
      #sh=act[index].shape
      sh=conv_act[index].shape
      #aa=act[index].reshape(1, nr)

      #this_act=np.zeros((1,nc))
      this_act=np.empty((1, nc), dtype="S40")
      for ti in range(0, len(this_act)):
        for tj in range(0, len(this_act[ti])):
          this_act[ti][tj]=var_names[index+1][0][ti][tj]
      ######
      for I in range(0, nc):
        if index+1==prior_layer+1 and not (current_J==I): continue
        constraint=[[],[]] 
        constraint[0].append(var_names[index+1][0][0][I])
        constraint[1].append(-1)
        for II in range(0, nr):
          #this_act[0][I]+=aa[0][II]*layer.w[II][I]
          constraint[0].append(vs[0][II])
          if cnnet.hidden_layers[index-1].is_conv: ### at least one convolutional layer before the 1st fully connected layer
              constraint[1].append(layer.w[II][I])
          elif (index==prior_layer and II==prior_layer_filter*sh[1]*sh[2] + prior_I*sh[2] + prior_J): #   i==prior_layer_filter and prior_I==0 and prior_J==I):
            aa=act[index].reshape(1, nr)
            if aa[0][II]>0:
              constraint[1].append(0)
            else:
              constraint[1].append(layer.w[II][I])
          else:
            aa=act[index].reshape(1, nr)
            if aa[0][II]>0 or index==0:
              constraint[1].append(layer.w[II][I])
            else:
              constraint[1].append(0)
        #this_act[0][I]+=layer.b[I]
        constraints.append(constraint)
        rhs.append(-layer.b[I])
        constraint_senses.append("E")
        constraint_names.append('')

        if True: #index < N-2:
          #this_act[0][I]=0
          _constraint=[[],[]]
          v=var_names[index+1][0][0][I]
          _constraint[0].append(v)
          _constraint[1].append(1)
          constraints.append(_constraint)
          rhs.append(0)              
          if (index+1==prior_layer and i==prior_layer_filter and prior_I==0 and prior_J==I) or (index+1==prior_layer+1 and i==current_layer_filter and current_I==0 and current_J==I):
            if act[index+1][0][0][I]>0: 
              constraint_senses.append("L")
            else:
              constraint_senses.append("G")
          elif index<N-2:
            if act[index+1][0][0][I]>0: 
              constraint_senses.append("G")
            else:
              constraint_senses.append("L")
          constraint_names.append("act{0}{1}{2}{3}>0".format(index+1, 0, 0, I))
        #if index==N-2:
        #  if (index+1==prior_layer and i==prior_layer_filter and prior_I==0 and prior_J==I) or (index+1==prior_layer+1 and i==current_layer_filter and current_I==0 and current_J==I):
        #    _constraint=[[],[]]
        #    v=var_names[index+1][0][0][I]
        #    _constraint[0].append(v)
        #    _constraint[1].append(1)
        #    constraints.append(_constraint)
        #    rhs.append(0)              
        #    if act[index+1][0][0][I]>0: 
        #      constraint_senses.append("L")
        #    else:
        #      constraint_senses.append("G")
        #    constraint_names.append("relu: "+v)
      conv_acts.append(np.array([this_act]))
            
    ##########
    if index==prior_layer: break
    index+=1
  ###### solve
  try:
    problem=cplex.Cplex()
    problem.variables.add(obj = objective,
                          lb = lower_bounds,
                          ub = upper_bounds,
                          names = var_names0)
    problem.linear_constraints.add(lin_expr=constraints,
                                   senses = constraint_senses,
                                   rhs = rhs,
                                   names = constraint_names)
    problem.solve()

    ####
    d=problem.solution.get_values("d")
    new_x=[]
    for i in range(0, len(var_names[0])):
      for var_x in var_names[0][i]:
        for var_xy in var_x:
          v=problem.solution.get_values(var_xy)
          if v<0 or v>1:
            print d
            print var_xy
            print v
            return False, X, -1
          new_x.append(v)

    #if d==0 or d==1:      
    #  return False, _, _

    return True, new_x, d
    #return True, X, d

  except:
    return False,[],-1
  #  
  #  try:
  #    d=problem.solution.get_values("d")
  #    print 'd is {0}'.format(d)
  #    new_x=[]
  #    #for i in len(X):
  #    #  new_x.append(problem.solution.get_values('x_0_'+str(i)))
  #    #return True, new_x, d
  #  except:
  #    print 'Exception for feasible model???'
  #    sys.exit(0)

