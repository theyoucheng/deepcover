
import cplex
import random
from util import *
from nnett import *
import sys

def rp_ssc(I, J, K, nnet, X, act):
  var_names=['d']
  objective=[1]
  lower_bounds=[0.0]
  upper_bounds=[1.0]
  
  N=len(act) # #layers
  for i in range(0, N):
    if i>I+1: break
    M=len(act[i]) # #neurons at layer i
    for j in range(0, M):
      if i==I+1 and j!=K: continue
      var_names.append('x_'+str(i)+'_'+str(j))
      objective.append(0)
      lower_bounds.append(-cplex.infinity)
      upper_bounds.append(cplex.infinity)

 
  constraints=[]
  rhs=[]
  constraint_senses=[]
  constraint_names=[]


  for i in range(0, len(X)):
    # x<=x0+d
    constraints.append([[0, i+1], [-1, 1]])
    rhs.append(X[i])
    constraint_senses.append("L")
    constraint_names.append("x<=x"+str(i)+"+d")
    # x>=x0-d
    constraints.append([[0, i+1], [1, 1]])
    rhs.append(X[i])
    constraint_senses.append("G")
    constraint_names.append("x>=x"+str(i)+"-d")
    # x<=1
    constraints.append([[i+1], [1]])
    rhs.append(1.0)
    constraint_senses.append("L")
    constraint_names.append("x<=1")
    # x>=0
    constraints.append([[i+1], [1]])
    rhs.append(0.0)
    constraint_senses.append("G")
    constraint_names.append("x>=0")

  # there is nothing to constrain for layer 0
  # and we start from layer 1
  # the last layer shall be handled individually
  for i in range(1, I+2):
    M=len(act[i]) # number of neurons at layer i 
    for j in range(0, M): 
      #### for layer (I+1) we only need to access one neuron
      if i==I+1 and j!=K: continue
      constraint=[[],[]] 
      constraint[0].append("x_"+str(i)+"_"+str(j))
      constraint[1].append(-1)
      for k in range(0, len(act[i-1])):
        constraint[0].append("x_"+str(i-1)+"_"+str(k))
        if i==1 or act[i-1][k]>0:
          if not (i-1==I and k==J):
            constraint[1].append(nnet.weights[i-1][k][j])
          else:
            constraint[1].append(0)
        else:
          if not (i-1==I and k==J):
            constraint[1].append(0)
          else:
            constraint[1].append(nnet.weights[i-1][k][j])
      constraints.append(constraint)
      rhs.append(-nnet.biases[i][j])
      constraint_senses.append("E")
      constraint_names.append("eq:"+"x_"+str(i)+"_"+str(j))

      ###### ReLU
      if i<N-1:
        _constraint=[[],[]]
        _constraint[0].append("x_"+str(i)+"_"+str(j))
        _constraint[1].append(1)
        constraints.append(_constraint)
        rhs.append(0)
        if not( (i==I and j==J) or (i==I+1 and j==K) ):
          if act[i][j]>0:
            constraint_senses.append("G")
          else:
            constraint_senses.append("L")
          constraint_names.append("relu:"+"x_"+str(i)+"_"+str(j))
        else:
          if act[i][j]>0:
            constraint_senses.append("L")
          else:
            constraint_senses.append("G")
          constraint_names.append("not relu:"+"x_"+str(i)+"_"+str(j))

  if I==N-2: # I+1==N-1
    #### Now, we are at the output layer
    #### x_{N-1, K}>=x_{N-1,old_label}
    label=np.argmax(act[N-1])
    for i in range(0, len(act[N-1])):
      if i!=K: continue
      constraint=[[],[]] 
      constraint[0].append("x_"+str(N-1)+"_"+str(i))
      constraint[1].append(1)
      #constraint[0].append("x_"+str(N-1)+"_"+str(label))
      #constraint[1].append(-1)
      constraints.append(constraint)
      rhs.append(0.0)
      if act[N-1][K]>0:
        constraint_senses.append("L")
      else:
        constraint_senses.append("G")
      constraint_names.append("not K")

  ###### solve
  try:
    problem=cplex.Cplex()
    problem.variables.add(obj = objective,
                          lb = lower_bounds,
                          ub = upper_bounds,
                          names = var_names)
    problem.linear_constraints.add(lin_expr=constraints,
                                   senses = constraint_senses,
                                   rhs = rhs,
                                   names = constraint_names)
    problem.solve()

    ####
    d=problem.solution.get_values("d")
    new_x=[]
    for i in range(0, len(X)):
      v=(problem.solution.get_values('x_0_'+str(i)))
      if v<0 or v>1: return False, _, _
      new_x.append(v)

    if d==0 or d==1:      
      return False, _, _, _, _


    #print problem.variables.get_num(), problem.linear_constraints.get_num()
    return True, new_x, d, problem.variables.get_num(), problem.linear_constraints.get_num()

  except:
    return False,[],-1, -1, -1
    
    try:
      d=problem.solution.get_values("d")
      print 'd is {0}'.format(d)
      new_x=[]
      #for i in len(X):
      #  new_x.append(problem.solution.get_values('x_0_'+str(i)))
      #return True, new_x, d
    except:
      print 'Exception for feasible model???'
      sys.exit(0)

def rp_dsc(I, J, nnet, X, act):

  var_names=['d']
  objective=[1]
  lower_bounds=[0.0]
  upper_bounds=[1.0]
  
  N=len(act) # #layers
  for i in range(0, N):
    M=len(act[i]) # #neurons at layer i
    for j in range(0, M):
      var_names.append('x_'+str(i)+'_'+str(j))
      objective.append(0)
      lower_bounds.append(-cplex.infinity)
      upper_bounds.append(cplex.infinity)

 
  constraints=[]
  rhs=[]
  constraint_senses=[]
  constraint_names=[]


  for i in range(0, len(X)):
    # x<=x0+d
    constraints.append([[0, i+1], [-1, 1]])
    rhs.append(X[i])
    constraint_senses.append("L")
    constraint_names.append("x<=x"+str(i)+"+d")
    # x>=x0-d
    constraints.append([[0, i+1], [1, 1]])
    rhs.append(X[i])
    constraint_senses.append("G")
    constraint_names.append("x>=x"+str(i)+"-d")
    # x<=1
    constraints.append([[i+1], [1]])
    rhs.append(1.0)
    constraint_senses.append("L")
    constraint_names.append("x<=1")
    # x>=0
    constraints.append([[i+1], [1]])
    rhs.append(0.0)
    constraint_senses.append("G")
    constraint_names.append("x>=0")

  # there is nothing to constrain for layer 0
  # and we start from layer 1
  # the last layer shall be handled individually
  for i in range(1, I+1):
    M=len(act[i]) # number of neurons at layer i 
    for j in range(0, M): 
      #### for layer (I+1) we only need to access one neuron
      if i==I and j!=J: continue
      constraint=[[],[]] 
      constraint[0].append("x_"+str(i)+"_"+str(j))
      constraint[1].append(-1)
      for k in range(0, len(act[i-1])):
        constraint[0].append("x_"+str(i-1)+"_"+str(k))
        if i==1 or act[i-1][k]>0:
           constraint[1].append(nnet.weights[i-1][k][j])
        else:
           constraint[1].append(0)
      constraints.append(constraint)
      rhs.append(-nnet.biases[i][j])
      constraint_senses.append("E")
      constraint_names.append("eq:"+"x_"+str(i)+"_"+str(j))

      ###### ReLU
      if i<N-1:
        _constraint=[[],[]]
        _constraint[0].append("x_"+str(i)+"_"+str(j))
        _constraint[1].append(1)
        constraints.append(_constraint)
        if not(i==I and j==J):
          rhs.append(0)
          if act[i][j]>0:
            constraint_senses.append("G")
          else:
            constraint_senses.append("L")
          constraint_names.append("relu:"+"x_"+str(i)+"_"+str(j))
        else: ## I+1, K
          ## ReLU sign does not change
          rhs.append(0)
          if act[i][j]>0:
            constraint_senses.append("L")
          else:
            constraint_senses.append("G")
          constraint_names.append("relu:"+"x_"+str(i)+"_"+str(j))

  if I==N-1: # I+1==N-1
    #### Now, we are at the output layer
    #### x_{N-1, K}>=x_{N-1,old_label}
    label=np.argmax(act[N-1])
    for i in range(0, len(act[N-1])):
      if i!=J: continue
      constraint=[[],[]] 
      constraint[0].append("x_"+str(N-1)+"_"+str(i))
      constraint[1].append(1)
      constraints.append(constraint)

      ##1) ReLU sign does not change
      rhs.append(0)
      if act[I][J]>0:
        constraint_senses.append("L")
      else:
        constraint_senses.append("G")
      constraint_names.append("relu sign:"+"x_"+str(I)+"_"+str(J))

  ###### solve
  try:
    problem=cplex.Cplex()
    problem.variables.add(obj = objective,
                          lb = lower_bounds,
                          ub = upper_bounds,
                          names = var_names)
    problem.linear_constraints.add(lin_expr=constraints,
                                   senses = constraint_senses,
                                   rhs = rhs,
                                   names = constraint_names)
    problem.solve()

    ####
    d=problem.solution.get_values("d")
    new_x=[]
    for i in range(0, len(X)):
      v=(problem.solution.get_values('x_0_'+str(i)))
      if v<0 or v>1: return False, _, _
      new_x.append(v)

    if d==0 or d==1:      
      return False, _, _

    return True, new_x, d

  except:
    return False,[],-1
    
    try:
      d=problem.solution.get_values("d")
      print 'd is {0}'.format(d)
      new_x=[]
      #for i in len(X):
      #  new_x.append(problem.solution.get_values('x_0_'+str(i)))
      #return True, new_x, d
    except:
      print 'Exception for feasible model???'
      sys.exit(0)

def rp_svc(I, J, K, nnet, X, act, sfactor):

  var_names=['d']
  objective=[1]
  lower_bounds=[0.0]
  upper_bounds=[1.0]
  
  N=len(act) # #layers
  for i in range(0, N):
    M=len(act[i]) # #neurons at layer i
    for j in range(0, M):
      var_names.append('x_'+str(i)+'_'+str(j))
      objective.append(0)
      lower_bounds.append(-cplex.infinity)
      upper_bounds.append(cplex.infinity)

 
  constraints=[]
  rhs=[]
  constraint_senses=[]
  constraint_names=[]


  for i in range(0, len(X)):
    # x<=x0+d
    constraints.append([[0, i+1], [-1, 1]])
    rhs.append(X[i])
    constraint_senses.append("L")
    constraint_names.append("x<=x"+str(i)+"+d")
    # x>=x0-d
    constraints.append([[0, i+1], [1, 1]])
    rhs.append(X[i])
    constraint_senses.append("G")
    constraint_names.append("x>=x"+str(i)+"-d")
    # x<=1
    constraints.append([[i+1], [1]])
    rhs.append(1.0)
    constraint_senses.append("L")
    constraint_names.append("x<=1")
    # x>=0
    constraints.append([[i+1], [1]])
    rhs.append(0.0)
    constraint_senses.append("G")
    constraint_names.append("x>=0")

  # there is nothing to constrain for layer 0
  # and we start from layer 1
  # the last layer shall be handled individually
  for i in range(1, I+2):
    M=len(act[i]) # number of neurons at layer i 
    for j in range(0, M): 
      #### for layer (I+1) we only need to access one neuron
      if i==I+1 and j!=K: continue
      constraint=[[],[]] 
      constraint[0].append("x_"+str(i)+"_"+str(j))
      constraint[1].append(-1)
      for k in range(0, len(act[i-1])):
        constraint[0].append("x_"+str(i-1)+"_"+str(k))
        if i==1 or act[i-1][k]>0:
          if not (i-1==I and k==J):
            constraint[1].append(nnet.weights[i-1][k][j])
          else:
            constraint[1].append(0)
        else:
          if not (i-1==I and k==J):
            constraint[1].append(0)
          else:
            constraint[1].append(nnet.weights[i-1][k][j])
      constraints.append(constraint)
      rhs.append(-nnet.biases[i][j])
      constraint_senses.append("E")
      constraint_names.append("eq:"+"x_"+str(i)+"_"+str(j))

      ###### ReLU
      if i<N-1:
        _constraint=[[],[]]
        _constraint[0].append("x_"+str(i)+"_"+str(j))
        _constraint[1].append(1)
        constraints.append(_constraint)
        if not( (i==I and j==J) or (i==I+1 and j==K) ):
          rhs.append(0)
          if act[i][j]>0:
            constraint_senses.append("G")
          else:
            constraint_senses.append("L")
          constraint_names.append("relu:"+"x_"+str(i)+"_"+str(j))
        elif (i==I and j==J): #Activation change
          rhs.append(0)
          if act[i][j]>0:
            constraint_senses.append("L")
          else:
            constraint_senses.append("G")
          constraint_names.append("not relu:"+"x_"+str(i)+"_"+str(j))
        else: ## I+1, K
          ## ReLU sign does not change
          rhs.append(0)
          if act[i][j]>0:
            constraint_senses.append("G")
          else:
            constraint_senses.append("L")
          constraint_names.append("relu:"+"x_"+str(i)+"_"+str(j))

          ## ReLU value changed
          _constraint=[[],[]]
          _constraint[0].append("x_"+str(i)+"_"+str(j))
          _constraint[1].append(1)
          constraints.append(_constraint)
          rhs.append(sfactor*act[I+1][K])
          if act[i][j]>0:
            if sfactor>1.0:
              constraint_senses.append("G")
            else:
              constraint_senses.append("L")
          else:
            if sfactor>1.0:
              constraint_senses.append("L")
            else:
              constraint_senses.append("G")
          constraint_names.append("relu value change:"+"x_"+str(i)+"_"+str(j))

  if I==N-2: # I+1==N-1
    #### Now, we are at the output layer
    #### x_{N-1, K}>=x_{N-1,old_label}
    label=np.argmax(act[N-1])
    for i in range(0, len(act[N-1])):
      if i!=K: continue
      constraint=[[],[]] 
      constraint[0].append("x_"+str(N-1)+"_"+str(i))
      constraint[1].append(1)
      constraints.append(constraint)

      ##1) ReLU sign does not change
      rhs.append(0)
      if act[I+1][K]>0:
        constraint_senses.append("G")
      else:
        constraint_senses.append("L")
      constraint_names.append("relu sign:"+"x_"+str(I+1)+"_"+str(K))

      ## ReLU value changed
      _constraint=[[],[]]
      _constraint[0].append("x_"+str(I+1)+"_"+str(K))
      _constraint[1].append(1)
      constraints.append(_constraint)
      rhs.append(sfactor*act[I+1][K])
      if act[I+1][K]>0:
        if sfactor>1.0:
          constraint_senses.append("G")
        else:
          constraint_senses.append("L")
      else:
        if sfactor>1.0:
          constraint_senses.append("L")
        else:
          constraint_senses.append("G")
      constraint_names.append("relu value change:"+"x_"+str(I+1)+"_"+str(K))


  ###### solve
  try:
    problem=cplex.Cplex()
    problem.variables.add(obj = objective,
                          lb = lower_bounds,
                          ub = upper_bounds,
                          names = var_names)
    problem.linear_constraints.add(lin_expr=constraints,
                                   senses = constraint_senses,
                                   rhs = rhs,
                                   names = constraint_names)
    problem.solve()

    ####
    d=problem.solution.get_values("d")
    new_x=[]
    for i in range(0, len(X)):
      v=(problem.solution.get_values('x_0_'+str(i)))
      if v<0 or v>1: return False, _, _
      new_x.append(v)

    if d==0 or d==1:      
      return False, _, _

    return True, new_x, d

  except:
    return False,[],-1
    
    try:
      d=problem.solution.get_values("d")
      print 'd is {0}'.format(d)
      new_x=[]
      #for i in len(X):
      #  new_x.append(problem.solution.get_values('x_0_'+str(i)))
      #return True, new_x, d
    except:
      print 'Exception for feasible model???'
      sys.exit(0)

def rp_dvc(I, J, nnet, X, act, sfactor):

  var_names=['d']
  objective=[1]
  lower_bounds=[0.0]
  upper_bounds=[1.0]
  
  N=len(act) # #layers
  for i in range(0, N):
    M=len(act[i]) # #neurons at layer i
    for j in range(0, M):
      var_names.append('x_'+str(i)+'_'+str(j))
      objective.append(0)
      lower_bounds.append(-cplex.infinity)
      upper_bounds.append(cplex.infinity)

 
  constraints=[]
  rhs=[]
  constraint_senses=[]
  constraint_names=[]


  for i in range(0, len(X)):
    # x<=x0+d
    constraints.append([[0, i+1], [-1, 1]])
    rhs.append(X[i])
    constraint_senses.append("L")
    constraint_names.append("x<=x"+str(i)+"+d")
    # x>=x0-d
    constraints.append([[0, i+1], [1, 1]])
    rhs.append(X[i])
    constraint_senses.append("G")
    constraint_names.append("x>=x"+str(i)+"-d")
    # x<=1
    constraints.append([[i+1], [1]])
    rhs.append(1.0)
    constraint_senses.append("L")
    constraint_names.append("x<=1")
    # x>=0
    constraints.append([[i+1], [1]])
    rhs.append(0.0)
    constraint_senses.append("G")
    constraint_names.append("x>=0")

  # there is nothing to constrain for layer 0
  # and we start from layer 1
  # the last layer shall be handled individually
  for i in range(1, I+1):
    M=len(act[i]) # number of neurons at layer i 
    for j in range(0, M): 
      #### for layer (I+1) we only need to access one neuron
      if i==I and j!=J: continue
      constraint=[[],[]] 
      constraint[0].append("x_"+str(i)+"_"+str(j))
      constraint[1].append(-1)
      for k in range(0, len(act[i-1])):
        constraint[0].append("x_"+str(i-1)+"_"+str(k))
        if i==1 or act[i-1][k]>0:
           constraint[1].append(nnet.weights[i-1][k][j])
        else:
           constraint[1].append(0)
      constraints.append(constraint)
      rhs.append(-nnet.biases[i][j])
      constraint_senses.append("E")
      constraint_names.append("eq:"+"x_"+str(i)+"_"+str(j))

      ###### ReLU
      if i<N-1:
        _constraint=[[],[]]
        _constraint[0].append("x_"+str(i)+"_"+str(j))
        _constraint[1].append(1)
        constraints.append(_constraint)
        if not(i==I and j==J):
          rhs.append(0)
          if act[i][j]>0:
            constraint_senses.append("G")
          else:
            constraint_senses.append("L")
          constraint_names.append("relu:"+"x_"+str(i)+"_"+str(j))
        else: ## I+1, K
          ## ReLU sign does not change
          rhs.append(0)
          if act[i][j]>0:
            constraint_senses.append("G")
          else:
            constraint_senses.append("L")
          constraint_names.append("relu:"+"x_"+str(i)+"_"+str(j))

          ## ReLU value changed
          _constraint=[[],[]]
          _constraint[0].append("x_"+str(i)+"_"+str(j))
          _constraint[1].append(1)
          constraints.append(_constraint)
          rhs.append(sfactor*act[I][J])
          if act[i][j]>0:
            if sfactor>1.0:
              constraint_senses.append("G")
            else:
              constraint_senses.append("L")
          else:
            if sfactor>1.0:
              constraint_senses.append("L")
            else:
              constraint_senses.append("G")
          constraint_names.append("relu value change:"+"x_"+str(i)+"_"+str(j))

  if I==N-1: # I+1==N-1
    #### Now, we are at the output layer
    #### x_{N-1, K}>=x_{N-1,old_label}
    label=np.argmax(act[N-1])
    for i in range(0, len(act[N-1])):
      if i!=J: continue
      constraint=[[],[]] 
      constraint[0].append("x_"+str(N-1)+"_"+str(i))
      constraint[1].append(1)
      constraints.append(constraint)

      ##1) ReLU sign does not change
      rhs.append(0)
      if act[I][J]>0:
        constraint_senses.append("G")
      else:
        constraint_senses.append("L")
      constraint_names.append("relu sign:"+"x_"+str(I)+"_"+str(J))

      ## ReLU value changed
      _constraint=[[],[]]
      _constraint[0].append("x_"+str(I)+"_"+str(J))
      _constraint[1].append(1)
      constraints.append(_constraint)
      rhs.append(sfactor*act[I][J])
      if act[I][J]>0:
        if sfactor>1.0:
          constraint_senses.append("G")
        else:
          constraint_senses.append("L")
      else:
        if sfactor>1.0:
          constraint_senses.append("L")
        else:
          constraint_senses.append("G")
      constraint_names.append("relu value change:"+"x_"+str(I)+"_"+str(J))


  ###### solve
  try:
    problem=cplex.Cplex()
    problem.variables.add(obj = objective,
                          lb = lower_bounds,
                          ub = upper_bounds,
                          names = var_names)
    problem.linear_constraints.add(lin_expr=constraints,
                                   senses = constraint_senses,
                                   rhs = rhs,
                                   names = constraint_names)
    problem.solve()

    ####
    d=problem.solution.get_values("d")
    new_x=[]
    for i in range(0, len(X)):
      v=(problem.solution.get_values('x_0_'+str(i)))
      if v<0 or v>1: return False, _, _
      new_x.append(v)

    if d==0 or d==1:      
      return False, _, _

    return True, new_x, d

  except:
    return False,[],-1
    
    try:
      d=problem.solution.get_values("d")
      print 'd is {0}'.format(d)
      new_x=[]
      #for i in len(X):
      #  new_x.append(problem.solution.get_values('x_0_'+str(i)))
      #return True, new_x, d
    except:
      print 'Exception for feasible model???'
      sys.exit(0)

