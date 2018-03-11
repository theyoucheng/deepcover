
import matplotlib.pyplot as plt
import numpy as np
import csv

"""
Simple demo of a scatter plot.
"""
import numpy as np
import matplotlib.pyplot as plt

results=['8-ss-results.txt', '9-ss-results.txt', '10-ss-results.txt']

x=np.arange(0.01, 0.31, 0.01)


tot8=0
vect8=np.zeros(30)
tot9=0
vect9=np.zeros(30)
tot10=0
vect10=np.zeros(30)

pre_nex=0
with open('8-ss-results.txt','r') as csvfile:
  plots = csv.reader(csvfile, delimiter=',')
  for row in plots:
    nex=int(row[3].split('=')[1])
    if nex>pre_nex:
      pre_nex=nex
      tot8+=1
      d=float(row[6].split('=')[1])
      for i in range(0, 30):
        if d<(i+1)*0.01: vect8[i]+=1
for i in range(0, 30):
  vect8[i]=1.0*vect8[i]/tot8

pre_nex=0
with open('9-ss-results.txt','r') as csvfile:
  plots = csv.reader(csvfile, delimiter=',')
  for row in plots:
    nex=int(row[3].split('=')[1])
    if nex>pre_nex:
      pre_nex=nex
      tot9+=1
      d=float(row[6].split('=')[1])
      for i in range(0, 30):
        if d<(i+1)*0.01: vect9[i]+=1
for i in range(0, 30):
  vect9[i]=1.0*vect9[i]/tot9

pre_nex=0
with open('10-ss-results.txt','r') as csvfile:
  plots = csv.reader(csvfile, delimiter=',')
  for row in plots:
    nex=int(row[3].split('=')[1])
    if nex>pre_nex:
      pre_nex=nex
      tot10+=1
      d=float(row[6].split('=')[1])
      for i in range(0, 30):
        if d<(i+1)*0.01: vect10[i]+=1
for i in range(0, 30):
  vect10[i]=1.0*vect10[i]/tot10


#plt.axis([0, 11, -0.5, +0.5])
plt.plot(x,vect10, '--', label='$\mathcal{N}_{10}$')
plt.plot(x,vect8, '--', label='$\mathcal{N}_8$')
plt.plot(x,vect9, '--', label='$\mathcal{N}_9$')

plt.xlabel('Input distance')
plt.ylabel('Accumulated adversarial examples')
plt.legend()
plt.legend(loc="lower right")
#plt.show()
plt.savefig("ss-distance-map.pdf", bbox_inches='tight')

    

