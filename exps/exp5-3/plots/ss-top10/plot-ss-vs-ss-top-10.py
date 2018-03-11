
import matplotlib.pyplot as plt
import numpy as np
import csv

"""
Simple demo of a scatter plot.
"""
import numpy as np
import matplotlib.pyplot as plt


N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors1 = np.random.rand(N)
colors2 = np.random.rand(N)
area = np.pi *5**2 #* (15 * np.random.rand(2))**2  # 0 to 15 point radii


nns=[]
ss1=[]
ae1=[]
ss2=[]
ae2=[]

ss1m2=[]
ae1m2=[]
  
with open('ss-vs-ss-top-10.csv','r') as csvfile:
  plots = csv.reader(csvfile, delimiter=',')
  for row in plots:
    print row
    nns.append(row[0])
    ss1.append(row[1])
    ae1.append(row[2])
    ss2.append(row[3])
    ae2.append(row[4])

    ss1m2.append(float(row[1])-float(row[3]))
    ae1m2.append(float(row[2])-float(row[4]))

plt.axis([0, 11, -0.1, +0.1])

plt.plot([0, 11],[0, 0], '--', alpha=0.5)
plt.scatter(nns, ss1m2, s=area, color='red', alpha=0.25, label='$Mcov_{SS}-Mcov_{SS}^{w10}$')
plt.scatter(nns, ae1m2, s=area, color='green', alpha=0.25, label='$AEcov_{SS}-AE_{SS}^{w10}$')

#
plt.xlabel('DNN index')
plt.ylabel('Difference in testing results')
plt.legend()
plt.savefig("ss-top10.pdf", bbox_inches='tight')

