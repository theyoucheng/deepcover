
import matplotlib.pyplot as plt
import numpy as np
import csv

import numpy as np
import matplotlib.pyplot as plt

x=np.arange(3, 6, 1)

n20_10=0
ae20_10=0
co20_10=0
#
n20_11=0
ae20_11=0
co20_11=0
#
n21_10=0
ae21_10=0
co21_10=0
#
n20_11=0
ae20_11=0
co20_11=0
#
n21_11=0
ae21_11=0
co21_11=0


N20_10=0
AE20_10=0
CO20_10=0
N20_11=0
AE20_11=0
CO20_11=0
N21_10=0
AE21_10=0
CO21_10=0

with open('cnn2-results.txt','r') as csvfile:
  plots = csv.reader(csvfile, delimiter=',')
  for row in plots:
    locs=row[0].split()
    I=int(locs[0].split('-')[1])
    J=int(locs[1].split('-')[1])
    cex=(row[3]==' cex=True')
    covered=(row[1]==' True')
    if I==0 and J==0:
      n20_10+=1
      if cex: ae20_10+=1
      if covered: co20_10+=1
    elif I==0 and J==1:
      n21_10+=1
      if cex: ae21_10+=1
      if covered: co21_10+=1
    elif I==1 and J==0:
      n20_11+=1
      if cex: ae20_11+=1
      if covered: co20_11+=1
    elif I==1 and J==1:
      n21_11+=1
      if cex: ae21_11+=1
      if covered: co21_11+=1

print "CNN2: "
print '{0}, {1}, {2}'.format(n20_10, ae20_10, co20_10)
print 'cf2,1: cf1,1==> {0}, {1}'.format(1.0*co20_10/n20_10, 1.0*ae20_10/n20_10)
print '{0}, {1}, {2}'.format(n21_10, ae21_10, co21_10)
print 'cf2,2: cf1,1==> {0}, {1}'.format(1.0*co21_10/n21_10, 1.0*ae21_10/n21_10)
print '{0}, {1}, {2}'.format(n20_11, ae20_11, co20_11)
print 'cf2,1: cf1,2==> {0}, {1}'.format(1.0*co20_11/n20_11, 1.0*ae20_11/n20_11)
print '{0}, {1}, {2}'.format(n21_11, ae21_11, co21_11)
print 'cf2,2: cf1,2==> {0}, {1}'.format(1.0*co21_11/n21_11, 1.0*ae21_11/n21_11)


with open('cnn1-results.txt','r') as csvfile:
  plots = csv.reader(csvfile, delimiter=',')
  for row in plots:
    locs=row[0].split()
    I=int(locs[0].split('-')[1])
    J=int(locs[1].split('-')[1])
    cex=(row[3]==' cex=True')
    covered=(row[1]==' True')
    if I==0 and J==0:
      N20_10+=1
      if cex: AE20_10+=1
      if covered: CO20_10+=1
    elif I==1 and J==0:
      N20_11+=1
      if cex: AE20_11+=1
      if covered: CO20_11+=1
    elif I==0 and J==1:
      N21_10+=1
      if cex: AE21_10+=1
      if covered: CO21_10+=1

print "\nCNN1: "
print '{0}, {1}, {2}'.format(N20_10, AE20_10, CO20_10)
print 'cf2,1: cf1,1==> {0}, {1}'.format(1.0*CO20_10/N20_10, 1.0*AE20_10/N20_10)
print '{0}, {1}, {2}'.format(N21_10, AE21_10,CO21_10)
print 'cf2,2: cf1,1==> {0}, {1}'.format(1.0*CO21_10/N21_10, 1.0*AE21_10/N21_10)

