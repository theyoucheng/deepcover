
import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 5
n8 = (0.35220126,0.49865229,0.1410602, 0.00808625, 0)
n9 = (0.37142857,0.59548872,0.03308271, 0, 0)
n10 = (0.27193619,0.38941262,0.24655547,0.08919507,  0.00290065)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8
 
rect_n8 = plt.bar(index+2*bar_width, n8, bar_width,
                 alpha=opacity,
                 color='b',
                 label='$\mathcal{N}_8$')

rect_n9 = plt.bar(index + 0*bar_width, n9, bar_width,
                 alpha=opacity,
                 color='g',
                 label='$\mathcal{N}_9$')
 
rect_n10 = plt.bar(index + 1*bar_width, n10, bar_width,
                 alpha=opacity,
                 color='red',
                 label='$\mathcal{N}_{10}$')

 
plt.xlabel('Adjacent layers')
plt.ylabel('Adversarial examples')
#plt.title('Scores by person')
plt.xticks(index + bar_width, ('$L2-3$', '$L3-4$', '$L4-5$', '$L5-6$', '$L6-7$'))
plt.legend()
 
plt.tight_layout()
#plt.show()
plt.savefig("layerwise-ss-bugs.pdf", bbox_inches='tight')

