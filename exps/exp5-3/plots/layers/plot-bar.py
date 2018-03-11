
import numpy as np
import matplotlib.pyplot as plt
 
# data to plot
n_groups = 5
n8 = (1, 1, 0.99621101, 0.96808511, 0)
n9 = (1, 1, 1, 0, 0)
n10 = (1, 1, 0.9783508, 0.64231293, 0.524)
 
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8
 
rect_n8 = plt.bar(index+2*bar_width, n8, bar_width,
                 alpha=opacity,
                 color='b',
                 label='$\mathcal{N}_{8}$')

rect_n9 = plt.bar(index + 0*bar_width, n9, bar_width,
                 alpha=opacity,
                 color='g',
                 label='$\mathcal{N}_9$')
 
rect_n10 = plt.bar(index + 1*bar_width, n10, bar_width,
                 alpha=opacity,
                 color='red',
                 label='$\mathcal{N}_{10}$')

 
plt.xlabel('Adjacent layers')
plt.ylabel('Layerwise SS coverage')
#plt.title('Scores by person')
plt.xticks(index + bar_width, ('$L2-3$', '$L3-4$', '$L4-5$', '$L5-6$', '$L6-7$'))
plt.legend()
 
plt.tight_layout()
#plt.show()
plt.savefig("layerwise-ss-coverage.pdf", bbox_inches='tight')

