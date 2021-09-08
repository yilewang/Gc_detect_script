import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#f = open("/home/wayne/tvb_inputs/sc.txt", "r")
f = open("/home/wayne/tvb_inputs/distance.txt", "r")
sc = np.loadtxt(f)
ax = sns.heatmap(sc)
ax.set_title('sc.txt')
plt.show()