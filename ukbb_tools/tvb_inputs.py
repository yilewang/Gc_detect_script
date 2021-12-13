import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#f = open("/home/wayne/tvb_inputs/sc.txt", "r")
f = open("C:/Users/Wayne/tvb/s123873_output/dMRI/distance.txt", "r")
sc = np.loadtxt(f)
ax = sns.heatmap(sc)
ax.set_title('distance.txt')
plt.show()