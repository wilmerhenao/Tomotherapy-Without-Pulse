
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

voxels = [59,97,131,329,671,2102,5000,8410]
seconds = [12.11,6.85,79,43,35.94,402,276,1600]

plt.plot(voxels, seconds)
plt.set_title('')
plt.show()