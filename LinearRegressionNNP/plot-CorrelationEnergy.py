import os
import numpy as np
import matplotlib.pyplot as plt

file = 'n2p2_energy_frames.out'

#read_file = np.fromstring(np.loadtxt(file))
read_file = np.loadtxt(file)

fig, ax = plt.subplots(dpi = 150)

#ax.errorbar(read_file[:, 0], read_file[:, 2], yerr = read_file[:, 3]/np.sqrt(31), fmt= 'o', markersize = 3, capsize = 3, color = 'b', label = 'Committee of NNs')
plt.plot(read_file[:, 1], read_file[:, 0], 'o', markersize = 7, color = 'g', label = 'N2P2')
#lims = [
#    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
#]

# now plot both limits against eachother
#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
#ax.set_aspect('equal')
#ax.set_xlim(lims)
#ax.set_ylim(lims)

plt.xticks(fontsize = 7)
plt.yticks(fontsize = 7)
plt.xlabel('True Energy (hartree)', fontsize = 11)
plt.ylabel('Predicted Energy (hartree)', fontsize = 11)
#plt.title('Test Set of 161 Structures', fontsize = 12)
plt.legend()
plt.tight_layout()
plt.show()
