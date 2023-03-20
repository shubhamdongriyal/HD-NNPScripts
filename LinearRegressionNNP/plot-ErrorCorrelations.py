import numpy as np
import matplotlib.pyplot as plt

DATA = np.loadtxt('/Users/shubhamdongriyal/Desktop/committee_mean_std_trained-9812345-10000.txt', usecols=(0, 1, 2, 3))

plt.figure(figsize=(8, 4), dpi = 150)
plt.subplot(1, 3, 1)
plt.scatter(np.abs(DATA[:, 0] - DATA[:, 2]), DATA[:, 3]/np.sqrt(31), s=2, c='k')
plt.xlabel('true - committee mean')
plt.ylabel('committee std error')

plt.subplot(1, 3, 2)
plt.scatter(np.abs(DATA[:, 0] - DATA[:, 1]), DATA[:, 3]/np.sqrt(31), s=2, c='k')
plt.xlabel('true - n2p2')
plt.ylabel('committee std error')

plt.subplot(1, 3, 3)
plt.scatter(np.abs(DATA[:, 0] - DATA[:, 1]), DATA[:, 0] - DATA[:, 2], s=2, c='k')
plt.xlabel('true - n2p2')
plt.ylabel('true - committee mean')
plt.tight_layout()
plt.show()