import os
import time
import pathlib
import shutil
import numpy as np
import datetime

PATH = '/u/shubsharma/HDNNP/Naphthalene-Multi/predictions/testing-newstyle/mulpred/predictions.txt'
ANALYSIS = '/u/shubsharma/HDNNP/Naphthalene-Multi/predictions/testing-newstyle/analysis/'
predictions = np.loadtxt(PATH)

p = pathlib.Path(PATH)

timestamp = datetime.datetime.fromtimestamp(p.stat().st_ctime).isoformat()

os.chdir(ANALYSIS)
os.mkdir(timestamp)
shutil.copy(PATH, timestamp)
os.chdir(timestamp)


diff = (predictions -  predictions[:,0].reshape(-1, 1))[: ,1: ]
np.savetxt(f'true-n2p2_predictions-{timestamp}.txt', diff)


frame = (np.array(range(predictions.shape[0])).reshape(-1, 1) + 1)

mean = np.mean(predictions[:, 1:], axis = 1).reshape(-1, 1)

committee = np.concatenate((frame, predictions[:, 0].reshape(-1, 1), mean, predictions[:, 0].reshape(-1, 1) - mean), axis = 1)
np.savetxt(f'committee-disagreement-{timestamp}.txt', committee, delimiter='\t', newline='\n', comments='#', header=f'frame\ttrue_energy\tcommittee_mean\tcommittee_disagreement', fmt = '%.8f')

np.savetxt(f'sorted-committee-disagreement-{timestamp}.txt', committee[np.abs(committee[:, 3]).argsort()[::-1]], delimiter='\t', newline='\n', comments='#', header=f'frame\ttrue_energy\tcommittee_mean\tcommittee_disagreement', fmt = '%.8f')
