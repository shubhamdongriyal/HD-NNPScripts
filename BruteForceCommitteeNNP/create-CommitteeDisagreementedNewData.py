# this script generate a file called add-input.data
# select structures which showed highest disagreement on committee predicted energies 
# this add-input.data you add to retrain the neural network potential

import os
import shutil
import numpy as np
import glob

NOFIDX = int(input("Number of structures to add for training"))


INPUT_PATH = '/u/shubsharma/HDNNP/Naphthalene-Multi/predictions/testing-newstyle/selection/'

committee = np.loadtxt(glob.glob('sorted-committee*')[0], usecols = 0, skiprows=0).astype(np.int64)

idxs = np.ndarray.tolist(committee[:NOFIDX])
print(idxs)


with open('add-input.data','wb') as wfd:
    for f in idxs:
        with open(os.path.join(INPUT_PATH, f"frame.{f}", "input.data"),'rb') as fd:
            shutil.copyfileobj(fd, wfd)
