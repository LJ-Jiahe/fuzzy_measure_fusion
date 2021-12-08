# Imports

import datetime
# import itertools
from itertools import permutations
import math
from multiprocessing import Pool
# import os
import pickle
import platform
import random
import sys
from tkinter import Tk

# from cvxopt import solvers, matrix
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import torch
# import torchvision
# from torchvision import transforms, models,datasets
# from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# from tools import *
import tools


if __name__ == '__main__':
    rep = int(sys.argv[1])
    max_Num_Source = int(sys.argv[2])

    reps = range(rep)
    max_N_Ss = (np.ones(rep) * max_Num_Source).astype(int)

    packed_params = [i for i in zip(reps, max_N_Ss)]

    if len(sys.argv >= 4):
        pool = Pool(int(sys.argv[3])) # Create a multiprocessing Pool
    else:
        pool = Pool()
        
    results = pool.starmap(tools.fusion_for_MP, packed_params)
    
    MSEs_seen_by_dim = {}
    MSEs_unseen_by_dim = {}
    FM_by_dim = {}
    
    for result in results:
        for k in result[0].keys():
            if k in FM_by_dim.keys():
                FM_by_dim[k] = np.append(FM_by_dim[k], result[0][k], axis=0)
                MSEs_seen_by_dim[k] = np.append(MSEs_seen_by_dim[k], result[1][k], axis=0)
                MSEs_unseen_by_dim[k] = np.append(MSEs_unseen_by_dim[k], result[2][k], axis=0)
            else:
                FM_by_dim[k] = result[0][k]
                MSEs_seen_by_dim[k] = result[1][k]
                MSEs_unseen_by_dim[k] = result[2][k]

    output_dir = 'output/'
    now = datetime.datetime.now().strftime("-%m-%d-%Y@%H.%M.%S")
    with open(output_dir + 'ChI_saved_file' + now, 'wb') as f:
        pickle.dump(FM_by_dim, f)
        pickle.dump(MSEs_seen_by_dim, f)
        pickle.dump(MSEs_unseen_by_dim, f)
        
        