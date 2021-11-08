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





def test(args):
    return args




if __name__ == '__main__':
    reps = range(100)
    pool = Pool()                         # Create a multiprocessing Pool
    results = pool.map(tools.fusion_for_MP, reps)
    
    FM_by_dim = {}
    MSEs_seen_by_dim = {}
    MSEs_unseen_by_dim = {}
    
    
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
        
        