# Imports

# import itertools
from itertools import permutations
import math
# import os
import pickle
import platform
import random
import timeit
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
from tqdm.auto import tqdm

# from tools import *
from tools.cho_integral import get_cal_chi, Choquet_Integral_NN, train_chinn, Choquet_Integral_QP, init_FM
from tools.data_source import create_dataset
from tools.process_bar import pbiter
from tools.w_avg import weighted_avg

import sys
sys.path.append('..')
import constants as c


def fusion(rep):

    ################################################################################
    # Part 1 - Parameters Init <START>
    num_source_list = c.num_source_list.copy() # Number of sources to be fused, a to b-1
    num_per_perm_list_train = c.num_per_perm_list_train.copy() # Each permutation gets the same number of samples, try different values here for train set
    num_per_perm_list_test = c.num_per_perm_list_test.copy() # Each permutation gets the same number of samples

    distributions = c.distributions.copy()

    weights = c.weights.copy()

    avg_funcs = c.avg_funcs.copy() # List of avg functions

    avg_names = c.avg_names.copy()

    train_group_num_limit = c.train_group_num_limit

    models = c.models.copy()
    model_names = c.model_names.copy()
    # Part 1 - Parameters Init <END>
    ################################################################################
    
    

    ################################################################################
    # Part 2 - Run <START>
    # Below are where we are going to store the results from this very repetition
    MSEs_seen_by_num_source = {}
    MSEs_unseen_by_num_source = {}
    FMs_by_num_source = {}

    pbar = tqdm(total=len(num_source_list) * 
                      len(distributions) * 
                      len(num_per_perm_list_train))

    ################################################################################
    # For Loop 1
    for num_source in num_source_list:
        # Shuffle the order of permutations fed to model in train session
        all_perms = list(permutations(list(range(num_source))))
        random.shuffle(all_perms) # or random.Random(random_seed).shuffle(all_perms)

        # Switch out arbitrary avg funcs to new num_source
        for avg_idx, weight_set in enumerate(weights[num_source]):
            avg_funcs[3+avg_idx] = weighted_avg(weight_set)

        # When the # of possible permutations exceed certain number (in here 5!), 
        # instead of feeding only one more permutation at a time, feed more.
        num_perms = math.factorial(num_source)
        if num_perms > train_group_num_limit:
            step = int(num_perms / train_group_num_limit)
        else:
            step = 1

        # Save a MSE value for each posibble condition
        MSEs_seen = np.zeros((len(distributions),
                              len(num_per_perm_list_train), 
                              len(range(step-1, num_perms, step)), 
                              len(avg_funcs), 
                              len(models)))
        MSEs_unseen = np.zeros((len(distributions),
                                len(num_per_perm_list_train), 
                                len(range(step-1, num_perms, step))-1, 
                                len(avg_funcs), 
                                len(models)))
        # Record FMs after train session
        FMs = np.zeros((len(distributions),
                        len(num_per_perm_list_train), 
                        len(range(step-1, num_perms, step)), 
                        len(avg_funcs), 
                        len(models), 
                        2**num_source-1))

        # Define subsets of 'X', or keys for fuzzy measure. 
        # Like '1 2' or '1 3 4 5' for g(x1, x2) or g(x1, x3, x4, x5)
        keys = list(init_FM(num_source).keys())

        ################################################################################
        # For Loop 2
        for dist_idx, distribution in enumerate(distributions):

            ################################################################################
            # For Loop 3
            for npp_idx, num_per_perm_train in enumerate(num_per_perm_list_train):
                # Get data_by_perm based on num_source & num_per_perm
                train_data_by_perm = \
                    create_dataset(num_source, num_per_perm_train, distribution, all_perms)
                test_data_by_perm = \
                    create_dataset(num_source, num_per_perm_list_test[0], distribution, all_perms)
                
                pbar.update(1)

                ################################################################################
                # For Loop 4
                for perc_idx, perc in enumerate(range(step-1, num_perms, step)):
                    # Find data sample through index
                    train_d = train_data_by_perm[0:num_per_perm_train*(perc+1), :]
                    np.random.shuffle(train_d)
                    test_d = test_data_by_perm[0:num_per_perm_list_test[0]*(perc+1), :]
                    test_d_unseen = test_data_by_perm[num_per_perm_list_test[0]*(perc+1):, :]
                    
                    ################################################################################
                    # For Loop 5
                    for avg_idx, avg_func in enumerate(avg_funcs):
                        # Calculate label with given avg function
                        train_label = avg_func(train_d, 1)
                        test_label = avg_func(test_d, 1)
                        if perc < num_perms-1:
                                test_label_unseen = avg_func(test_d_unseen, 1)
                        ################################################################################
                        # For Loop 6
                        for model_idx, model in enumerate(models):
                            if model_names[model_idx] == 'QP':
                                # Initialize ChIQP
                                chi_model = model()
                                # Train 
                                chi_model.train_chi(train_d, train_label)
                                # Get fuzzy measure learned
                                fm = chi_model.fm
                                
                            elif model_names[model_idx] == 'NN':
                                # Initialize ChINN
                                chi_model = model(num_source, 1)
                                # Parameters for training NN
                                lr = 0.05 # Learning rate
                                num_epoch = 100
                                criterion = torch.nn.MSELoss(reduction='mean')
                                optimizer = torch.optim.Adam(chi_model.parameters(), lr=lr)
                                
                                # Train 
                                train_chinn(chi_model, 
                                            lr, 
                                            criterion, 
                                            optimizer, 
                                            num_epoch, 
                                            torch.tensor(train_d, dtype=torch.float), 
                                            torch.tensor(train_label, dtype=torch.float))
                                # Get fuzzy measure learned
                                FMs_learned = (chi_model.chi_nn_vars(chi_model.vars).cpu()).detach().numpy()
                                fm_dict_binary = dict(zip(keys, FMs_learned[:,0]))
                                fm_dict_lexicographic = init_FM(num_source)
                                for key in fm_dict_lexicographic.keys():
                                    fm_dict_lexicographic[key] = fm_dict_binary[key]
                                fm = fm_dict_lexicographic
                            
                            FMs[dist_idx, npp_idx, perc_idx, avg_idx, model_idx, :] = np.asarray(list(fm.values()))
                            # Calculate result from integral with test data
                            test_output = np.apply_along_axis(get_cal_chi(fm), 1, test_d)
                            MSE = ((test_output - test_label)**2).mean()
                            MSEs_seen[dist_idx, npp_idx, perc_idx, avg_idx, model_idx] = MSE
                            # Calculate result from integral with test data - unseen
                            if perc < num_perms-1:
                                test_out_unseen = np.apply_along_axis(get_cal_chi(fm), 1, test_d_unseen)
                                MSE = ((test_out_unseen - test_label_unseen)**2).mean()
                                MSEs_unseen[dist_idx, npp_idx, perc_idx, avg_idx, model_idx] = MSE


        FMs_by_num_source[num_source] = FMs
        MSEs_seen_by_num_source[num_source] = MSEs_seen
        MSEs_unseen_by_num_source[num_source] = MSEs_unseen


            # if num_source in FMs_by_num_source.keys():
            #     FMs_by_num_source[num_source] = np.append(FMs_by_num_source[num_source], np.expand_dims(FMs, axis=0), axis=0)
            #     MSEs_seen_by_num_source[num_source] = np.append(MSEs_seen_by_num_source[num_source], np.expand_dims(MSEs_seen, axis=0), axis=0)
            #     MSEs_unseen_by_num_source[num_source] = np.append(MSEs_unseen_by_num_source[num_source], np.expand_dims(MSEs_unseen, axis=0), axis=0)
            # else:
            #     FMs_by_num_source[num_source] = np.expand_dims(FMs, axis=0)
            #     MSEs_seen_by_num_source[num_source] = np.expand_dims(MSEs_seen, axis=0)
            #     MSEs_unseen_by_num_source[num_source] = np.expand_dims(MSEs_unseen, axis=0)
    # Part 2 - Run <END>
    ################################################################################


    print('Rep ' + str(rep) + ' done.')
   
    return FMs_by_num_source, MSEs_seen_by_num_source, MSEs_unseen_by_num_source