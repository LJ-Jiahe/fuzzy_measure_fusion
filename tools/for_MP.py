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
from tqdm import tqdm

# from tools import *
import tools


def fusion_for_MP(rep):
    dim_list = list(range(3, 6))

    # num_per_perm_list_train = [30]
#     num_per_perm_list_train = [5]
    num_per_perm_list_train = [1, 10, 50, 100] # Every permutation gets the same number of train/test data samples
    num_per_perm_list_test = [10] # Every permutation gets the same number of train/test data samples

    superset_factor = 4 # Create a superset # times larger than what is needed to ensure each class would have sufficient number of samples

    distributions = ['uniform', 'normal', 'bimodal']
    distribution = distributions[0]

    weights = {3: np.asarray([[0.8, 0.1, 0.1],
                              [0.2, 0.7, 0.1],
                              [0.5, 0.2, 0.3],
                              [0.1, 0.3, 0.5]]),
               4: np.asarray([[0.1, 0.2, 0.3, 0.4],
                              [0.2, 0.2, 0.2, 0.4],
                              [0, 0.1, 0.3, 0.6],
                              [0, 0, 0.4, 0.6]]),
               5: np.asarray([[0.1, 0.1, 0.1, 0.2, 0.5],
                              [0, 0, 0.4, 0.3, 0.3],
                              [0, 0, 0, 0.4, 0.6],
                              [0.1, 0.8, 0.1, 0, 0]]),
               6: np.asarray([[0.1, 0.1, 0.1, 0.1, 0.2, 0.4],
                              [0, 0, 0.3, 0.3, 0.3, 0.1],
                              [0, 0, 0, 0, 0.4, 0.6],
                              [0.1, 0.7, 0.1, 0.1, 0, 0]]),}

    avg_funcs = [np.amin, np.amax, np.mean, None, None, None, None] # List of avg functions

    avg_names = {}
    for dim in dim_list:
        weight = weights[dim]
        weight_legend = []
        for w in weight:
            weight_legend.append(' '.join(map(str, (w*10).astype(int))))
        avg_names[dim] = ['Min', 'Max', 'Mean'] + weight_legend

    train_group_num_limit = math.factorial(4)

    models = [tools.Choquet_Integral_QP]
    model_names = ['QP']

    output_dir = 'output/'
    
    
    
    
    
    MSEs_seen_by_dim = {}
    MSEs_unseen_by_dim = {}
    FM_by_dim = {}
    
    
    
    train_data_source = tools.Data_Source(dim_list[-1], num_per_perm_list_train[-1], superset_factor, distribution)  
    train_superset = train_data_source.get_superset()

    test_data_source = tools.Data_Source(dim_list[-1], num_per_perm_list_test[-1], superset_factor, distribution)  
    test_superset = test_data_source.get_superset()
    
    random_seed = random.random()
    
    for dim in dim_list:

        for avg_idx in range(len(weights[dim])):
            avg_funcs[3+avg_idx] = tools.w_avg(weights[dim][avg_idx])

        all_perms = list(permutations(list(range(dim)))) # N! possible permutations
        random.Random(rep*random_seed).shuffle(all_perms)
#         random.shuffle(all_perms)
        
        # When the # of possible permutations exceed certain number (in here 5!), 
        # instead of feeding only one more permutation a time, feed more.
        if len(all_perms) > train_group_num_limit:
            step = int(len(all_perms) / train_group_num_limit)
        else:
            step = 1

        # Mean Squared Error [for each repetition, for each avg function, for each model, for each percentage, for each data#perperm, for each dim], of all test samples, for both seen and unseen data.
        MSEs_seen = np.zeros((len(num_per_perm_list_train), len(range(step-1, len(all_perms), step)), len(avg_funcs), len(models)))
        MSEs_unseen = np.zeros((len(num_per_perm_list_train), len(range(step-1, len(all_perms), step))-1, len(avg_funcs), len(models)))
        # Record FM after train session with both seen and unseen data pattern
        FM = np.zeros((len(num_per_perm_list_train), len(range(step-1, len(all_perms), step)), len(avg_funcs), len(models), 2**dim-1))

    
        for npp_idx, num_per_perm in enumerate(num_per_perm_list_train):
            
            train_idx_by_perm = train_data_source.get_data_idx(dim, num_per_perm)
            test_idx_by_perm = test_data_source.get_data_idx(dim, num_per_perm_list_test[0])
            for perc_idx, perc in enumerate(tqdm(range(step-1, len(all_perms), step))):
                
#             for perc_idx, perc in enumerate(range(step-1, len(all_perms), step)):
                # Find index of train/test sample in superset and shuffle
                train_idx = np.concatenate(train_idx_by_perm[0:perc+1])
                np.random.shuffle(train_idx)
                test_idx = np.concatenate(test_idx_by_perm[0:perc+1])
                # Find data sample through index
                train_d = train_superset[:, train_idx][0:dim]
                test_d = test_superset[:, test_idx][0:dim]
                # Define unseen test data samples when the train data doesn't cover 100% of the permutation
                if perc < len(all_perms)-1:
                    test_idx_unseen = np.concatenate(test_idx_by_perm[perc+1:])
                    test_d_unseen = test_superset[:, test_idx_unseen][0:dim]
                else:
                    test_d_unseen = []

                # Define subsets of 'X', or keys for fuzzy measure. Like '1 2' or '1 3 4 5' for g(x1, x2) or g(x1, x3, x4, x5)
                sourcesInNode, subset = tools.sources_and_subsets_nodes(dim)
                keys = [str(np.sort(i)+1) for i in sourcesInNode]
                
                for avg_idx, avg_func in enumerate(avg_funcs):
                    # Calculate label with given avg function
                    train_label = avg_func(train_d, 0)
                    test_label = avg_func(test_d, 0)
                    
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
                            chi_model = model(dim, 1)
                            # Parameters for training NN
                            lr = 0.05 # Learning rate
                            num_epoch = 100
                            criterion = torch.nn.MSELoss(reduction='mean')
                            optimizer = torch.optim.SGD(chi_model.parameters(), lr=lr)
                            
                            # Train 
                            tools.train_chinn(chi_model, lr, criterion, optimizer, num_epoch, torch.tensor(train_d, dtype=torch.float), torch.tensor(train_label, dtype=torch.float))
                            # Get fuzzy measure learned
                            FM_learned = (chi_model.chi_nn_vars(chi_model.vars).cpu()).detach().numpy()
                            fm_dict_binary = dict(zip(keys, FM_learned[:,0]))
                            fm_dict_lexicographic = tools.get_keys_index(dim)
                            for key in fm_dict_lexicographic.keys():
                                fm_dict_lexicographic[key] = fm_dict_binary[key]
                            fm = fm_dict_lexicographic
                            

                        
                        FM[npp_idx, perc_idx, avg_idx, model_idx, :] = np.asarray(list(fm.values()))
                        # Calculate result from integral with test data
                        test_output = np.apply_along_axis(tools.get_cal_chi(fm), 0, test_d)
                        MSE = ((test_output - test_label)**2).mean()
                        MSEs_seen[npp_idx, perc_idx, avg_idx, model_idx] = MSE
                        # Calculate result from integral with test data - unseen
                        if perc < len(all_perms)-1:
                            test_label_unseen = avg_func(test_d_unseen, 0)
                            test_out_unseen = np.apply_along_axis(tools.get_cal_chi(fm), 0, test_d_unseen)
                            MSEs_unseen[npp_idx, perc_idx, avg_idx, model_idx] = ((test_out_unseen - test_label_unseen)**2).mean()
                            
        if dim in FM_by_dim.keys():
            FM_by_dim[dim] = np.append(FM_by_dim[dim], np.expand_dims(FM, axis=0), axis=0)
            MSEs_seen_by_dim[dim] = np.append(MSEs_seen_by_dim[dim], np.expand_dims(MSEs_seen, axis=0), axis=0)
            MSEs_unseen_by_dim[dim] = np.append(MSEs_unseen_by_dim[dim], np.expand_dims(MSEs_unseen, axis=0), axis=0)
        else:
            FM_by_dim[dim] = np.expand_dims(FM, axis=0)
            MSEs_seen_by_dim[dim] = np.expand_dims(MSEs_seen, axis=0)
            MSEs_unseen_by_dim[dim] = np.expand_dims(MSEs_unseen, axis=0)
    print('Rep ' + str(rep) + ' done.')
    return FM_by_dim, MSEs_seen_by_dim, MSEs_unseen_by_dim