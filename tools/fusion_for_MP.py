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


def fusion_for_MP(rep, max_Num_Source):

    ################################################################################
    # Part 1 - Parameters Init
    ################################################################################

    ################################################################################
    # Data related parameters <START>

    num_source_list = list(range(3, max_Num_Source+1)) # Number of sources to be fused, a to b-1

    num_per_perm_list_train = [1, 10, 50, 100] # Each permutation gets the same number of samples, try different values here for train set
    num_per_perm_list_test = [10] # Each permutation gets the same number of samples

    distributions = ['uniform', 'Gaussian', 'polarized', 'random Gaussian'] # Use to discuss whether the distribution of data could have an impact on result

    weights = {3: np.asarray([[0.1, 0.8, 0.1],   # 1 large, 2 small, else 0
                              [0.0, 0.5, 0.5],   # 2 large, else 0
                              [0.3, 0.5, 0.2],   # 1 large = sum of else
                              [1/3, 1/3, 1/3]]), # 3 1/3 else 0
               4: np.asarray([[0.1, 0.8, 0.1, 0.0],
                              [0.0, 0.5, 0.5, 0.0],
                              [0.1, 0.5, 0.2, 0.2],
                              [1/3, 1/3, 1/3, 0.0]]),
               5: np.asarray([[0.1, 0.8, 0.1, 0.0, 0.0],
                              [0.0, 0.5, 0.5, 0.0, 0.0],
                              [0.1, 0.5, 0.1, 0.1, 0.2],
                              [1/3, 1/3, 1/3, 0.0, 0.0]]),
               6: np.asarray([[0.1, 0.8, 0.1, 0.0, 0.0, 0.0],
                              [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                              [0.1, 0.5, 0.1, 0.1, 0.1, 0.1],
                              [1/3, 1/3, 1/3, 0.0, 0.0, 0.0]]),
               7: np.asarray([[0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                              [0.1, 0.5, 0.1, 0.1, 0.1, .05, .05],
                              [1/3, 1/3, 1/3, 0.0, 0.0, 0.0, 0.0]]),
               8: np.asarray([[0.1, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.1, 0.5, 0.1, 0.1, .05, .05, .05, .05],
                              [1/3, 1/3, 1/3, 0.0, 0.0, 0.0, 0.0, 0.0]]),}

    avg_funcs = [np.amin, np.amax, np.mean, None, None, None, None] # List of avg functions

    avg_names = {} # List of names of avg functions
    for num_source in num_source_list:
        weight = weights[num_source]
        weight_legend = []
        for w in weight:
            weight_legend.append(' '.join(map(str, (w*10).astype(int))))
        avg_names[num_source] = ['Min', 'Max', 'Mean'] + weight_legend

    train_group_num_limit = math.factorial(5)

    # Data related parameters <END>
    ################################################################################


    ################################################################################
    # Other parameters <START>

    models = [tools.Choquet_Integral_QP]
    model_names = ['QP']

    # models = [tools.Choquet_Integral_QP, tools.Choquet_Integral_NN]
    # model_names = ['QP', 'NN']

    output_dir = 'output/'

    # Other parameters <END>
    ################################################################################
    
    

    ################################################################################
    # Part 2 - Run
    ################################################################################

    MSEs_seen_by_num_source = {}
    MSEs_unseen_by_num_source = {}
    FM_by_num_source = {}

    ################################################################################
    # For Loop 1
    for dist_idx, distribution in enumerate(distributions):
        # Create data source, an ndarray that contains input in the columns, grouped by permutation
        train_data_source = tools.Data_Source(num_source_list[-1], num_per_perm_list_train[-1], distribution)  
        test_data_source = tools.Data_Source(num_source_list[-1], num_per_perm_list_test[-1], distribution)  

        ################################################################################
        # For Loop 2
        for num_source in num_source_list:
            
            # Switch out arbitrary avg funcs to new num_source
            for avg_idx in range(len(weights[num_source])):
                avg_funcs[3+avg_idx] = tools.weighted_avg(weights[num_source][avg_idx])
             
            # When the # of possible permutations exceed certain number (in here 5!), 
            # instead of feeding only one more permutation a time, feed more.
            num_perms = math.factorial(num_source)
            if num_perms > train_group_num_limit:
                step = int(num_perms / train_group_num_limit)
            else:
                step = 1

            # Mean Squared Error [for each repetition, for each avg function, for each model, for each percentage, for each data#perperm, for each num_source], of all test samples, for both seen and unseen data.
            MSEs_seen = np.zeros( (len(distributions), len(num_per_perm_list_train), len(range(step-1, num_perms, step)), len(avg_funcs), len(models)) )
            MSEs_unseen = np.zeros( (len(distributions), len(num_per_perm_list_train), len(range(step-1, num_perms, step))-1, len(avg_funcs), len(models)) )
            # Record FM after train session with both seen and unseen data pattern
            FM = np.zeros( (len(distributions), len(num_per_perm_list_train), len(range(step-1, num_perms, step)), len(avg_funcs), len(models), 2**num_source-1) )

            ################################################################################
            # For Loop 3
            for npp_idx, num_per_perm_train in enumerate(num_per_perm_list_train):

                # Shuffle the order of permutations fed to model in train session
                all_perms = list(permutations(list(range(num_source))))
                random.shuffle(all_perms) # or random.Random(random_seed).shuffle(all_perms)

                # Get data_by_perm based on num_source & num_per_perm
                train_data_by_perm = train_data_source.get_data_by_perm(num_source, num_per_perm_train, all_perms)
                test_data_by_perm = test_data_source.get_data_by_perm(num_source, num_per_perm_list_test[0], all_perms)

                imbalanced_train_data_by_perm = train_data_source.get_data_by_perm(num_source, 1, all_perms)

                ################################################################################
                # For Loop 4
                for perc_idx, perc in enumerate(tqdm(range(step-1, num_perms, step))):
                    
                    # Find index of train/test sample in superset and shuffle
                    # train_idx = np.concatenate(train_idx_by_perm[0:perc+1])

                    # if data_imb == 'imbalanced' and perc < num_perms-1:
                    #     imb_data = np.concatenate(imbalanced_train_by_perm[perc+1:])
                    #     train_idx = np.concatenate((train_idx, imb_data))
                    
                    # np.random.shuffle(train_idx)
                    # test_idx = np.concatenate(test_idx_by_perm[0:perc+1])




                    # Find data sample through index
                    train_d = train_data_by_perm[0:num_per_perm_train*(perc+1), :]
                    train_d = tools.shuffle_array_columns(train_d)
                    test_d = test_data_by_perm[0:num_per_perm_list_test[0]*(perc+1), :]
                    test_d_unseen = test_data_by_perm[num_per_perm_list_test[0]*(perc+1):, :]

                    
                    
                    # Define unseen test data samples when the train data doesn't cover 100% of the permutation
                    # if perc < num_perms-1:
                    #     test_idx_unseen = np.concatenate(test_idx_by_perm[perc+1:])
                    #     test_d_unseen = test_superset[:, test_idx_unseen][0:num_source]
                    # else:
                    #     test_d_unseen = []

                    # Define subsets of 'X', or keys for fuzzy measure. Like '1 2' or '1 3 4 5' for g(x1, x2) or g(x1, x3, x4, x5)
                    # sourcesInNode, subset = tools.sources_and_subsets_nodes(num_source)
                    keys = list(tools.init_FM(num_source).keys())
                    
                    ################################################################################
                    # For Loop 5
                    for avg_idx, avg_func in enumerate(avg_funcs):
                        # Calculate label with given avg function
                        train_label = avg_func(train_d, 1)
                        test_label = avg_func(test_d, 1)
                        
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
                                optimizer = torch.optim.SGD(chi_model.parameters(), lr=lr)
                                
                                # Train 
                                tools.train_chinn(chi_model, lr, criterion, optimizer, num_epoch, torch.tensor(train_d, dtype=torch.float), torch.tensor(train_label, dtype=torch.float))
                                # Get fuzzy measure learned
                                FM_learned = (chi_model.chi_nn_vars(chi_model.vars).cpu()).detach().numpy()
                                fm_dict_binary = dict(zip(keys, FM_learned[:,0]))
                                fm_dict_lexicographic = tools.get_keys_index(num_source)
                                for key in fm_dict_lexicographic.keys():
                                    fm_dict_lexicographic[key] = fm_dict_binary[key]
                                fm = fm_dict_lexicographic
                                

                            
                            FM[dist_idx, npp_idx, perc_idx, avg_idx, model_idx, :] = np.asarray(list(fm.values()))
                            # Calculate result from integral with test data
                            test_output = np.apply_along_axis(tools.get_cal_chi(fm), 1, test_d)
                            MSE = ((test_output - test_label)**2).mean()
                            MSEs_seen[dist_idx, npp_idx, perc_idx, avg_idx, model_idx] = MSE
                            # Calculate result from integral with test data - unseen
                            if perc < num_perms-1:
                                test_label_unseen = avg_func(test_d_unseen, 1)
                                test_out_unseen = np.apply_along_axis(tools.get_cal_chi(fm), 1, test_d_unseen)
                                MSEs_unseen[dist_idx, npp_idx, perc_idx, avg_idx, model_idx] = ((test_out_unseen - test_label_unseen)**2).mean()
                                
            if num_source in FM_by_num_source.keys():
                FM_by_num_source[num_source] = np.append(FM_by_num_source[num_source], np.expand_dims(FM, axis=0), axis=0)
                MSEs_seen_by_num_source[num_source] = np.append(MSEs_seen_by_num_source[num_source], np.expand_dims(MSEs_seen, axis=0), axis=0)
                MSEs_unseen_by_num_source[num_source] = np.append(MSEs_unseen_by_num_source[num_source], np.expand_dims(MSEs_unseen, axis=0), axis=0)
            else:
                FM_by_num_source[num_source] = np.expand_dims(FM, axis=0)
                MSEs_seen_by_num_source[num_source] = np.expand_dims(MSEs_seen, axis=0)
                MSEs_unseen_by_num_source[num_source] = np.expand_dims(MSEs_unseen, axis=0)
                

    print('Rep ' + str(rep) + ' done.')
    return FM_by_num_source, MSEs_seen_by_num_source, MSEs_unseen_by_num_source