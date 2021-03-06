from itertools import permutations
import math
import random

import numpy as np
import scipy.stats


# class Data_Source:  
#     def __init__(self, max_num_source, max_num_per_perm, distribution):
#         self.max_num_source = max_num_source
#         self.max_num_per_perm = max_num_per_perm
#         self.data_by_perm, self.all_perms = self.create_dataset(max_num_source, 
#                                                                 max_num_per_perm, 
#                                                                 distribution)
    

#     def get_all_perms(self):
#         return self.all_perms
    

    # def get_data_by_perm(self, num_source, num_per_perm, all_perms):
    #     if num_source > self.max_num_source:
    #         exit('num_source exeeded maximum allowed from this data source')
    #     elif num_per_perm > self.max_num_per_perm:
    #         exit('num_per_perm exeeded maximum allowed from this data source')
            
    #     else:
    #         data_by_perm = []
    #         for p1 in all_perms:
    #             for p2 in self.all_perms:
    #                 if p1 == p2[0:num_source]:
    #                     first_idx = self.all_perms.index(p2) * self.max_num_per_perm
    #                     data_by_perm.append(self.data_by_perm[first_idx:first_idx+num_per_perm, 0:num_source])
    #                     break
    #         data_by_perm = np.concatenate(data_by_perm,  0)
            
    #     return data_by_perm
    

def create_dataset(num_source, num_per_perm, distribution, all_perms):
    """
    Create a dataset with all possible permutation, with each permutation having the same number of samples.
    
    :param num_source: Number of sources to be fused
    :param num_per_perm: Number of data samples for each permutation
    :param distribution: The distribution where each data sample is pulled from
    :return: ndarray (num_per_perm*num_perm, num_source), rows grouped by permutation
    """

    # all_perms = list(permutations(list(range(num_source))))

    data_by_perm = []

    # Choose data distribution
    if distribution == 'uniform' or distribution == 'Gaussian':
        if distribution == 'uniform':
            rand_func = np.random.random
            
        elif distribution == 'Gaussian':
            lower = 0
            upper = 1
            mu = 0.5
            sigma = 0.15
            rand_func = lambda num_source: \
                scipy.stats.truncnorm.rvs((lower-mu)/sigma, 
                                        (upper-mu)/sigma, 
                                        loc=mu, 
                                        scale=sigma, 
                                        size=num_source)
        
        for perm in all_perms:
            data = rand_func((num_per_perm, num_source))
            data.sort(axis=1)
            data_by_perm.append(data[:, np.array(perm)])
        
    
    elif distribution == 'polarized':
        lower = 0
        upper = 1
        mu1 = 0
        mu2 = 1
        sigma = 0.15
        rand_func1 = lambda num_source: \
            scipy.stats.truncnorm.rvs((lower-mu1)/sigma, 
                                    (upper-mu1)/sigma, 
                                    loc=mu1, 
                                    scale=sigma, 
                                    size=num_source)
        rand_func2 = lambda num_source: \
            scipy.stats.truncnorm.rvs((lower-mu2)/sigma, 
                                    (upper-mu2)/sigma, 
                                    loc=mu2, 
                                    scale=sigma, 
                                    size=num_source)

        for perm in all_perms:
            data1 = rand_func1((math.floor(num_per_perm/2), num_source))
            data2 = rand_func2((math.ceil(num_per_perm/2), num_source))
            data1.sort(axis=1)
            data2.sort(axis=1)
            data = np.concatenate([data1, data2], axis=0)
            np.random.shuffle(data)
            data.sort(axis=1)
            data_by_perm.append(data[:, np.array(perm)])

    elif distribution == 'random Gaussian':
        lower = 0
        upper = 1
        mu = random.uniform(0, 1)
        sigma = 0.15

        for perm in all_perms:
            data = np.zeros((num_per_perm, num_source))
            for i in range(num_per_perm):
                mu = random.uniform(0, 1)
                rand_func = lambda num_source: \
                    scipy.stats.truncnorm.rvs((lower-mu)/sigma, 
                                            (upper-mu)/sigma, 
                                            loc=mu, 
                                            scale=sigma, 
                                            size=num_source)
                d = rand_func(num_source)
                data[i, :] = d
            data.sort(axis=1)
            data_by_perm.append(data[:, np.array(perm)])
    
    data_by_perm = np.concatenate(data_by_perm, 0)
    return data_by_perm