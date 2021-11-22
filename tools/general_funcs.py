import itertools
from itertools import permutations
import random
import scipy.stats

import math
import numpy as np


class Data_Source:
    def __init__(self, num_source, num_per_perm, distribution='uniform'):
        self.num_source = num_source
        self.num_per_perm = num_per_perm
        self.data_by_perm, self.all_perms = create_dataset(num_source, num_per_perm, distribution)
    
    def get_superset(self):
        return self.superset
    
    def get_all_perms(self):
        return self.all_perms
    
    def get_data_by_perm(self, num_source, num_per_perm, all_perms):
        if num_source == self.num_source and num_per_perm == self.num_per_perm:
            data_by_perm = self.data_by_perm
            
        else:
            data_by_perm = []
            for p1 in all_perms:
                for p2 in self.all_perms:
                    if p1 == p2[0:num_source]:
                        p2_first_idx = self.all_perms.index(p2) * self.num_per_perm
                        data_by_perm.append(self.data_by_perm[0:num_source, p2_first_idx:p2_first_idx+num_per_perm])
                        break
            data_by_perm = np.concatenate(data_by_perm,  1)
            
        return data_by_perm


def create_dataset(num_source, num_per_perm, distribution):
    """
    Create a dataset with all possible permutation, with each permutation having the same number of samples.
    
    :param num_source: Number of sources to be fused
    :param all_perms: A list of permutation. 
                      Use as an input so that the algorithm creates data for different permutations in the order assigned.
    :param num_per_perm: Number of data samples for each permutation
    :param superset_factor: Data for each permutation are pulled from a randomly generated dataset. 
                            To ensure that each permutation gets at least #num_per_perm# data samples,
                            create a dataset #superset_factor* times (normally 3 will be enough) bigger than the dataset wanted.
    """
    # Every permutation gets the same number of train/test data samples,
    # To ensure that, calculate the number of data needed in total, 
    # and generate a super dataset that is multiple times bigger.
    num = math.factorial(num_source) * num_per_perm

    all_perms = list(permutations(list(range(num_source))))

    superset_factor = 4 # A factor needed specificly for [MY] random data creation.
                        # Explaination:
                        # When num_source=3, there are 6 possible permutations, 1 2 3, 1 3 2, 2 1 3, 2 3 1, 3 1 2, 3 2 1
                        # If we want 10 data points for each permutation, the dataset should have a size of 60, 10 samples in each permutation.
                        # But you can't ensure that by simply randomly sample 60 3-num_source entional data samples. At least one permutation will end up having less than 10 samples.
                        # What's a good number that you should sample from? 240, or 4 times more than your required number in this case, is what I found to be "Never fails on me".
    
    # Create superset
    
    data_by_perm = []
    for perm in all_perms:
        data = np.zeros((num_source, num_per_perm))
        
        # Choose data distribution
        if distribution == 'uniform':
            rand_func = np.random.random
            
        elif distribution == 'Gaussian':
            lower = 0
            upper = 1
            mu = 0.5
            sigma = 0.15
            rand_func = lambda num_source: scipy.stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=num_source)
            
        elif distribution == 'polarized':
            lower = 0
            upper = 1
            mu1 = 0
            mu2 = 1
            sigma = 0.15
            rand_func1 = lambda num_source: scipy.stats.truncnorm.rvs((lower-mu1)/sigma, (upper-mu1)/sigma, loc=mu1, scale=sigma, size=num_source)
            rand_func2 = lambda num_source: scipy.stats.truncnorm.rvs((lower-mu2)/sigma, (upper-mu2)/sigma, loc=mu2, scale=sigma, size=num_source)
#             rand_func = random.choice([rand_func1, rand_func2])
                
        elif distribution == 'random Gaussian':
            lower = 0
            upper = 1
            mu = random.uniform(0, 1)
            sigma = 0.15
#             rand_func = lambda num_source: scipy.stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=num_source)
            
            
        if distribution == 'uniform' or distribution == 'Gaussian':
            data = rand_func((num_source, num_per_perm))
            data.sort(axis=0)
            data_by_perm.append(data[np.array(perm), :])
        
        elif distribution == 'polarized' or distribution == 'random Gaussian':
            if distribution == 'polarized':
                rand_func = random.choice([rand_func1, rand_func2])
            elif distribution == 'random Gaussian':
                mu = random.uniform(0, 1)
                rand_func = lambda num_source: scipy.stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=num_source)
        
            for i in range(num_per_perm):
                d = rand_func(num_source)
                d.sort()
                data[:, i] = d[np.array(perm)]

            data_by_perm.append(data)
    
    data_by_perm = np.concatenate(data_by_perm, 1)
    return data_by_perm, all_perms


def cal_chi(fm, x):
    """
    Calculates ChI with given fuzzy measure and input
    
    :param fm: Fuzzy measure
    :param x: Input
    :return: Single value Chi output
    """
    pi_i = np.argsort(-x) + 1 # Arg sort of input, with the smallest index being 1
    ch = x[pi_i[0] - 1] * (fm[str(pi_i[:1])])
    for i in range(1, len(x)):
        latt_pti = np.sort(pi_i[:i+1])
        latt_ptimin1 = np.sort(pi_i[:i])
        ch = ch + x[pi_i[i] - 1] * (fm[str(latt_pti)] - fm[str(latt_ptimin1)])
    return ch


def get_cal_chi(fm):
    return lambda x: cal_chi(fm, x)


# Convert decimal to binary string
def sources_and_subsets_nodes(N):
    str1 = "{0:{fill}"+str(N)+"b}"
    a = []
    for i in range(1,2**N):
        a.append(str1.format(i, fill='0'))

    sourcesInNode = []
    sourcesNotInNode = []
    subset = []
    sourceList = list(range(N))
    # find subset nodes of a node
    def node_subset(node, sourcesInNodes):
        return [node - 2**(i) for i in sourcesInNodes]
    
    # convert binary encoded string to integer list
    def string_to_integer_array(s, ch):
        N = len(s) 
        return [(N - i - 1) for i, ltr in enumerate(s) if ltr == ch]
    
    for j in range(len(a)):
        # index from right to left
        idxLR = string_to_integer_array(a[j],'1')
        sourcesInNode.append(idxLR)  
        sourcesNotInNode.append(list(set(sourceList) - set(idxLR)))
        subset.append(node_subset(j,idxLR))

    return sourcesInNode, subset


def get_keys_index(num_source):
    """
    Sets up a dictionary for referencing FM.
    :return: The keys to the dictionary
    """
    vls = np.arange(1, num_source + 1)
    Lattice = {}
    for i in range(1, num_source + 1):
        A = np.array(list(itertools.combinations(vls, i)))
        for latt_pt in A:
            Lattice[str(latt_pt)] = 1/2
        if i == num_source:
            Lattice[str(A[-1])] = 1
    return Lattice


def get_min_fm_target(num_source):
    fm = get_keys_index(num_source)
    for key in fm.keys():
        if len(key.split()) != num_source:
            fm[key] = 0
        else:
            fm[key] = 1
    return fm
    
    
def get_max_fm_target(num_source):
    fm = get_keys_index(num_source)
    for key in fm.keys():
        fm[key] = 1
    return fm


def gmean(num_source):
    return lambda x, d: np.power(np.prod(x, d), 1/num_source)


def get_mean_fm_target(num_source):
    fm = get_keys_index(num_source)
    for key in fm.keys():
        fm[key] = len(key.split()) / num_source
    return fm


def get_gmean_fm_target(num_source):
    fm = get_mean_fm_target(num_source)
    return fm


def w_avg(weight):
    return lambda x, d: np.average(x, d, weight)

        
def w_avg_target(num_source, weight):
    fm = get_keys_index(num_source)
    for idx, key in enumerate(fm.keys()):
        if len(key.split()) == 1:
            fm[key] = weight[idx]
        elif len(key.split()) == 2:
            key1 = int(key[1:-1].split()[0])
            key2 = int(key[1:-1].split()[1])
            fm[key] = weight[key1-1] + weight[key2-1]
        elif len(key.split()) == 3:
            key1 = key[1:-1].split()[0]
            key2 = key[1:-1].split()[1]
            key3 = int(key[1:-1].split()[2])
            key12 = '[' + key1 + ' ' + key2 + ']'
            fm[key] = fm[key12] + weight[key3-1]
        elif len(key.split()) == 4:
            key1 = key[1:-1].split()[0]
            key2 = key[1:-1].split()[1]
            key3 = key[1:-1].split()[2]
            key4 = int(key[1:-1].split()[3])
            key123 = '[' + key1 + ' ' + key2 + ' ' + key3 + ']'
            fm[key] = fm[key123] + weight[key4-1]
    return fm

def get_w_avg_target(weight):
    return lambda num_source: w_avg_target(num_source, weight)