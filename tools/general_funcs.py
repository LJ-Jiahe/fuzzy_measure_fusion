import itertools
from itertools import permutations
import random
import scipy.stats

import math
import numpy as np


class Data_Source:
    def __init__(self, dim, num_per_perm, superset_factor, distribution='uniform'):
        self.dim = dim
        self.num_per_perm = num_per_perm
        self.all_perms = list(permutations(list(range(dim))))
        self.superset, _, self.data_idx_by_perm = create_dataset(dim, self.all_perms, num_per_perm, superset_factor, distribution)
    
    def get_superset(self):
        return self.superset
    
    def get_data_idx(self, dim, num_per_perm):
        if dim == self.dim and num_per_perm == self.num_per_perm:
            data_idx = self.data_idx_by_perm
        
        else:
            all_perms = list(permutations(list(range(dim))))
            data_idx = []
            for p1 in all_perms:
                for j, p2 in enumerate(self.all_perms):
                    if p1 == p2[0:dim]:
                        data_idx.append(self.data_idx_by_perm[j][0:num_per_perm])
                        break
            data_idx = np.asarray(data_idx)
        
        return data_idx


def create_dataset(dim, all_perms, num_per_perm, superset_factor, distribution):
    """
    Create a dataset with all possible permutation, with each permutation having the same number of samples.
    
    :param dim: Dimension of data sample
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
    num = math.factorial(dim) * num_per_perm
    # Create superset
    if distribution == 'uniform':
        data_superset = np.random.rand(dim, num*superset_factor)
    elif distribution == 'normal':
        lower = 0
        upper = 1
        mu = 0.5
        sigma = 0.15
        data_superset = scipy.stats.truncnorm.rvs((lower-mu)/sigma, (upper-mu)/sigma, loc=mu, scale=sigma, size=(dim, num*superset_factor))
    elif distribution == 'bimodal':
        lower = 0
        upper = 1
        mu1 = 0
        mu2 = 1
        sigma = 0.15
        data_superset1 = scipy.stats.truncnorm.rvs((lower-mu1)/sigma, (upper-mu1)/sigma, loc=mu1, scale=sigma, size=(dim, int(num*superset_factor/2)))
        data_superset2 = scipy.stats.truncnorm.rvs((lower-mu2)/sigma, (upper-mu2)/sigma, loc=mu2, scale=sigma, size=(dim, int(num*superset_factor/2)))
        data_superset = np.concatenate((data_superset1, data_superset2), 1)
    # Get permutation of each data sample
    data_perms = np.argsort(data_superset, 0)
    # N! possible permutations
#     all_perms = list(permutations(list(range(dim))))
    # Group data sample according to its permutation
    data_idx_superset_by_perm = []
    
    for i, current_perm in enumerate(all_perms):
        # Get index of data sample of certain permutation and save to list
        temp = np.where(data_perms[0, :]==current_perm[0])
        for idx, p in enumerate(current_perm):
            temp = np.intersect1d(temp, np.where(data_perms[idx, :]==p))
        if temp.size < num_per_perm:
            print('Current permutation doesn\'t have sufficient number of samples. Please regenerate!')
            exit()
        data_idx_superset_by_perm.append(temp)
    
    # Every permutation gets the same number of train/test data samples,
    # Data is randomly pull from superset each epoch
    data_idx_by_perm = []
    for i in range(len(all_perms)):
        temp = data_idx_superset_by_perm[i]
        random.shuffle(temp)
        data_idx_by_perm.append(temp[0:num_per_perm])
        
    data_idx_by_perm = np.asarray(data_idx_by_perm)
        
    return data_superset, data_idx_superset_by_perm, data_idx_by_perm


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


def get_keys_index(dim):
    """
    Sets up a dictionary for referencing FM.
    :return: The keys to the dictionary
    """
    vls = np.arange(1, dim + 1)
    Lattice = {}
    for i in range(1, dim + 1):
        A = np.array(list(itertools.combinations(vls, i)))
        for latt_pt in A:
            Lattice[str(latt_pt)] = 1/2
        if i == dim:
            Lattice[str(A[-1])] = 1
    return Lattice


def get_min_fm_target(dim):
    fm = get_keys_index(dim)
    for key in fm.keys():
        if len(key.split()) != dim:
            fm[key] = 0
        else:
            fm[key] = 1
    return fm
    
    
def get_max_fm_target(dim):
    fm = get_keys_index(dim)
    for key in fm.keys():
        fm[key] = 1
    return fm


def gmean(dim):
    return lambda x, d: np.power(np.prod(x, d), 1/dim)


def get_mean_fm_target(dim):
    fm = get_keys_index(dim)
    for key in fm.keys():
        fm[key] = len(key.split()) / dim
    return fm


def get_gmean_fm_target(dim):
    fm = get_mean_fm_target(dim)
    return fm


def w_avg(weight):
    return lambda x, d: np.average(x, d, weight)

        
def w_avg_target(dim, weight):
    fm = get_keys_index(dim)
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
    return lambda dim: w_avg_target(dim, weight)