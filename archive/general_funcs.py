import itertools
from itertools import permutations
import random

from numpy.random.mtrand import shuffle
import scipy.stats
import math
import numpy as np


class Data_Source:  
    def __init__(self, max_num_source, max_num_per_perm, distribution):
        self.max_num_source = max_num_source
        self.max_num_per_perm = max_num_per_perm
        self.data_by_perm, self.all_perms = create_dataset(max_num_source, 
                                                           max_num_per_perm, 
                                                           distribution)
    
    def get_all_perms(self):
        return self.all_perms
    
    def get_data_by_perm(self, num_source, num_per_perm, all_perms):
        if num_source > self.max_num_source:
            exit('num_source exeeded maximum allowed from this data source')
        if num_per_perm > self.max_num_per_perm:
            exit('num_per_perm exeeded maximum allowed from this data source')

        if num_source == self.max_num_source and num_per_perm == self.max_num_per_perm:
            data_by_perm = self.data_by_perm
            
        else:
            data_by_perm = []
            for p1 in all_perms:
                for p2 in self.all_perms:
                    if p1 == p2[0:num_source]:
                        first_idx = self.all_perms.index(p2) * self.max_num_per_perm
                        data_by_perm.append(self.data_by_perm[first_idx:first_idx+num_per_perm, 0:num_source])
                        break
            data_by_perm = np.concatenate(data_by_perm,  0)
            
        return data_by_perm


def create_dataset(num_source, num_per_perm, distribution):
    """
    Create a dataset with all possible permutation, with each permutation having the same number of samples.
    
    :param num_source: Number of sources to be fused
    :param num_per_perm: Number of data samples for each permutation
    :param distribution: The distribution where each data sample is pulled from
    :return: ndarray (num_per_perm*num_perm, num_source), rows grouped by permutation
    """

    all_perms = list(permutations(list(range(num_source))))

    data_by_perm = []

    for perm in all_perms:
        
        # Choose data distribution
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
                
        elif distribution == 'random Gaussian':
            lower = 0
            upper = 1
            mu = random.uniform(0, 1)
            sigma = 0.15
            
        if distribution == 'uniform' or distribution == 'Gaussian':
            data = rand_func((num_per_perm, num_source))
            data.sort(axis=1)
            data_by_perm.append(data[:, np.array(perm)])
        
        elif distribution == 'polarized' or distribution == 'random Gaussian':
            if distribution == 'polarized':
                rand_func = random.choice([rand_func1, rand_func2])
            elif distribution == 'random Gaussian':
                mu = random.uniform(0, 1)
                rand_func = lambda num_source: \
                    scipy.stats.truncnorm.rvs((lower-mu)/sigma, 
                                              (upper-mu)/sigma, 
                                              loc=mu, 
                                              scale=sigma, 
                                              size=num_source)
            
            data = np.zeros((num_per_perm, num_source))
            for i in range(num_per_perm):
                d = rand_func(num_source)
                d.sort()
                data[i, :] = d[np.array(perm)]

            data_by_perm.append(data)
    
    data_by_perm = np.concatenate(data_by_perm, 0)
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


def init_FM(num_source):
    """
    Sets up a dictionary for referencing FM.
    :return: FM
    """
    input_k = np.arange(1, num_source + 1)
    FM = {}
    for i in input_k:
        keys = np.array(list(itertools.combinations(input_k, i)))
        for k in keys:
            FM[str(k)] = 1/2
        if i == num_source:
            FM[str(keys[-1])] = 1
    return FM


def get_min_fm_target(num_source):
    fm = init_FM(num_source)
    for key in fm.keys():
        if len(key.split()) != num_source:
            fm[key] = 0
        else:
            fm[key] = 1
    return fm
    
    
def get_max_fm_target(num_source):
    fm = init_FM(num_source)
    for key in fm.keys():
        fm[key] = 1
    return fm


def get_mean_fm_target(num_source):
    fm = init_FM(num_source)
    for key in fm.keys():
        fm[key] = len(key.split()) / num_source
    return fm


def weighted_avg(weight):
    return lambda x, d: np.average(x, d, weight)


def w_avg_target(num_source, weight):
    fm = init_FM(num_source)
    for idx, key in enumerate(fm.keys()):
        key_split = key[1:-1].split()
        if len(key_split) == 1:
            fm[key] = weight[idx]
        else:
            key_part1 = key[0:-3] + ']'
            key_part2 = int(key_split[-1])
            fm[key] = fm[key_part1] + weight[key_part2-1]
    return fm

def get_w_avg_target(weight):
    return lambda num_source: w_avg_target(num_source, weight)

def shuffle_array_columns(arr):
    shuffled = arr.transpose()
    np.random.shuffle(shuffled)
    shuffled = shuffled.transpose()
    return shuffled






# Process Bar
def myprogress(current, whole=1, n=30, bars=u'▕▏▎▍▌▋▊▉', full='▉', empty='▕'): 
    """ current and whole can be an element of a list being iterated, or just two numbers """
    p = (whole.index(current))/len(whole)+1e-9 if type(whole)==list else current/whole+1e-9 
    return f"{full*int(p*n)}{bars[int(len(bars)*((p*n)%1))]}{empty*int((1-p)*n)} {p*100:>6.2f}%" 

def pbiter(it, *, total = None, width = 60, _cfg = {'idx': -1, 'pbs': {}, 'lline': 0}):
    try:
        total = total or len(it)
    except:
        total = None
    
    _cfg['idx'] += 1
    idx = _cfg['idx']
    pbs = _cfg['pbs']
    pbs[idx] = [0, total, 0]
    
    def Show():
        line2 = ' '.join([
            myprogress(e[1][0], max(e[1][0], e[1][1] or
                max(1, e[1][0]) / max(.1, e[1][2])), width // len(pbs))
            for e in sorted(pbs.items(), key = lambda e: e[0])
        ])
        line = line2 + ' ' * (max(0, _cfg['lline'] - len(line2)) + 0)
        print(line, end = '\r', flush = True)
        _cfg['lline'] = len(line2)
    
    try:
        Show()
        for e in it:
            yield e
            pbs[idx][0] += 1
            pbs[idx][2] += (1. - pbs[idx][2]) * .1
            Show()
        pbs[idx][2] = 1.
        Show()
    finally:
        del pbs[idx]