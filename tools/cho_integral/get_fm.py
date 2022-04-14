import itertools

import numpy as np


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
            FM[str(k)[1:-1]] = 1/2
        if i == num_source:
            FM[str(keys[-1])[1:-1]] = 1
    return FM


def init_Arbitrary_FM(num_source):
    fm = init_FM(num_source)
    singletons = np.random.random(num_source)
    for k, v in enumerate(singletons):
        fm[str(k+1)] = v

    for l_k in range(2, num_source):
        for key in fm.keys():
            if len(key.split()) == l_k:
                print(key)
                print(np.array(list(itertools.combinations(key.split(), l_k-1))))



    return fm

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


