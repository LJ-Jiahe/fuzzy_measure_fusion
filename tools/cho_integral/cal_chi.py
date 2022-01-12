import numpy as np


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