
import numpy as np


def weighted_avg(weight):
    return lambda x, d: np.average(x, d, weight)