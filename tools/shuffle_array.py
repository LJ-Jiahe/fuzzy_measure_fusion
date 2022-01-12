
import numpy as np


def shuffle_array_columns(arr):
    shuffled = arr.transpose()
    np.random.shuffle(shuffled)
    shuffled = shuffled.transpose()
    return shuffled