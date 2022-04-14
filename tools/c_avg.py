import numpy as np



def centered_average_multi(arr, axis):
    return np.apply_along_axis(centered_average, axis, arr)


def centered_average(nums):
    return (sum(nums) - max(nums) - min(nums)) / (len(nums) - 2) 