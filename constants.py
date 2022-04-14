# All constants are initialized here.
# Note that those  parameters are supposed to be copied when used in other modules.
# Otherwise conflict might happen.

import math

import numpy as np

from tools.c_avg import centered_average_multi
from tools.cho_integral import Choquet_Integral_QP, Choquet_Integral_NN



################################################################################
# Data related parameters <START>

num_source_list = list(range(3, 7)) # Number of sources to be fused, a to b-1

num_per_perm_list_train = [5] # Each permutation gets the same number of samples, try different values here for train set
num_per_perm_list_test = [10] # Each permutation gets the same number of samples

distributions = ['uniform', 'Gaussian', 'polarized']
# distributions = ['uniform', 'Gaussian', 'polarized', 'random Gaussian'] # Use to discuss whether the distribution of data could have an impact on result

weights = {3: np.asarray([[0.1, 0.8, 0.1],
                          [0.0, 0.5, 0.5],
                          [0.3, 0.5, 0.2],
                          [1/3, 1/3, 1/3]]),
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

avg_funcs = [np.amin, np.amax, np.mean, centered_average_multi, None, None, None, None, None] # List of avg functions

avg_names = {} # List of names of avg functions
for num_source in num_source_list:
    weight = weights[num_source]
    weight_legend = []
    for w in weight:
        weight_legend.append(' '.join(map(str, (w*10).astype(int))))
    avg_names[num_source] = ['Min', 'Max', 'Mean', 'Centered Mean'] + weight_legend + ['Arbitrary FM']

train_group_num_limit = math.factorial(6)

# Data related parameters <END>
################################################################################


################################################################################
# Other parameters <START>

# models = [tools.Choquet_Integral_QP]
# model_names = ['QP']

# models = [tools.Choquet_Integral_NN]
# model_names = ['NN']

models = [Choquet_Integral_QP, Choquet_Integral_NN]
model_names = ['QP', 'NN']

result_dir = 'output/results/'

params = {'num_source_list': num_source_list,
          'num_per_perm_list_train': num_per_perm_list_train,
          'num_per_perm_list_test': num_per_perm_list_test,
          'distributions': distributions,
          'avg_names': avg_names,
          'train_group_num_limit': train_group_num_limit,
          'model_names': model_names}



# Other parameters <END>
################################################################################


################################################################################
# Plot parameters <START>
plot_size = (4, 3)

font_size = 8

xlabel = 'Percentage of Observed Permutations'

ylabel = 'Mean Squared Error'

axis_left = 0.15

axis_right = 0.95
# Plot parameters <END>
################################################################################