import os
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from tools.plot import plot_distributions, plot_models, plot_NPP, plot_operators, plot_svsuns

# Load output file
output_file = askopenfilename(title='Load output file', initialdir='./output/result')
with open(output_file, 'rb') as f:
    FM_by_num_source = pickle.load(f)
    MSEs_seen_by_num_source = pickle.load(f)
    MSEs_unseen_by_num_source = pickle.load(f)
    params = pickle.load(f)

# for key in params.keys():
#     globals()[key] = params[key]



output_dir = './output/plots-distributions/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plot_distributions(MSEs_unseen_by_num_source, params, output_dir)

output_dir = './output/plots-models/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plot_models(MSEs_unseen_by_num_source, params, output_dir)

output_dir = './output/plots-NPP/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plot_NPP(MSEs_unseen_by_num_source, params, output_dir)

output_dir = './output/plots-operators/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plot_operators(MSEs_seen_by_num_source, MSEs_unseen_by_num_source, params, output_dir)

output_dir = './output/plots-seenVSunseen/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plot_svsuns(MSEs_seen_by_num_source, MSEs_unseen_by_num_source, params, output_dir)
