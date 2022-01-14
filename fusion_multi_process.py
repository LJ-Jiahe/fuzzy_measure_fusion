# Imports

import datetime
from multiprocessing import Pool
import os
import pickle
import sys

import numpy as np

from tools import fusion
import constants as c


def fusion_multi_process():
    ################################################################################
    # Read args <START>
    if len(sys.argv) == 3:
        pool = Pool()
    elif len(sys.argv) == 4:
        pool = Pool(int(sys.argv[3])) # Create a multiprocessing Pool
    else:
        sys.exit("Usage: python fusion-MP.py <repetition> <max_Num_Sources> <multi_process> \n" +
                  "If <multi_process> is not assigned, the program will use as many threads as possible, " +
                  "but not exceeding <repetition>.")    
    rep = int(sys.argv[1])
    max_Num_Source = int(sys.argv[2])
    c.num_source_list = list(range(3, max_Num_Source+1))
    # Read args <END>
    ################################################################################


    ################################################################################
    # Run child processes <START>
    reps = range(rep)
    # max_NSs = np.full(rep, max_Num_Source)
    # packed_params = [params for params in zip(reps, max_NSs)]
    results = pool.map(fusion, reps)
    # Run child processes <END>
    ################################################################################


    ################################################################################
    # Reorganize results <START>
    results_FMs = [result[0] for result in results]
    results_MSEs_seen = [result[1] for result in results]
    results_MSEs_unseen = [result[2] for result in results]

    FMs_by_num_source = {}
    MSEs_seen_by_num_source = {}
    MSEs_unseen_by_num_source = {}

    for num_source in results_FMs[0].keys():
        stacked_FMs = np.stack([r[num_source] for r in results_FMs], axis=0)
        stacked_MSEs_seen = np.stack([r[num_source] for r in results_MSEs_seen], axis=0)
        stacked_MSEs_unseen = np.stack([r[num_source] for r in results_MSEs_unseen], axis=0)
        
        FMs_by_num_source[num_source] = stacked_FMs
        MSEs_seen_by_num_source[num_source] = stacked_MSEs_seen
        MSEs_unseen_by_num_source[num_source] = stacked_MSEs_unseen
    # Reorganize results <END>
    ################################################################################


    ################################################################################
    # Save results to file <START>
    now = datetime.datetime.now().strftime("%m-%d-%Y@%H.%M.%S")

    if not os.path.exists(c.result_dir):
        os.makedirs(c.result_dir)

    with open(c.result_dir + 'result_' + now, 'wb') as f:
        pickle.dump(FMs_by_num_source, f)
        pickle.dump(MSEs_seen_by_num_source, f)
        pickle.dump(MSEs_unseen_by_num_source, f)
        pickle.dump(c.params, f)
    # Save results to file <END>
    ################################################################################


if __name__ == '__main__':
    fusion_multi_process()
