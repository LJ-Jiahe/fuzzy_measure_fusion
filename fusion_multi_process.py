
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

    if len(sys.argv) == 2:
        pool = Pool()
    elif len(sys.argv) == 3:
        pool = Pool(int(sys.argv[2])) # Create a multiprocessing Pool
    else:
        sys.exit("Usage: python fusion-MP.py <repetition> <multi_process> \n" +
                 "If <multi_process> is not assigned, the program will use as many threads as possible, " +
                 "but not exceeding <repetition>.")
    rep = int(sys.argv[1])

    # Read args <END>
    ################################################################################



    ################################################################################
    # Run child processes <START>

    # Send index of repetition to child process as an parameter.
    # <rep> currently unused, for future proof only.
    reps = range(rep)
    results = pool.map(fusion, reps)

    # Run child processes <END>
    ################################################################################



    ################################################################################
    # Reorganize results <START>
    
    # Merge results from all repetitions sent to different child processes

    results_FMs = [result['FMs_by_num_source'] for result in results]
    results_MSEs_seen = [result['MSEs_seen_by_num_source'] for result in results]
    results_MSEs_unseen = [result['MSEs_unseen_by_num_source'] for result in results]

    FMs_by_num_source_merged = {}
    MSEs_seen_by_num_source_merged = {}
    MSEs_unseen_by_num_source_merged = {}

    # All 3 <results_***> has the same exact keys.
    # <results_FMs['FMs_by_num_source']> is used for no specific reasons.
    for num_source in results_FMs['FMs_by_num_source'].keys():
        merged_FMs = np.stack([r[num_source] for r in results_FMs], axis=0)
        merged_MSEs_seen = np.stack([r[num_source] for r in results_MSEs_seen], axis=0)
        merged_MSEs_unseen = np.stack([r[num_source] for r in results_MSEs_unseen], axis=0)
        
        FMs_by_num_source_merged[num_source] = merged_FMs
        MSEs_seen_by_num_source_merged[num_source] = merged_MSEs_seen
        MSEs_unseen_by_num_source_merged[num_source] = merged_MSEs_unseen

    # Reorganize results <END>
    ################################################################################


    ################################################################################
    # Save results to file <START>

    now = datetime.datetime.now().strftime("%m-%d-%Y@%H.%M.%S")

    if not os.path.exists(c.result_dir):
        os.makedirs(c.result_dir)

    with open(c.result_dir + 'result_' + now, 'wb') as f:
        pickle.dump(FMs_by_num_source_merged, f)
        pickle.dump(MSEs_seen_by_num_source_merged, f)
        pickle.dump(MSEs_unseen_by_num_source_merged, f)
        pickle.dump(c.params, f)
        
    # Save results to file <END>
    ################################################################################


if __name__ == '__main__':
    fusion_multi_process()
