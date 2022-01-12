import datetime
import os
import pickle
import sys
from tkinter.filedialog import askopenfilenames

import numpy as np

def merge_output():
    # Load output file
    file_list = askopenfilenames(title='Load output file', initialdir='./output')

    first_file = file_list[0]
    with open(first_file, 'rb') as f:
        first_FM_by_num_source = pickle.load(f)
        first_MSEs_seen_by_num_source = pickle.load(f)
        first_MSEs_unseen_by_num_source = pickle.load(f)
        first_params = pickle.load(f)


    error = None
    for file in file_list[1:]:
        with open(file, 'rb') as f:
            FM_by_num_source = pickle.load(f)
            MSEs_seen_by_num_source = pickle.load(f)
            MSEs_unseen_by_num_source = pickle.load(f)
            params = pickle.load(f)
            if params.keys() != first_params.keys():
                sys.exit('Parameters key inconsistent')
            else:
                for key in first_params.keys():
                    try:
                        if params[key] == first_params[key]:
                            continue
                        else:
                            sys.exit('Parameter value inconsistent')
                    except ValueError as e:
                        print('Params consistency uncertain, see reason below')
                        print('ValueError:', e, '\n')
                        error = e
                
    if error == None:
        print('Params consistent')
    else:
        print('Params seems consistent, but it\'s uncertain due to error caught')


    Merged_FM_by_num_source = first_FM_by_num_source
    Merged_MSEs_seen_by_num_source = first_MSEs_seen_by_num_source
    Merged_MSEs_unseen_by_num_source = first_MSEs_unseen_by_num_source



    for file in file_list[1:]:
        with open(file, 'rb') as f:
            FM_by_num_source = pickle.load(f)
            MSEs_seen_by_num_source = pickle.load(f)
            MSEs_unseen_by_num_source = pickle.load(f)
            params = pickle.load(f)

        for num_source in Merged_FM_by_num_source.keys():
            Merged_FM_by_num_source[num_source] = np.append(Merged_FM_by_num_source[num_source], FM_by_num_source[num_source], axis=0)
            Merged_MSEs_seen_by_num_source[num_source] = np.append(Merged_MSEs_seen_by_num_source[num_source], MSEs_seen_by_num_source[num_source], axis=0)
            Merged_MSEs_unseen_by_num_source[num_source] = np.append(Merged_MSEs_unseen_by_num_source[num_source], MSEs_unseen_by_num_source[num_source], axis=0)



    output_dir = 'output/'
    now = datetime.datetime.now().strftime("%m-%d-%Y@%H.%M.%S")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    time_list = [file.split('_')[-1] for file in file_list]

    with open(output_dir + 'Merged_output_' + '+'.join(time_list) + '_' + now, 'wb') as f:
        pickle.dump(Merged_FM_by_num_source, f)
        pickle.dump(Merged_MSEs_seen_by_num_source, f)
        pickle.dump(Merged_MSEs_unseen_by_num_source, f)
        pickle.dump(first_params, f)

    print('Outputs successfully merged! Saved to ./' + output_dir)
    for key in first_params.keys():
        print(first_params[key])