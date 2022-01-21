import numpy as np
import os

"""
    Split the cup dataset in a reproducible way
    
    Parameters:
        -in_path: path of the cup dataset
        -out_tr_path: path where to save the training set
        -out_ts_path: path where to save the test set
        -tr_ratio: ratio of data to use for training 
"""


def cup_split(in_path, out_tr_path, out_ts_path, tr_ratio):
    seed = 42
    rng = np.random.default_rng(seed)

    # read lines, shuffle and split
    with open(in_path, 'r') as f_in:
        lines = f_in.readlines()
        headers = lines[:7]
        patterns = lines[7:]

        if 0 > tr_ratio > 1:
            raise ValueError("tr_ratio between 0 and 1")

        p_len = len(patterns)
        tr_len = int(p_len * tr_ratio)

        rng.shuffle(patterns)
        tr_set = patterns[:tr_len]
        ts_set = patterns[tr_len:]

    # write tr_set
    with open(out_tr_path, 'w') as f_tr_out:
        f_tr_out.writelines(headers)
        f_tr_out.writelines(tr_set)

    # write ts_set
    with open(out_ts_path, 'w') as f_ts_out:
        f_ts_out.writelines(headers)
        f_ts_out.writelines(ts_set)


cup_path = os.path.join('.', 'ML-CUP21-TR.csv')
tr_path = os.path.join('.', 'cup_tr.csv')
ts_path = os.path.join('.', 'cup_ts.csv')

cup_split(cup_path, tr_path, ts_path, 0.8)
