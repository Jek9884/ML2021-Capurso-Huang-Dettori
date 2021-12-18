import numpy as np


def read_monk(path, norm_data=True):
    with open(path) as f:
        data = f.readlines()
        targets = []
        patterns = []

        for line in data:
            elem = line.split()
            elem = elem[:-1]
            targets.append(elem[0])
            elem = elem[1:]
            patterns.append(elem)

        patterns = np.array(patterns, dtype=int)
        patterns = one_hot_encode(patterns)

        targets = np.array(targets, dtype=int, ndmin=2)

        if norm_data:
            patterns = normalise_data(patterns)

        if targets.shape[0] == 1:
            targets = targets.transpose()

    return patterns, targets


def one_hot_encode(patterns):

    final_patt = None

    for i in range(patterns.shape[1]):

        cur_col = patterns[:, i]
        unique_vals = np.unique(cur_col)
        num_values_var = len(unique_vals)
        feat_mat = np.zeros((patterns.shape[0], num_values_var))

        for val in unique_vals:
            #Note: could be a dangerous approach
            feat_mat[cur_col == val, val-1] = 1

        if final_patt is None:
            final_patt = feat_mat
        else:
            final_patt = np.concatenate((final_patt, feat_mat), axis=1)

    return final_patt


def normalise_data(patterns):
    patterns_mean = np.mean(patterns, axis=0)
    patterns_std = np.std(patterns, axis=0)
    patterns_norm = np.divide(np.subtract(patterns, patterns_mean), patterns_std)

    return patterns_norm
