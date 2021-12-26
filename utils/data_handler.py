import numpy as np


# Idea: split data handling from optimizer, handy for batch norm inference
class DataHandler:

    def __init__(self, data_x, data_y):

        self.data_x = data_x
        self.data_y = data_y

        self.n_patterns = self.data_x.shape[0]
        self.index_list = np.arange(self.n_patterns)


    def gen_minibatch_iter(self, batch_size, n_batches=-1, enforce_size=False):

        max_batches = int(np.ceil(self.n_patterns/batch_size))

        # Shuffle indexes of data for sampling
        np.random.shuffle(self.index_list)

        # Shortcut for full batch size
        if batch_size == -1:
            batch_size = n_patterns
        elif batch_size < 1:
            raise ValueError("DataHandler: invalid batch_size given")

        batch_x_list = []
        batch_y_list = []


        if n_batches == -1:
            n_batches = max_batches
        elif n_batches < 1 or n_batches > max_batches:
            raise ValueError("DataHandler: batch size should be >= 1 and <= l.\
                             If you want to use the full batch version use -1.")

        for i in range(n_batches):

            batch_idx_list = self.index_list[i * batch_size: (i + 1) * batch_size]
            cur_batch_size = len(batch_idx_list)

            if cur_batch_size != batch_size and enforce_size:
                raise ValueError("TODO: not implemented")
            else:
                batch_x_list.append(self.data_x[batch_idx_list])
                batch_y_list.append(self.data_y[batch_idx_list])

        return batch_x_list, batch_y_list


# Utilities
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
