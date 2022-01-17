import numpy as np


# Idea: split data handling from optimizer, handy for batch norm inference
class DataHandler:

    def __init__(self, data_x, data_y):

        self.data_x = data_x
        self.data_y = data_y

        self.avg_x = None
        self.std_x = None

        self.avg_y = None
        self.std_y = None

        self.n_patterns = self.data_x.shape[0]
        self.index_list = np.arange(self.n_patterns)

    def normalise_x(self):

        if self.avg_x is not None or self.std_x is not None:
            raise ValueError("normalise_x: data already normalised")

        self.data_x, self.avg_x, self.std_x = normalise_data(self.data_x)

    def denormalise_x(self):

        if self.avg_x is None or self.std_x is None:
            raise ValueError("denormalise_x: data is not normalised")

        self.data_x = self.data_x * self.std_x + self.avg_x
        self.avg_x = None
        self.std_x = None

    def normalise_y(self):

        if self.avg_y is not None or self.std_y is not None:
            raise ValueError("normalise_y: data already normalised")

        self.data_y, self.avg_y, self.std_y = normalise_data(self.data_y)

    def denormalise_y(self):

        if self.avg_y is None or self.std_y is None:
            raise ValueError("denormalise_y: data is not normalised")

        self.data_y = self.data_y * self.std_y + self.avg_y
        self.avg_y = None
        self.std_y = None

    # enforce_size: guarantee that mini-batch will be of the requested size
    def get_minibatch_list(self, batch_size, enforce_size=False):

        # Shortcut for full batch size
        if batch_size == -1:
            batch_size = self.n_patterns
        elif batch_size < 1:
            raise ValueError("DataHandler: invalid batch_size given")

        n_batches = int(np.ceil(self.n_patterns / batch_size))

        # Shuffle indexes of data for sampling
        # This way the original data is not modified
        np.random.shuffle(self.index_list)

        batch_x_list = []
        batch_y_list = []

        for i in range(n_batches):

            batch_idx_list = self.index_list[i * batch_size: (i + 1) * batch_size]
            cur_batch_size = len(batch_idx_list)

            if cur_batch_size < batch_size and enforce_size:
                other_idxs = self.index_list[0: i * batch_size]
                n_diff_batch = batch_size - cur_batch_size

                rand_sampled_idx = np.random.choice(other_idxs, n_diff_batch)
                batch_idx_list = np.concatenate((batch_idx_list, rand_sampled_idx))

            batch_x_list.append(self.data_x[batch_idx_list])
            batch_y_list.append(self.data_y[batch_idx_list])

        return batch_x_list, batch_y_list


# Utilities
def read_monk(path):
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

        if targets.shape[0] == 1:
            targets = targets.transpose()

    return patterns, targets


def read_cup(path, ts_set=False):
    with open(path) as f:
        data = f.readlines()
        targets = []
        patterns = []

    for line in data:
        if line[0] == '#':
            continue
        elem = line.split(',')
        elem = elem[1:]
        if ts_set:
            patterns.append(elem)
        else:
            targets.append(elem[-2:])
            patterns.append(elem[:-2])

    patterns = np.array(patterns, dtype=float)
    targets = np.array(targets, dtype=float)

    return patterns, targets


def one_hot_encode(patterns):
    final_patt = None

    for i in range(patterns.shape[1]):

        cur_col = patterns[:, i]
        unique_vals = np.unique(cur_col)
        num_values_var = len(unique_vals)
        feat_mat = np.zeros((patterns.shape[0], num_values_var))

        for val in unique_vals:
            # Note: could be a dangerous approach
            feat_mat[cur_col == val, val - 1] = 1

        if final_patt is None:
            final_patt = feat_mat
        else:
            final_patt = np.concatenate((final_patt, feat_mat), axis=1)

    return final_patt


def normalise_data(patterns):
    patterns_mean = np.mean(patterns, axis=0)
    patterns_std = np.std(patterns, axis=0)
    patterns_norm = np.divide(np.subtract(patterns, patterns_mean), patterns_std)

    return patterns_norm, patterns_mean, patterns_std
