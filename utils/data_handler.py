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
        targets = np.array(targets, dtype=int, ndmin=2)

        if norm_data:
            patterns = normalise_data(patterns)

        if targets.shape[0] == 1:
            targets = targets.transpose()

    return patterns, targets


def normalise_data(patterns):
    patterns_mean = np.mean(patterns, axis=0)
    patterns_std = np.std(patterns, axis=0)
    patterns_norm = np.divide(np.subtract(patterns, patterns_mean), patterns_std)

    return patterns_norm