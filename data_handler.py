import numpy as np


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

        patterns = np.matrix(patterns, dtype=int)
        targets = np.matrix(targets, dtype=int)

        if targets.shape[0] == 1:
            targets = targets.transpose()

    return patterns, targets
