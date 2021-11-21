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

        patterns = np.matrix(patterns).astype(int)
        print(patterns)

    return patterns, targets
