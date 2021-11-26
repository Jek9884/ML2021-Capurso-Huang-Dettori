import os

from network import Network
from optimizer import GradientDescent
import itertools as it
import numpy as np
from joblib import Parallel, delayed

key_names = ['conf_layer_list', 'init_func_list', 'act_func_list',
             'out_func_list', 'loss_func_list', 'bias_list', 'lr_list', 'batch_size_list',
             'reg_val_list', 'reg_type_list', 'epochs_list']
key_names.sort()


def accuracy(net, x_set, y_set):
    len_set = len(x_set)
    corr_count = 0

    for i, entry in enumerate(x_set):

        res = net.forward(np.asarray(x_set[i]).flatten())
        thresh_res = 1 if res.item() >= 0.5 else 0

        if y_set[i].item() == thresh_res:
            corr_count += 1

    return corr_count, len_set


def grid_search(train_x, train_y, parameters_dict):
    # TODO add validation step

    # Sorting helps to check missing hyper-parameters
    keys = list(parameters_dict.keys())
    keys.sort()
    if keys != key_names:
        raise ValueError("Missing hyper parameters")

    # It makes a cartesian product between all hyper-parameters
    combo_list = list(it.product(*(parameters_dict[k] for k in key_names)))

    results = Parallel(n_jobs=os.cpu_count(), verbose=50)(delayed(train)(
        train_x, train_y, combo) for i, combo in enumerate(combo_list))

    best_score = 0
    best_combo = None
    for i, result in enumerate(results):
        if best_score < result[0]:
            best_score = result[0]
            best_combo = result[1]

    return best_score, best_combo


def train(train_x, train_y, combo):
    best_score = 0
    best_combo = None

    network = Network(combo[key_names.index('conf_layer_list')],
                      combo[key_names.index('init_func_list')],
                      combo[key_names.index('act_func_list')],
                      combo[key_names.index('out_func_list')],
                      combo[key_names.index('loss_func_list')],
                      combo[key_names.index('bias_list')])

    gradient_descent = GradientDescent(network, combo[key_names.index('lr_list')],
                                       combo[key_names.index('batch_size_list')],
                                       combo[key_names.index('reg_val_list')],
                                       combo[key_names.index('reg_type_list')],
                                       combo[key_names.index('epochs_list')])

    gradient_descent.optimize(train_x, train_y)

    n_correct, len_y = accuracy(network, train_x, train_y)

    if (n_correct / len_y) > best_score:
        best_score = n_correct / len_y
        best_combo = combo

    return best_score, best_combo
