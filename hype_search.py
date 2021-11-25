from network import Network
from optimizer import GradientDescent
import itertools as it
import numpy as np


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
    """
    conf_layer_list, init_func_list, act_func_list,
                out_func_list, loss_func_list, bias_list, lr_list, batch_size_list,
                reg_val_list, reg_type_list, epochs_list
    """

    key_names = ['conf_layer_list', 'init_func_list', 'act_func_list',
                 'out_func_list', 'loss_func_list', 'bias_list', 'lr_list', 'batch_size_list',
                 'reg_val_list', 'reg_type_list', 'epochs_list']

    keys = list(parameters_dict.keys())
    key_names.sort()
    keys.sort()
    if keys != key_names:
        raise ValueError("Missing hyper parameters")

    combo_list = list(it.product(*(parameters_dict[k] for k in key_names)))

    best_score = 0
    best_combo = None

    for i, combo in enumerate(combo_list):
        net = Network(combo[key_names.index('conf_layer_list')],
                      combo[key_names.index('init_func_list')],
                      combo[key_names.index('act_func_list')],
                      combo[key_names.index('out_func_list')],
                      combo[key_names.index('loss_func_list')],
                      combo[key_names.index('bias_list')])

        gd = GradientDescent(net, combo[key_names.index('lr_list')],
                             combo[key_names.index('batch_size_list')],
                             combo[key_names.index('reg_val_list')],
                             combo[key_names.index('reg_type_list')],
                             combo[key_names.index('epochs_list')])

        gd.optimize(train_x, train_y)

        n_correct, len_y = accuracy(net, train_x, train_y)
        if (n_correct / len_y) > best_score:
            best_score = n_correct / len_y
            best_combo = combo

        # TODO add validation step

    return best_score, best_combo
