from network import Network
from optimizer import GradientDescent
import itertools as it
import numpy as np
from joblib import Parallel, delayed
from function import Function

key_names = ['conf_layer_list', 'init_func_list', 'act_func_list',
             'out_func_list', 'loss_func_list', 'bias_list', 'lr_list',
             'batch_size_list', 'reg_val_list', 'reg_type_list',
             'momentum_val_list', 'nesterov', 'epochs_list']
key_names.sort()


def accuracy(net, x_set, y_set):
    len_set = len(x_set)
    corr_count = 0

    for i, entry in enumerate(x_set):

        res = net.forward(np.asarray(x_set[i]).flatten())
        thresh_res = 1 if res.item() >= 0.5 else 0

        if y_set[i].item() == thresh_res:
            corr_count += 1

    return corr_count/len_set


def cleanup_par_combo(combo_list):

    new_list = []
    reg_bool = False  # Used to ignore multiple combos when reg_val = 0
    mom_bool = False  # Used to ignore multiple combos when momentum_val = 0

    for combo in combo_list:

        if combo[key_names.index('reg_val_list')] == 0 and reg_bool:
            continue

        if combo[key_names.index('momentum_val_list')] == 0 and mom_bool:
            continue

        new_list.append(combo)

    return new_list


def grid_search(train_x, train_y, parameters_dict, k):

    # Sorting helps to check missing hyper-parameters
    keys = list(parameters_dict.keys())
    keys.sort()
    if keys != key_names:
        raise ValueError("Missing hyper parameters")

    # It makes a cartesian product between all hyper-parameters
    combo_list = list(it.product(*(parameters_dict[k] for k in key_names)))
    combo_list = cleanup_par_combo(combo_list)

    list_tasks = [delayed(kfold_cv)(
        combo, train_x, train_y, k, accuracy) for i, combo in enumerate(combo_list)]

    print(f"Number of tasks to execute: {len(list_tasks)}")

    # -2: Leave one core free
    results = Parallel(n_jobs=-2, verbose=50)(list_tasks)

    best_score = 0
    best_combo = None
    for result in results:

        cur_val_score = result[1]

        if best_score < cur_val_score:
            best_score = cur_val_score
            best_combo = result[2]

    best_combo = {key_names[i]:best_combo[i] for i in len(best_combo)}

    return best_score, best_combo


def kfold_cv(par_combo, x_mat, y_mat, k, metric):

    num_fold = x_mat.shape[0] // k
    tot_tr_score = 0
    tot_val_score = 0

    for i in range(num_fold):

        train_idx = np.concatenate(
            (np.arange(i*k), np.arange((i+1)*k, x_mat.shape[0])), axis=0)

        train_x = x_mat[train_idx]
        train_y = y_mat[train_idx]

        val_x = x_mat[i*k:(i+1)*k]
        val_y = y_mat[i*k:(i+1)*k]

        cur_net = train(train_x, train_y, par_combo)

        tot_tr_score += metric(cur_net, train_x, train_y)
        tot_val_score += metric(cur_net, val_x, val_y)

    return tot_tr_score/num_fold, tot_val_score/num_fold, par_combo


def train(train_x, train_y, combo):

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
                                       combo[key_names.index('momentum_val_list')],
                                       combo[key_names.index('nesterov')],
                                       combo[key_names.index('epochs_list')])

    gradient_descent.train(train_x, train_y)

    return network
