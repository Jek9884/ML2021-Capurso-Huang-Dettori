import itertools as it
import time
from joblib import Parallel, delayed
import numpy as np

from network import Network
from optimizer import GradientDescent


def cleanup_par_combo(combo_list, key_list):

    new_list = []
    reg_bool = False  # Used to ignore multiple combos when reg_val = 0
    mom_bool = False  # Used to ignore multiple combos when momentum_val = 0

    for combo in combo_list:

        if combo[key_list.index('reg_val')] == 0 and reg_bool:
            continue

        if combo[key_list.index('momentum_val')] == 0 and mom_bool:
            continue

        new_list.append(combo)

    return new_list


def grid_search(train_x, train_y, par_dict_net, par_dict_opt, k, metric):

    # Obtain a fixed order list of corresponding key-value pairs
    net_keys, net_values = zip(*par_dict_net.items())
    opt_keys, opt_values = zip(*par_dict_opt.items())

    # It makes a cartesian product between all hyper-parameters
    combo_list = list(it.product(*(net_values+opt_values)))  # Join the two list of params
    combo_list = cleanup_par_combo(combo_list, net_keys+opt_keys)

    list_tasks = []

    for i, combo in enumerate(combo_list):

        combo_net = combo[0:len(net_keys)]
        combo_opt = combo[len(net_keys):]

        dict_net = {net_keys[i]: combo_net[i] for i in range(len(net_keys))}
        dict_opt = {opt_keys[i]: combo_opt[i] for i in range(len(opt_keys))}

        task = delayed(kfold_cv)(dict_net, dict_opt,
                                 train_x, train_y, k, metric)
        list_tasks.append(task)

    print(f"Number of tasks to execute: {len(list_tasks)}")

    # -2: Leave one core free
    # This is a barrier for the parallel computation
    results = Parallel(n_jobs=-2, verbose=0)(list_tasks)

    return compare_results_metric(results, metric)


def compare_results_metric(results, metric):

    if metric.name in ["miscl. error", "nll"]:
        best_score = np.inf
        sign = -1
    else:
        best_score = 0
        sign = 1

    best_combo = None

    # This is a barrier for the parallel computation
    for result in results:

        cur_val_score = result[1]

        if metric.name in ["nll"]:
            cur_val_score = np.average(cur_val_score, axis=0)

        c1 = best_score < cur_val_score and sign == 1
        c2 = best_score > cur_val_score and sign == -1

        if c1 or c2:
            best_score = cur_val_score
            best_combo = [result[2], result[3]]  # Return both dicts

    return best_score, best_combo

def kfold_cv(par_combo_net, par_combo_opt, x_mat, y_mat, k, metric, seed=42):

    num_fold = x_mat.shape[0] // k
    tot_tr_score = 0
    tot_val_score = 0
    pattern_idx = np.arange(x_mat.shape[0])

    np.random.seed(seed)
    np.random.shuffle(pattern_idx)

    for i in range(num_fold):

        # Everything except i*k:(i+1)*k segment
        train_idx = np.concatenate(
            (pattern_idx[:i*k],
             pattern_idx[(i+1)*k:]), axis=0)

        train_x = x_mat[train_idx]
        train_y = y_mat[train_idx]

        val_x = x_mat[i*k:(i+1)*k]
        val_y = y_mat[i*k:(i+1)*k]

        cur_net = train(train_x, train_y, par_combo_net, par_combo_opt)

        # Evaluate the net
        net_pred_tr = cur_net.forward(train_x)
        net_pred_tr[net_pred_tr < 0.5] = 0
        net_pred_tr[net_pred_tr >= 0.5] = 1

        net_pred_val = cur_net.forward(val_x)
        net_pred_val[net_pred_val < 0.5] = 0
        net_pred_val[net_pred_val >= 0.5] = 1

        tot_tr_score += metric(train_y, net_pred_tr)
        tot_val_score += metric(val_y, net_pred_val)

    return tot_tr_score/num_fold, tot_val_score/num_fold, par_combo_net, par_combo_opt


def train(train_x, train_y, par_combo_net, par_combo_opt): # combo* are dicts

    network = Network(**par_combo_net)

    gradient_descent = GradientDescent(**par_combo_opt)

    gradient_descent.train(network, train_x, train_y)

    return network
