import itertools as it
import time
from joblib import Parallel, delayed
import numpy as np

from network import Network
from optimizer import GradientDescent


def grid_search(train_x, train_y, par_dict_net, par_dict_opt, k, metric):

    # Obtain a fixed order list of corresponding key-value pairs
    net_keys, net_values = zip(*par_dict_net.items())
    opt_keys, opt_values = zip(*par_dict_opt.items())

    # It makes a cartesian product between all hyper-parameters
    combo_list = list(it.product(*(net_values+opt_values)))  # Join the two list of params

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

        c1 = best_score < cur_val_score and sign == 1
        c2 = best_score > cur_val_score and sign == -1

        if c1 or c2:
            best_score = cur_val_score
            best_combo = [result[2], result[3]]  # Return both dicts

    return best_score, best_combo

def kfold_cv(par_combo_net, par_combo_opt, x_mat, y_mat, k, metric, seed=42):

    fold_size = int(np.ceil(x_mat.shape[0] / k))
    tot_tr_score = 0
    tot_val_score = 0
    pattern_idx = np.arange(x_mat.shape[0])

    np.random.seed(seed)
    np.random.shuffle(pattern_idx)

    for i in range(k):

        # Everything except i*fold_size:(i+1)*fold_size segment
        train_idx = np.concatenate(
            (pattern_idx[:i*fold_size],
             pattern_idx[(i+1)*fold_size:]), axis=0)

        train_x = x_mat[train_idx]
        train_y = y_mat[train_idx]

        val_x = x_mat[i*fold_size:(i+1)*fold_size]
        val_y = y_mat[i*fold_size:(i+1)*fold_size]

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

    return tot_tr_score/k, tot_val_score/k, par_combo_net, par_combo_opt


def train(train_x, train_y, par_combo_net, par_combo_opt): # combo* are dicts

    network = Network(**par_combo_net)

    gradient_descent = GradientDescent(**par_combo_opt)

    gradient_descent.train(network, train_x, train_y)

    return network
