import itertools as it
from inspect import signature
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

from network import Network
from optimizer import GradientDescent


def grid_search(train_x, train_y, par_dict_net, par_dict_opt, k, metric, kfold_runs=10):

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
                                 train_x, train_y, k, metric, n_runs=kfold_runs)
        list_tasks.append(task)

    print(f"Number of tasks to execute: {len(list_tasks)}")

    # -2: Leave one core free
    # This is a barrier for the parallel computation
    results = Parallel(n_jobs=-2, verbose=50)(list_tasks)

    return compare_results_metric(results, metric)


def kfold_cv(par_combo_net, par_combo_opt, x_mat, y_mat, k, metric, n_runs=1,
             plot_bool=False):

    fold_size = int(np.ceil(x_mat.shape[0] / k))
    # Store all epochs results for all folds for both tr set and val set
    score_dict = {"tr": [], "val": []}
    pattern_idx = np.arange(x_mat.shape[0])

    np.random.shuffle(pattern_idx)

    for _ in range(n_runs):

        tmp_tr_scores = []
        tmp_val_scores = []

        for i in range(k):

            # Everything except i*fold_size:(i+1)*fold_size segment
            train_idx = np.concatenate(
                (pattern_idx[:i*fold_size],
                 pattern_idx[(i+1)*fold_size:]), axis=0)

            train_x = x_mat[train_idx]
            train_y = y_mat[train_idx]

            val_x = x_mat[i*fold_size:(i+1)*fold_size]
            val_y = y_mat[i*fold_size:(i+1)*fold_size]

            tr_score_list, val_score_list =\
                train_eval_fold(par_combo_net, par_combo_opt, train_x,
                                train_y, val_x, val_y, metric)

            tmp_tr_scores.append(tr_score_list)
            tmp_val_scores.append(val_score_list)

        score_dict["tr"].append(average_non_std_mat(tmp_tr_scores))
        score_dict["val"].append(average_non_std_mat(tmp_val_scores))

    avg_tr_score = average_non_std_mat(score_dict["tr"])
    avg_val_score = average_non_std_mat(score_dict["val"])

    if plot_bool:

        plt.plot(range(0, len(avg_tr_score)), avg_tr_score, label="tr")
        plt.plot(range(0, len(avg_val_score)), avg_val_score, label="val")

        plt.title(f"Results of kfold ({n_runs} runs)")
        plt.xlabel("Epochs")
        plt.ylabel(f"Metric ({metric.name})")
        plt.legend()
        plt.show(block=False)

    return avg_tr_score[-1], avg_val_score[-1], par_combo_net, par_combo_opt

def average_non_std_mat(value_mat):

    max_col = 0

    # Find longest row of non-standard matrix
    for row in value_mat:
        len_row = len(row)

        if len_row > max_col:
            max_col = len_row

    sum_vec = np.zeros((max_col,))  # Numerator
    count_vec = np.zeros((max_col,))  # Denominator

    for row in value_mat:
        len_row = len(row)
        # row with normalised length
        sum_vec += np.append(np.array(row), np.zeros((max_col - len_row,)))
        count_vec += np.append(np.ones((len_row,)), np.zeros((max_col - len_row,)))

    div_vec = np.divide(sum_vec, count_vec)

    return div_vec


def compare_results_metric(results, metric, topk=5):

    if metric.name in ["miscl. error", "nll"]:
        best_score = np.inf
        sign = -1
    else:
        best_score = 0
        sign = 1

    results = sorted(results, key=lambda i: i[1], reverse=(sign == 1))
    results = results[:topk]

    best_score = results[0][1]
    best_combo = [results[0][2], results[0][3]]

    return best_score, best_combo, results

def train_eval_fold(par_combo_net, par_combo_opt, train_x, train_y,
                    val_x, val_y, metric):

    cur_net = Network(**par_combo_net)
    gradient_descent = GradientDescent(**par_combo_opt)

    results_tr_list = []
    results_val_list = []

    train_bool = True

    while train_bool:

        train_bool = gradient_descent.train(cur_net, train_x, train_y, 1)

        results_tr_list.append(eval_dataset(cur_net, train_x, train_y, metric))
        results_val_list.append(eval_dataset(cur_net, val_x, val_y, metric))

    return results_tr_list, results_val_list


def eval_dataset(net, data_x, data_y, metric):

    net_pred = net.forward(data_x)

    if metric.name == "nll":
        res = np.squeeze(metric(data_y, net_pred))
    elif metric.name in ["miscl. error"]:
        # Note: it works only with classification
        net_pred[net_pred < 0.5] = 0
        net_pred[net_pred >= 0.5] = 1

        res = metric(data_y, net_pred)
    else:
        raise ValueError(f"Metric not supported in kfold ({metric.name})")

    return res
