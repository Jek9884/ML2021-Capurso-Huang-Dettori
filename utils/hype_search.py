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

def kfold_cv(par_combo_net, par_combo_opt, x_mat, y_mat, k, metric, n_runs=1,
             plot_bool=False):

    fold_size = int(np.ceil(x_mat.shape[0] / k))
    # Store all epochs results for all folds for both tr set and val set
    score_dict = {"tr": [], "val": []}
    pattern_idx = np.arange(x_mat.shape[0])

    if "lim_epochs" in par_combo_opt:
        lim_epochs = par_combo_opt["lim_epochs"]
    else:
        # Retrieve default from function paramenters
        lim_epochs = signature(GradientDescent).parameters["lim_epochs"].default

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
                                train_y, val_x, val_y, metric, lim_epochs)

            tmp_tr_scores.append(tr_score_list)
            tmp_val_scores.append(val_score_list)

        score_dict["tr"].append(np.average(tmp_tr_scores, axis=0))
        score_dict["val"].append(np.average(tmp_val_scores, axis=0))

    avg_tr_score = np.average(score_dict["tr"], axis=0)
    avg_val_score = np.average(score_dict["val"], axis=0)

    if plot_bool:

        plt.plot(range(0, lim_epochs), avg_tr_score, label="tr")
        plt.plot(range(0, lim_epochs), avg_val_score, label="val")

        plt.title(f"Results of kfold ({n_runs} runs)")
        plt.xlabel("Epochs")
        plt.ylabel(f"Metric ({metric.name})")
        plt.legend()
        plt.show(block=False)


    return avg_tr_score[-1], avg_val_score[-1], par_combo_net, par_combo_opt

def train_eval_fold(par_combo_net, par_combo_opt, train_x, train_y,
                    val_x, val_y, metric, lim_epochs):

    cur_net = Network(**par_combo_net)
    gradient_descent = GradientDescent(**par_combo_opt)

    results_tr_list = []
    results_val_list = []

    count = 0
    train_bool = True

    while train_bool:

        train_bool = gradient_descent.train(cur_net, train_x, train_y, 1)
        count += 1

        results_tr_list.append(eval_dataset(cur_net, train_x, train_y, metric))
        results_val_list.append(eval_dataset(cur_net, val_x, val_y, metric))

    # Increase length by repeating last result in order to 
    # have the same number of epochs in each output
    for i in range(lim_epochs - len(results_tr_list)):
        results_tr_list.append(results_tr_list[-1])
        results_val_list.append(results_val_list[-1])

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
