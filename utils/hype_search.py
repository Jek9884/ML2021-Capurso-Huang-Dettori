import itertools as it
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt

from network import Network
from optimizer import GradientDescent
from utils.helpers import average_non_std_mat


def grid_search(par_combo_net, par_combo_opt, train_x, train_y, metric,
                n_folds, n_runs=10):

    # Obtain a fixed order list of corresponding key-value pairs
    net_keys, net_values = zip(*par_combo_net.items())
    opt_keys, opt_values = zip(*par_combo_opt.items())

    # It makes a cartesian product between all hyper-parameters
    combo_list = list(it.product(*(net_values+opt_values)))  # Join the two list of params

    list_tasks = []

    for i, combo in enumerate(combo_list):

        combo_net = combo[0:len(net_keys)]
        combo_opt = combo[len(net_keys):]

        dict_net = {net_keys[i]: combo_net[i] for i in range(len(net_keys))}
        dict_opt = {opt_keys[i]: combo_opt[i] for i in range(len(opt_keys))}

        task = delayed(eval_model)(dict_net, dict_opt, train_x, train_y,
                                 metric, n_folds=n_folds, n_runs=n_runs)
        list_tasks.append(task)

    print(f"Number of tasks to execute: {len(list_tasks)}")

    # -2: Leave one core free
    # This is a barrier for the parallel computation
    results = Parallel(n_jobs=-2, verbose=50)(list_tasks)

    # List of best results containing: tr score, val score, combo
    best_results = compare_results(results, metric)
    best_score = (results[0][0], results[0][1])  # Return tr and val scores
    best_combo = [results[0][2], results[0][3]]

    return best_score, best_combo, best_results


def compare_results(results, metric, topk=5):

    if metric.name in ["miscl. error", "nll"]:
        best_score = np.inf
        sign = -1
    else:
        best_score = 0
        sign = 1

    # Check if validation score was computed, and use it to sort
    if results[0][1] != None:
        results = sorted(results, key=lambda i: i[1], reverse=(sign == 1))
    # Else sort by training score
    else:
        results = sorted(results, key=lambda i: i[0], reverse=(sign == 1))

    return results[:topk]


def eval_model(par_combo_net, par_combo_opt, x_mat, y_mat, metric, n_runs=10,
               n_folds=0, plot_bool=False):

    score_mat_dict = {"tr": []}

    for _ in range(n_runs):

        # Use kfold
        if n_folds > 0:
            tr_scores, val_scores = kfold_cv(par_combo_net, par_combo_opt,
                                             x_mat, y_mat, metric, n_folds)
            if "val" not in score_mat_dict:
                score_mat_dict["val"] = []

            score_mat_dict["tr"].append(tr_scores)
            score_mat_dict["val"].append(val_scores)
        # Train w/o validation set, used to estimate the avg performance
        else:
            tr_scores, val_scores = train_eval_dataset(par_combo_net, par_combo_opt,
                                                       x_mat, y_mat, metric)
            score_mat_dict["tr"].append(tr_scores)

    avg_score_dict = {}
    density_list = None
    perc_idx = None

    q_perc = 50

    for key in score_mat_dict:
        avg_score_dict[key], tmp_density_list = \
            average_non_std_mat(score_mat_dict[key])

        # Since the density of tr and val are the same initialise once
        if density_list is None:
            density_list = tmp_density_list

        if perc_idx is None:

            # Find index of q_perc point in the density data
            perc_val = np.percentile(density_list, q_perc, interpolation="nearest")
            perc_idx = np.argwhere(density_list == perc_val)[-1][0]

    if plot_bool:

        plot_dims = (2, 1)
        fig, axs = plt.subplots(*plot_dims, squeeze=False)

        for key in avg_score_dict:
            score_list = avg_score_dict[key]

            axs[0][0].plot(range(0, len(score_list)), score_list, label=key)

            if isinstance(score_list[perc_idx], np.ndarray):
                for i, val in enumerate(score_list[perc_idx]):
                    axs[0][0].scatter(perc_idx, val,
                                      label=f"{key} ({q_perc}-perc)", c="r")
            else:
                axs[0][0].scatter(perc_idx, score_list[perc_idx],
                                  label=f"{key} ({q_perc}-perc)", c="r")

        axs[1][0].plot(range(0, len(density_list)), density_list)
        axs[1][0].scatter(perc_idx, density_list[perc_idx],
                              label=f"{q_perc}-perc", c="r")

        axs[0][0].set_title(f"Model results ({n_runs} runs, {n_folds} kfolds)")
        axs[0][0].set_xlabel("Epochs")
        axs[0][0].set_ylabel(f"Metric ({metric.name})")
        axs[0][0].legend()

        axs[1][0].set_title(f"Distribution of model results ({n_runs} runs, {n_folds} kfolds)")
        axs[1][0].set_xlabel("Epochs")
        axs[1][0].set_ylabel(f"Number of datapoints")
        axs[1][0].legend()

        plt.show()

    if "val" in avg_score_dict:
        results = (avg_score_dict["tr"][perc_idx], avg_score_dict["val"][perc_idx],
                   par_combo_net, par_combo_opt)
    else:
        results = (avg_score_dict["tr"][perc_idx], None, par_combo_net, par_combo_opt)

    return results


def kfold_cv(par_combo_net, par_combo_opt, x_mat, y_mat, metric, n_folds):

    fold_size = int(np.ceil(x_mat.shape[0] / n_folds))
    tr_score_mat = []
    val_score_mat = []
    # Store all epochs results for all folds for both tr set and val set
    pattern_idx = np.arange(x_mat.shape[0])

    np.random.shuffle(pattern_idx)

    for i in range(n_folds):

        # Everything except i*fold_size:(i+1)*fold_size segment
        train_idx = np.concatenate(
            (pattern_idx[:i*fold_size],
             pattern_idx[(i+1)*fold_size:]), axis=0)

        train_x = x_mat[train_idx]
        train_y = y_mat[train_idx]

        val_x = x_mat[i*fold_size:(i+1)*fold_size]
        val_y = y_mat[i*fold_size:(i+1)*fold_size]

        tr_score_list, val_score_list =\
            train_eval_dataset(par_combo_net, par_combo_opt, train_x,
                               train_y, metric, val_x, val_y)

        tr_score_mat.append(tr_score_list)
        val_score_mat.append(val_score_list)

    # Lists shape = (num. epochs,), avg score of model accross folds
    avg_tr_score_list, _ = average_non_std_mat(tr_score_mat)
    avg_val_score_list, _ = average_non_std_mat(val_score_mat)

    return avg_tr_score_list, avg_val_score_list


def train_eval_dataset(par_combo_net, par_combo_opt, train_x, train_y,
                       metric, val_x=None, val_y=None):

    net = Network(**par_combo_net)
    gd = GradientDescent(**par_combo_opt)

    results_tr_list = []
    results_val_list = []

    train_bool = True

    while train_bool:

        train_bool = gd.train(net, train_x, train_y, 1)

        results_tr_list.append(eval_dataset(net, train_x, train_y, metric))

        # Check if the validation set is given as input
        if val_x is not None and val_y is not None:
            results_val_list.append(eval_dataset(net, val_x, val_y, metric))

    return results_tr_list, results_val_list


def eval_dataset(net, data_x, data_y, metric):

    net_pred = net.forward(data_x)

    if metric.name in ["nll", "squared"]:
        res = np.squeeze(metric(data_y, net_pred))
    elif metric.name in ["miscl. error"]:
        # Note: it works only with classification
        net_pred[net_pred < 0.5] = 0
        net_pred[net_pred >= 0.5] = 1

        res = metric(data_y, net_pred)
    else:
        raise ValueError(f"Metric not supported ({metric.name})")

    return res
