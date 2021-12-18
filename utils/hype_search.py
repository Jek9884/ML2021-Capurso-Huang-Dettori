import itertools as it
from joblib import Parallel, delayed
import numpy as np

import utils.helpers
from network import Network
from optimizer import GradientDescent
from utils.helpers import clean_combos


def grid_search(par_combo_net, par_combo_opt, train_x, train_y, metric,
                n_folds, n_runs=10):
    # Obtain a fixed order list of corresponding key-value pairs
    net_keys, net_values = zip(*par_combo_net.items())
    opt_keys, opt_values = zip(*par_combo_opt.items())

    # It makes a cartesian product between all hyper-parameters
    combo_list = list(it.product(*(net_values + opt_values)))  # Join the two list of params

    combo_list = clean_combos(par_combo_net, par_combo_opt, combo_list)

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

    return best_results


def compare_results(results, metric, topk=5):
    if metric.name in ["miscl. error", "nll"]:
        sign = -1
    elif metric.name in ["accuracy"]:
        sign = 1
    else:
        raise ValueError(f"Metric not supported {metric.name}")

    # TODO: consider adding comparison of std for best result
    # Check if the mean of validation score was computed, and use it to sort
    if results[0]['score_val'] is not None:
        results = sorted(results, key=lambda i: i['score_val'][0], reverse=(sign == 1))
    # Else sort by mean of training score
    else:
        results = sorted(results, key=lambda i: i['score_tr'][0], reverse=(sign == 1))

    return results[:topk]


def eval_model(par_combo_net, par_combo_opt, x_mat, y_mat, metric, n_runs=10,
               n_folds=0, plotter=None):

    score_results_dict = {"tr": []}  # Used to compute mean and std
    train_epoch_list = []

    for _ in range(n_runs):

        # Use kfold
        if n_folds > 0:

            avg_tr_res, avg_val_res, n_epochs = \
                kfold_cv(par_combo_net, par_combo_opt, x_mat, y_mat, metric,
                         n_folds, plotter=plotter)

            if "val" not in score_results_dict:
                score_results_dict["val"] = []

            score_results_dict["tr"].append(avg_tr_res)
            score_results_dict["val"].append(avg_val_res)
            train_epoch_list.append(n_epochs)

        # Train w/o validation set, used to estimate the avg performance
        else:
            tr_scores, _, n_epochs = train_eval_dataset(par_combo_net,
                                                        par_combo_opt,
                                                        x_mat, y_mat, metric,
                                                        plotter=plotter)
            score_results_dict["tr"].append(tr_scores[-1])
            train_epoch_list.append(n_epochs)

    avg_epochs = np.average(train_epoch_list)
    score_stats_dict = {"tr": (0, 0)}

    # Take average and std wrt runs
    for key in score_results_dict:
        mean = np.average(score_results_dict[key], axis=0)
        std = np.std(score_results_dict[key], axis=0)
        score_stats_dict[key] = (mean, std)

    if "val" in score_results_dict:
        results = {'combo_net': par_combo_net,
                   'combo_opt': par_combo_opt,
                   'score_tr': score_stats_dict["tr"],
                   'score_val': score_stats_dict["val"],
                   'metric': metric.name,
                   'epochs': avg_epochs}
    else:
        results = {'combo_net': par_combo_net,
                   'combo_opt': par_combo_opt,
                   'score_tr': score_stats_dict["tr"],
                   'score_val': None,
                   'metric': metric.name,
                   'epochs': avg_epochs}

    return results


def kfold_cv(par_combo_net, par_combo_opt, x_mat, y_mat, metric, n_folds, plotter=None):
    fold_size = int(np.floor(x_mat.shape[0] / n_folds))
    pattern_idx = np.arange(x_mat.shape[0])

    # Used to take average of the final nets result
    avg_tr_score = 0
    avg_val_score = 0
    avg_epochs = 0

    np.random.shuffle(pattern_idx)

    for i in range(n_folds):
        # Everything except i*fold_size:(i+1)*fold_size segment
        train_idx = np.concatenate(
            (pattern_idx[:i * fold_size],
             pattern_idx[(i + 1) * fold_size:]), axis=0)

        train_x = x_mat[train_idx]
        train_y = y_mat[train_idx]

        val_x = x_mat[i * fold_size:(i + 1) * fold_size]
        val_y = y_mat[i * fold_size:(i + 1) * fold_size]

        tr_score_list, val_score_list, n_epochs = \
            train_eval_dataset(par_combo_net, par_combo_opt, train_x,
                               train_y, metric, val_x, val_y, plotter=plotter)

        avg_tr_score += tr_score_list[-1]
        avg_val_score += val_score_list[-1]
        avg_epochs += n_epochs

    avg_tr_score /= n_folds
    avg_val_score /= n_folds
    avg_epochs /= n_folds

    return avg_tr_score, avg_val_score, avg_epochs


def train_eval_dataset(par_combo_net, par_combo_opt, train_x, train_y,
                       metric, val_x=None, val_y=None, plotter=None):
    net = Network(**par_combo_net)
    gd = GradientDescent(**par_combo_opt)

    epoch_res_tr_list = []
    epoch_res_val_list = []

    train_bool = True

    while train_bool:

        train_bool = gd.train(net, train_x, train_y, 1, plotter=plotter)

        epoch_res_tr_list.append(eval_dataset(net, train_x, train_y, metric))

        # Check if the validation set is given as input
        if val_x is not None and val_y is not None:
            epoch_res_val_list.append(eval_dataset(net, val_x, val_y, metric))

            if plotter is not None:
                plotter.add_lr_curve_datapoint(net, val_x, val_y, "val")

    if plotter is not None:
        plotter.add_new_plotline()

    return epoch_res_tr_list, epoch_res_val_list, gd.epoch_count


# Hp: all outputs from metric must be arrays
def eval_dataset(net, data_x, data_y, metric):
    net_pred = net.forward(data_x)

    if metric.name in ["nll", "squared"]:
        res = metric(data_y, net_pred)

    elif metric.name in ["miscl. error"]:
        # Note: it works only with classification
        net_pred[net_pred < 0.5] = 0
        net_pred[net_pred >= 0.5] = 1

        res = metric(data_y, net_pred)
    else:
        raise ValueError(f"Metric not supported ({metric.name})")

    return res
