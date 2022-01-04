import itertools as it
import copy
from joblib import Parallel, delayed
import numpy as np

import utils.helpers
from network import Network
from optimizer import GradientDescent
from utils.helpers import clean_combos
from utils.data_handler import DataHandler


def grid_search(par_combo_net, par_combo_opt, train_handler, metric,
                n_folds, n_runs=10, plotter=None):

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

        # Used to save images
        task_plotter = None

        # Use given plotter as model for all runs
        if plotter is not None:
            task_plotter = copy.deepcopy(plotter)

        task = delayed(eval_model_search)(dict_net, dict_opt, train_handler, metric,
                                          n_folds=n_folds, n_runs=n_runs,
                                          plotter=task_plotter, save_plot=True)
        list_tasks.append(task)

    print(f"Number of tasks to execute: {len(list_tasks)}")

    # -2: Leave one core free
    # This is a barrier for the parallel computation
    results = Parallel(n_jobs=-2, verbose=50)(list_tasks)

    # List of best results containing: tr score, val score, combo
    best_results = compare_results(results, metric)

    return best_results


def stoch_search(par_combo_net, par_combo_opt, train_handler, metric, n_jobs,
                n_folds, n_runs=10, plotter=None):

    par_combo = dict(par_combo_net, **par_combo_opt)
    rng = np.random.default_rng()
    list_tasks = []

    for _ in range(n_jobs):

        dict_net = {}
        dict_opt = {}

        for key, arr in par_combo.items():

            par_val = rng.choice(arr)

            if key in par_combo_net:
                dict_net[key] = par_val
            elif key in par_combo_opt:
                dict_opt[key] = par_val

        # Used to save images
        task_plotter = None

        # Use given plotter as model for all runs
        if plotter is not None:
            task_plotter = copy.deepcopy(plotter)

        task = delayed(eval_model_search)(dict_net, dict_opt, train_handler, metric,
                                          n_folds=n_folds, n_runs=n_runs,
                                          plotter=task_plotter, save_plot=True)
        list_tasks.append(task)

    print(f"Number of tasks to execute: {n_jobs}")

    results = Parallel(n_jobs=-2, verbose=50)(list_tasks)

    best_results = compare_results(results, metric)

    return best_results


def compare_results(results, metric, topk=50):

    if metric.name in ["miscl. error", "nll"]:
        sign = -1
    elif metric.name in ["accuracy"]:
        sign = 1
    else:
        raise ValueError(f"Metric not supported {metric.name}")

    # Remove all discarded results
    clean_results = []
    for res in results:
        if res is None:
            continue
        clean_results.append(res)
    results = clean_results

    # TODO: consider adding comparison of std for best result
    # Check if the mean of validation score was computed, and use it to sort
    if results[0]['score_val'] is not None:
        results = sorted(results, key=lambda i: i['score_val'][0], reverse=(sign == 1))
    # Else sort by mean of training score
    else:
        results = sorted(results, key=lambda i: i['score_tr'][0], reverse=(sign == 1))

    if topk is not None:
        results = results[:topk]

    print(f"Number of results kept: {len(results)}")

    return results


def eval_model_search(par_combo_net, par_combo_opt, train_handler, metric,
                      val_handler=None, n_runs=10, n_folds=0, plotter=None,
                      save_plot=False):

    np.seterr(divide="raise", over="raise")
    results = None

    try:
        results = eval_model(par_combo_net, par_combo_opt, train_handler, metric,
                   val_handler, n_runs, n_folds, plotter, save_plot)
    except FloatingPointError as e:
        print("FloatingPointError:", e, "(results discarded)")

    np.seterr(divide="print", over="print")
    return results


def eval_model(par_combo_net, par_combo_opt, train_handler, metric,
               val_handler=None, n_runs=10, n_folds=0, plotter=None,
               save_plot=False):

    score_results_dict = {"tr": []}  # Used to compute mean and std
    train_epoch_list = []
    train_age_list = []

    for _ in range(n_runs):

        # Use kfold
        if n_folds > 0:

            avg_tr_res, avg_val_res, n_epochs, age = \
                kfold_cv(par_combo_net, par_combo_opt, train_handler, metric,
                         n_folds, plotter=plotter)

            if "val" not in score_results_dict:
                score_results_dict["val"] = []

            score_results_dict["tr"].append(avg_tr_res)
            score_results_dict["val"].append(avg_val_res)
            train_epoch_list.append(n_epochs)
            train_age_list.append(age)

        # Train, then test on validation/test set
        elif val_handler is not None:

            tr_scores, val_scores, n_epochs, age = \
                train_eval_dataset(par_combo_net, par_combo_opt, train_handler,
                                   metric, val_handler=val_handler, plotter=plotter)

            if "val" not in score_results_dict:
                score_results_dict["val"] = []

            score_results_dict["tr"].append(tr_scores[-1])
            score_results_dict["val"].append(val_scores[-1])
            train_epoch_list.append(n_epochs)
            train_age_list.append(age)

        # Train w/o validation set, used to estimate the avg performance
        else:
            tr_scores, _, n_epochs, age = \
                train_eval_dataset(par_combo_net, par_combo_opt, train_handler,
                                   metric, plotter=plotter)
            score_results_dict["tr"].append(tr_scores[-1])
            train_epoch_list.append(n_epochs)
            train_age_list.append(age)

    avg_epochs = np.average(train_epoch_list)
    avg_age = np.average(train_age_list)
    score_stats_dict = {"tr": (0, 0), "val": None}

    # Take average and std wrt runs
    for key in score_results_dict:
        mean = np.average(score_results_dict[key], axis=0)
        std = np.std(score_results_dict[key], axis=0)
        score_stats_dict[key] = (mean, std)

    figure = None

    if save_plot and plotter is not None:
        figure = plotter.build_plot()

    results = {'combo_net': par_combo_net,
               'combo_opt': par_combo_opt,
               'score_tr': score_stats_dict["tr"],
               'score_val': score_stats_dict["val"],
               'metric': metric.name,
               'epochs': avg_epochs,
               'age': avg_age,
               'figure': figure}

    return results


def kfold_cv(par_combo_net, par_combo_opt, train_handler, metric, n_folds, plotter=None):

    x_mat = train_handler.data_x
    y_mat = train_handler.data_y

    fold_size = int(np.floor(x_mat.shape[0] / n_folds))
    pattern_idx = np.arange(x_mat.shape[0])

    # Used to take average of the final nets result
    avg_tr_score = 0
    avg_val_score = 0
    avg_epochs = 0
    avg_age = 0

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

        val_handler = DataHandler(val_x, val_y)

        tr_score_list, val_score_list, n_epochs, age = \
            train_eval_dataset(par_combo_net, par_combo_opt, train_handler,
                               metric, val_handler, plotter=plotter)

        avg_tr_score += tr_score_list[-1]
        avg_val_score += val_score_list[-1]
        avg_epochs += n_epochs
        avg_age += age

    avg_tr_score /= n_folds
    avg_val_score /= n_folds
    avg_epochs /= n_folds
    avg_age /= n_folds

    return avg_tr_score, avg_val_score, avg_epochs, avg_age


def train_eval_dataset(par_combo_net, par_combo_opt, train_handler,
                       metric, val_handler=None, plotter=None):

    net = Network(**par_combo_net)
    gd = GradientDescent(**par_combo_opt)

    epoch_res_tr_list = []
    epoch_res_val_list = []

    train_bool = True

    while train_bool:

        # Train 1 epoch at a time to plot the evolution of the lr curve
        train_bool = gd.train(net, train_handler, 1, plotter=plotter)

        epoch_res_tr_list.append(eval_dataset(net, train_handler, metric, True))

        # Check if the validation set is given as input
        if val_handler is not None:
            epoch_res_val_list.append(eval_dataset(net, val_handler, metric, False))

            if plotter is not None:
                plotter.add_lr_curve_datapoint(net, val_handler.data_x,
                                               val_handler.data_y, "val")

    if plotter is not None:
        plotter.add_new_plotline()

    return epoch_res_tr_list, epoch_res_val_list, gd.epoch_count, gd.age_count


# Hp: all outputs from metric must be arrays
def eval_dataset(net, data_handler, metric, training):

    data_x = data_handler.data_x
    data_y = data_handler.data_y
    net_pred = net.forward(data_x, training)

    if metric.name in ["nll", "squared"]:
        res = metric(data_y, net_pred, reduce_bool=True)

    elif metric.name in ["miscl. error"]:
        # Note: it works only with classification
        net_pred[net_pred < 0.5] = 0
        net_pred[net_pred >= 0.5] = 1

        res = metric(data_y, net_pred)
    else:
        raise ValueError(f"Metric not supported ({metric.name})")

    return res
