import itertools as it
import copy
from joblib import Parallel, delayed
import numpy as np

import utils.helpers
from utils.helpers import clean_combos
from utils.evaluation import ModelEvaluator


def grid_search(par_combo_net, par_combo_opt, train_handler, metric,
                n_folds, n_runs=10, val_handler=None, plotter=None, topk=50):

    # Obtain a fixed order list of corresponding key-value pairs
    net_keys, net_values = zip(*par_combo_net.items())
    opt_keys, opt_values = zip(*par_combo_opt.items())

    # It makes a cartesian product between all hyper-parameters
    combo_list = list(it.product(*(net_values + opt_values)))

    combo_list = clean_combos(par_combo_net, par_combo_opt, combo_list)

    list_tasks = []

    for i, combo in enumerate(combo_list):
        combo_net = combo[0:len(net_keys)]
        combo_opt = combo[len(net_keys):]

        dict_net = {net_keys[i]: combo_net[i] for i in range(len(net_keys))}
        dict_opt = {opt_keys[i]: combo_opt[i] for i in range(len(opt_keys))}

        # Use the given plotter as model for all results
        # All datapoints are collected, but the plots are generated
        # only for the best results at save-time/show-time
        res_plotter = copy.deepcopy(plotter)

        task = delayed(eval_model_search)(dict_net, dict_opt, train_handler, metric,
                                          n_folds=n_folds, n_runs=n_runs,
                                          val_handler=val_handler, plotter=res_plotter)
        list_tasks.append(task)

    print(f"Number of tasks to execute: {len(list_tasks)}")

    # -2: Leave one core free
    # This is a barrier for the parallel computation
    results = Parallel(n_jobs=-2, verbose=50)(list_tasks)

    # List of best results containing: tr score, val score, combo
    best_results = compare_results(results, metric, topk)

    return best_results


def stoch_search(par_combo_net, par_combo_opt, train_handler, metric, n_jobs,
                 n_folds, n_runs=10, val_handler=None, plotter=None, topk=50):

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

        # Use the given plotter as model for all results
        # All datapoints are collected, but the plots are generated
        # only for the best results at save-time/show-time
        res_plotter = copy.deepcopy(plotter)

        task = delayed(eval_model_search)(dict_net, dict_opt, train_handler, metric,
                                          n_folds=n_folds, n_runs=n_runs,
                                          val_handler=val_handler, plotter=res_plotter)
        list_tasks.append(task)

    print(f"Number of tasks to execute: {n_jobs}")

    results = Parallel(n_jobs=-2, verbose=50)(list_tasks)

    best_results = compare_results(results, metric, topk)

    return best_results


# Idea: order results and keep the k best ones
def compare_results(results, metric, topk=None):

    # Sign used to understand if min or max is better
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

    # Check if the validation score was computed, and use it to sort the results
    if results[0]['score_val'] is not None:
        results = sorted(results, key=lambda i: i['score_val']["perc_50"], reverse=(sign == 1))
    # Else sort by training score
    else:
        results = sorted(results, key=lambda i: i['score_tr']["perc_50"], reverse=(sign == 1))

    if topk is not None:
        results = results[:topk]

    return results


# Idea: wrapper around eval_model used only in _search methods since even if
# a thread, fails the process will go on
def eval_model_search(par_combo_net, par_combo_opt, train_handler, metric,
                      val_handler=None, n_folds=0, n_runs=10, plotter=None):

    np.seterr(divide="raise", over="raise")
    result = None

    try:
        ev_inst = ModelEvaluator(par_combo_net, par_combo_opt, train_handler, metric,
                                 val_handler, n_folds, n_runs, plotter)
        result = ev_inst.eval()

    except FloatingPointError as e:
        print("FloatingPointError:", e, "(results discarded)")

    np.seterr(divide="print", over="print")

    return result
