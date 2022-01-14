import itertools as it
import copy
from joblib import Parallel, delayed
import numpy as np

from utils.helpers import clean_combos
from utils.evaluation import ComboEvaluator


def grid_search(par_combo_net, par_combo_opt, tr_handler, metric, n_folds,
                n_runs=10, val_handler=None, plotter=None, topk=50,
                median_stop=None):

    # Obtain a fixed order list of corresponding key-value pairs
    net_keys, net_values = zip(*par_combo_net.items())
    opt_keys, opt_values = zip(*par_combo_opt.items())

    print("Cleaning-up combos...")
    # It makes a cartesian product between all hyper-parameters
    combo_list = list(it.product(*(net_values + opt_values)))
    combo_list = clean_combos(par_combo_net, par_combo_opt, combo_list)

    combo_eval_list = []

    print("Setting up tasks...")
    for i, combo in enumerate(combo_list):
        combo_net = combo[0:len(net_keys)]
        combo_opt = combo[len(net_keys):]

        dict_net = {net_keys[i]: combo_net[i] for i in range(len(net_keys))}
        dict_opt = {opt_keys[i]: combo_opt[i] for i in range(len(opt_keys))}

        # Use the given plotter as model for all results
        # All datapoints are collected, but the plots are generated
        # only for the best results at save-time/show-time
        res_plotter = copy.deepcopy(plotter)

        c_ev = ComboEvaluator(dict_net, dict_opt, tr_handler, metric,
                              n_folds=n_folds, n_runs=n_runs,
                              val_handler=val_handler, plotter=res_plotter)

        combo_eval_list.append(c_ev)

    if median_stop is not None:
        best_results = median_stop_run(combo_eval_list, metric, median_stop, topk)
    else:
        best_results = bruteforce_run(combo_eval_list, 200, metric, topk)

    best_stats = [res.last_results for res in best_results]

    return best_stats


def stoch_search(par_combo_net, par_combo_opt, tr_handler, metric, n_jobs,
                 n_folds, n_runs=10, val_handler=None, plotter=None, topk=50,
                 median_stop=None):

    par_combo = dict(par_combo_net, **par_combo_opt)
    rng = np.random.default_rng()
    combo_eval_list = []

    print("Setting up tasks...")

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

        c_ev = ComboEvaluator(dict_net, dict_opt, tr_handler, metric,
                              n_folds=n_folds, n_runs=n_runs,
                              val_handler=val_handler, plotter=res_plotter)

        combo_eval_list.append(c_ev)

    if median_stop is not None:
        best_results = median_stop_run(combo_eval_list, metric, median_stop, topk)
    else:
        best_results = bruteforce_run(combo_eval_list, 200, metric, topk)

    best_stats = [res.last_results for res in best_results]

    return best_stats


def median_stop_run(eval_list, metric, step_epochs, topk):

    raise RuntimeError("Not implemented yet")

    if step_epochs is None:
        raise ValueError("median_stop_run: step_epochs is None")

    iter_count = 0

    with Parallel(n_jobs=-2, verbose=50) as parallel:

        while len(eval_list) > topk:

            print(f"Median stop: iteration {iter_count}, " +
                  f"tasks to run {len(eval_list)}, " +
                  f"epochs check interval {step_epochs}," +
                  f"best to keep {topk}")

            job_list = generate_jobs(eval_list, step_epochs)
            res_list = parallel(job_list)
            clean_res = []

            # TODO: cleanup code and implement average accross epochs
            for res in res_list:
                if res is None:
                    continue
                clean_res.append(res)

            res_list = clean_res

            if res_list[0].last_results['score_val'] is not None:
                score_type = 'score_val'
            else:
                score_type = 'score_tr'

            avg_list = [ev.last_results[score_type]['avg'] for ev in res_list]
            median_avg_val = np.median(avg_list)

            pruned_list = []

            for res in res_list:

                if metric.aim == 'max' and\
                        res.last_results[score_type][metric.aim] > median_avg_val:
                    pruned_list.append(res)
                elif metric.aim == 'min' and\
                        res.last_results[score_type][metric.aim] < median_avg_val:
                    pruned_list.append(res)

            print(avg_list)
            print(median_avg_val)

            eval_list = pruned_list
            iter_count += 1

    return eval_list

# Idea: run the configurations and compare the final results, no early stop
# In order to optimise memory usage, define a chunk size and cleanup
# the results after chunk size tasks are completed
def bruteforce_run(eval_list, chunk_size, metric, topk):

    if topk > chunk_size:
        raise ValueError("bruteforce_run: topk is too big compared to chunk_size")

    res_list = []
    n_evals = len(eval_list)

    if chunk_size == -1:
        n_chunks = 1
    else:
        n_chunks = int(np.ceil(n_evals / chunk_size))

    # -2: Leave one core free
    with Parallel(n_jobs=-2, verbose=50) as parallel:

        for i in range(n_chunks):

            cur_start = i*chunk_size
            cur_stop = cur_start + chunk_size
            chunk_list = eval_list[cur_start: cur_stop]

            print(f"Number of tasks left: {len(eval_list[cur_start:])}")

            job_list = generate_jobs(chunk_list)
            res_list += parallel(job_list)

            # Keep the best results
            res_list = compare_results(res_list, metric, topk)

    return res_list


# Idea: order results and keep the k best ones
def compare_results(results, metric, topk=None):

    # Remove all discarded results
    clean_results = []

    for combo_ev in results:
        if combo_ev is None:
            continue

        clean_results.append(combo_ev)

    results = clean_results

    # Check if the validation score was computed, and use median to sort the results
    if results[0].last_results['score_val'] is not None:
        results = sorted(results, key=lambda ev: ev.last_results['score_val']["perc_50"],
                         reverse=(metric.aim == 'max'))
    # Else sort by training score median
    else:
        results = sorted(results, key=lambda ev: ev.last_results['score_tr']["perc_50"],
                         reverse=(metric.aim == 'max'))

    if topk is not None:
        results = results[:topk]

    return results


# Create tasks for Parallel starting from a list of ComboEvaluator
def generate_jobs(eval_list, step_epochs=None):

    job_list = []

    for ev in eval_list:
        job_list.append(delayed(eval_combo_search)(ev, step_epochs))

    return job_list


# Idea: wrapper around ComboEvaluator used only in _search methods
def eval_combo_search(combo_eval, step_epochs=None):

    res = None

    np.seterr(divide="raise", over="raise")

    try:
        combo_eval.eval(step_epochs)
        res = combo_eval

    except FloatingPointError as e:
        print("FloatingPointError:", e, "(results discarded)")

    np.seterr(divide="print", over="print")

    return res
