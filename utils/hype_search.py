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

    print("Setting up tasks...")
    combo_eval_list = random_sample_combo(par_combo_net, par_combo_opt, tr_handler,
                                          metric, n_jobs, n_folds, n_runs, val_handler,
                                          plotter)

    if median_stop is not None:
        best_results = median_stop_run(combo_eval_list, metric, median_stop, topk)
    else:
        best_results = bruteforce_run(combo_eval_list, 200, metric, topk)

    best_stats = [res.last_results for res in best_results]

    return best_stats


def hyperband_search(par_combo_net, par_combo_opt, tr_handler, metric,
                     n_folds, n_runs=10, val_handler=None, plotter=None, topk=50,
                     hb_R=10**3, hb_eta=3):

    # Initialisation
    s_max = int(np.log(hb_R) // np.log(hb_eta))
    B = (s_max+1)*hb_R
    best_results = []

    with Parallel(n_jobs=-2) as parallel:

        for s in range(s_max, -1, -1):

            print(f"Resource split countdown: {s}")
            n = int(np.ceil(B/hb_R * hb_eta**s/(s+1)))
            r = hb_R * hb_eta ** -s

            combo_eval_list = random_sample_combo(par_combo_net, par_combo_opt,
                                                  tr_handler, metric, n, n_folds,
                                                  n_runs, val_handler, plotter)

            # Note: compared with the standard implementation, this one doesn't
            # restart the training each time but "accumulates" epochs, therefore
            # hb_R no longer represents the max amount of epochs per combo
            for i in range(s+1):

                n_i = np.floor(n * hb_eta**-i)
                r_i = r * hb_eta**i

                print(f"Num tasks: {len(combo_eval_list)}, num epochs: {r_i}")

                jobs_list = generate_jobs(combo_eval_list, r_i)
                results = parallel(jobs_list)

                n_keep = int(n_i // hb_eta)
                combo_eval_list = compare_results(results, metric, n_keep)

                # Keep global best results found
                best_results += combo_eval_list
                best_results = compare_results(best_results, metric, topk)

    # Complete all training jobs if not already done
    best_results = bruteforce_run(best_results, 200, metric, topk)
    best_stats = [res.last_results for res in best_results]

    return best_stats

def random_sample_combo(par_combo_net, par_combo_opt, tr_handler, metric,
                        n_conf, n_folds, n_runs, val_handler, plotter):

    par_combo = dict(par_combo_net, **par_combo_opt)
    rng = np.random.default_rng()
    combo_eval_list = []

    for _ in range(n_conf):

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

    return combo_eval_list


def median_stop_run(eval_list, metric, step_epochs, topk):

    if step_epochs is None:
        raise ValueError("median_stop_run: step_epochs is None")

    iter_count = 0
    tr_complete = False

    with Parallel(n_jobs=-2, verbose=50) as parallel:

        while not tr_complete:

            print(f"Median stop: iteration {iter_count}, " +
                  f"tasks to run {len(eval_list)}, " +
                  f"epochs check interval {step_epochs}, " +
                  f"best to keep {topk}")

            job_list = generate_jobs(eval_list, step_epochs, True)
            res_list = parallel(job_list)
            clean_res = []

            # TODO: cleanup code and implement average accross epochs
            # Cleanup None evaluators
            for res in res_list:
                if res is None:
                    continue
                clean_res.append(res)

            res_list = clean_res

            avg_list = []
            for res in res_list:

                if res[1][2] is not None:
                    avg_list.append(res[1][2]) # Val avg
                else:
                    avg_list.append(res[1][0]) # Tr avg

            median_avg_val = np.median(avg_list)

            pruned_list = []

            for res in res_list:

                best_score = None

                if res[1][3] is not None:
                    best_score = res[1][3] # Val best
                else:
                    best_score = res[1][1] # Tr best
                print(best_score, median_avg_val)

                if metric.aim == 'max' and best_score > median_avg_val:
                    pruned_list.append(res[0])
                elif metric.aim == 'min' and best_score < median_avg_val:
                    pruned_list.append(res[0])

            tr_status_list = []

            for res in pruned_list:
                tr_status_list.append(res.last_results["tr_complete"])

            tr_complete = np.all(tr_status_list)

            eval_list = pruned_list
            iter_count += 1

    best_results = compare_results(eval_list, metric, topk)

    return best_results


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
def generate_jobs(eval_list, step_epochs=None, median_stop=False):

    job_list = []

    for ev in eval_list:
        job_list.append(delayed(eval_combo_search)(ev, step_epochs, median_stop))

    return job_list


# Idea: wrapper around ComboEvaluator used only in _search methods
def eval_combo_search(combo_eval, step_epochs=None, median_stop=False):

    res = None

    np.seterr(divide="raise", over="raise")

    try:
        combo_eval.eval(step_epochs)

        if median_stop:
            median_stop_stats = combo_eval.plotter.compute_median_stop_stats(combo_eval.metric)
            res = (combo_eval, median_stop_stats)
        else:
            res = combo_eval

    except FloatingPointError as e:
        print("FloatingPointError:", e, "(results discarded)")

    np.seterr(divide="print", over="print")

    return res
