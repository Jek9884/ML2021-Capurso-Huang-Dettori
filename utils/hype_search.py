import itertools as it
import copy, random
from joblib import Parallel, delayed
import numpy as np

from utils.helpers import clean_combos
from utils.evaluation import ComboEvaluator

"""
    Perform grid search. Returns a list of dictionaries containing the stats of the model and the model combo
    
    Parameters:
        -par_combo_net: dictionary containing a list for each hyperparameter of network
        -par_combo_opt: dictionary containing a list for each hyperparameter of optimizer
        -tr_handler: object containing both train_x and train_y (instance of DataHandler)
        -metric: metric function used to evaluate the performance of a model
        -n_folds: k value for the k-fold validation
        -n_runs: number of runs to perform
        -val_handler: object containing both valid_x and valid_y (instance of DataHandler) 
            (only used when n_folds not specified)
        -plotter: object that handles models plots
        -topk: number of best models to return. If None keeps everything
"""


def grid_search(par_combo_net, par_combo_opt, tr_handler, metric, n_folds,
                n_runs=10, val_handler=None, plotter=None, topk=50):
    # obtain a fixed order list of corresponding key-value pairs
    net_keys, net_values = zip(*par_combo_net.items())
    opt_keys, opt_values = zip(*par_combo_opt.items())

    print("Cleaning-up combos...")
    # it makes a cartesian product between all hyperparameters
    combo_list = list(it.product(*(net_values + opt_values)))
    combo_list = clean_combos(par_combo_net, par_combo_opt, combo_list)

    combo_eval_list = []

    print("Setting up tasks...")
    for i, combo in enumerate(combo_list):
        combo_net = combo[0:len(net_keys)]
        combo_opt = combo[len(net_keys):]

        dict_net = {net_keys[i]: combo_net[i] for i in range(len(net_keys))}
        dict_opt = {opt_keys[i]: combo_opt[i] for i in range(len(opt_keys))}

        # use the given plotter as model for all results
        # all datapoints are collected, but the plots are generated
        # only for the best results at save-time/show-time
        res_plotter = None

        if plotter is not None:
            res_plotter = copy.deepcopy(plotter)

        c_ev = ComboEvaluator(dict_net, dict_opt, tr_handler, metric,
                              n_folds=n_folds, n_runs=n_runs,
                              val_handler=val_handler, plotter=res_plotter)

        combo_eval_list.append(c_ev)

    # the chunk_size of 200 has been chosen empirically
    best_results = bruteforce_run(combo_eval_list, 200, metric, topk)
    best_stats = [res.last_results for res in best_results]

    return best_stats


"""
    Perform stochastic search. Returns a list of dictionaries containing the stats of the model and the model combo
    
    Parameters:
        -par_combo_net: dictionary containing a list for each hyperparameter of network
        -par_combo_opt: dictionary containing a list for each hyperparameter of optimizer
        -tr_handler: object containing both train_x and train_y (instance of DataHandler)
        -metric: metric function used to evaluate the performance of a model
        -n_jobs: number of combos to sample
        -n_folds: k value for the k-fold validation
        -n_runs: number of runs to perform
        -val_handler: object containing both valid_x and valid_y (instance of DataHandler) 
            (only used when n_folds not specified)
        -plotter: object that handles models plots
        -topk: number of best models to return. If None keeps everything
"""


def stoch_search(par_combo_net, par_combo_opt, tr_handler, metric, n_jobs,
                 n_folds, n_runs=10, val_handler=None, plotter=None, topk=50):
    print("Setting up tasks...")
    combo_eval_list = random_sample_combo(par_combo_net, par_combo_opt, tr_handler,
                                          metric, n_jobs, n_folds, n_runs, val_handler,
                                          plotter)

    best_results = bruteforce_run(combo_eval_list, 200, metric, topk)
    best_stats = [res.last_results for res in best_results]

    return best_stats


"""
    Perform hyperband-like search. Returns a list of dictionaries containing the stats of the model and the model combo
    
    Parameters:
        -par_combo_net: dictionary containing a list for each hyperparameter of network
        -par_combo_opt: dictionary containing a list for each hyperparameter of optimizer
        -tr_handler: object containing both train_x and train_y (instance of DataHandler)
        -metric: metric function used to evaluate the performance of a model
        -n_folds: k value for the k-fold validation
        -n_runs: number of runs to perform
        -val_handler: object containing both valid_x and valid_y (instance of DataHandler) 
            (only used when n_folds not specified)
        -plotter: object that handles models plots
        -topk: number of best models to return. If None keeps everything
        -hb_r: max number of epochs allocatable to a single model 
        -hb_eta: scalar that specifies how aggressively the combos are pruned 
"""


def hyperband_search(par_combo_net, par_combo_opt, tr_handler, metric,
                     n_folds, n_runs=10, val_handler=None, plotter=None, topk=50,
                     hb_r=10 ** 3, hb_eta=3):
    # initialisation
    s_max = int(np.log(hb_r) // np.log(hb_eta))
    B = (s_max + 1) * hb_r
    best_results = []

    with Parallel(n_jobs=-2) as parallel:

        for s in range(s_max, -1, -1):

            print(f"Resource split countdown: {s}")
            n = int(np.ceil(B / hb_r * hb_eta ** s / (s + 1)))
            r = hb_r * hb_eta ** -s

            combo_eval_list = random_sample_combo(par_combo_net, par_combo_opt,
                                                  tr_handler, metric, n, n_folds,
                                                  n_runs, val_handler, plotter)

            # Note: compared with the standard implementation, this one doesn't
            # restart the training each time but "accumulates" epochs, therefore
            # hb_R no longer represents the max amount of epochs per combo
            for i in range(s + 1):
                n_i = np.floor(n * hb_eta ** -i)
                # implementation detail: the number of epochs is roundup
                r_i = int(np.ceil(r * hb_eta ** i))
                n_keep = int(n_i // hb_eta)

                print(f"Num tasks: {len(combo_eval_list)}, num epochs: {r_i}")

                results = parallel(generate_jobs(combo_eval_list, r_i))

                combo_eval_list = compare_results(results, metric, n_keep)

                # keep global best results found
                best_results += combo_eval_list
                best_results = compare_results(best_results, metric, topk)

    # complete all training jobs if not already done
    best_results = bruteforce_run(best_results, 200, metric, topk)
    best_stats = [res.last_results for res in best_results]

    return best_stats


"""
    Randomly sample a combo from the given dictionaries. Returns a list of combos
    
    Parameters:
        -par_combo_net: dictionary containing a list for each hyperparameter of network
        -par_combo_opt: dictionary containing a list for each hyperparameter of optimizer
        -tr_handler: object containing both train_x and train_y (instance of DataHandler)
        -metric: metric function used to evaluate the performance of a model
        -n_conf: number of configurations to sample
        -n_folds: k value for the k-fold validation
        -n_runs: number of runs to perform
        -val_handler: object containing both valid_x and valid_y (instance of DataHandler) 
            (only used when n_folds not specified)
        -plotter: object that handles models plots
"""


def random_sample_combo(par_combo_net, par_combo_opt, tr_handler, metric,
                        n_conf, n_folds, n_runs, val_handler, plotter):
    par_combo = dict(par_combo_net, **par_combo_opt)
    combo_eval_list = []

    for _ in range(n_conf):

        dict_net = {}
        dict_opt = {}

        for key, arr in par_combo.items():

            par_val = random.choice(arr)

            if key in par_combo_net:
                dict_net[key] = par_val
            elif key in par_combo_opt:
                dict_opt[key] = par_val

        # use the given plotter as model for all results
        # all datapoints are collected, but the plots are generated
        # only for the best results at save-time/show-time
        res_plotter = None

        if plotter is not None:
            res_plotter = copy.deepcopy(plotter)

        c_ev = ComboEvaluator(dict_net, dict_opt, tr_handler, metric,
                              n_folds=n_folds, n_runs=n_runs,
                              val_handler=val_handler, plotter=res_plotter)

        combo_eval_list.append(c_ev)

    return combo_eval_list


"""
    Run the given configurations and compare the final results, no early stop in order 
    to optimise memory usage, define a chunk size and cleanup the results after chunk 
    size tasks are completed
    
    Parameters:
        -eval_list: list of configuration to evaluate
        -chunk_size: number of configurations to evaluate at each step
        -metric: metric used to measure the performance of the model
        -topk: number of best models to return. If None keeps everything
"""


def bruteforce_run(eval_list, chunk_size, metric, topk):
    best_list = []
    n_evals = len(eval_list)

    if chunk_size == -1:
        n_chunks = 1
    else:
        n_chunks = int(np.ceil(n_evals / chunk_size))

    # -2: Leave one core free
    with Parallel(n_jobs=-2, verbose=50) as parallel:

        for i in range(n_chunks):
            cur_start = i * chunk_size
            cur_stop = cur_start + chunk_size
            chunk_list = eval_list[cur_start: cur_stop]

            print(f"Number of tasks left: {len(eval_list[cur_start:])}")

            res_list = parallel(generate_jobs(chunk_list))

            # Keep the best results
            best_list = compare_results(best_list + res_list, metric, topk)

    return best_list


"""
    Compare results and keeps the best ones
        
    Parameters:
        -results: results of combos
        -metric: metric used to measure the performance of the model
        -topk: number of best models to return. If None keeps everything
"""


def compare_results(results, metric, topk=None):
    # remove all discarded results
    clean_results = []

    for combo_ev in results:
        if combo_ev is None:
            continue

        clean_results.append(combo_ev)

    results = clean_results

    # check if the validation score was computed, and use median to sort the results
    if results[0].last_results['score_val'] is not None:
        results = sorted(results, key=lambda ev: ev.last_results['score_val']["perc_50"],
                         reverse=(metric.aim == 'max'))
    # else sort by training score median
    else:
        results = sorted(results, key=lambda ev: ev.last_results['score_tr']["perc_50"],
                         reverse=(metric.aim == 'max'))

    if topk is not None:
        results = results[:topk]

    return results


"""
  Create tasks for Parallel starting from a list of ComboEvaluator
  
  Parameters:
    -eval_list: list of ComboEvaluator
    -step_epochs: number of epochs to train for or until stop criteria is met 
"""


def generate_jobs(eval_list, step_epochs=None):
    job_list = []

    # in case of exception the garbage collector will remove the object
    for ev in eval_list:
        job_list.append(delayed(eval_combo_search)(ev, step_epochs))

    return job_list


"""
    Wrapper around ComboEvaluator used only in _search methods
    
    Parameters:
        -combo_eval: a ComboEvaluator instance to evaluate
        -step_epochs: number of epochs to train for or until stop criteria is met 
"""


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
