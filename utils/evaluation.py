import numpy as np

from network import Network
from optimizer import GradientDescent
from utils.data_handler import DataHandler

def eval_model(par_combo_net, par_combo_opt, train_handler, metric,
               val_handler=None, n_folds=0, n_runs=10, plotter=None):

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

        avg = np.average(score_results_dict[key], axis=0)
        std = np.std(score_results_dict[key], axis=0)
        perc_25 = np.percentile(score_results_dict[key], 25, axis=0)
        perc_50 = np.percentile(score_results_dict[key], 50, axis=0)
        perc_75 = np.percentile(score_results_dict[key], 75, axis=0)

        score_stats_dict[key] = {"avg": avg,
                                 "std": std,
                                 "perc_25": perc_25,
                                 "perc_50": perc_50,
                                 "perc_75": perc_75}

    result = {'combo_net': par_combo_net,
              'combo_opt': par_combo_opt,
              'score_tr': score_stats_dict["tr"],
              'score_val': score_stats_dict["val"],
              'metric': metric.name,
              'epochs': avg_epochs,
              'age': avg_age,
              'plotter': plotter}

    return result


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

    training_complete = False

    while not training_complete:

        # Train 1 epoch at a time to plot the evolution of the lr curve
        training_complete = gd.train(net, train_handler, 1, plotter=plotter)

        tr_result = eval_dataset(net, train_handler, metric, True)
        epoch_res_tr_list.append(tr_result)

        # Check if the validation set is given as input
        if val_handler is not None:
            val_result = eval_dataset(net, val_handler, metric, False)
            epoch_res_val_list.append(val_result)

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
