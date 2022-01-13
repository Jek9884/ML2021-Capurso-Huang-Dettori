import numpy as np

from network import Network
from optimizer import GradientDescent
from utils.data_handler import DataHandler


class ComboEvaluator:

    def __init__(self, combo_net, combo_opt, tr_handler, metric,
                 val_handler=None, n_folds=0, n_runs=10, plotter=None):

        self.combo_net = combo_net
        self.combo_opt = combo_opt
        self.tr_handler = tr_handler
        self.metric = metric
        self.val_handler = val_handler
        self.n_folds = n_folds
        self.n_runs = n_runs
        self.plotter = plotter

        # List of cross-validation "instances" to run
        self.cv_list = []

        for i in range(self.n_runs):

            # Use kfold cv
            if self.n_folds > 0:

                kfold_cv = KFoldCV(self.combo_net, self.combo_opt,
                                   self.tr_handler, self.metric,
                                   self.n_folds, self.plotter)

                self.cv_list.append(kfold_cv)

            # Use a train-test split cv
            elif self.val_handler is not None:

                tr_ts_cv = ModelEvaluator(self.combo_net, self.combo_opt,
                                          self.tr_handler, self.metric,
                                          self.val_handler, self.plotter)

                self.cv_list.append(tr_ts_cv)

            # Evaluate model on only the training set
            else:

                tr_eval = ModelEvaluator(self.combo_net, self.combo_opt,
                                         self.tr_handler, self.metric,
                                         None, self.plotter)

                self.cv_list.append(tr_eval)

    def eval(self, step_epochs=None):

        tr_score_list = []
        val_score_list = []
        epoch_list = []
        age_list = []

        for cv in self.cv_list:

            tr_score, val_score, n_epochs, age = \
                cv.eval(step_epochs)

            tr_score_list.append(tr_score)

            if val_score is not None:
                val_score_list.append(val_score)

            epoch_list.append(n_epochs)
            age_list.append(age)

        tr_score_stats = compute_stats(tr_score_list)

        val_score_stats = None
        # List not empty
        if val_score_list:
            val_score_stats = compute_stats(val_score_list)

        avg_epochs = np.average(epoch_list)
        avg_age = np.average(age_list)

        result = {'combo_net': self.combo_net,
                  'combo_opt': self.combo_opt,
                  'score_tr': tr_score_stats,
                  'score_val': val_score_stats,
                  'metric': self.metric.name,
                  'epochs': avg_epochs,
                  'age': avg_age,
                  'plotter': self.plotter}

        return result

class KFoldCV:

    def __init__(self, combo_net, combo_opt, data_handler, metric,
                 n_folds, plotter=None):

        self.combo_net = combo_net
        self.combo_opt = combo_opt
        self.data_handler = data_handler
        self.metric = metric
        self.n_folds = n_folds
        self.plotter = plotter

        if self.n_folds < 1:
            raise ValueError("KFoldCV: invalid number of folds")

        # One model per fold
        self.model_list = []

        # Init folds_list with all models
        x_mat = data_handler.data_x
        y_mat = data_handler.data_y

        fold_size = int(np.floor(x_mat.shape[0] / self.n_folds))
        pattern_idx = np.arange(x_mat.shape[0])

        if fold_size < 1:
            raise ValueError("KFoldCV: fold size too big")

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

            tr_handler = DataHandler(train_x, train_y)
            val_handler = DataHandler(val_x, val_y)

            model = ModelEvaluator(self.combo_net, self.combo_opt,
                                   tr_handler, metric, val_handler,
                                   self.plotter)

            self.model_list.append(model)

    def eval(self, step_epochs=None):

        tr_score_list = []
        val_score_list = []
        epochs_list = []
        age_list = []

        for model in self.model_list:

            tr_score, val_score, n_epochs, age = \
                model.eval(step_epochs)

            tr_score_list.append(tr_score)
            val_score_list.append(val_score)
            epochs_list.append(n_epochs)
            age_list.append(age)

        # Use medians to represent the results
        avg_tr_score = np.median(tr_score_list)
        avg_val_score = np.median(val_score_list)
        avg_epochs = np.median(epochs_list)
        avg_age = np.median(age_list)

        return avg_tr_score, avg_val_score, avg_epochs, avg_age


class ModelEvaluator:

    def __init__(self, combo_net, combo_opt, tr_handler, metric,
                 val_handler=None, plotter=None):

        self.net = Network(**combo_net)
        self.opt = GradientDescent(**combo_opt)

        self.tr_handler = tr_handler
        self.val_handler = val_handler
        self.metric = metric
        self.plotter = plotter

        self.training_complete = False

    def eval(self, step_epochs=None):

        if step_epochs is None:
            # Guard value
            step_epochs = np.inf

        step_count = 0

        tr_score = None
        val_score = None

        while not self.training_complete and step_count < step_epochs:

            # Train 1 epoch at a time to plot the evolution of the lr curve
            self.training_complete = self.opt.train(self.net, self.tr_handler,
                                                    step_epochs=1, plotter=self.plotter)

            tr_score = eval_dataset(self.net, self.tr_handler, self.metric, True)

            # Check if the validation set is given as input
            if self.val_handler is not None:
                val_score = eval_dataset(self.net, self.val_handler,
                                         self.metric, False)

#                if self.plotter is not None:
#                    data_x = self.val_handler.data_x
#                    data_y = self.val_handler.data_y
#                    self.plotter.add_lr_curve_datapoint(self.net, data_x,
#                                                        data_y, "val")
            step_count += 1

#        if self.plotter is not None:
#            self.plotter.add_new_plotline()

        return tr_score, val_score, self.opt.epoch_count, self.opt.age_count


# Hp: all outputs from metric must be arrays
def eval_dataset(net, data_handler, metric, training):

    data_x = data_handler.data_x
    data_y = data_handler.data_y
    net_pred = net.forward(data_x, training)

    if metric.name in ["nll", "squared"]:
        score = metric(data_y, net_pred, reduce_bool=True)

    elif metric.name in ["miscl. error"]:
        # Note: it works only with classification
        net_pred[net_pred < 0.5] = 0
        net_pred[net_pred >= 0.5] = 1

        score = metric(data_y, net_pred)
    else:
        raise ValueError(f"Metric not supported ({metric.name})")

    return score


# Take stats wrt runs
def compute_stats(score_list):

    avg = np.average(score_list, axis=0)
    std = np.std(score_list, axis=0)
    perc_25 = np.percentile(score_list, 25, axis=0)
    perc_50 = np.percentile(score_list, 50, axis=0)
    perc_75 = np.percentile(score_list, 75, axis=0)

    stats_dict = {"avg": avg,
                  "std": std,
                  "perc_25": perc_25,
                  "perc_50": perc_50,
                  "perc_75": perc_75}

    return stats_dict