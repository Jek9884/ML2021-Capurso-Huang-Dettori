import numpy as np

from network import Network
from optimizer import GradientDescent
from utils.data_handler import DataHandler
from functions.loss_funcs import loss_dict
from functions.metric_funcs import metr_dict

"""
    ComboEvaluator
    
    Parameters:
        -combo_net: combo of hyperparameters for network
        -combo_opt: combo of hyperparameters for the optimizer
        -tr_handler: object containing both train_x and train_y (instance of DataHandler)
        -metric: metric used to evaluate combo
        -val_handler: object containing both valid_x and valid_y (instance of DataHandler) 
        -n_folds: k value for the k-fold validation
        -n_runs: number of runs to perform
        -plotter: object that handles models plots
        
    Attributes:
        -cv_list: list of cross-validation tasks to run
        -tr_complete: boolean specifying if the training is complete
        -last_results: dictionary of last results computed
"""


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

        self.cv_list = []

        self.tr_complete = False

        self.last_results = None

        for i in range(self.n_runs):

            # use kfold cv
            if self.n_folds > 0:

                kfold_cv = KFoldCV(self.combo_net, self.combo_opt,
                                   self.tr_handler, self.metric,
                                   self.n_folds, self.plotter, run_id=i)

                self.cv_list.append(kfold_cv)

            # use a train-test split cv
            elif self.val_handler is not None:

                tr_ts_cv = ModelEvaluator(self.combo_net, self.combo_opt,
                                          self.tr_handler, self.metric,
                                          self.val_handler, self.plotter,
                                          model_id=i)

                self.cv_list.append(tr_ts_cv)

            # evaluate model on only the training set
            else:

                tr_eval = ModelEvaluator(self.combo_net, self.combo_opt,
                                         self.tr_handler, self.metric,
                                         None, self.plotter, model_id=i)

                self.cv_list.append(tr_eval)

    """
        Evaluate the cross validation tasks. Returns the results of the evaluation
        
        Parameters:
            -step_epochs: number of epochs for which to run cross-validation tasks
    """

    def eval(self, step_epochs=None):

        # if all models have been trained don't go further
        if self.tr_complete:
            return self.last_results

        # lists of results used to compute last step stats
        # can contain either a matrix of k-fold runs (row: kfold run, column: fold)
        # or a matrix of single model scores (row: different run, column: single score)
        tr_score_mat = []
        val_score_mat = []
        epoch_list = []
        age_list = []

        # keep track of all the models training
        tr_status_list = []

        for cv_eval in self.cv_list:

            tr_score, val_score, n_epochs, age, tr_status = \
                cv_eval.eval(step_epochs)

            if isinstance(cv_eval, KFoldCV):
                tr_score_mat.append(tr_score)
                val_score_mat.append(val_score)

            elif isinstance(cv_eval, ModelEvaluator):
                tr_score_mat.append([tr_score])

                if val_score is not None:
                    val_score_mat.append(val_score)
            else:
                raise ValueError("ComboEvaluator: unknown cross-validation method")

            epoch_list.append(n_epochs)
            age_list.append(age)
            tr_status_list.append(tr_status)

        tr_score_stats = compute_stats(tr_score_mat)
        val_score_stats = None

        # list not empty
        if val_score_mat:
            val_score_stats = compute_stats(val_score_mat)

        avg_epochs = np.average(epoch_list)
        avg_age = np.average(age_list)
        self.tr_complete = np.all(tr_status_list)

        self.last_results = {'combo_net': self.combo_net,
                             'combo_opt': self.combo_opt,
                             'score_tr': tr_score_stats,
                             'score_val': val_score_stats,
                             'metric': self.metric.name,
                             'epochs': avg_epochs,
                             'age': avg_age,
                             'plotter': self.plotter,
                             'tr_complete': self.tr_complete}

        return self.last_results


"""
    KFoldCV
    

    
    Parameters:
        -combo_net: combo of hyperparameters for the network
        -combo_opt: combo of hyperparameters for the optimizer
        -data_handler: object containing both inputs and targets (instance of DataHandler)
        -metric: metric function used to evaluate the performance of a model
        -n_folds: k value for the k-fold validation
        -plotter: object that handles models plots
        -run_id: id used to identify a specific run (used for Plotter)
        
    Attributes:
        -tr_complete: boolean specifying if the training is complete
        -model_list: list of models of kfold, one per fold
"""


class KFoldCV:

    def __init__(self, combo_net, combo_opt, data_handler, metric,
                 n_folds, plotter=None, run_id=0):

        self.combo_net = combo_net
        self.combo_opt = combo_opt
        self.data_handler = data_handler
        self.metric = metric
        self.n_folds = n_folds
        self.plotter = plotter

        if self.n_folds < 1:
            raise ValueError("KFoldCV: invalid number of folds")

        # indicates if all the models have been trained
        self.tr_complete = False

        # one model per fold
        self.model_list = []

        # init folds_list with all models
        x_mat = data_handler.data_x
        y_mat = data_handler.data_y

        fold_size = int(np.floor(x_mat.shape[0] / self.n_folds))
        pattern_idx = np.arange(x_mat.shape[0])

        if fold_size < 1:
            raise ValueError("KFoldCV: fold size too big")

        np.random.shuffle(pattern_idx)

        for i in range(n_folds):
            # everything except i*fold_size:(i+1)*fold_size segment
            train_idx = np.concatenate(
                (pattern_idx[:i * fold_size],
                 pattern_idx[(i + 1) * fold_size:]), axis=0)

            train_x = x_mat[train_idx]
            train_y = y_mat[train_idx]

            val_x = x_mat[i * fold_size:(i + 1) * fold_size]
            val_y = y_mat[i * fold_size:(i + 1) * fold_size]

            tr_handler = DataHandler(train_x, train_y)
            val_handler = DataHandler(val_x, val_y)

            model_id = run_id * self.n_folds + i

            model = ModelEvaluator(self.combo_net, self.combo_opt,
                                   tr_handler, metric, val_handler,
                                   self.plotter, model_id=model_id)

            self.model_list.append(model)

    """
        Evaluate the kfold task. Returns a list of scores, one for each model/fold needs to be further handled
        
        Parameters:
            -step_epochs: number of epochs for which to run cross-validation tasks
    """

    def eval(self, step_epochs=None):

        tr_score_list = []
        val_score_list = []
        epochs_list = []
        age_list = []
        tr_status_list = []

        for model in self.model_list:
            tr_score, val_score, n_epochs, age, tr_status = \
                model.eval(step_epochs)

            tr_score_list.append(tr_score)
            val_score_list.append(val_score)
            epochs_list.append(n_epochs)
            age_list.append(age)
            tr_status_list.append(tr_status)

        avg_epochs = np.average(epochs_list)
        avg_age = np.average(age_list)
        self.tr_complete = np.all(tr_status_list)

        return tr_score_list, val_score_list, avg_epochs, avg_age, self.tr_complete


"""
    ModelEvaluator
    
    
    Parameters:
        -combo_net: combo of hyperparameters for the network
        -combo_opt: combo of hyperparameters for the optimizer
        -tr_handler: object containing both train_x and train_y (instance of DataHandler)
        -metric: metric used to evaluate combo
        -val_handler: object containing both valid_x and valid_y (instance of DataHandler) 
        -plotter: object that handles models plots
        -model_id: id used to identify a specific model/plotline (used for Plotter)
        
    Attributes:
        -tr_complete: boolean specifying if the training is complete
        -model_list: list of models of kfold, one per fold
"""


class ModelEvaluator:

    def __init__(self, combo_net, combo_opt, tr_handler, metric,
                 val_handler=None, plotter=None, model_id=0):

        self.net = Network(**combo_net)
        self.opt = GradientDescent(**combo_opt)

        self.tr_handler = tr_handler
        self.val_handler = val_handler
        self.metric = metric
        self.plotter = plotter
        self.model_id = model_id

        self.tr_score = None
        self.val_score = None

        self.tr_complete = False

    """
        Evaluate a single model. Returns the results of the evaluation

        Parameters:
            -step_epochs: number of epochs for which to run the model
    """

    def eval(self, step_epochs=None):

        if step_epochs is None:
            # guard value
            step_epochs = np.inf

        if self.plotter is not None:
            self.plotter.set_active_plotline(self.model_id)

        step_count = 0

        # stop if training complete or the number of epochs specified has been reached
        while not self.tr_complete and step_count < step_epochs:

            # train 1 epoch at a time to plot the evolution of the lr curve
            self.tr_complete = self.opt.train(self.net, self.tr_handler,
                                              step_epochs=1, plotter=self.plotter)

            # evaluate tr set as if using the trained model
            self.tr_score = eval_dataset(self.net, self.tr_handler, self.metric, False)

            # check if the validation set is given as input
            if self.val_handler is not None:

                self.val_score = eval_dataset(self.net, self.val_handler, self.metric, False)

                if self.plotter is not None:
                    data_x = self.val_handler.data_x
                    data_y = self.val_handler.data_y

                    self.plotter.add_lr_curve_datapoint(self.net, data_x,
                                                        data_y, "val")
            step_count += 1

        return self.tr_score, self.val_score, \
               self.opt.epoch_count, self.opt.age_count, \
               self.tr_complete


# Hp: all outputs from metric must be arrays
"""
    Evaluate the given dataset. Returns the score of the evaluation
    
    Parameters:
        -net: network used to evaluate
        -data_handler: object containing both inputs and targets (instance of DataHandler)
        -metric: metric used to evaluate combo
        -training: boolean specifying if the network is in training or inference mode
"""


def eval_dataset(net, data_handler, metric, training):
    data_x = data_handler.data_x
    data_y = data_handler.data_y
    net_pred = net.forward(data_x, training)

    if metric.name in loss_dict:
        score = metric(data_y, net_pred, reduce_bool=True)

    elif metric.name in metr_dict:
        # Note: it works only with classification
        net_pred[net_pred < 0.5] = 0
        net_pred[net_pred >= 0.5] = 1

        score = metric(data_y, net_pred)
    else:
        raise ValueError(f"Metric not supported ({metric.name})")

    return score


"""
    Compute stats of the predictions across runs. Returns a dictionary of stats
    
    Parameters:
        -score_list: list of scores to compute stats from
"""


def compute_stats(score_list):
    # take the stats across all the models trained
    avg = np.average(score_list)
    std = np.std(score_list)
    perc_25 = np.percentile(score_list, 25)
    perc_50 = np.percentile(score_list, 50)
    perc_75 = np.percentile(score_list, 75)

    stats_dict = {"avg": avg,
                  "std": std,
                  "perc_25": perc_25,
                  "perc_50": perc_50,
                  "perc_75": perc_75}

    return stats_dict
