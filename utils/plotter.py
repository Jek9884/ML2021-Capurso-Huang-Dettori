import matplotlib.pyplot as plt
import numpy as np

from utils.helpers import convert_ragged_mat_to_ma_array
from functions.loss_funcs import loss_dict
from functions.metric_funcs import metr_dict

"""
    Plotter
      
    Idea: each plotter can be used for either a single model or a set of models (ie multiple runs) 
    by constructing multiple plots and taking their stats
    
    Parameters:
        -type_plots: list of types of plots to create
            Values:
                -lr_curve: plot the learning curve
                -log_lr_curve: plot the learning curve in log scale
                -lr: plot the values of lr across time
                -grad_norm: plot the norm of the gradient per layer
                -delta_weights: plot the norm of the changing weights per layer 
                -log_delta_weights: plot the norm of the changing weights per layer in log scale
                -act_val: values of activation function per layer
                
        -lr_metric_list: list of metric functions to generate the lr curve
        -n_cols: number of columns of the plots grid
        
    Attributes:
        -n_models: number of models that plotter keeps track of 
        -fig: figure composed by multiple plots (matplotlib object)
        -active_plt: id of currently selected plot/model to update
        -results_dict: tree of dictionaries containing the requested plots
        -param_dict: dictionary containing relevant constants
        -data_labels: list of labels of datasets used to evaluate net
"""


class Plotter:

    def __init__(self, type_plots=None, lr_metric_list=None, n_cols=1):

        if type_plots is None:
            raise ValueError("plotter: type_plots must differ from None value")

        self.lr_metric_list = lr_metric_list
        self.type_plots = type_plots
        self.n_cols = n_cols
        self.n_models = 0
        self.fig = None

        # indicates which plotline new information will be appended to
        self.active_plt = 0

        # each "leaf" of the dict is a list of plotlines
        self.results_dict = {}

        self.param_dict = {}

        self.data_labels = []

        if "lr_curve" in self.type_plots and self.lr_metric_list is None:
            raise ValueError("plotter: to print the learning curve a metric is needed")

    """
        Add to results_dict the requested plots
        
        Parameters:
            -network: network to plot
            -optimizer: optimizer to plot
            -data_x: input patterns
            -data_y: targets
    """

    def add_plot_datapoint(self, network, optimizer, data_x, data_y):

        if "lr_curve" in self.type_plots or "log_lr_curve" in self.type_plots:
            self.add_lr_curve_datapoint(network, data_x, data_y)
        if "lr" in self.type_plots:
            self.add_lr_rate_datapoint(optimizer)
        if "grad_norm" in self.type_plots:
            self.add_grad_norm_datapoint(network)
        if "delta_weights" in self.type_plots or "log_delta_weights" in self.type_plots:
            self.add_delta_weights_datapoint(network, optimizer)
        if "act_val" in self.type_plots:
            self.add_activ_val_datapoint(network, data_x)

    """
        Select active model/plotline
        
        Parameters:
            -model_id: id of the active model
    """

    def set_active_plotline(self, model_id):

        # + 1 since we want to avoid jumps but allow for a single plotline addition
        if model_id < 0 or model_id > self.n_models:
            raise ValueError("Plotter/set_active_plotline: invalid plot_id")

        if model_id == self.n_models:
            self.add_new_plotline()
            self.n_models += 1

        self.active_plt = model_id

    """
        Append an empty list for each type of plot to results_dict
        
        Parameters:
            -plot_dict: used for recursion purposes
    """

    def add_new_plotline(self, plot_dict=None):

        if plot_dict is None:
            plot_dict = self.results_dict

        # add an empty list to each list of lists
        for k, v in plot_dict.items():

            if isinstance(v, list):
                v.append([])

            elif isinstance(v, dict):
                self.add_new_plotline(v)

    """
        Add new datapoint to the active lr curve plot in the results_dict
        
        Parameters:
            -network: network to plot 
            -data_x: input patterns
            -data_y: targets
            -data_label: define if data belongs to training or test set
    """

    def add_lr_curve_datapoint(self, network, data_x, data_y, data_label="tr"):

        if data_label not in self.data_labels:
            self.data_labels.append(data_label)

        for metric in self.lr_metric_list:

            training = False

            if data_label == "tr":
                training = True

            if metric.name in loss_dict:

                pred_vec = network.forward(data_x, training)
                metric_res = metric(data_y, pred_vec, reduce_bool=True)

            elif metric.name in metr_dict:
                pred_vec = network.forward(data_x, training)
                pred_vec[pred_vec < 0.5] = 0
                pred_vec[pred_vec >= 0.5] = 1
                metric_res = metric(data_y, pred_vec)
            else:
                raise ValueError(f"add_lr_curve_datapoint: unsupported metric {metric.name}")

            plot_name = f"lr_curve ({metric.name}) ({data_label})"

            if plot_name not in self.results_dict:
                self.results_dict[plot_name] = [[]]

            self.results_dict[plot_name][self.active_plt].append(metric_res)

    """
        Add new datapoint to the active lr rate plot in the results_dict
        
        Parameters:
            -optimizer: optimizer to plot 
    """

    def add_lr_rate_datapoint(self, optimizer):

        if "lr" not in self.results_dict:
            self.results_dict["lr"] = [[]]

        self.results_dict["lr"][self.active_plt].append(optimizer.lr)

    """
        Add new datapoint to the active grad norm plot in the results_dict
        
        Note: use after a backward pass
        
        Parameters:
            -network: network to plot
    """

    def add_grad_norm_datapoint(self, network):

        if "grad_norm" not in self.results_dict:
            self.results_dict["grad_norm"] = {}

        for i, layer in enumerate(network.layers):

            # uses frobenius norm on the joint weights (bias included) matrix
            grad_layer = \
                np.hstack((layer.grad_w, np.expand_dims(layer.grad_b, axis=1)))
            norm_grad = np.linalg.norm(grad_layer)

            if i not in self.results_dict["grad_norm"]:
                self.results_dict["grad_norm"][i] = [[]]

            self.results_dict["grad_norm"][i][self.active_plt].append(norm_grad)

    """
        Add new datapoint to the active delta weights plot in the results_dict
        
        Parameters:
            -network: network to plot
            -optimizer: optimizer to plot
    """

    def add_delta_weights_datapoint(self, network, optimizer):

        if "delta_weights" not in self.results_dict:
            self.results_dict["delta_weights"] = {}

        for i, layer in enumerate(network.layers):

            # uses frobenius norm on the joint weights (bias included) matrix
            delta_layer = \
                np.hstack((layer.delta_w_old, np.expand_dims(layer.delta_b_old, axis=1)))
            norm_delta = np.linalg.norm(delta_layer)

            if i not in self.results_dict["delta_weights"]:
                self.results_dict["delta_weights"][i] = [[]]

            if "delta_eps" not in self.param_dict:
                self.param_dict["delta_eps"] = optimizer.epsilon

            self.results_dict["delta_weights"][i][self.active_plt].append(norm_delta)

    """
        Add new datapoint to the active activation values plot in the results_dict

        Parameters:
            -network: network to plot
            -data_x: input patterns
    """

    def add_activ_val_datapoint(self, network, data_x):

        if "act_val" not in self.results_dict:
            self.results_dict["act_val"] = {}

        network.forward(data_x, training=True)

        for i, layer in enumerate(network.layers):

            if i not in self.results_dict["act_val"]:
                self.results_dict["act_val"][i] = [[]]

            self.results_dict["act_val"][i][self.active_plt].append(np.average(layer.out))

    # stats utilities functions

    """
        Recursively compute the stats of the types of plots across plotlines/models
        
        Parameters:
            -in_dict: dictionary from which to start the recursion
            -out_dict: dictionary in which to save the results
            -log_bool: boolean used to apply natural log to the results
    """

    def compute_stats_plotlines(self, in_dict=None, out_dict=None, log_bool=False):

        if in_dict is None:
            in_dict = self.results_dict

        if out_dict is None:
            raise ValueError("compute_stats_plotlines: need to provide an out_dict")

        # used to avoid log(0) problem
        log_eps = 10 ** -6

        # add an empty list to each list of lists
        for k, v in in_dict.items():

            if isinstance(v, dict):

                if k not in out_dict:
                    out_dict[k] = {}

                self.compute_stats_plotlines(v, out_dict[k], log_bool)

            elif isinstance(v, list):
                ma_matrix = convert_ragged_mat_to_ma_array(v)
                # compute stats
                ma_average = np.ma.average(ma_matrix, axis=0)
                ma_std = np.ma.std(ma_matrix, axis=0)

                if log_bool:
                    ma_average = np.log(ma_average + log_eps)
                    ma_std = np.log(ma_std + log_eps)

                # return only data from masked array
                out_dict[k] = {"avg": ma_average.data, "std": ma_std.data}

    """
        Recursively compute the number of models still active per epoch
        
        Parameters:
            -in_dict: dictionary from which to start the recursion
    """

    def compute_models_distr(self, in_dict=None):

        if in_dict is None:
            in_dict = self.results_dict

        model_distr = None

        for k, v in in_dict.items():

            if model_distr is not None:
                break

            if isinstance(v, dict):
                model_distr = self.compute_models_distr(v)

            # Use the first list of lists encountered to compute distr
            elif isinstance(v, list):
                ma_matrix = convert_ragged_mat_to_ma_array(v)

                if model_distr is None:
                    model_distr = np.ma.count(ma_matrix, axis=0)

                    # needed for multi-out network
                    if model_distr.ndim > 1:
                        model_distr = model_distr[:, 0]

        return model_distr


    # plot generation functions

    """
        Build the dictionary containing each requested plot's data. Returns the dict
    """

    def build_plot_dict(self):

        plot_dict = {}

        for lr_curve_type in ["lr_curve", "log_lr_curve"]:

            if lr_curve_type not in self.type_plots:
                continue

            for metric in self.lr_metric_list:
                for label in self.data_labels:
                    plot_name = f"{lr_curve_type} ({metric.name}) ({label})"
                    results_name = f"lr_curve ({metric.name}) ({label})"

                    log_bool = "log" in lr_curve_type

                    res_dict = {plot_name: self.results_dict[results_name]}
                    self.compute_stats_plotlines(res_dict, res_dict, log_bool)
                    plot_dict[plot_name] = res_dict[plot_name]

        for plt_type in ["lr", "act_val", "grad_norm"]:
            if plt_type in self.type_plots:
                res_dict = {plt_type: self.results_dict[plt_type]}
                self.compute_stats_plotlines(res_dict, res_dict)
                plot_dict[plt_type] = res_dict[plt_type]

        for delta_type in ["delta_weights", "log_delta_weights"]:

            if delta_type not in self.type_plots:
                continue

            log_bool = "log" in delta_type

            res_dict = {delta_type: self.results_dict["delta_weights"]}
            self.compute_stats_plotlines(res_dict, res_dict, log_bool)
            plot_dict[delta_type] = res_dict[delta_type]

        return plot_dict
 

    """
        Build the figure containing each requested plot. Returns the figure
    """

    def build_plot(self):

        if self.fig is not None:
            return self.fig

        if self.results_dict == {}:
            raise RuntimeError("plotter: no results to plot")

        # used to avoid log(0) problem
        log_eps = 10 ** -6

        plot_dict = self.build_plot_dict()

        model_distr = self.compute_models_distr()
        tot_epochs = len(model_distr)

        if self.n_models > 0:
            plot_dict["model_distr"] = model_distr

        # The subplots are dynamically generated step by step
        fig_dim = (15, 10)
        plot_dim = (len(plot_dict) // self.n_cols + 1, self.n_cols)
        fig, axs = plt.subplots(*plot_dim, squeeze=False, figsize=fig_dim)

        for i, plt_type in enumerate(plot_dict):

            cur_row = i // self.n_cols
            cur_col = i % self.n_cols
            cur_ax = axs[cur_row][cur_col]

            # needed to handle matrix of values in these plots
            if plt_type in ["grad_norm", "act_val"]:

                for n_layer, val in plot_dict[plt_type].items():
                    cur_ax.errorbar(range(tot_epochs), val["avg"], val["std"],
                                    label=f"Layer {n_layer}", linestyle="None",
                                    marker=".", alpha=0.6)
                cur_ax.legend()
                cur_ax.set_ylabel(f"{plt_type}")

            elif "delta_weights" in plt_type:

                # compute the log of the delta_weights
                for n_layer, val in plot_dict[plt_type].items():

                    cur_ax.errorbar(range(tot_epochs), val["avg"], val["std"],
                                    label=f"Layer {n_layer}", linestyle="None",
                                    marker=".", alpha=0.6, zorder=2)

                delta_eps = self.param_dict["delta_eps"]

                if "log" in plt_type:
                    delta_eps = np.log(delta_eps + log_eps)

                cur_ax.plot(range(tot_epochs), [delta_eps] * tot_epochs, zorder=3,
                            label="Delta eps", linestyle="dashed", color="black")
                cur_ax.legend()
                cur_ax.set_ylabel(plt_type)

            elif "lr_curve" in plt_type:

                lr_stats = plot_dict[plt_type]

                if "log" in plt_type:
                    # Remove "log_" from plt_type
                    lines_list = self.results_dict[plt_type.split("_", 1)[1]]

                    # Plot all individual lines
                    for line in lines_list:
                        line_len = len(line)
                        cur_ax.plot(range(line_len), np.log(np.add(line, log_eps)),
                                    alpha=0.1, color="gray")
                else:
                    # plot all individual lines
                    for line in self.results_dict[plt_type]:
                        line_len = len(line)
                        cur_ax.plot(range(line_len), line, alpha=0.1, color="gray")

                cur_ax.plot(range(tot_epochs), lr_stats["avg"], label="Avg score")
                cur_ax.set_ylabel(plt_type)
                cur_ax.legend()

            elif plt_type in ["lr"]:
                cur_ax.plot(range(tot_epochs),
                            np.around(plot_dict[plt_type]["avg"], decimals=5))
                cur_ax.set_ylabel(f"{plt_type}")

            elif plt_type == "model_distr":
                cur_ax.plot(range(tot_epochs), plot_dict[plt_type])
                cur_ax.set_ylabel(f"{plt_type}")

            else:
                raise ValueError(f"Unknown plt_type ({plt_type})")

            cur_ax.set_xlabel("Epochs")

        # hide unused plots
        n_blank_axs = self.n_cols - len(plot_dict) % self.n_cols

        for i in range(1, n_blank_axs + 1):
            axs[-1][-i].axis('off')

        self.fig = fig

        return self.fig


    def save_lr_curves(self, path):

        plot_dict = self.build_plot_dict()

        # The subplots are dynamically generated step by step
        fig_dim = (15, 5)
        plot_dim = (1, len(self.lr_metric_list))
        fig, axs = plt.subplots(*plot_dim, squeeze=False, figsize=fig_dim)

        for i, lr_metric in enumerate(self.lr_metric_list):

            lr_type = lr_metric.name

            for plt_type in plot_dict:

                if "lr_curve" not in plt_type or lr_type not in plt_type:
                    continue

                lr_stats = plot_dict[plt_type]

                if "tr" in plt_type:
                    axs[0][i].plot(range(len(lr_stats["avg"])), lr_stats["avg"],
                                   label="Avg score (train)")

                if "val" in plt_type:
                    axs[0][i].plot(range(len(lr_stats["avg"])), lr_stats["avg"],
                                   label="Avg score (test)", linestyle="dashed")

                axs[0][i].set_ylabel(lr_type)
                axs[0][i].set_xlabel("Epochs")
                axs[0][i].legend()

        fig.savefig(path)
        plt.close(fig)

    """
        Show the resulting plots
    """

    def plot(self):

        if self.fig is None:
            self.fig = self.build_plot()

        # Note: show() shows all plots created and not closed/shown
        plt.show()
        plt.close(self.fig)

    """
        Save generated figure

        Parameters:
            -path: path where to save the figure
    """
    def savefig(self, path):

        if self.fig is None:
            self.fig = self.build_plot()

        self.fig.savefig(path)
        plt.close(self.fig)
