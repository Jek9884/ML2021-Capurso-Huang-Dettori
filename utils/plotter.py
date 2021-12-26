import matplotlib.pyplot as plt
import numpy as np

from utils.helpers import convert_ragged_mat_to_ma_array


# Idea: each plotter can be used for either a model or a full eval by
# constructing multiple plots and taking their stats
class Plotter:

    def __init__(self, type_plots=[], lr_metric_list=None, n_cols=1):

        self.lr_metric_list = lr_metric_list
        self.type_plots = type_plots
        self.n_cols = n_cols
        self.n_plots = 0
        self.fig = None

        # Each "leaf" of the dict is a list of plotlines
        self.results_dict = {}

        if "lr_curve" in self.type_plots and self.lr_metric_list is None:
            raise ValueError("To print the learning curve a metric is needed")

    def add_plot_datapoint(self, network, optimizer, data_x, data_y):

        for plt_type in self.type_plots:

            if plt_type == "lr_curve":
                self.add_lr_curve_datapoint(network, data_x, data_y)
            elif plt_type == "lr":
                self.add_lr_rate_datapoint(optimizer)
            elif plt_type == "grad_norm":
                self.add_grad_norm_datapoint(network)
            elif plt_type == "delta_weights":
                self.add_delta_weights_datapoint(network)
            elif plt_type == "act_val":
                self.add_activ_val_datapoint(network, data_x)

    # Ideally called at the end of a model training process
    def add_new_plotline(self, plot_dict=None):

        if plot_dict is None:
            plot_dict = self.results_dict

        # Add an empty list to each list of lists
        for k, v in plot_dict.items():

            if isinstance(v, list):
                v.append([])

            elif isinstance(v, dict):
                self.add_new_plotline(v)

        self.n_plots += 1

    def plot(self):

        if self.fig is None:
            self.fig = self.build_plot()

        # Note: show() shows all plots created and not closed/shown
        plt.show()

    def save_fig(self, path):

        if self.fig is None:
            self.fig = self.build_plot()

        self.fig.savefig(path)

    def build_plot(self):

        if self.fig is not None:
            return self.fig

        if self.results_dict == {}:
            raise RuntimeError("Plotter: no results to plot")

        # Substitute lists of list with their average row-wise
        popul_distr = self.compute_stats_plotlines(self.results_dict)

        if self.n_plots > 0:
            self.results_dict["popul_distr"] = popul_distr

        plot_dim = ((len(self.results_dict) + 1) // self.n_cols, self.n_cols)
        fig_dim = (15, 10)
        fig, axs = plt.subplots(*plot_dim, squeeze=False, figsize=fig_dim)
        tot_epochs = len(popul_distr)

        for i, plt_type in enumerate(self.results_dict):

            cur_row = i // self.n_cols
            cur_col = i % self.n_cols
            cur_ax = axs[cur_row][cur_col]

            # Needed to handle matrix of values in these plots
            if plt_type in ["grad_norm", "act_val", "delta_weights"]:

                for n_layer, val in self.results_dict[plt_type].items():
                    cur_ax.errorbar(range(tot_epochs), val["avg"], val["std"],
                                    label=f"Layer {n_layer}", linestyle="None",
                                    marker=".", alpha=0.6)
                cur_ax.legend()

            elif "lr_curve" in plt_type:

                for i, data_label in enumerate(self.results_dict[plt_type]):
                    val = self.results_dict[plt_type][data_label]
                    cur_ax.plot(range(tot_epochs), val["avg"], label=data_label)
                    cur_ax.plot(range(tot_epochs), [val["avg_final"]]*tot_epochs,
                                label=f"Avg {data_label}", linestyle="dashed")
                cur_ax.legend()

            elif plt_type in ["lr"]:
                cur_ax.plot(range(tot_epochs),
                            np.around(self.results_dict[plt_type]["avg"], decimals=5))

            elif plt_type == "popul_distr":
                cur_ax.plot(range(tot_epochs), self.results_dict[plt_type])

            else:
                raise ValueError(f"Unknown plt_type ({plt_type})")

            cur_ax.set_xlabel("Epochs")
            cur_ax.set_ylabel(f"{plt_type}")

        n_blank_axs = len(self.results_dict) % self.n_cols

        # Hide unused plots
        for i in range(1, n_blank_axs + 1):
            axs[-1][-i].axis('off')

        self.fig = fig

        return self.fig

    def add_lr_curve_datapoint(self, network, data_x, data_y, data_label="tr"):

        for metric in self.lr_metric_list:

            training = False

            if data_label == "tr":
                training = True

            if metric.name == "nll":

                # Cannot apply nll to a net that doesn't use it as its criterion
                if network.loss_func.name != "nll":
                    raise ValueError("add_lr_curve_datapoint: unsupported use")

                network.forward(data_x, training)
                metric_res = network.eval_loss(data_y, reduce_bool=True)

            elif metric.name == "squared":
                pred_vec = network.forward(data_x, training)
                metric_res = metric(data_y, pred_vec, reduce_bool=True)

            elif metric.name in ["miscl. error"]:
                pred_vec = network.forward(data_x, training)
                pred_vec[pred_vec < 0.5] = 0
                pred_vec[pred_vec >= 0.5] = 1
                metric_res = metric(data_y, pred_vec)
            else:
                raise ValueError("add_lr_curve_datapoint: unsupported metric")

            plot_name = f"lr_curve ({metric.name})"

            if plot_name not in self.results_dict:
                self.results_dict[plot_name] = {}

            if data_label not in self.results_dict[plot_name]:
                self.results_dict[plot_name][data_label] = [[]]

            self.results_dict[plot_name][data_label][-1].append(metric_res)

    def add_lr_rate_datapoint(self, optimizer):

        if "lr" not in self.results_dict:
            self.results_dict["lr"] = [[]]

        self.results_dict["lr"][-1].append(optimizer.lr)

    # Note: use after a backward pass
    def add_grad_norm_datapoint(self, network):

        if "grad_norm" not in self.results_dict:
            self.results_dict["grad_norm"] = {}

        for i, layer in enumerate(network.layers):

            # Uses frobenius norm on the joint weights (bias included) matrix
            grad_layer = \
                np.hstack((layer.grad_w, np.expand_dims(layer.grad_b, axis=1)))
            norm_grad = np.linalg.norm(grad_layer)

            if i not in self.results_dict["grad_norm"]:
                self.results_dict["grad_norm"][i] = [[]]

            self.results_dict["grad_norm"][i][-1].append(norm_grad)

    def add_delta_weights_datapoint(self, network):

        if "delta_weights" not in self.results_dict:
            self.results_dict["delta_weights"] = {}

        for i, layer in enumerate(network.layers):

            # Uses frobenius norm on the joint weights (bias included) matrix
            delta_layer = \
                np.hstack((layer.delta_w_old, np.expand_dims(layer.delta_b_old, axis=1)))
            norm_delta = np.linalg.norm(delta_layer)

            if i not in self.results_dict["delta_weights"]:
                self.results_dict["delta_weights"][i] = [[]]

            self.results_dict["delta_weights"][i][-1].append(norm_delta)

    def add_activ_val_datapoint(self, network, data_x):

        if "act_val" not in self.results_dict:
            self.results_dict["act_val"] = {}

        network.forward(data_x, training=True)

        for i, layer in enumerate(network.layers):

            if i not in self.results_dict["act_val"]:
                self.results_dict["act_val"][i] = [[]]

            self.results_dict["act_val"][i][-1].append(np.average(layer.out))

    def compute_stats_plotlines(self, plot_dict=None, parent=None):

        if plot_dict is None:
            plot_dict = self.results_dict

        popul_distr = None
        # Add an empty list to each list of lists
        for k, v in plot_dict.items():

            if isinstance(v, dict):
                popul_distr = self.compute_stats_plotlines(v, parent=k)

            elif isinstance(v, list):
                ma_matrix = convert_ragged_mat_to_ma_array(v)
                # Compute stats
                ma_average = np.ma.average(ma_matrix, axis=0)
                ma_std = np.ma.std(ma_matrix, axis=0)

                if popul_distr is None:
                    popul_distr = np.ma.count(ma_matrix, axis=0)

                    # Needed for multi-out network
                    if popul_distr.ndim > 1:
                        popul_distr = popul_distr[:, 0]

                plot_dict[k] = {"avg": ma_average, "std": ma_std}

                if isinstance(parent, str) and "lr_curve" in parent:
                    ma_elem_len = len(ma_matrix[0][0])
                    # Take the last non-masked element of each row
                    last_ma_idx = np.ma.notmasked_edges(ma_matrix, axis=1)[1][1]
                    # Each idx is repeated for each element in the matrix cell
                    last_ma_idx = last_ma_idx[::ma_elem_len]
                    # Generate list of position in the matrix to compute average
                    final_pred_idx = (range(len(last_ma_idx)), last_ma_idx)
                    plot_dict[k]["avg_final"] = np.average(ma_matrix[final_pred_idx])


        return popul_distr
