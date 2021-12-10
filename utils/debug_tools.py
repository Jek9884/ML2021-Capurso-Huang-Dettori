import matplotlib.pyplot as plt
import numpy as np


# TODO: lookup gradient checking technique


class Plotter:

    def __init__(self, type_plots=[], lr_metric_list=None, n_cols=1, epoch_interval=1):

        self.epoch_interval = epoch_interval
        self.num_intervals = 0
        self.lr_metric_list = lr_metric_list
        self.type_plots = type_plots
        self.n_cols = n_cols
        self.results_dict = None

        if "lr_curve" in self.type_plots and self.lr_metric_list is None:
            raise ValueError("To print the learning curve a metric is needed")

    def build_plot(self, network, optimizer, data_x, data_y, cur_epoch):

        # If this is the first epoch, reset the dictionary
        if cur_epoch == 0:
            self.results_dict = {}

        # The results are updated only every epoch_interval
        if cur_epoch % self.epoch_interval != 0:
            return

        for plt_type in self.type_plots:

            if plt_type == "lr_curve":
                self.plot_learning_curve(network, data_x, data_y)
            elif plt_type == "lr":
                self.plot_learning_rate(optimizer)
            elif plt_type == "grad_norm":
                self.plot_gradient_norm(network)
            elif plt_type == "act_val":
                self.plot_activ_func_out(network, data_x)

        self.num_intervals += 1

    def plot(self):

        if self.results_dict is None:
            raise RuntimeError("Plotter: no results to plot")

        plot_dims = ((len(self.results_dict)+1)//self.n_cols, self.n_cols)
        fig, axs = plt.subplots(*plot_dims, squeeze=False)
        tot_epochs = self.num_intervals * self.epoch_interval

        # TODO: add more info in label and titles
        for i, plt_type in enumerate(self.results_dict):

            cur_row = i // self.n_cols
            cur_col = i % self.n_cols
            cur_ax = axs[cur_row][cur_col]

            # Needed to handle matrix of values in these plots
            if plt_type in ["grad_norm", "act_val"]:

                mat_val = np.transpose(self.results_dict[plt_type])

                for j in range(len(mat_val)):
                    cur_ax.plot(range(0, tot_epochs, self.epoch_interval),
                                mat_val[j], label=f"Layer {j}")

                cur_ax.legend()

            elif "lr_curve" in plt_type:

                for data_label in self.results_dict[plt_type]:
                    cur_ax.plot(range(0, tot_epochs, self.epoch_interval),
                                self.results_dict[plt_type][data_label],
                                label=data_label)
                cur_ax.legend()

            else:
                cur_ax.plot(range(0, tot_epochs, self.epoch_interval),
                            self.results_dict[plt_type])

            cur_ax.set_xlabel("Epochs")
            cur_ax.set_ylabel(f"{plt_type}")

        plt.show()


    def plot_learning_curve(self, network, data_x, data_y, data_label="tr"):

        for metric in self.lr_metric_list:

            if metric.name == "nll":
                pred_vec = network.forward(data_x, net_out=True)
            elif metric.name == "squared":
                pred_vec = network.forward(data_x)
            elif metric.name in ["miscl. error"]:
                pred_vec = network.forward(data_x)
                pred_vec[pred_vec < 0.5] = 0
                pred_vec[pred_vec >= 0.5] = 1
            else:
                raise ValueError("plot_learning_curve: unsupported metric")

            metric_res = metric(data_y, pred_vec)

            plot_name = f"lr_curve ({metric.name})"

            if plot_name not in self.results_dict:
                self.results_dict[plot_name] = {}

            if data_label not in self.results_dict[plot_name]:
                self.results_dict[plot_name][data_label] = []

            self.results_dict[plot_name][data_label].append(metric_res)

    def plot_learning_rate(self, optimizer):

        if "lr" not in self.results_dict:
            self.results_dict["lr"] = []

        self.results_dict["lr"].append(optimizer.lr)

    # Note: needs to be used after a backward pass
    def plot_gradient_norm(self, network):

        norm_grad_list = []

        for layer in network.layers:

            bias_shape = (layer.grad_w.shape[0], 1)

            # TODO: check for spike in plot
            # Uses frobenius norm on the joint weights (bias included) matrix
            grad_layer = np.hstack((layer.grad_w, layer.grad_b.reshape(bias_shape)))
            norm_grad = np.linalg.norm(grad_layer)

            norm_grad_list.append(norm_grad)

        if "grad_norm" not in self.results_dict:
            self.results_dict["grad_norm"] = []

        self.results_dict["grad_norm"].append(norm_grad_list)

    def plot_activ_func_out(self, network, data_x):

        act_list = []

        network.forward(data_x)

        for layer in network.layers:
            act_list.append(np.average(layer.out))

        if "act_val" not in self.results_dict:
            self.results_dict["act_val"] = []

        self.results_dict["act_val"].append(act_list)
