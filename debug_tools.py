import matplotlib.pyplot as plt
import numpy as np


# TODO: lookup gradient checking technique


class Plotter:

    def __init__(self, epoch_interval, type_plots=[], lr_metric=None, n_cols=1):

        self.epoch_interval = epoch_interval
        self.lr_metric = lr_metric
        self.type_plots = type_plots
        self.n_cols = n_cols
        self.plot_dims = (len(self.type_plots)//n_cols, n_cols)
        self.results_dict = {}

        if "lr_curve" in self.type_plots and lr_metric is None:
            raise ValueError("To print the learning curve a metric is needed")

    def build_plot(self, network, optimizer, train_x, train_y, cur_epoch):

        # If this is the first epoch, reset the dictionary
        if cur_epoch == 0:
            self.results_dict = {type_plt: [] for type_plt in self.type_plots}

        # The results are updated only every epoch_interval
        if cur_epoch % self.epoch_interval != 0:
            return

        for plt_type in self.type_plots:

            if plt_type == "lr_curve":
                self.plot_learning_curve(network, train_x, train_y)
            elif plt_type == "lr":
                self.results_dict["lr"].append(optimizer.lr)
            elif plt_type == "grad_norm":
                self.plot_gradient_norm(network, train_x)
            elif plt_type == "act_val":
                self.plot_activ_func_out(network, train_x)

    def plot(self):

        fig, axs = plt.subplots(*self.plot_dims, squeeze=False)
        num_intervals = len(self.results_dict[self.type_plots[0]])
        tot_epochs = num_intervals * self.epoch_interval

        # TODO: add more info in label and titles
        for i, plt_type in enumerate(self.type_plots):

            cur_row = i // self.n_cols
            cur_col = i % self.n_cols
            cur_ax = axs[cur_row][cur_col]

            if plt_type in ["grad_norm", "act_val"]:

                mat_val = np.transpose(self.results_dict[plt_type])

                for j in range(len(mat_val)):
                    cur_ax.plot(range(0, tot_epochs, self.epoch_interval),
                                   mat_val[j], label=f"Layer {j}")
            else:
                cur_ax.plot(range(0, tot_epochs, self.epoch_interval),
                               self.results_dict[plt_type])

            cur_ax.set_xlabel("Epochs")
            cur_ax.set_ylabel(f"{plt_type}")
            cur_ax.legend()

        plt.show()


    def plot_learning_curve(self, network, train_x, train_y):

        if self.lr_metric.name == "nll":
            pred_vec = network.forward(train_x, net_out=True)
        elif self.lr_metric.name == "squared":
            pred_vec = network.forward(train_x)
        else:
            pred_vec = network.forward(train_x)
            pred_vec[pred_vec < 0.5] = 0
            pred_vec[pred_vec >= 0.5] = 1

        metric_res = self.lr_metric(train_y, pred_vec)

        self.results_dict["lr_curve"].append(metric_res)

    # Note: needs to be used after a backward pass
    # TODO: check if there are problems with optimizer op order
    def plot_gradient_norm(self, network, train_x):

        norm_grad_list = []

        for layer in network.layers:

            bias_shape = (layer.grad_w.shape[0], 1)

            # Uses frobenius norm on the joint weights (bias included) matrix
            grad_layer = np.hstack((layer.grad_w, layer.grad_b.reshape(bias_shape)))
            norm_grad = np.linalg.norm(grad_layer)

            norm_grad_list.append(norm_grad)

        self.results_dict["grad_norm"].append(norm_grad_list)

    def plot_activ_func_out(self, network, train_x):

        act_list = []

        for layer in network.layers:
            act_list.append(np.average(layer.out))

        self.results_dict["act_val"].append(act_list)
