import numpy as np

from utils.debug_tools import check_gradient_net


class GradientDescent:

    def __init__(self, lr, batch_size, reg_val=0, reg_type=2, momentum_val=0,
                 nesterov=False, lim_epochs=10**4, lr_decay_type=None,
                 lr_dec_lin_tau=None, lr_dec_exp_k=None, stop_crit_type='fixed',
                 epsilon=None, patient=5, norm_clipping=0, check_gradient=False):

        if lr <= 0 or lr > 1:
            raise ValueError('lr should be a value between 0 and 1')

        if momentum_val < 0 or momentum_val > 1:
            raise ValueError('momentum should be a value between 0 and 1')

        self.lr = lr
        self.reg_val = reg_val
        self.reg_type = reg_type
        self.batch_size = batch_size
        self.momentum_val = momentum_val
        self.nesterov = nesterov
        self.lim_epochs = lim_epochs
        self.lr_decay_type = lr_decay_type
        self.stop_crit_type = stop_crit_type
        self.epsilon = epsilon
        self.patient = patient
        self.norm_clipping = norm_clipping
        self.check_gradient = check_gradient

        # Used to avoid stopping training prematurely in case of delta_w crit
        self.count_patient = 0

        # Used to implement stop criteria
        self.delta_w_norm = None

        # linear decay heuristic
        if self.lr_decay_type == "lin":
            self.lr_decay_tau = lr_dec_lin_tau
            self.initial_lr = self.lr
            self.eta_tau = self.initial_lr / 100
        elif self.lr_decay_type == "exp":
            self.lr_decay_k = lr_dec_exp_k
            self.initial_lr = self.lr
        elif self.lr_decay_type is not None:
            raise ValueError(f"lr_decay type not supported {self.lr_decay_type}")

        self.reset_optimizer()

    # Reset "moving parts" of the optimizer
    # To use after completing the whole training on the model
    # Or when switching models
    def reset_optimizer(self):

        # Used to keep track of cumulative amount of epochs/updates trained
        # Useful to compare accross batch sizes
        self.age_count = 0
        # Useful to know when to reset lr_decay's alpha
        self.epoch_count = 0

        if self.stop_crit_type == 'delta_w':
            self.delta_w_norm = np.inf
            self.count_patient = 0

    def train(self, net, tr_handler, step_epochs=None, plotter=None):

        # Allow for more flexibility in using the optimizer with different epochs
        if step_epochs is None:
            lim_step = self.lim_epochs
        else:
            lim_step = self.epoch_count + step_epochs

        while not self.is_training_complete():

            # Check if number of epochs requested in train() has been reached
            if self.epoch_count >= lim_step:
                break

            # Batch normalisation needs minibatch of fixed size
            enforce_size = net.batch_norm
            # Already randomized
            mb_x_list, mb_y_list = tr_handler.get_minibatch_list(self.batch_size,
                                                                 enforce_size)
            n_minibatch = len(mb_x_list)
            mb_count = 0

            while mb_count < n_minibatch and not self.is_training_complete():
                mini_batch_x = mb_x_list[mb_count]
                mini_batch_y = mb_y_list[mb_count]

                # Depending on situation the mb size might differ from the chosen one
                # See DataHandler parameters
                mb_size = len(mini_batch_x)

                self.update_weights(net, mini_batch_x, mini_batch_y)
                self.update_stop_crit()

                self.age_count += mb_size
                mb_count += 1

            if plotter is not None:
                # Approximates the plot by taking a screenshot of the net after
                # each epoch
                plotter.add_plot_datapoint(net, self, tr_handler.data_x, tr_handler.data_y)

            if mb_count == n_minibatch:
                self.epoch_count += 1

        # Used to determine if there needs to be further training
        return self.is_training_complete()

    # Check if the stop criterion has been reached
    def is_training_complete(self):

        if self.stop_crit_type == 'fixed':
            complete_bool = self.epoch_count >= self.lim_epochs

        elif self.stop_crit_type == 'delta_w':

            if self.epoch_count >= self.lim_epochs:
                complete_bool = True
            elif self.delta_w_norm > self.epsilon:
                complete_bool = False
            else:
                # Mx number of patient reached
                complete_bool = self.count_patient >= self.patient
        else:
            raise ValueError('Invalid stop criteria')

        return complete_bool

    def update_weights(self, net, sub_train_x, sub_train_y):

        net.null_grad()

        if self.nesterov and self.momentum_val > 0:
            for layer in net.layers:
                layer.weights += self.momentum_val * layer.delta_w_old
                layer.bias += self.momentum_val * layer.delta_b_old

        if self.check_gradient:
            check_gradient_net(net, sub_train_x, sub_train_y)

        net.forward(sub_train_x, training=True)
        net.backward(sub_train_y)

        self.compute_deltas(net, sub_train_x.shape[0])

        # Compute the norm of the delta of the weights
        if self.stop_crit_type == 'delta_w':
            norm_weights = []

            for layer in net.layers:
                deltaw_layer = \
                    np.hstack((layer.delta_w_old,
                               np.expand_dims(layer.delta_b_old, axis=1)))
                norm_delta = np.linalg.norm(deltaw_layer)

                norm_weights.append(norm_delta)

            # Take the biggest change in weights to determine stop cond
            self.delta_w_norm = np.max(norm_weights)

    def compute_deltas(self, net, num_patt):

        # LR decay techniques
        if self.lr_decay_type == "lin":
            # Avoid having an alpha > 1 due to difference in lim_epochs/tau
            alpha = min(self.epoch_count / self.lr_decay_tau, 1)
            self.lr = self.initial_lr*(1-alpha) + alpha*self.eta_tau

        elif self.lr_decay_type == "exp":
            exp_fact = np.exp(-1*self.lr_decay_k*self.epoch_count)
            self.lr = self.initial_lr * exp_fact

        for layer in net.layers:

            # Gradient norm clipping
            if self.norm_clipping > 0:

                layer_weights = np.hstack((layer.grad_w,
                                           np.expand_dims(layer.grad_b, axis=1)))
                norm_weights = np.linalg.norm(layer_weights)

                if norm_weights > self.norm_clipping:
                    layer.grad_w = self.norm_clipping*layer.grad_w/norm_weights
                    layer.grad_b = self.norm_clipping*layer.grad_b/norm_weights

            # Take average of weights wrt number of samples
            delta_w = -1 * self.lr * layer.grad_w
            delta_b = -1 * self.lr * layer.grad_b

            # Momentum
            if self.momentum_val > 0:
                delta_w += self.momentum_val * layer.delta_w_old
                delta_b += self.momentum_val * layer.delta_b_old

            # Regularisation
            if self.reg_val > 0 and self.reg_type == 2:
                delta_w += -2 * self.reg_val * layer.weights

            # Update parameters
            layer.weights = layer.weights + delta_w
            layer.bias = layer.bias + delta_b
            layer.delta_w_old = delta_w
            layer.delta_b_old = delta_b

            if layer.batch_norm:
                layer.batch_gamma -= self.lr*layer.grad_gamma
                layer.batch_beta -= self.lr*layer.grad_beta

    # Update the variables concerning the stop criterion
    def update_stop_crit(self):

        if self.stop_crit_type == 'delta_w':

            # epsilon is the criterion tolerance
            if self.delta_w_norm > self.epsilon:
                self.count_patient = 0
            else:
                self.count_patient += 1
