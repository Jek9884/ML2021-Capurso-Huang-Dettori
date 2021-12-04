import numpy as np


class GradientDescent:

    def __init__(self, lr, batch_size, reg_val=0, reg_type=2, momentum_val=0,
                 nesterov=False, epochs=None, lr_decay=False, lr_decay_tau=None, stop_crit_type='fixed',
                 epsilon=None, patient=None):

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
        self.epochs = epochs
        self.lr_decay = lr_decay
        self.lr_decay_tau = lr_decay_tau
        self.stop_crit_type = stop_crit_type
        self.epsilon = epsilon
        self.patient = patient

        if self.stop_crit_type == 'weights_change':
            self.count_patient = 0

    def train(self, net, train_x, train_y, epochs=None):

        # Allow for more flexibility in using the optimizer with different epochs
        if epochs is None:
            epochs = self.epochs

        n_patterns = train_x.shape[0]
        index_list = np.arange(n_patterns)
        np.random.shuffle(index_list)  # Bengio et al suggest that one shuffle is enough

        if self.lr_decay:
            eta_0 = self.lr
            eta_tau = eta_0 / 100

        n_epochs = 0

        delta_weights = np.inf

        if epochs is None:
            raise ValueError('epochs should not be None')

        while self.check_stop_crit(delta_weights) and n_epochs < epochs:

            old_weights = []

            if self.stop_crit_type == 'weights_change':
                for layer in net.layers:
                    old_weights.append(layer.weights)

            if self.lr_decay:
                alpha = n_epochs / self.lr_decay_tau
                self.lr = eta_0 * (1 - alpha) + alpha * eta_tau

            if self.batch_size == -1:  # Batch version
                self.__step(net, train_x, train_y)

            elif 1 <= self.batch_size < n_patterns:  # Online/mini-batch version

                n_mini_batch = int(np.ceil(n_patterns / self.batch_size))

                for i in range(n_mini_batch):
                    idx_list = index_list[i * self.batch_size: (i + 1) * self.batch_size]
                    mini_batch_x = train_x[idx_list]
                    mini_batch_y = train_y[idx_list]
                    self.__step(net, mini_batch_x, mini_batch_y)

            else:
                raise ValueError("Mini-batch size should be >= 1 and < l.\
                                 If you want to use the batch version use -1.")

            if self.stop_crit_type == 'weights_change':
                norm_weights = []

                for i, layer in enumerate(net.layers):
                    norm_weights.append(np.linalg.norm(np.subtract(layer.weights, old_weights[i])))

                delta_weights = np.average(norm_weights)
                print('delta weights: ' + str(delta_weights))
                print('epoch: ' + str(n_epochs))
            n_epochs += 1

    def check_stop_crit(self, delta_w=None):

        if self.stop_crit_type == 'fixed':
            result = True
        elif self.stop_crit_type == 'weights_change':
            epsilon = self.epsilon

            if delta_w > epsilon:
                self.count_patient = 0
                result = True
            else:
                self.count_patient += 1
                if self.count_patient >= self.patient:
                    result = False
                else:
                    result = True
        else:
            raise ValueError('Invalid stop criteria')

        return result

    def __step(self, net, sub_train_x, sub_train_y):

        net.null_grad()

        if self.nesterov:
            for layer in net.layers:
                layer.weights += self.momentum_val * layer.delta_w_old

        out = net.forward(sub_train_x)
        net.backward(sub_train_y, out)

        self.__update_weights(net, sub_train_x.shape[0])

    def __update_weights(self, net, num_patt):

        for layer in net.layers:
            avg_grad_w = np.divide(layer.grad_w, num_patt)
            avg_grad_b = np.divide(layer.grad_b, num_patt)
            delta_w = self.lr * avg_grad_w
            delta_b = self.lr * avg_grad_b

            if self.momentum_val != 0:
                delta_w += self.momentum_val * layer.delta_w_old

            # TODO: take decision regarding norm-1 and reg_type parameter
            if self.reg_val > 0 and self.reg_type == 2:
                delta_w += 2 * self.reg_val * layer.weights

            layer.weights = layer.weights - delta_w
            layer.bias = layer.bias - delta_b
            layer.delta_w_old = delta_w
