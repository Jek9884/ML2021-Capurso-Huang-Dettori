import numpy as np


class GradientDescent:

    def __init__(self, network, lr, batch_size, reg_val=0, reg_type=2, momentum_val=0, nesterov=False, epochs=None):

        if lr <= 0 or lr > 1:
            raise ValueError('lr should be a value between 0 and 1')

        if momentum_val < 0 or momentum_val > 1:
            raise ValueError('momentum should be a value between 0 and 1')

        self.lr = lr
        self.reg_val = reg_val
        self.reg_type = reg_type
        self.network = network
        self.batch_size = batch_size
        self.momentum_val = momentum_val
        self.nesterov = nesterov
        self.epochs = epochs

    def train(self, train_x, train_y):

        n_patterns = train_x.shape[0]
        index_list = np.arange(n_patterns)
        np.random.shuffle(index_list)  # Bengio et al suggest that one shuffle is enough

        #TODO: add alternative stop criterions
        if self.epochs is not None:

            for e in range(self.epochs):

                if self.batch_size == 1:  # Online version
                    for i, _ in enumerate(train_x):
                        self.__step(train_x[i], train_y[i])

                #TODO: parallelise mini-batch execution
                elif 1 < self.batch_size < n_patterns:  # Mini-batch version
                    n_mini_batch = n_patterns // self.batch_size
                    for i in range(n_mini_batch):
                        #TODO: check for correct dimensions
                        mini_batch_x = train_x[index_list[i * self.batch_size:
                                                          (i + 1) * self.batch_size]]
                        mini_batch_y = train_y[index_list[i * self.batch_size:
                                                          (i + 1) * self.batch_size]]
                        self.__step(mini_batch_x, mini_batch_y)

                elif self.batch_size == -1:  # Batch version
                    self.__step(train_x, train_y)
                else:
                    raise ValueError("Batch size should be >= 1 and <= l")
        else:
            raise ValueError('limit_step should be not None')

    def __step(self, sub_train_x, sub_train_y):

        self.network.null_grad()

        if self.nesterov:
            for layer in self.network.layers:
                layer.weights += self.momentum_val * layer.delta_w_old

        out = self.network.forward(sub_train_x)
        self.network.backward(sub_train_y, out)

        self.__update_weights()

    def __update_weights(self):

        for layer in self.network.layers:
            delta_w = self.lr * layer.grad_w
            delta_b = self.lr * layer.grad_b

            if self.momentum_val != 0:
                delta_w += self.momentum_val * layer.delta_w_old

            # TODO: take decision regarding norm-1 and reg_type parameter
            if self.reg_val > 0 and self.reg_type == 2:
                delta_w += 2 * self.reg_val * layer.weights

            layer.weights = layer.weights - delta_w
            layer.bias = layer.bias - delta_b
            layer.delta_w_old = delta_w
