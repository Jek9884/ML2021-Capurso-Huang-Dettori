import numpy as np

class GradientDescent:

    def __init__(self, network, lr, batch_size, reg_val=0, reg_type=2, epochs=None):

        if lr <= 0 or lr > 1:
            raise ValueError('lr should be a value between 0 and 1')

        self.lr = lr
        self.reg_val = reg_val
        self.reg_type = reg_type
        self.network = network
        self.batch_size = batch_size
        self.epochs = epochs

    def optimize(self, train_x, train_y):

        if self.epochs is not None:  # TODO check that train_x and train_y are matrix
            l = len(train_x)

            for e in range(self.epochs):
                batch = []
                sub_train_x = []
                sub_train_y = []

                if self.batch_size == 1:  # Online version
                    for i, _ in enumerate(train_x):
                        self.__step(train_x[i], train_y[i])
                elif 1 < self.batch_size < l: #TODO missing Stochastic batch and fix lr
                    pass
                elif self.batch_size == l:  # Batch version
                    self.__step(train_x, train_y)
                else:
                    raise ValueError("Batch size should be >= 1 and <= l")
        else:
            raise ValueError('limit_step should be not None')

    def __step(self, sub_train_x, sub_train_y):

        self.network.null_grad()

        for i, _ in enumerate(sub_train_x):
            self.network.forward(np.asarray(sub_train_x[i]).flatten())
            self.network.backward(np.asarray(sub_train_y[i]).flatten())

        self.__update_weights()

    def __update_weights(self):

        for layer in self.network.layers:

            delta_w = self.lr * layer.grad_w
            delta_b = self.lr * layer.grad_b

            # TODO: take decision regarding norm-1 and reg_type parameter
            if self.reg_val > 0 and self.reg_type == 2:
                delta_w += 2*self.reg_val*layer.weights
                delta_b += 2*self.reg_val*layer.bias

            layer.weights = layer.weights - delta_w
            layer.bias = layer.bias - delta_b
