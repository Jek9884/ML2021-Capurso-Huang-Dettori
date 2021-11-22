import numpy as np

class GradientDescent:

    def __init__(self, network, lr, batch_size, epochs=None):

        if lr <= 0 or lr > 1:
            raise ValueError('lr should be a value between 0 and 1')
        self.lr = lr
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
                elif 1 < self.batch_size < l:
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
            print(layer.weights)
            layer.weights = layer.weights + (self.lr * layer.grad)


