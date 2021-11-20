import numpy as np

class Layer:

    # Network may not pass an act_func to the last layer
    def __init__(self, n_in, n_out, init_func=None, act_func=None):

        self.n_in = n_in
        self.n_out = n_out
        self.bias = 0
        self.act_func = act_func

        if init_func is None:
            self.weights = np.ones((n_out, n_in))  # Each row is composed of the weights of the unit
        else:
            self.weights = init_func(n_in, n_out, 1)
            print(self.weights)

    def forward(self, in_vec):

        net = np.matmul(self.weights, in_vec)
        net = np.add(net, self.bias)
        out = self.act_func.func(net)

        return out

    def __str__(self):

        return f"number units: {self.n_out}, number weigths: {self.n_in}"
