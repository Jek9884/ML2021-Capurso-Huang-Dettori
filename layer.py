import numpy as np

class Layer:

    def __init__(self, n_in, n_out, act_func=None): #Network may not pass an act_func to the last layer

        self.n_in = n_in
        self.n_out = n_out
        self.weights = np.zeros((n_out, n_in))  # Each row is composed of the weights of the unit
        self.bias = 0
        self.act_func = act_func

        # TODO: implement weight init functions
        self.weights = np.ones((n_out, n_in))

    def forward(self, in_vec):

        net = np.matmul(self.weights, in_vec)
        net = np.add(net, self.bias)
        out = self.act_func.func(net)

        return out

    def __str__(self):

        return f"number units: {self.n_out}, number weigths: {self.n_in}"
