import numpy as np

class Layer:

    # Network may not pass an act_func to the last layer
    def __init__(self, n_in, n_out, init_func=None, act_func=None):

        self.n_in = n_in
        self.n_out = n_out
        self.bias = 0
        self.act_func = act_func
        self.grad = None

        if init_func is None:
            self.weights = np.ones((n_out, n_in))  # Each row is composed of the weights of the unit
        else:
            self.weights = init_func(n_in, n_out, 1) #TODO: standardise input args
            print(self.weights)

    def forward(self, in_vec):

        self.in_val = in_vec
        tmp_net = np.matmul(self.weights, self.in_val)
        self.net = np.add(tmp_net, self.bias)

        return self.act_func.func(self.net)

    # sum_prod_delta: sum of the product of the deltas of the layer above with correspondant weights
    def backward(self, deriv_err):

        self.delta = np.multiply(deriv_err, self.act_func.deriv(self.net))
        self.grad = np.multiply(self.delta, self.in_val)

        return np.matmul(self.weights, self.delta)


    def __str__(self):

        cur_str = f"\tnumber units: {self.n_out}, number weigths: {self.n_in}"

        if self.grad is not None:
            cur_str += f"\n\tlayer gradient vector: {self.grad}\n"

        return cur_str
