import numpy as np

"""

Layer class

Parameters:
    -n_out: number of units of the layer
    -n_in: number of input weights
    -init_func: weights init function
    -act_func: activation function object (DerivableFunction)
    -bias: starting value for bias
    -debug_bool: print debug information from the layer

Attributes:
    -grad: gradient of error w.r.t. weights of the layer
    -in_val: input value for the layer (vector)
    -net: Dot product btw weights (matrix) and input (vector)
"""


class Layer:

    # Network may not pass an act_func to the last layer
    def __init__(self, n_out, n_in, init_func=None, act_func=None, bias=0, debug_bool=False):

        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.act_func = act_func
        self.debug_bool = debug_bool
        self.grad_w = np.zeros((self.n_out, self.n_in))
        self.grad_b = np.zeros(self.n_out)
        self.in_val = None
        self.net = None
        self.delta_w_old = 0  # Used by optimizer with momentum

        if init_func is None:
            self.weights = np.ones((n_out, n_in))  # Each row is composed of the weights of the unit
        else:
            self.weights = init_func(n_out, n_in, 0)  # TODO add parameter for sparse count
            # self.bias = init_func(n_out, 1, 0)

    """
        Computes layer forward pass

        Parameters:
            -in_vec: vector of input data

        Returns:
            -out: vector of layer's outputs
    """

    def forward(self, in_vec):

        self.in_val = in_vec
        tmp_net = np.matmul(self.weights, self.in_val)
        self.net = np.add(tmp_net, self.bias)

        return self.act_func.func(self.net)

    """
        Computes layer backward

        Parameters:
            -deriv_err: derivative of error w.r.t. weights of layer

        Returns:
            -deriv_err for next layer's backward
    """

    def backward(self, deriv_err):

        # delta = deriv_err * f'(net_t)
        delta = np.multiply(deriv_err, self.act_func.deriv(self.net))

        # grad += delta_i * output of previous layer (o_u)
        self.grad_w += np.outer(delta, self.in_val)
        self.grad_b += delta
        new_deriv_err = np.matmul(np.transpose(self.weights), delta)

        if self.debug_bool:
            print("Layer")
            print("Activation function: ", self.act_func.name)
            print("\tOut: ", self.act_func.func(self.net))
            print("\tNet with bias: ", self.net)
            print("\tBias: ", self.bias)
            print("\tDelta: ", delta)
            print("\tDeriv_err: ", deriv_err)
            print("\tDeriv act(net): ", self.act_func.deriv(self.net))
            print("\tGrad weights: \n", self.grad_w)
            print("\tGrad bias: ", self.grad_b)
            print()

        return new_deriv_err

    def __str__(self):

        cur_str = f"\tnumber units: {self.n_out}, number weights: {self.n_in}"
        cur_str += f"\n\tlayer weights matrix: {self.weights}\n"
        cur_str += f"\n\tlayer gradient weights matrix: {self.grad_w}\n"
        cur_str += f"\n\tlayer gradient bias vector: {self.grad_b}\n"
        cur_str += f"\n\tactivation function: {self.act_func.name}"

        return cur_str

    def null_grad(self):

        self.grad_w = np.zeros((self.n_out, self.n_in))
        self.grad_b = np.zeros(self.n_out)
