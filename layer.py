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
    def __init__(self, n_out, n_in, init_func=None, act_func=None, bias=None,
                 init_scale=0, debug_bool=False):

        self.n_in = n_in  # Number of units in previous layer
        self.n_out = n_out  # Number of units in this layer
        self.bias = bias
        self.act_func = act_func
        self.init_func = init_func
        self.init_scale = init_scale

        # Variables used to implement backpropagation
        self.grad_w = None
        self.grad_b = None
        self.in_val = None
        self.net = None

        # Variable used to keep track of the layer's last output
        self.out = None

        # Variable used to print some debug information
        self.debug_bool = debug_bool

        self.reset_parameters()

    '''
        Reset the parameters of the layer

        Used instead of generating a new network
    '''
    def reset_parameters(self):

        # Variables used by the optimizer to implement momentum
        self.delta_w_old = np.zeros((self.n_out, self.n_in))
        self.delta_b_old = np.zeros((self.n_out,))

        if self.init_func is None:
            # Each row is composed of the weights of the unit
            self.weights = np.ones((self.n_out, self.n_in))
        # TODO add parameter for sparse count

        elif self.init_func.name == "std" and self.init_scale != 0:
            self.weights = self.init_func((self.n_out, self.n_in), self.init_scale, 0)

        else:
            self.weights = self.init_func((self.n_out, self.n_in), 0)

        self.null_grad()

    """
        Computes layer forward pass

        Parameters:
            -in_mat: matrix of input data
            -net_out: boolean used to return the net() result w/o activ func

        Returns:
            -matrix of layer's outputs
    """

    def forward(self, in_mat, net_out=False):

        self.in_val = in_mat
        net_wo_bias = np.matmul(self.in_val, np.transpose(self.weights))
        self.net = np.add(net_wo_bias, self.bias)

        if net_out:
            self.out = self.net
        else:
            self.out = self.act_func(self.net)

        return self.out

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
        self.grad_w = np.dot(np.transpose(delta), self.in_val)
        self.grad_b = np.sum(delta, axis=0)

        new_deriv_err = np.dot(delta, self.weights)

        if self.debug_bool:
            print("Layer")
            print("Activation function: ", self.act_func.name)
            print("\tOut: ", self.act_func(self.net))
            print("\tNet with bias: ", self.net)
            print("\tBias: ", self.bias)
            print("\tDelta: ", delta)
            print("\tDeriv_err: ", deriv_err)
            print("\tDeriv act(net): ", self.act_func.deriv(self.net))
            print("\tGrad weights: \n", self.grad_w)
            print("\tGrad bias: ", self.grad_b)
            print()

        return new_deriv_err

    '''
        Zero out the gradient variables of the layer
    '''
    def null_grad(self):

        self.grad_w = np.zeros((self.n_out, self.n_in))
        self.grad_b = np.zeros(self.n_out)

    def __str__(self):

        cur_str = f"\tnumber units: {self.n_out}, number weights: {self.n_in}"
        cur_str += f"\n\tlayer weights matrix: {self.weights}\n"
        cur_str += f"\n\tlayer bias vector: {self.bias}\n"
        cur_str += f"\n\tlayer gradient weights matrix: {self.grad_w}\n"
        cur_str += f"\n\tlayer gradient bias vector: {self.grad_b}\n"
        cur_str += f"\n\tactivation function: {self.act_func.name}"

        return cur_str
