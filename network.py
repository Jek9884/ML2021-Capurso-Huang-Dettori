from layer import Layer
import numpy as np

"""

Network class

Parameters:
    -conf_layers: vector containing for each position the corresponding number of units in that layer.
            The first position contains the size of the input vectors.
    -init_func: weights init function
    -act_func: activation function object (DerivableFunction)
    -out_func: output function object (DerivableFunction)
    -loss_func: loss function object (DerivableFunction)
    -bias: vector containing the starting values for each layer's bias
    -debug_bool: print debug information from network and layers

Attributes:
    -out: Output vector of network
    -layers: list of network's layers objects

"""


# TODO add method for bias init

class Network:

    def __init__(self, conf_layers, init_func=None, act_func=None, out_func=None,
                 loss_func=None, bias=None, debug_bool=False):

        self.conf_layers = conf_layers
        self.init_func = init_func
        self.act_func = act_func
        self.out_func = out_func
        self.loss_func = loss_func
        self.bias = bias
        self.layers = []
        self.debug_bool = debug_bool

        if self.bias is None:
            # Init layer bias with heuristic value based on act/out func
            bias_dict = {"identity": 0,
                         "sigm": 0.5,
                         "tanh": 0}
            act_bias = [bias_dict[act_func.name]]*(len(conf_layers)-2)
            out_bias = bias_dict[out_func.name]
            self.bias = [*act_bias, out_bias]

        # layers init
        for i in range(len(conf_layers) - 2):
            self.layers.append(Layer(conf_layers[i + 1], conf_layers[i],
                                     init_func, act_func, self.bias[i], debug_bool))

        # init of output layer. Needed for different out_func
        self.layers.append(Layer(conf_layers[-1], conf_layers[-2],
                                 init_func, out_func, self.bias[len(conf_layers) - 2], debug_bool))

    """
        Computes network forward pass

        Parameters:
            -in_mat: matrix of input data

        Returns:
            -matrix of network's outputs
    """

    def forward(self, in_mat):

        if in_mat.ndim == 1:
            in_mat = np.asmatrix(in_mat)

        # net and act functions
        fw_mat = in_mat

        for i, layer in enumerate(self.layers):
            fw_mat = layer.forward(fw_mat)

        return fw_mat


    """
        Computes network backward

        Parameters:
            -cur_out: matrix of forward results
            -exp_out: matrix of expected results
    """

    def backward(self, exp_out, cur_out):

        if self.debug_bool:
            print("Network-wise info:")
            print("\tOut: ", cur_out)
            print("\tExpected: ", exp_out)
            print("\tLoss: ", self.loss_func.func(exp_out, cur_out))
            print("\tDeriv Loss: ", self.loss_func.deriv(exp_out, cur_out))
            print()

        # derivative of error w.r.t. the output of the last layer
        deriv_err = self.loss_func.deriv(exp_out, cur_out)

        # compute derivative of error w.r.t the i-th layer
        for i in reversed(range(len(self.layers))):
            deriv_err = self.layers[i].backward(deriv_err)

    """
        Zeros out the gradients stored in the network's layers
    """

    def null_grad(self):

        for layer in self.layers:
            layer.null_grad()

    def __str__(self):

        str_var = ""

        for i, layer in enumerate(self.layers):
            str_var += f"Layer id {i} " + layer.__str__() + "\n"

        return str_var
