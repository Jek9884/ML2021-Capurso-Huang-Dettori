import numpy as np
from layer import Layer
from functions.act_funcs import identity_act_func
from functools import partial

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


class Network:

    def __init__(self, conf_layers, init_func=None, act_func=None, out_func=None,
                 loss_func=None, bias=None, init_scale=0, batch_norm=False, debug_bool=False):

        self.conf_layers = conf_layers
        self.init_func = init_func
        self.act_func = act_func
        self.out_func = out_func
        self.loss_func = loss_func
        self.bias = bias
        self.layers = []
        self.batch_norm = batch_norm

        self.debug_bool = debug_bool

        if self.bias is None:
            # Init layer bias with heuristic value based on act/out func
            bias_dict = {"identity": 0,
                         "sigm": 0.5,
                         "tanh": 0,
                         "relu": 0.1}

            act_bias = [bias_dict[act_func.name]]*(len(conf_layers)-2)
            out_bias = bias_dict[out_func.name]
            self.bias = [*act_bias, out_bias]

        # layers init
        for i in range(len(conf_layers) - 2):
            self.layers.append(Layer(conf_layers[i + 1], conf_layers[i],
                                     self.init_func, self.act_func,
                                     self.bias[i], init_scale, batch_norm, debug_bool))

        # init of output layer is handled at the network level to avoid numerical problems
        self.layers.append(Layer(conf_layers[-1], conf_layers[-2],
                                 self.init_func, self.out_func,
                                 self.bias[len(conf_layers) - 2], init_scale,
                                 batch_norm, debug_bool))

    """
        Computes network forward pass

        Parameters:
            -in_mat: matrix of input data

        Returns:
            -matrix of network's outputs
    """

    def forward(self, in_mat, training=False):

        fw_mat = in_mat

        for i, layer in enumerate(self.layers):

            fw_mat = layer.forward(fw_mat, training)

        return fw_mat


    """
        Computes network backward

        Parameters:
            -cur_out: matrix of forward results
            -exp_out: matrix of expected results
    """

    def backward(self, exp_out):

        if self.debug_bool:
            print("Network-wise info:")
            print("\tActual out: ", self.layers[-1].out)
            print("\tExpected y: ", exp_out)
            print("\tLoss: ", self.eval_loss(exp_out))
            print("\tDeriv Loss: ", self.eval_deriv_loss(exp_out))
            print()

        d_err_d_y = None

        if self.loss_func.name == "nll":
            # nll returns the derivative w.r.t the input of non-linearity
            d_err_d_y = self.eval_deriv_loss(exp_out)
        else:
            d_err_d_out = self.eval_deriv_loss(exp_out)

        # compute derivative of error w.r.t the i-th layer
        for i in reversed(range(len(self.layers))):

            layer = self.layers[i]

            if d_err_d_y is None:
                d_err_d_y = d_err_d_out*layer.act_func.deriv(layer.y)

            d_err_d_out = layer.backward(d_err_d_y)

            d_err_d_y = None


    def eval_loss(self, exp_out, reduce_bool=False):

        res = None
        out_layer = self.layers[-1]

        # If used to avoid surprises
        if self.loss_func.name == "nll" and out_layer.act_func.name == "sigm":
            res = self.loss_func(exp_out, self.layers[-1].out, reduce_bool)

        elif self.loss_func.name in ["squared"]:
            res = self.loss_func(exp_out, self.layers[-1].out, reduce_bool)

        else:
            raise ValueError(f"network: unknown loss conf {self.loss_func.name}")

        return res


    def eval_deriv_loss(self, exp_out):

        res = None
        out_layer = self.layers[-1]

        # If used to avoid surprises
        if self.loss_func.name == "nll" and out_layer.act_func.name == "sigm":
            res = self.loss_func.deriv(exp_out, out_layer.out)

        elif self.loss_func.name in ["squared"]:
            res = self.loss_func.deriv(exp_out, out_layer.out)

        else:
            raise ValueError(f"network: unknown loss {self.loss_func.name}")

        return res


    '''
        Reset network parameters dependent on training
    '''
    def reset_parameters(self):

        for layer in self.layers:
            layer.reset_parameters()

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
