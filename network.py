import numpy as np
from layer import Layer

class Network:

    '''
    layer_unit_vec: vector containing for each position the corresponding number of units in that layer.
            The first position contains the size of the input vectors.
    act_func: activation function object
    out_func: output function object
    '''
    def __init__(self, layer_unit_vec, init_func=None, act_func=None, out_func=None, loss_func=None):

        self.layer_unit_vec = layer_unit_vec
        self.init_func = init_func
        self.act_func = act_func
        self.out_func = out_func
        self.loss_func = loss_func
        self.layers = []

        for i in range(len(layer_unit_vec)-1):

            self.layers.append(Layer(layer_unit_vec[i], layer_unit_vec[i+1], init_func, act_func))

    def forward(self, in_vec):
        # net and act functions
        fw_vec = in_vec

        for i, layer in enumerate(self.layers):

            fw_vec = layer.forward(fw_vec)

        # Used for computing the gradient
        self.out = self.out_func.func(fw_vec)

        return self.out

    def backward(self, res_vec):

        deriv_err = self.out - res_vec # TODO: generalise to other losses

        for i in reversed(range(len(self.layers))):

            deriv_err = self.layers[i].backward(deriv_err)

    def __str__(self):

        str_var = ""

        for i, layer in enumerate(self.layers):

            str_var += f"Layer id {i} " + layer.__str__() + "\n"

        return str_var
