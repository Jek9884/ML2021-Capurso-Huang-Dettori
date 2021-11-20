from layer import Layer

"""

Network class 

Parameters:
    -layer_unit_vec: vector containing for each position the corresponding number of units in that layer.
            The first position contains the size of the input vectors.
    -init_func: weights init function
    -act_func: activation function object (DerivableFunction)
    -out_func: output function object (DerivableFunction)
    -loss_func: loss function object (DerivableFunction)
    -bias: starting value for bias

Attributes:
    -out: Output vector of network
    -layers: list of network's layers objects
    
"""


class Network:

    def __init__(self, layer_unit_vec, init_func=None, act_func=None, out_func=None, loss_func=None, bias=0):

        self.layer_unit_vec = layer_unit_vec
        self.init_func = init_func
        self.act_func = act_func
        self.out_func = out_func
        self.loss_func = loss_func
        self.bias = bias
        self.out = None
        self.layers = []

        # layers init
        for i in range(len(layer_unit_vec) - 1):
            self.layers.append(Layer(layer_unit_vec[i + 1], layer_unit_vec[i], init_func, act_func, bias))

    """
        Computes network forward pass
        
        Parameters:
            -in_vec: vector of input data
            
        Returns:
            -out: vector of network's outputs
    """

    def forward(self, in_vec):
        # net and act functions
        fw_vec = in_vec

        for i, layer in enumerate(self.layers):
            fw_vec = layer.forward(fw_vec)

        # used for computing the gradient
        self.out = self.out_func.func(fw_vec)

        return self.out

    """
        Computes network backward
        
        Parameters:
            -res_vec: vector of expected results
    """

    def backward(self, res_vec):

        # derivative of error w.r.t. the output of the last layer
        deriv_err = self.loss_func.deriv(res_vec, self.out)

        # compute derivative of error w.r.t the i-th layer
        for i in reversed(range(len(self.layers))):
            deriv_err = self.layers[i].backward(deriv_err)

    def __str__(self):

        str_var = ""

        for i, layer in enumerate(self.layers):
            str_var += f"Layer id {i} " + layer.__str__() + "\n"

        return str_var
