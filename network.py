from layer import Layer

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
        -init_scale: scale of uniform distribution in std weight init
        -batch_norm: boolean for batch normalization activation
        -batch_momentum: momentum value for the moving avg of batch normalization (inference mode)
        -dropout: boolean for dropout activation
        -dropout_in_keep: probability of keeping an input unit
        -dropout_hid_keep: probability of keeping an hidden unit
        -debug_bool: print debug information from network and layers
    
    Attributes:
        -layers: list of network's layers objects
"""


class Network:

    def __init__(self, conf_layers, init_func=None, act_func=None, out_func=None,
                 loss_func=None, bias=None, init_scale=0, batch_norm=False, batch_momentum=0.99,
                 dropout=False, dropout_in_keep=0.8, dropout_hid_keep=0.5, debug_bool=False):

        self.conf_layers = conf_layers
        self.init_func = init_func
        self.act_func = act_func
        self.out_func = out_func
        self.loss_func = loss_func
        self.bias = bias
        self.layers = []

        self.batch_norm = batch_norm
        self.batch_momentum = batch_momentum

        self.dropout = dropout
        self.dropout_in_keep = dropout_in_keep
        self.dropout_hid_keep = dropout_hid_keep

        self.debug_bool = debug_bool

        if out_func.name != "sigm" and loss_func.name == "nll":
            raise ValueError("Network: {out_func.name}/nll combination not supported")

        if 0 > dropout_in_keep > 1 or 0 > dropout_hid_keep > 1:
            raise ValueError("Network: invalid values for dropout probabilities")

        if self.bias is None:
            # Init layer bias with heuristic value based on act/out func
            bias_dict = {"identity": 0,
                         "sigm": 0.5,
                         "tanh": 0,
                         "relu": 0.1,
                         "lrelu": 0.1}

            act_bias = [bias_dict[act_func.name]] * (len(conf_layers) - 2)
            out_bias = bias_dict[out_func.name]
            self.bias = [*act_bias, out_bias]

        # Initialise layers
        for i in range(0, len(conf_layers) - 1):

            # Defaults
            act_func = self.act_func
            dropout_keep = self.dropout_hid_keep

            # Special cases
            if i == 0:
                dropout_keep = self.dropout_in_keep

            if i == (len(conf_layers) - 2):
                act_func = self.out_func

            self.layers.append(Layer(conf_layers[i + 1], conf_layers[i],
                                     self.init_func, act_func, self.bias[i],
                                     init_scale, self.batch_norm,
                                     self.batch_momentum, self.dropout,
                                     dropout_keep, self.debug_bool))

    """
        Computes network forward pass. Returns matrix of network's outputs

        Parameters:
            -in_mat: matrix of input data
            -training: boolean used to distinguish between training and inference mode for 
                batch_norm and dropout purposes
    """

    def forward(self, in_mat, training=False):

        fw_mat = in_mat

        for i, layer in enumerate(self.layers):
            fw_mat = layer.forward(fw_mat, training)

        return fw_mat

    """
        Computes network backward

        Parameters:
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

        # Derivative of the error w.r.t. output of the net/layer
        d_err_d_y = None

        if self.loss_func.name == "nll":
            # nll returns the derivative w.r.t the input of non-linearity
            d_err_d_y = self.eval_deriv_loss(exp_out)
        else:
            # Other losses return the derivative w.r.t the out of the net
            d_err_d_out = self.eval_deriv_loss(exp_out)

        # compute derivative of error w.r.t the i-th layer
        for i in reversed(range(len(self.layers))):

            layer = self.layers[i]

            if d_err_d_y is None:
                d_err_d_y = d_err_d_out * layer.act_func.deriv(layer.y)

            d_err_d_out = layer.backward(d_err_d_y)

            d_err_d_y = None

    """
        Evaluate loss function
        
        Parameters:
            -exp_out: expected results
            -reduce_bool: boolean to return the reduction (avg) of the loss func result
    """

    def eval_loss(self, exp_out, reduce_bool=False):

        res = self.loss_func(exp_out, self.layers[-1].out, reduce_bool)

        return res

    """    
        Evaluate derivative of loss function
    
        Parameter:
            -exp_out: expected results
    """

    def eval_deriv_loss(self, exp_out):

        res = self.loss_func.deriv(exp_out, self.layers[-1].out)

        return res

    '''
        Reset network parameters dependant on training/set on initialization
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
