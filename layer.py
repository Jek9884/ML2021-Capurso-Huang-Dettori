import numpy as np

"""
    Layer class
    
    Parameters:
        -n_out: number of units of the layer
        -n_in: number of inputs to the layer
        -init_func: weights init function
        -act_func: activation function object (DerivableFunction)
        -bias_init: bias initialization value
        -init_scale: scale of uniform distribution in std weight init
        -batch_norm: boolean for batch normalization activation
        -batch_momentum: momentum value for the moving avg of batch normalization (testing mode)
        -dropout: boolean for dropout activation
        -dropout_keep: probability of keeping a unit
        -debug_bool: print debug information from the layer
    
    Attributes:
        -weights: matrix of weights of the layer (excluding bias)
        -bias: bias vector of the layer
        -batch_gamma: vector of gamma values for batch normalization
        -batch_beta: vector of beta values for batch normalization
        -grad_w: gradient of error w.r.t. weights
        -grad_b: gradient of error w.r.t. bias
        -net: dot product btw weights matrix and input matrix
        -delta_w_old: difference between current and previous weights (for momentum)
        -delta_b_old: difference between current and previous bias (for momentum)
        -batch_mean: vector of means of units w.r.t. the patterns (batch normalization training) 
        -batch_var: vector of variance of units w.r.t. the patterns (batch normalization training)
        -moving_mean: vector of moving avg of means of units (batch normalization inference)
        -moving_var: vector of moving avg of variance of units (batch normalization inference)
        -net_hat: standardized net (batch normalization)
        -batch_eps: constant used for stability of batch normalization (avoid dividing by 0)
        -grad_gamma: gradient of error w.r.t. batch_gamma
        -grad_beta: gradient of error w.r.t. batch_beta
        -layer_in: input value for the layer (vector)
        -y: Variable representing the input of the non-linearity: 
                -net in the case of normal backpropagation
                -the transformed net in case of batch normalisation
        -out: output of the layer's non-linearity
"""


class Layer:

    def __init__(self, n_out, n_in, init_func=None, act_func=None, bias_init=None,
                 init_scale=0, batch_norm=False, batch_momentum=0.99, dropout=False,
                 dropout_keep=0.5, debug_bool=False):

        self.n_in = n_in
        self.n_out = n_out
        self.bias_init = bias_init

        # Functions
        self.act_func = act_func
        self.init_func = init_func
        self.init_scale = init_scale

        # Layer parameters
        self.weights = None
        self.bias = None
        self.batch_gamma = None
        self.batch_beta = None

        # Variables used to implement standard backpropagation
        self.grad_w = None
        self.grad_b = None
        self.net = None
        self.delta_w_old = None
        self.delta_b_old = None

        # Variables used to perform batch normalisation
        self.batch_norm = batch_norm
        self.batch_mean = None  # Training
        self.batch_var = None  # Training
        self.moving_mean = 0  # Inference
        self.moving_var = 0  # Inference
        self.batch_momentum = batch_momentum  # Inference
        self.net_hat = None
        self.batch_eps = 10**-6

        # Variables used to perform batch normalisation backpropagation
        self.grad_gamma = None
        self.grad_beta = None

        # Dropout hyper parameters
        self.dropout = dropout
        self.dropout_keep = dropout_keep

        self.layer_in = None

        self.y = None

        self.out = None

        self.debug_bool = debug_bool

        self.reset_parameters()

    '''
        Reset the parameters of the layer

        Used instead of generating a new layer
    '''
    def reset_parameters(self):

        # Variables used by the optimizer to implement momentum
        self.delta_w_old = np.zeros((self.n_out, self.n_in))
        self.delta_b_old = np.zeros((self.n_out,))

        if self.init_func is None:
            # Each row is composed of the weights of the unit
            self.weights = np.ones((self.n_out, self.n_in))

        # Specify init scale in case of standard initialisation
        elif self.init_func.name == "std":
            self.weights = self.init_func((self.n_out, self.n_in), self.init_scale)

        else:
            self.weights = self.init_func((self.n_out, self.n_in))

        # Default value for bias is a vector of 0, unless otherwise specified
        if self.bias_init is None:
            self.bias = np.zeros((self.n_out,))
        else:
            self.bias = np.full((self.n_out,), self.bias_init, dtype=np.float64)

        self.batch_gamma = np.full((self.n_out,), 1, dtype=np.float64)
        self.batch_beta = np.full((self.n_out,), 1, dtype=np.float64)

        self.null_grad()

    """
        Computes layer forward pass. Returns matrix of layer's outputs

        Parameters:
            -in_mat: matrix of input data
            -training:  boolean used to distinguish between training and inference mode for 
                batch_norm and dropout purposes
    """

    def forward(self, in_mat, training=False):

        self.layer_in = in_mat

        # dropout implementation
        if training and self.dropout:
            dropout_mask = np.random.binomial(1, self.dropout_keep,
                                              size=self.layer_in.shape)
            # Used to implement inverted dropout
            dropout_scale = 1 / self.dropout_keep
            self.layer_in = self.layer_in * dropout_mask * dropout_scale

        net_wo_bias = np.matmul(self.layer_in, np.transpose(self.weights))
        self.net = np.add(net_wo_bias, self.bias)

        if training:

            # batch normalization implementation
            if self.batch_norm:
                self.batch_mean = np.mean(self.net, axis=0)
                self.batch_var = np.var(self.net, axis=0)

                # standardised net
                self.net_hat = (self.net-self.batch_mean) / \
                    np.sqrt(self.batch_var+self.batch_eps)

                # update moving stats for inference
                self.moving_mean = self.moving_mean*self.batch_momentum +\
                    (1-self.batch_momentum)*np.mean(self.net, axis=0)

                self.moving_var = self.moving_var*self.batch_momentum +\
                    (1-self.batch_momentum)*np.var(self.net, axis=0)

                # batch normalised net
                self.y = self.batch_gamma*self.net_hat+self.batch_beta

            else:
                self.y = self.net

        else:

            if self.batch_norm:
                # note: moving stats break gradient checking
                self.net_hat = (self.net-self.moving_mean) / \
                    np.sqrt(self.moving_var+self.batch_eps)

                # batch normalised net
                self.y = self.batch_gamma*self.net_hat+self.batch_beta

            else:
                self.y = self.net

        self.out = self.act_func(self.y)

        return self.out

    """
        Computes layer backward. Returns derivative of error w.r.t. the output of the next 
        (w.r.t. backward order) layer for next layer's backward

        Parameters:
            -d_err_d_y: derivative of error w.r.t. the input of the non-linearity (delta)
    """

    def backward(self, d_err_d_y):

        if self.batch_norm:

            # Used multiple times
            inv_sqrt_var = 1/np.sqrt(self.batch_var+self.batch_eps)
            batch_size = self.net.shape[0]

            self.grad_gamma = np.sum(d_err_d_y*self.net_hat, axis=0)  # scalar output
            self.grad_beta = np.sum(d_err_d_y, axis=0)  # scalar output

            d_err_d_net_hat = d_err_d_y * self.batch_gamma

            d_var_sum_elem = d_err_d_net_hat*(self.net-self.batch_mean)*\
                (-1/2)*inv_sqrt_var**3
            d_err_d_batch_var = np.sum(d_var_sum_elem, axis=0)

            d_err_d_batch_mean = np.sum(d_err_d_net_hat*(-inv_sqrt_var), axis=0)

            d_err_d_net = d_err_d_net_hat*inv_sqrt_var +\
                d_err_d_batch_var*2*(self.net-self.batch_mean)/batch_size +\
                d_err_d_batch_mean/batch_size
        else:
            d_err_d_net = d_err_d_y

        # grad += delta_i * output of previous layer (o_u)
        self.grad_w = np.dot(np.transpose(d_err_d_net), self.layer_in)
        self.grad_b = np.sum(d_err_d_net, axis=0)

        # take the average of the gradients across patterns
        num_patt = len(self.layer_in)
        self.grad_w = np.divide(self.grad_w, num_patt)
        self.grad_b = np.divide(self.grad_b, num_patt)

        new_d_err_d_out = np.dot(d_err_d_net, self.weights)

        if self.debug_bool:
            print("Layer")
            print("Activation function: ", self.act_func.name)
            print("\tInput: ", self.layer_in)
            print("\tExpected out: ", self.act_func(self.y))
            print("\ty (net or batch-normalised net): ", self.y)
            print("\tBias: ", self.bias)
            print("\tDelta: ", d_err_d_y)
            print("\tDeriv act(y): ", self.act_func.deriv(self.y))
            print("\tGrad weights: \n", self.grad_w)
            print("\tGrad bias: ", self.grad_b)
            print("\tBatch normalisation: ", self.batch_norm)
            print()

        return new_d_err_d_out

    '''
        Zero out the gradient variables of the layer
    '''
    def null_grad(self):

        self.grad_w = None
        self.grad_b = None

        self.grad_gamma = None
        self.grad_beta = None

    def __str__(self):

        cur_str = f"\tnumber units: {self.n_out}, number weights: {self.n_in}"
        cur_str += f"\n\tlayer weights matrix: {self.weights}\n"
        cur_str += f"\n\tlayer bias vector: {self.bias}\n"
        cur_str += f"\n\tlayer gradient weights matrix: {self.grad_w}\n"
        cur_str += f"\n\tlayer gradient bias vector: {self.grad_b}\n"
        cur_str += f"\n\tactivation function: {self.act_func.name}"

        return cur_str
