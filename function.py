import numpy as np


class Function:

    def __init__(self, func, name):
        self.__func = func
        self.__name = name

    @property
    def func(self):
        return self.__func

    @property
    def name(self):
        return self.__name


class DerivableFunction(Function):

    def __init__(self, func, deriv, name):
        super(DerivableFunction, self).__init__(func=func, name=name)
        self.__deriv = deriv

    @property
    def deriv(self):
        return self.__deriv


# Loss functions
def squared_loss(exp_val, pred_val):
    return np.square(np.subtract(exp_val, pred_val))


def squared_loss_deriv(exp_val, pred_val):
    return np.subtract(exp_val, pred_val)


# Loss function dictionary
squared_loss_func = DerivableFunction(squared_loss, squared_loss_deriv, 'squared')
loss_dict = {
    'squared': squared_loss_func
}


# Activation functions
def identity(x):
    return x


def identity_deriv(x):
    return np.ones(len(x))


def sigm(x):
    return 1 / (1 + np.power(np.e, -x))


def sigm_deriv(x):
    return np.power(np.e, -x) / np.power(1 + np.power(np.e, -x), 2)


# Activation function dictionary
identity_act_func = DerivableFunction(identity, identity_deriv, 'identity')
sigm_act_func = DerivableFunction(sigm, sigm_deriv, 'sigm')
act_dict = {
    'identity': identity_act_func,
    'sigm': sigm_act_func
}


# Weight initialisation functions (layer-wise)
def std_weight_init(n_out, n_in, sparse_count=0):
    init_mat = np.random.uniform(low=-1 / np.sqrt(n_in),
                                 high=1 / np.sqrt(n_in),
                                 size=(n_out, n_in))

    if 0 < sparse_count < n_in:

        zeroed_ind = np.arange(n_in)

        for i in range(n_out):
            np.random.shuffle(zeroed_ind)
            init_mat[i][zeroed_ind[:sparse_count]] = 0

    return init_mat


# Xavier init
def norm_weight_init(n_out, n_in, sparse_count=0):
    init_mat = np.random.uniform(low=-np.sqrt(6 / (n_in + n_out)),
                                 high=np.sqrt(6 / (n_in + n_out)),
                                 size=(n_out, n_in))

    if 0 < sparse_count < n_in:

        zeroed_ind = np.arange(n_in)

        for i in range(n_out):
            np.random.shuffle(zeroed_ind)
            init_mat[i][zeroed_ind[:sparse_count]] = 0

    return init_mat


# Weight initialisation dictionary
init_dict = {
    'std': std_weight_init,
    'norm': norm_weight_init
}
