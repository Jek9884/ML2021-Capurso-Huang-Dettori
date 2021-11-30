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
    return np.subtract(pred_val, exp_val)


def nll_loss(exp_val, pred_val):  # Negative log-likelihood

    if exp_val == 0:
        return -np.log(1 - pred_val)
    elif exp_val == 1:
        return -np.log(pred_val)
    else:
        raise ValueError("Supports only binary classification")


def nll_loss_deriv(exp_val, pred_val):
    # Use only with sigmoid!!!

    return np.subtract(pred_val, exp_val)


# Loss function dictionary
squared_loss_func = DerivableFunction(squared_loss, squared_loss_deriv, 'squared')
nll_loss_func = DerivableFunction(nll_loss, nll_loss_deriv, 'nll')
loss_dict = {
    'squared': squared_loss_func,
    'nll': nll_loss_func
}


# Activation functions
def identity(x):
    return x


def identity_deriv(x):
    return np.ones(len(x))


def sigm(x):
    return 1 / (1 + np.power(np.e, -x))


def sigm_deriv(x):
    return np.multiply(sigm(x), 1 - sigm(x))


def tanh(x):
    return 2 * sigm(2 * x) - 1


def tanh_deriv(x):
    return 4 * sigm_deriv(2 * x)


# Activation function dictionary
identity_act_func = DerivableFunction(identity, identity_deriv, 'identity')
sigm_act_func = DerivableFunction(sigm, sigm_deriv, 'sigm')
tanh_act_func = DerivableFunction(tanh, tanh_deriv, 'tanh')
act_dict = {
    'identity': identity_act_func,
    'sigm': sigm_act_func,
    'tanh': tanh_act_func
}


# Weight initialisation functions (layer-wise, assume a shape (unit, feature))
def std_weight_init(shape, sparse_count=0):
    init_matrix = np.random.uniform(low=-1 / np.sqrt(shape[1]),
                                    high=1 / np.sqrt(shape[1]),
                                    size=shape)

    if 0 < sparse_count < shape[1]:

        init_matrix = sparsify_weight_matrix(init_matrix, sparse_count)

    elif sparse_count >= shape[1]:
        raise ValueError("Invalid value for weight sparsification")

    return init_matrix


# Xavier init
def norm_weight_init(shape, sparse_count=0):
    init_matrix = np.random.uniform(low=-np.sqrt(6 / (shape[0] + shape[1])),
                                    high=np.sqrt(6 / (shape[0] + shape[1])),
                                    size=shape)

    if 0 < sparse_count < shape[1]:

        init_matrix = sparsify_weight_matrix(init_matrix, sparse_count)

    elif sparse_count >= shape[1]:
        raise ValueError("Invalid value for weight sparsification")

    return init_matrix


def sparsify_weight_matrix(matrix, sparse_count):

    shape = matrix.shape
    shape_ones_mat = (shape[0], shape[1]-sparse_count)
    shape_zeros_mat = (shape[0], sparse_count)

    mask_ones_mat = np.ones(shape=shape_ones_mat)
    mask_zeros_mat = np.zeros(shape=shape_zeros_mat)

    final_mask = np.concatenate((mask_ones_mat, mask_zeros_mat), axis=1)

    rng = np.random.default_rng()
    final_mask = rng.permuted(final_mask, axis=1)

    return np.multiply(matrix, final_mask)


# Weight initialisation dictionary
std_init_func = Function(std_weight_init, "std init")
norm_init_func = Function(norm_weight_init, "norm init")
init_dict = {
    'std': std_init_func,
    'norm': norm_init_func
}
