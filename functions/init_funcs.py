import numpy as np
from functions.function import Function

# weight initialisation functions (layer-wise, assume a shape (unit, feature))

"""
    Standard initialization of weights matrix
    
    Parameters:
        -shape: shape of the weights matrix
        -scale: width of the uniform distribution
"""


def std_weight_init(shape, scale=0):

    # defaults to number of nodes in layer as scale for uniform distr
    if scale == 0:
        scale = shape[1]

    init_matrix = np.random.uniform(low=-1 / np.sqrt(scale),
                                    high=1 / np.sqrt(scale),
                                    size=shape)

    return init_matrix


"""
    Xavier init
"""


def norm_weight_init(shape):
    init_matrix = np.random.uniform(low=-np.sqrt(6 / (shape[0] + shape[1])),
                                    high=np.sqrt(6 / (shape[0] + shape[1])),
                                    size=shape)

    return init_matrix


"""
    He init (for linear rectifiers)    
"""


def he_weight_init(shape):
    init_matrix = np.random.normal(0, np.sqrt(2 / shape[1]), size=shape)

    return init_matrix


# Weight initialisation dictionary
std_init_func = Function(std_weight_init, "std")
norm_init_func = Function(norm_weight_init, "norm")
he_init_func = Function(he_weight_init, "he")
init_dict = {
    'std': std_init_func,
    'norm': norm_init_func,
    'he': he_init_func
}
