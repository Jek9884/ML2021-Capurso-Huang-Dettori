import numpy as np
from functions.function import Function, DerivableFunction


# Weight initialisation functions (layer-wise, assume a shape (unit, feature))
def std_weight_init(shape, scale=0, sparse_count=0):

    # Defaults to number of nodes in layer as scale for uniform distr
    if scale == 0:
        scale = shape[1]

    init_matrix = np.random.uniform(low=-1 / np.sqrt(scale),
                                    high=1 / np.sqrt(scale),
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
std_init_func = Function(std_weight_init, "std")
norm_init_func = Function(norm_weight_init, "norm")
init_dict = {
    'std': std_init_func,
    'norm': norm_init_func
}
