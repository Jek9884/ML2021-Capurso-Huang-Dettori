import numpy as np
from functions.function import Function, DerivableFunction

# Loss functions return a 1-dim vector
# Loss derivatives return a vector consisting of the derivative wrt feature

def squared_loss(exp_val, pred_val, reduce_bool=False):
    loss_vec = 1/2*np.square(np.subtract(pred_val, exp_val))

    # In case of multiple targets sum them together
    loss_vec = np.sum(loss_vec, axis=1)

    if reduce_bool:
        loss_vec = np.average(loss_vec, axis=0)

    return loss_vec

def squared_loss_deriv(exp_val, pred_val):
    return np.subtract(pred_val, exp_val)

def euclidean_loss(exp_val, pred_val, reduce_bool=False):
    loss_vec = np.linalg.norm(np.subtract(pred_val, exp_val), axis=1)

    if reduce_bool:
        loss_vec = np.average(loss_vec, axis=0)

    return loss_vec

def euclidean_loss_deriv(exp_val, pred_val):
    loss_num = np.subtract(pred_val, exp_val)
    loss_den = euclidean_loss(exp_val, pred_val)

    # Add an empty dimension to the denominator to make divide work
    return np.divide(loss_num, loss_den[:, None])

# Note: nll for now supports only a sigmoid output 
# eps value suggested by sklearn log_loss implementation
def nll_loss_bin(exp_val, pred_val, reduce_bool=False, eps=10**-15):
    t1 = exp_val * np.log(np.maximum(pred_val, eps))
    t2 = (1-exp_val) * np.log(np.maximum(1-pred_val, eps))

    loss_vec = -np.add(t1, t2)

    if reduce_bool:
        loss_vec = np.average(loss_vec, axis=0)

    return loss_vec

def nll_loss_bin_deriv(exp_val, pred_val):
    return np.subtract(pred_val, exp_val)


# Loss function dictionary
squared_loss_func = DerivableFunction(squared_loss, squared_loss_deriv, 'squared', 'min')
euclidean_loss_func = DerivableFunction(euclidean_loss, euclidean_loss_deriv, 'euclidean', 'min')
nll_loss_bin_func = DerivableFunction(nll_loss_bin, nll_loss_bin_deriv, 'nll', 'min')

loss_dict = {
    'squared': squared_loss_func,
    'euclidean': euclidean_loss_func,
    'nll': nll_loss_bin_func
}
