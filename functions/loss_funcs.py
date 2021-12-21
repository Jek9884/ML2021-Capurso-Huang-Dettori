import numpy as np
from functions.function import Function, DerivableFunction

# Loss functions

def squared_loss(exp_val, pred_val, reduce_bool=False):
    loss_vec = 1/2*np.square(np.subtract(pred_val, exp_val))

    if reduce_bool:
        loss_vec = np.average(loss_vec, axis=0)

    return loss_vec

def squared_loss_deriv(exp_val, pred_val):
    return np.subtract(pred_val, exp_val)


# Note: nll for now supports only a sigmoid output 
def nll_loss_bin(exp_val, pred_val, reduce_bool=False):
    t1 = exp_val * np.log(pred_val)
    t2 = (1-exp_val) * np.log(1-pred_val)

    loss_vec = -np.add(t1, t2)

    if reduce_bool:
        loss_vec = np.average(loss_vec, axis=0)

    return loss_vec

def nll_loss_bin_deriv(exp_val, pred_val):
    return np.subtract(pred_val, exp_val)


# Loss function dictionary
squared_loss_func = DerivableFunction(squared_loss, squared_loss_deriv, 'squared')
nll_loss_bin_func = DerivableFunction(nll_loss_bin, nll_loss_bin_deriv, 'nll')
loss_dict = {
    'squared': squared_loss_func,
    'nll': nll_loss_bin_func
}
