import numpy as np
from functions.function import Function, DerivableFunction


# Loss functions
def squared_loss(exp_val, pred_val):
    return 1/2*np.square(np.subtract(exp_val, pred_val))

def squared_loss_deriv(exp_val, pred_val):
    return np.subtract(pred_val, exp_val)


# Note: nll for now supports only a sigmoid output 
# nll_loss assumes the network output net as input, the sigmoid is already factored in
def nll_loss_bin(exp_val, pred_val_net):  # Negative log-likelihood
    t1 = np.multiply(np.subtract(1, exp_val), pred_val_net)
    t2 = np.log(1+np.power(np.e, np.negative(pred_val_net)))

    return np.add(t1, t2)

def nll_loss_bin_deriv(exp_val, pred_val):
    return np.subtract(pred_val, exp_val)


# Loss function dictionary
squared_loss_func = DerivableFunction(squared_loss, squared_loss_deriv, 'squared')
nll_loss_bin_func = DerivableFunction(nll_loss_bin, nll_loss_bin_deriv, 'nll')
loss_dict = {
    'squared': squared_loss_func,
    'nll': nll_loss_bin_func
}


# Error functions
def squared_error(exp_val, pred_val):
    res = np.multiply(1/2, squared_loss(exp_val, pred_val))
    return np.average(res, axis=0)

def nll_error(exp_val, pred_val):
    return np.average(nll_loss_bin(exp_val, pred_val), axis=0)

# Error function dictionary
squared_error_func = Function(squared_error, 'squared')
nll_error_func = Function(nll_error, 'nll')
error_dict = {
    'squared': squared_error_func,
    'nll': nll_error_func
}
