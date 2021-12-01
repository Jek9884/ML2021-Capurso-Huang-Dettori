import numpy as np
from functions.Function import Function, DerivableFunction


# Loss functions
def squared_loss(exp_val, pred_val):
    return np.square(np.subtract(exp_val, pred_val))


def squared_loss_deriv(exp_val, pred_val):
    return np.subtract(pred_val, exp_val)


def nll_loss(exp_val, pred_val):  # Negative log-likelihood

    if exp_val == 0:
        return np.negative(np.log(1 - pred_val))
    elif exp_val == 1:
        return np.negative(np.log(pred_val))
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
