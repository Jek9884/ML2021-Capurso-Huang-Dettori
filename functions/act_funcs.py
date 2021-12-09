import numpy as np
from functions.Function import Function, DerivableFunction

# Activation functions
def identity(x):
    return x


def identity_deriv(x):
    return np.ones(x.shape)


def sigm(x):
    return np.divide(1, np.add(1, np.power(np.e, np.negative(x))))


def sigm_deriv(x):
    return np.multiply(sigm(x), np.subtract(1, sigm(x)))


def tanh(x):
    prod = np.multiply(2, sigm(np.multiply(2, x)))
    return np.subtract(prod, 1)


def tanh_deriv(x):
    return np.multiply(4, sigm_deriv(np.multiply(2, x)))


def relu(x):
    return np.maximum(x, 0)

def relu_deriv(x):
    return np.maximum(np.sign(x), 0)

# Activation function dictionary
identity_act_func = DerivableFunction(identity, identity_deriv, 'identity')
sigm_act_func = DerivableFunction(sigm, sigm_deriv, 'sigm')
tanh_act_func = DerivableFunction(tanh, tanh_deriv, 'tanh')
relu_act_func = DerivableFunction(relu, relu_deriv, 'relu')
act_dict = {
    'identity': identity_act_func,
    'sigm': sigm_act_func,
    'tanh': tanh_act_func,
    'relu': relu_act_func
}
