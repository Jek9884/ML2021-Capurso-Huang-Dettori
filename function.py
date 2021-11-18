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
def squared_loss(y, y1):
    return np.square(np.subtract(y, y1))


def squared_loss_deriv(y, y1):
    return np.subtract(y, y1)


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


# Activation function dictionary
identity_act_func = DerivableFunction(identity, identity_deriv, 'identity')
act_dict = {
    'identity': identity_act_func
}
