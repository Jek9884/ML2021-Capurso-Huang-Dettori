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
