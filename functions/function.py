class Function:

    def __init__(self, func, name, aim=None):
        self.__func = func
        self.__name = name
        self.__aim = aim  # Min/Max/None

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    @property
    def func(self):
        return self.__func

    @property
    def name(self):
        return self.__name

    @property
    def aim(self):
        return self.__aim

    def __repr__(self):
        return self.__name


class DerivableFunction(Function):

    def __init__(self, func, deriv, name, aim=None):
        super(DerivableFunction, self).__init__(func=func, name=name, aim=aim)
        self.__deriv = deriv

    @property
    def deriv(self):
        return self.__deriv
