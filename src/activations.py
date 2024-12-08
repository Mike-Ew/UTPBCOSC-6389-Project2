import math


class ActivationFunction:
    def __init__(self, func, dfunc_output_based=False, dfunc=None):
        # dfunc_output_based indicates if derivative is computed using function output instead of input
        self.func = func
        self.dfunc_output_based = dfunc_output_based
        self.dfunc = dfunc


# Sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def d_sigmoid(y):
    # derivative given output y of sigmoid
    return y * (1.0 - y)


SIGMOID = ActivationFunction(func=sigmoid, dfunc_output_based=True, dfunc=d_sigmoid)


# Tanh
def d_tanh(y):
    # derivative given output y of tanh
    return 1 - y * y


TANH = ActivationFunction(func=math.tanh, dfunc_output_based=True, dfunc=d_tanh)


# ReLU
def relu(x):
    return x if x > 0 else 0


def d_relu(x):
    # For ReLU, we need the input x (pre-activation)
    return 1.0 if x > 0 else 0.0


RELU = ActivationFunction(func=relu, dfunc_output_based=False, dfunc=d_relu)
