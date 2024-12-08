import math


class CostFunction:
    def __init__(self, func, dfunc):
        # func(y_pred, y_true), dfunc(y_pred, y_true)
        self.func = func
        self.dfunc = dfunc


# Mean Squared Error
def mse(y_pred, y_true):
    return 0.5 * (y_pred - y_true) ** 2


def d_mse(y_pred, y_true):
    return y_pred - y_true


MSE = CostFunction(func=mse, dfunc=d_mse)


# Cross-Entropy
def cross_entropy(y_pred, y_true):
    eps = 1e-9
    return -(
        y_true * math.log(y_pred + eps) + (1 - y_true) * math.log(1 - y_pred + eps)
    )


def d_cross_entropy(y_pred, y_true):
    eps = 1e-9
    return (y_pred - y_true) / ((y_pred * (1 - y_pred)) + eps)


CROSS_ENTROPY = CostFunction(func=cross_entropy, dfunc=d_cross_entropy)
