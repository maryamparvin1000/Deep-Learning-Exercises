import numpy as np

"""
Ex1
"""
class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):  # based on formula slide 1.11
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        Optimizer.__init__(self)
    #regularization page 13 and 14
    def calculate_update(self, weight_tensor, gradient_tensor):
        shrinkage_term = weight_tensor
        self.weights = weight_tensor
        # apply regularizeration
        if self.regularizer:
            reg = self.regularizer
            if reg.type == "L2":
                shrinkage_term = (1.0 - self.learning_rate * reg.alpha) * weight_tensor
            elif reg.type == "L1":
                shrinkage_term = weight_tensor - self.learning_rate * reg.alpha * np.sign(weight_tensor)
        new_weight = shrinkage_term - self.learning_rate * gradient_tensor
        return new_weight


"""
Ex2
"""
class SgdWithMomentum(Optimizer):  # slide 2.7
    def __init__(self, learning_rate: float, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.vk = 0
        Optimizer.__init__(self)

    def calculate_update(self, weight_tensor, gradient_tensor):
        shrinkage_term = weight_tensor
        self.weights = weight_tensor
        vk = (self.momentum_rate * self.vk) - (self.learning_rate * gradient_tensor)
        #new_weight = weight_tensor + vk
        self.vk = vk

        # apply regularizeration
        if self.regularizer:
            reg = self.regularizer
            if reg.type == "L2":
                shrinkage_term = (1.0 - self.learning_rate * reg.alpha) * weight_tensor
            elif reg.type == "L1":
                shrinkage_term = weight_tensor - self.learning_rate * reg.alpha * np.sign(weight_tensor)
        new_weight = shrinkage_term + self.vk
        return new_weight


class Adam(Optimizer):  # slide 2.9
    def __init__(self, learning_rate: float, mu, rho):
        self.iteration = 1.
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho

        self.vk = 0
        self.rk = 0
        Optimizer.__init__(self)

    def calculate_update(self, weight_tensor, gradient_tensor):
        gk = gradient_tensor
        self.vk = (self.mu * self.vk) + (1. - self.mu) * gk
        vk_bias = self.vk / (1. - np.power(self.mu, self.iteration))  # Bias correction
        self.rk = (self.rho * self.rk) + (1. - self.rho) * gk * gk
        rk_bias = self.rk / (1. - np.power(self.rho, self.iteration))  # Bias correction

        self.iteration += 1

        # apply regularizeration
        shrinkage_term = weight_tensor
        self.weights = weight_tensor
        if self.regularizer:
            reg = self.regularizer
            if reg.type == "L2":
                shrinkage_term = (1.0 - self.learning_rate * reg.alpha) * weight_tensor
            elif reg.type == "L1":
                shrinkage_term = weight_tensor - self.learning_rate * reg.alpha * np.sign(weight_tensor)
        new_weight = shrinkage_term - self.learning_rate * (vk_bias / (np.sqrt(rk_bias) + np.finfo(float).eps))

        #new_weight = weight_tensor - (self.learning_rate * (vk_bias / (np.sqrt(rk_bias) + np.finfo(float).eps)))

        return new_weight
