import numpy as np

class L2_Regularizer:
    #alpha representing regulariyation weights
    def __init__(self, alpha):
        self.alpha = alpha
        self.type = "L2"

    # calculates a (sub-)gradient on the weights needed for the optimizer
    def calculate_gradient(self, weights):
        return self.alpha * weights

    # calculates the norm enhanced loss
    def norm(self, weights):
        return self.alpha * np.square(np.linalg.norm(weights))

class L1_Regularizer:

    def __init__(self, alpha):
        self.alpha = alpha
        self.type = "L1"

    # calculates a (sub-)gradient on the weights needed for the optimizer
    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)


    # calculates the norm enhanced loss
    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))

