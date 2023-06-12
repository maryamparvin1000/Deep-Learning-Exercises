import numpy as np


class Constant:
    def __init__(self, constant):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.zeros((fan_in, fan_out)) + self.constant


class UniformRandom:
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        return np.random.rand(fan_in, fan_out)

#slide 5
class Xavier: #normal Gaussian
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt(2 / (fan_out + fan_in)), weights_shape)

#slide 6
class He:
    @staticmethod
    def initialize(weights_shape, fan_in, fan_out):
        return np.random.normal(0, np.sqrt(2 / fan_in), weights_shape)
