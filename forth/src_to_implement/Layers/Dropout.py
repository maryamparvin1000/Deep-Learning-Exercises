import numpy as np


#used to regulaize fully connected layers
from Layers.Base import BaseLayer


class Dropout(BaseLayer):
    probability: float

    # probability: determining the fraction units to keep
    def __init__(self, probability):
        self.trainable = False
        self.probability = probability
        self.mask = None
        self.testing_phase = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        #testing phase
        if self.testing_phase:
            return self.input_tensor

        # training phase
        else:
            mask = (np.random.uniform(0, 1, input_tensor.shape) < self.probability)
            self.mask = mask
            self.input_tensor = np.multiply(self.input_tensor, self.mask)
            self.input_tensor = np.multiply(self.input_tensor, 1 / self.probability)
            return self.input_tensor
            #return input_tensor / self.probability

    def backward(self, error_tensor):
        #testing phase
        if self.testing_phase:
            return error_tensor
        # training phase
        else:
            error_tensor = np.multiply(error_tensor, self.mask)
            error_tensor = np.multiply(error_tensor, 1 / self.probability)
            return  error_tensor
            #return error_tensor / self.probability