from Layers.Base import BaseLayer
import numpy as np


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(self.input_size+1,self.output_size)  # adding one dimention for bias
        self.trainable = True
        self._optimizer = None
        self._gradient_weights = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, x):
        self._optimizer = x

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, gradient):
        self._gradient_weights = gradient

    # based on slide 16/ 4
    def forward(self, input_tensor):
        adding_array = np.ones(len(input_tensor))
        input = np.array([adding_array])
        # one layer for bias will be added
        self.input_tensor = np.concatenate((input_tensor, np.transpose(input)), axis=1)
        y_hat = np.dot(self.input_tensor, self.weights)
        return y_hat

    # based on slide 21 /5
    def backward(self, error_tensor):
        self.error_tensor = np.dot(error_tensor, self.weights.T)
        # we should remove the last bias column
        self.error_tensor = np.delete(self.error_tensor, -1, axis=1)  # -1: removing last element
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        # update the weights if the optimizer is not none
        if self.optimizer != None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        # the output batch_size x input_size
        return self.error_tensor



