import numpy as np
from Layers.Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.prediction = None

    def forward(self, input_tensor):
        # based on slide 14 and 15
        x_k = input_tensor - np.max(input_tensor)  # To increase numerical stability
        self.prediction = np.divide(np.exp(x_k), np.expand_dims(np.sum(np.exp(x_k), axis=1), axis=1))
        # Formula 13
        # np.expand_dims: it has to be column
        return self.prediction  # it is used in backward's formula

    def backward(self, error_tensor):
        # based on slide 16
        """
        note: we are not allowed to use for loop. So, we have to use
        numpy methods, such as np.expand_dims, in order to iterate
        all columns, and as for Integral, we use np.sum.
        """
        backward_input = np.sum(error_tensor * self.prediction, axis=1, keepdims=True)  # Colum vectors
        backward_output = self.prediction * (error_tensor - backward_input)  # Formula 14
        return backward_output
