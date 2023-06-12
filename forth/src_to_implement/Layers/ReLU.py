import numpy as np
from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    # return the input_tensor for the next layer
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    # based on slide 13, formula 12
    def backward(self, error_tensor):
        """
        Note: it is equivalent to If/ Else.
        if self.input_tensor > 0:
            error_tensor
        else:
            0
        """
        return np.where(self.input_tensor > 0, error_tensor, 0)


