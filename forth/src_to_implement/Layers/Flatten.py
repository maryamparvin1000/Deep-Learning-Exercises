from Layers.Base import BaseLayer
import numpy as np

"""
Flattening is converting the data into a 1-dimensional array for inputting it to the next layer.
"""


class Flatten(BaseLayer):
    def __init__(self):
        super(Flatten, self).__init__()
        pass

    def forward(self, input_tensor):
        self.tensor_shape = input_tensor.shape  # we used it later in backward
        # np.prod: Return the product of array elements.
        #reshape the input tensor
        return np.reshape(input_tensor, (input_tensor.shape[0], np.prod(input_tensor.shape[1:])))

    def backward(self, error_tensor):  # Just changing the shape
        return np.reshape(error_tensor, self.tensor_shape)

