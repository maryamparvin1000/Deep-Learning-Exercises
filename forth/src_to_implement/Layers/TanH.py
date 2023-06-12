import numpy as np
from Layers.Base import BaseLayer


class TanH(BaseLayer):
    input_tensor: np.ndarray
    activations = None



    def __init__(self):
        self.trainable = False
    #activation functions page 18
    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        back = np.multiply(1 - np.power(self.activations,2),error_tensor)
        return back
