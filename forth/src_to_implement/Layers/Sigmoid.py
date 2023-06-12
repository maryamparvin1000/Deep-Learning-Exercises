import numpy as np

class Sigmoid:
    def __init__(self):
        input_tensor: np.ndarray
        self.activations = None
        self.trainable = False

    # activation functions page 14
    def forward(self,input_tensor):
        self.activations = 1/(1 + np.exp(-input_tensor))
        return self.activations

    def backward(self,error_tensor):

        back = np.multiply(1 - self.activations , self.activations)
        back = np.multiply(back,error_tensor)
        return back