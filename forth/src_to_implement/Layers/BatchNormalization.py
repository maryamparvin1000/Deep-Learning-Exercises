import numpy as np
from .Base import BaseLayer
from .Helpers import compute_bn_gradients
from copy import deepcopy


class BatchNormalization(BaseLayer):
    channels: int
    _optimizer: object = None
    _gradient_weights: np.ndarray(shape=None, dtype=np.float64) = None
    _gradient_bias: np.ndarray(shape=None, dtype=np.float64) = None
    weights = None
    bias = None
    mean = None
    var = None
    mean_tilda = None
    var_tilda = None
    B = None
    H = None
    M = None
    N = None
    def __init__(self, channels):
        self.trainable = True
        self.channels = channels  # The number of channels of input tensor
        self.testing_phase = False
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))

    def initialize(self, weights_initializer, bias_initializer):

        self.weights = weights_initializer.initialize(
            (1, self.channels), 1, self.channels)
        self.bias = bias_initializer.initialize(
            (1, self.channels), 1, self.channels
        )

    def reformat(self, tensor):
        # B × H × M × N tensor to B × H × M · N
        if len(tensor.shape) == 4:  # image_like to vector_like
            self.B = tensor.shape[0]
            self.H = tensor.shape[1]
            self.M = tensor.shape[2]
            self.N = tensor.shape[3]

            tensor = np.reshape(tensor, (self.B, self.H, self.M * self.N))
            tensor = tensor.transpose(0, 2, 1)# self.B, self.M * self.N, self.H
            output = np.reshape(tensor, (self.B * self.M * self.N, self.H))
        else:  # vector_like to image_like
            tensor = np.reshape(tensor, (self.B, self.M * self.N, self.H))
            tensor = tensor.transpose(0, 2, 1)#self.B, self.H, self.M * self.N
            output = np.reshape(tensor, (self.B, self.H, self.M, self.N))
        return output

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        is_image = False  # for reformating
        alpha = .8  # For moving average estimation, moving average decay

        if len(input_tensor.shape) == 4:
            self.input_tensor = self.reformat(input_tensor)
            is_image = True  # for reformating

        if not self.testing_phase:
            self.mean = np.mean(self.input_tensor, axis=0, keepdims=1)  # for 4d array for each batch and channel
            self.var = np.var(self.input_tensor, axis=0, keepdims=1)

            if self.mean_tilda is None:  # Initialize with first batche's mean and var
                self.mean_tilda = self.mean
                self.var_tilda = self.var
            #page 27 reg
            self.input_tensor_tilda = np.divide(self.input_tensor - self.mean, np.sqrt(self.var + 1e-10))
            output_tensor = self.input_tensor_tilda.transpose(0, 1) * self.weights + self.bias

            # Moving average estimation, page31
            self.mean_tilda = alpha * self.mean_tilda + (1 - alpha) * self.mean
            self.var_tilda = alpha * self.var_tilda + (1 - alpha) * self.var

        else:  # Test time
            input_tensor_tilda = np.divide(self.input_tensor - self.mean_tilda, np.sqrt(self.var_tilda + 1e-10))
            output_tensor = input_tensor_tilda.transpose(0, 1) * self.weights + self.bias

        if is_image:  # If it was 4D reformat to the original shapes
            output_tensor = self.reformat(output_tensor)
        return output_tensor

    def backward(self, error_tensor):
        is_image = False
        # input_tensor = self.input_tensor  # pass input to compute_bn_gradients
        if len(error_tensor.shape) == 4:
            error_tensor = self.reformat(error_tensor)
            is_image = True  # for reformating
        # Gradiant with respect to weight for 2d
        gradiant_with_respect_to_X = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean,
                                                          self.var, eps=1e-10)
        self._gradient_weights = np.sum(error_tensor * self.input_tensor_tilda, axis=0,
                                        keepdims=True)  # Sum over channels
        self._gradient_bias = np.sum(error_tensor, axis=0, keepdims=True)  # Sum over channels

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        error_tensor = gradiant_with_respect_to_X
        if is_image:
            error_tensor = self.reformat(error_tensor)
        return error_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, var):
        self._optimizer = var
        self._optimizer_bias = deepcopy(var)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, var):
        self._gradient_weights = var

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, var):
        self._gradient_bias = var
