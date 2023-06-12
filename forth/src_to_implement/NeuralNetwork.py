import numpy as np
from Optimization.Optimizers import *
import copy


class NeuralNetwork:
    testing_phase = bool
    regularizatin_loss = 0
    def __init__(self, optimizers, weights_initializer, bias_initializer):
        self.optimizers = optimizers
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.label_tensor = None
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        input_tensor = input_tensor.astype('float64')
        self.label_tensor = self.label_tensor.astype('float64')
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if layer.trainable and layer.optimizer.regularizer:
                self.regularizatin_loss += layer.optimizer.regularizer.norm(layer.weights)  # λ||w||2
        if self.loss_layer is not None:
            input_tensor = self.loss_layer.forward(input_tensor, self.label_tensor)
            input_tensor += self.regularizatin_loss
        return input_tensor

    def backward(self):
        output_back = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:
            output_back = layer.backward(output_back)

    def append_layer(self, layer):
        if layer.trainable:
            optimizer = copy.deepcopy(self.optimizers)
            layer.optimizer = optimizer
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = False
        for _ in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        self.phase = True
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    @property
    def phase(self):
        return self.testing_phase

    @phase.setter
    def phase(self, var):
        self.testing_phase = var
