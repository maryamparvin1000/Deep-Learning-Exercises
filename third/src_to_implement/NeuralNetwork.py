import numpy as np
from Optimization.Optimizers import *
import copy


class NeuralNetwork:
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
        self.input_tensor, self.label_tensor = self.data_layer.forward()

        layer_pointer = 1
        for layer in self.layers:
            # print("layer ",layer)
            if layer_pointer == 1:
                out = layer.forward(self.input_tensor)
                layer_pointer = 2
            else:
                # print("out size", out.shape)
                out = layer.forward(out)
            self.pred = out

        return self.loss_layer.forward(out, self.label_tensor)

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
        for _ in range(iterations):
            loss = self.forward()
            self.backward()
            self.loss.append(loss)

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
