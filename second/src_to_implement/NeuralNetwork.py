import numpy as np
import copy


class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []  # contain the loss value for each iteration after calling train
        self.layers = []  # holds the architecture (fully connected layer)
        self.data_layer = None  # provides input_tensor and label_tensor
        self.loss_layer = None  # special layer providing loss and prediction
        self.label_tensor = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()  # Ex0 method that we implement
        entry = np.copy(self.input_tensor)

        for l in self.layers:
            output = l.forward(entry)
            entry = output

        output = self.loss_layer.forward(entry, self.label_tensor)
        return output

    def backward(self):

        output = self.loss_layer.backward(self.label_tensor)
        for layers in self.layers[::-1]:  # backpropogate from the last layer
            output_for = layers.backward(output)
            output = output_for

        return output

    def append_layer(self, layer):
        if layer.trainable:
            optimizer = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer
        self.layers.append(layer)

    # trains the network and stores the loss in each iteration
    def train(self, iterations):
        for i in range(iterations):
            loss_feedforward = self.forward()
            error_backward = self.backward()
            self.loss.append(loss_feedforward)
        return np.array(self.loss)

    def test(self, input_tensor):
        test_data = input_tensor
        for layer in self.layers:
            output = layer.forward(test_data)
            test_data = output
        return test_data





