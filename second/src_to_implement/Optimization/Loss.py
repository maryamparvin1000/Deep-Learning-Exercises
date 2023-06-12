import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        # According to slide 17, Formula 15
        self.prediction_tensor = prediction_tensor
        prediction_tensor = prediction_tensor[label_tensor == 1]  # where y_k == 1
        # Integral(-1 * ln(y_hat_k + eps))
        return np.sum((-1) * np.log(prediction_tensor + np.finfo(float).eps))

    def backward(self, label_tensor):
        """
        According to slide 18, formula 16
        E_n = (-1) * y / y_hat
        """
        return np.multiply((-1), np.divide(label_tensor, self.prediction_tensor))
