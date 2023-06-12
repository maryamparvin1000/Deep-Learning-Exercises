class BaseLayer:
    def __init__(self):
        self.testing_phase = False
        self.trainable = False
        self.weights = None
