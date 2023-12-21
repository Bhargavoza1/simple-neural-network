# Base class
class Module:
    def __init__(self):
        pass

    def forward(self, input_features: float):
        pass

    def backpropagation(self, output_error: float, learning_rate: float):
        pass