# Base class
class Module:
    def __init__(self):
       pass

    def forward(self, input_features: float):
        raise NotImplementedError("Subclasses must implement my_abstract_method")

    def backpropagation(self, output_error: float, learning_rate: float):
        raise NotImplementedError("Subclasses must implement my_abstract_method")