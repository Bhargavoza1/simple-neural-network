from module import Module
import numpy as np

class Linear(Module):
    def __init__(self, input_size: int, output_size: int, bias: bool = True, w_b_range: float = 0.5):
        # Initialize weights with random values between -w_b_range and w_b_range
        self.weights = np.random.rand(input_size, output_size) - w_b_range

        # Initialize bias if bias is True, otherwise set bias to 0.0
        self.bias = np.random.rand(1, output_size) - w_b_range if bias else 0.0

    def forward(self, input_features: float):
        self.input_features = input_features
        self.out_features = np.dot(self.input_features, self.weights) + self.bias
        return self.out_features

    # here output_error can be dC/dS x dS/dy or dC/dR x dR/dH depend on activation function and layer type
    def backpropagation(self, output_error, learning_rate):

        #input_error is input for previous layer
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot( self.input_features.T, output_error)

        # Update weights and bias using gradient descent
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error