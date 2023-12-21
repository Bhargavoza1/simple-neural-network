from module import Module
import numpy as np

class ReLU(Module):
    def __init__(self):
        # Constructor, but it's empty in this case
        pass
    def relu_derivative(self, x):
        # Subgradient of the ReLU activation function
        return np.where(x > 0, 1, 0)

    def forward(self, input_features):
        # Forward pass of the ReLU activation function
        # Save the input features for potential use in backward pass
        self.input_features = input_features

        # Apply ReLU activation element-wise
        self.activation = np.maximum(0, self.input_features)

        # Return the result of the activation
        return self.activation

    def backpropagation(self, output_error: float, learning_rate: float):
        return self.relu_derivative(self.input_features) * output_error

class Sigmoid(Module):
    def __init__(self):
        # Constructor, but it's empty in this case
        pass

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of the sigmoid activation function
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)

    def forward(self, input_features):
        # Forward pass of the sigmoid activation function
        # Save the input features for potential use in the backward pass
        self.input_features = input_features

        # Apply the sigmoid activation element-wise
        self.activation = self.sigmoid(self.input_features)

        # Return the result of the activation
        return self.activation

    def backpropagation(self, output_error: float, learning_rate: float):
        return self.sigmoid_derivative(self.input_features) * output_error


class Tanh(Module):
    def __init__(self):
      pass

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        # Derivative of the tanh activation function
        return 1 - np.tanh(x) ** 2

    def forward(self, input_features):
        # Forward pass of the tanh activation function
        # Save the input features for potential use in the backward pass
        self.input_features = input_features

        # Apply the tanh activation element-wise
        self.activation = self.tanh(self.input_features)

        # Return the result of the activation
        return self.activation

    def backpropagation(self, output_error: float, learning_rate: float):
        return self.tanh_derivative(self.input_features) * output_error

class BinaryStep(Module):
    def __init__(self):
        pass

    def binary_step(self, x):
        return 1 if x >= 0 else 0

    def forward(self, input_features):
        # Forward pass of the binary step activation function

        # Save the input features for potential use in the backward pass
        self.input_features = input_features
        # Apply the binary step activation element-wise using np.vectorize
        self.activation = np.vectorize(self.binary_step)(self.input_features)

        # Return the result of the activation
        return self.activation

    def backpropagation(self, output_error: float, learning_rate: float):
        # derivative of binary step function is not possible
        return 0