from module import Module
import numpy as np


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, w_b_range: float = 0.5):
        # Initialize weights with random values between -w_b_range and w_b_range
        self.weights = np.random.rand(in_features, out_features) - w_b_range

        # Initialize bias if bias is True, otherwise set bias to 0.0
        self.bias = np.random.rand(1, out_features) - w_b_range if bias else 0.0

    def forward(self, input_features: int):
        self.input_features = input_features
        self.out_features = np.dot(self.input_features, self.weights) + self.bias
        return self.out_features