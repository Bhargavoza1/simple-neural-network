from module import Module
from linear import Linear
from activation import ReLU,Sigmoid
class MLP(Module):
    # define model elements
    def __init__(self, input_size:int, output_size:int , hiddenlayer:int = 1, h_l_dimension:int = 10):

        self.h_l_dimension = h_l_dimension
        self.hiddenlayer = hiddenlayer
        # input to first hidden layer
        self.hidden1 = Linear(input_size, self.h_l_dimension)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(self.h_l_dimension, self.h_l_dimension)
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(self.h_l_dimension , output_size)
        self.act3 = Sigmoid()

    # forward propagate input
    def forward(self, input_features: float ):
        # input to first hidden layer

        X = self.hidden1.forward(input_features)
        X = self.act1.forward(X)

        for _ in range(self.hiddenlayer):
            # second hidden layer
            X = self.hidden2.forward(X)
            X = self.act2.forward(X)

        # third hidden layer and output
        X = self.hidden3.forward(X)
        X = self.act3.forward(X)
        return X

    def backpropagation(self, output_error: float, learning_rate: float):
        X = self.act3.backpropagation(output_error, learning_rate)
        X = self.hidden3.backpropagation(X, learning_rate)

        for _ in range(self.hiddenlayer):
            # second hidden layer
            X = self.act2.backpropagation(X, learning_rate)
            X = self.hidden2.backpropagation(X, learning_rate)


        X = self.act1.backpropagation(X , learning_rate)
        X = self.hidden1.backpropagation(X , learning_rate)

