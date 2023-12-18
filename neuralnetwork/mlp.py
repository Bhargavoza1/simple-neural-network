from module import Module
from linear import Linear
from activation import ReLU,Sigmoid, Tanh
class MLP(Module):
    # define model elements
    def __init__(self, input_size:float, output_size:float):

        # input to first hidden layer
        self.hidden1 = Linear(input_size, 3)
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(3, 2)
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(2, output_size)
        self.act3 = Tanh()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1.forward(X)
        X = self.act1.forward(X)
        # second hidden layer
        X = self.hidden2.forward(X)
        X = self.act2.forward(X)
        # third hidden layer and output
        X = self.hidden3.forward(X)
        X = self.act3.forward(X)
        return X
