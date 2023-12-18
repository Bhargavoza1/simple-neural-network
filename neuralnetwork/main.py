import numpy as np
from mlp import MLP_XOR

x_train = np.array([[1,1], [0,1], [1,0], [0,0]])
y_train = np.array([[0], [1], [1], [0]])

def predict(row, model):

    # make prediction
    Y = model.forward(row)
    return Y


model = MLP_XOR(input_size=2 ,output_size=1)

output = predict(x_train, model)

print(output)