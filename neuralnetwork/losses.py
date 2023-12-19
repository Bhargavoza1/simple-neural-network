import numpy as np

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # small value to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip to avoid log(0) or log(1)
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(loss)


def binary_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-15  # small value to prevent division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip to avoid division by zero
    output = - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)
    return output

def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_derivative(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size
