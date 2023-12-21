import numpy as np
from mlp import MLP_XOR
from losses import mse, mse_derivative

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[1,0]], [[0,1]], [[0,1]], [[1,0]]])

def predict_and_print_results( model , input_data, target_data):
    num_samples = len(input_data)

    for sample_index in range(num_samples):
        input_sample = input_data[sample_index]
        predicted_output = model.forward(input_sample)

        actual_output = 0 if target_data[sample_index][0][0] == 1 else 1

        print(f'{predicted_output[0] }')

        if predicted_output[0][0] >= 0.5:
            print(f'1st XOR value: {input_sample[0][0]} 2nd XOR value: {input_sample[0][1]} '
                  f'Neural Network output is 0, actual output is: {actual_output}')
        else:
            print(f'1st XOR value: {input_sample[0][0]} 2nd XOR value: {input_sample[0][1]} '
                  f'Neural Network output is 1, actual output is: {actual_output}')


def train_neural_network(model, input_data, target_data, num_epochs, learning_rate):
    # Get the number of samples
    num_samples = len(input_data)

    # Training loop
    for epoch in range(num_epochs):
        total_error = 0

        for sample_index in range(num_samples):
            # Forward propagation
            input_sample = input_data[sample_index]
            predicted_output = model.forward(input_sample)

            # Compute loss (for display purposes only)
            total_error += mse(target_data[sample_index], predicted_output)

            # Backward propagation
            output_error = mse_derivative(target_data[sample_index], predicted_output)
            model.backpropagation(output_error, learning_rate)

        # Calculate the average error on all samples
        average_error = total_error / num_samples
        print(f'Epoch {epoch + 1}/{num_epochs}   Mean Squared Error: {average_error:.6f}')




model = MLP_XOR(input_size=2 ,output_size=2)
train_neural_network(model , x_train, y_train, num_epochs=2000, learning_rate=0.3)
predict_and_print_results(model , x_train, y_train )
