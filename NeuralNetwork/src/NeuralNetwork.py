import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(
        self,
        input_vector: list = [np.random.randn(), np.random.randn()],
        learning_rate: float = 0.1
    ):
        self.__input_vector = np.array(input_vector)
        self.__weights = np.array([np.random.randn(), np.random.randn()])
        self.__bias = np.random.randn()
        self.__prediction: float = float("inf")
        self.__learning_rate: float = learning_rate

    def __str__(self):
        output: tuple[str] = (
            "Predicted: {}\n".format(self.__prediction),
            "MSE: {}\n".format(self.mse),
            "Learning rate: {}\n".format(self.__learning_rate)
        )
        return ''.join(output)

    @property
    def input_vector(self):
        return self.__input_vector

    @input_vector.setter
    def input_vector(self, new_vector: list):
        self.__input_vector = np.array(new_vector)

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, new_weights: list):
        self.__weights = np.array(new_weights)

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, new_bias: list):
        self.__bias = np.array(new_bias)

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, new_rate):
        self.__learning_rate = new_rate

    @property
    def mse(self):
        """Mean Squared Error.

        Squared distance between predicted and actual values.
        """
        # return np.square(self.__prediction - self.__target)
        return "TODO"

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_deriv(self, x):
        return self.__sigmoid(x) * (1-self.__sigmoid(x))

    def predict(self, input_vector=None):
        if (input_vector is not None):
            layer_1 = np.dot(input_vector, self.__weights) + self.__bias
        else:
            layer_1 = np.dot(self.__input_vector, self.__weights) + self.__bias
        layer_2 = self.__sigmoid(layer_1)
        self.__prediction = layer_2
        return layer_2

    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derivative_error_bias, derivative_weights = self.__compute_gradients(
                input_vector, target
            )

            self.__update_parameters(derivative_error_bias, derivative_weights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors

    def __compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.__weights) + self.__bias
        layer_2 = self.__sigmoid(layer_1)
        self.__prediction = layer_2

        derivative_prediction_error = 2 * (self.__prediction - target)
        derivative_layer_1_prediction = self.__sigmoid_deriv(layer_1)
        layer_1_bias = 1
        layer_1_weights = (0 * self.__weights) + (1 * input_vector)

        derivative_error_bias = (
            derivative_prediction_error * derivative_layer_1_prediction * layer_1_bias
        )

        derivative_error_weights = (
            derivative_prediction_error * derivative_layer_1_prediction * layer_1_weights
        )

        return (derivative_error_bias, derivative_error_weights)

    def __update_parameters(
        self,
        derivative_error_bias,
        derivative_error_weights
    ):
        self.__bias = (
            self.bias - (derivative_error_bias * self.__learning_rate)
        )
        self.__weights = self.__weights - (
            derivative_error_weights * self.learning_rate
        )

    @staticmethod
    def error_plot(training_error):
        plt.plot(training_error)
        plt.xlabel("Iterations")
        plt.ylabel("Error for all training instances")
        plt.savefig("plots/cumulative_error.png")
