import sys
import ast
import numpy as np

from NeuralNetwork.src.NeuralNetwork import NeuralNetwork


class Main:
    @staticmethod
    def main(args: list[str] = []):
        neuralnet = NeuralNetwork()
        if len(args) > 0:
            neuralnet.input_vector = args[1]
            print(neuralnet.predict())
        else:
            # with open("NeuralNetwork/input.txt", 'r') as reader:
            #     lines = reader.readlines()
            # neuralnet.input_vector = ast.literal_eval(lines[0])
            input_vectors = np.array([
                [3, 1.5],
                [2, 1],
                [4, 1.5],
                [3, 4],
                [3.5, 0.5],
                [2, 0.5],
                [5.5, 1],
                [1, 1],
            ])
            targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
            training_error = neuralnet.train(input_vectors, targets, 10000)
            neuralnet.error_plot(training_error)
            print(neuralnet.predict())
            print(neuralnet)


if __name__ == '__main__':
    Main.main(sys.argv)
