


import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, output_size, learning_rate):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        print(x, 'x')
        print(1 / (1 + np.exp(-x)))
        print(max(max(0.1*x), max(x)))
        return [max(0.1*x), max(x)]#1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return 1#self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        return layer_2

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = input_vector.reshape(-1, 1)
        #print(dprediction_dlayer1)
        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias -= self.learning_rate * derror_dbias
        self.weights -= self.learning_rate * derror_dweights

    def train(self, input_vectors, targets, iterations):
        epoch_result = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_vector_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_vector_index]
            target = targets[random_vector_index]
            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(input_vector, target)

            self._update_parameters(derror_dbias, derror_dweights)
            if current_iteration % 10 == 0:
                mapp = np.zeros((100, 100))
                got_right = 0
                greater_than = 0
                for x in range(100):
                    for y in range(100):
                        answer = self.answer([x/100, y/100])
                        mapp[x, y] = 1 if answer[0] > answer[1] else 0
                        if mapp[x, y] == 1 and x/100+y/100 > 0.75:
                            greater_than += 1
                            got_right += 1
                        elif mapp[x, y] == 0 and x/100+y/100 < 0.75:
                            got_right += 1
                epoch_result.append(got_right)
        plt.plot(np.arange(len(epoch_result)), epoch_result)
        plt.show()

    def answer(self, input_vector):
        predictions = self.predict(input_vector)#[self.predict(input_vector) for input_vector in input_vectors] for more elements
        return predictions