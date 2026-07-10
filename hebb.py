import numpy as np


class HebbNetwork:

    def __init__(self):

        self.weights = []
        self.bias = []
        self.labels = []

    # --------------------------------------------------
    # Training using Hebb Learning Rule
    # --------------------------------------------------
    def train(self, patterns):

        self.labels = list(patterns.keys())

        n_inputs = len(next(iter(patterns.values())))

        self.weights = []
        self.bias = []

        for current_label in self.labels:

            weight = np.zeros(n_inputs)
            bias = 0

            for label, pattern in patterns.items():

                if label == current_label:
                    target = 1
                else:
                    target = -1

                # Hebb Learning Rule
                weight = weight + (pattern * target)

                bias = bias + target

            self.weights.append(weight)
            self.bias.append(bias)

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def predict(self, pattern):

        net_values = {}

        for label, weight, bias in zip(
                self.labels,
                self.weights,
                self.bias):

            net = np.dot(weight, pattern) + bias

            net_values[label] = float(net)

        # Best character
        prediction = max(net_values, key=net_values.get)

        # Confidence calculation
        values = np.array(list(net_values.values()))

        if values.max() == values.min():

            confidence = 100.0

        else:

            confidence = (
                (values.max() - values.min())
                /
                (abs(values).max() + 1e-8)
            ) * 100

        return prediction, net_values, round(confidence, 2)

    # --------------------------------------------------
    # Weight Dictionary
    # --------------------------------------------------
    def get_weights(self):

        weights = {}

        for label, weight in zip(
                self.labels,
                self.weights):

            weights[label] = weight

        return weights

    # --------------------------------------------------
    # Bias Dictionary
    # --------------------------------------------------
    def get_bias(self):

        bias = {}

        for label, value in zip(
                self.labels,
                self.bias):

            bias[label] = value

        return bias

    # --------------------------------------------------
    # Activation Function
    # --------------------------------------------------
    def activation(self, value):

        if value >= 0:
            return 1

        return -1

    # --------------------------------------------------
    # Display Weight Matrix
    # --------------------------------------------------
    def weight_matrix(self):

        matrices = {}

        for label, weight in zip(
                self.labels,
                self.weights):

            matrices[label] = weight.reshape(5, 5)

        return matrices
