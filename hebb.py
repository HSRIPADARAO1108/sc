# hebb.py

import numpy as np


class HebbNetwork:

    def __init__(self):

        self.weights = []
        self.bias = []
        self.labels = []

    # -------------------------------------------------
    # Train the Hebb Network
    # -------------------------------------------------
    def train(self, patterns):

        self.labels = list(patterns.keys())

        size = len(next(iter(patterns.values())))

        self.weights = []
        self.bias = []

        for label in self.labels:

            weight = np.zeros(size)
            bias = 0

            for name, pattern in patterns.items():

                # Target Output
                if name == label:
                    target = 1
                else:
                    target = -1

                # Hebb Learning Rule
                weight = weight + (pattern * target)

                # Bias Update
                bias = bias + target

            self.weights.append(weight)
            self.bias.append(bias)

    # -------------------------------------------------
    # Predict Character
    # -------------------------------------------------
    def predict(self, pattern):

        net_values = {}

        for label, weight, bias in zip(
                self.labels,
                self.weights,
                self.bias):

            net = np.dot(weight, pattern) + bias

            net_values[label] = float(net)

        # Character having highest net value
        prediction = max(net_values, key=net_values.get)

        return prediction, net_values

    # -------------------------------------------------
    # Get Weight Matrix
    # -------------------------------------------------
    def get_weights(self):

        return dict(zip(self.labels, self.weights))

    # -------------------------------------------------
    # Get Bias Values
    # -------------------------------------------------
    def get_bias(self):

        return dict(zip(self.labels, self.bias))
