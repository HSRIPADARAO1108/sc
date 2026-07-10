import numpy as np


class HebbNetwork:
    def __init__(self):
        # Stores weight vector for each character
        self.weights = []
        # Stores bias for each character
        self.bias = []
        # Stores character labels
        self.labels = []

    # -------------------------------------------------
    # Train Hebb Network
    # -------------------------------------------------
    def train(self, patterns):
        # Store labels
        self.labels = list(patterns.keys())
        # Number of inputs (15 for 3x5)
        size = len(next(iter(patterns.values())))
        self.weights = []
        self.bias = []
        # Train one neuron for each character
        for label in self.labels:
            weight = np.zeros(size)
            bias = 0
            for name, pattern in patterns.items():
                # Desired output
                if name == label:
                    target = 1
                else:
                    target = -1
                # Hebb Learning Rule
                weight = weight + (pattern * target)
                # Bias update
                bias = bias + target
            self.weights.append(weight)
            self.bias.append(bias)

    # -------------------------------------------------
    # Predict Character
    # -------------------------------------------------
    def predict(self, pattern):
        net_values = {}
        # Calculate net value for every character
        for label, weight, bias in zip(
                self.labels,
                self.weights,
                self.bias):
            net = np.dot(weight, pattern) + bias
            net_values[label] = float(net)
        # Character having maximum net value
        prediction = max(net_values, key=net_values.get)
        return prediction, net_values

    # -------------------------------------------------
    # Return Weight Matrix
    # -------------------------------------------------
    def get_weights(self):
        weights = {}
        for label, weight in zip(self.labels, self.weights):
            weights[label] = weight
        return weights

    # -------------------------------------------------
    # Return Bias
    # -------------------------------------------------
    def get_bias(self):
        bias_values = {}
        for label, bias in zip(self.labels, self.bias):
            bias_values[label] = bias
        return bias_values
