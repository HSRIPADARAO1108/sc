# hebb.py

import numpy as np


class HebbNetwork:

    def __init__(self):

        self.weights = None
        self.bias = 0
        self.labels = []

    # -----------------------------
    # Training
    # -----------------------------
    def train(self, patterns):

        self.labels = list(patterns.keys())

        n = len(self.labels)

        size = len(next(iter(patterns.values())))

        self.weights = []

        self.bias = []

        for label in self.labels:

            w = np.zeros(size)

            b = 0

            for name, pattern in patterns.items():

                if name == label:
                    target = 1
                else:
                    target = -1

                w = w + pattern * target

                b = b + target

            self.weights.append(w)

            self.bias.append(b)

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, pattern):

        nets = []

        for w, b in zip(self.weights, self.bias):

            net = np.dot(w, pattern) + b

            nets.append(net)

        index = np.argmax(nets)

        return self.labels[index], nets[index]
