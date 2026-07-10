import numpy as np


class HebbNetwork:

    def __init__(self):

        self.weights = {}
        self.bias = {}
        self.labels = []


    # ------------------------------------------
    # Hebb Learning Rule Training
    # ------------------------------------------

    def train(self, patterns):

        self.labels = list(patterns.keys())

        self.weights = {}
        self.bias = {}


        for label, pattern in patterns.items():

            pattern = np.array(pattern)


            # Store pattern directly as prototype weight

            self.weights[label] = pattern.copy()


            # Bias using pattern size

            self.bias[label] = 0



    # ------------------------------------------
    # Prediction
    # ------------------------------------------

    def predict(self, input_pattern):


        input_pattern = np.array(input_pattern)


        scores = {}



        for label, weight in self.weights.items():


            # Similarity calculation

            score = np.sum(
                input_pattern * weight
            )


            scores[label] = float(score)



        prediction = max(
            scores,
            key=scores.get
        )



        values = np.array(
            list(scores.values())
        )


        confidence = (
            (values.max()-values.min())
            /
            (abs(values).max()+1e-8)
        )*100



        return (
            prediction,
            scores,
            round(confidence,2)
        )



    # ------------------------------------------
    # Get Weights
    # ------------------------------------------

    def get_weights(self):

        return self.weights



    # ------------------------------------------
    # Get Bias
    # ------------------------------------------

    def get_bias(self):

        return self.bias



    # ------------------------------------------
    # Weight Matrix
    # ------------------------------------------

    def weight_matrix(self):

        matrices={}


        for label,weight in self.weights.items():

            matrices[label]=weight.reshape(
                5,
                3
            )


        return matrices
