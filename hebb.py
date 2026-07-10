import numpy as np


class HebbNetwork:

    def __init__(self):

        self.weights = {}
        self.bias = {}
        self.labels = []


    # ------------------------------------------
    # Training
    # ------------------------------------------

    def train(self, patterns):

        self.labels = list(patterns.keys())

        self.weights = {}
        self.bias = {}


        for label, pattern in patterns.items():

            pattern = np.array(
                pattern
            )


            # Store character prototype

            self.weights[label] = pattern.copy()


            # Bias

            self.bias[label] = 0



    # ------------------------------------------
    # Prediction
    # ------------------------------------------

    def predict(self, input_pattern):


        input_pattern = np.array(
            input_pattern
        )


        net_values = {}



        for label, weight in self.weights.items():


            # Similarity score

            score = np.dot(
                weight,
                input_pattern
            )


            net_values[label] = float(
                score
            )



        prediction = max(
            net_values,
            key=net_values.get
        )



        values = np.array(
            list(net_values.values())
        )


        if abs(values).max()==0:

            confidence = 0

        else:

            confidence = (
                (values.max()-values.min())
                /
                abs(values).max()
            )*100



        return (
            prediction,
            net_values,
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

        result={}


        for label,weight in self.weights.items():

            result[label]=np.array(weight).reshape(
                5,
                3
            )


        return result
