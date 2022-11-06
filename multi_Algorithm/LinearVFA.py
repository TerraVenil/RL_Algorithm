import numpy as np

# ------------------------------------------------------------------------------
# 4. Value function approximation
# ------------------------------------------------------------------------------

# Implementation of linear value function approximation through linear
# combination of features
class LinearVFA:
    # Intialize the weights vector to a fixed  value
    def setUpWeights(self, dimensions, value = 1):
        self.weights = np.ones(dimensions) * value

    def returnWeights(self, dimensions, value = 1):
        return np.ones(dimensions) * value

    def getValue(self, features):
        return np.dot(features.T, self.weights)
        # if features.shape == (8, 1) and self.weights.shape == (8, 1):
        #     return np.dot(features.T, self.weights)
        # elif self.weights.shape == (8, 1):
        #     return np.dot(features[:-1].T, self.weights)
        # else:
        #     return np.dot(features.T, self.weights)

    def getGradient(self, features):
        return features

    def updateWeightsDelta(self, delta_weight):
        self.weights += delta_weight

    def updateWeightsMatrix(self, A, b):
        self.weights = np.matmul(np.linalg.inv(A), b)

    def getWeights(self):
        return self.weights

    def setWeights(self, weights):
        self.weights = weights