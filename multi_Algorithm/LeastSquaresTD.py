# Import necessary libraries and functions
import numpy as np
from BatchAgent import BatchAgent

# Implementation of the Linear Least Squares TD batch prediction algorithm
class LeastSquaresTD(BatchAgent):
    def batchUpdate(self):
        A = np.zeros((self.nA * self.nS, self.nA * self.nS))
        b = np.zeros((self.nA * self.nS, 1))
        # -------------------------------------------------------
        # A = np.zeros((2 * self.nS - 1, 2 * self.nS - 1))
        # b = np.zeros((2 * self.nS - 1, 1))
        # -------------------------------------------------------
        # A = np.zeros((2 * self.nS - 2, 2 * self.nS - 2))
        # b = np.zeros((2 * self.nS - 2, 1))
        for di, dn in enumerate(self.sequence):
            # Get data from array
            state, action, reward, state_prime, action_prime, E = dn

            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)
            if state_prime == 4 and action_prime == 0:
                ...
            if state == 4 and action == 0:
                ...

            A_delta = np.matmul(features, (features - self.gamma * features_prime).T)
            A += A_delta

            b_delta = reward * features
            b += b_delta

        # A = A[:-1, :-1]
        # b = b[:-1]
        if np.linalg.det(A) != 0: self.VFA.updateWeightsMatrix(A, b)