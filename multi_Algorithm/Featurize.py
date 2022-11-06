import numpy as np

# ------------------------------------------------------------------------------
# 3. Featurize states
# ------------------------------------------------------------------------------

# General object to featurize states:
#   -featureState(state) returns the feature corresponding to state s
#   -featureStateAction(state, action) returns the feature corresponding to the
#       state-action pair (s,a)
class Featurize():
    def set_nSnA(self, nS, nA):
        self.nS = nS
        self.nA = nA

    def set_action_space(self, action_space):
        self.action_space = action_space

    def featureState(self, state):
        return featureTableState(state, self.nS)

    def featureStateAction(self, state, action):
        return featureTableStateAction(state, action, self.nS, self.nA)
        # nS = 5
        # nA = 2 # let's try to delete Sleep state because it introduces zero row that impossibble to calculate matrix determinant
        # feature = np.zeros((nS * nA - 1, 1))
        # feature[state * nA + action] = 1
        # return feature

# Specific implementation of feature functions

# 1. Table lookup
# Function to featurize a state using table lookup
def featureTableState(state, nS):
    feature = np.zeros((nS, 1))
    feature[state] = 1
    return feature

# Function to featurize a state-action pair using table lookup
def featureTableStateAction(state, action, nS, nA):
    feature = np.zeros((nS * nA, 1))
    feature[state * nA + action] = 1
    return feature