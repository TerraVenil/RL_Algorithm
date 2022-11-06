import numpy as np

# ------------------------------------------------------------------------------
# 2. Policies
# ------------------------------------------------------------------------------

# Implements an e-greedy policy for an agent using value function approximation.
# The policy returns an action given an input state and VFA function.
# The epsilon of the policy decays according to the parameter 'decay'
class EGreedyPolicyVFA:
    def __init__(self, epsilon, decay = 1):
        self.epsilon = epsilon
        self.decay = decay

    def setNActions(self, nA):
        self.nA = nA

    def set_action_space(self, action_space):
        self.action_space = action_space

    def getAction(self, VFA, featurize, state):
        # VFA is the value function approximator
        if np.random.random() > self.epsilon:
            # Take a greedy action
            return self.greedyAction(VFA, featurize, state)
        # Take an exploratory action
        else: return self.randomAction(state)

    # Returns a random action
    def randomAction(self, state):
        # return np.random.randint(self.nA)
        return np.random.randint(self.action_space.spaces[state].n)

    # Returns a greedy action
    def greedyAction(self, VFA, featurize, state):
        maxima_index = [] # Actions with maximum value
        maxVal = None # Value of the current best actions

        for action in range(self.nA):
        # for action in range(self.action_space.spaces[state].n):
             # Get the value of the state action pair from VFA
            features = featurize.featureStateAction(state, action)
            value = VFA.getValue(features)

            if maxVal is None: # For the fist (s,a), intialize 'maxVal'
                maxVal = value
            if value > maxVal: # If the action is better than previus ones, update
                maxima_index = [action]
                maxVal = value
            elif value == maxVal: # If the action is equally good, add it
                maxima_index.append(action)

        # Randomly choose one of the best actions
        return np.random.choice(maxima_index)

    # Returns an array containing the action with the highest value for every state
    def getDetArray(self, VFA, featurize, nS):
        detActions = np.zeros((nS, 1))
        actionVals = np.zeros((self.nA, 1)) # Stores the values for all actions
                                            # in a given state
        for state in range(nS):
            for action in range(self.nA):
                features = featurize.featureStateAction(state, action)
                actionVals[action] = VFA.getValue(features)
            detActions[state] = np.argmax(actionVals) # Choose highest value
        return detActions

    def epsilonDecay(self):
        self.epsilon *= self.decay

    # The policy update consists only on epsilon decay
    def episodeUpdate(self):
        self.epsilonDecay()