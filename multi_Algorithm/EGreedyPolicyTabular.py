import numpy as np

# ------------------------------------------------------------------------------
# 2. Policies
# ------------------------------------------------------------------------------

# Implements an e-greedy policy for a tabular agent.
# The policy returns an action given an input state and state-action value function.
# The epsilon of the policy decays according to the parameter 'decay'
class EGreedyPolicyTabular:
    def __init__(self, epsilon, decay = 1):
        self.epsilon = epsilon
        self.decay = decay

    def getAction(self, Q, state):
        # Q(s, a) should be addressable as Q[s][a]

        if np.random.random() > self.epsilon:
            # Take greedy action
            return self.greedyAction(Q, state)
        # Take an exploratory action
        else: return self.randomAction(Q)

    # Returns a random action
    def randomAction(self, Q):
        nA = Q[0].shape[0]
        return np.random.randint(nA)

    # Returns a greedy action
    def greedyAction(self, Q, state):
        nA = Q[0].shape[0]
        maxima_index = [] # Actions with maximum value
        maxVal = None # Value of the current best actions

        for action in range(nA):
            value = Q[state][action] # Get the value from the state-action value function.
            if maxVal == None: # For the fist (s,a), intialize 'maxVal'
                maxVal = value
            if value > maxVal: # If the action is better than previus ones, update
                maxima_index = [action]
                maxVal = value
            elif value == maxVal: # If the action is equally good, add it
                maxima_index.append(action)

        # Randomly choose one of the best actions
        return np.random.choice(maxima_index)

    def epsilonDecay(self):
        self.epsilon *= self.decay

    # The policy update consists only on epsilon decay
    def episodeUpdate(self):
        self.epsilonDecay()