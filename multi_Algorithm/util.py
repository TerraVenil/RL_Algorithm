# Contains objects and functions required for the following scripts:
#   -agentTabular.py
#   -agentIncrementalVFA.py
#   -agentBatchVFA.py

import numpy as np
import pdb

# ------------------------------------------------------------------------------
# 2. Policies
# ------------------------------------------------------------------------------

# Implements a softmax policy for an agent using linear combination of features
# for value function approximation.
# The policy returns an action given an input state and VFA function.
class SoftmaxPolicyVFA:
    def __init__(self, tau = 1):
        self.tau = tau

    # Intialize the weights vector to a fixed  value
    def setUpWeights(self, dimensions, value = 1):
        self.weights = np.ones(dimensions) * value

    def setNActions(self, nA):
        self.nA = nA

    def getAction(self, featurize, state):
        probabilities = self.computeWeights(featurize, state)

        # Sample action according to the computed probabilities
        return np.random.choice(range(self.nA), p = probabilities)

    # Returns a greedy action
    def greedyAction(self, featurize, state):
        probabilities = self.computeWeights(VFA, featurize, state)

        # Return action with the highest probability
        return np.argmax(probabilities)

    # Compute the probability of sampling each action in a softmax maner
    def computeWeights(self, featurize, state):
        # Compute the feature vector
        values = np.zeros((self.nA, 1))
        for action in range(self.nA):
            feature = featurize.featureStateAction(state, action)
            values[action] = np.dot(feature.T, self.weights)

        # Get the weight of each action
        values_exp = np.exp(values / self.tau - max(values))
        probabilities = (values_exp / sum(values_exp)).flatten()
        #print probabilities
        return probabilities

    # Compute the policy gradient for the state-action pair
    def getGradient(self, featurize, state, action):
        # Compute the feature for every action
        features = featurize.featureStateAction(state, 0) # Array to store the features
        for a in range(1, self.nA): features = np.hstack([features, featurize.featureStateAction(state, a)])
        mean_feature = np.mean(features, 1).reshape(-1,1)  # Mean of the features
        gradient = (features[:, action].reshape(-1, 1) - mean_feature) / self.tau  # Compute gradient
        return gradient

    # Update the parameter theta
    def updateWeightsDelta(self, delta):
        self.weights += delta

# ------------------------------------------------------------------------------
# 5. Models
# ------------------------------------------------------------------------------

# Implementation of a Table Lookup Model as showed by David Silver in
# COMPM050/COMPGI13 Lecture 8, slide 15
class TableLookupModel:
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.N = np.zeros((nS, nA)) # Keep track of the number of times (s,a)
                                    # has appeared
        self.SprimeCounter = np.zeros((nS, nA, nS)) # Number of times (s,a)
                                                    # resulted in s'
        self.Rcounter = np.zeros((nS, nA)) # Total reward obtained by (s,a)
        self.observedStates = [] # states that have appeared before
        self.observedActions = [[] for i in range(nS)] # actions observed before
                                                       # at every state
        self.terminalStates = [] # No knowledge about terminal states assumed

    # Experience is considered as a tuple of (state, action, reward, state_prime)
    def addExperience(self, experience):
        s, a, r, s_prime = experience
        self.N[s][a] += 1
        self.SprimeCounter[s][a][s_prime] += 1
        self.Rcounter[s][a] += r
        if not s in self.observedStates: self.observedStates.append(s)
        if not a in self.observedActions[s]: self.observedActions[s].append(a)

    # Samples the resulting state of (s,a)
    def sampleStatePrime(self, state, action):
        # If there is no information about (s,a), then sample randomly
        if self.N[state][action] == 0: return np.random.choice(range(self.nS))

        prob = self.SprimeCounter[state][action] / self.N[state][action]
        return np.random.choice(range(self.nS), p = prob)

    # Samples the resulting reward of (s,a)
    def sampleReward(self, state, action):
        # If there is no information about (s,a), then return a fixed reward
        if self.N[state][action] == 0: return 0

        return self.Rcounter[state][action] / self.N[state][action]

    # Sample a random state that has been observed before
    def sampleRandState(self):
        return np.random.choice(self.observedStates)

    # Sample a random action previously observed in a given state
    def sampleRandAction(self, state):
        return np.random.choice(self.observedActions[state])

    # Give model knowledge about terminal states
    def addTerminalStates(self, term_states):
        self.terminalStates = term_states

    # Check wether a state is terminal (assuming model has knowledge about
    # terminal states)
    def isTerminal(self, state):
        return state in self.terminalStates
