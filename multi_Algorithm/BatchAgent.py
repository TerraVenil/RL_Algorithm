# Import necessary libraries and functions
import numpy as np
from Agent import Agent
from Featurize import Featurize
from LinearVFA import LinearVFA

# Implements the specific functionality of a batch value function approximation
# agent, such as to initialize the agent or run episodes.
class BatchAgent(Agent):
    def __init__(self, env, policy, VFA: LinearVFA, featurize: Featurize, alpha, batchSize = 100,
        lamda = 0, gamma = 1, eps = 1, horizon = 1000, verbosity = 0):
        # Inputs:
        #   -env: openAI gym environment object
        #   -policy: object containing a policy from which to sample actions
        #   -VFA: object containing the value function approximator
        #   -featurize: object which featurizes states
        #   -alpha: step size parameter
        #   -batchSize: number of episodes of experience before policy evaluation
        #   -lamda: trace discount paramater
        #   -gamma: discount-rate parameter
        #   -eps: minimum difference in a weight update for methods that require
        #       convergence
        #   -horizon: finite horizon steps
        #   -verbosity: if TRUE, prints to screen additional information

        self.env = env
        self.policy = policy
        self.featurize = featurize
        self.VFA = VFA
        self.alpha = alpha
        self.batchSize = batchSize
        self.lamda = lamda
        self.gamma = gamma
        self.eps = eps
        self.horizon = horizon
        self.verbosity = verbosity

        self.nS = env.observation_space.n   # Number of states
        self.nA = env.action_space.n    # Number of actions
        # self.nA = 0
        # for space in env.action_space:
        #     self.nA += space.n

        self.policy.setNActions(self.nA)
        # self.policy.set_action_space(env.action_space)
        self.featurize.set_nSnA(self.nS, self.nA)
        self.featDim = featurize.featureStateAction(0,0).shape # Dimensions of the
                                                               # feature vector
        self.VFA.setUpWeights(self.featDim) # Initialize weights for the VFA
        self.learn = 0 # Initially prevent agent from learning

        self.batch_i = 0 # To keep track of the number of stored experience episodes
        self.sequence =  [] # Array to store episode sequences

    def setUpTrace(self):
        self.E = np.zeros(self.featDim)

    # Computes a single episode.
    # Returns the episode reward return.
    def episode(self):
        episodeReward = 0
        self.setUpTrace()

        # Initialize S, A
        state = self.env.reset()
        # if state == 4:
        #     episodeReward += 0
        #     return episodeReward
        # else:
        action = self.policy.getAction(self.VFA, self.featurize, state)

        # Repeat for each episode
        for t in range(self.horizon):
            # Take action A, observe R, S'
            state, action, reward, done = self.step(state, action)

            # Update the total episode return
            episodeReward += reward

            # Finish the loop if S' is a terminal state
            if done: break

        # Update the policy if the agent is learning and the amount of required
        # experience is met.
        if self.learn:
             self.batch_i += 1
             if (self.batch_i+1) % self.batchSize == 0: self.batchUpdate()
        # self.batchUpdate()

        # self.policy.episodeUpdate()

        return episodeReward

    def step(self, state, action):
        # Take A, observe R and S'
        state_prime, reward, done, info = self.env.step(action)

        # Choose A' using a policy derived from S'
        # if state_prime == 4:
        #     action_prime = 0
        # else:
        action_prime = self.policy.getAction(self.VFA, self.featurize, state_prime)

        # Store experience
        if self.learn:
            # If traces are being used, update them
            if self.lamda != 0:
                features = self.featurize.featureStateAction(state, action)
                self.E = (self.gamma * self.lamda * self.E) + self.VFA.getGradient(features)

            # Store experience
            self.sequence.append((state, action, reward, state_prime, action_prime, self.E))

        return state_prime, action_prime, reward, done