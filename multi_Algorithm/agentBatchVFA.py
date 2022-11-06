# Implementation of the following batch methods for value function approximation
# with policy iteration, using linear combination of features and table lookup features:
#   - Least Squares TD(0) [1]
#   - Least Squares TD(lamda) [1]
#   - Least Squares TDQ [2]
#   - Least Squares Policy Iteration TD [3]
# to be used with OpenAI Gym environments. Demonstrations are included with the
# following environments: GridWorld-v0.
#
# The control implementation for this batch methods are not efficient, but rather
# demonstrate their ability to be used for function value evaluation given some
# training experience.
#
# [1] - David Silver (2015), COMPM050/COMPGI13 Lecture 6, slide 45
# [2] - David Silver (2015), COMPM050/COMPGI13 Lecture 6, slide 50
# [3] - David Silver (2015), COMPM050/COMPGI13 Lecture 6, slide 51
#
# By Ricardo Dominguez Olmedo, Aug-2017

# Import necessary libraries and functions
import numpy as np
from Agent import Agent
from BatchAgent import BatchAgent
from Featurize import Featurize
from LinearVFA import LinearVFA
from EGreedyPolicyVFA import EGreedyPolicyVFA
from LeastSquaresTD import LeastSquaresTD
from StudentEnv import StudentEnv
from gridworld import GridworldEnv

# Implementation of the Linear Least Squares TD batch prediction algorithm using
# eligibility traces.
class LSTDlamda(BatchAgent):
    def batchUpdate(self):
        A = np.zeros((self.nA * self.nS, self.nA * self.nS))
        b = np.zeros((self.nA * self.nS, 1))
        for di, dn in enumerate(self.sequence):
            # Get data from array
            state, action, reward, state_prime, action_prime, E = dn

            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_prime)

            A_delta = np.matmul(E, (features - self.gamma * features_prime).T)
            A += A_delta

            b_delta = reward * E
            b += b_delta

        if np.linalg.det(A) != 0: self.VFA.updateWeightsMatrix(A, b)

# Implementation of the Linear Least Squares TDQ batch prediction algorithm
class LSTDQ(BatchAgent):
    def batchUpdate(self):
        A = np.zeros((self.nA * self.nS, self.nA * self.nS))
        b = np.zeros((self.nA * self.nS, 1))
        for di, dn in enumerate(self.sequence):
            # Get data from array
            state, action, reward, state_prime, action_prime, E = dn

            # Compute A' greedily from S'
            action_star = self.policy.greedyAction(self.VFA, self.featurize, state_prime)

            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_star)

            A_delta = np.matmul(features, (features - self.gamma * features_prime).T)
            A += A_delta

            b_delta = reward * features
            b += b_delta

        if np.linalg.det(A) != 0: self.VFA.updateWeightsMatrix(A, b)

# Implementation of the Linear Least Squares Policy Iteration with LSTDQ
# batch evaluation method
class LSPITD(BatchAgent):
    def batchUpdate(self):
        pi_prime = self.policy.getDetArray(self.VFA, self.featurize, self.nS)
        while 1:
            pi = pi_prime
            self.updateWeights()
            pi_prime = self.policy.getDetArray(self.VFA, self.featurize, self.nS)
            if np.array_equal(pi, pi_prime): break

    def updateWeights(self):
        A = np.zeros((self.nA * self.nS, self.nA * self.nS))
        b = np.zeros((self.nA * self.nS, 1))
        for di, dn in enumerate(self.sequence):
            # Get data from array
            state, action, reward, state_prime, action_prime, E = dn

            # Compute A' greedily from S'
            action_star = self.policy.greedyAction(self.VFA, self.featurize, state_prime)

            # Compute the pertinent feature vectors
            features = self.featurize.featureStateAction(state, action)
            features_prime = self.featurize.featureStateAction(state_prime, action_star)

            A_delta = np.matmul(features, (features - self.gamma * features_prime).T)
            A += A_delta

            b_delta = reward * features
            b += b_delta

        if np.linalg.det(A) != 0: self.VFA.updateWeightsMatrix(A, b)

# This function demonstrates how the above methods can be used with OpenAI gym
# environments, while also demonstrating the differences in performance between
# these methods.
def compareMethods():
    import gym
    import matplotlib.pyplot as plt
    # import gym_gridworlds
    # from gridworld_env import GridworldEnv
    # from ... import GridworldEnv
    # from gridworld import GridworldEnv

    # env = gym.make('Gridworld-v0')
    # env = gym.make('gym_gridworlds/GridWorld-v0')
    # env = StudentEnv()
    env = GridworldEnv()
    policy = EGreedyPolicyVFA(0.1)
    VFA = LinearVFA()
    feature = Featurize()

    # training_episodes = 1000
    training_episodes = 1_000
    # n_plot_points = 100
    n_plot_points = 100
    eps_benchmark = 100
    # fixedHorizon = 20
    fixedHorizon = 20

    # Initialize agents
    # alpha1 = 0.4
    alpha1 = 0.1
    agent1 = LeastSquaresTD(env, policy, VFA, feature, alpha1, batchSize=100, horizon = fixedHorizon)

    alpha2 = 0.4
    lamda2 = 0.8
    agent2 = LSTDlamda(env, policy, VFA, feature, alpha2, lamda2, horizon = fixedHorizon)

    alpha3 = 0.4
    agent3 = LSTDQ(env, policy, VFA, feature, alpha3, horizon = fixedHorizon)

    alpha4 = 0.4
    agent4 = LSPITD(env, policy, VFA, feature, alpha4, horizon = fixedHorizon)

    # agents = [agent1, agent2, agent3, agent4]
    agents = [agent1]

    eps_per_point = int(training_episodes / n_plot_points)
    benchmark_data = np.zeros((len(agents), n_plot_points))
    # Benchmark agents without training
    # for agent_i in range(len(agents)):
    #     benchmark_data[agent_i][0] = agents[agent_i].benchmark(eps_benchmark)
    # Train and benchmark agents
    for point_i in range(1, n_plot_points):
        for agent_i in range(len(agents)):
            # print('Agent ' + str(agent_i) + ', Episode ' + str((point_i+1)*eps_per_point))
            agents[agent_i].train(eps_per_point)
            benchmark_data[agent_i][point_i] = agents[agent_i].benchmark(eps_benchmark)

    # Plot results
    plt.figure(figsize=(12, 10))
    xaxis = [eps_per_point*(i+1) for i in range(n_plot_points)]
    title1 = 'LSTD(0), a = ' + str(alpha1)
    title2 = 'LSTD(lamda), a = ' + str(alpha2) + ', l = ' + str(lamda2)
    title3 = 'LSTDQ, a = ' + str(alpha3)
    title4 = 'LSPITD, a = ' + str(alpha4)
    titles = [title1, title2, title3, title4]
    for i in range(len(agents)):
        plt.subplot(221+i)
        plt.plot(xaxis, benchmark_data[i])
        plt.xlabel('Training episodes')
        plt.ylabel('Average reward per episode')
        plt.title(titles[i])
    plt.show()
    print()

compareMethods()
