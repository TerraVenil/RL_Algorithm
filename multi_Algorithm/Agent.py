# ------------------------------------------------------------------------------
# 1. Agent framework
# ------------------------------------------------------------------------------

# Contains an agent's basic functionality, such as to train and benchmark
class Agent:
    def allowlearn(self):
        self.learn = 1

    def preventlearn(self):
        self.learn = 0

    # Trains the agent by computing 'n_episodes' episodes.
    # Returns the average reward per episode.
    def train(self, n_episodes):
        if self.verbosity: print('Training...')
        self.allowlearn()
        return self.runEpisodes(n_episodes) / n_episodes

    # Benchmark the agent by computing 'n_episodes' episodes.
    # Returns the average reward per episode.
    def benchmark(self, n_episodes):
        if self.verbosity: print('benchmarking...')
        self.preventlearn()
        return self.runEpisodes(n_episodes) / n_episodes

    # Computes 'n_episodes' episodes.
    # Returns the average reward per episode.
    def runEpisodes(self, n_episodes):
        accumulatedReward = 0
        for episode_i in range(n_episodes):
            # if self.verbosity: print('Episode ' + str(episode_i))
            accumulatedReward += self.episode()  # Update cumulative reward
        return accumulatedReward