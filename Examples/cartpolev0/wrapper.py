import gym
import numpy as np


class DiscreteObservationSpaceWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_bins, low=None, high=None):
        """
        :param num_bins: The number of bins each state component is divided into.
        :param low: A list containing the starting (smallest) values for each state component.
        :param high: A list containing the starting (smallest) values for each state component.
        """
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)

        self.num_bins = num_bins
        # For each observation, use linspace to split into equal sized bins.
        self.value_bins = [np.linspace(lo, hi, num_bins) for lo, hi in zip(low, high)]
        # The number of possible states using the number of bins.
        self.observation_space.n = gym.spaces.Discrete(num_bins ** len(low))

    def observation(self, observation):
        """
        :param observation:
        :return: A
        """
        return [np.digitize([state_value], bins)[0] for state_value, bins in zip(observation, self.value_bins)]
