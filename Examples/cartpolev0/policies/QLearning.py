import gym
import numpy as np
from collections import defaultdict


class TrainingConfig:
    alpha = 0.1
    epsilon = 1.0
    gamma = 0.9
    num_episodes = 5000


class QLearningPolicy():
    def __init__(self, env, training_config=TrainingConfig):
        """
        :param env: The discrete environment
        :param training_config: Container for training hyperparameters
        """

        # Observation and action spaces must be discrete.
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Discrete)

        self.env = env
        self.Q = defaultdict(float)
        self.tc = training_config

    def update_Q(self, reward, next_state):
        """
        Update all values in Q table using
        Q(s,a) += alpha * (reward + max_a'(Q(next_state, a') - Q(s,a))
        """
        max_q_value = max(self.Q[next_state, a] for a in range(self.env.action_space.n))

        # Update each s,a pair of Q
        for key in self.Q.keys():
            self.Q[key] += self.tc.alpha * (reward + self.tc.gamma*max_q_value - self.Q[key])

    def get_action(self, state, epsilon=0):
        '''
        Decide which action will be taken given the current state
        using epsilon-greedy.
        :param state: The current state.
        :param epsilon: Probability of choosing random action.
        :return: An action that should be taken.
        '''

        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            # Determine action(s) 'a' which maximize Q(state, a).
            q_values = {a: self.Q[state, a] for a in range(self.env.action_space.n)}
            max_q_value = max(q_values.values())

            argmax_actions = [a for a, q in q_values.items() if q == max_q_value]
            return np.random.choice(argmax_actions)
    def train(self):
        pass