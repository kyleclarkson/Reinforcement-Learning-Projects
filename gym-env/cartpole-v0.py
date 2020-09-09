import gym
import numpy as np
import matplotlib.pyplot as plt

"""
    A Temporal Difference Q Learning 
    approach for CartPole environment.
"""


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    alpha = 0.05
    gamma = 0.95


    states = np.linespace()

class Agent():

    def __init__(self,
                 lr,
                 gamma,
                 n_actions,
                 state_space,
                 epsil_start,
                 epsil_end,
                 epsil_dec):

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsil_start
        self.epsil_end = epsil_end
        self.epsil_dec = epsil_dec

        self.n_actions = n_actions
        self.state_space = state_space

        self.actions = [i for i in range(self.n_actions)]

        self.Q = {}

        self.init_Q()

    def init_Q(self):
        """ Init all Q value pairs to 0. """
        for state in self.state_space:
            for action in self.actions:
                self.Q[(state, action)] = 0

    def max_action(self, state):
        """ Return action corresponding to max Q(s,a). """
        actions = np.array([self.Q[(state, a)] for a in self.actions])
        action = np.argmax(actions)
        return action

    def choose_action(self, state):
        """ Use epsilon greedy to choose action. """
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.max_action(state)

        return action

    def decrease_epsilon(self):
        if self.epsilon > self.epsil_end:
            self.epsilon -= self.epsil_dec
        else:
            self.epsilon = self.epsil_end

    def update_Q(self, state, action, reward, state_):
        """ Update Q values using TD update rule. """
        a_max = self.max_action(state_)

        self.Q[(state, action)] = self.Q[(state, action)] +\
                                  self.lr*(reward + self.gamma*self.Q[(state, a_max)]-self.Q[(state,action)])





