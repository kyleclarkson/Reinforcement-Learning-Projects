import numpy as np

"""
    A Temporal Difference Q Learning 
    approach for CartPole environment.
"""

class Agent():
    def __init__(self,
                 lr,
                 gamma,
                 n_actions,
                 state_space,
                 epsilon_start,
                 epsilon_end,
                 epsilon_dec):

        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_dec = epsilon_dec

        self.n_actions = n_actions
        self.state_space = state_space
        self.action_space = [i for i in range(self.n_actions)]

        self.Q = {}
        self.init_Q()

    def init_Q(self):
        """ Init all Q value pairs to 0. """
        for state in self.state_space:
            for action in self.action_space:
                self.Q[(state, action)] = 0.0

    def max_action(self, state):
        """ Return action corresponding to max Q(s,a). """
        actions = np.array([self.Q[(state, a)] for a in self.action_space])
        action = np.argmax(actions)
        return action

    def choose_action(self, state):
        """ Use epsilon greedy to choose action. """
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.max_action(state)

        return action

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon = self.epsilon - self.epsilon_dec
        else:
            self.epsilon = self.epsilon_end

    def update_Q(self, state, action, reward, state_):
        """ Update Q values using TD update rule. """
        a_max = self.max_action(state_)

        self.Q[(state, action)] = self.Q[(state, action)] + self.lr*\
                                  (reward + self.gamma*self.Q[(state_, a_max)]-self.Q[(state, action)])

