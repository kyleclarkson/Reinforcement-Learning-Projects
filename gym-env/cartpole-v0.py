import gym
import numpy as np
import matplotlib.pyplot as plt

"""
    A Temporal Difference Q Learning 
    approach for CartPole environment.
"""

class CartPoleStateDigitizer():
    def __init__(self, bounds=(2.4, 4, 0.209, 4), n_bins=10):
        self.position_space = np.linspace(-1*bounds[0], bounds[0], n_bins)
        self.velocity_space = np.linspace(-1*bounds[1], bounds[1], n_bins)
        self.pole_angle_space = np.linspace(-1*bounds[2], bounds[2], n_bins)
        self.pole_velocity_space = np.linspace(-1*bounds[3], bounds[3], n_bins)

        self.states = self.get_state_space()

    def get_state_space(self):
        states = []
        for i in range(len(self.position_space)+1):
            for j in range(len(self.velocity_space)+1):
                for k in range(len(self.pole_angle_space)+1):
                    for l in range(len(self.pole_velocity_space)+1):
                        states.append((i, j, k, l))
        return states

    def digitize(self, observation):
        x, x_dot, theta, theta_dot = observation
        cart_x = int(np.digitize(x, self.position_space))
        cart_x_dot = int(np.digitize(x_dot, self.velocity_space))
        pole_theta = int(np.digitize(theta, self.pole_angle_space))
        pole_theta_dot = int(np.digitize(theta_dot, self.pole_velocity_space))

        return (cart_x, cart_x_dot, pole_theta, pole_theta_dot)


def plot_learning_curve(scores, x):
    """ Plot running average over 100 episodes. """
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])

    plt.plot(x, running_avg)
    plt.title("Running Average of Previous 100 Scores")
    plt.show()


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
        self.action_space = [i for i in range(self.n_actions)]

        self.Q = {}
        self.init_Q()

    def init_Q(self):
        """ Init all Q value pairs to 0. """
        for state in self.state_space:
            for action in self.action_space:
                self.Q[(state, action)] = 0

    def max_action(self, state):
        """ Return action corresponding to max Q(s,a). """
        actions = np.array(self.Q[(state, a)] for a in self.action_space)
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
        if self.epsilon > self.epsil_end:
            self.epsilon = self.epsilon - self.epsil_dec
        else:
            self.epsilon = self.epsil_end

    def update_Q(self, state, action, reward, state_):
        """ Update Q values using TD update rule. """
        a_max = self.max_action(state_)

        self.Q[(state, action)] = self.Q[(state, action)] + self.lr*\
                                  (reward + self.gamma*self.Q[(state, a_max)]-self.Q[(state, action)])


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    n_eps = 30_000
    epsil_dec = 2 / n_eps

    digitizer = CartPoleStateDigitizer()
    agent = Agent(lr=0.01, gamma=0.99, n_actions=2,
                  epsil_start=1.0, epsil_end=0.01,
                  epsil_dec=epsil_dec,
                  state_space=digitizer.states)

    scores = []

    for i in range(n_eps):
        obs = env.reset()
        done = False

        score = 0
        state = digitizer.digitize(obs)
        while not done:
            action = agent.choose_action(state)
            obs_, reward, done, info = env.step(action)
            state_ = digitizer.digitize(obs_)
            agent.update_Q(state, action, reward, state_)
            state = state_
            score += reward

        if i % 2500 == 0:
            print('episode ', i, ' score %.1f' % score, ' epsilon %.3f' % agent.epsilon)

        agent.decrease_epsilon()
        scores.append(score)


    # Plot average rewards.
    x = [i+1 for i in range(n_eps)]
    plot_learning_curve(scores, x)

    env.close()

