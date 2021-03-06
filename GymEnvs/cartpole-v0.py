import gym
import numpy as np
import matplotlib.pyplot as plt

from LearningMethods.td_0 import Agent

"""
A Class to make the state space from continuous to discrete.
"""
class CartPoleStateDigitizer():
    def __init__(self, bounds=(2.4, 4, 0.209, 4), n_bins=16):
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


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    n_eps = 80_000
    epsil_dec = 2.5 / n_eps

    digitizer = CartPoleStateDigitizer()
    agent = Agent(lr=0.01, gamma=0.99, n_actions=2,
                  epsilon_start=1.0,
                  epsilon_end=0.01,
                  epsilon_dec=epsil_dec,
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

        if i % 5000 == 0:
            print('episode ', i, ' score %.1f' % score, ' epsilon %.3f' % agent.epsilon)

        agent.decrease_epsilon()
        scores.append(score)


    # Plot average rewards.
    x = [i+1 for i in range(n_eps)]
    plot_learning_curve(scores, x)

    env.close()

