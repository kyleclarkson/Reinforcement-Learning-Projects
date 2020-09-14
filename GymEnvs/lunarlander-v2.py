import gym
import matplotlib.pyplot as plt
import numpy as np

from LearningMethods.PolicyGradient import PolicyGradientAgent

def plot_learning_curve(scores, x, figure_file):
    """ Plot running averages over 100 episodes and save to file. """

    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])

    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 episodes')
    plt.savefig(figure_file)


if __name__ == '__main__':

    env = gym.make('LunarLander-v2')

    n_games = 3000
    agent = PolicyGradientAgent(gamma=0.99, lr=0.005, input_dims=[8], n_actions=4)
    print("Device: " + str(agent.policy.device))
    fname = 'REINFORCE_' + 'lunar_lander_lr' + str(agent.lr) + '_'\
        + str(n_games) + 'games'

    figure_file = 'plots/'+fname+'.png'

    # Run games
    scores = []
    for i in range(n_games):
        done = False
        obs = env.reset()
        score = 0

        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            score += reward

            agent.store_rewards(reward)
            obs = obs_
        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)
