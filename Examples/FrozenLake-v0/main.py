import gym
import numpy as np
import random

class TrainingConfig():

    def __init__(self):
        self.num_of_episodes = 1000
        self.display_rewards_every = self.num_of_episodes // (self.num_of_episodes // 10)
        self.max_steps_per_episode = 100

        self.learning_rate = 0.1
        self.discount_rate = 0.95

        self.explore_rate = 1
        self.max_explore_rate = 1
        self.min_explore_rate = 0.01
        self.explore_decay_rate = 0.001

    def decay_explore_rate(self, current_episode):
        self.explore_rate = (self.max_explore_rate - self.min_explore_rate) \
                            * np.exp(-self.explore_decay_rate*current_episode) + self.min_explore_rate


class BaseLearningMethod():
    pass


class QLearning(BaseLearningMethod):
    def __init__(self, num_of_states, num_of_actions, training_config):
        self.q_table = np.zeros((num_of_states, num_of_actions))
        self.tc = training_config

    def update_q(self, state, action, new_state, reward):
        '''
        Update Q table using Q(s,a) += learning_rate(r+discount*Q(s',:) - Q(s,a) (typed differently)
        :param state: current state
        :param action: action take
        :param next_state: new state
        :param reward: reward amount
        :return: ---
        '''
        self.q_table[state, action] = self.q_table[state, action]*(1-self.tc.learning_rate) \
            + self.tc.learning_rate*(reward + self.tc.discount_rate*np.max(self.q_table[new_state, :]))


if __name__ == '__main__':
    # Create gym environment
    env = gym.make("FrozenLake-v0")

    tc = TrainingConfig()
    train = QLearning(env.observation_space.n, env.action_space.n, tc)

    # Maintain reward over all episodes.
    rewards_all_episodes = []

    for episode in range(tc.num_of_episodes):
        state = env.reset()

        done = False
        rewards_per_ep = 0

        if (episode+1) % 10_000 == 0:
            print("Episode: ", episode+1)

        for step in range(tc.max_steps_per_episode):
            # Decide action using epsilon-greedy
            if random.uniform(0, 1) > tc.explore_rate:
                action = np.argmax(train.q_table[state, :])
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)

            train.update_q(state, action, new_state, reward)

            # Update state, rewards
            state = new_state
            rewards_per_ep += reward

            if done:
                break

        # Decay explore rate.
        tc.decay_explore_rate(episode)
        # Update episode rewards
        rewards_all_episodes.append(rewards_per_ep)

    # Display average reward.

    print(tc.display_rewards_every)
    display_rewards = np.split(np.array(rewards_all_episodes), 10)
    count = 100
    for r in display_rewards:
        print(count, ": ", str(np.sum(r/100)))
        count += 100

