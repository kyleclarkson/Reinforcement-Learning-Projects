import gym
import numpy as np
import random

class TrainingConfig():

    def __init__(self):
        self.num_of_episodes = 1_000_000
        self.display_rewards_every = self.num_of_episodes // (self.num_of_episodes // 10)
        self.display_training_iterations = self.num_of_episodes // 10
        self.max_steps_per_episode = 100

        self.learning_rate = 0.1
        self.discount_rate = 0.95

        self.explore_rate = 1
        self.max_explore_rate = 1
        self.min_explore_rate = 0.01
        self.explore_decay_rate = 0.005

    def decay_explore_rate(self, current_episode):
        self.explore_rate = (self.max_explore_rate - self.min_explore_rate) \
                            * np.exp(-self.explore_decay_rate*current_episode) + self.min_explore_rate


class BaseLearningMethod():
    pass


class QLearning(BaseLearningMethod):
    def __init__(self, env, training_config):
        # An environment that has OpenAi Gym structure.
        self.env = env
        # size: num of states x num of actions.
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.tc = training_config

        self.rewards_all_episodes = []

    def learn(self):
        """
        Use tc parameters to learning Q values for environment.
        """

        # maintain list of rewards for all episodes
        self.rewards_all_episodes = []

        for episode in range(self.tc.num_of_episodes):
            state = self.env.reset()

            done = False
            rewards_per_ep = 0

            if (episode + 1) % self.tc.display_training_iterations == 0:
                print("Episode: ", episode + 1)

            for step in range(self.tc.max_steps_per_episode):
                # Decide action using epsilon-greedy
                if random.uniform(0, 1) > self.tc.explore_rate:
                    action = np.argmax(self.q_table[state, :])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, done, info = self.env.step(action)

                self.update_q(state, action, new_state, reward)

                # Update state, rewards
                state = new_state
                rewards_per_ep += reward

                if done:
                    break

            # Decay explore rate.
            self.tc.decay_explore_rate(episode)
            # Update episode rewards
            self.rewards_all_episodes.append(rewards_per_ep)

    def update_q(self, state, action, new_state, reward):
        """
        Update Q table using Q(s,a) += learning_rate(r+discount*Q(s',:) - Q(s,a) (typed differently)
        :param state: current state
        :param action: action take
        :param next_state: new state
        :param reward: reward amount
        :return: ---
        """
        self.q_table[state, action] = self.q_table[state, action]*(1-self.tc.learning_rate) \
            + self.tc.learning_rate*(reward + self.tc.discount_rate*np.max(self.q_table[new_state, :]))

    def ave_reward_over_window(self):
        """
        Compute average reward over a window of training episodes.
        :return: List of average rewards
        """
        if len(self.rewards_all_episodes) == 0:
            return []
        send = []
        for r in np.split(np.array(self.rewards_all_episodes), 10):
            # Display average reward for this time window.
            send.append(np.sum(r/ len(r)))
        return send

        # count = tc.num_of_episodes / 10
        # for r in np.split(np.array(rewards_all_episodes), 10):
        #     # Display average reward for this time window.
        #     print(count, ": ", str(np.sum(r / 100)))
        #     count += tc.num_of_episodes / 10

    def save_q_table(self, name):
        np.save(name+".npy", self.q_table)

    def load_q_table(self, name):
        return np.load(name+".npy")


if __name__ == '__main__':
    # Create gym environment
    env = gym.make("FrozenLake-v0")
    tc = TrainingConfig()
    model = QLearning(env, tc)

    model.save_q_table("saved_models/before")
    print("Learning")
    model.learn()
    ave_rewards = model.ave_reward_over_window()

    for r in ave_rewards:
        print(r)

    model.save_q_table("saved_models/after")