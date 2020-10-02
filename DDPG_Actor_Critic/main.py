import torch as T
import torch.nn.functional as F
import numpy as np

from DDPG_Actor_Critic.ReplayBuffer import ReplayBuffer
from DDPG_Actor_Critic.Networks import ActorNetwork, CriticNetwork
from DDPG_Actor_Critic.OUNoise import QUNoise

import gym
import matplotlib.pyplot as plt

class Agent():

    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99, max_size=1_000_000,
                 fc1_dims=400, fc2_dims=300, batch_size=64):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = QUNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions, name='actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions, name='target_actor')
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, name='target_critic')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):

        self.actor.eval()
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)
        # Add noise for exploration.
        # print('mu: ', mu.shape)
        # print('mu_: ', T.tensor(self.noise(), dtype=T.float).to(self.actor.device).shape)
        mu_ = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)

        self.actor.train()

        # Return action as numpy array.
        return mu_.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()


    def learn(self):
        # if buffer is not full, let agent play
        if self.memory.memory_counter < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        dones = T.tensor(dones, dtype=T.bool).to(self.actor.device)

        # feed to actor network
        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)

        # critic value for actions taken by agent
        critic_value = self.critic.forward(states, actions)

        # set any new states that are terminal to 0.
        critic_value_[dones] = 0.0

        # collapse critic_value_ dimension
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        # compute critic loss
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        # compute actor loss
        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        # update network parameters
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        # Update critic parameters using current parameters
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        # convert to dicts
        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_state_dict =  dict(target_actor_params)
        target_critic_state_dict = dict(target_critic_params)

        # Iterate over keys of dict, use update rule to set.
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() +\
                                      (1-tau)*target_actor_state_dict[name].clone()
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() +\
                                      (1-tau)*target_critic_state_dict[name].clone()

        # set parameters via load from dict function.
        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

if __name__ == '__main__':

    env = gym.make('LunarLanderContinuous-v2')

    fc1_dims = 128
    fc2_dims = 64
    agent = Agent(0.001, beta=0.001, input_dims=env.observation_space.shape, tau=0.001,
                  batch_size=64, fc1_dims=fc1_dims, fc2_dims=fc2_dims, n_actions=env.action_space.shape[0])

    n_games = 1500
    filename = f'LunarLander_alpha_{str(agent.alpha)}_beta_{str(agent.beta)}_games_' \
               f'{str(n_games)}_shape_{str(fc1_dims)}-{str(fc2_dims)}'
    figure_file = 'plots/' + filename + '.png'
    print("device: ", agent.actor.device)
    print("filename: ", filename)

    best_score = env.reward_range[0]
    score_history = []

    for i in range(1, n_games+1):

        obs = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, obs_, done)
            agent.learn()
            score += reward
            obs = obs_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score and i > 800:
            best_score = avg_score
            agent.save_models()

        print(f'episode: {i}, score: {score:.2f}, average score: {avg_score:.2f}')

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)