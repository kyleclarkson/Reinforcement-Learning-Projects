import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):

    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()

        # Construct network layers
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # Ensure tensor is cuda tensor.
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


class PolicyGradientAgent():

    def __init__(self, lr, input_dims, gamma=0.99, n_actions=4):
        self.gamma = gamma
        self.lr = lr

        self.reward_memory = []
        self.action_memory = []

        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)

    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.policy.device)
        probs = F.softmax(self.policy.forward(state))
        action_probs = T.distributions.Categorical(probs)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        # append to memory.
        self.action_memory.append(log_probs)

        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)


    def learn(self):
        # Zero gradient
        self.policy.optimizer.zero_grad()

        # Compute return
        # G_t = sum k=0 to k=T gamma**k * R_{t+k+1}
        G = np.zeros_like(self.reward_memory, dtype=np.float64)

        # Compute return.
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        # Convert to Pytorch tensor.
        G = T.tensor(G, dtype=T.float).to(self.policy.device)
        # Compute loss
        # Loss = sum over actions taken G_t * \ln(\pi(a |s))
        loss = T.tensor()
        for g, logprob in zip(G, self.action_memory):
            loss += -g * logprob

        # back prob loss through network and apply update.
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory=[]
        self.reward_memory=[]