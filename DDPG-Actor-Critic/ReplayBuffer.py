import numpy as np

class ReplayBuffer():

    def __init__(self, mem_size, input_shape, n_actions):
        self.memory_size = mem_size
        self.memory_counter = 0

        self.state_memory = np.zeros((self.memory_size, *input_shape))
        self.new_state_memory = np.zeros((self.memory_size, *input_shape))
        self.action_memory = np.zeros((self.memory_size, n_actions))
        self.reward_memory = np.zeros((self.memory_size))
        # A mask for setting terminal states to zero.
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):

        index = self.memory_counter % self.memory_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_memory = min(self.memory_counter, self.memory_size)

        # Generates random indices from zero to max_memory into list of size batch_size.
        batch = np.random.choice(max_memory, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones
