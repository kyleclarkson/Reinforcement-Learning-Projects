"""
Follow along code for ML Phil's Gridworld example

Agent is attempting to move to bottom-right cell of a grid
while avoid

"""

import numpy as np
import matplotlib.pyplot as plt

class GridWorld(object):

    def __init__(self, m, n, magicSquares):
        self.grid = np.zeros((m,n))
        self.m = m
        self.n = n
        # A list of non-terminal states.
        self.stateSpace = [i for i in range(self.m*self. n)]
        self.stateSpace.remove(self.m*self.n - 1)
        # A list of all states.
        self.stateSpacePlus = [i for i in range(self.m*self.n)]

        # Define mapping of how action will change state.
        self.actionSpace = {'U': -self.m,
                            'D': self.m,
                            'L': -1,
                            'R': 1}

        self.possibleActions = ['U', 'D', 'L', 'R']
        # A dict containing magic squares.
        self.addMagicSquares(magicSquares)
        # Initial position is top-left of grid
        self.agentPosition = 0

    def addMagicSquares(self, magicSquares):
        self.magicSquares = magicSquares
        # only two magic squares.
        i = 2
        for square in magicSquares:
            x = square // self.m
            y = square % self.n
            self.grid[x][y] = i
            i += 1
            x = magicSquares[square] // self.m
            y = magicSquares[square] % self.n
            self.grid[x][y] = i
            i += 1

    def isTerminalState(self, state):
        return state in self.stateSpacePlus and state not in self.stateSpace

    def getAgentRowAndCol(self):
        return self.agentPosition // self.m, self.agentPosition % self.n

    def setState(self, state):
        # 0/1 indicates if agent is here/ not here.
        x, y = self.getAgentRowAndCol()
        self.grid[x][y] = 0
        self.agentPosition = state
        x, y = self.getAgentRowAndCol()
        self.grid[x][y] = 1


    def offGridMove(self, newState, oldState):
        '''
        Determine if agent moves off grid.
        :param newState:
        :param oldState:
        :return:
        '''
        if newState not in self.stateSpacePlus:
            return True
        elif oldState % self.m == 0 and oldState % self.m == self.m - 1:
            return True
        elif oldState % self.n == 0 and oldState % self.n == self.n - 1:
            return True
        else:
            return False

    def step(self, action):
        x, y = self.getAgentRowAndCol()
        newState = self.agentPosition + self.actionSpace[action]

        # Key for magic square represents how to enter.
        if newState in self.magicSquares.keys():
            newState = self.magicSquares[newState]

        reward = -1 if not self.isTerminalState(newState) else 0

        if not self.offGridMove(newState, self.agentPosition):
            self.setState(newState)
            return newState, reward, self.isTerminalState(newState), None
        else:
            return self.agentPosition, reward, self.isTerminalState(self.agentPosition), None

    def reset(self):
        self.agentPosition = 0
        self.grid = np.zeros((self.m, self.n))
        self.addMagicSquares(self.magicSquares)
        return self.agentPosition

    def render(self):
        print('---------------')
        for row in self.grid:
            for col in row:
                if col == 0:
                    print('-', end='\t')
                elif col == 1:
                    print('X', end='\t') # agent
                elif col == 2:
                    print('Ain', end='\t')
                elif col == 3:
                    print('Aout', end='\t')
                elif col == 4:
                    print('Bin', end='\t')
                elif col == 5:
                    print('Bout', end='\t')

            print("\n")
        print('---------------')

    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)

    def maxAction(self, Q, state, actions):
        values = np.array(Q[state, a] for a in actions)
        action = np.argmax(values)
        return actions[action]

if __name__ == '__main__':
    magicSquares = {18: 54, 63: 14}

    env = GridWorld(9, 9, magicSquares)

    # == Hyperparams ==
    ALPHA = 0.1
    GAMMA = 9.0
    EPSILON = .8

    NUMEPS = 50000

    # Create Q table
    Q = {}
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[state, action] = 0

    totalRewards = np.zeros(NUMEPS)
    env.render()
    for ep in range(NUMEPS):
        if ep % 5000 == 0:
            print('Episode: ', ep)

        done = False
        epRewards = 0
        observation = env.reset()

        while not done:
            rand = np.random.random()
            action = env.maxAction(Q, observation, env.possibleActions) if rand < (1-EPSILON) \
                else env.actionSpaceSample()

            # take action
            observation_, reward, done, info = env.step(action)
            epRewards += reward

            action_ = env.maxAction(Q, observation, env.possibleActions)

            Q[observation, action] += ALPHA * (reward + GAMMA*Q[observation_, action_] - Q[observation, action])

            observation = observation_

        if EPSILON - 2 / NUMEPS > 0:
            EPSILON -= 2 / NUMEPS
        else:
            EPSILON = 0

        totalRewards[ep] = epRewards

    plt.plot(totalRewards)
    plt.show()