import gym

env = gym.make('CartPole-v0')

for eps in range(20):

    obs = env.reset()

    for t in range(1000):

        env.render()
        action = env.action_space.sample()

        o, r, done, info = env.step(action)

        if done:
            print("Episode finished after {} steps".format(t+1))
            break

env.close()