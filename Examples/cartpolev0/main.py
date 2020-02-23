import gym

from cartpolev0.utils.wrapper import DiscreteObservationSpaceWrapper
from cartpolev0.policies.QLearning import QLearningPolicy

if __name__ == '__main__':
    env = DiscreteObservationSpaceWrapper(
        env=gym.envs.make('CartPole-v0'),
        num_bins=10,
        low=[-2.4, -2.0, -0.42, -3.5],
        high=[2.4, 2.0, 0.42, 3.5]
    )

    learning = QLearningPolicy(env=env)

    print(learning.env.observation_space)
    action = learning.get_action(env.observation([-2.4, -2, -0.2, -3.5]))
    print(action)

    learning.update_Q(1, env.observation([-2.4, -2, -0.2, -3.5]))