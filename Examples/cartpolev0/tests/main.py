import gym
from cartpolev0.utils.wrapper import DiscreteObservationSpaceWrapper


def test_wrapper():
    print("=== test_wrapper() ===")
    env = gym.make('CartPole-v0')
    env = DiscreteObservationSpaceWrapper(
        env,
        num_bins=10,
        low=[-2.4, -2.0, -0.42, -3.5],
        high=[2.4, 2.0, 0.42, 3.5])

    print(env.observation([-2.4, -2.0, -0.42, -3.5]))
    print(env.observation([-.4, 1.3, -0.42, -1.5]))
    print(env.observation([-2.4, .54, 0.42, 2.5]))
    print(env.observation([2.4, 2.0, 0.42, 3.5]))


if __name__ == '__main__':
    test_wrapper()