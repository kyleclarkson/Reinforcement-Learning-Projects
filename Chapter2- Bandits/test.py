import random as rd
import numpy as np
import matplotlib.pyplot as plt

# A run is a single instance of a k-armed bandit
num_of_runs = 100
# A pull (time step) is the chooseing of an action, specified by the action selection rule.
num_of_pulls = 2000
# The number of actions (arms) in which one is choosen per pull.
num_of_arms = 10
# For each run, sample the true reward for each arm.
q_true = np.random.normal(0, 1, (num_of_runs, num_of_arms))

# An array of the indices of the true optimal arms for each run.
true_opt_arms = np.argmax(q_true, 1)

# The various epsilon values for the epsilon greedy strategy
epsilon_values = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 0.9, 1] #10 difference values

# Run epsilon-greedy stratergy

for epsilon_i in range(len(epsilon_values)):
    print("Running epsilon value: ", epsilon_values[epsilon_i])

    # Reward estimate
    Q = np.zeros((num_of_runs, num_of_arms))

    # Number of times an arm is pulled (init to 1)
    N = np.ones((num_of_runs, num_of_arms))

    # Initially pull all arms
    Q_init = np.random.normal(q_true, 1)

    # Set reward array.
    R_ep = []
    R_ep.append(0)
    R_ep.append(np.mean(Q_init))
    R_ep_opt = []

    for pull in range(2, num_of_pulls + 1):
        R_pull = []
        opt_arm_pull = 0  # Number of pulls of for the best arm (in this time step)

        for i in range(num_of_runs):
            # Choice arm uniformly at random
            if rd.random() < epsilon_values[epsilon_i]:
                arm = np.random.randint(num_of_arms)
            # Choose best arm.
            else:
                arm = np.argmax(Q[i])

            # The optimal arm was pulled.
            if arm == true_opt_arms[i]:
                opt_arm_pull += 1

            # Reward is estimated by sampling around the true action-value.
            reward_i = np.random.normal(q_true[i][arm], 1)

            R_pull.append(reward_i)
            N[i][arm] += 1
            Q[i][arm] = Q[i][arm] + (reward_i - Q[i][arm]) / N[i][arm]

        # == End runs ==
        average_R_pull = np.mean(R_pull)
        R_ep.append(average_R_pull)
        R_ep_opt.append(float(opt_arm_pull) * 100 / num_of_pulls)

    # == End pulls ==
    print("Average reward: ", np.average(R_ep))


