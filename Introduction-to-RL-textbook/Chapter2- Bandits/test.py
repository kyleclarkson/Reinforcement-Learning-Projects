import random as rd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# A run is a single instance of a k-armed bandit
num_of_runs = 2000

# A pull is the decision of which arm to pull.
num_of_pulls = 4000

# The number of arms available to be pulled.
num_of_arms = 10

# For each run, sample the true reward for each arm.
q_true = np.random.normal(0, 1, (num_of_runs, num_of_arms))

# An array of the indices of the true optimal arms for each run.
true_opt_arms = np.argmax(q_true, 1)

# The various epsilon values for the epsilon greedy strategy
epsilon_values = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1]

# Set up plots for graphing rewards.
fig1 = plt.figure().add_subplot(111)
fig2 = plt.figure().add_subplot(111)

# Run epsilon-greedy strategy for each value.
for eps_i in range(len(epsilon_values)):

    # Reward estimate
    Q = np.zeros((num_of_runs, num_of_arms))

    # Number of times an arm is pulled (init to 1)
    N = np.ones((num_of_runs, num_of_arms))

    # Initially pull all arms
    Q_init = np.random.normal(q_true, 1)

    # Reward for this epsilon run.
    reward_eps = []
    reward_eps.append(0)
    # reward_eps.append(np.mean(Q_init))
    reward_eps_opt = []

    for pull in range(1, num_of_pulls + 1):
        reward_pull = []
        opt_arm_pull = 0  # Number of pulls of for the best arm (in this time step)

        for i in range(num_of_runs):
            # Choice arm uniformly at random
            if rd.random() < epsilon_values[eps_i]:
                arm = np.random.randint(num_of_arms)
            # Choose best arm.
            else:
                arm = np.argmax(Q[i])

            # The optimal arm was pulled.
            if arm == true_opt_arms[i]:
                opt_arm_pull += 1

            # Reward is estimated by sampling around the true action-value.
            reward_i = np.random.normal(q_true[i][arm], 1)

            reward_pull.append(reward_i)
            N[i][arm] += 1
            Q[i][arm] = Q[i][arm] + (reward_i - Q[i][arm]) / N[i][arm]

        # == End runs ==
        # Append average reward to rewards list.
        average_reward_pull = np.mean(reward_pull)
        reward_eps.append(average_reward_pull)
        # Percentage optimal arm was pulled.
        reward_eps_opt.append(float(opt_arm_pull) * 100 / num_of_pulls)

    fig1.plot(range(0, num_of_pulls + 1), reward_eps, label="$\epsilon$="+str(epsilon_values[eps_i]))
    fig2.plot(range(1, num_of_pulls + 1), reward_eps_opt, label="$\epsilon$="+str(epsilon_values[eps_i]))

    # == End pulls ==
    print("Epsilon value: ", epsilon_values[eps_i], "Average reward: ", np.average(reward_eps))

fig1.title.set_text('$\epsilon$-greedy : Average Reward vs Number of Pulls')
fig1.set_xlabel("Pulls")
fig1.set_ylabel("Average Reward")
fig1.legend(loc="best")

fig2.title.set_text("$\epsilon$-greedy : Percentage of Optimal Action vs Number of Pulls")
fig2.set_xlabel("Pulls")
fig2.set_ylabel("Percentage of Optimal Action")
fig2.legend(loc="best")
plt.show()


