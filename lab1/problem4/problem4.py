# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import matplotlib.pyplot as plt
import itertools
import pickle
from tqdm import tqdm


def epsilon_greedy(k, epsilon, W, N, state):
    # Explore randomly with probability epsilon, otherwise exploit the best policy for that state
    action = 0
    if np.random.rand() <= epsilon:
        action = np.random.choice(k)
    else:
        best_action = float('-inf')
        for a in range(k):
            q = W[:, a].reshape(-1, 1).T @ compute_phi(N, state)
            if best_action < q:
                best_action = q
                action = a

    return action


# Functions used during training
def running_average(x, N):
    """
    Function used to compute the running mean
    of the last N elements of a vector x
    """
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def scale_state_variables(s, low, high):
    """ Rescaling of s to the box [0,1]^2 """
    x = (s - low) / (high - low)
    return x


def elegibility_greedy_update(Z, discount_factor, lbd, action_t, phi, k):
    for action in range(k):
        if action == action_t:
            Z[:, action] = discount_factor * lbd * Z[:, action] + phi.reshape(-1, )
        else:
            Z[:, action] = discount_factor * lbd * Z[:, action]
    Z = np.clip(Z, -5, 5)

    return Z


def update_w(momentum, V, alpha, delta, Z, W):
    # Update Velocity term
    V = momentum * V + alpha * delta * Z
    # Update Weights
    W = W + momentum * V + alpha * delta * Z

    return W


def compute_phi(N, state):
    # Initialize Fourier Basis for particular state
    phi = np.cos(np.pi * N.T @ np.array(state).reshape(-1, 1))

    return phi


def SARSA(action, reward, next_action, discount_factor, W, phi, phi_next):
    delta = reward + discount_factor * W[:, next_action].reshape(-1, 1).T @ phi_next - \
            W[:, action].reshape(-1, 1).T @ phi

    return float(delta)


def compute_learning_rate(lr, N):
    alpha = []
    for eta in N.T:  # Transpose in order to get each of the etas
        module = np.linalg.norm(eta)
        if module != 0:
            alpha.append(lr / module)
        else:
            alpha.append(lr)

    return np.array(alpha)


def main():
    # Import and initialize Mountain Car Environment
    env = gym.make('MountainCar-v0')
    env.reset()
    k = env.action_space.n  # tells you the number of actions (push left, push right, no push)
    # Parameters
    N_episodes = 200  # Number of episodes to run for training
    discount_factor = 1.  # Value of gamma
    lbd = 0.7  # Eligibility parameter
    p = 2  # Order of the problem
    momentum = 0.6  # momentum
    dimensions = 2  # Velocity and height
    epsilon = np.arange(0, 1, 1/N_episodes)[::-1]  # Epsilon value
    # Initialize learning rate
    m = pow(p + 1, dimensions)  # Number of basis functions
    W = np.random.random((m, k))  # weights
    N = np.array(list(itertools.product(range(p + 1), range(p + 1)))).T  # Size (dimensions x m)
    Z = np.zeros(W.shape)  # Initialize eligibility traces
    V = np.zeros(W.shape)  # Velocity term
    lr = 0.001
    alpha = compute_learning_rate(lr, N)

    # Reward
    episode_reward_list = []  # Used to save episodes reward

    # Training process
    for i in tqdm(range(N_episodes)):
        # Reset enviroment data
        done = False
        state = scale_state_variables(env.reset(), env.observation_space.low, env.observation_space.high)
        total_episode_reward = 0.
        while not done:
            # env.render()
            # Take a random action
            # env.action_space.n tells you the number of actions
            # available
            action = epsilon_greedy(k, epsilon[i], W, N, state)

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)
            next_state = scale_state_variables(next_state, env.observation_space.low, env.observation_space.high)

            # Update episode reward
            total_episode_reward += reward

            # Compute phi's
            phi_current = compute_phi(N, state)
            phi_next = compute_phi(N, next_state)

            # Compute next action based on next state
            next_action = epsilon_greedy(k, epsilon[i], W, N, next_state)

            # Compute total temporal error SARSA(state, action, reward, next_state, next_action)
            delta = SARSA(action, reward, next_action, discount_factor, W, phi_current, phi_next)

            # Update state for next iteration
            state = next_state

            # Update parameters Z and W
            Z = elegibility_greedy_update(Z, discount_factor, lbd, action, phi_current, k)
            for idx, lr in enumerate(alpha):
                W[idx] = update_w(momentum, V[idx], lr, delta, Z[idx], W[idx])

        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        # Close environment
        env.close()

    data = {'W': W.T,
            "N": N.T
            }

    pickle.dump(data, open("weights.pkl", "wb"))

    # Plot Rewards
    plt.plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward')
    plt.plot([i for i in range(1, N_episodes + 1)], running_average(episode_reward_list, 10),
             label='Average episode reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
