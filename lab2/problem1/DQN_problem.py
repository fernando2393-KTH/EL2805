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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
from collections import deque
import numpy as np
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import trange
import copy
from DQN_agent import RandomAgent, Agent

L = 5000  # Size of the experiences buffer
N = 64  # Batch size
HIDDEN_NODES = 128
MAX_EPS = 0.99  # Maximum value for epsilon
MIN_EPS = 0.05  # Minimum value for epsilon
Z = 0.925
C = round(L / N)  # Steps for target update


class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)


class MyNetwork(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, input_size, output_size, hidden_layer_size):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.input_layer_activation = nn.ReLU()

        # Create middle layer with ReLU activation
        self.middle_layer = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.middle_layer_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        # Compute second layer
        l2 = self.middle_layer(l1)
        l2 = self.middle_layer_activation(l2)

        # Compute output layer
        out = self.output_layer(l2)
        return out


def epsilon_greedy(epsilon, values):
    # Explore randomly with probability epsilon, otherwise exploit the best policy for that state
    if np.random.rand() <= epsilon:
        action = np.random.choice(range(len(values.detach().numpy().T)))
    else:
        action = values.max(1)[1].item()

    return action


def running_average(x, N):
    """ Function used to compute the running average
        of the last N elements of a vector x
    """
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def epsilon_decay(k, n_episodes):
    return max(MIN_EPS, MAX_EPS - ((MAX_EPS - MIN_EPS) * (k - 1)) / (n_episodes * Z - 1))


def compute_target(discount_factor, next_state, reward, done, target_network, agent):
    if not done:
        actions = agent.forward(next_state, target_network, buffer=True)
        best_choice = actions.max(1)[0].item()
        return reward + discount_factor * best_choice
    else:
        return reward


def main():

    # Import and initialize the discrete Lunar Laner Environment
    env = gym.make('LunarLander-v2')
    env.reset()

    # Parameters
    N_episodes = 100  # Number of episodes
    discount_factor = 0.95  # Value of the discount factor
    n_ep_running_average = 50  # Running average of 50 episodes
    n_actions = env.action_space.n  # Number of available actions
    dim_state = len(env.observation_space.high)  # State dimensionality
    lr = pow(10, -3)  # Learning rate

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode
    episode_reward_list = []  # this list contains the total reward per episode
    episode_number_of_steps = []  # this list contains the number of steps per episode

    # Random agent initialization
    agent = Agent(n_actions)

    # Buffer and network(s) initialization
    network = MyNetwork(input_size=dim_state, output_size=n_actions, hidden_layer_size=HIDDEN_NODES)
    target_network = MyNetwork(input_size=dim_state, output_size=n_actions, hidden_layer_size=HIDDEN_NODES)
    buffer = ExperienceReplayBuffer(maximum_length=L)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # -------- Training process -------- #

    # trange is an alternative to range in python, from the tqdm library
    # It shows a nice progression bar that you can update with useful information
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset enviroment data and initialize variables
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        epsilon = epsilon_decay(i + 1, N_episodes)

        while not done:
            # Take a random action
            values = agent.forward(state, network, buffer=False)  # Compute possible actions
            action = epsilon_greedy(epsilon, values)  # Take greedy action
            # Get next state and reward. The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)
            experience = (state, action, reward, next_state, done)  # Create the experience
            buffer.append(experience)  # Append the experience to the buffer

            if len(buffer) >= N:
                # Sample N elements from the buffer
                states, actions, rewards, next_states, dones = buffer.sample_batch(n=N)
                targets = []
                values = []
                
                for j in range(len(states)):
                    targets.append(compute_target(discount_factor, next_states[j],
                                                  rewards[j], dones[j], target_network, agent))
                    values.append(agent.forward(states[j], network, buffer=True)[0][actions[j]].double())
                targets = torch.tensor(targets)  # Convert targets to tensor
                values = torch.stack(values)  # Stack values as a single tensor

                # Training process, set gradients to 0
                optimizer.zero_grad()

                # Compute loss function
                loss = nn.functional.mse_loss(
                    values,
                    targets
                    )

                network = agent.backward(loss, optimizer, network)

                optimizer.step()

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state

            # TODO: How should we update this?
            if t % C == 0:
                target_network.load_state_dict(network.state_dict())
            
            t += 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

        # Updates the tqdm update bar with fresh information
        # (episode number, total reward of the last episode, total number of Steps
        # of the last episode, average reward, average number of steps)
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t,
                running_average(episode_reward_list, n_ep_running_average)[-1],
                running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, N_episodes + 1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, N_episodes + 1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, N_episodes + 1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, N_episodes + 1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
