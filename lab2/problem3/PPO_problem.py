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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 3
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
from collections import deque
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from PPO_agent import RandomAgent, AgentQ

M = 100
L = 30000
THRESHOLD = 200
computePDF = lambda mu, sigma_square, action: torch.pow(2 * np.pi * sigma_square, -1 / 2) * torch.exp(
    - torch.pow(action - mu, 2) / (2 * sigma_square))


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
        indices = np.arange(len(self.buffer))

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        return zip(*batch)


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N,)) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", dev)

    # Import and initialize Mountain Car Environment
    env = gym.make('LunarLanderContinuous-v2')
    env.reset()

    # Parameters
    N_episodes = 1600  # Number of episodes to run for training
    discount_factor = 0.99  # Value of gamma
    n_ep_running_average = 50  # Running average of 20 episodes
    m = len(env.action_space.high)  # dimensionality of the action
    epsilon = 0.2
    lr_critic = pow(10, -3)
    lr_actor = pow(10, -5)
    dim_state = len(env.observation_space.high)  # State dimensionality

    # Reward
    episode_reward_list = []  # Used to save episodes reward
    episode_number_of_steps = []

    # Agent initialization
    agent = AgentQ(m, dim_state, lr_critic, lr_actor, N_episodes, discount_factor, dev, epsilon)

    # Training process
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    for i in EPISODES:
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        t = 0
        # Initialize Buffer
        buffer = ExperienceReplayBuffer(maximum_length=L)
        while not done:
            # Take a random action
            state_tensor = torch.tensor(state, device=dev, dtype=torch.float32)
            mu_t, sigma_square_t = agent.forward_actor(state_tensor)  # Compute possible actions
            action = torch.distributions.multivariate_normal.MultivariateNormal(mu_t,
                                                                                sigma_square_t * torch.eye(m).to(dev)
                                                                                ).sample()
            pdf_old = computePDF(mu_t[0], sigma_square_t[0], action[0]) * computePDF(mu_t[1], sigma_square_t[1],
                                                                                     action[1])
            # Get next state and reward. The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action.cpu().detach().numpy())
            experience = (pdf_old, state, action, reward, next_state, done)  # Create the experience
            buffer.append(experience)  # Append the experience to the buffer
            # Update episode reward
            total_episode_reward += reward
            # Update state for next iteration
            state = next_state
            t += 1

        # Get params from buffer
        pdfs_old, states, actions, rewards, next_states, _ = buffer.sample_batch(n=t)
        pdfs_old_tensor = torch.tensor(pdfs_old, dtype=torch.float32, device=dev).reshape(-1, 1)
        states_tensor = torch.tensor(states, dtype=torch.float32, device=dev)
        actions_tensor = torch.stack(actions)
        targets_list = np.zeros(t)
        for idx in range(t):
            for jdx in range(idx, t):
                targets_list[idx] += pow(discount_factor, (jdx - idx)) * rewards[jdx]
        targets = torch.tensor(targets_list, device=dev, dtype=torch.float32).reshape(-1, 1)

        # Compute values
        values = agent.forward_critic(states_tensor)

        # Compute Psi
        psi = targets - values

        for n in range(M):
            # Compute values
            values = agent.forward_critic(states_tensor)
            
            # Update w (critic network)
            agent.backward_critic(targets, values)
            
            # Update theta (actor network)
            agent.backward_actor(states_tensor, actions_tensor, psi.detach(), pdfs_old_tensor)
        
        # Append episode reward
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

        if running_average(episode_reward_list, n_ep_running_average)[-1] > THRESHOLD:  # Early stopping
            break

    # Save network
    torch.save(agent.actor_network, 'neural-network-3-actor.pth')
    torch.save(agent.critic_network, 'neural-network-3-critic.pth')

    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, len(episode_reward_list) + 1)], episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, len(episode_reward_list) + 1)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, len(episode_number_of_steps) + 1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, len(episode_number_of_steps) + 1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.savefig('Result_problem3.png')
    plt.show()


if __name__ == "__main__":
    main()
