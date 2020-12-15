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
# Last update: 29th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn

HIDDEN_NODES = [400, 200]
computePDF = lambda mu, sigma_square, action: torch.pow(2 * np.pi * sigma_square, -1 / 2) * torch.exp(
    - torch.pow(action - mu, 2) / (2 * sigma_square))


class CriticNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_units):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_units[0])
        self.input_layer_activation = nn.ReLU()

        # Create middle layer with ReLU activation
        self.middle_layer = nn.Linear(hidden_units[0], hidden_units[1])
        self.middle_layer_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(hidden_units[1], output_size)

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


class ActorNetwork(nn.Module):
    """ Create a feedforward neural network """

    def __init__(self, input_size, output_size, hidden_units):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_units[0])
        self.input_layer_activation = nn.ReLU()

        # Create middle layer head 1 - mu with ReLU activation
        self.middle_layer_head_1 = nn.Linear(hidden_units[0], hidden_units[1])
        self.middle_layer_activation_head_1 = nn.ReLU()

        # Create output layer head 1 - mu with Tanh
        self.output_layer_head_1 = nn.Linear(hidden_units[1], output_size)
        self.output_layer_activation_head_1 = nn.Tanh()

        # Create middle layer head 2 - sigma with ReLU activation
        self.middle_layer_head_2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.middle_layer_activation_head_2 = nn.ReLU()

        # Create output layer head 2 - sigma
        self.output_layer_head_2 = nn.Linear(hidden_units[1], output_size)
        self.output_layer_activation_head_2 = nn.Sigmoid()

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        # Compute second layer
        l2_mu = self.middle_layer_head_1(l1)
        l2_mu = self.middle_layer_activation_head_1(l2_mu)

        # Compute output layer
        out_mu = self.output_layer_head_1(l2_mu)
        out_mu = self.output_layer_activation_head_1(out_mu)

        # Compute second layer
        l2_sigma = self.middle_layer_head_2(l1)
        l2_sigma = self.middle_layer_activation_head_2(l2_sigma)

        # Compute output layer
        out_sigma = self.output_layer_head_2(l2_sigma)
        out_sigma = self.output_layer_activation_head_2(out_sigma)

        return out_mu, out_sigma


class AgentQ(object):
    """ Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    """

    def __init__(self, n_actions: int, dim_state: int, critic_lr, actor_lr, N_episodes, discount_factor, dev, epsilon):
        self.last_action = None
        self.actor_network = ActorNetwork(input_size=dim_state, output_size=n_actions,
                                          hidden_units=HIDDEN_NODES).to(dev)
        self.critic_network = CriticNetwork(input_size=dim_state, output_size=1,
                                            hidden_units=HIDDEN_NODES).to(dev)
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.discount_factor = discount_factor
        self.dev = dev
        self.m = n_actions
        self.n = np.zeros(n_actions)
        self.epsilon = epsilon

    def forward_critic(self, state):
        """ Performs a forward computation """
        return self.critic_network(state)

    def forward_actor(self, state):
        """ Performs a forward computation """
        return self.actor_network(state)

    def backward_critic(self, values, targets):
        """ Performs a backward pass on the network """
        # Compute gradient and Perform backward pass (backpropagation)
        # Training process, set gradients to 0

        # Compute loss function
        loss = nn.functional.mse_loss(
            values,
            targets
        )

        self.critic_optimizer.zero_grad()
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.critic_network.parameters(), max_norm=1.)
        self.critic_optimizer.step()

    def backward_actor(self, states, actions, psi, pdfs_old):
        mu_t, sigma_square_t = self.actor_network(states)  # Compute possible actions
        pdf = (computePDF(mu_t[:, 0], sigma_square_t[:, 0], actions[:, 0]) *
               computePDF(mu_t[:, 1], sigma_square_t[:, 1], actions[:, 1])).reshape(-1, 1)
        r_theta = pdf / pdfs_old
        clip = torch.clamp(r_theta, (1 - self.epsilon), (1 + self.epsilon))
        j = torch.min(r_theta * psi, clip * psi)
        loss = - torch.mean(j)
        # Compute loss
        self.actor_optimizer.zero_grad()
        loss.backward()
        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.actor_network.parameters(), max_norm=1.)

        # Update theta
        self.actor_optimizer.step()


class Agent(object):
    """ Base agent class

        Args:
            n_actions (int): actions dimensionality

        Attributes:
            n_actions (int): where we store the dimensionality of an action
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    """ Agent taking actions uniformly at random, child of the class Agent"""

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        """ Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent
        """
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)
