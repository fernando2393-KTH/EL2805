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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
from DDPG_soft_updates import soft_updates

HIDDEN_NODES = [400, 200]

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_units):
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(input_size, hidden_units[0])
        self.input_layer_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(hidden_units[0], output_size)

    def forward(self, x):
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        # Compute output layer
        out = self.output_layer(l1)
        return out


class MyNetwork(nn.Module):
    """ Create a feedforward neural network """

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
        self.output_layer_activation = nn.Tanh()

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


class AgentQ(object):
    """ Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    """

    def __init__(self, n_actions: int, dim_state: int, actor_lr, target_lr, N_episodes, discount_factor, mu, sigma, tau, dev):
        self.n_actions = n_actions
        self.last_action = None
        # Buffer and network(s) initialization
        self.network = MyNetwork(input_size=dim_state + n_actions, output_size=1, hidden_units=HIDDEN_NODES).to(dev)
        self.target_network = MyNetwork(input_size=dim_state + n_actions, output_size=1,
                                        hidden_units=HIDDEN_NODES).to(dev)
        self.policy_network = PolicyNetwork(input_size=dim_state, output_size=n_actions,
                                            hidden_units=HIDDEN_NODES).to(dev)
        self.target_policy_network = PolicyNetwork(input_size=dim_state, output_size=n_actions,
                                                   hidden_units=HIDDEN_NODES).to(dev)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=actor_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=target_lr)
        self.episodes = N_episodes
        self.discount_factor = discount_factor
        self.dev = dev
        self.m = n_actions
        self.n = np.zeros(n_actions)
        self.mu = mu
        self.sigma = sigma
        self.tau = tau

    def forward(self, state: np.ndarray, grad):
        """ Performs a forward computation """
        state_tensor = torch.tensor(state,
                                    requires_grad=grad,
                                    dtype=torch.float32,
                                    device=self.dev)
        if grad:
            return self.network(state_tensor)
        else:
            action = self.policy_network(state_tensor).detach().numpy() + self.n

            return action

    def forward_target(self, states: np.ndarray):
        """ Performs a forward computation """
        state_tensor = torch.tensor(states,
                                    requires_grad=False,
                                    dtype=torch.float32,
                                    device=self.dev)

        actions_tensor = self.target_policy_network(state_tensor).type(torch.int64)

        net_input = torch.cat((state_tensor, actions_tensor), 1)

        values = self.target_network(net_input)

        return values

    def noise(self):
        self.n = -self.mu * self.n + \
                 np.random.multivariate_normal(np.zeros(self.m), pow(self.sigma, 2) * np.identity(self.m))

    def backward(self, values, targets, t, C):
        """ Performs a backward pass on the network """
        # Compute gradient and Perform backward pass (backpropagation)
        # Training process, set gradients to 0
        self.optimizer.zero_grad()

        # Compute loss function
        loss = nn.functional.mse_loss(
            values,
            targets
        )

        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.)
        self.optimizer.step()
        if t % C == 0:
            self.target_network.load_state_dict(self.network.state_dict())

    def policy_backward(self, states, N):
        state_tensor = torch.tensor(states,
                                    requires_grad=False,
                                    dtype=torch.float32,
                                    device=self.dev)
        self.policy_optimizer.zero_grad()
        jacobian = -1/N * sum([torch.gather(self.network(state_tensor), 1, self.policy_network(state_tensor))])
        jacobian.backward()
        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.)
        self.policy_optimizer.step()
        soft_updates(self.policy_network, self.target_policy_network, self.tau)
        soft_updates(self.network, self.target_network, self.tau)


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
        """ Performs a forward computation """
        pass

    def backward(self):
        """ Performs a backward pass on the network """
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
                    the parent class Agent.
        """
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)
