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
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn

N = 64  # Batch size
HIDDEN_NODES = 64


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
        action = values.max(0)[1].item()

    return action


class AgentQ(object):
    """ Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    """

    def __init__(self, n_actions: int, dim_state: int, lr, N_episodes, discount_factor):
        self.n_actions = n_actions
        self.last_action = None
        # Buffer and network(s) initialization
        self.network = MyNetwork(input_size=dim_state, output_size=n_actions, hidden_layer_size=HIDDEN_NODES)
        self.target_network = MyNetwork(input_size=dim_state, output_size=n_actions, hidden_layer_size=HIDDEN_NODES)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.episodes = N_episodes
        self.discount_factor = discount_factor

    def forward(self, state: np.ndarray, epsilon, grad):
        """ Performs a forward computation """
        state_tensor = torch.tensor(state,
                                    requires_grad=grad,
                                    dtype=torch.float32)
        values = self.network(state_tensor)

        if grad:
            return values
        else:
            action = epsilon_greedy(epsilon, values)

            return action

    def forward_target(self, states: np.ndarray):
        """ Performs a forward computation """
        state_tensor = torch.tensor(states,
                                    requires_grad=False,
                                    dtype=torch.float32)

        values = self.target_network(state_tensor).max(axis=1)[0]

        return values

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


class Agent(object):
    """ Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        """ Performs a forward computation """
        pass

    def backward(self):
        """ Performs a backward pass on the network """
        pass


class RandomAgent(Agent):
    """ Agent taking actions uniformly at random, child of the class Agent """

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        """ Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        """
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action
