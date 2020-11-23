import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

PLOTTING = False

GRID = np.array([
    [1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1]
])

UP = lambda pos: (pos[0] - 1, pos[1])
DOWN = lambda pos: (pos[0] + 1, pos[1])
LEFT = lambda pos: (pos[0], pos[1] - 1)
RIGHT = lambda pos: (pos[0], pos[1] + 1)
STILL = lambda pos: pos

PLAYER_POS = (0, 0)
POLICE_POS = (1, 2)
MAX_IT = 1000


def compute_reward(state):
    if state[0] == state[1]:
        return -50
    elif GRID[state[0]]:
        return 10
    else:
        return 0


def get_possible_states():
    rows, columns = np.where(GRID != 2)
    possible_moves = [(x, y) for x, y in zip(rows, columns)]
    possible_states = list(itertools.product(possible_moves, possible_moves))
    states = {}
    for idx, state in enumerate(possible_states):
        states[state] = idx

    return possible_states, states


def value_iteration(possible_states, states, gamma, tolerance):
    # Initial best value
    best_V = np.ones((len(possible_states), 1))
    V = np.zeros((len(possible_states), 1))
    # Initial best policy
    value = np.empty((), dtype=object)
    value[()] = (0, 0)
    best_policy = np.full((len(possible_states), 1), value, dtype=object)
    # Iterate until convergence
    n = 0
    while np.linalg.norm(V - best_V) >= tolerance:
        # Update the value function
        V = np.copy(best_V)
        # Compute the new BV
        for state in possible_states:
            rewards = []
            player = Player()
            player.position = state[0]
            police = Police()
            police.position = state[1]
            p_actions = player.possible_moves()
            pl_actions = police.possible_moves(player.position)
            for action in p_actions:
                reward = compute_reward(state)
                next_possible_states = list(itertools.product([action], pl_actions))
                for next_state in next_possible_states:
                    reward += gamma * (1 / len(next_possible_states)) * V[states[next_state]]
                rewards.append(reward)
            # Define best reward
            u = max(rewards)
            best_V[states[state]] = u
            # Compute best policy
            a = rewards.index(u)
            best_policy[states[state]] = [p_actions[a]]
        n += 1
    # Return the best possible moves for each state
    print("error: ", np.linalg.norm(V - best_V))
    return best_policy, best_V[states[(PLAYER_POS, POLICE_POS)]]


class Player:
    def __init__(self):
        self.position = PLAYER_POS
        self.actions = [UP, DOWN, LEFT, RIGHT, STILL]

    def possible_moves(self):
        moves = []
        for action in self.actions:
            move = action(self.position)
            if (- 1 < move[0] < GRID.shape[0]) and (- 1 < move[1] < GRID.shape[1]):
                moves.append(move)

        return moves


class Police:
    def __init__(self):
        self.position = POLICE_POS
        self.actions = [UP, DOWN, LEFT, RIGHT, STILL]

    def possible_moves(self, player_position):
        moves = []
        for action in self.actions:
            move = action(self.position)
            original_dist = np.linalg.norm(np.array(self.position) - np.array(player_position))
            new_dist = np.linalg.norm(np.array(move) - np.array(player_position))
            if (- 1 < move[0] < GRID.shape[0]) and (- 1 < move[1] < GRID.shape[1]) \
                    and (new_dist - original_dist < 1):
                if player_position[0] == self.position[0] or player_position[1] == self.position[1]:  # Same line case
                    moves.append(move)
                else:  # Different line case
                    if new_dist < original_dist:
                        moves.append(move)

        return moves

    def generate_moves(self, player_position):
        moves = self.possible_moves(player_position)
        return moves[np.random.choice(np.arange(len(moves)))]


def plot_grid(player_pos, police_pos):
    fig, ax = plt.subplots()
    ax.text(y=player_pos[0], x=player_pos[1], s="Player", va='center', ha='center', fontsize=15, color='b')
    ax.text(y=police_pos[0], x=police_pos[1], s="Police", va='center', ha='center', fontsize=15, color='r')
    rows, columns = np.where(GRID == 1)
    for idx, (x, y) in enumerate(zip(rows, columns)):
        ax.text(y=x + 0.3, x=y, s="Bank " + str(idx + 1), va='center', ha='center', fontsize=10, color='g')
    ax.imshow(GRID, cmap="Pastel1")
    plt.show(block=False)
    plt.pause(0.35)
    plt.close()


def main():
    epsilon = pow(10, -4)
    gammas = np.round(np.arange(0, 1, 0.05), 2)
    possible_states, states = get_possible_states()
    rewards = [[] for _ in range(len(gammas))]
    best_values = []
    for idx, gamma in tqdm(enumerate(gammas)):
        it = 0
        tolerance = (1 - gamma) * epsilon / (gamma + np.finfo(float).eps)
        best_policy, best_value = value_iteration(possible_states, states, gamma, tolerance)
        best_values.append(best_value)
        player = Player()
        police = Police()
        accumulative_reward = 0
        while it < MAX_IT:
            if PLOTTING:
                plot_grid(player.position, police.position)
            accumulative_reward += compute_reward((player.position, police.position))
            rewards[idx].append(accumulative_reward)
            if (np.array(player.position) == np.array(police.position)).all():
                player.position = PLAYER_POS
                police.position = POLICE_POS
            previous_player_position = player.position  # Save the previous position as the
            # police don't know what player will do
            player.position = best_policy[states[(player.position, police.position)]][0]
            police.position = police.generate_moves(previous_player_position)
            it += 1
    for idx, reward in enumerate(rewards):
        plt.plot(range(MAX_IT), reward, label="Gamma=" + str(gammas[idx]))
    plt.legend()
    plt.title("Reward w.r.t gamma")
    plt.xlabel("Time")
    plt.ylabel("Reward")
    plt.show()

    plt.plot(gammas, best_values)
    plt.title("Value Function w.r.t gamma")
    plt.xlabel("Gamma")
    plt.ylabel("Value")
    plt.show()


if __name__ == "__main__":
    main()
