import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

GRID = np.array([
    [1, 1, -1, 1, 1, 1, 1, 1],
    [1, 1, -1, 1, 1, -1, 1, 1],
    [1, 1, -1, 1, 1, -1, -1, -1],
    [1, 1, -1, 1, 1, -1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, -1, -1, -1, -1, -1, -1, 1],
    [1, 1, 1, 1, -1, 1, 1, 1]
])

EXIT = (6, 5)

UP = lambda pos: (pos[0] + 1, pos[1])
DOWN = lambda pos: (pos[0] - 1, pos[1])
LEFT = lambda pos: (pos[0], pos[1] - 1)
RIGHT = lambda pos: (pos[0], pos[1] + 1)
STILL = lambda pos: pos


# TODO: Fix reward for blocked paths
def compute_reward(state):
    if state[0] == state[1]:
        return -100
    elif state[0] == EXIT:
        return 1
    else:
        return -1


def bellman(max_time):
    rows, columns = np.where(GRID == 1)
    possible_moves = [(x, y) for x, y in zip(rows, columns)]
    possible_states = list(itertools.product(possible_moves, possible_moves))
    states = {}
    for idx, state in enumerate(possible_states):
        states[state] = idx
    u_star = np.zeros((len(possible_states), max_time))
    value = np.empty((), dtype=object)
    value[()] = (0, 0)
    a_star = np.full((u_star.shape[0], u_star.shape[1] - 1), value, dtype=object)

    for time in tqdm(reversed(range(max_time))):
        if time == max_time - 1:  # Last state: only reward is computed
            for state in possible_states:
                u_star[states[state], time] = compute_reward(state)
        else:
            for state in possible_states:
                rewards = []
                player = Player()
                player.position = state[0]
                p_actions = player.possible_moves()
                minotaur = Minotaur()
                minotaur.position = state[1]
                m_actions = minotaur.possible_moves()
                for action in p_actions:
                    reward = compute_reward(state)  # TODO: Check this reward
                    next_possible_states = list(itertools.product([action], m_actions))
                    for next_state in next_possible_states:
                        reward += (1 / len(next_possible_states)) * u_star[states[next_state], time + 1]
                    rewards.append(reward)
                u = max(rewards)
                a = rewards.index(u)
                u_star[states[state], time] = u
                a_star[states[state], time] = p_actions[a]

    return u_star, a_star, states


class Minotaur:
    def __init__(self):
        self.position = (6, 5)
        self.actions = [UP, DOWN, LEFT, RIGHT, STILL]

    def possible_moves(self):
        moves = []
        for action in self.actions:
            move = action(self.position)
            if (- 1 < move[0] < GRID.shape[0]) and (- 1 < move[1] < GRID.shape[1]) and (GRID[move[0], move[1]] != -1):
                moves.append(move)
            else:
                move = action(move)
                if (- 1 < move[0] < GRID.shape[0]) and (- 1 < move[1] < GRID.shape[1]) and \
                        (GRID[move[0], move[1]] != -1):
                    moves.append(move)

        return moves

    def generate_move(self):
        moves = self.possible_moves()

        return moves[np.random.choice(np.arange(len(moves)))]


class Player:
    def __init__(self):
        self.position = (0, 0)
        self.actions = [UP, DOWN, LEFT, RIGHT, STILL]

    def possible_moves(self):
        moves = []
        for action in self.actions:
            move = action(self.position)
            if (- 1 < move[0] < GRID.shape[0]) and (- 1 < move[1] < GRID.shape[1]) and GRID[move[0], move[1]] != -1:
                moves.append(move)

        return moves


def plot_grid(player_pos, minotaur_pos):
    fig, ax = plt.subplots()
    ax.text(y=player_pos[0], x=player_pos[1], s="P", va='center', ha='center', fontsize=25, color='b')
    ax.text(y=minotaur_pos[0], x=minotaur_pos[1], s="M", va='center', ha='center', fontsize=25, color='b')
    ax.text(y=EXIT[0]+0.3, x=EXIT[1], s="Exit", va='center', ha='center', fontsize=10, color='g')
    ax.imshow(GRID, cmap="gray")
    plt.show(block=False)
    plt.pause(0.35)
    plt.close()


def main():
    player = Player()
    minotaur = Minotaur()
    player_positions = [player.position]
    minotaur_positions = [minotaur.position]
    time = 0
    max_time = 20
    u_star, a_star, states = bellman(max_time)
    while time < max_time - 1:
        player.position = a_star[states[(player.position, minotaur.position)], time]
        player_positions.append(player.position)
        minotaur.position = minotaur.generate_move()
        minotaur_positions.append(minotaur.position)
        time += 1
        if player.position == EXIT:
            for i in range(len(player_positions)):
                plot_grid(player_positions[i], minotaur_positions[i])
            print("Winner!")
            print("Won after " + str(time) + " actions.")
            exit(0)
        elif player.position == minotaur.position:
            for i in range(len(player_positions)):
                plot_grid(player_positions[i], minotaur_positions[i])
            print("Game Over!")
            exit(0)
    for i in range(len(player_positions)):
        plot_grid(player_positions[i], minotaur_positions[i])
    print("You did not reach the exit in time!")
    exit(0)


if __name__ == "__main__":
    main()
