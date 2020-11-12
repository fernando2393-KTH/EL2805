import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1337)

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


def compute_reward(player_position, minotaur_position):
    distance = abs(player_position[0] - EXIT[0]) + abs(player_position[1] - EXIT[1])
    if player_position == minotaur_position:
        return float('-inf')
    elif distance == 0:
        return float('inf')
    else:
        return 10 / distance


def bellman(time, max_time, player, minotaur):
    if time != max_time:
        states_reward = []
        actions = player.possible_moves()
        minotaur_moves = minotaur.possible_moves()  # Generating possible minotaur moves
        for action in actions:
            next_player = Player()
            next_player.position = action  # Updating player state with new action
            reward = compute_reward(next_player.position, minotaur.position)
            next_states_reward = []
            for move in minotaur_moves:
                next_minotaur = Minotaur()
                next_minotaur.position = move
                next_states_reward.append((1 / len(minotaur_moves)) *
                                          bellman(time + 1, max_time, next_player, next_minotaur)[0])
            states_reward.append(reward + sum(next_states_reward))
        best_action = states_reward.index(max(states_reward))

        return states_reward[best_action], actions[best_action]

    else:
        states_reward = []
        actions = player.possible_moves()
        for action in actions:
            next_player = Player()
            next_player.position = action  # Updating player state with new action
            states_reward.append(compute_reward(next_player.position, minotaur.position))
        best_action = states_reward.index(max(states_reward))

        return states_reward[best_action], actions[best_action]


class Minotaur:
    def __init__(self):
        self.position = (6, 5)
        self.actions = [UP, DOWN, LEFT, RIGHT]

    def possible_moves(self):
        moves = []
        for action in self.actions:
            move = action(self.position)
            if (- 1 < move[0] < GRID.shape[0]) and (- 1 < move[1] < GRID.shape[1]):
                moves.append(move)

        return moves

    def generate_move(self):
        moves = []
        for action in self.actions:
            move = action(self.position)
            if (- 1 < move[0] < GRID.shape[0]) and (- 1 < move[1] < GRID.shape[1]):
                moves.append(move)

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
    plt.pause(0.25)


def main():
    player = Player()
    minotaur = Minotaur()
    player_positions = [player.position]
    minotaur_positions = [minotaur.position]
    time = 0
    max_time = 6
    while time <= max_time:
        reward, action = bellman(time, max_time, player, minotaur)
        player.position = action
        player_positions.append(player.position)
        minotaur.position = minotaur.generate_move()
        minotaur_positions.append(minotaur.position)
        time += 1
        if reward == float('inf'):
            for i in range(len(player_positions)):
                plot_grid(player_positions[i], minotaur_positions[i])
            print("Winner!")
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
