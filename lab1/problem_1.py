import numpy as np
import matplotlib.pyplot as plt

GRID = np.array([
    [1, 1, -1, 1, 1, 1, 1, 1],
    [1, 1, -1, 1, 1, -1, 1, 1],
    [1, 1, -1, 1, 1, -1, -1, -1],
    [1, 1, -1, 1, 1, -1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, -1, -1, -1, -1, -1, -1, 1],
    [1, 1, 1, 1, -1, 1, 1, 1]
])


UP = lambda pos: (pos[0] + 1, pos[1])
DOWN = lambda pos: (pos[0] - 1, pos[1])
LEFT = lambda pos: (pos[0], pos[1] - 1)
RIGHT = lambda pos: (pos[0], pos[1] + 1)

ACTIONS = [UP, DOWN, LEFT, RIGHT]


class Minotaur:
    def __init__(self):
        self.position = (6, 5)

    def generate_move(self):
        moves = []
        for action in ACTIONS:
            move = action(self.position)
            if (- 1 < move[0] < GRID.shape[0]) and (- 1 < move[1] < GRID.shape[1]):
                moves.append(move)

        return moves[np.random.choice(np.arange(len(moves)))]


class Player:
    def __init__(self):
        self.position = (0, 0)

    def generate_move(self):
        moves = []
        for action in ACTIONS:
            move = action(self.position)
            if (- 1 < move[0] < GRID.shape[0]) and (- 1 < move[1] < GRID.shape[1]) and GRID[move[0], move[1]] != -1:
                moves.append(move)

        return moves[np.random.choice(np.arange(len(moves)))]


def plot_grid(player_pos, minotaur_pos):
    fig, ax = plt.subplots()
    ax.text(y=player_pos[0], x=player_pos[1], s="P", va='center', ha='center', fontsize=25, color='b')
    ax.text(y=minotaur_pos[0], x=minotaur_pos[1], s="M", va='center', ha='center', fontsize=25, color='b')
    ax.text(y=6+0.3, x=5, s="Exit", va='center', ha='center', fontsize=10, color='g')
    ax.imshow(GRID, cmap="gray")
    plt.show(block=False)
    plt.pause(0.2)


def main():
    player = Player()
    minotaur = Minotaur()
    plot_grid(player.position, minotaur.position)
    while player.position != minotaur.position:
        player.position = player.generate_move()
        minotaur.position = minotaur.generate_move()
        plot_grid(player.position, minotaur.position)


if __name__ == "__main__":
    main()
