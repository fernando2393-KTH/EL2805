import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

PLOTTING = False

GRID = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

UP = lambda pos: (pos[0] + 1, pos[1])
DOWN = lambda pos: (pos[0] - 1, pos[1])
LEFT = lambda pos: (pos[0], pos[1] - 1)
RIGHT = lambda pos: (pos[0], pos[1] + 1)
STILL = lambda pos: pos

PLAYER_POS = (0, 0)
POLICE_POS = (3, 3)
CONVERGENCE_STEPS = pow(10, 6)

def compute_reward(state):
    if state[0] == state[1]:
        return -10
    elif GRID[state[0]]: # Where in the grid the value is 1, i.e. the bank
        return 1
    else:
        return 0

def initialize_Q(states, possible_actions):
    return np.ones((len(states), len(possible_actions)))

def get_possible_states():
    rows, columns = np.where(GRID != 2)
    possible_moves = [(x, y) for x, y in zip(rows, columns)]
    possible_states = list(itertools.product(possible_moves, possible_moves))
    states = {}
    for idx, state in enumerate(possible_states):
        states[state] = idx

    return possible_states, states

class Police:
    def __init__(self):
        self.position = POLICE_POS
        self.actions = [UP, DOWN, LEFT, RIGHT]

    def possible_moves(self):
        moves = []
        for action in self.actions:
            move = action(self.position)
            if (- 1 < move[0] < GRID.shape[0]) and (- 1 < move[1] < GRID.shape[1]):
                moves.append(move)

        return moves

    def generate_move(self):
        moves = self.possible_moves()

        return moves[np.random.choice(np.arange(len(moves)))]


class Player:
    def __init__(self):
        self.position = PLAYER_POS
        self.actions = [UP, DOWN, LEFT, RIGHT, STILL]

    def possible_moves(self):
        moves = []
        actions = []
        for idx, action in enumerate(self.actions):
            move = action(self.position)
            if (- 1 < move[0] < GRID.shape[0]) and (- 1 < move[1] < GRID.shape[1]):
                moves.append(move)
                actions.append(idx)

        return moves, actions

def Q_Learning(Q, discount, possible_states, states, player, police):
    step = 0
    Q_t = 0
    while step < CONVERGENCE_STEPS:
        alpha_step = 1 / (step + 1)
        next_player_pos, next_player_action = player.possible_moves()
        next_police_pos = police.possible_moves()
        current_reward = compute_reward((player.position, police.position))

        for idx, position in enumerate(next_player_pos):
            max_q = []
            Q_aux = 0
            next_possible_states = list(itertools.product([position], next_police_pos))
            for next_state in next_possible_states:
                Q_aux += (1 / len(next_possible_states)) * Q[states[next_state], next_player_action[idx]]
            max_q.append(Q_aux)
        
        # Define best reward
        best_q = max(max_q)
        action = max_q.index(best_q)

        # Update Q
        Q[states[(player.position, police.position)], action] = Q_t + alpha_step * (current_reward + discount * best_q - Q_t)
        Q_t = Q[states[(player.position, police.position)], action]

        # Move player
        player.position = next_player_pos[action]
        # Move police
        police.position = police.generate_move()





        step += 1 
    return

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
    discount = 0.8
    possible_states, states = get_possible_states()
    player = Player()
    police = Police()
    Q_init = initialize_Q(possible_states, player.actions)
    optimal_Q = Q_Learning(Q_init, discount, possible_states, states, player, police)
    

if __name__ == "__main__":
    main()