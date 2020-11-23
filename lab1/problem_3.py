import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

PLOTTING = False

np.random.seed(42)

GRID = np.array([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

UP = lambda pos: (pos[0] - 1, pos[1])
DOWN = lambda pos: (pos[0] + 1, pos[1])
LEFT = lambda pos: (pos[0], pos[1] - 1)
RIGHT = lambda pos: (pos[0], pos[1] + 1)
STILL = lambda pos: pos

PLAYER_POS = (0, 0)
POLICE_POS = (3, 3)
CONVERGENCE_STEPS = pow(10, 6)


def compute_reward(state):
    if state[0] == state[1]:
        return -10
    elif GRID[state[0]]:  # Where in the grid the value is 1, i.e. the bank
        return 1
    else:
        return 0


def initialize_Q(states, possible_actions):
    return np.zeros((len(states), len(possible_actions)))


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


def Q_Learning(Q, discount, states, player, police):
    step = 0
    best_q_initial_state = []
    alpha = {}
    pbar = tqdm(total=CONVERGENCE_STEPS - step)
    while step < CONVERGENCE_STEPS:
        current_state = (player.position, police.position)
        next_player_pos, next_player_action = player.possible_moves()
        action = np.random.choice(next_player_action)
        next_police_pos = police.possible_moves()
        current_reward = compute_reward(current_state)
        max_q = []
        for idx, position in enumerate(next_player_pos):
            Q_aux = 0
            next_possible_states = list(itertools.product([position], next_police_pos))
            for next_state in next_possible_states:
                Q_aux += (1 / len(next_possible_states)) * Q[states[next_state], next_player_action[idx]]
            max_q.append(Q_aux)

        # Define best reward
        best_q = max(max_q)
        # future_action = next_player_action[max_q.index(best_q)]
        if (current_state, action) not in alpha:
            alpha[(current_state, action)] = 1
        else:
            alpha[(current_state, action)] += 1
        # Update Q
        Q[states[current_state], action] += (1/pow(alpha[(current_state, action)], 2/3)) * (
                    current_reward + discount * best_q - Q[states[current_state], action])
        player.position = next_player_pos[next_player_action.index(action)]
        # Move police
        police.position = police.generate_move()
        step += 1
        aux = np.copy(Q[states[(PLAYER_POS, POLICE_POS)]])
        best_q_initial_state.append(aux)
        pbar.update(1)
    pbar.close()

    return best_q_initial_state, Q, alpha


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
    initial_q, optimal_Q, alpha = Q_Learning(Q_init, discount, states, player, police)
    initial_actions = ["Up", "Down", "Left", "Right", "Still"]
    for i in tqdm(range(len(initial_actions))):
        aux_list = []
        for j in range(len(initial_q)):
            aux_list.append(initial_q[j][i])
        plt.plot(range(len(initial_q)), aux_list, label=initial_actions[i])
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Value function")
    plt.show()


if __name__ == "__main__":
    main()
