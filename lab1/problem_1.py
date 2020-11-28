'''
@Authors:
Fernando Garcia Sanz - 970718-0312
Gustavo Teodoro Döhler Beck - 940218-0195
'''

import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

FINITE = False  # Set this to true in order to play a single game
PLOTTING = False  # Set this to true in order to plot a single game

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
UP = lambda pos: (pos[0] - 1, pos[1])
DOWN = lambda pos: (pos[0] + 1, pos[1])
LEFT = lambda pos: (pos[0], pos[1] - 1)
RIGHT = lambda pos: (pos[0], pos[1] + 1)
STILL = lambda pos: pos


def compute_reward(state):
    if state[0] == state[1]:
        return 0
    elif state[0] == EXIT:
        return 1
    else:
        return 0


def get_possible_states():
    rows, columns = np.where(GRID == 1)
    possible_moves = [(x, y) for x, y in zip(rows, columns)]
    possible_states = list(itertools.product(possible_moves, possible_moves))
    states = {}
    for idx, state in enumerate(possible_states):
        states[state] = idx
    
    return possible_states, states


def bellman(max_time):
    # Generate states and possible states
    possible_states, states = get_possible_states()
    u_star = np.zeros((len(possible_states), max_time))
    value = np.empty((), dtype=object)
    value[()] = (0, 0)
    a_star = np.full((u_star.shape[0], u_star.shape[1] - 1), value, dtype=object)
    for time in reversed(range(max_time)):
        if time == max_time - 1:  # Last state: only reward is computed
            for state in possible_states:
                u_star[states[state], time] = compute_reward(state)
        else:
            for state in possible_states:
                rewards = []
                player = Player()
                player.position = state[0]
                minotaur = Minotaur()
                minotaur.position = state[1]
                if player.position == (EXIT or minotaur.position):
                    u_star[states[state], time] = compute_reward(state)
                    a_star[states[state], time] = player.position
                    continue
                p_actions = player.possible_moves()
                m_actions = minotaur.possible_moves()
                for action in p_actions:
                    reward = compute_reward(state)
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
    ax.text(y=EXIT[0] + 0.3, x=EXIT[1], s="Exit", va='center', ha='center', fontsize=10, color='g')
    ax.imshow(GRID, cmap="gray")
    plt.show(block=False)
    plt.pause(0.35)
    plt.close()


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
    while np.linalg.norm(V - best_V) >= tolerance and n < 50:
        # Update the value function
        V = np.copy(best_V)
        # Compute the new BV
        for state in possible_states:
            rewards = []
            player = Player()
            player.position = state[0]
            minotaur = Minotaur()
            minotaur.position = state[1]
            if player.position == (EXIT or minotaur.position):
                best_V[states[state]] = compute_reward(state)
                best_policy[states[state]] = [player.position]
                continue
            p_actions = player.possible_moves()
            m_actions = minotaur.possible_moves()
            for action in p_actions:
                reward = compute_reward(state)
                next_possible_states = list(itertools.product([action], m_actions))
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
    return best_policy


def main():
    if FINITE:
        probabilities = []
        max_time = 20
        for max_t in tqdm(range(1, max_time)):
            time = 0
            # Initialize players
            player = Player()
            minotaur = Minotaur()
            player_positions = [player.position]
            minotaur_positions = [minotaur.position]
            u_star, a_star, states = bellman(max_t)
            probabilities.append(u_star[states[(player.position, minotaur.position)], time])
            while time < max_t - 1:
                player.position = a_star[states[(player.position, minotaur.position)], time]
                player_positions.append(player.position)
                if player.position != (EXIT or minotaur.position):
                    minotaur.position = minotaur.generate_move()
                minotaur_positions.append(minotaur.position)
                time += 1
            if PLOTTING:
                for i in range(len(player_positions)):
                    plot_grid(player_positions[i], minotaur_positions[i])
                    if player_positions[i] == minotaur_positions[i]:
                        print("Game Over!")
                        exit(0)
                    elif player_positions[i] == EXIT:
                        print("Winner!")
                        print("Won after " + str(i) + " actions.")
                        exit(0)
                print("You did not reach the exit in time!")
        plt.plot(range(1, len(probabilities) + 1), probabilities)
        plt.title("Probability vs Maximum Time")
        plt.xticks(np.arange(1, len(probabilities) + 1))
        plt.xlabel("Maximum Time")
        plt.ylabel("Probability")
        plt.show()
    
    else:
        # Discount Factor 
        mean = 30
        gamma = (mean - 1) / mean
        # Accuracy treshold 
        epsilon = 0.0001
        # Tolerance error
        tolerance = (1 - gamma) * epsilon / gamma
        # Define all states
        possible_states, states = get_possible_states()
        # Compute Value Iteration (VI)
        best_policy = value_iteration(possible_states, states, gamma, tolerance)
        # Use same policy over 10,000 games
        simulations = 10 ** 4
        life_distribution = np.random.geometric(p=1/mean, size=simulations)
        games_outcome = {
            "win": 0,
            "eaten": 0,
            "times up": 0,
            "win distribution": [],
            "loss distribution": []
        }

        for life_time in life_distribution:
            # Initialize players
            player = Player()
            minotaur = Minotaur()
            # Define final position
            terminal_state = False
            for _ in range(life_time):
                # Get best movement based on the state (player, minotaur)
                player.position = best_policy[states[(player.position, minotaur.position)]][0]
                if player.position != (EXIT or minotaur.position):
                    # Generate new movement if game if not over
                    minotaur.position = minotaur.generate_move()
                elif player.position == minotaur.position:
                    games_outcome["eaten"] += 1
                    terminal_state = True
                    games_outcome["loss distribution"].append(life_time)
                    break
                elif player.position == EXIT:
                    games_outcome["win"] += 1
                    terminal_state = True
                    games_outcome["win distribution"].append(life_time)
                    break
            if not terminal_state:
                games_outcome["times up"] += 1
                games_outcome["loss distribution"].append(life_time)
            
        print("All games ended")
        print("Ratio (Not Finished)/(Won) :", games_outcome["times up"]/games_outcome["win"])

        plt.hist(games_outcome["win distribution"], color="b", edgecolor="black",
                 bins=int(len(set(games_outcome["win distribution"]))/5), label="Win")
        plt.hist(games_outcome["loss distribution"], color="r", edgecolor="black",
                 bins=int(len(set(games_outcome["loss distribution"]))/5), label="Loss")
        plt.title("Wins vs Losses split by maximum time")
        plt.legend()
        plt.xlabel("Maximum time")
        plt.ylabel("Frequency")
        plt.show()


if __name__ == "__main__":
    main()
