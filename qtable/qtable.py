import numpy as np
import random
import sys
import time

epsilon = 0.3
epsilon_min = 0.01
alpha = 0.1
gamma = 0.9
epsilon_decay_min = 0.9

def load_maze(file):
    maze = []
    with open(file, "r") as f:
        for line in f:
            maze.append([int(x) for x in line.strip().split()])
    maze = np.array(maze)
    start = tuple(int(i) for i in np.argwhere(maze == 3)[0])
    end = tuple(int(i) for i in np.argwhere(maze == 4)[0])
    return maze, start, end

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
action_vectors = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

Q = {}
def get_q(state, action):
    return Q.get((state, action), 0.0)
def set_q(state, action, value):
    Q[(state, action)] = value

def step_env(maze, state, action, end, goal_reward, step_reward):
    dx, dy = action_vectors[action]
    x, y = state
    nx, ny = x + dx, y + dy
    if nx < 0 or ny < 0 or nx >= maze.shape[0] or ny >= maze.shape[1]:
        nx, ny = x, y
    if maze[nx, ny] == 1:
        nx, ny = x, y
        reward = -1
    elif (nx, ny) == end:
        reward = goal_reward
    else:
        reward = step_reward
    done = (nx, ny) == end
    return (nx, ny), reward, done

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    q_vals = [get_q(state, a) for a in ACTIONS]
    max_q = max(q_vals)
    best_actions = [a for a, q in zip(ACTIONS, q_vals) if q == max_q]
    return random.choice(best_actions)

def print_maze(maze, path=[], start=None, end=None):
    RED = "\033[91m"
    RESET = "\033[0m"
    lines = []
    for i in range(maze.shape[0]):
        line = ""
        for j in range(maze.shape[1]):
            pos = (i, j)
            if start and pos == start:
                line += "S "
            elif end and pos == end:
                line += "E "
            elif pos in path:
                line += f"{RED}*{RESET} "
            elif maze[i,j] == 1:
                line += "# "
            else:
                line += ". "
        lines.append(line)
    return "\n".join(lines) + "\n"

def log(message, file):
    print(message)
    file.write(message + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qtable.py maze.txt")
        sys.exit(1)

    maze_file = sys.argv[1]
    maze, start, end = load_maze(maze_file)

    maze_height, maze_width = maze.shape
    total_cells = maze_height * maze_width
    max_steps = 8 * total_cells
    goal_reward = 100
    step_reward = -0.1

    max_epsilon_decay = 1 - (1 - epsilon_min) / total_cells
    epsilon_decay = max_epsilon_decay
    delta_decay = max_epsilon_decay / total_cells * 0.1301

    log_file_name = "training_log.txt"
    with open(log_file_name, "w") as f:
        log(f"Starting epsilon decay: {epsilon_decay:.6f}", f)

        episode = 0
        epsilon_history = []
        best_solution_steps = float('inf')

        start_training_time = time.time()

        while True:
            episode += 1
            state = start
            total_reward = 0
            start_time = time.time()

            for step_num in range(max_steps):
                action = choose_action(state, epsilon)
                next_state, reward, done = step_env(maze, state, action, end, goal_reward, step_reward)
                old_q = get_q(state, action)
                next_max = max([get_q(next_state, a) for a in ACTIONS])
                new_q = old_q + alpha * (reward + gamma * next_max - old_q)
                set_q(state, action, new_q)
                state = next_state
                total_reward += reward
                if done:
                    break

            steps_this_episode = step_num + 1
            episode_time = time.time() - start_time

            if state == end and steps_this_episode < best_solution_steps:
                best_solution_steps = steps_this_episode
                epsilon_decay = max(epsilon_decay - delta_decay, epsilon_decay_min)
                log(f"🔥 New best solution! Steps: {best_solution_steps}. Decay decreased to {epsilon_decay:.6f}", f)

            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            epsilon_history.append(epsilon)

            result = "WIN" if state == end else "LOSE"
            log(f"Episode {episode} | Reward: {total_reward} | Steps: {steps_this_episode} | "
                f"Epsilon: {epsilon:.3f} | Result: {result} | Decay: {epsilon_decay:.6f} | Time: {episode_time:.4f}s", f)

            if epsilon <= epsilon_min + 1e-5:
                log(f"\nTraining stopped after {episode} episodes because epsilon is low ({epsilon:.3f})", f)
                break

        total_training_time = time.time() - start_training_time
        log(f"\nTotal training time: {total_training_time:.4f}s", f)

        state = start
        epsilon = 0
        path = [state]

        while state != end:
            action = choose_action(state, epsilon)
            state, _, done = step_env(maze, state, action, end, goal_reward, step_reward)
            if state in path:
                log("⚠️ Loop detected — stopping evaluation...", f)
                break
            path.append(state)

        path_clean = [(int(x), int(y)) for x, y in path]
        num_steps = len(path_clean) - 1

        maze_solution = print_maze(maze, path=path_clean, start=start, end=end)

        log(f"\nLearned path: {path_clean}", f)
        log(f"Path length: {num_steps} steps", f)
        log("Maze solution:", f)
        log(maze_solution, f)
