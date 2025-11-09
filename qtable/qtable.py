import numpy as np
import random
import sys
import time

epsilon_min = 0.05
epsilon_decay = 0.995
alpha = 0.1
gamma = 0.9

reward_goal = 100 
reward_wall = -1
reward_step = -0.1
reward_visited = -0.1

def load_maze(file):
    maze = []
    with open(file) as f:
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

def step_env(maze, state, action, end, visited):
    dx, dy = action_vectors[action]
    x, y = state
    nx, ny = x + dx, y + dy

    if nx < 0 or ny < 0 or nx >= maze.shape[0] or ny >= maze.shape[1] or maze[nx, ny] == 1:
        nx, ny = x, y
        reward = reward_wall
    elif (nx, ny) == end:
        reward = reward_goal
    else:
        reward = reward_step
        if (nx, ny) in visited:
            reward += reward_visited
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python qtable.py maze.txt")
        sys.exit(1)

    maze_file = sys.argv[1]
    maze, start, end = load_maze(maze_file)

    maze_height, maze_width = maze.shape
    total_cells = maze_height * maze_width
    max_steps = total_cells * 8

    if maze_height > 20 or maze_width > 20:
        epsilon_start = 0.5
    elif maze_height > 14 or maze_width > 14:
        epsilon_start = 0.3
    elif maze_height > 10 or maze_width > 10:
        epsilon_start = 0.2
    else:
        epsilon_start = 0.1

    epsilon = epsilon_start
    last_results = []
    last_rewards = []
    win_window = 100
    reward_window = 50

    start_training_time = time.time()
    episode = 0
    best_reward_overall = -float('inf')

    while True:
        episode += 1
        state = start
        total_reward = 0
        start_time = time.time()
        visited = set()
        visited.add(state)

        for step_num in range(max_steps):
            action = choose_action(state, epsilon)
            next_state, reward, done = step_env(maze, state, action, end, visited)
            old_q = get_q(state, action)
            next_max = max([get_q(next_state, a) for a in ACTIONS])
            new_q = old_q + alpha * (reward + gamma * next_max - old_q)
            set_q(state, action, new_q)
            state = next_state
            visited.add(state)
            total_reward += reward
            if done:
                break

        steps_this_episode = step_num + 1
        episode_time = time.time() - start_time
        result = "WIN" if state == end else "LOSE"

        print(f"Episode {episode} | Reward: {total_reward:.2f} | Steps: {steps_this_episode} | "
              f"Epsilon: {epsilon:.2f} | Result: {result} | Time: {episode_time:.5f}s")

        last_results.append(result == "WIN")
        if len(last_results) > win_window:
            last_results.pop(0)

        last_rewards.append(total_reward)
        if len(last_rewards) > reward_window:
            last_rewards.pop(0)

        max_reward_last_50 = max(last_rewards) if last_rewards else -float('inf')
        higher_reward_in_last_50 = max_reward_last_50 > best_reward_overall
        if total_reward > best_reward_overall:
            best_reward_overall = total_reward

        if (epsilon <= epsilon_min and
            len(last_results) == win_window and all(last_results) and
            not higher_reward_in_last_50):
            print(f"\nConverged at episode {episode} — all stopping conditions met")
            break

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    total_training_time = time.time() - start_training_time
    print(f"\nTotal training time: {total_training_time:.4f}s")

    state = start
    epsilon = 0
    path = [state]
    visited = set()
    visited.add(state)

    while state != end:
        action = choose_action(state, epsilon)
        state, _, done = step_env(maze, state, action, end, visited)
        if state in visited:
            print("Loop detected — stopping evaluation...")
            break
        path.append(state)
        visited.add(state)

    path_clean = [(int(x), int(y)) for x, y in path]
    num_steps = len(path_clean) - 1
    maze_solution = print_maze(maze, path=path_clean, start=start, end=end)

    print(f"\nLearned path: {path_clean}")
    print(f"Path length: {num_steps} steps")
    print("Maze solution:")
    print(maze_solution)
