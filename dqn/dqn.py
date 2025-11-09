import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame
import time
from collections import deque

REWARDS = {
    'step': -0.01,
    'wall': -5.0,
    'goal': 100.0,
    'explore': 0.5,
}

CONFIG = {
    'episodes': 2000,
    'learning_rate': 1e-3,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay_episodes': 1500,
    'batch_size': 64,
    'memory_size': 20000,
    'target_update': 10,
    'print_every': 25,
    'visualize': True,
    'display_delay': 0.0,
    'early_stopping': True,
    'early_stopping_patience': 100,
    'early_stopping_threshold': 1.0,
    'min_episodes': 500,
    'max_steps_multiplier': 3,
    'distance_reward_scale': 2.0,
}

def load_maze(filename="maze.txt"):
    maze = []
    start = None
    goal = None
    with open(filename, 'r') as f:
        for r, line in enumerate(f):
            row = list(map(int, line.strip().split()))
            for c, val in enumerate(row):
                if val == 3:
                    start = (r, c)
                elif val == 4:
                    goal = (r, c)
            maze.append(row)
    return np.array(maze), start, goal

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.bool_),
        )
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )
    def forward(self, x):
        return self.net(x)

class MazeEnv:
    def __init__(self, maze, start, goal, max_steps_multiplier=3, distance_reward_scale=2.0):
        self.maze = maze
        self.start = start
        self.goal = goal
        self.pos = start
        self.steps = 0
        self.path = []
        self.visited = set()
        self.state_size = 8
        self.action_size = 4
        self.distance_reward_scale = distance_reward_scale
        self.max_steps = int(maze.shape[0] * maze.shape[1] * max_steps_multiplier)
    def reset(self):
        self.pos = self.start
        self.steps = 0
        self.path = [self.start]
        self.visited = {self.start}
        return self._get_state()
    def _get_state(self):
        r, c = self.pos
        gr, gc = self.goal
        norm_r = r / self.maze.shape[0]
        norm_c = c / self.maze.shape[1]
        norm_gr = gr / self.maze.shape[0]
        norm_gc = gc / self.maze.shape[1]
        dist_r = (gr - r) / self.maze.shape[0]
        dist_c = (gc - c) / self.maze.shape[1]
        walls_count = 0
        open_count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.maze.shape[0] and 0 <= nc < self.maze.shape[1]):
                if self.maze[nr, nc] == 1:
                    walls_count += 1
                else:
                    open_count += 1
        walls_ratio = walls_count / 4.0
        open_ratio = open_count / 4.0
        return np.array([norm_r, norm_c, norm_gr, norm_gc, dist_r, dist_c, walls_ratio, open_ratio], dtype=np.float32)
    def step(self, action):
        r, c = self.pos
        gr, gc = self.goal
        prev_distance = abs(gr - r) + abs(gc - c)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc
        if (0 <= nr < self.maze.shape[0] and 0 <= nc < self.maze.shape[1] and self.maze[nr, nc] != 1):
            self.pos = (nr, nc)
            self.path.append(self.pos)
            is_new_exploration = self.pos not in self.visited
            if is_new_exploration:
                self.visited.add(self.pos)
            if self.pos == self.goal:
                reward = REWARDS['goal']
                done = True
            else:
                reward = REWARDS['step']
                if is_new_exploration:
                    reward += REWARDS['explore']
                done = False
            new_distance = abs(gr - nr) + abs(gc - nc)
            distance_delta = prev_distance - new_distance
            reward += self.distance_reward_scale * distance_delta
        else:
            reward = REWARDS['wall']
            done = False
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        return self._get_state(), reward, done
    def get_path_length(self):
        return len(set(self.path))

class PygameVisualizer:
    def __init__(self, maze, start, goal):
        pygame.init()
        self.cell_size = 30
        self.maze = maze
        self.start = start
        self.goal = goal
        self.maze_width = maze.shape[1] * self.cell_size
        self.maze_height = maze.shape[0] * self.cell_size
        self.metrics_width = 250
        self.width = self.maze_width + self.metrics_width
        self.height = self.maze_height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("DQN Maze Solver Training")
        pygame.font.init()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.DARK_GREEN = (0, 100, 0)
        self.GRAY = (128, 128, 128)
        self.YELLOW = (255, 255, 0)
        self.draw_maze()
    def draw_maze(self):
        self.screen.fill(self.BLACK)
        metrics_x = self.maze_width
        pygame.draw.rect(self.screen, (20, 20, 20), (metrics_x, 0, self.metrics_width, self.height))
        for r in range(self.maze.shape[0]):
            for c in range(self.maze.shape[1]):
                x = c * self.cell_size
                y = r * self.cell_size
                if self.maze[r, c] == 1:
                    pygame.draw.rect(self.screen, self.GREEN, (x, y, self.cell_size, self.cell_size))
                else:
                    pygame.draw.rect(self.screen, self.DARK_GREEN, (x, y, self.cell_size, self.cell_size))
        sx, sy = self.start[1] * self.cell_size, self.start[0] * self.cell_size
        pygame.draw.rect(self.screen, self.BLUE, (sx, sy, self.cell_size, self.cell_size))
        gx, gy = self.goal[1] * self.cell_size, self.goal[0] * self.cell_size
        pygame.draw.rect(self.screen, self.RED, (gx, gy, self.cell_size, self.cell_size))
    def draw_metrics(self, episode, success_rate, current_reward, current_steps, current_length, elapsed_time):
        metrics_x = self.width - 240
        metrics_start_y = self.height - 200
        episode_text = self.font_medium.render(f"Episode: {episode}", True, self.YELLOW)
        self.screen.blit(episode_text, (metrics_x, metrics_start_y))
        metrics_start_y += 30
        success_text = self.font_medium.render(f"Success Rate: {success_rate:.1f}%", True, self.WHITE)
        self.screen.blit(success_text, (metrics_x, metrics_start_y))
        metrics_start_y += 30
        reward_text = self.font_medium.render(f"Reward: {current_reward:.1f}", True, self.WHITE)
        self.screen.blit(reward_text, (metrics_x, metrics_start_y))
        metrics_start_y += 30
        steps_text = self.font_medium.render(f"Steps: {current_steps}", True, self.WHITE)
        self.screen.blit(steps_text, (metrics_x, metrics_start_y))
        metrics_start_y += 30
        length_text = self.font_medium.render(f"Length: {current_length}", True, self.WHITE)
        self.screen.blit(length_text, (metrics_x, metrics_start_y))
        metrics_start_y += 30
        time_text = self.font_medium.render(f"Time: {elapsed_time:.1f}s", True, self.WHITE)
        self.screen.blit(time_text, (metrics_x, metrics_start_y))
    def update(self, path, current_pos, episode=None, success_rate=None, current_reward=None, current_steps=None, current_length=None, elapsed_time=None, delay=0.0):
        self.draw_maze()
        for pos in path[:-1]:
            x = pos[1] * self.cell_size + self.cell_size // 4
            y = pos[0] * self.cell_size + self.cell_size // 4
            pygame.draw.rect(self.screen, self.BLUE, (x, y, self.cell_size // 2, self.cell_size // 2))
        if current_pos:
            x = current_pos[1] * self.cell_size + self.cell_size // 8
            y = current_pos[0] * self.cell_size + self.cell_size // 8
            pygame.draw.rect(self.screen, self.WHITE, (x, y, self.cell_size * 3 // 4, self.cell_size * 3 // 4))
        if (episode is not None and success_rate is not None and current_reward is not None and current_steps is not None and current_length is not None and elapsed_time is not None):
            self.draw_metrics(episode, success_rate, current_reward, current_steps, current_length, elapsed_time)
        pygame.display.flip()
        if delay > 0:
            time.sleep(delay)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
    def close(self):
        pygame.quit()

def train_dqn(env, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU detected - using CPU")
    print(f"Using device: {device}\n")
    policy_net = DQN(env.state_size, env.action_size).to(device)
    target_net = DQN(env.state_size, env.action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=config['learning_rate'])
    memory = ReplayMemory(config['memory_size'])
    rewards_history = []
    success_history = []
    steps_history = []
    length_history = []
    episode_times = []
    visualizer = PygameVisualizer(env.maze, env.start, env.goal) if config['visualize'] else None
    print("="*80)
    print("STARTING DQN TRAINING")
    print("="*80)
    print(f"Max Episodes: {config['episodes']}")
    print(f"State size: {env.state_size}")
    print(f"Action size: {env.action_size}")
    print(f"Max steps per episode: {env.max_steps}")
    print("="*80 + "\n")
    start_time = time.time()
    for episode in range(1, config['episodes'] + 1):
        ep_start = time.time()
        state = env.reset()
        total_reward = 0
        done = False
        epsilon_decay_episodes = config.get('epsilon_decay_episodes', config['episodes'])
        epsilon_progress = min(1.0, (episode - 1) / max(1, epsilon_decay_episodes))
        epsilon = config['epsilon_end'] + (config['epsilon_start'] - config['epsilon_end']) * (1 - epsilon_progress)
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, env.action_size - 1)
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_t)
                    action = q_values.argmax().item()
            next_state, reward, done = env.step(action)
            total_reward += reward
            memory.push(state, action, reward, next_state, done)
            state = next_state
            if len(memory) >= config['batch_size']:
                states, actions, rewards, next_states, dones = memory.sample(config['batch_size'])
                states_t = torch.from_numpy(states).float().to(device)
                actions_t = torch.from_numpy(actions).long().unsqueeze(1).to(device)
                rewards_t = torch.from_numpy(rewards).float().to(device)
                next_states_t = torch.from_numpy(next_states).float().to(device)
                dones_t = torch.from_numpy(dones.astype(np.float32)).float().to(device)
                current_q = policy_net(states_t).gather(1, actions_t).squeeze(1)
                with torch.no_grad():
                    next_actions = policy_net(next_states_t).argmax(1, keepdim=True)
                    next_q = target_net(next_states_t).gather(1, next_actions).squeeze(1)
                    target_q = rewards_t + config['gamma'] * next_q * (1 - dones_t)
                loss = nn.SmoothL1Loss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
            if visualizer and (env.steps % 50 == 0 or done):
                window = min(50, episode)
                current_success_rate = np.mean(success_history[-window:]) * 100 if len(success_history) > 0 else 0.0
                current_length = env.get_path_length()
                elapsed_time = time.time() - start_time
                if not visualizer.update(env.path, env.pos, episode=episode, success_rate=current_success_rate, current_reward=total_reward, current_steps=env.steps, current_length=current_length, elapsed_time=elapsed_time, delay=config.get('display_delay', 0.0)):
                    visualizer.close()
                    visualizer = None
        if episode % config['target_update'] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        success = 1 if env.pos == env.goal else 0
        rewards_history.append(total_reward)
        success_history.append(success)
        steps_history.append(env.steps)
        length_history.append(env.get_path_length())
        episode_times.append(time.time() - ep_start)
        if config.get('early_stopping', False) and episode >= config.get('min_episodes', 100) and len(success_history) >= config.get('early_stopping_patience', 50):
            patience = config.get('early_stopping_patience', 50)
            threshold = config.get('early_stopping_threshold', 1.0)
            total_successes = sum(success_history)
            if total_successes > 0:
                recent_window = min(patience, len(success_history))
                recent_success_rate = np.mean(success_history[-recent_window:]) * 100
                if len(success_history) >= patience * 2:
                    earlier_window = min(patience, len(success_history) - recent_window)
                    earlier_success_rate = np.mean(success_history[-(recent_window + earlier_window):-recent_window]) * 100
                    improvement = recent_success_rate - earlier_success_rate
                    if improvement < threshold and recent_success_rate > 80:
                        print(f"\n{'='*80}")
                        print("EARLY STOPPING - Converged!")
                        print(f"{'='*80}\n")
                        break
        if episode % config['print_every'] == 0:
            window = min(50, episode)
            avg_reward = np.mean(rewards_history[-window:])
            success_rate = np.mean(success_history[-window:]) * 100
            avg_steps = np.mean(steps_history[-window:])
            avg_length = np.mean(length_history[-window:])
            total_time = time.time() - start_time
            status = "✓" if success else "✗"
            print(f"{status} Ep {episode:4d} | Reward: {avg_reward:7.2f} | Success: {success_rate:5.1f}% | Steps: {avg_steps:6.1f} | Length: {avg_length:5.1f} | ε: {epsilon:.3f} | Time: {total_time:.1f}s")
    if visualizer:
        print("\nClose Pygame window to continue...")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        visualizer.close()
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total Episodes: {episode}")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"\nFinal Success Rate: {np.mean(success_history[-50:]) * 100:.1f}%")
    print(f"Best Path Length: {min([l for l, s in zip(length_history, success_history) if s == 1], default=0)}")
    print("="*80 + "\n")
    return policy_net, rewards_history, success_history, steps_history

def plot_results(rewards, successes, steps, episodes):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    window = 10
    if len(successes) >= window:
        success_rate = np.convolve(successes, np.ones(window)/window, mode='valid') * 100
        episodes_plot = range(window-1, len(successes))
        ax1.plot(episodes_plot, success_rate, 'g-', linewidth=2)
        ax1.fill_between(episodes_plot, 0, success_rate, alpha=0.3, color='green')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    ax2.plot(steps, 'b-', alpha=0.4)
    if len(steps) >= window:
        steps_avg = np.convolve(steps, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(steps)), steps_avg, 'r-', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Steps per Episode')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("✓ Results saved to 'training_results.png'\n")
    plt.show()

def main():
    print("\n" + "="*80)
    print("DQN MAZE SOLVER - FIXED VERSION")
    print("="*80)
    #maze, start, goal = load_maze("../maze_map/perfect_medium_maze.txt")
    maze, start, goal = load_maze("../maze_map/unperfect_medium_maze.txt")
    print(f"Maze shape: {maze.shape}")
    print(f"Start: {start}")
    print(f"Goal: {goal}\n")
    env = MazeEnv(maze, start, goal, max_steps_multiplier=CONFIG['max_steps_multiplier'], distance_reward_scale=CONFIG['distance_reward_scale'])
    model, rewards, successes, steps = train_dqn(env, CONFIG)
    plot_results(rewards, successes, steps, CONFIG['episodes'])

if __name__ == "__main__":
    main()
