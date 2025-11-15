# Acknowledgement
# Portions of this DQN implementation were created with guidance and support from
# generative AI tools.

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pygame
import time
import math
from collections import deque

REWARDS = {
    'step': -0.1,
    'wall': -1.0,
    'goal': 100.0,
    'explore': 0.0,
    'revisit': -0.1,
}

CONFIG = {
    'learning_rate': 0.0001,
    'gamma': 0.99,
    'epsilon_start': 0.8,
    'epsilon_end': 0.01,
    'epsilon_decay_episodes': 400,
    'batch_size': 64,
    'memory_size': 20000,
    'target_update': 10,
    'visualize': False,
    'display_delay': 0.0,
    'early_stopping': True,
    'early_stopping_patience': 150,
    'early_stopping_threshold': 1.0,
    'early_stopping_success_threshold': 70.0,
    'min_episodes': 100,
    'max_episodes': 3000,
    'max_steps_factor': 8,
    'distance_reward_scale': 0.0,
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
    def __init__(self, maze, start, goal, max_steps_factor=8, distance_reward_scale=1.0):
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
        self.max_steps = int(maze.shape[0] * maze.shape[1] * max_steps_factor)
    
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
            if (0 <= nr < self.maze.shape[0] and 
                0 <= nc < self.maze.shape[1]):
                if self.maze[nr, nc] == 1:
                    walls_count += 1
                else:
                    open_count += 1
        
        walls_ratio = walls_count / 4.0
        open_ratio = open_count / 4.0

        return np.array([norm_r, norm_c, norm_gr, norm_gc, dist_r, dist_c, 
                        walls_ratio, open_ratio], dtype=np.float32)
    
    def step(self, action):
        r, c = self.pos
        gr, gc = self.goal
        prev_distance = abs(gr - r) + abs(gc - c)

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc

        if (0 <= nr < self.maze.shape[0] and 
            0 <= nc < self.maze.shape[1] and 
            self.maze[nr, nc] != 1):
            
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
                else:
                    reward += REWARDS['revisit']
                
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
        return max(0, len(set(self.path)) - 1)


class PygameVisualizer:
    def __init__(self, maze, start, goal):
        pygame.init()
        
        self.maze = maze
        self.start = start
        self.goal = goal
        
        max_maze_width = 1200
        max_maze_height = 800
        
        cell_size_by_width = max_maze_width // maze.shape[1]
        cell_size_by_height = max_maze_height // maze.shape[0]
        self.base_cell_size = min(cell_size_by_width, cell_size_by_height, 50)
        self.base_cell_size = max(self.base_cell_size, 10)
        
        self.zoom_level = 1.0
        self.min_zoom = 0.05
        self.max_zoom = 3.0
        self.zoom_step = 0.1
        
        self.offset_x = 0
        self.offset_y = 0
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        
        self.metrics_width = 300
        self.screen_width = 1450 
        self.screen_height = 800
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("DQN Maze Solver Training - Scroll to Zoom, Drag to Pan")
        
        pygame.font.init()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.DARK_GREEN = (0, 100, 0)
        self.GRAY = (128, 128, 128)
        self.YELLOW = (255, 255, 0)
        self.WALL_COLOR = (0, 255, 0)
        
        self.maze_surface = pygame.Surface((self.screen_width - self.metrics_width, self.screen_height))
        
    @property
    def cell_size(self):
        return int(self.base_cell_size * self.zoom_level)
    
    def draw_maze(self):
        self.maze_surface.fill(self.BLACK)
        
        cell_size = self.cell_size
        wall_thickness = max(1, cell_size // 5)
        
        for r in range(self.maze.shape[0]):
            for c in range(self.maze.shape[1]):
                x = c * cell_size + self.offset_x
                y = r * cell_size + self.offset_y
                
                if (x + cell_size < 0 or x > self.maze_surface.get_width() or
                    y + cell_size < 0 or y > self.maze_surface.get_height()):
                    continue
                
                if self.maze[r, c] == 1:
                    pygame.draw.rect(self.maze_surface, self.WALL_COLOR, 
                                   (x, y, cell_size, cell_size))
                else:
                    pygame.draw.rect(self.maze_surface, self.DARK_GREEN, 
                                   (x, y, cell_size, cell_size))
        
        sx = self.start[1] * cell_size + self.offset_x
        sy = self.start[0] * cell_size + self.offset_y
        pygame.draw.rect(self.maze_surface, self.BLUE, 
                        (sx, sy, cell_size, cell_size))
        
        gx = self.goal[1] * cell_size + self.offset_x
        gy = self.goal[0] * cell_size + self.offset_y
        pygame.draw.rect(self.maze_surface, self.RED, 
                        (gx, gy, cell_size, cell_size))
    
    def draw_metrics(self, episode, success_rate, current_reward, current_steps, 
                    current_length, elapsed_time):
        metrics_x = self.screen_width - self.metrics_width + 10
        y = 20
        line_height = 35
        
        title = self.font_large.render("Training Stats", True, self.YELLOW)
        self.screen.blit(title, (metrics_x, y))
        y += line_height + 10
        
        stats = [
            f"Episode: {episode}",
            f"Success: {success_rate:.1f}%",
            f"Reward: {current_reward:.1f}",
            f"Steps: {current_steps}",
            f"Length: {current_length}",
            f"Time: {elapsed_time:.1f}s",
            "",
            f"Zoom: {self.zoom_level:.1f}x",
        ]
        
        for stat in stats:
            if stat:
                text = self.font_medium.render(stat, True, self.WHITE)
                self.screen.blit(text, (metrics_x, y))
            y += line_height
        
        y += 20
        controls_title = self.font_medium.render("Controls:", True, self.YELLOW)
        self.screen.blit(controls_title, (metrics_x, y))
        y += line_height
        
        controls = [
            "Scroll: Zoom",
            "Drag: Pan",
            "R: Reset View",
        ]
        
        for control in controls:
            text = self.font_small.render(control, True, self.GRAY)
            self.screen.blit(text, (metrics_x, y))
            y += 25
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEWHEEL:
                old_zoom = self.zoom_level
                if event.y > 0:
                    self.zoom_level = min(self.max_zoom, self.zoom_level + self.zoom_step)
                else: 
                    self.zoom_level = max(self.min_zoom, self.zoom_level - self.zoom_step)
                
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if mouse_x < self.screen_width - self.metrics_width:
                    zoom_factor = self.zoom_level / old_zoom
                    self.offset_x = mouse_x - (mouse_x - self.offset_x) * zoom_factor
                    self.offset_y = mouse_y - (mouse_y - self.offset_y) * zoom_factor
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if mouse_x < self.screen_width - self.metrics_width:
                        self.dragging = True
                        self.last_mouse_pos = pygame.mouse.get_pos()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    dx = mouse_x - self.last_mouse_pos[0]
                    dy = mouse_y - self.last_mouse_pos[1]
                    self.offset_x += dx
                    self.offset_y += dy
                    self.last_mouse_pos = (mouse_x, mouse_y)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.zoom_level = 1.0
                    self.offset_x = 0
                    self.offset_y = 0
        
        return True
    
    def update(self, path, current_pos, episode=None, success_rate=None, 
               current_reward=None, current_steps=None, current_length=None, 
               elapsed_time=None, delay=0.0):
        if not self.handle_events():
            return False
        
        self.screen.fill(self.BLACK)
        
        metrics_rect = pygame.Rect(self.screen_width - self.metrics_width, 0, 
                                   self.metrics_width, self.screen_height)
        pygame.draw.rect(self.screen, (20, 20, 20), metrics_rect)
        
        self.draw_maze()
        
        cell_size = self.cell_size
        path_size = max(2, cell_size // 3)
        
        for pos in path[:-1]:
            x = pos[1] * cell_size + self.offset_x + (cell_size - path_size) // 2
            y = pos[0] * cell_size + self.offset_y + (cell_size - path_size) // 2
            
            if (x + path_size >= 0 and x < self.maze_surface.get_width() and
                y + path_size >= 0 and y < self.maze_surface.get_height()):
                pygame.draw.rect(self.maze_surface, self.BLUE, 
                               (x, y, path_size, path_size))
        
        if current_pos:
            agent_size = max(3, cell_size // 2)
            x = current_pos[1] * cell_size + self.offset_x + (cell_size - agent_size) // 2
            y = current_pos[0] * cell_size + self.offset_y + (cell_size - agent_size) // 2
            
            if (x + agent_size >= 0 and x < self.maze_surface.get_width() and
                y + agent_size >= 0 and y < self.maze_surface.get_height()):
                pygame.draw.rect(self.maze_surface, self.WHITE, 
                               (x, y, agent_size, agent_size))
        
        self.screen.blit(self.maze_surface, (0, 0))
        
        if (episode is not None and success_rate is not None and 
            current_reward is not None and current_steps is not None and 
            current_length is not None and elapsed_time is not None):
            self.draw_metrics(episode, success_rate, current_reward, 
                            current_steps, current_length, elapsed_time)
        
        pygame.display.flip()
        
        if delay > 0:
            time.sleep(delay)
        
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
    epsilon_history = []
    decile_stats = []
    
    def safe_mean(values):
        return float(np.mean(values)) if len(values) > 0 else 0.0
    
    visualizer = PygameVisualizer(env.maze, env.start, env.goal) if config.get('visualize') else None
    
    max_episodes = config.get('max_episodes')
    target_label = "∞" if not max_episodes else str(max_episodes)
    
    print("="*80)
    print("STARTING DQN TRAINING")
    print("="*80)
    print(f"Target Episodes: {target_label} (auto-stop on plateau)")
    print(f"State size: {env.state_size}")
    print(f"Action size: {env.action_size}")
    print(f"Max steps per episode: {env.max_steps}")
    print(f"Visualization: {'ON' if visualizer else 'OFF'}")
    print("="*80 + "\n")
    
    start_time = time.time()
    episode = 0
    stop_reason = None
    
    while True:
        episode += 1
        ep_start = time.time()
        
        state = env.reset()
        total_reward = 0
        done = False
        
        epsilon_decay_episodes = config.get('epsilon_decay_episodes', max(episode, 1))
        epsilon_progress = min(1.0, (episode - 1) / max(1, epsilon_decay_episodes))
        epsilon = (
            config['epsilon_end']
            + (config['epsilon_start'] - config['epsilon_end']) * (1 - epsilon_progress)
        )
        
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
                
                if not visualizer.update(env.path, env.pos, 
                                        episode=episode,
                                        success_rate=current_success_rate,
                                        current_reward=total_reward,
                                        current_steps=env.steps,
                                        current_length=current_length,
                                        elapsed_time=elapsed_time,
                                        delay=config.get('display_delay', 0.0)):
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
        epsilon_history.append(epsilon)
        
        if episode % 10 == 0:
            block_slice = slice(max(0, episode - 10), episode)
            avg_reward = safe_mean(rewards_history[block_slice])
            success_rate = safe_mean(success_history[block_slice]) * 100.0
            avg_steps = safe_mean(steps_history[block_slice])
            avg_length = safe_mean(length_history[block_slice])
            avg_epsilon = safe_mean(epsilon_history[block_slice])
            avg_time = safe_mean(episode_times[block_slice])
            block_duration = sum(episode_times[block_slice])
            
            status = "✓" if success_history[episode - 1] else "✗"
            
            print(f"{status} Ep {episode:4d} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Steps: {avg_steps:6.1f} | "
                  f"Length: {avg_length:5.1f} | "
                  f"ε: {avg_epsilon:6.3f} | "
                  f"Time: {avg_time:5.2f}s | "
                  f"Duration: {block_duration:6.2f}s")
        
        # Early stopping check
        if config.get('early_stopping', False) and episode >= config.get('min_episodes', 0):
            patience = config.get('early_stopping_patience', 50)
            threshold = config.get('early_stopping_threshold', 1.0)
            success_threshold = config.get('early_stopping_success_threshold', 80.0)
            
            if len(success_history) >= patience:
                recent_window = patience
                recent_success_rate = np.mean(success_history[-recent_window:]) * 100
                
                if len(success_history) >= patience * 2:
                    earlier_window = patience
                    earlier_success_rate = np.mean(success_history[-(recent_window + earlier_window):-recent_window]) * 100
                else:
                    earlier_success_rate = np.mean(success_history[:-recent_window]) * 100 if len(success_history) > recent_window else 0.0
                
                improvement = recent_success_rate - earlier_success_rate
                
                if improvement < threshold and recent_success_rate >= success_threshold:
                    stop_reason = (f"Plateau detected: ΔSuccess {improvement:.2f}% < "
                                   f"{threshold}% with recent success {recent_success_rate:.1f}%")
        
        if max_episodes and episode >= max_episodes:
            stop_reason = f"Reached max episodes limit ({max_episodes})"
        
        if stop_reason:
            print(f"\n{'='*80}")
            print(f"EARLY STOPPING - {stop_reason}")
            print(f"{'='*80}\n")
            break
    
    if visualizer:
        print("\nClose Pygame window to continue...")
        running = True
        while running:
            running = visualizer.handle_events()
        visualizer.close()
    
    total_time = time.time() - start_time
    total_episodes = episode
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    final_success_window = success_history[-min(50, len(success_history)):]
    print(f"\nFinal Success Rate (last {len(final_success_window)} eps): {safe_mean(final_success_window) * 100:.1f}%")
    best_success_lengths = [l for l, s in zip(length_history, success_history) if s == 1]
    print(f"Best Path Length: {min(best_success_lengths) if best_success_lengths else 0}")
    print("="*80 + "\n")
    
    print("SUMMARY BY TRAINING PROGRESS (10% BUCKETS)")
    if total_episodes >= 1:
        decile_edges = [math.ceil(total_episodes * i / 10) for i in range(1, 11)]
        prev_idx = 0
        for idx, edge in enumerate(decile_edges):
            edge = min(edge, total_episodes)
            if edge <= prev_idx:
                continue
            block_slice = slice(prev_idx, edge)
            avg_reward = safe_mean(rewards_history[block_slice])
            success_rate = safe_mean(success_history[block_slice]) * 100.0
            avg_steps = safe_mean(steps_history[block_slice])
            avg_length = safe_mean(length_history[block_slice])
            avg_time = safe_mean(episode_times[block_slice])
            block_duration = sum(episode_times[block_slice])
            avg_epsilon = safe_mean(epsilon_history[block_slice])
            
            start_percent = idx * 10
            end_percent = (idx + 1) * 10
            episodes_label = f"Episodes {prev_idx + 1}-{edge}"
            progress_label = f"{start_percent}-{end_percent}%"
            
            print(f"[{progress_label}] {episodes_label} | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Steps: {avg_steps:6.1f} | "
                  f"Length: {avg_length:5.1f} | "
                  f"ε: {avg_epsilon:6.3f} | "
                  f"Time/Ep: {avg_time:5.2f}s | "
                  f"Duration: {block_duration:6.2f}s")
            
            decile_stats.append({
                'percent_range': (start_percent, end_percent),
                'episode_range': (prev_idx + 1, edge),
                'avg_reward': avg_reward,
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'avg_length': avg_length,
                'avg_time': avg_time,
                'duration': block_duration,
                'avg_epsilon': avg_epsilon,
            })
            
            prev_idx = edge
    print("="*80 + "\n")
    
    return policy_net, rewards_history, success_history, steps_history, decile_stats


def plot_results(decile_stats):
    if not decile_stats:
        print("No decile statistics recorded; skipping plotting.\n")
        return

    percent_points = [stats['percent_range'][1] for stats in decile_stats]
    avg_rewards = [stats['avg_reward'] for stats in decile_stats]
    success_rates = [stats['success_rate'] for stats in decile_stats]
    avg_steps = [stats['avg_steps'] for stats in decile_stats]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(percent_points, avg_rewards, marker='o', color='skyblue')
    axes[0].set_title('Average Reward by Training Progress')
    axes[0].set_xlabel('Training Progress (%)')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_xticks(list(range(10, 101, 10)))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(percent_points, success_rates, marker='o', color='green')
    axes[1].set_title('Success Rate by Training Progress')
    axes[1].set_xlabel('Training Progress (%)')
    axes[1].set_ylabel('Success Rate (%)')
    axes[1].set_xticks(list(range(10, 101, 10)))
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(percent_points, avg_steps, marker='o', color='orange')
    axes[2].set_title('Average Steps by Training Progress')
    axes[2].set_xlabel('Training Progress (%)')
    axes[2].set_ylabel('Average Steps')
    axes[2].set_xticks(list(range(10, 101, 10)))
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    print("✓ Results saved to 'training_results.png'\n")
    plt.show()


def evaluate_model(env, model, num_episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    success_count = 0
    path_lengths = []
    
    print("\n" + "="*80)
    print("EVALUATING TRAINED MODEL (ε=0 - Pure Exploitation)")
    print("="*80)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = model(state_t)
                action = q_values.argmax().item()
            
            state, reward, done = env.step(action)
        
        if env.pos == env.goal:
            success_count += 1
            path_lengths.append(env.get_path_length())
        
        if (episode + 1) % 10 == 0:
            current_success_rate = (success_count / (episode + 1)) * 100
            print(f"Episode {episode + 1}/{num_episodes} - Success Rate: {current_success_rate:.1f}%")
    
    success_rate = (success_count / num_episodes) * 100
    avg_path_length = np.mean(path_lengths) if path_lengths else 0
    
    print("="*80)
    print(f"EVALUATION RESULTS:")
    print(f"Success Rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
    print(f"Average Path Length: {avg_path_length:.1f}")
    print(f"Best Path Length: {min(path_lengths) if path_lengths else 0}")
    print("="*80 + "\n")
    
    return success_rate, avg_path_length


def main():
    print("\n" + "="*80)
    print("DQN MAZE SOLVER - WITH VISUALIZATION")
    print("="*80)
    
    maze, start, goal = load_maze("maze_map/perfect_small_maze.txt")
    print(f"Maze shape: {maze.shape}")
    print(f"Start: {start}")
    print(f"Goal: {goal}\n")
    
    env = MazeEnv(maze, start, goal,
                max_steps_factor=CONFIG['max_steps_factor'],
                distance_reward_scale=CONFIG['distance_reward_scale'])
    
    model, rewards, successes, steps, decile_stats = train_dqn(env, CONFIG)
    
    plot_results(decile_stats)
    
    evaluate_model(env, model, num_episodes=100)


if __name__ == "__main__":
    main()
