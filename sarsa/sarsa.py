import sys
import random
import numpy as np
import time
from typing import Tuple, Optional

EPSILON_MIN = 0.05
ALPHA = 0.1
GAMMA = 0.9
LAMBDA = 0.9
EPSILON_DECAY = 0.995

REWARD_GOAL = 100
REWARD_WALL = -1
REWARD_STEP = -0.1
REWARD_VISITED = -0.1
MAX_EPISODES = 50000

def load_txt_intgrid(path: str) -> Tuple[np.ndarray, Optional[Tuple[int,int]], Optional[Tuple[int,int]]]:
    rows, start, goal = [], None, None
    with open(path, "r") as f:
        for r, line in enumerate(l.strip() for l in f if l.strip()):
            vals = []
            for c, tok in enumerate(line.split()):
                v = int(tok)
                vals.append(v)
                if v == 3 and start is None: start = (r, c)
                if v == 4 and goal is None: goal = (r, c)
            rows.append(vals)
    return np.array(rows, dtype=int), start, goal

class MazeEnv:
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
    N_ACTIONS = 4
    def __init__(self, grid, start, goal, step_penalty=REWARD_STEP, wall_penalty=REWARD_WALL, goal_reward=REWARD_GOAL):
        self.grid, self.start, self.goal = grid, start, goal
        self.step_penalty, self.wall_penalty, self.goal_reward = step_penalty, wall_penalty, goal_reward
        H, W = grid.shape
        self.max_steps = H * W * 8
        self.reset()

    def reset(self):
        self.pos = tuple(self.start)
        self.t = 0
        self.visited = set()
        self.visited.add(self.pos)
        return self._sid(self.pos)

    def _sid(self,pos):
        return pos[0]*self.grid.shape[1]+pos[1]

    def _in_bounds(self,r,c): return 0<=r<self.grid.shape[0] and 0<=c<self.grid.shape[1]
    def _is_wall(self,r,c): return self.grid[r,c] == 1

    def step(self, action: int):
        dr, dc = MazeEnv.ACTIONS[action]
        r, c = self.pos
        nr, nc = r + dr, c + dc
        self.t += 1
        reward = self.step_penalty
        done = False
        if not self._in_bounds(nr,nc) or self._is_wall(nr,nc):
            nr, nc = r, c
            reward += self.wall_penalty
        if (nr, nc) in self.visited:
            reward += REWARD_VISITED
        self.pos = (nr, nc)
        self.visited.add(self.pos)
        if self.pos == tuple(self.goal):
            reward = self.goal_reward
            done = True
        if self.t >= self.max_steps:
            done = True
        return self._sid(self.pos), reward, done

class SarsaLambdaAgent:
    def __init__(self, n_states, n_actions, alpha=ALPHA, gamma=GAMMA, lam=LAMBDA, eps_start=1.0, eps_min=EPSILON_MIN, eps_decay=0.995):
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.E = np.zeros_like(self.Q)
        self.alpha, self.gamma, self.lam = alpha, gamma, lam
        self.eps, self.eps_min, self.eps_decay = eps_start, eps_min, eps_decay
        self.n_actions = n_actions

    def policy(self, s: int) -> int:
        if np.random.rand() < self.eps: return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[s]))

    def update(self, s,a,r,s_next,a_next):
        td = r + self.gamma*self.Q[s_next,a_next] - self.Q[s,a]
        self.E[s,a] += 1.0
        self.Q += self.alpha*td*self.E
        self.E *= self.gamma*self.lam

    def decay_eps(self): self.eps = max(self.eps_min, self.eps*self.eps_decay)

def train(env: MazeEnv, agent: SarsaLambdaAgent, max_episodes=MAX_EPISODES):
    success_hist, returns_hist = [], []
    last_rewards, last_results = [], []
    best_reward = -float('inf')
    
    for ep in range(1, max_episodes+1):
        s = env.reset()
        agent.E[:] = 0.0
        a = agent.policy(s)
        done = False
        G = 0.0
        steps_path = []

        start_time = time.time()

        while not done:
            s_next, r, done = env.step(a)
            a_next = agent.policy(s_next)
            agent.update(s, a, r, s_next, a_next)
            s, a = s_next, a_next
            G += r
            r_pos, c_pos = s // env.grid.shape[1], s % env.grid.shape[1]
            steps_path.append((r_pos, c_pos))

        episode_time = time.time() - start_time
        agent.decay_eps()
        success = 1 if env.pos == tuple(env.goal) else 0

        success_hist.append(success)
        returns_hist.append(G)

        last_results.append(success == 1)
        if len(last_results) > 100: last_results.pop(0)

        last_rewards.append(G)
        if len(last_rewards) > 50: last_rewards.pop(0)

        higher_reward_in_last_50 = any(r > best_reward for r in last_rewards)
        if G > best_reward: best_reward = G

        print(f"Episode {ep} | Reward: {G:.2f} | Steps: {len(steps_path)} | "
              f"Epsilon: {agent.eps:.2f} | Result: {'WIN' if success else 'LOSE'} | "
              f"Time: {episode_time:.5f}s")

        if (agent.eps <= agent.eps_min and
            len(last_results) == 100 and all(last_results) and
            not higher_reward_in_last_50):
            print(f"Converged at episode {ep}")
            break

    return success_hist, returns_hist, agent

def greedy_rollout(env: MazeEnv, Q: np.ndarray, max_steps=None):
    s = env.reset()
    path = []
    limit = max_steps or env.max_steps
    for _ in range(limit):
        valid_actions = []
        for a, (dr, dc) in enumerate(env.ACTIONS):
            r, c = env.pos
            nr, nc = r+dr, c+dc
            if env._in_bounds(nr,nc) and not env._is_wall(nr,nc):
                valid_actions.append(a)
        if not valid_actions:
            break
        a = max(valid_actions, key=lambda x: Q[s,x])
        s, _, done = env.step(a)
        r, c = s // env.grid.shape[1], s % env.grid.shape[1]
        path.append((r,c))
        if done:
            return True, len(path), path
    return False, len(path), path

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python3 sarsa.py <maze_filename.txt>")
        sys.exit(1)

    MAZE_TXT = sys.argv[1]
    arr_int, start, goal = load_txt_intgrid(MAZE_TXT)
    grid = arr_int

    maze_height, maze_width = grid.shape

    if maze_height > 20 or maze_width > 20:
        epsilon_start = 0.5
    elif maze_height > 14 or maze_width > 14:
        epsilon_start = 0.3
    elif maze_height > 10 or maze_width > 10:
        epsilon_start = 0.2
    else:
        epsilon_start = 0.1

    n_states = maze_height * maze_width
    env = MazeEnv(grid, start, goal)
    agent = SarsaLambdaAgent(n_states, MazeEnv.N_ACTIONS, eps_start=epsilon_start, eps_decay=EPSILON_DECAY)

    start_training = time.time()
    success_hist, returns_hist, agent = train(env, agent, max_episodes=MAX_EPISODES)
    total_training_time = time.time() - start_training

    print(f"\nTraining finished. Total training time: {total_training_time:.5f}s")

    reached, steps, path = greedy_rollout(env, agent.Q)
    print(f"Greedy reached goal: {reached} in {steps} steps.")
    print("Learned path:", path)
