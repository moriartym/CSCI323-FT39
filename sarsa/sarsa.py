PRESETS = {
    "small":  dict(max_episodes=10_000, window=100, eps_decay=0.997, alpha=0.20, gamma=0.99, lam=0.90),
    "medium": dict(max_episodes=15_000, window=120, eps_decay=0.997, alpha=0.18, gamma=0.995, lam=0.90),
    "big":    dict(max_episodes=35_000, window=200, eps_decay=0.999, alpha=0.15, gamma=0.995, lam=0.90),
}

import sys
import os, time, random
from collections import deque
from typing import Tuple, List, Optional
import numpy as np

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

def is_cell_coord(rc: Tuple[int,int]) -> bool:
    r, c = rc
    return (r % 2 == 1) and (c % 2 == 1)

def is_path_bool(grid_bool: np.ndarray, rc: Tuple[int,int]) -> bool:
    r, c = rc
    return 0 <= r < grid_bool.shape[0] and 0 <= c < grid_bool.shape[1] and grid_bool[r, c] == False

def bfs_shortest_steps_expanded(grid_bool: np.ndarray, start_rc: Tuple[int,int], goal_rc: Tuple[int,int]):
    H, W = grid_bool.shape
    (sr, sc), (gr, gc) = start_rc, goal_rc
    if not is_path_bool(grid_bool, start_rc) or not is_path_bool(grid_bool, goal_rc): return False, -1
    q = deque([(sr, sc, 0)])
    seen = {(sr, sc)}
    while q:
        r, c, d = q.popleft()
        if (r, c) == (gr, gc): return True, d
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen and grid_bool[nr, nc] == False:
                seen.add((nr, nc))
                q.append((nr, nc, d + 1))
    return False, -1

def first_valid_cell_expanded(grid_bool: np.ndarray) -> Tuple[int,int]:
    H, W = grid_bool.shape
    for r in range(1,H,2):
        for c in range(1,W,2):
            if not grid_bool[r,c]: return (r,c)
    return (1,1)

def snap_to_nearest_path_expanded(grid_bool: np.ndarray, rc: Tuple[int,int], max_radius: int = 8) -> Tuple[int,int]:
    r, c = rc
    if 0 <= r < grid_bool.shape[0] and 0 <= c < grid_bool.shape[1]:
        if is_cell_coord(rc) and not grid_bool[r, c]: return rc
    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_bool.shape[0] and 0 <= nc < grid_bool.shape[1]:
                    if is_cell_coord((nr, nc)) and not grid_bool[nr, nc]:
                        return (nr, nc)
    return first_valid_cell_expanded(grid_bool)

class MazeEnv:
    ACTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
    N_ACTIONS = 4
    def __init__(self, grid_bool, start, goal, step_penalty=-0.1, wall_penalty=-1, goal_reward=100.0, max_steps=None):
        self.grid, self.start, self.goal = grid_bool, start, goal
        self.step_penalty, self.wall_penalty, self.goal_reward = step_penalty, wall_penalty, goal_reward
        self.max_steps = max_steps or (grid_bool.shape[0]*grid_bool.shape[1])
        self.reset()
    def reset(self):
        self.pos = tuple(self.start)
        self.t = 0
        return self._sid(self.pos)
    def _sid(self,pos): return pos[0]*self.grid.shape[1]+pos[1]
    def _in_bounds(self,r,c): return 0<=r<self.grid.shape[0] and 0<=c<self.grid.shape[1]
    def _is_wall(self,r,c): return self.grid[r,c]
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
        self.pos = (nr,nc)
        if self.pos == tuple(self.goal):
            reward = self.goal_reward
            done = True
        if self.t >= self.max_steps: done = True
        return self._sid(self.pos), reward, done

class SarsaLambdaAgent:
    def __init__(self, n_states, n_actions, alpha=0.2, gamma=0.99, lam=0.9, eps_start=1.0, eps_min=0.05, eps_decay=0.997):
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

def train(env: MazeEnv, agent: SarsaLambdaAgent, max_episodes=10_000, window=100, progress_every=1, patience=50):
    t0 = time.perf_counter()
    success_hist, steps_hist, returns_hist, eps_hist, ep_time_hist = [], [], [], [], []
    best_ma_succ = -1.0
    last_improve_ep = 0
    for ep in range(1, max_episodes+1):
        ep_start = time.perf_counter()
        s = env.reset()
        agent.E[:] = 0.0
        a = agent.policy(s)
        done = False
        steps, G = 0, 0.0
        while not done:
            s_next, r, done = env.step(a)
            a_next = agent.policy(s_next)
            agent.update(s,a,r,s_next,a_next)
            s, a = s_next, a_next
            steps += 1
            G += r
        ep_time = time.perf_counter() - ep_start
        ep_time_hist.append(ep_time)
        success = 1 if env.pos==tuple(env.goal) else 0
        success_hist.append(success)
        steps_hist.append(steps)
        returns_hist.append(G)
        agent.decay_eps()
        eps_hist.append(agent.eps)
        result = "WIN" if success else "LOSE"
        print(f"Episode {ep} | Reward: {G:.2f} | Steps: {steps} | Epsilon: {agent.eps:.2f} | Result: {result} | Time: {ep_time:.4f}s")
        if len(success_hist)>=window:
            sm = float(np.mean(success_hist[-window:]))
            if sm>best_ma_succ: best_ma_succ=sm; last_improve_ep=ep
            elif ep-last_improve_ep>=patience: break
    return dict(
        success_hist=success_hist,
        steps_hist=steps_hist,
        returns_hist=returns_hist,
        eps_hist=eps_hist,
        ep_time_hist=ep_time_hist,
        episodes_trained=len(success_hist),
        final_epsilon=agent.eps,
        total_time=time.perf_counter()-t0
    )

def greedy_rollout(env: MazeEnv, Q: np.ndarray, max_steps=None):
    s = env.reset()
    steps = 0
    path = [tuple(env.start)]
    limit = max_steps or env.max_steps
    for _ in range(limit):
        a = int(np.argmax(Q[s]))
        s, _, done = env.step(a)
        r, c = s//env.grid.shape[1], s%env.grid.shape[1]
        path.append((r,c))
        steps += 1
        if done: return env.pos==tuple(env.goal), steps, path
    return False, steps, path

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 sarsa.py <maze_filename.txt>")
        sys.exit(1)
    MAZE_TXT = sys.argv[1]
    lower = MAZE_TXT.lower()
    arr_int, start, goal = load_txt_intgrid(MAZE_TXT)
    grid_bool = (arr_int==1)
    if start is None: start = (1,1) if is_path_bool(grid_bool,(1,1)) else first_valid_cell_expanded(grid_bool)
    if goal is None:
        H,W = grid_bool.shape
        cells = [(r,c) for r in range(1,H,2) for c in range(1,W,2) if not grid_bool[r,c] and (r,c)!=start]
        random.shuffle(cells); goal=cells[0] if cells else start
    start = snap_to_nearest_path_expanded(grid_bool,start,8)
    goal  = snap_to_nearest_path_expanded(grid_bool,goal,8)
    ok, sp = bfs_shortest_steps_expanded(grid_bool,start,goal)
    assert ok, "Goal not reachable from start (BFS failed)."
    target_steps = int(sp*1.5)
    size_key = "small" if "small" in lower else ("medium" if "medium" in lower else ("big" if "big" in lower else "small"))
    cfg = PRESETS[size_key]
    env = MazeEnv(grid_bool,start,goal)
    n_states = grid_bool.shape[0]*grid_bool.shape[1]
    agent = SarsaLambdaAgent(n_states,MazeEnv.N_ACTIONS,
                             alpha=cfg["alpha"], gamma=cfg["gamma"], lam=cfg["lam"],
                             eps_start=1.0, eps_min=0.05, eps_decay=cfg["eps_decay"])
    logs = train(env,agent,max_episodes=cfg["max_episodes"],window=cfg["window"],progress_every=1,patience=50)
    print(f"\nTraining finished in {logs['total_time']:.2f}s")
    reached, steps, path = greedy_rollout(env,agent.Q)
    print(f"Greedy reached goal: {reached} in {steps -1} steps.")
