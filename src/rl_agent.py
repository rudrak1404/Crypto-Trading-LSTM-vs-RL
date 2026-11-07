"""Custom Gymnasium trading environment + PPO agent (Stable-Baselines3).

State: last `window` hours of standardised features, flattened, plus the
current position. Action: target position (flat/long, or short/flat/long).
Reward: log growth of the portfolio over the next hour, net of transaction
costs, scaled by 100, minus an optional downside penalty.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from backtest import metrics


@dataclass
class RLConfig:
    window: int = 24
    episode_len: int = 720          # 30 days of hourly bars per training episode
    timesteps: int = 200_000
    lr: float = 3e-4
    n_steps: int = 2048
    batch: int = 64
    gamma: float = 0.99
    ent_coef: float = 0.0
    net: int = 64
    downside_penalty: float = 0.0
    allow_short: bool = False
    decide_every: int = 1        # each action holds for N hours
    n_envs: int = 4
    seed: int = 0


class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, X, fwd_ret, idx, window, cost, episode_len=None,
                 allow_short=False, downside_penalty=0.0, seed=None, decide_every=1):
        super().__init__()
        self.every = decide_every
        self.X, self.idx = X, idx[idx >= window - 1]
        self.simple = np.expm1(fwd_ret)
        self.window, self.cost = window, cost
        self.episode_len = episode_len
        self.levels = np.array([-1.0, 0.0, 1.0] if allow_short else [0.0, 1.0])
        self.penalty = downside_penalty
        self.action_space = spaces.Discrete(len(self.levels))
        self.observation_space = spaces.Box(-np.inf, np.inf, (window * X.shape[1] + 1,), np.float32)
        self.rng = np.random.default_rng(seed)

    def _obs(self):
        t = self.idx[self.k]
        w = self.X[t - self.window + 1:t + 1].ravel()
        return np.append(w, self.pos).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.episode_len:
            self.k = int(self.rng.integers(0, len(self.idx) - self.episode_len))
            self.end = self.k + self.episode_len
        else:
            self.k, self.end = 0, len(self.idx)
        # Evaluation can start from the position held at the end of the previous
        # walk-forward segment, so a model hand-over doesn't force a round trip.
        self.pos = float((options or {}).get("initial_pos", 0.0))
        return self._obs(), {}

    def step(self, action):
        """Hold the chosen position for the next `decide_every` bars."""
        new = float(self.levels[action])
        reward, held = 0.0, []
        for _ in range(self.every):
            if self.k >= self.end:
                break
            t = self.idx[self.k]
            r = new * self.simple[t] - self.cost * abs(new - self.pos)
            reward += 100 * np.log1p(r) - self.penalty * 1e4 * min(r, 0.0) ** 2
            self.pos = new
            self.k += 1
            held.append(new)
        done = self.k >= self.end
        obs = self._obs() if not done else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), done, False, {"pos": held}


class PPOStrategy:
    def __init__(self, cfg: RLConfig, cost: float):
        self.cfg, self.cost, self.model = cfg, cost, None

    def fit(self, X, fwd_ret, train_idx, verbose=0, monitor_path=None):
        c = self.cfg
        # Drop the last train bar: its fwd_ret is the first return of the next split.
        tr = train_idx[:-1]
        envs = DummyVecEnv([
            (lambda i=i: TradingEnv(X, fwd_ret, tr, c.window, self.cost, c.episode_len,
                                    c.allow_short, c.downside_penalty, seed=c.seed * 100 + i,
                                    decide_every=c.decide_every))
            for i in range(c.n_envs)])
        if monitor_path:                      # episode rewards -> CSV (training curves)
            envs = VecMonitor(envs, filename=str(monitor_path))
        self.model = PPO("MlpPolicy", envs, learning_rate=c.lr, n_steps=c.n_steps,
                         batch_size=c.batch, gamma=c.gamma, ent_coef=c.ent_coef,
                         policy_kwargs={"net_arch": [c.net, c.net]},
                         seed=c.seed, verbose=verbose, device="cpu")
        self.model.learn(total_timesteps=c.timesteps)
        return self

    def save(self, path):
        self.model.save(str(path))

    @classmethod
    def load(cls, path, cfg: RLConfig, cost: float):
        s = cls(cfg, cost)
        s.model = PPO.load(str(path), device="cpu")
        return s

    def positions(self, X, fwd_ret, idx, initial_pos: float = 0.0) -> np.ndarray:
        env = TradingEnv(X, fwd_ret, idx, self.cfg.window, self.cost,
                         allow_short=self.cfg.allow_short, decide_every=self.cfg.decide_every)
        assert len(env.idx) == len(idx), "eval split needs `window` bars of history"
        obs, _ = env.reset(options={"initial_pos": initial_pos})
        pos, done = [], False
        while not done:
            a, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(a))
            pos.extend(info["pos"])
        return np.array(pos)

    def evaluate(self, X, fwd_ret, idx) -> dict:
        return metrics(self.positions(X, fwd_ret, idx), fwd_ret[idx], self.cost)


def config_dict(cfg: RLConfig) -> dict:
    return asdict(cfg)
