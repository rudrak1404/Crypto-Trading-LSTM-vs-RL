"""Process-pool workers for PPO training.

PPO with a small MLP is CPU-bound and single-threaded per fit, so independent
fits (Optuna trials, seeds, walk-forward segments) run in parallel processes.
Functions here must be importable at module top level (Windows uses spawn).
"""
from __future__ import annotations


def init_worker():
    import torch
    torch.set_num_threads(1)


def ppo_fit_job(job: dict) -> dict:
    """job keys: cfg (dict), cost, X, fwd, fit_idx, save_path (optional),
    monitor_path (optional), evals: list of (name, idx) evaluated from flat.
    Returns {name: positions} for each requested evaluation."""
    from rl_agent import PPOStrategy, RLConfig

    s = PPOStrategy(RLConfig(**job["cfg"]), job["cost"])
    s.fit(job["X"], job["fwd"], job["fit_idx"], monitor_path=job.get("monitor_path"))
    if job.get("save_path"):
        s.save(job["save_path"])
    return {name: s.positions(job["X"], job["fwd"], idx) for name, idx in job.get("evals", [])}
