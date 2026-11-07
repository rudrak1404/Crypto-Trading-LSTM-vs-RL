"""Train and evaluate the LSTM and PPO strategies; save every position series.

Protocol (fixed before the test period was evaluated)
  * Splits: train 2019-2023, validation 2024, test 2025-01 .. 2026-10.
  * untuned:   library defaults, hourly decisions, fit once on train.
  * tuned:     Optuna-tuned on validation Sharpe (strategies with < 10% market
               exposure rejected), then refit once on train+val for the test.
  * tuned_wf:  same hyper-parameters, but walk-forward through the test period:
               refit every quarter on all data before that quarter (expanding
               window, scaler refit too), then trade the quarter.
               ** Primary result: tuned_wf seed ensemble. **
  * Each variant runs several seeds; "ens" = seed ensemble (LSTM: mean forecast,
    PPO: majority vote).

All metrics, statistics and plots are computed afterwards by report.py from the
saved positions, so every number in the README is reproducible from results/.

Usage
  python src/run_all.py --part lstm     # GPU            } run in parallel
  python src/run_all.py --part ppo      # CPU processes  }
  python src/run_all.py --part report   # baselines, statistics, plots
  python src/run_all.py --quick         # smoke test of everything
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import optuna
import torch

import data
import features as F
import jobs
from backtest import metrics
from lstm_model import LSTMConfig, LSTMStrategy, permutation_importance
from lstm_model import positions as lstm_positions
from rl_agent import PPOStrategy, RLConfig

ROOT = Path(__file__).resolve().parents[1]
COST = 0.001                       # 0.1% per unit traded (Binance spot taker fee)
THRESHOLDS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
DECIDE_EVERY = [1, 4, 12, 24]      # hours between position decisions
MIN_EXPOSURE = 0.10                # tuned strategies must be in the market >= 10% of the time


def score(m: dict) -> float:
    """Tuning objective: validation Sharpe. A strategy that is almost never in
    the market can post a high Sharpe from a handful of trades -- rejected."""
    return m["sharpe"] if m["exposure"] >= MIN_EXPOSURE else -10.0


class Ctx:
    """Shared data, logging and result storage for one part of the run."""

    def __init__(self, a, part: str):
        self.a = a
        self.out = ROOT / "results" / ("quick" if a.quick else "")
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "models").mkdir(exist_ok=True)
        self.t0 = time.time()
        self.logf = open(self.out / f"run_{part}.log", "w", encoding="utf-8")
        raw = data.load()
        self.log("data:", data.report_gaps(raw))
        self.f = F.build(raw)
        self.sp = F.split_index(self.f)
        self.sp["trainval"] = np.r_[self.sp["train"], self.sp["val"]]
        self.segments = F.walkforward_segments(self.f, self.sp["test"])
        self.fwd = self.f["fwd_ret"].to_numpy()
        self._X, self._y = {}, {}
        self.positions, self.preds, self.meta = {}, {}, {}
        for k in ("train", "val", "test"):
            v = self.sp[k]
            self.log(f"{k:5s}: {len(v):6d} bars  {self.f.index[v[0]]} -> {self.f.index[v[-1]]}")
        self.log(f"walk-forward: {len(self.segments)} segments: " +
                 ", ".join(str(self.f.index[t[0]].date()) for _, t in self.segments))

    def log(self, *m):
        s = f"[{(time.time() - self.t0) / 60:5.1f}m] " + " ".join(str(x) for x in m)
        print(s, flush=True)
        self.logf.write(s + "\n")
        self.logf.flush()

    def X(self, fit_idx: np.ndarray) -> np.ndarray:
        """Feature matrix standardised with statistics of the fit window only."""
        key = (int(fit_idx[0]), int(fit_idx[-1]), len(fit_idx))
        if key not in self._X:
            self._X[key] = F.scaled_matrix(self.f, fit_idx)
        return self._X[key]

    def y(self, h: int) -> np.ndarray:
        if h not in self._y:
            self._y[h] = F.target(self.f, h).to_numpy()
        return self._y[h]

    def put(self, model, variant, seed, split, pos):
        pos = np.asarray(pos, dtype=float)
        assert len(pos) == len(self.sp[split]), (model, variant, seed, split, len(pos))
        self.positions[f"{model}|{variant}|{seed}|{split}"] = pos
        m = metrics(pos, self.fwd[self.sp[split]], COST)
        self.log(f"  {model:5s} {variant:9s} {split:4s} seed={seed!s:3s} sharpe={m['sharpe']:+.2f} "
                 f"ret={m['total_return']:+.1%} mdd={m['max_drawdown']:.1%} "
                 f"exp={m['exposure']:.0%} trades={m['trades']}")

    def save(self, part: str):
        np.savez_compressed(self.out / f"positions_{part}.npz", **self.positions)
        if self.preds:
            np.savez_compressed(self.out / f"preds_{part}.npz", **self.preds)
        with open(self.out / f"meta_{part}.json", "w", encoding="utf-8") as fh:
            json.dump(self.meta, fh, indent=2, default=str)
        self.log(f"saved {len(self.positions)} position series -> {self.out}")


# ================================ LSTM ================================
def run_lstm(c: Ctx):
    a, sp = c.a, c.sp
    torch.set_num_threads(4)
    base = LSTMConfig(epochs=2 if a.quick else 25)
    seeds = list(range(a.seeds))

    def fit(cfg, fit_idx, val_idx=None, epochs=None):
        s = LSTMStrategy(cfg)
        s.fit(c.X(fit_idx), c.y(cfg.horizon), fit_idx, val_idx, log=lambda *_: None, epochs=epochs)
        return s

    def to_pos(pred, cfg):
        return lstm_positions(pred, cfg.threshold, cfg.allow_short, cfg.decide_every)

    def best_threshold(pred, cfg):
        res = {}
        for th in THRESHOLDS:
            m = metrics(lstm_positions(pred, th, cfg.allow_short, cfg.decide_every), c.fwd[sp["val"]], COST)
            res[th] = (score(m), m["sharpe"])
        th = max(res, key=lambda k: res[k][0])
        return th, res[th][0], res[th][1]

    def record_preds(variant, seed, split, pred, cfg):
        c.preds[f"{variant}|{seed}|{split}"] = pred
        c.put("LSTM", variant, seed, split, to_pos(pred, cfg))

    def ensemble(variant, cfg, splits):
        for split in splits:
            p = np.mean([c.preds[f"{variant}|{s}|{split}"] for s in seeds], axis=0)
            record_preds(variant, "ens", split, p, cfg)

    # ---- untuned ----
    c.log("\n== LSTM untuned (defaults, hourly, fit once on train)")
    Xtr = c.X(sp["train"])
    for seed in seeds:
        s = fit(replace(base, seed=seed), sp["train"], sp["val"])
        for split in ("val", "test"):
            record_preds("untuned", seed, split, s.predict(Xtr, sp[split]), base)
    ensemble("untuned", base, ("val", "test"))

    # ---- tuning ----
    c.log(f"\n== LSTM Optuna: {a.lstm_trials} trials, objective = validation Sharpe "
          f"(exposure >= {MIN_EXPOSURE:.0%})")

    def objective(trial):
        cfg = replace(
            base,
            seq_len=trial.suggest_categorical("seq_len", [24, 48, 96, 168]),
            horizon=trial.suggest_categorical("horizon", [1, 4, 12, 24]),
            hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
            layers=trial.suggest_int("layers", 1, 2),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_categorical("weight_decay", [0.0, 1e-4, 1e-3]),
            decide_every=trial.suggest_categorical("decide_every", DECIDE_EVERY),
        )
        s = fit(cfg, sp["train"], sp["val"])
        th, sc, raw = best_threshold(s.predict(Xtr, sp["val"]), cfg)
        trial.set_user_attr("threshold", th)
        trial.set_user_attr("val_sharpe", raw)
        trial.set_user_attr("best_epoch", s.best_epoch)
        c.log(f"  trial {trial.number:2d} score={sc:+.2f} sharpe={raw:+.2f} th={th} "
              f"epochs={s.best_epoch} {trial.params}")
        return sc

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=a.lstm_trials)
    best = replace(base, **study.best_params, threshold=study.best_trial.user_attrs["threshold"])
    c.log("best LSTM:", best)

    # ---- tuned: val from train-only fit; test from single refit and from walk-forward ----
    c.log("\n== LSTM tuned (single refit on train+val) and tuned_wf (quarterly walk-forward)")
    best_epochs, diag = {}, {}
    for seed in seeds:
        cfg = replace(best, seed=seed)
        s0 = fit(cfg, sp["train"], sp["val"])
        best_epochs[seed] = s0.best_epoch
        record_preds("tuned", seed, "val", s0.predict(Xtr, sp["val"]), cfg)
        c.preds[f"tuned_wf|{seed}|val"] = c.preds[f"tuned|{seed}|val"]
        if seed == 0:
            diag["history"] = s0.history
            diag["importance"] = permutation_importance(
                s0, Xtr, c.y(cfg.horizon), sp["val"], F.FEATURES, n_repeats=1 if a.quick else 3)
        s1 = fit(cfg, sp["trainval"], None, epochs=s0.best_epoch)
        record_preds("tuned", seed, "test", s1.predict(c.X(sp["trainval"]), sp["test"]), cfg)
        wf = []
        for fit_idx, trade_idx in c.segments:
            s2 = fit(cfg, fit_idx, None, epochs=s0.best_epoch)
            wf.append(s2.predict(c.X(fit_idx), trade_idx))
        record_preds("tuned_wf", seed, "test", np.concatenate(wf), cfg)
    ensemble("tuned", best, ("val", "test"))
    ensemble("tuned_wf", best, ("test",))
    c.positions["LSTM|tuned_wf|ens|val"] = c.positions["LSTM|tuned|ens|val"]
    for seed in seeds:
        c.positions[f"LSTM|tuned_wf|{seed}|val"] = c.positions[f"LSTM|tuned|{seed}|val"]

    c.meta = {
        "untuned": asdict(base), "tuned": asdict(best), "best_epochs": best_epochs,
        "trials": [{"number": t.number, "score": t.value, **t.params, **t.user_attrs}
                   for t in study.trials],
        **diag,
    }
    c.save("lstm")


# ================================ PPO ================================
def run_ppo(c: Ctx):
    a, sp = c.a, c.sp
    base = RLConfig(timesteps=a.rl_steps)
    seeds = list(range(a.seeds))
    models_dir = c.out / "models"

    def job(cfg, fit_idx, evals=(), save=None, monitor=None):
        return {"cfg": asdict(cfg), "cost": COST, "X": c.X(fit_idx), "fwd": c.fwd,
                "fit_idx": fit_idx, "evals": list(evals),
                "save_path": str(save) if save else None,
                "monitor_path": str(monitor) if monitor else None}

    def vote(keys):
        return (np.mean([c.positions[k] for k in keys], axis=0) >= 0.5).astype(float)

    with ProcessPoolExecutor(max_workers=a.workers, initializer=jobs.init_worker) as pool:
        # ---- untuned ----
        c.log(f"\n== PPO untuned (SB3 defaults, hourly, fit once on train), {a.workers} workers")
        js = [job(replace(base, seed=s), sp["train"], [("val", sp["val"]), ("test", sp["test"])])
              for s in seeds]
        for seed, res in zip(seeds, pool.map(jobs.ppo_fit_job, js)):
            for split in ("val", "test"):
                c.put("PPO", "untuned", seed, split, res[split])
        for split in ("val", "test"):
            c.put("PPO", "untuned", "ens", split, vote([f"PPO|untuned|{s}|{split}" for s in seeds]))

        # ---- tuning: parallel batches through Optuna's ask/tell interface ----
        n_batches = int(np.ceil(a.rl_trials / a.batch))
        c.log(f"\n== PPO Optuna: {n_batches} x {a.batch} parallel trials, objective = validation "
              f"Sharpe (exposure >= {MIN_EXPOSURE:.0%})")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(
            seed=0, constant_liar=True, n_startup_trials=min(10, a.rl_trials)))
        for _ in range(n_batches):
            trials = [study.ask() for _ in range(a.batch)]
            cfgs = [replace(
                base,
                window=t.suggest_categorical("window", [12, 24, 48]),
                lr=t.suggest_float("lr", 3e-5, 1e-3, log=True),
                n_steps=t.suggest_categorical("n_steps", [256, 512, 1024]),
                gamma=t.suggest_categorical("gamma", [0.9, 0.97, 0.99]),
                ent_coef=t.suggest_float("ent_coef", 1e-4, 0.05, log=True),
                net=t.suggest_categorical("net", [64, 128]),
                downside_penalty=t.suggest_categorical("downside_penalty", [0.0, 0.5, 2.0]),
                decide_every=t.suggest_categorical("decide_every", DECIDE_EVERY),
            ) for t in trials]
            results = pool.map(jobs.ppo_fit_job, [job(cfg, sp["train"], [("val", sp["val"])]) for cfg in cfgs])
            for t, res in zip(trials, results):
                m = metrics(res["val"], c.fwd[sp["val"]], COST)
                t.set_user_attr("val_sharpe", m["sharpe"])
                t.set_user_attr("exposure", m["exposure"])
                study.tell(t, score(m))
                c.log(f"  trial {t.number:2d} score={score(m):+.2f} sharpe={m['sharpe']:+.2f} "
                      f"exp={m['exposure']:.0%} trades={m['trades']} {t.params}")
        best = replace(base, **study.best_params)
        c.log("best PPO:", best, "| val score", round(study.best_value, 3))

        # ---- tuned: every fit in parallel; walk-forward models saved, evaluated in order ----
        c.log("\n== PPO tuned (single refit on train+val) and tuned_wf (quarterly walk-forward)")
        js, keys = [], []
        for seed in seeds:
            cfg = replace(best, seed=seed)
            js.append(job(cfg, sp["train"], [("val", sp["val"])],
                          monitor=(c.out / "ppo_train_seed0") if seed == 0 else None))
            keys.append(("val", seed, None))
            js.append(job(cfg, sp["trainval"], [("test", sp["test"])]))
            keys.append(("test", seed, None))
            for i, (fit_idx, _) in enumerate(c.segments):
                js.append(job(cfg, fit_idx, save=models_dir / f"ppo_wf_seed{seed}_seg{i}"))
                keys.append(("wf", seed, i))
        results = list(pool.map(jobs.ppo_fit_job, js))

    for (kind, seed, _), res in zip(keys, results):
        if kind == "val":
            c.put("PPO", "tuned", seed, "val", res["val"])
            c.put("PPO", "tuned_wf", seed, "val", res["val"])
        elif kind == "test":
            c.put("PPO", "tuned", seed, "test", res["test"])
    for seed in seeds:
        cfg = replace(best, seed=seed)
        pos, prev = [], 0.0
        for i, (fit_idx, trade_idx) in enumerate(c.segments):
            s = PPOStrategy.load(models_dir / f"ppo_wf_seed{seed}_seg{i}", cfg, COST)
            p = s.positions(c.X(fit_idx), c.fwd, trade_idx, initial_pos=prev)
            pos.append(p)
            prev = float(p[-1])
        c.put("PPO", "tuned_wf", seed, "test", np.concatenate(pos))
    for variant in ("tuned", "tuned_wf"):
        for split in ("val", "test"):
            c.put("PPO", variant, "ens", split, vote([f"PPO|{variant}|{s}|{split}" for s in seeds]))

    c.meta = {
        "untuned": asdict(base), "tuned": asdict(best),
        "trials": [{"number": t.number, "score": t.value, **t.params, **t.user_attrs}
                   for t in study.trials],
    }
    c.save("ppo")


def keep_awake():
    """Ask Windows not to sleep while this process runs (released automatically
    when it exits). No-op elsewhere."""
    import sys
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["lstm", "ppo", "report", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lstm-trials", type=int, default=40)
    ap.add_argument("--rl-trials", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8, help="parallel PPO trials per Optuna batch")
    ap.add_argument("--workers", type=int, default=12, help="PPO worker processes")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rl-steps", type=int, default=300_000)
    a = ap.parse_args()
    if a.quick:
        a.lstm_trials, a.rl_trials, a.batch, a.seeds, a.rl_steps = 2, 2, 2, 2, 4096
    if a.part in ("lstm", "all"):
        run_lstm(Ctx(a, "lstm"))
    if a.part in ("ppo", "all"):
        run_ppo(Ctx(a, "ppo"))
    if a.part in ("report", "all"):
        import report
        report.main(quick=a.quick)


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
