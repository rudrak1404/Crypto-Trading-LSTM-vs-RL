"""Statistical checks on strategy returns.

* Stationary block bootstrap (Politis & Romano, 1994): resamples blocks of
  random (geometric) length so the hourly returns keep their autocorrelation
  and volatility clustering. Used for Sharpe-ratio confidence intervals and for
  the paired Sharpe difference between a strategy and buy-and-hold.
* Probabilistic Sharpe Ratio (Bailey & Lopez de Prado, 2012): probability that
  the true Sharpe exceeds a benchmark, given sample length, skew and kurtosis.
* Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014): PSR against the Sharpe
  you would expect from the best of N tried configurations by luck alone, i.e.
  a correction for hyper-parameter search.

All Sharpe ratios inside these formulas are per-period (hourly), not annualised.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st

from backtest import BARS_PER_YEAR

EULER_GAMMA = 0.5772156649015329


def iter_bootstrap_indices(n: int, mean_block: int, n_boot: int, seed: int = 0, chunk: int = 200):
    """Yield (m, n) int32 arrays of stationary-bootstrap positions, m <= chunk,
    n_boot rows in total. Each row starts at a random hour; each next hour
    continues the current block with probability 1 - 1/mean_block, otherwise a
    new block starts at a random hour (indices wrap around)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = np.empty((m, n), dtype=np.int32)
        idx[:, 0] = rng.integers(0, n, m)
        jump = rng.random((m, n)) < p
        fresh = rng.integers(0, n, (m, n), dtype=np.int32)
        for t in range(1, n):
            idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
        yield idx
        done += m


def annual_sharpe(r: np.ndarray, axis=-1) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    sd = r.std(axis=axis)
    mu = r.mean(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), 0.0) * np.sqrt(BARS_PER_YEAR)
    return s


def bootstrap_sharpes(returns: dict, n_boot: int = 2000, mean_block: int = 168, seed: int = 0) -> dict:
    """Annualised Sharpe of every strategy on the *same* bootstrap resamples, so
    differences between strategies are paired. Returns {name: (n_boot,) array}."""
    names = list(returns)
    n = len(returns[names[0]])
    assert all(len(returns[k]) == n for k in names), "all return series must be aligned"
    sims = {k: [] for k in names}
    for idx in iter_bootstrap_indices(n, mean_block, n_boot, seed):
        for k in names:
            sims[k].append(annual_sharpe(np.asarray(returns[k])[idx]))
    return {k: np.concatenate(v) for k, v in sims.items()}


def ci(sims: np.ndarray, level: float = 0.95):
    lo, hi = np.quantile(sims, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """PSR: P(true per-period Sharpe > sr_benchmark)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    sd = r.std(ddof=1)
    if n < 3 or sd == 0:
        return float("nan")
    sr = r.mean() / sd
    skew = st.skew(r)
    kurt = st.kurtosis(r, fisher=False)          # non-excess kurtosis
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(st.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(trial_sharpes_per_period: np.ndarray) -> float:
    """Expected maximum per-period Sharpe among N independent trials with the
    observed cross-trial variance, under the null of zero true Sharpe."""
    s = np.asarray(trial_sharpes_per_period, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 2:
        return 0.0
    sd = s.std(ddof=1)
    return float(sd * ((1 - EULER_GAMMA) * st.norm.ppf(1 - 1 / n)
                       + EULER_GAMMA * st.norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe(r: np.ndarray, trial_sharpes_annual: np.ndarray) -> float:
    """DSR of the selected strategy's returns `r`, deflated by the search over
    the trials whose (annualised) Sharpe ratios are given."""
    per_period = np.asarray(trial_sharpes_annual, dtype=float) / np.sqrt(BARS_PER_YEAR)
    return probabilistic_sharpe(r, expected_max_sharpe(per_period))
