"""Vectorised backtest and performance metrics.

Convention: position p[t] is decided at the close of bar t and earns fwd_ret[t]
(close t -> close t+1). Changing position costs `cost` per unit traded
(0.001 = 0.1%, Binance spot taker fee).
"""
from __future__ import annotations

import numpy as np

BARS_PER_YEAR = 24 * 365


def hold(pos, every: int) -> np.ndarray:
    """Only allow position changes every `every` bars (decision interval):
    bar i keeps the decision made at bar i - i % every."""
    pos = np.asarray(pos, dtype=float)
    if every <= 1:
        return pos
    return pos[np.arange(len(pos)) // every * every]


def strategy_returns(pos, fwd_ret, cost: float = 0.001) -> np.ndarray:
    pos = np.asarray(pos, dtype=float)
    simple = np.expm1(np.asarray(fwd_ret, dtype=float))
    prev = np.r_[0.0, pos[:-1]]
    return pos * simple - cost * np.abs(pos - prev)


def max_drawdown(rets) -> float:
    eq = np.cumprod(1 + np.asarray(rets))
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    return float((eq / peak - 1).min())


def metrics(pos, fwd_ret, cost: float = 0.001) -> dict:
    pos = np.asarray(pos, dtype=float)
    r = strategy_returns(pos, fwd_ret, cost)
    years = len(r) / BARS_PER_YEAR
    total = float(np.prod(1 + r) - 1)
    sd = r.std()
    down = r[r < 0].std() if (r < 0).any() else np.nan
    mdd = max_drawdown(r)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    prev = np.r_[0.0, pos[:-1]]
    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": float(r.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0,
        "sortino": float(r.mean() / down * np.sqrt(BARS_PER_YEAR)) if down and down > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else 0.0,
        "exposure": float(np.mean(pos != 0)),
        "trades": int(np.sum(np.abs(pos - prev) > 0)),
    }


def equity(pos, fwd_ret, cost: float = 0.001) -> np.ndarray:
    return np.cumprod(1 + strategy_returns(pos, fwd_ret, cost))
