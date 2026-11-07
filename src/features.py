"""Feature engineering and chronological splits.

Every feature at bar t uses only data up to and including bar t's close.
`fwd_ret` is the log return from close t to close t+1: the return a position
decided at bar t earns. It is a label, never a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_4", "ret_24", "ret_72",
    "vol_24", "vol_168",
    "rsi_14", "macd_hist",
    "dist_sma24", "dist_sma168",
    "vol_z", "hl_range",
    "hour_sin", "hour_cos",
]

SPLITS = {
    "train": ("2019-01-01", "2023-12-31 23:00"),
    "val": ("2024-01-01", "2024-12-31 23:00"),
    "test": ("2025-01-01", "2026-10-31 23:00"),
}


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build(df: pd.DataFrame) -> pd.DataFrame:
    full = pd.date_range(df.index[0], df.index[-1], freq="1h", name="time")
    df = df.reindex(full)
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)

    c = df["close"]
    lr = np.log(c).diff()
    f = pd.DataFrame(index=df.index)
    f["ret_1"] = lr
    for k in (4, 24, 72):
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    f["vol_24"] = lr.rolling(24).std()
    f["vol_168"] = lr.rolling(168).std()
    f["rsi_14"] = _rsi(c) / 100 - 0.5
    macd = (c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()) / c
    f["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    sma24, sma168 = c.rolling(24).mean(), c.rolling(168).mean()
    f["dist_sma24"] = c / sma24 - 1
    f["dist_sma168"] = c / sma168 - 1
    lv = np.log1p(df["volume"])
    f["vol_z"] = (lv - lv.rolling(168).mean()) / lv.rolling(168).std()
    f["hl_range"] = np.log(df["high"] / df["low"])
    hour = df.index.hour
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Labels / helpers (not model inputs)
    f["close"] = c
    f["fwd_ret"] = lr.shift(-1)
    f["sma_signal"] = (sma24 > sma168).astype(float)
    f = f.replace([np.inf, -np.inf], np.nan)
    return f.dropna(subset=FEATURES + ["fwd_ret"])


def target(f: pd.DataFrame, horizon: int) -> pd.Series:
    """Forward log return over `horizon` bars, starting at the close of bar t."""
    return np.log(f["close"].shift(-horizon) / f["close"])


def split_index(f: pd.DataFrame) -> dict[str, np.ndarray]:
    """Integer positions of each split inside f (f keeps the full timeline so
    windows at the start of val/test can look back into the previous split)."""
    out = {}
    for name, (a, b) in SPLITS.items():
        mask = (f.index >= pd.Timestamp(a, tz="UTC")) & (f.index <= pd.Timestamp(b, tz="UTC"))
        out[name] = np.flatnonzero(mask)
    return out


def walkforward_segments(f: pd.DataFrame, split: np.ndarray, freq: str = "QS"):
    """Split the positions `split` (e.g. the test period) into calendar segments
    (quarters by default). For each segment return (fit_idx, trade_idx): the model
    is (re)trained on everything strictly before the segment (expanding window)
    and then trades the segment."""
    times = f.index[split]
    starts = pd.date_range(times[0].floor("D"), times[-1], freq=freq)
    starts = starts.union(pd.DatetimeIndex([times[0]]))
    out = []
    for i, a in enumerate(starts):
        b = starts[i + 1] if i + 1 < len(starts) else None
        mask = (times >= a) & ((times < b) if b is not None else True)
        trade = split[mask]
        if len(trade):
            out.append((np.arange(0, trade[0]), trade))
    return out


def scaled_matrix(f: pd.DataFrame, train_idx: np.ndarray) -> np.ndarray:
    """Standardise features with train-only statistics, clipped to +/-5 sd."""
    X = f[FEATURES].to_numpy(dtype=np.float32)
    mu, sd = X[train_idx].mean(0), X[train_idx].std(0) + 1e-8
    return np.clip((X - mu) / sd, -5, 5).astype(np.float32)
