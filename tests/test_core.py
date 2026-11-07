"""Checks for the parts that silently ruin trading backtests: look-ahead and cost accounting."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features as F  # noqa: E402
from backtest import max_drawdown, metrics, strategy_returns  # noqa: E402
from rl_agent import TradingEnv  # noqa: E402


def _fake_ohlcv(n=600, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC", name="time")
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.uniform(1, 10, n), "trades": 1.0}, index=idx)


def test_features_have_no_lookahead():
    df = _fake_ohlcv()
    a = F.build(df)
    df2 = df.copy()
    df2.iloc[400:, :5] *= 3.0          # change only the future
    b = F.build(df2)
    cut = a.index[a.index < df.index[400]]
    pd.testing.assert_frame_equal(a.loc[cut, F.FEATURES], b.loc[cut, F.FEATURES])


def test_fwd_ret_is_next_bar_return():
    df = _fake_ohlcv()
    f = F.build(df)
    t = f.index[10]
    nxt = df.index[df.index.get_loc(t) + 1]
    assert np.isclose(f.loc[t, "fwd_ret"], np.log(df.loc[nxt, "close"] / df.loc[t, "close"]))


def test_costs_and_returns():
    fwd = np.log(np.array([1.10, 1.0, 0.5]))      # +10%, 0%, -50%
    r = strategy_returns([1, 1, 0], fwd, cost=0.01)
    assert np.allclose(r, [0.10 - 0.01, 0.0, -0.01])
    assert np.isclose(metrics([1, 1, 1], fwd, 0)["total_return"], 1.1 * 1.0 * 0.5 - 1)
    assert np.isclose(max_drawdown([0.10, 0.0, -0.5]), -0.5)


def test_env_reward_matches_backtest():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3)).astype(np.float32)
    fwd = rng.normal(0, 0.01, 200)
    idx = np.arange(50, 150)
    env = TradingEnv(X, fwd, idx, window=10, cost=0.001)
    env.reset()
    acts = rng.integers(0, 2, len(idx))
    total, done, i = 0.0, False, 0
    while not done:
        _, rew, done, _, _ = env.step(acts[i])
        total += rew
        i += 1
    expected = 100 * np.log1p(strategy_returns(acts.astype(float), fwd[idx], 0.001)).sum()
    assert np.isclose(total, expected)


def test_hold_and_env_decision_interval():
    from backtest import hold
    assert hold([1, 0, 0, 1, 0, 1, 1], 3).tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert hold([0, 1, 1, 0, 1], 2).tolist() == [0, 0, 1, 1, 1]
    rng = np.random.default_rng(2)
    X = rng.normal(size=(200, 2)).astype(np.float32)
    fwd = rng.normal(0, 0.01, 200)
    idx = np.arange(50, 151)                       # 101 bars, not a multiple of 4
    env = TradingEnv(X, fwd, idx, window=10, cost=0.001, decide_every=4)
    env.reset()
    total, held, done, k = 0.0, [], False, 0
    while not done:
        _, rew, done, _, info = env.step(k % 2)
        total += rew
        held += info["pos"]
        k += 1
    assert len(held) == len(idx)
    expected = 100 * np.log1p(strategy_returns(np.array(held), fwd[idx], 0.001)).sum()
    assert np.isclose(total, expected)


def test_env_observation_uses_only_past():
    X = np.arange(100, dtype=np.float32)[:, None]
    env = TradingEnv(X, np.zeros(100), np.arange(20, 30), window=5, cost=0.0)
    obs, _ = env.reset()
    assert obs[:-1].tolist() == [16, 17, 18, 19, 20]   # window ends at the current bar


def test_walkforward_segments_cover_test_without_lookahead():
    df = _fake_ohlcv(n=24 * 400)                     # ~13 months of hourly bars from 2020-01-01
    f = F.build(df)
    split = np.flatnonzero(f.index >= pd.Timestamp("2020-07-01", tz="UTC"))
    segs = F.walkforward_segments(f, split)
    trades = np.concatenate([t for _, t in segs])
    assert np.array_equal(trades, split)             # every test hour traded exactly once, in order
    for fit_idx, trade in segs:
        assert fit_idx[-1] == trade[0] - 1           # fit window ends right before the segment
        assert f.index[trade[0]].day == 1 and f.index[trade[0]].month in (1, 4, 7, 10)


def test_sharpe_statistics():
    from stats import annual_sharpe, bootstrap_sharpes, deflated_sharpe, probabilistic_sharpe
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 5000)
    assert np.isclose(annual_sharpe(r), r.mean() / r.std() * np.sqrt(24 * 365))
    sims = bootstrap_sharpes({"a": r, "b": r * 0.5}, n_boot=200, mean_block=24)
    assert np.allclose(sims["a"], sims["b"])         # Sharpe is scale invariant, paired resamples
    assert probabilistic_sharpe(r) > 0.95            # clearly positive drift
    assert abs(probabilistic_sharpe(rng.normal(0, 0.01, 5000)) - 0.5) < 0.45
    # deflating by many lucky trials can only lower the probability
    assert deflated_sharpe(r, np.full(1, 2.0)) == probabilistic_sharpe(r)
    assert deflated_sharpe(r, rng.normal(0, 1.0, 50)) < probabilistic_sharpe(r)


def test_env_initial_position_carries_over():
    X = np.zeros((60, 2), dtype=np.float32)
    fwd = np.full(60, 0.01)
    env = TradingEnv(X, fwd, np.arange(10, 20), window=5, cost=0.01)
    obs, _ = env.reset(options={"initial_pos": 1.0})
    assert obs[-1] == 1.0
    _, rew, _, _, _ = env.step(1)                    # stay long: no cost
    assert np.isclose(rew, 100 * np.log1p(np.expm1(0.01)))
