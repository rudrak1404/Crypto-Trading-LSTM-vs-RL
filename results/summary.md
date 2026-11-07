# Results: BTC/USDT hourly, 0.10% cost per unit traded

Primary result (declared before the test period was evaluated): **tuned + walk-forward (WF), seed ensemble**. Seed rows are mean ± std over seeds; `ens` = seed ensemble.


## Validation (2024)

| model | variant | runs | Sharpe | total return | max drawdown | Calmar | exposure | trades |
|---|---|---|---|---|---|---|---|---|
| Buy&Hold | - | - | +1.78 | +122.0% | -32.3% | +3.76 | +100.0% | 1 |
| SMA 24/168 | - | - | +1.77 | +81.6% | -15.4% | +5.27 | +56.3% | 68 |
| LSTM | untuned | 3 seeds | +0.61 ± 0.53 | +21.3% ± 27.8% | -40.7% ± 4.0% | +0.56 ± 0.70 | +67.3% ± 11.4% | 669 |
| LSTM | untuned | ens | +0.44 | +10.4% | -39.6% | +0.26 | +67.8% | 672 |
| LSTM | tuned | 3 seeds | +2.18 ± 0.63 | +123.6% ± 31.2% | -32.0% ± 10.5% | +4.46 ± 2.82 | +64.9% ± 19.5% | 122 |
| LSTM | tuned | ens | +1.91 | +114.3% | -31.6% | +3.61 | +72.1% | 132 |
| LSTM | tuned_wf | 3 seeds | +2.18 ± 0.63 | +123.6% ± 31.2% | -32.0% ± 10.5% | +4.46 ± 2.82 | +64.9% ± 19.5% | 122 |
| LSTM | tuned_wf | ens | +1.91 | +114.3% | -31.6% | +3.61 | +72.1% | 132 |
| PPO | untuned | 3 seeds | -2.15 ± 0.50 | -57.9% ± 7.5% | -63.5% ± 4.4% | -0.91 ± 0.06 | +48.0% ± 2.8% | 1430 |
| PPO | untuned | ens | -3.46 | -74.8% | -75.6% | -0.99 | +47.2% | 1675 |
| PPO | tuned | 3 seeds | +0.85 ± 0.88 | +31.9% ± 43.8% | -29.9% ± 11.5% | +1.74 ± 2.65 | +50.3% ± 5.2% | 162 |
| PPO | tuned | ens | +1.11 | +43.4% | -27.9% | +1.55 | +52.5% | 152 |
| PPO | tuned_wf | 3 seeds | +0.85 ± 0.88 | +31.9% ± 43.8% | -29.9% ± 11.5% | +1.74 ± 2.65 | +50.3% ± 5.2% | 162 |
| PPO | tuned_wf | ens | +1.11 | +43.4% | -27.9% | +1.55 | +52.5% | 152 |

## Test (2025-01 to 2026-10, out of sample)

| model | variant | runs | Sharpe | total return | max drawdown | Calmar | exposure | trades |
|---|---|---|---|---|---|---|---|---|
| Buy&Hold | - | - | -0.03 | -17.9% | -53.7% | -0.19 | +100.0% | 1 |
| SMA 24/168 | - | - | -0.86 | -40.5% | -52.0% | -0.47 | +49.6% | 172 |
| LSTM | untuned | 3 seeds | -1.69 ± 0.39 | -68.1% ± 2.9% | -73.6% ± 3.9% | -0.63 ± 0.01 | +68.4% ± 14.0% | 1131 |
| LSTM | untuned | ens | -1.61 | -67.6% | -72.3% | -0.64 | +69.1% | 1140 |
| LSTM | tuned | 3 seeds | +0.02 ± 0.16 | -10.4% ± 10.2% | -47.8% ± 7.0% | -0.12 ± 0.11 | +84.6% ± 3.1% | 136 |
| LSTM | tuned | ens | +0.04 | -10.6% | -47.1% | -0.13 | +88.2% | 117 |
| LSTM | tuned_wf | 3 seeds | +0.17 ± 0.10 | -0.1% ± 6.9% | -42.9% ± 10.0% | +0.01 ± 0.09 | +70.4% ± 11.2% | 203 |
| LSTM | tuned_wf | ens | +0.18 | +0.2% | -39.0% | +0.00 | +73.1% | 233 |
| PPO | untuned | 3 seeds | -4.00 ± 0.37 | -89.9% ± 3.1% | -90.2% ± 3.3% | -0.79 ± 0.02 | +47.3% ± 8.0% | 2354 |
| PPO | untuned | ens | -4.69 | -93.4% | -93.4% | -0.83 | +46.4% | 2847 |
| PPO | tuned | 3 seeds | -0.53 ± 0.49 | -30.6% ± 21.3% | -48.4% ± 7.9% | -0.36 ± 0.24 | +47.7% ± 5.7% | 287 |
| PPO | tuned | ens | -0.09 | -12.9% | -38.4% | -0.19 | +46.3% | 277 |
| PPO | tuned_wf | 3 seeds | -0.16 ± 0.34 | -15.7% ± 16.3% | -37.1% ± 5.8% | -0.24 ± 0.24 | +53.0% ± 6.4% | 272 |
| PPO | tuned_wf | ens | -0.43 | -30.5% | -44.0% | -0.41 | +54.1% | 270 |

## Statistical significance on the test period

Stationary block bootstrap (2000 resamples, mean block 168 h), paired across strategies. p = share of resamples where the strategy's Sharpe does not exceed buy-and-hold's (one-sided). PSR = probability the true Sharpe is above 0.

| strategy | Sharpe | 95% CI | Sharpe − B&H [95% CI], p | PSR |
|---|---|---|---|---|
| Buy & Hold | -0.03 | [-1.35, +1.31] | - | 0.48 |
| SMA 24/168 | -0.86 | [-2.29, +0.64] | -0.83 [-2.03, +0.33], p=0.923 | 0.12 |
| LSTM untuned | -1.61 | [-3.02, -0.07] | -1.58 [-2.42, -0.74], p=0.999 | 0.01 |
| LSTM tuned | +0.04 | [-1.34, +1.43] | +0.07 [-0.41, +0.54], p=0.409 | 0.52 |
| LSTM tuned + WF | +0.18 | [-1.16, +1.48] | +0.21 [-0.60, +1.04], p=0.322 | 0.60 |
| PPO untuned | -4.69 | [-6.01, -3.29] | -4.66 [-6.01, -3.43], p=1.000 | 0.00 |
| PPO tuned | -0.09 | [-1.42, +1.42] | -0.06 [-0.98, +0.91], p=0.543 | 0.45 |
| PPO tuned + WF | -0.43 | [-1.78, +0.95] | -0.40 [-1.31, +0.61], p=0.789 | 0.28 |

## Is the validation edge more than search luck? (Deflated Sharpe Ratio)

DSR = probability that the tuned model's validation Sharpe beats the best Sharpe expected by chance from that many Optuna trials (Bailey & López de Prado, 2014).

| model | Optuna trials | val Sharpe (ens) | PSR | DSR |
|---|---|---|---|---|
| LSTM | 40 | +1.91 | 0.97 | 0.63 |
| PPO | 24 | +1.11 | 0.87 | 0.00 |

## Transaction-cost sensitivity (test): Sharpe / total return

| strategy | 0 bps | 5 bps | 10 bps | 20 bps |
|---|---|---|---|---|
| Buy & Hold | -0.03 / -18% | -0.03 / -18% | -0.03 / -18% | -0.03 / -18% |
| SMA 24/168 | -0.53 / -29% | -0.69 / -35% | -0.86 / -41% | -1.18 / -50% |
| LSTM untuned | +0.19 / +1% | -0.71 / -43% | -1.61 / -68% | -3.39 / -90% |
| LSTM tuned | +0.20 / +1% | +0.12 / -5% | +0.04 / -11% | -0.12 / -20% |
| LSTM tuned + WF | +0.54 / +26% | +0.36 / +13% | +0.18 / +0% | -0.17 / -21% |
| PPO untuned | +0.40 / +15% | -2.16 / -72% | -4.69 / -93% | -9.58 / -100% |
| PPO tuned | +0.40 / +15% | +0.16 / +0% | -0.09 / -13% | -0.58 / -34% |
| PPO tuned + WF | +0.01 / -9% | -0.21 / -20% | -0.43 / -31% | -0.88 / -47% |

## Volatility-targeting overlay (test, target 40% annualised, rebalanced daily, max 1x)

Position size = min(1, target / trailing-1-week realised volatility). Same rule for every strategy, including buy-and-hold; the target was fixed in advance.

| strategy | Sharpe | total return | max drawdown |
|---|---|---|---|
| Buy & Hold | -0.03 → -0.08 | -17.9% → -16.7% | -53.7% → -51.9% |
| SMA 24/168 | -0.86 → -0.79 | -40.5% → -33.9% | -52.0% → -44.7% |
| LSTM untuned | -1.61 → -1.66 | -67.6% → -63.1% | -72.3% → -67.6% |
| LSTM tuned | +0.04 → +0.03 | -10.6% → -8.1% | -47.1% → -45.4% |
| LSTM tuned + WF | +0.18 → +0.08 | +0.2% → -4.0% | -39.0% → -38.0% |
| PPO untuned | -4.69 → -5.02 | -93.4% → -91.5% | -93.4% → -91.6% |
| PPO tuned | -0.09 → -0.10 | -12.9% → -9.8% | -38.4% → -31.6% |
| PPO tuned + WF | -0.43 → -0.47 | -30.5% → -26.6% | -44.0% → -39.7% |

## Return by quarter (test)

| strategy | 2025 Q1 | 2025 Q2 | 2025 Q3 | 2025 Q4 | 2026 Q1 | 2026 Q2 | 2026 Q3 | 2026 Q4 |
|---|---|---|---|---|---|---|---|---|
| Buy & Hold | -12.5% | +29.9% | +6.4% | -23.1% | -22.2% | -14.3% | +42.6% | -7.1% |
| SMA 24/168 | -23.4% | +7.2% | +4.8% | -20.3% | -14.6% | -2.9% | +14.4% | -8.6% |
| LSTM untuned | -20.1% | +10.1% | -6.7% | -34.3% | -28.3% | -13.3% | +10.4% | -12.5% |
| LSTM tuned | -15.1% | +22.7% | +9.9% | -16.8% | -19.8% | -10.7% | +42.1% | -7.7% |
| LSTM tuned + WF | -15.1% | +23.3% | -5.2% | -16.7% | -0.8% | +10.3% | +24.9% | -11.3% |
| PPO untuned | -41.5% | -9.8% | -13.7% | -45.7% | -43.3% | -31.8% | -21.5% | -11.6% |
| PPO tuned | -24.7% | +14.0% | +3.1% | -5.3% | -10.3% | -2.6% | +27.0% | -6.3% |
| PPO tuned + WF | -24.7% | +14.6% | -1.9% | -14.9% | -5.4% | -11.2% | +21.2% | -5.2% |

## LSTM forecast quality (seed ensemble)

IC = Spearman rank correlation between the forecast and the realised forward return over the forecast horizon; hit rate = share of hours where the forecast has the right sign.

| variant | split | horizon (h) | IC | hit rate |
|---|---|---|---|---|
| untuned | val | 4 | +0.028 | 50.8% |
| untuned | test | 4 | -0.025 | 49.2% |
| tuned | val | 24 | +0.109 | 54.0% |
| tuned | test | 24 | +0.000 | 51.6% |
| tuned_wf | test | 24 | +0.041 | 50.8% |

## Tuned LSTM configuration

```json
{
  "seq_len": 168,
  "horizon": 24,
  "hidden": 32,
  "layers": 2,
  "dropout": 0.12144511904097338,
  "lr": 0.002648990715782306,
  "weight_decay": 0.0001,
  "batch": 256,
  "epochs": 25,
  "patience": 3,
  "threshold": 0.0,
  "allow_short": false,
  "decide_every": 24,
  "seed": 0
}
```

## Tuned PPO configuration

```json
{
  "window": 24,
  "episode_len": 720,
  "timesteps": 300000,
  "lr": 0.00017277216481159552,
  "n_steps": 256,
  "batch": 64,
  "gamma": 0.97,
  "ent_coef": 0.0009079154170956705,
  "net": 64,
  "downside_penalty": 0.0,
  "allow_short": false,
  "decide_every": 24,
  "n_envs": 4,
  "seed": 0
}
```
