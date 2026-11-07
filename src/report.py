"""Every number and plot in the README is computed here from the saved
position series (results/positions_*.npz), the predictions and the run metadata.

Outputs (results/): summary.md, metrics.csv, stats.json and the plots.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sst

import data
import features as F
from backtest import BARS_PER_YEAR, hold, metrics, strategy_returns
from stats import bootstrap_sharpes, ci, deflated_sharpe, probabilistic_sharpe

ROOT = Path(__file__).resolve().parents[1]
COST = 0.001
COSTS_BPS = [0, 5, 10, 20]
TARGET_VOL = 0.40          # annualised target for the volatility overlay, fixed in advance
BOOT_BLOCK = 168           # mean bootstrap block: one week of hourly returns

FOCUS = {                  # label -> "model|variant|seed" (ensembles for the learned models)
    "Buy & Hold": "Buy&Hold|-|-",
    "SMA 24/168": "SMA 24/168|-|-",
    "LSTM untuned": "LSTM|untuned|ens",
    "LSTM tuned": "LSTM|tuned|ens",
    "LSTM tuned + WF": "LSTM|tuned_wf|ens",
    "PPO untuned": "PPO|untuned|ens",
    "PPO tuned": "PPO|tuned|ens",
    "PPO tuned + WF": "PPO|tuned_wf|ens",
}
PRIMARY = ["LSTM tuned + WF", "PPO tuned + WF"]
ORDER = [("Buy&Hold", "-"), ("SMA 24/168", "-"), ("LSTM", "untuned"), ("LSTM", "tuned"),
         ("LSTM", "tuned_wf"), ("PPO", "untuned"), ("PPO", "tuned"), ("PPO", "tuned_wf")]


def pct(x):
    return f"{x:+.1%}"


def num(x):
    return f"{x:+.2f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(quick: bool = False):
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = ROOT / "results" / ("quick" if quick else "")
    raw = data.load()
    f = F.build(raw)
    sp = F.split_index(f)
    fwd = f["fwd_ret"].to_numpy()
    segments = F.walkforward_segments(f, sp["test"])
    n_boot = 300 if quick else 2000

    pos = {}
    for part in ("lstm", "ppo"):
        p = out / f"positions_{part}.npz"
        if p.exists():
            pos.update(dict(np.load(p)))
    meta = {}
    for part in ("lstm", "ppo"):
        p = out / f"meta_{part}.json"
        if p.exists():
            meta[part] = json.loads(p.read_text(encoding="utf-8"))
    for split in ("val", "test"):
        pos[f"Buy&Hold|-|-|{split}"] = np.ones(len(sp[split]))
        pos[f"SMA 24/168|-|-|{split}"] = f["sma_signal"].to_numpy()[sp[split]]
    focus = {k: v for k, v in FOCUS.items() if f"{v}|test" in pos}

    # ------------------------------------------------------------------ metrics
    rows = []
    for key, p in pos.items():
        model, variant, seed, split = key.split("|")
        rows.append({"model": model, "variant": variant, "seed": seed, "split": split,
                     **metrics(p, fwd[sp[split]], COST)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)

    lines = ["# Results: BTC/USDT hourly, 0.10% cost per unit traded\n",
             "Primary result (declared before the test period was evaluated): **tuned + "
             "walk-forward (WF), seed ensemble**. Seed rows are mean ± std over seeds; "
             "`ens` = seed ensemble.\n"]
    cols = [("sharpe", num), ("total_return", pct), ("max_drawdown", pct), ("calmar", num),
            ("exposure", pct)]
    for split in ("val", "test"):
        tab = []
        for model, variant in ORDER:
            g = df[(df.split == split) & (df.model == model) & (df.variant == variant)]
            if g.empty:
                continue
            for label, sub in (("seeds", g[~g.seed.isin(["ens", "-"])]), ("ens", g[g.seed == "ens"]),
                               ("-", g[g.seed == "-"])):
                if sub.empty:
                    continue
                cells = []
                for col, fmt in cols:
                    s = fmt(sub[col].mean())
                    if len(sub) > 1:
                        s += f" ± {sub[col].std():.1%}" if fmt is pct else f" ± {sub[col].std():.2f}"
                    cells.append(s)
                runs = f"{len(sub)} seeds" if label == "seeds" else label
                tab.append([model, variant, runs, *cells, f"{sub['trades'].mean():.0f}"])
        title = "Validation (2024)" if split == "val" else "Test (2025-01 to 2026-10, out of sample)"
        lines += [f"\n## {title}\n", md_table(
            ["model", "variant", "runs", "Sharpe", "total return", "max drawdown", "Calmar",
             "exposure", "trades"], tab)]

    # ------------------------------------------------------- significance (test)
    test_r = {lab: strategy_returns(pos[f"{key}|test"], fwd[sp["test"]], COST) for lab, key in focus.items()}
    sims = bootstrap_sharpes(test_r, n_boot=n_boot, mean_block=BOOT_BLOCK, seed=0)
    stat_rows, stats_json = [], {"test": {}, "val_dsr": {}}
    bh = "Buy & Hold"
    for lab in focus:
        sh = float(np.mean(test_r[lab]) / np.std(test_r[lab]) * np.sqrt(BARS_PER_YEAR)) if np.std(test_r[lab]) > 0 else 0.0
        lo, hi = ci(sims[lab])
        psr = probabilistic_sharpe(test_r[lab])
        if lab == bh:
            dcell = "-"
            d = dlo = dhi = p = None
        else:
            dsims = sims[lab] - sims[bh]
            d = sh - float(np.mean(test_r[bh]) / np.std(test_r[bh]) * np.sqrt(BARS_PER_YEAR))
            dlo, dhi = ci(dsims)
            p = float(np.mean(dsims <= 0))
            dcell = f"{d:+.2f} [{dlo:+.2f}, {dhi:+.2f}], p={p:.3f}"
        stat_rows.append([lab, f"{sh:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]", dcell, f"{psr:.2f}"])
        stats_json["test"][lab] = {"sharpe": sh, "ci": [lo, hi], "psr": psr,
                                   "diff_vs_bh": d, "diff_ci": [dlo, dhi], "p_beats_bh": p,
                                   **metrics(pos[f"{focus[lab]}|test"], fwd[sp["test"]], COST)}
    lines += ["\n## Statistical significance on the test period\n",
              f"Stationary block bootstrap ({n_boot} resamples, mean block {BOOT_BLOCK} h), paired "
              "across strategies. p = share of resamples where the strategy's Sharpe does not "
              "exceed buy-and-hold's (one-sided). PSR = probability the true Sharpe is above 0.\n",
              md_table(["strategy", "Sharpe", "95% CI", "Sharpe − B&H [95% CI], p", "PSR"], stat_rows)]

    # --------------------------------------------- validation deflated Sharpe
    dsr_rows = []
    for model, part in (("LSTM", "lstm"), ("PPO", "ppo")):
        if part not in meta or f"{model}|tuned|ens|val" not in pos:
            continue
        trial_sh = [t.get("val_sharpe") for t in meta[part]["trials"] if t.get("val_sharpe") is not None]
        r_val = strategy_returns(pos[f"{model}|tuned|ens|val"], fwd[sp["val"]], COST)
        dsr = deflated_sharpe(r_val, np.array(trial_sh))
        sh = metrics(pos[f"{model}|tuned|ens|val"], fwd[sp["val"]], COST)["sharpe"]
        dsr_rows.append([model, len(trial_sh), f"{sh:+.2f}", f"{probabilistic_sharpe(r_val):.2f}", f"{dsr:.2f}"])
        stats_json["val_dsr"][model] = {"trials": len(trial_sh), "val_sharpe": sh, "dsr": dsr}
    if dsr_rows:
        lines += ["\n## Is the validation edge more than search luck? (Deflated Sharpe Ratio)\n",
                  "DSR = probability that the tuned model's validation Sharpe beats the best Sharpe "
                  "expected by chance from that many Optuna trials (Bailey & López de Prado, 2014).\n",
                  md_table(["model", "Optuna trials", "val Sharpe (ens)", "PSR", "DSR"], dsr_rows)]

    # ------------------------------------------------------- cost sensitivity
    cost_rows, cost_curve = [], {}
    for lab, key in focus.items():
        p = pos[f"{key}|test"]
        ms = [metrics(p, fwd[sp["test"]], b / 1e4) for b in COSTS_BPS]
        cost_curve[lab] = [m["sharpe"] for m in ms]
        cost_rows.append([lab] + [f"{m['sharpe']:+.2f} / {m['total_return']:+.0%}" for m in ms])
    lines += ["\n## Transaction-cost sensitivity (test): Sharpe / total return\n",
              md_table(["strategy"] + [f"{b} bps" for b in COSTS_BPS], cost_rows)]

    # ------------------------------------------------ volatility-target overlay
    vol_ann = f["vol_168"].to_numpy()[sp["test"]] * np.sqrt(BARS_PER_YEAR)
    scale = hold(np.clip(TARGET_VOL / vol_ann, 0.0, 1.0), 24)
    vt_rows = []
    for lab, key in focus.items():
        p = pos[f"{key}|test"]
        m0 = metrics(p, fwd[sp["test"]], COST)
        m1 = metrics(p * scale, fwd[sp["test"]], COST)
        vt_rows.append([lab, f"{m0['sharpe']:+.2f} → {m1['sharpe']:+.2f}",
                        f"{m0['total_return']:+.1%} → {m1['total_return']:+.1%}",
                        f"{m0['max_drawdown']:.1%} → {m1['max_drawdown']:.1%}"])
        stats_json["test"].setdefault(lab, {})["vol_target"] = m1
    lines += [f"\n## Volatility-targeting overlay (test, target {TARGET_VOL:.0%} annualised, "
              "rebalanced daily, max 1x)\n",
              "Position size = min(1, target / trailing-1-week realised volatility). Same rule for "
              "every strategy, including buy-and-hold; the target was fixed in advance.\n",
              md_table(["strategy", "Sharpe", "total return", "max drawdown"], vt_rows)]

    # ------------------------------------------------------- per-quarter returns
    q_rows = []
    test_pos0 = sp["test"][0]
    for lab in focus:
        r = test_r[lab]
        cells = []
        for _, trade in segments:
            seg = r[trade - test_pos0]
            cells.append(f"{np.prod(1 + seg) - 1:+.1%}")
        q_rows.append([lab] + cells)
    q_names = [f"{f.index[t[0]].year} Q{(f.index[t[0]].month - 1) // 3 + 1}" for _, t in segments]
    lines += ["\n## Return by quarter (test)\n", md_table(["strategy"] + q_names, q_rows)]

    # ------------------------------------------------------ LSTM forecast quality
    pp = out / "preds_lstm.npz"
    if pp.exists() and "lstm" in meta:
        preds = dict(np.load(pp))
        fc_rows = []
        for variant in ("untuned", "tuned", "tuned_wf"):
            h = meta["lstm"]["untuned" if variant == "untuned" else "tuned"]["horizon"]
            y = F.target(f, h).to_numpy()
            for split in ("val", "test"):
                k = f"{variant}|ens|{split}"
                if k not in preds:
                    continue
                yy, pr = y[sp[split]], preds[k]
                ok = np.isfinite(yy) & (yy != 0)
                ic = sst.spearmanr(pr[ok], yy[ok]).statistic
                hit = float(np.mean(np.sign(pr[ok]) == np.sign(yy[ok])))
                fc_rows.append([variant, split, h, f"{ic:+.3f}", f"{hit:.1%}"])
                stats_json.setdefault("lstm_forecast", {})[f"{variant}|{split}"] = {"ic": ic, "hit_rate": hit, "horizon": h}
        lines += ["\n## LSTM forecast quality (seed ensemble)\n",
                  "IC = Spearman rank correlation between the forecast and the realised forward "
                  "return over the forecast horizon; hit rate = share of hours where the forecast "
                  "has the right sign.\n",
                  md_table(["variant", "split", "horizon (h)", "IC", "hit rate"], fc_rows)]

    # ------------------------------------------------------------ hyper-params
    for part in ("lstm", "ppo"):
        if part in meta:
            lines += [f"\n## Tuned {part.upper()} configuration\n", "```json",
                      json.dumps(meta[part]["tuned"], indent=2), "```"]

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "stats.json", "w", encoding="utf-8") as fh:
        json.dump(stats_json, fh, indent=2, default=float)
    plots(out, f, sp, fwd, pos, focus, cost_curve, meta)
    print("\n".join(lines))


def plots(out, f, sp, fwd, pos, focus, cost_curve, meta):
    t = f.index[sp["test"]]
    colors = {"LSTM untuned": "#8fd18f", "LSTM tuned": "#f4a259", "LSTM tuned + WF": "#1f5fbf",
              "PPO untuned": "#b08b7d", "PPO tuned": "#e58fc2", "PPO tuned + WF": "#7b2cbf"}
    styles = {"Buy & Hold": dict(color="black", lw=1.8), "SMA 24/168": dict(color="grey", lw=1.2, ls="--")}
    for lab, col in colors.items():
        styles[lab] = dict(color=col, lw=1.8 if lab in PRIMARY else 1.0, alpha=1.0 if lab in PRIMARY else 0.75)
    palette = plt.get_cmap("tab10")

    # equity + drawdown
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for i, (lab, key) in enumerate(focus.items()):
        r = strategy_returns(pos[f"{key}|test"], fwd[sp["test"]], COST)
        eq = np.cumprod(1 + r)
        dd = eq / np.maximum.accumulate(eq) - 1
        st_ = styles.get(lab, dict(color=palette(i), lw=1.6 if lab in PRIMARY else 0.9,
                                   alpha=1.0 if lab in PRIMARY else 0.55))
        ax1.plot(t, eq, label=lab, **st_)
        ax2.plot(t, dd, **st_)
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of $1 (log)")
    ax1.set_title("Out-of-sample test, BTC/USDT hourly, 0.10% cost (seed ensembles)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "equity_test.png", dpi=130)
    plt.close(fig)

    # cost sensitivity
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (lab, sh) in enumerate(cost_curve.items()):
        ax.plot(COSTS_BPS, sh, marker="o", label=lab, **styles.get(lab, dict(color=palette(i))))
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("cost per unit traded (bps)")
    ax.set_ylabel("test Sharpe")
    ax.set_title("Transaction-cost sensitivity")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "cost_sensitivity.png", dpi=130)
    plt.close(fig)

    # LSTM importance + training curves
    if "lstm" in meta and "importance" in meta["lstm"]:
        imp = pd.Series(meta["lstm"]["importance"]["delta_mse"]).sort_values()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(imp.index, imp.values, color="tab:blue")
        ax.set_xlabel("increase in validation MSE when the feature is shuffled")
        ax.set_title("LSTM permutation feature importance (tuned, seed 0)")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(out / "lstm_feature_importance.png", dpi=130)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "lstm" in meta and meta["lstm"].get("history"):
        h = pd.DataFrame(meta["lstm"]["history"])
        axes[0].plot(h["epoch"], h["train_mse"], marker="o", label="train")
        if "val_mse" in h:
            axes[0].plot(h["epoch"], h["val_mse"], marker="o", label="validation")
        axes[0].set_title("LSTM loss (tuned, seed 0)")
        axes[0].set_xlabel("epoch")
        axes[0].set_ylabel("MSE (target in sd units)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    mon = out / "ppo_train_seed0.monitor.csv"
    if mon.exists():
        m = pd.read_csv(mon, skiprows=1)
        axes[1].plot(np.cumsum(m["l"]), m["r"].rolling(50, min_periods=1).mean())
        axes[1].set_title("PPO episode reward (tuned, seed 0, rolling mean of 50)")
        axes[1].set_xlabel("environment steps")
        axes[1].set_ylabel("episode reward (100 x log growth, net of costs)")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=130)
    plt.close(fig)

    # behaviour: exposure by trend and volatility regime (validation)
    keys = {"LSTM tuned": "LSTM|tuned|ens|val", "PPO tuned": "PPO|tuned|ens|val"}
    keys = {k: v for k, v in keys.items() if v in pos}
    if keys:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        val = sp["val"]
        for ax, feat, title in ((axes[0], "dist_sma168", "trend: price vs 1-week average"),
                                (axes[1], "vol_168", "trailing 1-week volatility")):
            dec = pd.qcut(f[feat].to_numpy()[val], 10, labels=False, duplicates="drop")
            for i, (lab, k) in enumerate(keys.items()):
                expo = pd.Series(pos[k]).groupby(dec).mean()
                ax.plot(expo.index + 1, expo.values, marker="o", label=lab, color=palette(i + 2))
            ax.set_xlabel(f"decile of {title}")
            ax.set_ylabel("share of time long")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("What the tuned models do (validation 2024)")
        fig.tight_layout()
        fig.savefig(out / "behaviour.png", dpi=130)
        plt.close(fig)


if __name__ == "__main__":
    import sys
    main(quick="--quick" in sys.argv)
