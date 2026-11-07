"""LSTM return forecaster turned into a long/flat (optionally short) strategy.

The LSTM reads the last `seq_len` hours of standardised features and predicts
the forward `horizon`-hour log return. Rule: go long when the prediction exceeds
`threshold` (in units of the training target's std), flat otherwise, short when
below -threshold if `allow_short`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from backtest import hold, metrics


@dataclass
class LSTMConfig:
    seq_len: int = 48
    horizon: int = 4
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch: int = 256
    epochs: int = 20
    patience: int = 3
    threshold: float = 0.0
    allow_short: bool = False
    decide_every: int = 1        # re-decide position every N hours (hold in between)
    seed: int = 0


class Net(nn.Module):
    def __init__(self, n_feat, hidden, layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _windows(Xt: torch.Tensor, idx: np.ndarray, L: int) -> torch.Tensor:
    """Stack Xt[i-L+1 : i+1] for each i in idx -> (len(idx), L, n_feat), on Xt's device."""
    i = torch.as_tensor(idx, device=Xt.device)[:, None] + torch.arange(-L + 1, 1, device=Xt.device)[None, :]
    return Xt[i]


def positions(pred_z: np.ndarray, threshold: float, allow_short: bool, every: int = 1) -> np.ndarray:
    pos = (pred_z > threshold).astype(float)
    if allow_short:
        pos -= (pred_z < -threshold).astype(float)
    return hold(pos, every)


class LSTMStrategy:
    def __init__(self, cfg: LSTMConfig):
        self.cfg = cfg
        self.model = None
        self.y_sd = 1.0

    def fit(self, X, y, train_idx, val_idx=None, log=print, epochs=None):
        """With val_idx: early-stop on validation MSE and remember the best epoch.
        Without val_idx (final refit on train+val): train exactly `epochs` epochs."""
        c = self.cfg
        torch.manual_seed(c.seed)
        np.random.seed(c.seed)
        # Drop samples whose window is incomplete or whose label isn't known,
        # and purge the last `horizon` train bars (their labels reach into the next split).
        tr = train_idx[(train_idx >= c.seq_len - 1)][:-c.horizon]
        tr = tr[np.isfinite(y[tr])]
        self.y_sd = float(np.std(y[tr]))
        yt = torch.from_numpy(np.nan_to_num(y / self.y_sd).astype(np.float32)).to(DEVICE)
        Xt = torch.from_numpy(X).to(DEVICE)

        self.model = Net(X.shape[1], c.hidden, c.layers, c.dropout).to(DEVICE)
        opt = torch.optim.AdamW(self.model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
        loss_fn = nn.MSELoss()
        if val_idx is not None:
            va = val_idx[:-c.horizon]
            va = va[np.isfinite(y[va])]
            Xva, yva = _windows(Xt, va, c.seq_len), yt[va]
        best, best_state, bad = np.inf, None, 0
        self.best_epoch = epochs or c.epochs
        self.history = []                      # per-epoch train / val MSE (target in sd units)
        for ep in range(epochs or c.epochs):
            self.model.train()
            perm = np.random.permutation(tr)
            tot, nb = 0.0, 0
            for k in range(0, len(perm), c.batch):
                b = perm[k:k + c.batch]
                opt.zero_grad()
                loss = loss_fn(self.model(_windows(Xt, b, c.seq_len)), yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot, nb = tot + loss.item(), nb + 1
            if val_idx is None:
                self.history.append({"epoch": ep + 1, "train_mse": tot / nb})
                continue
            self.model.eval()
            with torch.no_grad():
                vl = float(loss_fn(self._predict_t(Xva), yva))
            self.history.append({"epoch": ep + 1, "train_mse": tot / nb, "val_mse": vl})
            log(f"  epoch {ep + 1:2d} train_mse {tot / nb:.4f} val_mse {vl:.4f}")
            if vl < best - 1e-4:
                best, bad, self.best_epoch = vl, 0, ep + 1
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= c.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best

    def _predict_t(self, Xw):
        out = []
        for k in range(0, len(Xw), 2048):
            out.append(self.model(Xw[k:k + 2048]))
        return torch.cat(out)

    def predict(self, X, idx) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X).to(DEVICE)
            return self._predict_t(_windows(Xt, idx, self.cfg.seq_len)).cpu().numpy()

    def positions(self, X, idx) -> np.ndarray:
        return positions(self.predict(X, idx), self.cfg.threshold, self.cfg.allow_short,
                         self.cfg.decide_every)

    def evaluate(self, X, idx, fwd_ret, cost) -> dict:
        return metrics(self.positions(X, idx), fwd_ret[idx], cost)


def config_dict(cfg: LSTMConfig) -> dict:
    return asdict(cfg)


def permutation_importance(strat: "LSTMStrategy", X: np.ndarray, y: np.ndarray, idx: np.ndarray,
                           names: list, n_repeats: int = 3, seed: int = 0) -> dict:
    """Increase in forecast MSE (target in training-sd units) when one feature's
    values are shuffled across the hours the evaluation windows read. Larger means
    the model relies more on that feature."""
    rng = np.random.default_rng(seed)
    idx = idx[np.isfinite(y[idx])]
    yt = y[idx] / strat.y_sd
    base = float(np.mean((strat.predict(X, idx) - yt) ** 2))
    lo = idx[0] - strat.cfg.seq_len + 1
    rows = np.arange(lo, idx[-1] + 1)
    out = {}
    for j, name in enumerate(names):
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[rows, j] = Xp[rng.permutation(rows), j]
            deltas.append(float(np.mean((strat.predict(Xp, idx) - yt) ** 2)) - base)
        out[name] = float(np.mean(deltas))
    return {"base_mse": base, "delta_mse": out}
