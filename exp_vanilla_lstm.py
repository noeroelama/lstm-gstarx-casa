#!/usr/bin/env python3
"""Vanilla LSTM (no GSTAR) -- is the GSTAR spatial lag F* actually urgent?

LSTM-GSTARX(K;1) feeds the LSTM a per-timestep input [Y_t, F*_t, X_t] (82-dim), where
F* = W.Y is the GSTAR spatial lag. This vanilla LSTM DROPS F* entirely: input [Y_t, X_t]
(42-dim) -- a plain multivariate LSTM over all 40 provinces' rainfall + climate, with NO
spatial weight matrix and NO spatial-lag operator. Same K, same 5-fold walk-forward CV,
same hidden size / optimizer / epochs as the GSTAR models.

Comparison target: LSTM-GSTARX(12;1) = 71.37 mm (single-seed CV).
  - vanilla ~ 71  -> the explicit GSTAR spatial lag is NOT urgent (the LSTM learns spatial
    structure from the raw multivariate input). Reinforces 'spatial is immaterial'.
  - vanilla > 71  -> F* adds value -> the GSTAR component is justified.
"""
import sys, csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_casa_n40 import (normalize, build_window, inverse_y_idn,
                            N_TOTAL, N_IDN, HIDDEN_SIZE, EPOCHS, LR, CLIP_NORM,
                            N_SPLITS, MIN_TRAIN)

PANEL = r"data/panel_data.npz"
CASA_DIR = r"."
K = 12
SEEDS = [7, 42, 123]   # multi-seed, to compare fairly with LSTM-GSTARX(12;1) = 71.99 +- 0.57

panel = np.load(PANEL, allow_pickle=True)
Y = panel['Y_rainfall'].astype(float)
X = panel['X_climate'].astype(float)
idn = np.where(panel['region_countries'] == 'IDN')[0]
T, N = Y.shape
FOLD = (T - MIN_TRAIN) // N_SPLITS


class VanillaLSTM(nn.Module):
    """Plain multivariate LSTM: input [Y_all(40), X(2)] per timestep -- no F*, no CASA."""

    def __init__(self, N=N_TOTAL, N_out=N_IDN, hidden=HIDDEN_SIZE, n_exog=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=N + n_exog, hidden_size=hidden, batch_first=True)
        self.linear = nn.Linear(hidden, N_out)

    def forward(self, Y_window, X_window):
        x = torch.cat([Y_window, X_window], dim=-1)      # (B, K, N+2)
        h, _ = self.lstm(x)
        return self.linear(h[:, -1, :])


def run(seed):
    rmses = []
    for f in range(N_SPLITS):
        tr_end = MIN_TRAIN + f * FOLD; te_s, te_e = tr_end, tr_end + FOLD
        if te_e > T: break
        Yn, Xn, Ystats = normalize(Y, X, tr_end)
        Yw, Xw, _, _, Ytgt = build_window(Yn, Xn, X, idn, K)
        torch.manual_seed(seed + f); np.random.seed(seed + f)
        Yw_t = torch.tensor(Yw, dtype=torch.float32)
        Xw_t = torch.tensor(Xw, dtype=torch.float32)
        Ytgt_t = torch.tensor(Ytgt, dtype=torch.float32)
        tr_lo, tr_hi = 0, tr_end - K
        te_lo, te_hi = tr_end - K, te_e - K
        model = VanillaLSTM()
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        for _ in range(EPOCHS):
            opt.zero_grad()
            yh = model(Yw_t[tr_lo:tr_hi], Xw_t[tr_lo:tr_hi])
            loss = F.mse_loss(yh, Ytgt_t[tr_lo:tr_hi]) * 0.5
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
        model.eval()
        with torch.no_grad():
            yh = model(Yw_t[te_lo:te_hi], Xw_t[te_lo:te_hi])
        yhmm = inverse_y_idn(yh.numpy(), idn, Ystats)
        rmses.append(float(np.sqrt(((yhmm - Y[te_s:te_e][:, idn]) ** 2).mean())))
    r = np.array(rmses)
    return r.mean(), r.std(ddof=1), rmses


if __name__ == "__main__":
    print(f"T={T} N={N} IDN={len(idn)} K={K} | input={N+2}-dim (no F*) | "
          f"hidden={HIDDEN_SIZE} epochs={EPOCHS} lr={LR}")
    print(f"compare: LSTM-GSTARX(12;1) with F* = 71.37 mm")
    rows = []
    seed_means = []
    for seed in SEEDS:
        t0 = time.time()
        m, s, per = run(seed)
        seed_means.append(m)
        print(f"  vanilla LSTM(K=12) seed={seed}: CV RMSE = {m:.2f} +- {s:.2f}  "
              f"folds={[round(v,1) for v in per]}  ({time.time()-t0:.0f}s)")
        rows.append(["vanilla_lstm_k12", K, seed, round(m, 3), round(s, 3)])
    agg_mean = float(np.mean(seed_means)); agg_std = float(np.std(seed_means, ddof=1))
    print(f"\n  multi-seed aggregate: {agg_mean:.2f} +- {agg_std:.2f} mm  "
          f"(vs LSTM-GSTARX(12;1) 71.99 +- 0.57 -> F* adds {agg_mean-71.99:+.2f} mm)")
    rows.append(["vanilla_lstm_k12_MULTISEED", K, "7;42;123", round(agg_mean, 3), round(agg_std, 3)])
    with open(CASA_DIR + r"\vanilla_lstm_rmse.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "K", "seed", "rmse_mean", "rmse_std"])
        w.writerows(rows)
    print("\nWrote vanilla_lstm_rmse.csv")
