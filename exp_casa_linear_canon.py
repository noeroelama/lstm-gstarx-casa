#!/usr/bin/env python3
"""Canonical (shared-parameter) version of the CASA-in-a-linear-model ablation.

Same question as exp_casa_linear.py -- does an adaptive/learned spatial weight matrix help a
LINEAR (no-LSTM) GSTAR(K;1) -- but with the canonical GSTAR estimator (homogeneous dynamics
shared across locations + per-location intercept, one stacked OLS), to match exp_gstar_canonical.

Per fold, per mode:
  static   : F = A_geo @ Y           (no CASA training)
  learned  : F = A_learned @ Y       (CASA no_geo, alpha=0)        -- stage-1 GD trains CASA
  adaptive : F = W_t @ Y             (CASA gate, neighbours_only)  -- stage-1 GD trains CASA
Stage 2 (all modes): freeze F, fit canonical GSTAR coefficients (shared phi_own, phi_sp +
per-location intercept; + shared climate-lag coefs for GSTARX) by stacked OLS.

Validation: static GSTAR / GSTARX at K=12 must reproduce canonical GSTAR (81.25 / 79.04 mm).
"""
import sys, csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from casa_torch import CASATorch
from exp_gstar_canonical import (normalize, _design, idn, NID, A, Y, X, T, N,
                                 MIN_TRAIN, N_SPLITS, FOLD, CASA_DIR)

K = 12
CASA_EPOCHS, CASA_LR, CLIP, SEED = 800, 1e-3, 1.0, 42
A_t = torch.tensor(A, dtype=torch.float32)
idn_t = torch.tensor(idn, dtype=torch.long)
Xnino = torch.tensor(X[:, 0], dtype=torch.float32)
Xdmi  = torch.tensor(X[:, 1], dtype=torch.float32)


def casa_fsrc(mode, Yn, tr_end, use_climate, seed):
    """Return F_src (T,N): per-month spatial lag from the chosen weight matrix.
    For learned/adaptive, stage-1 trains CASA (GD) with a shared-parameter linear head."""
    if mode == 'static':
        return Yn @ A.T, 1.0
    ablation = 'no_geo' if mode == 'learned' else 'neighbours_only'
    nmask = (A_t > 0) if ablation == 'neighbours_only' else None
    Yn_t = torch.tensor(Yn, dtype=torch.float32)
    Xn_loc = torch.tensor(((X - X[:tr_end].min(0)) / (X[:tr_end].max(0) - X[:tr_end].min(0) + 1e-9)),
                          dtype=torch.float32)
    torch.manual_seed(seed)
    casa = CASATorch(N=N, b_alpha_init=2.0, ablation=ablation, freeze_alpha=None)
    phi_own = nn.Parameter(torch.zeros(K)); phi_sp = nn.Parameter(torch.zeros(K))
    bias = nn.Parameter(torch.zeros(NID))
    params = list(casa.parameters()) + [phi_own, phi_sp, bias]
    if use_climate:
        phi_cl = nn.Parameter(torch.zeros(K, 2)); params.append(phi_cl)
    opt = torch.optim.Adam(params, lr=CASA_LR)
    ts = torch.arange(K, tr_end)
    tgt = Yn_t[ts][:, idn_t]                                          # (n, NID)
    for _ in range(CASA_EPOCHS):
        opt.zero_grad()
        F_all, _, a_all, _ = casa(Yn_t, nino_t=Xnino, dmi_t=Xdmi, A_geo=A_t, neighbor_mask=nmask)
        own = torch.stack([Yn_t[ts - (k + 1)][:, idn_t] for k in range(K)], dim=2)   # (n,NID,K)
        sp  = torch.stack([F_all[ts - (k + 1)][:, idn_t] for k in range(K)], dim=2)
        yh = (torch.einsum('nik,k->ni', own, phi_own)
              + torch.einsum('nik,k->ni', sp, phi_sp) + bias)
        if use_climate:
            clim = torch.stack([torch.stack([Xn_loc[ts - (k + 1), 0], Xn_loc[ts - (k + 1), 1]], 1)
                                for k in range(K)], dim=1)             # (n,K,2)
            yh = yh + torch.einsum('nkc,kc->n', clim, phi_cl)[:, None]
        loss = F.mse_loss(yh, tgt) * 0.5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        opt.step()
    with torch.no_grad():
        F_all, _, a_all, _ = casa(Yn_t, nino_t=Xnino, dmi_t=Xdmi, A_geo=A_t, neighbor_mask=nmask)
    return F_all.detach().numpy(), float(a_all.mean())


def run(mode, use_climate):
    rmses, alphas = [], []
    for f in range(N_SPLITS):
        tr_end = MIN_TRAIN + f * FOLD; te_s, te_e = tr_end, tr_end + FOLD
        if te_e > T: break
        Yn, Xn, (ymin, ymax, yr) = normalize(Y, X, tr_end)
        F_src, a_mean = casa_fsrc(mode, Yn, tr_end, use_climate, SEED + f)
        tr_ts = np.arange(K, tr_end); te_ts = np.arange(te_s, te_e)
        D_tr, r_tr = _design(tr_ts, Yn, Xn, F_src, K, use_climate)
        beta, *_ = np.linalg.lstsq(D_tr, r_tr, rcond=None)
        D_te, _ = _design(te_ts, Yn, Xn, F_src, K, use_climate)
        pred = (D_te @ beta).reshape(len(te_ts), NID) * yr[idn] + ymin[idn]
        rmses.append(float(np.sqrt(((pred - Y[te_s:te_e][:, idn]) ** 2).mean())))
        alphas.append(a_mean if mode != 'learned' else 0.0)
    r = np.array(rmses); return r.mean(), r.std(), float(np.mean(alphas))


CONFIGS = [
    ("static   GSTAR",  'static',   False),
    ("learned  GSTAR",  'learned',  False),
    ("adaptive GSTAR",  'adaptive', False),
    ("static   GSTARX", 'static',   True),
    ("learned  GSTARX", 'learned',  True),
    ("adaptive GSTARX", 'adaptive', True),
]

if __name__ == "__main__":
    print(f"Canonical CASA-linear, K={K}  (static must match canonical GSTAR 81.25 / GSTARX 79.04)")
    print(f"{'config':18} | {'RMSE mm':>14} | {'mean alpha':>10}")
    print("-" * 50)
    rows = []
    for label, mode, uc in CONFIGS:
        t0 = time.time()
        m, s, al = run(mode, uc)
        print(f"{label:18} | {m:7.2f} +- {s:4.2f} | {al:10.3f}  ({time.time()-t0:.0f}s)")
        rows.append([label, K, mode, uc, round(m, 3), round(s, 3), round(al, 4)])
    with open(CASA_DIR + r"\canonical_casa_linear_rmse.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "K", "mode", "use_climate", "rmse_mean", "rmse_std", "alpha_mean"])
        w.writerows(rows)
    print("\nWrote canonical_casa_linear_rmse.csv")
