#!/usr/bin/env python3
"""Experiment: memory vs teleconnection. Run the canonical LSTM-GSTARX(12;1) coupled
5-fold walk-forward CV (mode window_f) WITH climate inputs (reproduce ~71.37) and
WITHOUT climate inputs (Nino3.4 & DMI zeroed -> X_norm=0). Reuses the real model/CV
from train_casa_n40. A_geo rebuilt from centroids (verified to match a_geo.npy).
"""
import sys, numpy as np
import train_casa_n40 as T

panel = np.load(r"data/panel_data.npz", allow_pickle=True)
Y = panel['Y_rainfall'].astype(float)
X = panel['X_climate'].astype(float)
dates = panel['dates']
region_ids = panel['region_ids']
idn = np.where(panel['region_countries'] == 'IDN')[0]
A_geo = np.load("data/a_geo.npy")

def run(Xin, label):
    res, _ = T.walk_forward_cv(
        Y, Xin, dates, A_geo, idn,
        n_splits=T.N_SPLITS, epochs=T.EPOCHS, lr=T.LR, seed=T.SEED,
        min_train=T.MIN_TRAIN, ablation='neighbours_only',
        b_alpha_init=T.B_ALPHA_INIT, alpha_entropy_reg=0.0,
        lookback=12, rich_context=False, coupled_fstar=True)
    r = np.array([d['rmse'] for d in res])
    print(f">>> {label}: RMSE = {r.mean():.2f} +- {r.std():.2f} mm | folds {np.round(r, 2)}")
    return float(r.mean()), float(r.std())

print("########## (12;1) WITH climate — should reproduce ~71.37 ##########")
m1, s1 = run(X, "with-climate (12;1)")
print("\n########## (12;1) NO climate — Nino3.4 & DMI zeroed ##########")
m0, s0 = run(np.zeros_like(X), "no-climate  (12;1)")
print("\n==================== RESULT ====================")
print(f"single-step K=1 (context)   : ~80.34 mm")
print(f"(12;1) WITH climate          : {m1:.2f} +- {s1:.2f} mm")
print(f"(12;1) NO climate (X zeroed) : {m0:.2f} +- {s0:.2f} mm")
print(f"--> climate-input contribution at K=12 : {m0 - m1:+.2f} mm")
print(f"--> memory contribution (K=1 -> 12, with climate) : {m1 - 80.34:+.2f} mm")
