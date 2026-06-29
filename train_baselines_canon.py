#!/usr/bin/env python3
"""Canonical GSTAR(1;1)/GSTARX(1;1) baselines -- stacked shared-parameter OLS.

Replaces the per-location OLS GSTAR/GSTARX in train_baselines.py with the canonical estimator
(homogeneous dynamics shared across locations + per-location intercept, one stacked OLS), emits
per-month predictions in the SAME npz format, and MERGES the unchanged lstm_static predictions
from the existing outputs/predictions_baselines.npz (backed up to *_perloc.npz first).

Then re-run compare_models.py to refresh cv_summary_table.csv + dm_test_matrix.csv +
regime_stratified.csv with canonical linear baselines (neural models untouched).

Local validation (build_a_geo): canonical GSTAR(1;1)=88.79, GSTARX(1;1)=88.23.
On the VPS it loads data/a_geo.npy (identical KNN k=5).
"""
import sys, shutil
from pathlib import Path
import numpy as np

PANEL = 'data/panel_data.npz'
A_GEO = 'data/a_geo.npy'
OUT   = 'outputs'
MIN_TRAIN, N_SPLITS = 380, 5

panel = np.load(PANEL, allow_pickle=True)
Y = panel['Y_rainfall'].astype(float)
X = panel['X_climate'].astype(float)
dates = panel['dates']
idn = np.where(panel['region_countries'] == 'IDN')[0]
try:
    A = np.load(A_GEO)
except FileNotFoundError:
            A = np.load("data/a_geo.npy")
T, N = Y.shape
FOLD = (T - MIN_TRAIN) // N_SPLITS
NID = len(idn)
K = 1   # baselines are GSTAR(1;1)


def normalize(Y, X, tr):
    ymin, ymax = Y[:tr].min(0), Y[:tr].max(0); yr = ymax - ymin + 1e-9
    xmin, xmax = X[:tr].min(0), X[:tr].max(0); xr = xmax - xmin + 1e-9
    return (Y - ymin) / yr, (X - xmin) / xr, (ymin, ymax, yr)


def _design(ts, Yn, Xn, F, use_climate):
    n = len(ts)
    own = np.stack([Yn[ts - k][:, idn] for k in range(1, K + 1)], axis=2)   # (n,NID,K)
    sp  = np.stack([F [ts - k][:, idn] for k in range(1, K + 1)], axis=2)
    feat = np.concatenate([own, sp], axis=2)
    if use_climate:
        clim = np.stack([np.stack([Xn[ts - k, 0], Xn[ts - k, 1]], axis=1)
                         for k in range(1, K + 1)], axis=1).reshape(n, 2 * K)
        clim = np.repeat(clim[:, None, :], NID, axis=1)
        feat = np.concatenate([feat, clim], axis=2)
    feat = feat.reshape(n * NID, feat.shape[2])
    inter = np.tile(np.eye(NID), (n, 1))
    return np.concatenate([inter, feat], axis=1), Yn[ts][:, idn].reshape(n * NID)


def canon_predictions(use_climate):
    """Return dict fold->(te_size, NID) predictions in mm, and per-fold RMSE."""
    preds, rmses = {}, []
    for f in range(N_SPLITS):
        tr_end = MIN_TRAIN + f * FOLD; te_s, te_e = tr_end, tr_end + FOLD
        if te_e > T: break
        Yn, Xn, (ymin, ymax, yr) = normalize(Y, X, tr_end)
        F = Yn @ A.T
        D_tr, r_tr = _design(np.arange(K, tr_end), Yn, Xn, F, use_climate)
        beta, *_ = np.linalg.lstsq(D_tr, r_tr, rcond=None)
        te_ts = np.arange(te_s, te_e)
        D_te, _ = _design(te_ts, Yn, Xn, F, use_climate)
        pred = (D_te @ beta).reshape(len(te_ts), NID) * yr[idn] + ymin[idn]
        preds[f + 1] = pred.astype(np.float64)
        rmses.append(float(np.sqrt(((pred - Y[te_s:te_e][:, idn]) ** 2).mean())))
    return preds, np.array(rmses)


if __name__ == "__main__":
    gstar_preds, gr = canon_predictions(False)
    gstarx_preds, gxr = canon_predictions(True)
    print(f"canonical GSTAR(1;1)  RMSE = {gr.mean():.2f} +- {gr.std(ddof=1):.2f}  (target 88.79)")
    print(f"canonical GSTARX(1;1) RMSE = {gxr.mean():.2f} +- {gxr.std(ddof=1):.2f}  (target 88.23)")

    npz_path = Path(OUT) / 'predictions_baselines.npz'
    if not npz_path.exists():
        print(f"\n[local] {npz_path} not found -- validation only, no merge written.")
        sys.exit(0)
    # back up the per-location predictions, then merge canonical gstar/gstarx + old lstm_static
    backup = Path(OUT) / 'predictions_baselines_perloc.npz'
    if not backup.exists():
        shutil.copy(npz_path, backup)
        print(f"backed up per-location predictions -> {backup}")
    old = np.load(npz_path, allow_pickle=True)
    out = {}
    for k in range(1, N_SPLITS + 1):
        out[f'y_hat_gstar_fold_{k}']       = gstar_preds[k]
        out[f'y_hat_gstarx_fold_{k}']      = gstarx_preds[k]
        out[f'y_hat_lstm_static_fold_{k}'] = old[f'y_hat_lstm_static_fold_{k}']
        out[f'y_true_fold_{k}']            = old[f'y_true_fold_{k}']
        out[f'dates_fold_{k}']             = old[f'dates_fold_{k}']
    np.savez_compressed(npz_path, **out)
    print(f"wrote canonical baselines into {npz_path} (lstm_static preserved). "
          f"Now run: python compare_models.py")
