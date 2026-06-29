#!/usr/bin/env python3
"""Reference skill (vs climatology + persistence) and MAE for the paper.

The manuscript reports RMSE in mm with no reference skill. This adds two naive
references on the SAME walk-forward folds, then the RMSE-based skill score
SS = 1 - RMSE_model / RMSE_ref (>0 means the model beats the reference):

  - climatology : forecast = training-period mean rainfall for that calendar
                  month, per province (the seasonal cycle; walk-forward, no leak).
  - persistence : forecast = previous month's observed rainfall, y_hat_t = y_{t-1}.

Model RMSE/MAE are read straight from the saved prediction NPZs so they match
cv_summary_table.csv exactly. Per-fold metrics are averaged across the 5 folds,
identical to the headline reporting. Output: outputs/skill_scores.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("data")
OUT = Path("outputs")
MIN_TRAIN, FOLD, NSPLITS = 380, 20, 5

panel = np.load(ROOT / "panel_data.npz", allow_pickle=True)
Y = panel['Y_rainfall'].astype(float)
dates = pd.to_datetime(panel['dates'])
idn = np.where(panel['region_countries'] == 'IDN')[0]
moy = dates.month.values
T = Y.shape[0]


def fold_metrics(pred_fn):
    """pred_fn(train_end, test_idx) -> (n_test, 38) forecast in mm. Returns
    (mean RMSE, mean MAE) across folds, per-fold-mean like the headline."""
    rmse, mae = [], []
    for f in range(NSPLITS):
        tr_end = MIN_TRAIN + f * FOLD
        idx = np.arange(tr_end, tr_end + FOLD)
        yhat = pred_fn(tr_end, idx)
        ytrue = Y[idx][:, idn]
        e = yhat - ytrue
        rmse.append(np.sqrt((e ** 2).mean()))
        mae.append(np.abs(e).mean())
    return float(np.mean(rmse)), float(np.mean(mae))


def clim_pred(tr_end, idx):
    out = np.empty((len(idx), len(idn)))
    train_moy = moy[:tr_end]
    for i, ti in enumerate(idx):
        sel = train_moy == moy[ti]
        out[i] = Y[:tr_end][sel][:, idn].mean(axis=0)
    return out


def pers_pred(tr_end, idx):
    return Y[idx - 1][:, idn]


def model_metrics(npz_name, prefix=None):
    """Per-fold-mean RMSE/MAE read from a predictions NPZ (matches cv_summary)."""
    npz = np.load(OUT / npz_name, allow_pickle=True)
    rmse, mae = [], []
    for k in range(1, NSPLITS + 1):
        yk = f'y_hat_{prefix}_fold_{k}' if prefix else f'y_hat_fold_{k}'
        e = npz[yk] - npz[f'y_true_fold_{k}']
        rmse.append(np.sqrt((e ** 2).mean()))
        mae.append(np.abs(e).mean())
    return float(np.mean(rmse)), float(np.mean(mae))


# references
clim_rmse, clim_mae = fold_metrics(clim_pred)
pers_rmse, pers_mae = fold_metrics(pers_pred)

# models (read from saved predictions)
models = {
    'GSTAR(1;1)':            model_metrics('predictions_baselines.npz', 'gstar'),
    'GSTARX(1;1)':           model_metrics('predictions_baselines.npz', 'gstarx'),
    'LSTM-GSTARX(1;1)':      model_metrics('predictions_casa_neighbours_only.npz'),
    'LSTM-GSTARX(12;0)+F*':  model_metrics('predictions_casa_neighbours_only_lb12.npz'),
    'LSTM-GSTARX(12;1)':     model_metrics('predictions_casa_neighbours_only_lb12_gstarK1.npz'),
}

print(f"References (per-fold-mean, mm):")
print(f"  climatology : RMSE {clim_rmse:6.2f}  MAE {clim_mae:6.2f}")
print(f"  persistence : RMSE {pers_rmse:6.2f}  MAE {pers_mae:6.2f}")
print(f"\n{'model':22s} {'RMSE':>7s} {'MAE':>7s} {'SS_clim':>8s} {'SS_pers':>8s}")
rows = [dict(reference='climatology', rmse=clim_rmse, mae=clim_mae),
        dict(reference='persistence', rmse=pers_rmse, mae=pers_mae)]
for name, (rm, ma) in models.items():
    ss_c = 1 - rm / clim_rmse
    ss_p = 1 - rm / pers_rmse
    print(f"{name:22s} {rm:7.2f} {ma:7.2f} {ss_c:8.3f} {ss_p:8.3f}")
    rows.append(dict(model=name, rmse=rm, mae=ma, ss_clim=ss_c, ss_pers=ss_p))

pd.DataFrame(rows).to_csv(OUT / 'skill_scores.csv', index=False)
print(f"\nSaved {OUT / 'skill_scores.csv'}")
