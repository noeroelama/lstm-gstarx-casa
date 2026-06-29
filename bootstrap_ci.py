#!/usr/bin/env python3
"""
bootstrap_ci.py -- Compute bootstrap 95% CI for RMSE per model.

Per-month squared error vectors (across IDN provinces averaged) are
bootstrap-resampled B times with replacement. RMSE = sqrt(mean(MSE)) per
resample; 2.5%-97.5% quantiles yield 95% CI.

Targets top-K models by aggregate RMSE (default top 5) plus LSTM-static
baseline as reference.

Output: outputs/bootstrap_ci.csv with columns
  [model, rmse_mean, rmse_ci_lo, rmse_ci_hi, mae_mean, mae_ci_lo, mae_ci_hi]
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUTS_DIR = 'outputs'
B = 1000               # bootstrap resamples
N_TOP = 5              # report top-K models
SEED = 42
INCLUDE_BASELINES = ('lstm_static', 'gstarx', 'gstar')


def discover_models():
    """Mirror compare_models.py discovery logic."""
    models = []
    for p in sorted(Path(OUTPUTS_DIR).glob('predictions_casa_*.npz')):
        name = p.stem.replace('predictions_casa_', 'casa_')
        models.append((name, p.name, None))
    bpath = Path(OUTPUTS_DIR) / 'predictions_baselines.npz'
    if bpath.exists():
        for prefix in ('gstar', 'gstarx', 'lstm_static'):
            models.append((prefix, bpath.name, prefix))
    return models


def load_per_month_errors(models):
    """For each model: concatenate per-month MSE (averaged over IDN) across
    all folds. Returns dict {model_name: (T_total,) MSE array}."""
    cache = {}
    out = {}
    for name, fname, prefix in models:
        path = f'{OUTPUTS_DIR}/{fname}'
        if not Path(path).exists():
            continue
        if fname not in cache:
            cache[fname] = np.load(path, allow_pickle=True)
        npz = cache[fname]
        fold_nums = sorted(set(
            int(k.split('fold_')[1])
            for k in npz.files if 'fold_' in k
        ))
        mse_all = []
        ae_all  = []
        for n in fold_nums:
            yhat_key = (f'y_hat_{prefix}_fold_{n}' if prefix
                        else f'y_hat_fold_{n}')
            if yhat_key not in npz.files:
                continue
            yhat = npz[yhat_key]
            ytrue = npz[f'y_true_fold_{n}']
            err = yhat - ytrue                  # (T_fold, 38)
            mse_t = (err ** 2).mean(axis=1)     # (T_fold,)
            ae_t  = np.abs(err).mean(axis=1)    # (T_fold,)
            mse_all.append(mse_t)
            ae_all.append(ae_t)
        if mse_all:
            out[name] = (np.concatenate(mse_all),
                         np.concatenate(ae_all))
    return out


def bootstrap_metric(values, b=B, rng=None):
    """Bootstrap (B,)-array of sqrt(mean(values_resample))-equivalent.

    For RMSE: values = per-month MSE; reduction = sqrt(mean).
    For MAE:  values = per-month |err|; reduction = mean.
    """
    n = len(values)
    rng = rng or np.random.default_rng(SEED)
    samples = rng.choice(values, size=(b, n), replace=True)
    return samples


def main():
    models = discover_models()
    err_dict = load_per_month_errors(models)

    # Rank by aggregate RMSE
    ranking = sorted(
        ((name, float(np.sqrt(mse.mean())))
         for name, (mse, ae) in err_dict.items()),
        key=lambda x: x[1]
    )
    print("\n=== Model ranking (aggregate RMSE, mm) ===")
    for i, (name, rmse) in enumerate(ranking):
        print(f"  {i+1:2d}. {name:40s} {rmse:7.3f}")

    # Pick top-N + baselines
    top_names = [n for n, _ in ranking[:N_TOP]]
    for b in INCLUDE_BASELINES:
        if b in err_dict and b not in top_names:
            top_names.append(b)

    print(f"\n=== Bootstrap 95% CI (B = {B} resamples, seed = {SEED}) ===")
    rows = []
    rng = np.random.default_rng(SEED)
    for name in top_names:
        mse_per_month, ae_per_month = err_dict[name]
        n_months = len(mse_per_month)
        # Resample indices ONCE per model
        idx = rng.choice(n_months, size=(B, n_months), replace=True)
        mse_resamp = mse_per_month[idx]   # (B, n_months)
        ae_resamp  = ae_per_month[idx]
        rmse_b = np.sqrt(mse_resamp.mean(axis=1))
        mae_b  = ae_resamp.mean(axis=1)
        rmse_lo, rmse_hi = np.quantile(rmse_b, [0.025, 0.975])
        mae_lo,  mae_hi  = np.quantile(mae_b,  [0.025, 0.975])
        rmse_mean = float(np.sqrt(mse_per_month.mean()))
        mae_mean  = float(ae_per_month.mean())
        rows.append({
            'model':       name,
            'rmse_mean':   rmse_mean,
            'rmse_ci_lo':  float(rmse_lo),
            'rmse_ci_hi':  float(rmse_hi),
            'mae_mean':    mae_mean,
            'mae_ci_lo':   float(mae_lo),
            'mae_ci_hi':   float(mae_hi),
            'n_months':    n_months,
        })
        print(f"  {name:40s} RMSE {rmse_mean:6.2f} "
              f"[{rmse_lo:6.2f}, {rmse_hi:6.2f}]   "
              f"MAE {mae_mean:6.2f} "
              f"[{mae_lo:6.2f}, {mae_hi:6.2f}]")

    df = pd.DataFrame(rows)
    out_path = Path(OUTPUTS_DIR) / 'bootstrap_ci.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")


if __name__ == '__main__':
    main()
