#!/usr/bin/env python3
"""
per_province_rmse.py -- Per-province RMSE breakdown across top-N models.

For each IDN province (38 forecast targets), compute RMSE across all
test months pooled (5 folds × ~20 months = ~100 months). Returns:
  - outputs/per_province_rmse.csv (long form: model × province × rmse)
  - outputs/per_province_rmse_matrix.csv (pivot: province × model)
  - Identifies "hard provinces" (top decile RMSE) and "easy provinces"
    (bottom decile RMSE) per model and across models.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUTS_DIR = 'outputs'
PANEL_NPZ   = 'data/panel_data.npz'
TOP_MODELS  = [
    'casa_neighbours_only_lb24',     # 1
    'casa_neighbours_only_lb12',     # 5 (paper default)
    'casa_neighbours_only',          # baseline single-step CASA
    'lstm_static',                   # single-step deep baseline
    'gstarx',                        # linear baseline
]


def discover_npz(model_name):
    """Return (path, prefix). For CASA, path = predictions_<model>.npz, no prefix.
    For baselines (gstar/gstarx/lstm_static), path = predictions_baselines.npz with prefix."""
    if model_name in ('gstar', 'gstarx', 'lstm_static'):
        return f'{OUTPUTS_DIR}/predictions_baselines.npz', model_name
    return f'{OUTPUTS_DIR}/predictions_{model_name}.npz', None


def load_per_province_errors(model_name):
    """Returns (T_total, 38) array of (y_hat - y_true) per province."""
    path, prefix = discover_npz(model_name)
    if not Path(path).exists():
        return None
    npz = np.load(path, allow_pickle=True)
    fold_nums = sorted(set(
        int(k.split('fold_')[1]) for k in npz.files if 'fold_' in k
    ))
    err_all = []
    for n in fold_nums:
        yhat_key = (f'y_hat_{prefix}_fold_{n}' if prefix
                    else f'y_hat_fold_{n}')
        if yhat_key not in npz.files:
            continue
        yhat  = npz[yhat_key]
        ytrue = npz[f'y_true_fold_{n}']
        err_all.append(yhat - ytrue)
    return np.concatenate(err_all, axis=0) if err_all else None


def main():
    # Load province metadata
    panel = np.load(PANEL_NPZ, allow_pickle=True)
    # panel['regions'] is 40 nodes; 38 IDN are forecast targets
    if 'regions' in panel.files:
        all_regions = list(panel['regions'])
    else:
        # fallback: chirps_monthly_v10.csv ordering
        regions_csv = pd.read_csv('data/chirps_monthly_v10.csv')
        all_regions = (regions_csv[['country', 'region_id']]
                       .drop_duplicates()
                       .sort_values(['country', 'region_id'])['region_id']
                       .tolist())
    # Filter to IDN
    idn_regions = [r for r in all_regions if r.startswith('ID-')]
    if len(idn_regions) != 38:
        print(f"  [WARN] expected 38 IDN regions, got {len(idn_regions)}")

    rows = []
    print(f"\n=== Per-province RMSE per model ===\n")
    print(f"{'province':10s}", end='')
    for m in TOP_MODELS:
        print(f" {m[:12]:>12s}", end='')
    print()
    print('-' * (10 + 13 * len(TOP_MODELS)))

    per_model_err = {}
    for m in TOP_MODELS:
        err = load_per_province_errors(m)
        if err is None:
            print(f"  [WARN] {m} npz not found, skipping")
            continue
        per_model_err[m] = err

    # Per-province RMSE per model
    for j, prov in enumerate(idn_regions):
        print(f"{prov:10s}", end='')
        for m in TOP_MODELS:
            if m not in per_model_err:
                print(f" {'NA':>12s}", end='')
                continue
            err = per_model_err[m]
            rmse_j = float(np.sqrt((err[:, j] ** 2).mean()))
            print(f" {rmse_j:>12.2f}", end='')
            rows.append({'province': prov, 'model': m, 'rmse': rmse_j})
        print()

    df = pd.DataFrame(rows)
    out_long = Path(OUTPUTS_DIR) / 'per_province_rmse.csv'
    df.to_csv(out_long, index=False)
    pivot = df.pivot(index='province', columns='model', values='rmse')
    out_pivot = Path(OUTPUTS_DIR) / 'per_province_rmse_matrix.csv'
    pivot.to_csv(out_pivot)

    # Hard / easy provinces analysis
    print("\n=== HARD provinces (top decile RMSE across models) ===")
    median_rmse = pivot.median(axis=1).sort_values(ascending=False)
    print(f"  Top 10 hardest provinces (median RMSE across models):")
    for prov, rmse in median_rmse.head(10).items():
        print(f"    {prov:10s} median {rmse:6.2f}")

    print(f"\n  Top 10 easiest provinces:")
    for prov, rmse in median_rmse.tail(10).items():
        print(f"    {prov:10s} median {rmse:6.2f}")

    # Model improvement (lb24 vs lstm_static)
    if 'casa_neighbours_only_lb24' in pivot.columns and 'lstm_static' in pivot.columns:
        print("\n=== Per-province IMPROVEMENT (lb24 vs lstm_static) ===")
        delta = pivot['casa_neighbours_only_lb24'] - pivot['lstm_static']
        delta_sorted = delta.sort_values()
        print(f"  Top 10 provinces with LARGEST IMPROVEMENT (lb24 << lstm_static):")
        for prov, d in delta_sorted.head(10).items():
            print(f"    {prov:10s} Δ {d:+6.2f} mm   ({pivot.loc[prov, 'lstm_static']:.1f} → {pivot.loc[prov, 'casa_neighbours_only_lb24']:.1f})")

        print(f"\n  Top 10 provinces with SMALLEST IMPROVEMENT (lb24 ~ lstm_static):")
        for prov, d in delta_sorted.tail(10).items():
            print(f"    {prov:10s} Δ {d:+6.2f} mm")

    print(f"\nSaved {out_long}")
    print(f"Saved {out_pivot}")


if __name__ == '__main__':
    main()
