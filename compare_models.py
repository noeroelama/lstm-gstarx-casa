#!/usr/bin/env python3
"""
compare_models.py -- Merge CASA ablations + baseline predictions, produce:
  - cv_summary_table.csv : RMSE/MAE per model (mean +/- std across folds)
  - dm_test_matrix.csv   : pairwise Diebold-Mariano (HLN) p-values
  - regime_stratified.csv: RMSE per ENSO regime per model

Assumes train_casa_n40.py and train_baselines.py have been run for all
target ablations. Reads predictions NPZs from outputs/.

DM test follows Diebold & Mariano (1995) with Harvey-Leybourne-Newbold
(1997) small-sample correction. For h=1 one-step forecasts:
    DM_HLN = (d_mean / sqrt(gamma_0/T)) * sqrt((T-1)/T)
under H0 (equal predictive accuracy), DM_HLN ~ t_{T-1}.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUTPUTS_DIR = 'outputs'
PANEL_NPZ   = 'data/panel_data.npz'

def discover_models():
    """Auto-discover all CASA experiment NPZ + baselines bundle in OUTPUTS_DIR.

    Returns list of tuples (display_name, npz_filename, key_prefix).
    key_prefix is None for single-model NPZ, or 'gstar'/'gstarx'/'lstm_static'
    for the bundled baselines NPZ.
    """
    models = []
    out_dir = Path(OUTPUTS_DIR)
    if not out_dir.exists():
        return models
    # CASA experiments: one NPZ per ablation/experiment
    for p in sorted(out_dir.glob('predictions_casa_*.npz')):
        name = p.stem.replace('predictions_casa_', 'casa_')
        models.append((name, p.name, None))
    # Baselines bundle: 3 models share one NPZ
    bpath = out_dir / 'predictions_baselines.npz'
    if bpath.exists():
        for prefix in ('gstar', 'gstarx', 'lstm_static'):
            models.append((prefix, bpath.name, prefix))
    return models


MODELS = discover_models()


# ============= LOAD ===========================================
def load_predictions():
    cache = {}
    out = {}
    for name, fname, prefix in MODELS:
        path = f'{OUTPUTS_DIR}/{fname}'
        if not Path(path).exists():
            print(f"  [WARN] {path} missing, skipping {name}")
            continue
        if fname not in cache:
            cache[fname] = np.load(path, allow_pickle=True)
        npz = cache[fname]

        # find available fold numbers
        fold_nums = sorted(set(
            int(k.split('fold_')[1])
            for k in npz.files if 'fold_' in k
        ))
        folds = []
        for n in fold_nums:
            y_hat_key = (f'y_hat_{prefix}_fold_{n}' if prefix
                         else f'y_hat_fold_{n}')
            if y_hat_key not in npz.files:
                continue
            folds.append({
                'y_hat':  npz[y_hat_key],
                'y_true': npz[f'y_true_fold_{n}'],
                'dates':  npz[f'dates_fold_{n}'],
            })
        out[name] = folds
    return out


# ============= 1. CV SUMMARY ==================================
def compute_cv_summary(preds):
    rows = []
    for name, folds in preds.items():
        for i, f in enumerate(folds):
            err = f['y_hat'] - f['y_true']
            rmse = float(np.sqrt((err ** 2).mean()))
            mae  = float(np.abs(err).mean())
            rows.append({'model': name, 'fold': i + 1,
                         'rmse': rmse, 'mae': mae,
                         'test_size': f['y_true'].shape[0]})
    df = pd.DataFrame(rows)
    summary = df.groupby('model').agg(
        rmse_mean=('rmse', 'mean'), rmse_std=('rmse', 'std'),
        mae_mean =('mae',  'mean'), mae_std =('mae',  'std'),
        n_folds  =('fold', 'count'),
    ).reset_index().sort_values('rmse_mean').reset_index(drop=True)
    return df, summary


# ============= 2. DIEBOLD-MARIANO (HLN) =======================
def dm_hln(errors_A, errors_B, h=1):
    """DM with Harvey-Leybourne-Newbold small-sample correction.

    errors_A, errors_B : (T,) per-month squared errors averaged across IDN
    h                  : forecast horizon (1 for one-step ahead)
    Returns: (dm_stat_hln, p_value_two_sided). Negative dm => A better.
    """
    d = np.asarray(errors_A) - np.asarray(errors_B)
    T = len(d)
    if T < 2:
        return float('nan'), float('nan')
    d_mean = d.mean()
    gamma_0 = d.var(ddof=1)
    if gamma_0 <= 0:
        return float('nan'), float('nan')
    dm_stat = d_mean / np.sqrt(gamma_0 / T)
    correction = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln_stat = dm_stat * correction
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_hln_stat), df=T - 1))
    return float(dm_hln_stat), float(p_value)


def compute_pairwise_dm(preds):
    err_per_model = {}
    for name, folds in preds.items():
        all_e = []
        for f in folds:
            e = ((f['y_hat'] - f['y_true']) ** 2).mean(axis=1)   # (T_fold,)
            all_e.append(e)
        err_per_model[name] = np.concatenate(all_e)

    names = list(err_per_model)
    rows = []
    for i, A in enumerate(names):
        for j, B in enumerate(names):
            if i >= j:
                continue
            dm, p = dm_hln(err_per_model[A], err_per_model[B])
            rows.append({
                'model_A': A, 'model_B': B,
                'dm_hln': dm, 'p_value': p,
                'A_better': (dm < 0) if not np.isnan(dm) else None,
                'sig_005': (p < 0.05) if not np.isnan(p) else None,
                'sig_001': (p < 0.01) if not np.isnan(p) else None,
            })
    return pd.DataFrame(rows)


# ============= 3. REGIME STRATIFIED ===========================
def regime_stratified(preds, nino_threshold=0.5):
    panel = np.load(PANEL_NPZ, allow_pickle=True)
    dates_panel = pd.to_datetime(panel['dates'])
    nino_panel  = panel['X_climate'][:, 0]
    # Map: date -> nino value
    nino_by_date = dict(zip(dates_panel, nino_panel))

    rows = []
    for name, folds in preds.items():
        for f in folds:
            err_t = ((f['y_hat'] - f['y_true']) ** 2).mean(axis=1)
            dts = pd.to_datetime(f['dates'])
            nino_te = np.array([nino_by_date[d] for d in dts])
            regime = np.full(len(nino_te), 'Neutral', dtype=object)
            regime[nino_te >  nino_threshold] = 'ElNino'
            regime[nino_te < -nino_threshold] = 'LaNina'
            for r in ('ElNino', 'Neutral', 'LaNina'):
                m = regime == r
                if m.sum() == 0:
                    continue
                rows.append({
                    'model': name, 'regime': r,
                    'n_months': int(m.sum()),
                    'rmse_sq_mean': float(err_t[m].mean()),
                })
    df = pd.DataFrame(rows)
    # Aggregate across folds: weighted by n_months
    grouped = df.groupby(['model', 'regime']).apply(
        lambda g: pd.Series({
            'n_months_total': int(g['n_months'].sum()),
            'rmse': float(np.sqrt(
                (g['rmse_sq_mean'] * g['n_months']).sum() / g['n_months'].sum()
            )),
        })
    ).reset_index()
    pivot = grouped.pivot(index='model', columns='regime', values='rmse')
    return grouped, pivot


# ============= MAIN ===========================================
def main():
    print("=== Loading predictions ===")
    preds = load_predictions()
    print(f"  loaded {len(preds)} models")
    for n, f in preds.items():
        total = sum(fld['y_true'].shape[0] for fld in f)
        print(f"    {n:25s}: {len(f)} folds, {total} test months")

    # 1. CV Summary
    print("\n=== 1. CV Summary (per-fold mean +/- std) ===")
    df_cv, summary = compute_cv_summary(preds)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    summary.to_csv(f'{OUTPUTS_DIR}/cv_summary_table.csv', index=False)

    # 2. Pairwise DM (HLN) test
    print("\n=== 2. Pairwise DM (HLN) Test ===")
    df_dm = compute_pairwise_dm(preds)
    sig = df_dm[df_dm['p_value'] < 0.05].sort_values('p_value')
    print(f"  significant pairs (p<0.05): {len(sig)} of {len(df_dm)}")
    if len(sig):
        cols = ['model_A', 'model_B', 'dm_hln', 'p_value', 'A_better', 'sig_001']
        print(sig[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:.4f}"))
    df_dm.to_csv(f'{OUTPUTS_DIR}/dm_test_matrix.csv', index=False)

    # 3. Regime-stratified RMSE
    print("\n=== 3. Regime-Stratified RMSE (mm) ===")
    df_regime, pivot = regime_stratified(preds)
    print(pivot.round(2).to_string())
    df_regime.to_csv(f'{OUTPUTS_DIR}/regime_stratified.csv', index=False)

    print("\nAll outputs saved to outputs/")


if __name__ == '__main__':
    main()
