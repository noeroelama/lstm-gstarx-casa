#!/usr/bin/env python3
"""
train_baselines.py -- Baseline models for CASA comparison (N=40, IDN forecast).

Three baselines (per spec sect 7.2):
  1. GSTAR(1;1)         -- linear, OLS, no exogenous
  2. GSTARX(1;1)        -- linear, OLS, with exogenous (Nino 3.4, DMI)
  3. LSTM-GSTARX static -- deep, F_t = A_geo @ Y_prev (no CASA adaptation)

All three use the same data, same fold splits, same seed as
train_casa_n40.py for fair downstream DM-test comparison.

Saves per-fold predictions for all three models to
outputs/predictions_baselines.npz.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============= CONFIG (mirror train_casa_n40.py) ==============
PANEL_NPZ    = 'data/panel_data.npz'
A_GEO_NPY    = 'data/a_geo.npy'
OUTPUT_DIR   = 'outputs'

N_TOTAL      = 40
N_IDN        = 38
N_EXOG       = 2
HIDDEN_SIZE  = 64
EPOCHS       = 200
LR           = 1e-3
CLIP_NORM    = 1.0
N_SPLITS     = 5
MIN_TRAIN    = 380
SEED         = 42


# ============= GSTAR / GSTARX (canonical stacked LS) ==========
def _stacked_ols(feat_by_loc, target_by_loc):
    """Single stacked least-squares fit of the GSTAR baseline.

    `feat_by_loc` is an (N, T-1, m) array of the m regressors for each of the N
    locations; `target_by_loc` is (N, T-1). The N per-location parameter vectors
    (the diagonal entries phi^(1..N) of the GSTAR parameter matrices) are stacked
    into one block-diagonal design over every (location, time) pair and solved
    with ONE ordinary-least-squares fit -- the canonical GSTAR LS estimator of
    Ruchjana, Borovkova & Lopuhaa (2012). Because each location's columns enter
    only its own equations the normal matrix is block-diagonal, so this stacked
    estimate coincides exactly with the per-location least-squares solution; this
    is GSTAR, not its homogeneous STAR restriction that shares a single scalar
    across locations.

    Returns the (N, m) coefficient matrix.
    """
    N, n_t, m = feat_by_loc.shape
    nrows = N * n_t
    D = np.zeros((nrows, N * m))                       # block-diagonal design
    z = np.empty(nrows)
    for j in range(N):
        r0 = j * n_t
        D[r0:r0 + n_t, j * m:(j + 1) * m] = feat_by_loc[j]
        z[r0:r0 + n_t] = target_by_loc[j]
    beta, *_ = np.linalg.lstsq(D, z, rcond=None)
    return beta.reshape(N, m)


class GSTAR:
    """GSTAR(1;1) baseline, estimated by a single stacked OLS.

    For location j: y_j(t) = phi0_j + phi1_j y_j(t-1) + phi2_j (W y(t-1))_j + e_j(t),
    i.e. the GSTAR parameter matrices Phi_10, Phi_11 are N x N DIAGONAL with one
    entry per location. All N triples are estimated jointly by one stacked OLS
    (see _stacked_ols)."""

    n_feat = 3

    def __init__(self):
        self.coefs = None                              # (N, n_feat)

    def _feats(self, Y_norm, F_norm):
        T, N = Y_norm.shape
        # (N, T-1, 3): intercept, own-lag, spatial-lag
        ones = np.ones((N, T - 1))
        return np.stack([ones, Y_norm[:-1].T, F_norm[:-1].T], axis=2)

    def fit(self, Y_norm, W):
        F_norm = Y_norm @ W.T
        feats = self._feats(Y_norm, F_norm)
        self.coefs = _stacked_ols(feats, Y_norm[1:].T)
        return self

    def predict_onestep(self, Y_norm, W):
        F_norm = Y_norm @ W.T
        c = self.coefs                                 # (N, 3)
        preds = c[:, 0] + c[:, 1] * Y_norm[:-1] + c[:, 2] * F_norm[:-1]
        return np.clip(preds, 0, 1)


class GSTARX(GSTAR):
    """GSTARX(1;1) = GSTAR + lagged Nino 3.4 and DMI (per-location climate loadings)."""

    n_feat = 5

    def _feats(self, Y_norm, F_norm, X_norm):
        T, N = Y_norm.shape
        ones = np.ones((N, T - 1))
        nino = np.tile(X_norm[:-1, 0], (N, 1))         # shared regressor, per-loc coef
        dmi  = np.tile(X_norm[:-1, 1], (N, 1))
        return np.stack([ones, Y_norm[:-1].T, F_norm[:-1].T, nino, dmi], axis=2)

    def fit(self, Y_norm, W, X_norm):
        F_norm = Y_norm @ W.T
        feats = self._feats(Y_norm, F_norm, X_norm)
        self.coefs = _stacked_ols(feats, Y_norm[1:].T)
        return self

    def predict_onestep(self, Y_norm, W, X_norm):
        F_norm = Y_norm @ W.T
        c = self.coefs                                 # (N, 5)
        preds = (c[:, 0] + c[:, 1] * Y_norm[:-1] + c[:, 2] * F_norm[:-1]
                 + c[:, 3] * X_norm[:-1, 0][:, None] + c[:, 4] * X_norm[:-1, 1][:, None])
        return np.clip(preds, 0, 1)


# ============= LSTM-GSTARX static =============================
class LSTMGSTARX_Static(nn.Module):
    """LSTM-GSTARX with STATIC F = A_geo @ Y_prev (no CASA)."""

    def __init__(self, N=N_TOTAL, N_out=N_IDN, hidden=HIDDEN_SIZE, n_exog=N_EXOG):
        super().__init__()
        self.N, self.N_out = N, N_out
        d_in = 2 * N + n_exog
        self.lstm   = nn.LSTM(input_size=d_in, hidden_size=hidden, batch_first=True)
        self.linear = nn.Linear(hidden, N_out)

    def forward(self, Y_prev_norm, F_static, X_norm):
        x = torch.cat([Y_prev_norm, F_static, X_norm], dim=-1).unsqueeze(1)
        h, _ = self.lstm(x)
        return self.linear(h.squeeze(1))


# ============= NORMALIZATION ==================================
def normalize(Y, X, tr_end):
    Y_min = Y[:tr_end].min(axis=0); Y_max = Y[:tr_end].max(axis=0)
    Y_range = Y_max - Y_min + 1e-9
    Y_norm = (Y - Y_min) / Y_range
    X_min = X[:tr_end].min(axis=0); X_max = X[:tr_end].max(axis=0)
    X_range = X_max - X_min + 1e-9
    X_norm = (X - X_min) / X_range
    return Y_norm, X_norm, (Y_min, Y_max, Y_range)


def inverse_y_idn(y_hat_norm, idn_indices, Y_stats):
    Y_min, _, Y_range = Y_stats
    return y_hat_norm * Y_range[idn_indices] + Y_min[idn_indices]


def train_lstm_static(model, Y_prev, F_static, X_norm, Y_tgt,
                      epochs, lr, clip_norm):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        y_hat = model(Y_prev, F_static, X_norm)
        loss = F.mse_loss(y_hat, Y_tgt) * 0.5
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()


# ============= WALK-FORWARD CV ================================
def walk_forward_cv(Y, X, dates, A_geo, idn_indices, n_splits, epochs, lr,
                    seed, min_train, reuse_lstm=None):
    T, N = Y.shape
    fold_size = (T - min_train) // n_splits
    print(f"\nWalk-forward CV: T={T}, min_train={min_train}, "
          f"fold_size={fold_size}, n_splits={n_splits}")

    results = []
    predictions = {}

    for fold in range(n_splits):
        train_end  = min_train + fold * fold_size
        test_start = train_end
        test_end   = test_start + fold_size
        if test_end > T:
            break

        Y_norm, X_norm, Y_stats = normalize(Y, X, train_end)
        y_true_te_mm = Y[test_start:test_end][:, idn_indices]
        test_dates = dates[test_start:test_end]

        # --- 1. GSTAR ------------------------------------------------
        t0 = time.time()
        gstar = GSTAR().fit(Y_norm[:train_end], A_geo)
        pred_n = gstar.predict_onestep(Y_norm, A_geo)
        pred_te    = pred_n[test_start - 1:test_end - 1][:, idn_indices]
        pred_te_mm_gstar = inverse_y_idn(pred_te, idn_indices, Y_stats)
        rmse_gstar = float(np.sqrt(((pred_te_mm_gstar - y_true_te_mm) ** 2).mean()))
        mae_gstar  = float(np.abs(pred_te_mm_gstar - y_true_te_mm).mean())
        t_gstar = time.time() - t0

        # --- 2. GSTARX -----------------------------------------------
        t0 = time.time()
        gstarx = GSTARX().fit(Y_norm[:train_end], A_geo, X_norm[:train_end])
        pred_n = gstarx.predict_onestep(Y_norm, A_geo, X_norm)
        pred_te    = pred_n[test_start - 1:test_end - 1][:, idn_indices]
        pred_te_mm_gstarx = inverse_y_idn(pred_te, idn_indices, Y_stats)
        rmse_gstarx = float(np.sqrt(((pred_te_mm_gstarx - y_true_te_mm) ** 2).mean()))
        mae_gstarx  = float(np.abs(pred_te_mm_gstarx - y_true_te_mm).mean())
        t_gstarx = time.time() - t0

        # --- 3. LSTM-GSTARX static -----------------------------------
        # The static LSTM is a neural baseline and is unchanged by the GSTAR
        # estimator rewrite; --reuse-lstm loads its (already-trained) predictions
        # so a baseline rebuild does not retrain it (avoids cross-platform drift).
        t0 = time.time()
        if reuse_lstm is not None:
            y_hat_mm_lstm = np.asarray(reuse_lstm[f'y_hat_lstm_static_fold_{fold + 1}'],
                                       dtype=np.float64)
        else:
            F_static_all = Y_norm @ A_geo.T
            Y_prev_all = torch.tensor(Y_norm[:-1],             dtype=torch.float32)
            F_static_t = torch.tensor(F_static_all[:-1],       dtype=torch.float32)
            X_norm_t   = torch.tensor(X_norm[1:],              dtype=torch.float32)
            Y_tgt_t    = torch.tensor(Y_norm[1:, idn_indices], dtype=torch.float32)

            tr_lo, tr_hi = 0, train_end - 1
            te_lo, te_hi = train_end - 1, test_end - 1

            torch.manual_seed(seed + fold)
            np.random.seed(seed + fold)
            model = LSTMGSTARX_Static()
            train_lstm_static(
                model,
                Y_prev_all[tr_lo:tr_hi], F_static_t[tr_lo:tr_hi],
                X_norm_t[tr_lo:tr_hi],   Y_tgt_t[tr_lo:tr_hi],
                epochs=epochs, lr=lr, clip_norm=CLIP_NORM,
            )
            model.eval()
            with torch.no_grad():
                y_hat_n = model(
                    Y_prev_all[te_lo:te_hi], F_static_t[te_lo:te_hi],
                    X_norm_t[te_lo:te_hi],
                )
            y_hat_mm_lstm = inverse_y_idn(y_hat_n.numpy(), idn_indices, Y_stats)
        rmse_lstm = float(np.sqrt(((y_hat_mm_lstm - y_true_te_mm) ** 2).mean()))
        mae_lstm  = float(np.abs(y_hat_mm_lstm - y_true_te_mm).mean())
        t_lstm = time.time() - t0

        print(f"  fold {fold+1}: "
              f"GSTAR rmse={rmse_gstar:.2f} ({t_gstar:.2f}s) | "
              f"GSTARX rmse={rmse_gstarx:.2f} ({t_gstarx:.2f}s) | "
              f"LSTM-static rmse={rmse_lstm:.2f} ({t_lstm:.1f}s)")

        results.append({
            'fold': fold + 1, 'train_size': train_end - 1, 'test_size': test_end - train_end,
            'rmse_gstar':       rmse_gstar,  'mae_gstar':       mae_gstar,
            'rmse_gstarx':      rmse_gstarx, 'mae_gstarx':      mae_gstarx,
            'rmse_lstm_static': rmse_lstm,   'mae_lstm_static': mae_lstm,
            'time_total':       t_gstar + t_gstarx + t_lstm,
        })

        # Save predictions for downstream analysis
        k = fold + 1
        predictions[f'y_hat_gstar_fold_{k}']       = pred_te_mm_gstar.astype(np.float64)
        predictions[f'y_hat_gstarx_fold_{k}']      = pred_te_mm_gstarx.astype(np.float64)
        predictions[f'y_hat_lstm_static_fold_{k}'] = y_hat_mm_lstm.astype(np.float64)
        predictions[f'y_true_fold_{k}']            = y_true_te_mm.astype(np.float64)
        predictions[f'dates_fold_{k}']             = test_dates

    return results, predictions


# ============= MAIN ===========================================
def main(epochs, n_splits, seed, min_train, reuse_lstm_npz=None):
    print(f"=== Baselines (GSTAR, GSTARX, LSTM-GSTARX static) at N=40 ===")
    print(f"  EPOCHS={epochs}, N_SPLITS={n_splits}, SEED={seed}, "
          f"MIN_TRAIN={min_train}")

    panel = np.load(PANEL_NPZ, allow_pickle=True)
    Y         = panel['Y_rainfall']
    X         = panel['X_climate']
    dates     = panel['dates']
    countries = panel['region_countries']
    A_geo     = np.load(A_GEO_NPY)
    idn_indices = np.where(countries == 'IDN')[0]

    print(f"Loaded: T={Y.shape[0]} N={Y.shape[1]} IDN_targets={len(idn_indices)}")

    reuse_lstm = None
    if reuse_lstm_npz:
        reuse_lstm = dict(np.load(reuse_lstm_npz, allow_pickle=True))
        print(f"  Reusing static-LSTM predictions from {reuse_lstm_npz} (no retrain)")

    results, predictions = walk_forward_cv(
        Y, X, dates, A_geo, idn_indices,
        n_splits=n_splits, epochs=epochs, lr=LR, seed=seed, min_train=min_train,
        reuse_lstm=reuse_lstm,
    )

    df = pd.DataFrame(results)
    print(f"\n=== SUMMARY ===")
    for col in ['rmse_gstar', 'rmse_gstarx', 'rmse_lstm_static']:
        print(f"  {col:25s}: {df[col].mean():.2f} +- {df[col].std():.2f} mm")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    csv_out = f'{OUTPUT_DIR}/cv_results_baselines.csv'
    npz_out = f'{OUTPUT_DIR}/predictions_baselines.npz'
    df.to_csv(csv_out, index=False)
    np.savez_compressed(npz_out, **predictions)
    print(f"\nSaved {csv_out}")
    print(f"Saved {npz_out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',    type=int, default=EPOCHS)
    p.add_argument('--folds',     type=int, default=N_SPLITS)
    p.add_argument('--seed',      type=int, default=SEED)
    p.add_argument('--min-train', type=int, default=MIN_TRAIN)
    p.add_argument('--reuse-lstm', default=None,
                   help='NPZ with y_hat_lstm_static_fold_* to reuse instead of '
                        'retraining the static LSTM (keeps the neural baseline fixed)')
    args = p.parse_args()
    main(epochs=args.epochs, n_splits=args.folds, seed=args.seed,
         min_train=args.min_train, reuse_lstm_npz=args.reuse_lstm)
