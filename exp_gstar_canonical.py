#!/usr/bin/env python3
"""GSTAR(K;1) vs its homogeneous (STAR) restriction.

GSTAR (Borovkova-Lopuhaa-Ruchjana 2008/2012) is HETEROGENEOUS: the autoregressive and
space-time parameters vary PER LOCATION -- the parameter matrices are diagonal,
Phi_k0 = diag(phi_k0^(1..N)), Phi_k1 = diag(phi_k1^(1..N)). Its homogeneous restriction
-- a single SCALAR shared by every location, Phi = phi*I -- is the STAR model
(Pfeifer-Deutsch 1980), NOT a more "canonical" GSTAR. For a model with fully per-location
parameters, the stacked/block least-squares fit over all (location, time) pairs is
mathematically identical to 38 independent per-province regressions (the design is
block-diagonal), so the per-location number IS the GSTAR least-squares estimate.

    Z_i(t) = mu_i + sum_{k=1..K} [ phi_k0^(i) Z_i(t-k) + phi_k1^(i) (W Z(t-k))_i ] + e_i(t)

This script computes, for K in {1,6,12,18,24} and for GSTAR / GSTARX:
  - per-location = GSTAR (the paper baseline). Reproduces
    linear_multistep_rmse.csv: K1 GSTAR 87.09, K12 GSTAR 77.94, ...
  - "canonical"  = the HOMOGENEOUS (STAR) restriction, a shared scalar phi + per-location
    intercept (reported as a contrast only; K1 GSTAR 88.79 / GSTARX 88.23). The CSV column
    keeps the name "canonical_*" for backward compatibility.
Same panel, same A_geo (KNN k=5 Haversine), same 5-fold walk-forward CV (min_train=380,
fold_size=20), same min-max normalisation as every other model.
"""
import sys, csv
import numpy as np


PANEL    = r"data/panel_data.npz"
CASA_DIR = r"."
MIN_TRAIN, N_SPLITS = 380, 5
KS = [1, 6, 12, 18, 24]

panel = np.load(PANEL, allow_pickle=True)
Y = panel['Y_rainfall'].astype(float)
X = panel['X_climate'].astype(float)
idn = np.where(panel['region_countries'] == 'IDN')[0]
A = np.load("data/a_geo.npy")
T, N = Y.shape
FOLD = (T - MIN_TRAIN) // N_SPLITS
NID = len(idn)


def normalize(Y, X, tr):
    ymin, ymax = Y[:tr].min(0), Y[:tr].max(0); yr = ymax - ymin + 1e-9
    xmin, xmax = X[:tr].min(0), X[:tr].max(0); xr = xmax - xmin + 1e-9
    return (Y - ymin) / yr, (X - xmin) / xr, (ymin, ymax, yr)


def perloc_cv(K, use_climate):
    """Per-location OLS (old parameterisation). Validation anchor."""
    fold = []
    for f in range(N_SPLITS):
        tr_end = MIN_TRAIN + f * FOLD; te_s, te_e = tr_end, tr_end + FOLD
        if te_e > T: break
        Yn, Xn, (ymin, ymax, yr) = normalize(Y, X, tr_end)
        F = Yn @ A.T
        preds = np.zeros((te_e - te_s, NID))
        for jj, j in enumerate(idn):
            def feats(ts):
                cols = [np.ones(len(ts))]
                for k in range(1, K + 1):
                    cols += [Yn[ts - k, j], F[ts - k, j]]
                if use_climate:
                    for k in range(1, K + 1):
                        cols += [Xn[ts - k, 0], Xn[ts - k, 1]]
                return np.stack(cols, axis=1)
            tr_ts = np.arange(K, tr_end)
            coef, *_ = np.linalg.lstsq(feats(tr_ts), Yn[tr_ts, j], rcond=None)
            preds[:, jj] = feats(np.arange(te_s, te_e)) @ coef
        pred_mm = preds * yr[idn] + ymin[idn]
        fold.append(float(np.sqrt(((pred_mm - Y[te_s:te_e][:, idn]) ** 2).mean())))
    r = np.array(fold); return r.mean(), r.std()


def _design(ts, Yn, Xn, F, K, use_climate):
    """Stacked design for target times `ts` over all IDN locations.
    Rows ordered (t, i); columns = [per-location intercept (NID)] + shared dynamics.
    Returns (D, target)."""
    n = len(ts)
    # shared dynamic features per (t, i): own-lag k, spatial-lag k
    own = np.stack([Yn[ts - k][:, idn] for k in range(1, K + 1)], axis=2)   # (n, NID, K)
    sp  = np.stack([F [ts - k][:, idn] for k in range(1, K + 1)], axis=2)   # (n, NID, K)
    feat = np.concatenate([own, sp], axis=2)                               # (n, NID, 2K)
    if use_climate:
        clim = np.stack([np.stack([Xn[ts - k, 0], Xn[ts - k, 1]], axis=1)
                         for k in range(1, K + 1)], axis=1)                # (n, K, 2)
        clim = clim.reshape(n, 2 * K)
        clim = np.repeat(clim[:, None, :], NID, axis=1)                    # (n, NID, 2K)
        feat = np.concatenate([feat, clim], axis=2)
    n_dyn = feat.shape[2]
    feat = feat.reshape(n * NID, n_dyn)                                    # rows (t,i)
    # per-location intercept one-hot
    inter = np.tile(np.eye(NID), (n, 1))                                   # (n*NID, NID)
    D = np.concatenate([inter, feat], axis=1)
    target = Yn[ts][:, idn].reshape(n * NID)
    return D, target


def canonical_cv(K, use_climate):
    """Canonical GSTAR(K;1): shared dynamics + per-location intercept, one stacked OLS."""
    fold = []
    for f in range(N_SPLITS):
        tr_end = MIN_TRAIN + f * FOLD; te_s, te_e = tr_end, tr_end + FOLD
        if te_e > T: break
        Yn, Xn, (ymin, ymax, yr) = normalize(Y, X, tr_end)
        F = Yn @ A.T
        tr_ts = np.arange(K, tr_end)
        D_tr, r_tr = _design(tr_ts, Yn, Xn, F, K, use_climate)
        beta, *_ = np.linalg.lstsq(D_tr, r_tr, rcond=None)
        te_ts = np.arange(te_s, te_e)
        D_te, _ = _design(te_ts, Yn, Xn, F, K, use_climate)
        pred = (D_te @ beta).reshape(len(te_ts), NID)
        pred_mm = pred * yr[idn] + ymin[idn]
        fold.append(float(np.sqrt(((pred_mm - Y[te_s:te_e][:, idn]) ** 2).mean())))
    r = np.array(fold); return r.mean(), r.std(), len(beta)


# reference per-location numbers (linear_multistep_rmse.csv) for the validation check
REF = {(1, False): 87.09, (6, False): 80.54, (12, False): 77.94, (18, False): 77.35, (24, False): 78.05,
       (1, True): 86.82, (6, True): 80.93, (12, True): 78.71, (18, True): 81.19, (24, True): 82.82}

if __name__ == "__main__":
    print(f"T={T} N={N} IDN={NID} fold_size={FOLD}")
    print(f"{'K':>3} {'clim':>5} | {'per-loc OLS':>14} {'(ref)':>7} {'ok':>3} | {'canonical shared':>18} {'npar':>5} | {'canon-perloc':>12}")
    print("-" * 92)
    rows = []
    for use_climate in (False, True):
        for K in KS:
            pm, ps = perloc_cv(K, use_climate)
            cm, cs, npar = canonical_cv(K, use_climate)
            ref = REF[(K, use_climate)]
            ok = "OK" if abs(pm - ref) < 0.15 else "!!"
            tag = "GSTARX" if use_climate else "GSTAR "
            print(f"{K:>3} {tag:>5} | {pm:7.2f}+-{ps:4.2f} {ref:7.2f} {ok:>3} | "
                  f"{cm:7.2f}+-{cs:4.2f} {npar:>5} | {cm-pm:+12.2f}")
            rows.append([tag.strip(), K, round(pm, 3), round(ps, 3), round(cm, 3), round(cs, 3), npar])
    with open(CASA_DIR + r"\canonical_gstar_rmse.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "K", "perloc_rmse", "perloc_std", "canonical_rmse", "canonical_std", "canonical_nparams"])
        w.writerows(rows)
    print("\nWrote canonical_gstar_rmse.csv")
