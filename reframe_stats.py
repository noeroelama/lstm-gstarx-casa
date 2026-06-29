#!/usr/bin/env python3
"""Statistics for the Spatial Statistics reframe (weight-matrix protagonist).

Three products:
 (1) LINEAR GSTAR(1;1) weight-matrix comparison -- anchors the equivalence claim in
     the spatial-weight-selection tradition: refit the SAME linear GSTAR with three W
     specifications (KNN k=5; inverse-distance; training-correlation), all leak-free
     (W fixed across folds; correlation uses only months 0..min_train-1, which are
     training in every fold). Reuses train_baselines.GSTAR.
 (2) Formal TOST equivalence test for the deep spatial-graph variants (static a=1,
     learned a=0, adaptive) on the per-month loss, via moving-block bootstrap. Margin
     delta=0.5 mm (below the 0.57 mm three-seed dispersion). Equivalent at alpha=0.05
     if the 90% CI of the RMSE difference lies inside (-delta, +delta).
 (3) Gate-ENSO correlation r with a moving-block bootstrap 95% CI (replaces the naive
     p-value on r=+0.79).
Output: outputs/reframe_stats.csv (+ console).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = "."
ROOT = "data"
sys.path.insert(0, HERE)
import train_baselines as TB  # noqa: E402

OUT = Path(HERE) / "outputs"
MIN_TRAIN, FOLD, NSPLITS = 380, 20, 5
L, B, SEED = 12, 10000, 42

panel = np.load(ROOT + "/panel_data.npz", allow_pickle=True)
Y = panel['Y_rainfall'].astype(float)
X = panel['X_climate'].astype(float)
dates = pd.to_datetime(panel['dates'])
region_ids = panel['region_ids']
countries = panel['region_countries']
idn = np.where(countries == 'IDN')[0]
A_knn = np.load(ROOT + "/a_geo.npy")
N = Y.shape[0]


# ---------- (1) build alternative W specifications (panel node order) ----------
def row_norm(W):
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    s = W.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return W / s


cent = pd.read_csv(ROOT + "/a_geo_centroids.csv").set_index('region_id')
lat = np.array([cent.loc[r, 'lat'] for r in region_ids], float)
lon = np.array([cent.loc[r, 'lon'] for r in region_ids], float)
la = np.radians(lat); lo = np.radians(lon)
dlat = la[:, None] - la[None, :]
dlon = lo[:, None] - lo[None, :]
hav = np.sin(dlat / 2)**2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlon / 2)**2
dist = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(hav, 0, 1)))   # km
with np.errstate(divide='ignore'):
    W_inv = np.where(dist > 0, 1.0 / dist, 0.0)
W_inv = row_norm(W_inv)

corr = np.corrcoef(Y[:MIN_TRAIN].T)        # leak-free: months 0..379 = train in all folds
W_corr = row_norm(np.clip(corr, 0, None))  # keep positive correlations

W_SPECS = {'KNN k=5 (paper)': A_knn, 'inverse-distance': W_inv, 'training-correlation': W_corr}


def gstar_rmse(W):
    """Linear GSTAR(1;1) per-fold-mean RMSE with weight matrix W (mirrors train_baselines)."""
    rmse = []
    for f in range(NSPLITS):
        tr_end = MIN_TRAIN + f * FOLD
        ts, te = tr_end, tr_end + FOLD
        Y_norm, X_norm, Y_stats = TB.normalize(Y, X, tr_end)
        g = TB.GSTAR().fit(Y_norm[:tr_end], W)
        pred = g.predict_onestep(Y_norm, W)[ts - 1:te - 1][:, idn]
        pred_mm = TB.inverse_y_idn(pred, idn, Y_stats)
        true_mm = Y[ts:te][:, idn]
        rmse.append(float(np.sqrt(((pred_mm - true_mm) ** 2).mean())))
    return float(np.mean(rmse)), float(np.std(rmse))


print("=== (1) LINEAR GSTAR(1;1) by weight-matrix specification (per-fold-mean RMSE mm) ===")
rows = []
for name, W in W_SPECS.items():
    m, s = gstar_rmse(W)
    print(f"  {name:24s}: {m:6.2f} +- {s:4.2f}")
    rows.append(dict(kind='linear_W', spec=name, rmse=m, std=s))
lin = {r['spec']: r['rmse'] for r in rows}
span = max(lin.values()) - min(lin.values())
print(f"  --> linear-GSTAR W-specification span = {span:.2f} mm")


# ---------- moving-block bootstrap machinery ----------
def load_loss(npz_name, prefix=None):
    npz = np.load(OUT / npz_name, allow_pickle=True)
    nums = sorted({int(k.split('fold_')[1]) for k in npz.files if 'fold_' in k})
    parts = []
    for n in nums:
        yk = f'y_hat_{prefix}_fold_{n}' if prefix else f'y_hat_fold_{n}'
        if yk in npz.files:
            parts.append(((npz[yk] - npz[f'y_true_fold_{n}']) ** 2).mean(axis=1))
    return np.concatenate(parts)


loss = {k: load_loss(v) for k, v in {
    'static':   'predictions_casa_neighbours_only_lb12_freezeA1.npz',
    'learned':  'predictions_casa_neighbours_only_lb12_freezeA0.npz',
    'adaptive': 'predictions_casa_neighbours_only_lb12_gstarK1.npz',
}.items()}
n = len(loss['adaptive'])
rng = np.random.default_rng(SEED)
nb = int(np.ceil(n / L))
idx = (rng.integers(0, n - L + 1, size=(B, nb))[:, :, None] + np.arange(L)[None, None, :]).reshape(B, nb * L)[:, :n]

# ---------- (2) TOST equivalence ----------
print("\n=== (2) TOST equivalence (deep static/learned/adaptive W); margin delta=0.5 mm ===")
DELTA = 0.5
for a, b in [('adaptive', 'static'), ('learned', 'static'), ('adaptive', 'learned')]:
    diff = np.sqrt(loss[a][idx].mean(1)) - np.sqrt(loss[b][idx].mean(1))
    obs = np.sqrt(loss[a].mean()) - np.sqrt(loss[b].mean())
    lo90, hi90 = np.quantile(diff, [0.05, 0.95])      # 90% CI for alpha=0.05 TOST
    p_up = (diff >= DELTA).mean()      # H0: true diff >= +delta
    p_lo = (diff <= -DELTA).mean()     # H0: true diff <= -delta
    tost_p = max(p_up, p_lo)
    eq = (lo90 > -DELTA) and (hi90 < DELTA)
    print(f"  {a:8s} - {b:8s}: dRMSE={obs:+.3f}  90%CI=[{lo90:+.3f},{hi90:+.3f}]  "
          f"TOST p={tost_p:.4f}  equivalent={eq}")
    rows.append(dict(kind='TOST', spec=f'{a}-{b}', rmse=obs, ci_lo=lo90, ci_hi=hi90,
                     tost_p=tost_p, equivalent=eq))

# ---------- (3) gate-ENSO correlation CI ----------
print("\n=== (3) gate alpha vs signed Nino3.4: r with moving-block bootstrap 95% CI ===")
npz = np.load(OUT / 'predictions_casa_neighbours_only_lb12_gstarK1.npz', allow_pickle=True)
nums = sorted({int(k.split('fold_')[1]) for k in npz.files if 'alpha_fold_' in k})
nino_by_date = dict(zip(dates, X[:, 0]))
alpha_all, nino_all = [], []
for k in nums:
    a = npz[f'alpha_fold_{k}']
    d = pd.to_datetime(npz[f'dates_fold_{k}'])
    alpha_all.append(a); nino_all.append(np.array([nino_by_date[x] for x in d]))
alpha_all = np.concatenate(alpha_all); nino_all = np.concatenate(nino_all)
r_obs = float(np.corrcoef(alpha_all, nino_all)[0, 1])
m = len(alpha_all)
nb2 = int(np.ceil(m / L))
idx2 = (rng.integers(0, m - L + 1, size=(B, nb2))[:, :, None] + np.arange(L)[None, None, :]).reshape(B, nb2 * L)[:, :m]
rb = np.array([np.corrcoef(alpha_all[i], nino_all[i])[0, 1] for i in idx2])
rlo, rhi = np.quantile(rb, [0.025, 0.975])
print(f"  r(alpha, signed Nino3.4) = {r_obs:+.3f}  95% block-bootstrap CI=[{rlo:+.3f},{rhi:+.3f}]  (n={m})")
rows.append(dict(kind='gate_enso', spec='r_alpha_nino', rmse=r_obs, ci_lo=rlo, ci_hi=rhi))

pd.DataFrame(rows).to_csv(OUT / 'reframe_stats.csv', index=False)
print(f"\nSaved {OUT / 'reframe_stats.csv'}")
