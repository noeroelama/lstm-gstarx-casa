#!/usr/bin/env python3
"""Experiment: X-timing confound (climate at target month t vs lag-1).

Reviewer concern (Journal of Hydrology): the deep LSTM-GSTARX models read the
climate vector X=[Nino3.4, DMI] at the TARGET month t (concurrent), while the
linear GSTARX baseline uses X at t-1 (lag-1). So part of the deep models'
RMSE advantage could be a pure information advantage, not a modelling one.

This harmonizes the deep model's climate information set with the baseline by
shifting X back one month wherever it is consumed -- BOTH the per-timestep LSTM
X-channel AND the CASA climate-context c_t -- via a single data-level shift
X_lag[m] = X[m-1]. Because every mode (default / window / window_f) reads
"climate at month m" through the same X array, one shift lags them all
consistently. Y (rainfall, autoregressive terms + target) is UNCHANGED.

We re-run the three models that build the headline + decomposition, each with
lag-0 (reproduce the frozen CSV) and lag-1 (harmonized):
  - LSTM-GSTARX(1;1)      lookback=1,  coupled=False  -> frozen 80.34
  - LSTM-GSTARX(12;0)+F*  lookback=12, coupled=False  -> frozen 74.03
  - LSTM-GSTARX(12;1)     lookback=12, coupled=True   -> frozen 71.37  (headline)

Reuses the REAL model/CV from train_casa_n40 (no reimplementation). Single seed
(42) so the lag-0 vs lag-1 delta isolates the timing effect (seed held fixed).
Linear baselines (GSTAR 87.09 / GSTARX 86.82) are frozen references.
"""
import sys
import numpy as np
import pandas as pd

HERE = "."
ROOT = "data"
sys.path.insert(0, HERE)
import train_casa_n40 as T  # noqa: E402

# ---- data (canonical: load a_geo.npy directly, as train_casa_n40 does) -------
panel = np.load(ROOT + "/panel_data.npz", allow_pickle=True)
Y = panel['Y_rainfall'].astype(float)
X = panel['X_climate'].astype(float)
dates = panel['dates']
idn = np.where(panel['region_countries'] == 'IDN')[0]
A_geo = np.load(ROOT + "/a_geo.npy")
print(f"Loaded: T={Y.shape[0]} N={Y.shape[1]} IDN={len(idn)} "
      f"A_geo={A_geo.shape}")


def make_lag1(Xin):
    """X_lag[m] = X[m-1] (climate of the previous month placed at index m).

    Pads index 0 with X[0] (a single training-edge row, never a test target;
    no wrap-around of the last month to the front, unlike np.roll)."""
    Xl = np.empty_like(Xin)
    Xl[1:] = Xin[:-1]
    Xl[0] = Xin[0]
    return Xl


def run(Xin, lookback, coupled, label):
    res, _ = T.walk_forward_cv(
        Y, Xin, dates, A_geo, idn,
        n_splits=T.N_SPLITS, epochs=T.EPOCHS, lr=T.LR, seed=T.SEED,
        min_train=T.MIN_TRAIN, ablation='neighbours_only',
        b_alpha_init=T.B_ALPHA_INIT, alpha_entropy_reg=0.0,
        lookback=lookback, rich_context=False, coupled_fstar=coupled)
    r = np.array([d['rmse'] for d in res])
    m = np.array([d['mae'] for d in res])
    print(f">>> {label:32s}: RMSE={r.mean():6.2f} +- {r.std():5.2f} | "
          f"MAE={m.mean():6.2f} | folds {np.round(r, 2)}")
    return float(r.mean()), float(r.std()), float(m.mean())


CONFIGS = [
    # key,            lookback, coupled, frozen_lag0, model label
    ("11",            1,  False, 80.34, "LSTM-GSTARX(1;1)"),
    ("12d",           12, False, 74.03, "LSTM-GSTARX(12;0)+F*"),
    ("121",           12, True,  71.37, "LSTM-GSTARX(12;1)"),
]
X_lag = make_lag1(X)
GSTAR, GSTARX = 87.09, 86.82  # frozen linear baselines (lag-1, see cv_summary)

rows = []
for key, lb, cp, frozen, lbl in CONFIGS:
    print(f"\n##### {lbl}  lag-0 (reproduce frozen {frozen}) #####")
    m0, s0, mae0 = run(X, lb, cp, f"{lbl} lag0")
    print(f"\n##### {lbl}  lag-1 (harmonized with GSTARX) #####")
    m1, s1, mae1 = run(X_lag, lb, cp, f"{lbl} lag1")
    rows.append(dict(key=key, model=lbl, frozen_lag0=frozen,
                     rmse_lag0=m0, std_lag0=s0, mae_lag0=mae0,
                     rmse_lag1=m1, std_lag1=s1, mae_lag1=mae1,
                     delta_lag1_minus_lag0=m1 - m0,
                     repro_err_vs_frozen=m0 - frozen))

df = pd.DataFrame(rows)
out = HERE + "/outputs/xtiming_result.csv"
df.to_csv(out, index=False)

# ---- decomposition + headline, lag-0 vs lag-1 --------------------------------
d = {r['key']: r for r in rows}
print("\n" + "=" * 72)
print("VALIDATION (lag-0 must reproduce frozen):")
for r in rows:
    flag = "OK" if abs(r['repro_err_vs_frozen']) < 0.5 else "** MISMATCH **"
    print(f"  {r['model']:24s} lag0={r['rmse_lag0']:6.2f} "
          f"(frozen {r['frozen_lag0']:.2f}, d={r['repro_err_vs_frozen']:+.2f}) {flag}")

print("\nHEADLINE (% below linear GSTAR 87.09):")
for tag, key in [("lag0", "rmse_lag0"), ("lag1", "rmse_lag1")]:
    v = d['121'][key]
    print(f"  LSTM-GSTARX(12;1) {tag}: {v:6.2f}  ->  "
          f"{(GSTAR - v) / GSTAR * 100:5.1f}% below GSTAR | "
          f"{(GSTARX - v) / GSTARX * 100:5.1f}% below GSTARX")

print("\nDECOMPOSITION:")
for tag, key in [("lag0", "rmse_lag0"), ("lag1", "rmse_lag1")]:
    temporal = d['121'][key] - d['11'][key]      # (12;1) - (1;1)
    fstar = d['121'][key] - d['12d'][key]        # (12;1) - (12;0)+F*
    print(f"  {tag}: temporal-order (12;1)-(1;1) = {temporal:+.2f} mm | "
          f"F*-placement (12;1)-(12;0)+F* = {fstar:+.2f} mm")

print(f"\nSaved {out}")
