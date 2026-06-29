# LSTM-GSTARX(K;1) + CASA — monthly rainfall forecasting for Indonesia

[![CI](https://github.com/noeroelama/lstm-gstarx-casa/actions/workflows/ci.yml/badge.svg)](https://github.com/noeroelama/lstm-gstarx-casa/actions/workflows/ci.yml)
<!-- DOI badge: add after archiving this release to Zenodo -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--8802--9629-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0005-8802-9629)

Code and derived data for a hybrid LSTM-GSTARX rainfall-forecasting model and CASA
(Climate-Adaptive Spatial Attention), a compact 265-parameter module that replaces the
static GSTAR spatial weight matrix `W` with a time-varying, climate-conditioned blend
`W_t = α_t · A_geo + (1 − α_t) · A_learned_t`.

The study forecasts next-month rainfall for 38 Indonesian provinces (plus two exogenous
boundary nodes, N = 40) over 484 months (1985–2025), under five-fold walk-forward
cross-validation with Diebold–Mariano testing. The main finding is that **temporal
recurrence, not spatial adaptivity, drives forecasting skill**: lengthening the temporal
order lowers RMSE by ~9 mm, while the choice of spatial graph (static, learned, or
adaptive) changes accuracy by less than 0.25 mm. CASA's value is interpretability — its
gate tracks the signed Niño 3.4 index (r = +0.79) — and a no-cost fallback to the
geographic prior. The best model, LSTM-GSTARX(12;1), reaches 71.99 ± 0.57 mm RMSE,
about 17% below the GSTAR baseline (87.09 mm).

An interactive results dashboard accompanies the study: https://rainfall.lms.id

## Repository layout

```
casa_torch.py                  CASA module + the LSTM-GSTARX-CASA forward pass (PyTorch)
train_casa_n40.py              training + 5-fold walk-forward CV (main entry point)
build_a_geo_k.py               KNN spatial-weight graph (k sensitivity: k=3, k=8)
train_baselines.py             GSTAR/GSTARX baselines (stacked OLS) + static LSTM
exp_gstar_canonical.py         GSTAR (per-location) vs homogeneous-STAR contrast, K sweep
train_baselines_canon.py       homogeneous-STAR baseline variant (contrast only)
exp_casa_linear_canon.py       linear CASA spatial-weight ablation
exp_noclim.py                  ablation: model without climate inputs
exp_vanilla_lstm.py            ablation: plain LSTM, no spatial lag
exp_linear_multistep.py        GSTAR temporal-order sweep (linear)
compare_models.py              cross-model comparison + Diebold–Mariano tests
bootstrap_ci.py                bootstrap confidence intervals
per_province_rmse.py           per-province error breakdown
alpha_enso_analysis.py         CASA gate vs ENSO interpretability
make_paper_figs.py             figures 1–4
make_forecast_fig.py           observed-vs-forecast figure
make_fig_linear_vs_neural_canon.py   memory-vs-nonlinearity figure
data/                          derived inputs (see "Data")
results/                       headline result tables (CSV)
```

## Installation

Python 3.10+:

```
pip install -r requirements.txt
```

Models train on CPU (no GPU required), in minutes per cross-validation fold.

## Data

`data/` ships the derived series used in the paper:

- `panel_data.npz` — the N = 40 monthly panel: `Y_rainfall` (areal-mean CHIRPS per
  province), `X_climate` (Niño 3.4, DMI), `dates`, `region_ids`, `region_countries`.
- `a_geo.npy` — the static geographic prior `A_geo`, a row-stochastic KNN graph (k = 5,
  inverse-distance over equal-area centroids).
- `a_geo_centroids.csv` — province centroids (lat/lon), from which `a_geo.npy` and the
  k = 3 / k = 8 sensitivity graphs are reconstructed by `build_a_geo_k.py`.

The raw drivers are publicly available from their providers and are not redistributed
here: CHIRPS v2.0 (Climate Hazards Center, UC Santa Barbara), the Niño 3.4 index (NOAA),
and the Dipole Mode Index derived from HadISST1.1 (UK Met Office Hadley Centre).

## Reproducing the results

```
# 1. main model — LSTM-GSTARX(12;1): K=12 temporal order, spatial lag kept inside the
#    recurrence (the --coupled-fstar flag = the GSTAR-consistent (K;1) design)
python train_casa_n40.py --lookback 12 --coupled-fstar

# 2. linear baselines: GSTAR/GSTARX + static LSTM -> predictions_baselines.npz
python train_baselines.py
#    (optional GSTAR-vs-homogeneous-STAR K-sweep contrast: python exp_gstar_canonical.py)

# 3. cross-model comparison + Diebold–Mariano tests
python compare_models.py

# 4. interpretability (gate vs ENSO) and figures
python alpha_enso_analysis.py
python make_paper_figs.py
python make_forecast_fig.py
```

Per-model predictions are written to `outputs/` and consumed by `compare_models.py`,
`bootstrap_ci.py`, and `per_province_rmse.py`. Each training script documents its
command-line switches in its module docstring (`python train_casa_n40.py --help`).
The spatial-graph sensitivity (k = 3 / 5 / 8) uses `build_a_geo_k.py` followed by
`train_casa_n40.py --a-geo-path data/a_geo_k3.npy`.

## Results

The headline tables are in `results/` (mean RMSE across the five walk-forward folds,
Diebold–Mariano test matrix, ENSO-regime stratification, linear K sweeps). The
headline comparison: GSTAR(1;1) 87.09, GSTARX(1;1) 86.82, LSTM-GSTARX(1;1) 80.36,
LSTM-GSTARX(12;0) 74.03, LSTM-GSTARX(12;1) 71.37 mm (71.99 ± 0.57 over three seeds).
(The linear baselines are GSTAR; its homogeneous STAR restriction,
88.79 / 88.23, is reported as a contrast in `results/canonical_gstar_rmse.csv`.)

Further analyses are reproduced by `skill_scores.py` (skill vs a same-calendar-month
climatology and persistence, plus MAE), `block_bootstrap.py` (moving-block-bootstrap
significance and a TOST equivalence test for the static / learned / adaptive spatial
weights), `reframe_stats.py` (a linear-GSTAR weight-matrix comparison and the gate–ENSO
correlation interval), and `exp_xtiming.py` (a climate-timing robustness check); their
tables are written to `results/`.

## Citation

If you use this code or data, please cite the accompanying paper (see `CITATION.cff`):

> Nurwahid, A. (2026). Climate-adaptive spatial weights for space-time autoregression:
> an equivalence result for monthly rainfall forecasting.

## License

MIT — see `LICENSE`.
