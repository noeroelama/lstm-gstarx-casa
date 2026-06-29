#!/usr/bin/env python3
"""make_forecast_fig.py -- observed-vs-forecast panel for the paper (Fig 5).

Loads LSTM-GSTARX(12;1) predictions (the paper's K=12 model), picks four
provinces spanning the per-province RMSE distribution (~25/50/75/90th
percentile -- representative, not cherry-picked), and plots observed vs
forecast monthly rainfall over the concatenated walk-forward test period.

Run from the repo root after training:
    python make_forecast_fig.py
Writes paper_figs/fig5_forecast.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRED = 'outputs/predictions_casa_neighbours_only_lb12_gstarK1.npz'
CENT = 'data/a_geo_centroids.csv'
OUT  = 'paper_figs/fig5_forecast.png'

d = np.load(PRED, allow_pickle=True)
yh, yt, dts = [], [], []
for k in range(1, 11):
    hk, tk, dk = f'y_hat_fold_{k}', f'y_true_fold_{k}', f'dates_fold_{k}'
    if hk in d and tk in d and dk in d:
        yh.append(d[hk]); yt.append(d[tk]); dts.append(d[dk])
yh = np.concatenate(yh, axis=0)
yt = np.concatenate(yt, axis=0)
dts = pd.to_datetime(np.concatenate(dts))
order = np.argsort(dts.values)
dts, yh, yt = dts[order], yh[order], yt[order]

cent = pd.read_csv(CENT, dtype={'region_id': str})
idn = cent[cent['country'] == 'IDN'].reset_index(drop=True)
n = min(len(idn), yh.shape[1])

rmse = np.sqrt(((yh[:, :n] - yt[:, :n]) ** 2).mean(axis=0))
ranks = np.argsort(rmse)                       # ascending RMSE
picks = [int(ranks[int(round(p * (n - 1)))]) for p in (0.25, 0.50, 0.75, 0.90)]

fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex=True)
for ax, j in zip(axes.ravel(), picks):
    ax.plot(dts, yt[:, j], color='#1e293b', lw=1.5, label='Observed')
    ax.plot(dts, yh[:, j], color='#dd4814', lw=1.5, ls='--', label='Forecast')
    nm = str(idn.loc[j, 'name'])
    ax.set_title(f"{nm}  (RMSE {rmse[j]:.0f} mm)", fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8)
axes[0, 0].legend(loc='upper right', fontsize=8, framealpha=0.85)
for ax in axes[:, 0]:
    ax.set_ylabel('Monthly rainfall (mm)', fontsize=9)
fig.suptitle('Observed vs forecast monthly rainfall, LSTM-GSTARX(12;1) '
             '(walk-forward test period)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print('Wrote', OUT, '| shape', yh.shape,
      '| provinces:', [str(idn.loc[j, 'name']) for j in picks],
      '| RMSE:', [round(float(rmse[j]), 1) for j in picks])
