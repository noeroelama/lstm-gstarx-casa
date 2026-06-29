#!/usr/bin/env python3
"""Moving-block bootstrap for the CASA paper (rigorous for serially-correlated
monthly forecast-error differentials; replaces the i.i.d. Diebold-Mariano-HLN
assumption that the manuscript flagged as a limitation).

Two products, both on the per-month squared error averaged over the 38 targets,
concatenated across the 5 walk-forward folds (the SAME loss series the DM test
uses):

  (1) EFFECT p-values -- two-sided block-bootstrap test of H0: mean loss
      differential = 0 for the headline effects (deep vs linear, temporal order,
      F* placement, and the borderline K=6 placement).
  (2) EQUIVALENCE CIs -- block-bootstrap 95% CI of the RMSE difference between
      spatial-graph variants (static / learned / adaptive). If the whole CI lies
      within a small negligible margin, "the graph type does not matter" is an
      equivalence result, not merely a non-significant difference.

Moving-block bootstrap: Kunsch (1989); block length L=12 (one year, the
seasonally-motivated choice), B=10000 resamples, seed 42. Automatic block-length
selection (Politis & White 2004) gives a similar L for monthly data.
Output: outputs/block_bootstrap.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs")
L, B, SEED = 12, 10000, 42

MODELS = {
    'GSTAR(1;1)':           ('predictions_baselines.npz', 'gstar'),
    'GSTARX(1;1)':          ('predictions_baselines.npz', 'gstarx'),
    'LSTM-static':          ('predictions_baselines.npz', 'lstm_static'),
    'LSTM-GSTARX(1;1)':     ('predictions_casa_neighbours_only.npz', None),
    'LSTM-GSTARX(6;0)+F*':  ('predictions_casa_neighbours_only_lb6.npz', None),
    'LSTM-GSTARX(6;1)':     ('predictions_casa_neighbours_only_lb6_gstarK1.npz', None),
    'LSTM-GSTARX(12;0)+F*': ('predictions_casa_neighbours_only_lb12.npz', None),
    'LSTM-GSTARX(12;1)':    ('predictions_casa_neighbours_only_lb12_gstarK1.npz', None),
    'graph-static(a=1)':    ('predictions_casa_neighbours_only_lb12_freezeA1.npz', None),
    'graph-learned(a=0)':   ('predictions_casa_neighbours_only_lb12_freezeA0.npz', None),
    'graph-adaptive':       ('predictions_casa_neighbours_only_lb12_gstarK1.npz', None),
}


def load_loss(npz_name, prefix):
    """Per-month MSE (averaged over 38 targets), concatenated across folds."""
    npz = np.load(OUT / npz_name, allow_pickle=True)
    nums = sorted({int(k.split('fold_')[1]) for k in npz.files if 'fold_' in k})
    parts = []
    for n in nums:
        yk = f'y_hat_{prefix}_fold_{n}' if prefix else f'y_hat_fold_{n}'
        if yk not in npz.files:
            continue
        e = npz[yk] - npz[f'y_true_fold_{n}']
        parts.append((e ** 2).mean(axis=1))
    return np.concatenate(parts)


loss = {name: load_loss(*spec) for name, spec in MODELS.items()}
n = len(loss['LSTM-GSTARX(12;1)'])
assert all(len(v) == n for v in loss.values()), "fold lengths differ"

rng = np.random.default_rng(SEED)
nb = int(np.ceil(n / L))
starts = rng.integers(0, n - L + 1, size=(B, nb))
idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, nb * L)[:, :n]


def effect(a_name, b_name):
    """Two-sided block-bootstrap p for H0: E[loss_a - loss_b]=0. Negative mean
    diff => a better. Also report observed RMSE of each."""
    a, b = loss[a_name], loss[b_name]
    d = a - b
    dbar = (a[idx] - b[idx]).mean(axis=1)
    p = 2 * min((dbar >= 0).mean(), (dbar <= 0).mean())
    return dict(pair=f'{a_name} - {b_name}', rmse_a=np.sqrt(a.mean()),
                rmse_b=np.sqrt(b.mean()), mean_loss_diff=d.mean(),
                a_better=bool(d.mean() < 0), p_block=min(p, 1.0))


def equivalence(a_name, b_name, margin=0.5):
    """Block-bootstrap 95% CI of RMSE_a - RMSE_b (mm). Equivalence at `margin`
    if the whole CI lies within +/- margin."""
    a, b = loss[a_name], loss[b_name]
    diff = np.sqrt(a[idx].mean(axis=1)) - np.sqrt(b[idx].mean(axis=1))
    lo, hi = np.quantile(diff, [0.025, 0.975])
    obs = np.sqrt(a.mean()) - np.sqrt(b.mean())
    return dict(pair=f'{a_name} - {b_name}', rmse_diff=obs, ci_lo=lo, ci_hi=hi,
                margin=margin, equivalent=bool(abs(lo) < margin and abs(hi) < margin))


EFFECTS = [
    ('LSTM-GSTARX(12;1)', 'GSTAR(1;1)'),        # deep vs linear (headline)
    ('LSTM-GSTARX(12;1)', 'LSTM-GSTARX(1;1)'),  # temporal order (-8.97)
    ('LSTM-GSTARX(12;1)', 'LSTM-GSTARX(12;0)+F*'),  # F* placement (-2.66)
    ('LSTM-GSTARX(6;1)',  'LSTM-GSTARX(6;0)+F*'),   # borderline K=6 placement
]
EQUIV = [
    ('graph-adaptive', 'graph-static(a=1)'),
    ('graph-learned(a=0)', 'graph-static(a=1)'),
    ('graph-adaptive', 'graph-learned(a=0)'),
]

print(f"Moving-block bootstrap: L={L}, B={B}, seed={SEED}, n={n} test months\n")
print("=== EFFECT TESTS (two-sided block-bootstrap p; DM-HLN p in paper for comparison) ===")
rows = []
dm_ref = {'LSTM-GSTARX(12;1) - LSTM-GSTARX(1;1)': '3.6e-6',
          'LSTM-GSTARX(12;1) - LSTM-GSTARX(12;0)+F*': '2.6e-5',
          'LSTM-GSTARX(6;1) - LSTM-GSTARX(6;0)+F*': '0.079 (n.s.)'}
for a, b in EFFECTS:
    r = effect(a, b)
    r['kind'] = 'effect'
    rows.append(r)
    print(f"  {r['pair']:48s} dRMSE={r['rmse_a']-r['rmse_b']:+6.2f}  "
          f"betterA={r['a_better']!s:5s}  p_block={r['p_block']:.4g}   "
          f"[DM-HLN: {dm_ref.get(r['pair'],'-')}]")

print("\n=== EQUIVALENCE (RMSE-diff 95% block-bootstrap CI; margin +/-0.5 mm) ===")
for a, b in EQUIV:
    r = equivalence(a, b)
    r['kind'] = 'equivalence'
    rows.append(r)
    print(f"  {r['pair']:42s} dRMSE={r['rmse_diff']:+.3f}  "
          f"95% CI=[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  equivalent(<0.5mm)={r['equivalent']}")

pd.DataFrame(rows).to_csv(OUT / 'block_bootstrap.csv', index=False)
print(f"\nSaved {OUT / 'block_bootstrap.csv'}")
