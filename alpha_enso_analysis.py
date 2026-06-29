"""alpha_t vs ENSO interpretability for the LSTM-GSTARX(12;1) model (Proposition 4).

Uses the saved gate alpha_t (test months) from the coupled prediction NPZ, joins
the Nino 3.4 index, and quantifies the relationship. Produces a print-quality
two-panel figure: (1) alpha_t timeline vs Nino 3.4; (2) alpha_t vs Nino 3.4
scatter coloured by ENSO regime, with OLS fit.

Run from the repo root after training (needs data/ + outputs/). Writes outputs/alpha_enso.png + stats.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

OUT = 'outputs'
NPZ = f'{OUT}/predictions_casa_neighbours_only_lb12_gstarK1.npz'  # LSTM-GSTARX(12;1)

npz = np.load(NPZ, allow_pickle=True)
pan = np.load('data/panel_data.npz', allow_pickle=True)
nino_by = dict(zip(pd.to_datetime(pan['dates']), pan['X_climate'][:, 0]))

al, nn, ds = [], [], []
for k in range(1, 6):
    a, d = npz[f'alpha_fold_{k}'], pd.to_datetime(npz[f'dates_fold_{k}'])
    for ai, di in zip(a, d):
        al.append(float(ai)); nn.append(float(nino_by[di])); ds.append(di)
al, nn = np.array(al), np.array(nn)
order = np.argsort(ds); ds = np.array(ds)[order]; al, nn = al[order], nn[order]

# regimes (Nino 3.4 +/- 0.5)
reg = np.where(nn > 0.5, 'El Nino', np.where(nn < -0.5, 'La Nina', 'Neutral'))
rs, ps = pearsonr(al, nn)
ra, pa = pearsonr(al, np.abs(nn))
b1, b0 = np.polyfit(nn, al, 1)
print(f'n={len(al)}  alpha mean={al.mean():.3f} sd={al.std():.3f} '
      f'[{al.min():.3f},{al.max():.3f}]')
print(f'corr(alpha, signed Nino3.4): r={rs:+.3f} p={ps:.4f}')
print(f'corr(alpha, |Nino3.4|)     : r={ra:+.3f} p={pa:.4f}')
for r in ('El Nino', 'Neutral', 'La Nina'):
    m = reg == r
    print(f'  alpha[{r:8s}] mean={al[m].mean():.3f} (n={m.sum()})')

# ---- figure ----
C = {'El Nino': '#d62728', 'Neutral': '#7f7f7f', 'La Nina': '#1f77b4'}
fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                               gridspec_kw=dict(width_ratios=[1.55, 1]))

# Panel A: timeline
axA.plot(ds, al, color='#157A3A', lw=2, label=r'$\alpha_t$ (gate)')
axA.set_ylabel(r'$\alpha_t$  (1 = geography, 0 = learned)', color='#157A3A')
axA.tick_params(axis='y', labelcolor='#157A3A')
axA.set_ylim(0.5, 1.0)
ax2 = axA.twinx()
ax2.plot(ds, nn, color='#d62728', lw=1.4, ls='--', label='Niño 3.4')
ax2.axhline(0.5, color='#d62728', lw=0.6, alpha=0.4)
ax2.axhline(-0.5, color='#1f77b4', lw=0.6, alpha=0.4)
ax2.set_ylabel('Niño 3.4 (°C)', color='#d62728')
ax2.tick_params(axis='y', labelcolor='#d62728')
ax2.set_ylim(-2.6, 2.6)
axA.set_title('(a) Gate $\\alpha_t$ tracks the ENSO phase (test period)',
              fontsize=11, fontweight='bold')
axA.set_xlabel('Month (walk-forward test period)')

# Panel B: scatter + fit
for r in ('El Nino', 'Neutral', 'La Nina'):
    m = reg == r
    axB.scatter(nn[m], al[m], s=26, c=C[r], label=r, edgecolor='white', lw=0.4)
xs = np.linspace(nn.min(), nn.max(), 50)
axB.plot(xs, b1 * xs + b0, color='#333', lw=1.6,
         label=f'OLS  (r={rs:.2f})')
axB.set_xlabel('Niño 3.4 (°C)')
axB.set_ylabel(r'$\alpha_t$')
axB.set_title('(b) La Niña → lower $\\alpha_t$ → more learned attention',
              fontsize=11, fontweight='bold')
axB.legend(frameon=False, fontsize=8.5, loc='lower right')
fig.tight_layout()
fig.savefig(f'{OUT}/alpha_enso.png', dpi=200, bbox_inches='tight')
print(f'wrote {OUT}/alpha_enso.png')
