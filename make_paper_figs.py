"""Print-quality paper figure set (dpi=300), consistent style.

Produces into paper_figs/:
  fig1_ranking.png       6-model RMSE comparison (Table 1)
  fig2_decomposition.png confound decomposition single/decoupled/coupled
  fig3_regime.png        regime-stratified RMSE (3 models x 3 ENSO states)
  fig4_graph_ablation.png static/learned/adaptive + KNN (Table 2)

All numbers are the 5-fold walk-forward CV results (see CASA_Changelog.md).
alpha_enso.png is produced separately by alpha_enso_analysis.py (needs the VPS
per-month gate series).

Usage:  python make_paper_figs.py [outdir]
"""
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else 'paper_figs'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'axes.linewidth': 0.8,
})

C_LIN = '#B33A3A'    # linear GSTAR(1;1)
C_SS = '#E0A235'     # LSTM-GSTARX(1;1) deep
C_DEC = '#2E8B97'    # LSTM-GSTARX(K;0)
C_COU = '#1FA84F'    # LSTM-GSTARX(K;1) (proposed)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches='tight')
    plt.close(fig)
    print('wrote', os.path.join(OUT, name))


# ===== Fig 1: 6-model ranking =====
models = ['GSTAR(1;1)', 'GSTARX(1;1)', 'LSTM-GSTARX(1;1)\nstatic W',
          'LSTM-GSTARX(1;1)\n+ CASA', 'LSTM-GSTARX(12;0)\n+ CASA',
          'LSTM-GSTARX(12;1)\n+ CASA']
rmse = [87.09, 86.82, 80.36, 80.34, 74.03, 71.37]
err = [4.50, 5.13, 7.64, 7.62, 7.07, 6.82]
cols = [C_LIN, C_LIN, C_SS, C_SS, C_DEC, C_COU]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
y = np.arange(len(models))[::-1]
ax.barh(y, rmse, xerr=err, color=cols, edgecolor='white', height=0.66,
        error_kw=dict(ecolor='#555', lw=1.1, capsize=3))
ax.set_yticks(y)
ax.set_yticklabels(models, fontsize=8.5)
ax.set_xlabel('RMSE (mm) — lower is better')
ax.set_xlim(0, 100)
for yi, r in zip(y, rmse):
    ax.text(r + max(err) + 1.5, yi, f'{r:.2f}', va='center', fontsize=9,
            fontweight='bold')
ax.axvspan(0, 0, color='none')
ax.set_title('Six-model comparison (5-fold walk-forward CV)',
             fontsize=11.5, fontweight='bold')
save(fig, 'fig1_ranking.png')


# ===== Fig 2: confound decomposition =====
labels = ['LSTM-GSTARX\n(1;1)+CASA', 'LSTM-GSTARX(12;0)\n(F* after LSTM)',
          'LSTM-GSTARX(12;1)\n(F* in recurrence)']
vals = [80.34, 74.03, 71.37]
sd = [7.62, 7.07, 6.82]
cols2 = [C_SS, C_DEC, C_COU]
fig, ax = plt.subplots(figsize=(7.2, 4.8))
x = np.arange(3)
ax.bar(x, vals, width=0.55, color=cols2, edgecolor='white', linewidth=1.2,
       yerr=sd, capsize=6, error_kw=dict(ecolor='#444', lw=1.2))
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel('RMSE (mm) — lower is better')
ax.set_ylim(0, 113)
for xi, v, s in zip(x, vals, sd):
    ax.text(xi, v + s + 2.2, f'{v:.2f}', ha='center', fontweight='bold', fontsize=11)
ax.annotate('', xy=(2, 95), xytext=(1, 95),
            arrowprops=dict(arrowstyle='-|>', color='#157A3A', lw=2.2))
ax.text(1.5, 96.5, 'spatial lag in recurrence:  −2.66 mm  (p<0.001)', ha='center',
        va='bottom', color='#157A3A', fontsize=9.5, fontweight='bold')
ax.annotate('', xy=(2, 104.5), xytext=(0, 104.5),
            arrowprops=dict(arrowstyle='-|>', color='#555', lw=1.7))
ax.text(1.0, 106, 'pure temporal-order effect:  −8.97 mm  (p<0.001)', ha='center',
        va='bottom', color='#333', fontsize=9.5)
ax.set_title('Decomposing the (1;1) → (K;1) gain',
             fontsize=11.5, fontweight='bold', pad=10)
save(fig, 'fig2_decomposition.png')


# ===== Fig 3: regime-stratified =====
regimes = ['El Niño', 'Neutral', 'La Niña']
single = [71.39, 77.50, 89.42]
dec = [65.92, 73.01, 80.71]
cou = [63.34, 71.28, 77.02]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
x = np.arange(3); w = 0.26
ax.bar(x - w, single, w, label='LSTM-GSTARX(1;1) CASA', color=C_SS, edgecolor='white')
ax.bar(x, dec, w, label='LSTM-GSTARX(K;0)', color=C_DEC, edgecolor='white')
ax.bar(x + w, cou, w, label='LSTM-GSTARX(K;1) (proposed)', color=C_COU,
       edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(regimes)
ax.set_ylabel('RMSE (mm) — lower is better')
ax.set_ylim(0, 100)
for i in range(3):
    for off, v in [(-w, single[i]), (0, dec[i]), (w, cou[i])]:
        ax.text(i + off, v + 1.2, f'{v:.0f}', ha='center', fontsize=8)
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.set_title('Forecast error by ENSO regime (K=12)',
             fontsize=11.5, fontweight='bold')
save(fig, 'fig3_regime.png')


# ===== Fig 4: spatial-graph ablation =====
glabels = ['Static\nKNN (α=1)', 'Learned\nonly (α=0)', 'Adaptive\nCASA',
           'Static\nk=3', 'Static\nk=8']
gvals = [71.36, 71.42, 71.37, 71.18, 71.32]
gcols = [C_COU, '#888', C_COU, '#bbb', '#bbb']
fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(5)
ax.bar(x, gvals, width=0.6, color=gcols, edgecolor='white', linewidth=1.0)
ax.set_xticks(x); ax.set_xticklabels(glabels, fontsize=9)
ax.set_ylabel('RMSE (mm)')
ax.set_ylim(70.5, 72.0)
for xi, v in zip(x, gvals):
    ax.text(xi, v + 0.02, f'{v:.2f}', ha='center', fontsize=9.5, fontweight='bold')
ax.axhline(71.36, color='#157A3A', lw=0.8, ls='--', alpha=0.6)
ax.set_title('Spatial-graph type has no effect on accuracy (LSTM-GSTARX(12;1))\n'
             'static / learned / adaptive all within 0.24 mm',
             fontsize=11, fontweight='bold', pad=8)
save(fig, 'fig4_graph_ablation.png')

print('DONE — 4 figures in', OUT)
