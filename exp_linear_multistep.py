#!/usr/bin/env python3
"""Confound control: MEMORY vs the LSTM nonlinearity.
LINEAR GSTAR(K;1) — per-province OLS on K lags of [own Y, spatial lag F=A_geo@Y]
(+climate lags for GSTARX) — same 5-fold walk-forward CV as the neural models.
K=1 must reproduce ~87.09/86.82 (validation). Writes linear_multistep_rmse.csv and the
figure fig_linear_vs_neural.png (neural curve read live from cv_summary_table.csv).
"""
import sys, csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASA = r"."
FIGDST = r"paper_figs"
panel = np.load(r"data/panel_data.npz", allow_pickle=True)
Y = panel['Y_rainfall'].astype(float); X = panel['X_climate'].astype(float)
idn = np.where(panel['region_countries'] == 'IDN')[0]
A = np.load("data/a_geo.npy")
T, N = Y.shape
MIN_TRAIN, N_SPLITS = 380, 5
FOLD = (T - MIN_TRAIN) // N_SPLITS
KS = [1, 6, 12, 18, 24]

def normalize(Y, X, tr):
    ymin, ymax = Y[:tr].min(0), Y[:tr].max(0); yr = ymax - ymin + 1e-9
    xmin, xmax = X[:tr].min(0), X[:tr].max(0); xr = xmax - xmin + 1e-9
    return (Y - ymin) / yr, (X - xmin) / xr, (ymin, ymax, yr)

def gstar_cv(K, use_climate):
    fold = []
    for f in range(N_SPLITS):
        tr_end, te_s, te_e = MIN_TRAIN + f * FOLD, MIN_TRAIN + f * FOLD, MIN_TRAIN + f * FOLD + FOLD
        if te_e > T: break
        Yn, Xn, (ymin, ymax, yr) = normalize(Y, X, tr_end)
        F = Yn @ A.T
        preds = np.zeros((te_e - te_s, len(idn)))
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

# --- neural curve from cv_summary_table.csv (live) ---
SUM = {}
with open(CASA + r"\cv_summary_table.csv") as fh:
    for row in csv.DictReader(fh):
        SUM[row['model']] = float(row['rmse_mean'])
NEU = {1: SUM['casa_neighbours_only'], 6: SUM['casa_neighbours_only_lb6_gstarK1'],
       12: SUM['casa_neighbours_only_lb12_gstarK1'], 18: SUM['casa_neighbours_only_lb18_gstarK1'],
       24: SUM['casa_neighbours_only_lb24_gstarK1']}

R = {}
print(f"{'K':>3} | {'linear GSTAR(K;1)':>18} | {'linear GSTARX(K;1)':>18} | {'neural LSTM-GSTARX(K;1)':>24}")
for K in KS:
    m, s = gstar_cv(K, False); mx, sx = gstar_cv(K, True); R[K] = (m, s, mx, sx)
    print(f"{K:>3} | {m:7.2f} +- {s:4.2f}     | {mx:7.2f} +- {sx:4.2f}     | {NEU[K]:7.2f}")

with open(CASA + r"\linear_multistep_rmse.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["K", "gstar_rmse", "gstar_std", "gstarx_rmse", "gstarx_std", "neural_lstm_gstarK1_rmse"])
    for K in KS: w.writerow([K, *[round(v, 3) for v in R[K]], round(NEU[K], 3)])

# --- figure ---
lin = [R[k][0] for k in KS]; neu = [NEU[k] for k in KS]
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(KS, lin, 's--', color='#B33A3A', lw=2.2, ms=9, label='Linear GSTAR(K;1)  — no LSTM')
ax.plot(KS, neu, 'o-', color='#1FA84F', lw=2.6, ms=9, label='LSTM-GSTARX(K;1)  — neural')
ax.annotate("", xy=(12, neu[2]), xytext=(12, lin[2]), arrowprops=dict(arrowstyle='<->', color='0.35', lw=1.5))
ax.text(12.6, (lin[2] + neu[2]) / 2, f"the LSTM adds\n{neu[2]-lin[2]:.2f} mm", fontsize=9.5, color='0.15', va='center')
ax.annotate("longer memory\n(linear too)", xy=(6, lin[1]), xytext=(3.0, 83.5), fontsize=8.5, color='#B33A3A',
            arrowprops=dict(arrowstyle='->', color='#B33A3A', alpha=0.6))
ax.text(1.2, 80.0, "single-step\n80.34", fontsize=8.3, color='#1FA84F')
ax.set_xticks(KS); ax.set_xlabel("Temporal order K  (months of memory)")
ax.set_ylabel("RMSE (mm) — lower is better"); ax.set_ylim(68, 90); ax.set_xlim(0, 25.5)
ax.set_title("You need BOTH: longer memory AND the LSTM\nLinear with long memory stalls at ~78 mm; only the LSTM reaches 71 mm", fontsize=10.5)
ax.legend(fontsize=9, loc='upper right'); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(FIGDST + "/fig_linear_vs_neural.png", dpi=300); plt.close(fig)
print("\nWrote linear_multistep_rmse.csv + fig_linear_vs_neural.png")
print(f"At K=12: linear {R[12][0]:.2f} vs neural {NEU[12]:.2f}  -> LSTM adds {NEU[12]-R[12][0]:.2f} mm")
print(f"Memory (linear, K1->12): {R[1][0]-R[12][0]:.2f} mm ; (neural): {NEU[1]-NEU[12]:.2f} mm")
