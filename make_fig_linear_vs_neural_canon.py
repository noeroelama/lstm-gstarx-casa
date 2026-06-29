#!/usr/bin/env python3
"""Regenerate fig_linear_vs_neural.png with the CANONICAL linear GSTAR(K;1) curve.
Linear canonical from canonical_gstar_rmse.csv; neural LSTM-GSTARX(K;1) from cv_summary_table.csv."""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASA = r"."
FIGDST = r"paper_figs"
KS = [1, 6, 12, 18, 24]

# canonical linear GSTAR(K;1)
lin = {}
with open(CASA + r"\canonical_gstar_rmse.csv") as fh:
    for row in csv.DictReader(fh):
        if row["model"] == "GSTAR":
            lin[int(row["K"])] = float(row["canonical_rmse"])

# neural LSTM-GSTARX(K;1) from cv_summary
SUM = {}
with open(CASA + r"\cv_summary_table.csv") as fh:
    for row in csv.DictReader(fh):
        SUM[row["model"]] = float(row["rmse_mean"])
NEU = {1: SUM["casa_neighbours_only"], 6: SUM["casa_neighbours_only_lb6_gstarK1"],
       12: SUM["casa_neighbours_only_lb12_gstarK1"], 18: SUM["casa_neighbours_only_lb18_gstarK1"],
       24: SUM["casa_neighbours_only_lb24_gstarK1"]}

linv = [lin[k] for k in KS]
neu  = [NEU[k] for k in KS]

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(KS, linv, 's--', color='#B33A3A', lw=2.2, ms=9, label='Linear GSTAR(K;1)  — no LSTM (canonical)')
ax.plot(KS, neu, 'o-', color='#1FA84F', lw=2.6, ms=9, label='LSTM-GSTARX(K;1)  — neural')
ax.annotate("", xy=(12, neu[2]), xytext=(12, linv[2]), arrowprops=dict(arrowstyle='<->', color='0.35', lw=1.5))
ax.text(12.6, (linv[2] + neu[2]) / 2, f"the LSTM adds\n{linv[2]-neu[2]:.1f} mm", fontsize=9.5, color='0.15', va='center')
ax.annotate("longer memory\n(linear too)", xy=(6, linv[1]), xytext=(3.0, 86.5), fontsize=8.5, color='#B33A3A',
            arrowprops=dict(arrowstyle='->', color='#B33A3A', alpha=0.6))
ax.text(1.2, 80.6, "single-step\n80.3", fontsize=8.3, color='#1FA84F')
ax.set_xticks(KS); ax.set_xlabel("Temporal order K  (months of memory)")
ax.set_ylabel("RMSE (mm) — lower is better"); ax.set_ylim(68, 92); ax.set_xlim(0, 25.5)
ax.set_title("You need BOTH: longer memory AND the LSTM\nLinear with long memory stalls at ~81 mm; only the LSTM reaches 71 mm", fontsize=10.5)
ax.legend(fontsize=9, loc='upper right'); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(FIGDST + "/fig_linear_vs_neural.png", dpi=300); plt.close(fig)
print(f"linear canonical: {[round(v,2) for v in linv]}")
print(f"neural:           {[round(v,2) for v in neu]}")
print(f"At K=12: linear {linv[2]:.2f} vs neural {neu[2]:.2f} -> LSTM adds {linv[2]-neu[2]:.2f} mm")
print("Wrote fig_linear_vs_neural.png (canonical)")
