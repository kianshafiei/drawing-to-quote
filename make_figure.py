"""Results figure for the README (results/summary.png)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = json.loads(Path("results/metrics.json").read_text())["by_level"]
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
LEVELS = {"0": ("clean", "#86b6ef"), "1": ("light scan noise", "#2a78d6"), "2": ("heavy scan noise", "#104281")}
FIELDS = ["family", "material", "finish", "qty", "tol", "holes", "threads", "envelope"]
NAMES = ["part type", "material", "finish", "quantity", "tolerance", "holes", "threads", "size (envelope)"]

fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.6, 1]}, facecolor=SURF)
for ax in (a, b):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

w = 0.26
for i, (lv, (lab, col)) in enumerate(LEVELS.items()):
    acc = [M[lv]["field_accuracy"][f] * 100 for f in FIELDS]
    a.plot([x + (i - 1) * w for x in range(len(FIELDS))], acc, "o", ms=8, color=col, label=lab,
           markeredgecolor=SURF, markeredgewidth=1.5)
a.set_xticks(range(len(FIELDS)), NAMES, rotation=20, ha="right", color=INK, fontsize=9)
a.set_ylabel("read correctly (%)", color=MUTED)
a.set_title("Fields read from 60 unseen drawings, by scan quality", loc="left", color=INK, fontsize=11)
a.set_ylim(70, 101)
a.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK, ncol=3)

m = M["all"]
rows = [("Price per kg rule", m["median_ape_perkg_true"]), ("Average of 5 similar jobs", m["median_ape_knn_true"]),
        ("Model, raw reading", m["median_ape_gbm_raw"]), ("Model, after review of flags", m["median_ape_gbm_reviewed"]),
        ("Model, perfect reading", m["median_ape_gbm_true"])]
y = range(len(rows))[::-1]
b.barh(list(y), [r[1] * 100 for r in rows], 0.6, color="#2a78d6")
for yy, (_, v) in zip(y, rows):
    b.text(v * 100 + 0.8, yy, f"{v * 100:.1f}%", va="center", color=INK, fontsize=9)
b.set_yticks(list(y), [r[0] for r in rows], color=INK, fontsize=9)
b.xaxis.grid(True, color=GRID, lw=0.8)
b.yaxis.grid(False)
b.set_xlabel("median error vs. the shop's price (%)", color=MUTED)
b.set_title("Quote error", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig("results/summary.png", dpi=150, facecolor=SURF)
print("saved")
