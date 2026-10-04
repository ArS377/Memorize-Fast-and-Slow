#!/usr/bin/env python3
"""Render the static figures embedded in the GitHub technical report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


with (ROOT / "summary.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))

labels = [f"Cell {row['cell_id']}" for row in rows]
accuracy = np.array([float(row["accuracy"]) for row in rows])
ci_low = np.array([float(row["ci95_low"]) for row in rows])
ci_high = np.array([float(row["ci95_high"]) for row in rows])
errors = np.vstack((accuracy - ci_low, ci_high - accuracy))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor="white")
x = np.arange(len(rows))
bars = ax.bar(
    x,
    accuracy,
    width=0.62,
    color="#2f6fa3",
    edgecolor="#17364f",
    linewidth=1.1,
    yerr=errors,
    capsize=6,
    error_kw={"ecolor": "#242424", "elinewidth": 1.4, "capthick": 1.4},
)
for bar, row in zip(bars, rows):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + float(row["ci95_high"]) - float(row["accuracy"]) + 0.025,
        f"{100 * float(row['accuracy']):.0f}%\n({row['correct']}/50)",
        ha="center",
        va="bottom",
        color="#242424",
        fontsize=10,
    )
ax.axhline(0.25, color="#9a6b16", linestyle="--", linewidth=1.4, label="Four-choice chance (25%)")
ax.set_xticks(x, labels)
ax.set_ylim(0, 0.60)
ax.set_ylabel("Accuracy")
fig.suptitle("Accuracy by experimental cell", x=0.10, y=0.98, ha="left", fontsize=16, weight="bold", color="#242424")
fig.text(0.10, 0.925, "Same 50-example seed-0 cohort; error bars are 95% Wilson intervals", ha="left", color="#5d5d5d", fontsize=10)
ax.yaxis.set_major_formatter(lambda value, _position: f"{100 * value:.0f}%")
ax.yaxis.grid(True, color="#dedede", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(FIGURES / "accuracy_by_cell.png", dpi=180, bbox_inches="tight")
plt.close(fig)


kg = json.loads((ROOT / "run_metadata.json").read_text())["kg"]
accepted = int(kg["scallop_accepted"])
rejected = int(kg["scallop_rejected"])
total = int(kg["candidate_facts"])

fig, ax = plt.subplots(figsize=(9.2, 3.2), facecolor="white")
accepted_share = accepted / total
rejected_share = rejected / total
ax.barh([0], [accepted_share], color="#2f6fa3", edgecolor="#17364f", height=0.46, label="Accepted")
ax.barh([0], [rejected_share], left=[accepted_share], color="#d58b32", edgecolor="#6d481e", height=0.46, label="Rejected")
ax.text(accepted_share / 2, 0, f"Accepted\n{accepted:,} ({accepted_share:.1%})", ha="center", va="center", color="white", weight="bold")
ax.text(accepted_share + rejected_share / 2, 0, f"Rejected\n{rejected:,} ({rejected_share:.1%})", ha="center", va="center", color="#242424", weight="bold")
ax.set_xlim(0, 1)
ax.set_yticks([])
ax.set_xlabel("Share of frozen candidate facts")
fig.suptitle("Scallop disposition of the frozen KG candidate corpus", x=0.06, y=0.98, ha="left", fontsize=16, weight="bold", color="#242424")
fig.text(0.06, 0.89, f"{total:,} candidates extracted exhaustively before validation", ha="left", color="#5d5d5d", fontsize=10)
ax.xaxis.set_major_formatter(lambda value, _position: f"{100 * value:.0f}%")
ax.spines[["top", "right", "left"]].set_visible(False)
ax.xaxis.grid(True, color="#dedede", linewidth=0.8)
ax.set_axisbelow(True)
fig.tight_layout(rect=(0, 0, 1, 0.80))
fig.savefig(FIGURES / "kg_fact_filtering.png", dpi=180, bbox_inches="tight")
plt.close(fig)
