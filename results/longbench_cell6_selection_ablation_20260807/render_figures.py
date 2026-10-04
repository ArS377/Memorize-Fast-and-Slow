#!/usr/bin/env python3
"""Renders figures/accuracy_by_condition.png from summary.csv.

Usage: python3 results/longbench_cell6_selection_ablation_20260807/render_figures.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent


def main() -> None:
    with (ROOT / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))

    labels = [r["label"] for r in rows]
    acc = [float(r["accuracy"]) * 100 for r in rows]
    lo = [float(r["accuracy"]) * 100 - float(r["ci95_low"]) * 100 for r in rows]
    hi = [float(r["ci95_high"]) * 100 - float(r["accuracy"]) * 100 for r in rows]
    colors = ["#888888" if r["condition"] == "no_local_changes" else "#2f6fed" for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    y = range(len(labels))
    ax.barh(y, acc, xerr=[lo, hi], color=colors, capsize=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(25, color="black", linestyle="--", linewidth=1, label="25% chance (4-choice)")
    ax.set_xlabel("Accuracy (%) with 95% Wilson interval")
    ax.set_title("Cell 5/6 LongBench pilot accuracy by condition")
    ax.legend(loc="lower right")
    grey_patch = plt.Rectangle((0, 0), 1, 1, color="#888888", label="no local changes")
    blue_patch = plt.Rectangle((0, 0), 1, 1, color="#2f6fed", label="local changes")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [grey_patch, blue_patch], loc="lower right", fontsize=8)
    fig.tight_layout()

    figures_dir = ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig.savefig(figures_dir / "accuracy_by_condition.png", dpi=150)
    print(f"Wrote {figures_dir / 'accuracy_by_condition.png'}")


if __name__ == "__main__":
    main()
