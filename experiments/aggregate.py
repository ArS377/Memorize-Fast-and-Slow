"""Aggregate per-cell results into a single CSV + figure.

Robust to partial runs: any cell whose ``results.jsonl`` does not exist
is rendered as a blank row in the CSV and a muted "n/a" bar in the
figure.

Importable without ``neo4j`` / ``rlm`` / ``scallopy``: only depends on
``experiments.common`` (stdlib) and ``matplotlib`` for the figure.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurosym.application.experiment_io import CELLS, cell_output_path
from neurosym.reporting import read_jsonl, summarize_experiment_rows

SUMMARY_COLUMNS = [
    "cell_id", "label", "retrieval", "validator", "recursion",
    "n_examples", "n_answered", "accuracy",
    "n_grounded", "grounded_accuracy", "grounded_coverage",
    "n_fallback", "fallback_accuracy", "mean_latency_s",
    "mean_triples", "n_errors", "n_diagnostic_answered", "diagnostic_accuracy",
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return summarize_experiment_rows(rows)


def collect_summary(results_dir: Path) -> List[Dict[str, Any]]:
    """Build the 6-row summary table. Missing cells produce blank rows."""
    out: List[Dict[str, Any]] = []
    for cell in CELLS:
        path = cell_output_path(cell["cell_id"], cell["label"], results_dir)
        row = {
            "cell_id": cell["cell_id"],
            "label": cell["label"],
            "retrieval": cell["retrieval"],
            "validator": cell["validator"],
            "recursion": cell["recursion"],
            "n_examples": 0,
            "n_answered": 0,
            "accuracy": None,
            "n_grounded": 0,
            "grounded_accuracy": None,
            "grounded_coverage": None,
            "n_fallback": 0,
            "fallback_accuracy": None,
            "mean_latency_s": None,
            "mean_triples": None,
            "n_errors": 0,
            "n_diagnostic_answered": 0,
            "diagnostic_accuracy": None,
        }
        if path.exists():
            try:
                rows = _load_jsonl(path)
                row.update(_summarize_rows(rows))
            except Exception as e:
                print(f"[aggregate] failed to read {path}: {e}", file=sys.stderr)
        out.append(row)
    return out


def write_summary_csv(summary: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for row in summary:
            out_row = dict(row)
            for k in (
                "accuracy",
                "grounded_accuracy",
                "grounded_coverage",
                "fallback_accuracy",
                "mean_latency_s",
                "mean_triples",
                "diagnostic_accuracy",
            ):
                v = out_row.get(k)
                if isinstance(v, float):
                    out_row[k] = f"{v:.4f}"
            w.writerow({k: out_row.get(k, "") for k in SUMMARY_COLUMNS})


def render_figure(summary: List[Dict[str, Any]], path: Path) -> None:
    """2x3 grouped bar chart. Rows = {Flat, RLM}, columns = {Raw, KG, KG+Scallop}."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[aggregate] matplotlib unavailable, skipping figure: {e}", file=sys.stderr)
        return

    by_label = {row["label"]: row for row in summary}
    columns = [("Raw", "flat_raw", "rlm_raw"),
               ("KG", "flat_kg_noscallop", "rlm_kg_noscallop"),
               ("KG+Scallop", "flat_kg_scallop", "rlm_kg_scallop")]

    import numpy as np  # matplotlib hard-depends on numpy
    x = np.arange(len(columns))
    width = 0.38

    flat_acc, rlm_acc, flat_label, rlm_label = [], [], [], []
    flat_missing, rlm_missing = [], []
    for _name, flat_lab, rlm_lab in columns:
        f = by_label.get(flat_lab, {})
        r = by_label.get(rlm_lab, {})
        f_acc = f.get("accuracy")
        r_acc = r.get("accuracy")
        flat_missing.append(f_acc is None)
        rlm_missing.append(r_acc is None)
        flat_acc.append(0.0 if f_acc is None else float(f_acc))
        rlm_acc.append(0.0 if r_acc is None else float(r_acc))
        flat_label.append(
            "n/a" if f_acc is None else f"{f_acc:.2f} (n={f.get('n_examples', 0)})"
        )
        rlm_label.append(
            "n/a" if r_acc is None else f"{r_acc:.2f} (n={r.get('n_examples', 0)})"
        )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    flat_colors = ["#cccccc" if m else "#7fbf7f" for m in flat_missing]
    rlm_colors = ["#999999" if m else "#1f7a1f" for m in rlm_missing]
    bars1 = ax.bar(x - width / 2, flat_acc, width, label="Flat LLM", color=flat_colors, edgecolor="black")
    bars2 = ax.bar(x + width / 2, rlm_acc, width, label="RLM", color=rlm_colors, edgecolor="black")

    for b, lab in zip(bars1, flat_label):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, lab,
                ha="center", va="bottom", fontsize=8)
    for b, lab in zip(bars2, rlm_label):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, lab,
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in columns])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("LongBench-v2 Ablation: Flat LLM vs RLM x {Raw, KG, KG+Scallop}")
    ax.legend(loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> Dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--figure-path", type=Path, default=None)
    args = parser.parse_args(argv)

    summary_path = args.summary_path or (args.results_dir / "summary.csv")
    figure_path = args.figure_path or (args.results_dir / "figures" / "accuracy_grid.png")

    summary = collect_summary(args.results_dir)
    write_summary_csv(summary, summary_path)
    render_figure(summary, figure_path)
    print(f"[aggregate] summary -> {summary_path}", file=sys.stderr)
    print(f"[aggregate] figure  -> {figure_path}", file=sys.stderr)
    return {"summary": summary_path, "figure": figure_path}


if __name__ == "__main__":
    main()
