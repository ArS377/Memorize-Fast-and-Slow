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
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.common import CELLS, cell_output_path

SUMMARY_COLUMNS = [
    "cell_id", "label", "retrieval", "validator", "recursion",
    "n_examples", "n_answered", "accuracy", "mean_latency_s",
    "mean_triples", "n_errors", "n_diagnostic_answered", "diagnostic_accuracy",
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_examples = len(rows)
    answered = [r for r in rows if r.get("predicted") and not r.get("error")]
    n_answered = len(answered)
    correct = sum(1 for r in answered if r.get("correct"))
    accuracy = (correct / n_answered) if n_answered else None
    diagnostic_answered = [r for r in rows if r.get("diagnostic_predicted")]
    n_diagnostic_answered = len(diagnostic_answered)
    diagnostic_correct = sum(1 for r in diagnostic_answered if r.get("diagnostic_correct"))
    diagnostic_accuracy = (
        diagnostic_correct / n_diagnostic_answered if n_diagnostic_answered else None
    )
    latencies = [r.get("elapsed_seconds") for r in rows if isinstance(r.get("elapsed_seconds"), (int, float))]
    mean_latency = (sum(latencies) / len(latencies)) if latencies else None
    triples = [r.get("n_triples") for r in rows if isinstance(r.get("n_triples"), (int, float))]
    mean_triples = (sum(triples) / len(triples)) if triples else None
    n_errors = sum(1 for r in rows if r.get("error"))
    return {
        "n_examples": n_examples,
        "n_answered": n_answered,
        "accuracy": accuracy,
        "mean_latency_s": mean_latency,
        "mean_triples": mean_triples,
        "n_errors": n_errors,
        "n_diagnostic_answered": n_diagnostic_answered,
        "diagnostic_accuracy": diagnostic_accuracy,
    }


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
            for k in ("accuracy", "mean_latency_s", "mean_triples", "diagnostic_accuracy"):
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
            "n/a" if f_acc is None else f"{f_acc:.2f} (n={f.get('n_answered', 0)})"
        )
        rlm_label.append(
            "n/a" if r_acc is None else f"{r_acc:.2f} (n={r.get('n_answered', 0)})"
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
