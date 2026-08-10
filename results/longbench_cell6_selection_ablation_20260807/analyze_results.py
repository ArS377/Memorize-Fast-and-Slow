#!/usr/bin/env python3
"""Regenerates summary.csv, paired_comparison.json, and run_metadata.json
from the raw/ result files in this directory. See README.md for methodology.

Usage: python3 results/longbench_cell6_selection_ablation_20260807/analyze_results.py
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).parent
RAW = ROOT / "raw"

# (row_label, cell_id, condition, kg_variant, retrieval_mode, path)
RUNS = [
    ("gitmain_cell5_hybrid", 5, "no_local_changes", "gitmain_cap3", "hybrid",
     RAW / "gitmain/cell5_hybrid/results.jsonl"),
    ("gitmain_cell6_hybrid", 6, "no_local_changes", "gitmain_cap3", "hybrid",
     RAW / "gitmain/cell6_hybrid/results.jsonl"),
    ("gitmain_cell6_dense_ppr", 6, "no_local_changes", "gitmain_cap3", "dense_ppr",
     RAW / "gitmain/cell6_dense_ppr/results.jsonl"),
    ("semantic20_hybrid", 6, "local_changes", "semantic20_flat", "hybrid",
     RAW / "semantic20/hybrid/results.jsonl"),
    ("semantic20_dense_ppr", 6, "local_changes", "semantic20_flat", "dense_ppr",
     RAW / "semantic20/dense_ppr/results.jsonl"),
    ("peropt_hybrid", 6, "local_changes", "semantic20_peropt", "hybrid",
     RAW / "peropt/hybrid/results.jsonl"),
    ("peropt_dense_ppr", 6, "local_changes", "semantic20_peropt", "dense_ppr",
     RAW / "peropt/dense_ppr/results.jsonl"),
    ("relfloor_hybrid", 6, "local_changes", "semantic20_relfloor", "hybrid",
     RAW / "relfloor/hybrid/results.jsonl"),
    ("relfloor_dense_ppr", 6, "local_changes", "semantic20_relfloor", "dense_ppr",
     RAW / "relfloor/dense_ppr/results.jsonl"),
    ("mmr_hybrid", 6, "local_changes", "semantic20_mmr", "hybrid",
     RAW / "mmr/hybrid/results.jsonl"),
    ("mmr_dense_ppr", 6, "local_changes", "semantic20_mmr", "dense_ppr",
     RAW / "mmr/dense_ppr/results.jsonl"),
    ("combined_hybrid", 6, "local_changes", "semantic20_combined", "hybrid",
     RAW / "combined/hybrid/results.jsonl"),
    ("combined_dense_ppr", 6, "local_changes", "semantic20_combined", "dense_ppr",
     RAW / "combined/dense_ppr/results.jsonl"),
]


def wilson_ci(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((center - margin) / denom, (center + margin) / denom)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def fact_count(rows: list[dict]) -> "int | None":
    for r in rows:
        idx = r.get("dense_index_identity")
        if idx:
            return idx[0].get("fact_count")
    return None


def snapshot_sha(rows: list[dict]) -> "str | None":
    for r in rows:
        idx = r.get("dense_index_identity")
        if idx:
            return idx[0].get("source_snapshot_sha256")
    return None


def main() -> None:
    summary_rows = []
    by_label = {}
    for label, cell_id, condition, kg_variant, mode, path in RUNS:
        if not path.exists():
            continue
        rows = load(path)
        n = len(rows)
        answered = [r for r in rows if r.get("predicted")]
        correct = sum(1 for r in rows if r.get("correct"))
        errors = sum(1 for r in rows if r.get("error"))
        latencies = [r["elapsed_seconds"] for r in rows if r.get("elapsed_seconds") is not None]
        triples = [r.get("n_triples", 0) or 0 for r in rows]
        zero_triple = sum(1 for t in triples if t == 0)
        lo, hi = wilson_ci(correct, n)
        by_label[label] = rows
        summary_rows.append({
            "label": label,
            "cell_id": cell_id,
            "condition": condition,
            "kg_variant": kg_variant,
            "retrieval_mode": mode,
            "n_examples": n,
            "n_answered": len(answered),
            "correct": correct,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "ci95_low": round(lo, 4),
            "ci95_high": round(hi, 4),
            "mean_latency_s": round(statistics.mean(latencies), 2) if latencies else None,
            "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
            "mean_triples": round(statistics.mean(triples), 2) if triples else None,
            "zero_triple_examples": zero_triple,
            "n_errors": errors,
            "kg_fact_count": fact_count(rows),
        })

    with (ROOT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Paired comparison: no-local-changes vs. best local-changes variant
    # (combined), matched by example_id, per retrieval mode.
    def paired(label_a: str, label_b: str) -> dict:
        rows_a = {r["example_id"]: r for r in by_label[label_a]}
        rows_b = {r["example_id"]: r for r in by_label[label_b]}
        shared = sorted(set(rows_a) & set(rows_b))
        both_correct = a_only = b_only = both_wrong = 0
        for ex in shared:
            ca, cb = bool(rows_a[ex]["correct"]), bool(rows_b[ex]["correct"])
            if ca and cb:
                both_correct += 1
            elif ca and not cb:
                a_only += 1
            elif cb and not ca:
                b_only += 1
            else:
                both_wrong += 1
        # exact two-sided binomial (McNemar) over discordant pairs
        n_disc = a_only + b_only
        if n_disc == 0:
            p = 1.0
        else:
            k = min(a_only, b_only)
            p = 0.0
            for i in range(0, k + 1):
                p += math.comb(n_disc, i) * (0.5 ** n_disc)
            p = min(1.0, 2 * p)
        return {
            "a_label": label_a,
            "b_label": label_b,
            "n_shared": len(shared),
            "both_correct": both_correct,
            "a_only_correct": a_only,
            "b_only_correct": b_only,
            "both_wrong": both_wrong,
            "same_prediction_or_both_wrong_count": both_correct + both_wrong,
            "mcnemar_exact_two_sided_p": round(p, 4),
        }

    paired_comparison = {
        "hybrid": paired("gitmain_cell6_hybrid", "combined_hybrid"),
        "dense_ppr": paired("gitmain_cell6_dense_ppr", "combined_dense_ppr"),
    }
    (ROOT / "paired_comparison.json").write_text(
        json.dumps(paired_comparison, indent=2) + "\n"
    )

    metadata = {
        "sample": {"dataset": "LongBench v2", "limit": 50},
        "model": "Qwen/Qwen3-4B",
        "server": "serrano",
        "code": {
            "no_local_changes_kg_build": "a0678d8 (origin/main)",
            "no_local_changes_eval_controller": "08f1bc8 (origin/agent-set-rlm-depth-one, unmerged as of this run -- has the intended --max-depth 1; main still defaults to 2)",
            "local_changes": (
                "uncommitted local working-tree changes at the time of these runs "
                "(semantic chunking, per-option query fan-out, relevance-floor "
                "filtering, MMR diversity reranking in "
                "neurosym/adapters/chunk_selection.py and longbench_kg_pipeline.py), "
                "stashed after these runs and not yet pushed"
            ),
        },
        "kg_fact_counts": {row["kg_variant"]: row["kg_fact_count"] for row in summary_rows},
        "kg_snapshot_sha256": {
            label: snapshot_sha(rows) for label, rows in by_label.items()
            if snapshot_sha(rows)
        },
    }
    (ROOT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote summary.csv ({len(summary_rows)} rows), paired_comparison.json, run_metadata.json")


if __name__ == "__main__":
    main()
