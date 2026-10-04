#!/usr/bin/env python3
"""Validate and summarize the August 2026 exhaustive Cells 1--4 run.

The script uses only the Python standard library.  It verifies that all four
cells contain the same ordered 50-example cohort before producing the summary,
paired comparison, run metadata, and an optional portable-report artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any


RUN_ID = "cells123_exhaustive_ynez_20260805"
GENERATED_AT = "2026-08-06T23:30:00Z"
EXPECTED_INPUT_SHA256 = "82b7f112eaca68f2456283745f72e495c351f85f2a37921f7a9871f73ffa4b53"
CELL_SPECS = (
    (1, "cell1_flat_raw", "Flat / raw 32k prefix", "Raw text", "n/a", "flat"),
    (2, "cell2_flat_kg_noscallop", "Flat / KG", "KG", "none", "flat"),
    (3, "cell3_flat_kg_scallop", "Flat / KG + Scallop", "KG", "scallop", "flat"),
    (4, "cell4_rlm_raw", "RLM / raw full source", "Raw text", "n/a", "rlm"),
)
EXPECTED_LABELS = {
    1: "flat_raw",
    2: "flat_kg_noscallop",
    3: "flat_kg_scallop",
    4: "rlm_raw",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    # Some LongBench strings contain Unicode line-separator characters.  They
    # are valid inside JSON strings but ``str.splitlines()`` treats them as
    # record boundaries, so split only on the JSONL newline delimiter.
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    denominator = 1 + z * z / total
    center = (correct / total + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            (correct / total * (1 - correct / total) + z * z / (4 * total))
            / total
        )
        / denominator
    )
    return center - radius, center + radius


def exact_paired_p(cell_a_only: int, cell_b_only: int) -> float:
    discordant = cell_a_only + cell_b_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(cell_a_only, cell_b_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def chance_lower_tail(correct: int, total: int, chance: float = 0.25) -> float:
    return sum(
        math.comb(total, k) * chance**k * (1 - chance) ** (total - k)
        for k in range(correct + 1)
    )


def parse_kg_counts(log_path: Path) -> dict[str, int | float]:
    text = log_path.read_text()
    no_scallop = re.search(r"noscallop committed=(\d+).*mirrored=(\d+)", text)
    rejected_matches = re.findall(r"scallop_rejected=(\d+)", text)
    scallop = re.search(
        rf"session={RUN_ID}_scallop committed=(\d+).*mirrored=(\d+)",
        text,
    )
    if not no_scallop or not scallop or not rejected_matches:
        raise AssertionError("could not recover KG counts from assembly log")
    candidates = int(no_scallop.group(1))
    rejected = int(rejected_matches[-1])
    accepted = int(scallop.group(1))
    assert candidates == int(no_scallop.group(2))
    assert accepted == int(scallop.group(2))
    assert accepted + rejected == candidates
    return {
        "candidate_facts": candidates,
        "scallop_accepted": accepted,
        "scallop_rejected": rejected,
        "scallop_retention": accepted / candidates,
    }


def source(source_id: str, label: str, path: str, description: str) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "python",
            "language": "python",
            "sql": "python3 results/cells1_4_exhaustive_20260805/analyze_results.py",
            "description": description,
            "executed_at": GENERATED_AT,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional frozen pilot_input.jsonl; when supplied, its hash, IDs, and gold labels are audited.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument(
        "--write-portable-artifact",
        action="store_true",
        help="Also write artifact.json for a future portable-report renderer.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    raw_root = output_dir / "raw"
    rows_by_cell: dict[int, list[dict[str, Any]]] = {}
    result_paths: dict[int, Path] = {}

    if args.input:
        pilot_rows = read_jsonl(args.input)
        assert len(pilot_rows) == 50, f"expected 50 input examples, found {len(pilot_rows)}"
        assert sha256(args.input) == EXPECTED_INPUT_SHA256, "input hash differs from the frozen run input"
        pilot_ids = [str(row["_id"]) for row in pilot_rows]
        input_by_id = {str(row["_id"]): row for row in pilot_rows}
        input_audit = "Frozen input hash, ordered IDs, and gold labels revalidated."
    else:
        # The full LongBench contexts are intentionally not duplicated in the
        # results package.  Use Cell 1 as the archived cohort manifest and
        # cross-check every other cell against its ordered IDs and gold labels.
        manifest_rows = read_jsonl(raw_root / "cell1_flat_raw" / "results.jsonl")
        pilot_ids = [str(row["example_id"]) for row in manifest_rows]
        input_by_id = {
            str(row["example_id"]): {"answer": row["gold"]}
            for row in manifest_rows
        }
        input_audit = "Archived result cohorts revalidated; the frozen source input was not duplicated in this package."
    assert len(set(pilot_ids)) == 50, "cohort contains duplicate example IDs"

    for cell_id, directory, *_ in CELL_SPECS:
        path = raw_root / directory / "results.jsonl"
        rows = read_jsonl(path)
        assert len(rows) == 50, f"Cell {cell_id}: expected 50 rows, found {len(rows)}"
        assert [str(row["example_id"]) for row in rows] == pilot_ids
        assert len({str(row["example_id"]) for row in rows}) == 50
        assert all(int(row["cell_id"]) == cell_id for row in rows)
        assert all(row["label"] == EXPECTED_LABELS[cell_id] for row in rows)
        assert all(row["run_id"] == RUN_ID for row in rows)
        assert all(row["gold"] == input_by_id[str(row["example_id"])]["answer"] for row in rows)
        assert all(
            not row.get("predicted") or row.get("predicted") in {"A", "B", "C", "D"}
            for row in rows
        )
        assert all(not row.get("error") for row in rows)
        rows_by_cell[cell_id] = rows
        result_paths[cell_id] = path

    assert all(
        rows_by_cell[1][index]["gold"] == rows_by_cell[cell][index]["gold"]
        for cell in (2, 3, 4)
        for index in range(50)
    )
    assert all(row["configured_retrieval_mode"] == "hybrid" for row in rows_by_cell[2])
    assert all(row["configured_retrieval_mode"] == "hybrid" for row in rows_by_cell[3])
    assert all(not row["retrieval_degraded"] for row in rows_by_cell[2])
    assert all(not row["retrieval_degraded"] for row in rows_by_cell[3])
    assert all(row["memory_scope"] == "example" for row in rows_by_cell[2])
    assert all(row["memory_scope"] == "example" for row in rows_by_cell[3])
    assert all(row["rrf_k"] == 60 for row in rows_by_cell[2])
    assert all(row["rrf_k"] == 60 for row in rows_by_cell[3])

    summaries: list[dict[str, Any]] = []
    for cell_id, _directory, display, evidence, validator, recursion in CELL_SPECS:
        rows = rows_by_cell[cell_id]
        correct = sum(bool(row["correct"]) for row in rows)
        low, high = wilson_interval(correct, len(rows))
        triples = [int(row.get("n_triples") or 0) for row in rows]
        context_chars = [int(row.get("n_context_chars") or 0) for row in rows]
        latencies = [float(row.get("elapsed_seconds") or 0) for row in rows]
        sparse_zero = sum(
            int((row.get("retrieval_branch_counts") or {}).get("sparse", -1)) == 0
            for row in rows
        ) if cell_id in {2, 3} else None
        summaries.append(
            {
                "cell_id": cell_id,
                "cell": f"Cell {cell_id}",
                "display": display,
                "label": EXPECTED_LABELS[cell_id],
                "evidence": evidence,
                "validator": validator,
                "recursion": recursion,
                "n_examples": len(rows),
                "n_answered": sum(row.get("predicted") in {"A", "B", "C", "D"} for row in rows),
                "correct": correct,
                "accuracy": correct / len(rows),
                "accuracy_pct": round(100 * correct / len(rows), 1),
                "ci95_low": low,
                "ci95_high": high,
                "ci95": f"{100 * low:.1f}%–{100 * high:.1f}%",
                "mean_latency_s": mean(latencies),
                "median_latency_s": median(latencies),
                "mean_triples": mean(triples),
                "zero_triple_examples": sum(value == 0 for value in triples),
                "mean_context_chars": mean(context_chars),
                "sparse_zero_examples": sparse_zero,
                "n_errors": sum(bool(row.get("error")) for row in rows),
            }
        )

    cell2 = {row["example_id"]: row for row in rows_by_cell[2]}
    cell3 = {row["example_id"]: row for row in rows_by_cell[3]}
    paired_counts = Counter(
        (bool(cell2[example_id]["correct"]), bool(cell3[example_id]["correct"]))
        for example_id in pilot_ids
    )
    cell2_only = paired_counts[(True, False)]
    cell3_only = paired_counts[(False, True)]
    paired = {
        "both_correct": paired_counts[(True, True)],
        "cell2_only_correct": cell2_only,
        "cell3_only_correct": cell3_only,
        "both_wrong": paired_counts[(False, False)],
        "same_prediction": sum(cell2[key]["predicted"] == cell3[key]["predicted"] for key in pilot_ids),
        "different_prediction": sum(cell2[key]["predicted"] != cell3[key]["predicted"] for key in pilot_ids),
        "mcnemar_exact_two_sided_p": exact_paired_p(cell2_only, cell3_only),
        "cell2_chance_lower_tail_p": chance_lower_tail(sum(row["correct"] for row in rows_by_cell[2]), 50),
    }

    kg = parse_kg_counts(output_dir / "logs" / "cells23-assemble-5c1e0dd.log")
    assert summaries[1]["zero_triple_examples"] == 1
    assert summaries[2]["zero_triple_examples"] == 1

    summary_columns = [
        "cell_id", "label", "evidence", "validator", "recursion", "n_examples",
        "n_answered", "correct", "accuracy", "ci95_low", "ci95_high",
        "mean_latency_s", "median_latency_s", "mean_triples",
        "zero_triple_examples", "sparse_zero_examples", "n_errors",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        for summary in summaries:
            row = {key: summary.get(key) for key in summary_columns}
            for key in ("accuracy", "ci95_low", "ci95_high", "mean_latency_s", "median_latency_s", "mean_triples"):
                if isinstance(row[key], float):
                    row[key] = f"{row[key]:.4f}"
            writer.writerow(row)

    paired_rows = [
        {"paired_outcome": "Both correct", "examples": paired["both_correct"]},
        {"paired_outcome": "Cell 2 only correct", "examples": paired["cell2_only_correct"]},
        {"paired_outcome": "Cell 3 only correct", "examples": paired["cell3_only_correct"]},
        {"paired_outcome": "Both wrong", "examples": paired["both_wrong"]},
    ]
    (output_dir / "paired_comparison.json").write_text(json.dumps(paired, indent=2) + "\n")

    valid_predictions = sum(
        row.get("predicted") in {"A", "B", "C", "D"}
        for rows in rows_by_cell.values()
        for row in rows
    )
    blank_predictions = sum(len(rows) for rows in rows_by_cell.values()) - valid_predictions
    run_metadata = {
        "run_id": RUN_ID,
        "completed_date": "2026-08-06",
        "completed_at_note": "The exact completion timestamp is not recoverable from the copied logs.",
        "report_generated_at": GENERATED_AT,
        "server": "ynez",
        "model": "Qwen/Qwen3-4B",
        "sample": {"dataset": "LongBench v2", "limit": 50, "seed": 0},
        "input_sha256": EXPECTED_INPUT_SHA256,
        "input_audit": input_audit,
        "code": {
            "cell1_and_candidate_extraction": "76f72138e065afa2c5260efbe2e18531c746e447",
            "cell2_cell3_and_cell4_runner": "5c1e0dd241d537d79a743adacee2557450a6839f",
            "note": "The later commit changes Cell 4 limits and exhaustive defaults; Cell 2/3 answer logic is unchanged from the extraction checkout.",
        },
        "kg": kg,
        "results_sha256": {
            f"cell{cell_id}": sha256(path) for cell_id, path in result_paths.items()
        },
        "validation": {
            "same_ordered_example_ids": True,
            "same_gold_labels": True,
            "valid_mcq_predictions": valid_predictions,
            "blank_or_invalid_predictions": blank_predictions,
            "runtime_errors": 0,
            "retrieval_degradations_cells2_3": 0,
            "status": "share_with_caveats",
        },
        "remote_cleanup": {
            "completed": True,
            "gpu_memory_after_mib": [4, 4, 4, 4],
            "gpu_utilization_after_pct": [0, 0, 0, 0],
            "tmux_sessions_remaining": 0,
            "run_graph_nodes_deleted": 29810,
            "run_graph_relationships_deleted": 27264,
            "orphan_endpoint_entities_deleted": 5908,
            "run_graph_prefix_remaining": 0,
            "isolated_paths_remaining": 0,
            "note": "Only locally archived result rows and logs remain; shared model caches and the shared Neo4j installation were not run artifacts and were not removed.",
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")

    result_source = source(
        "four_cell_results",
        "Verified Cells 1–4 result rows",
        "results/cells1_4_exhaustive_20260805/summary.csv",
        "Validates the four raw JSONL files and recomputes observed accuracy, Wilson intervals, latency, and evidence counts.",
    )
    kg_source = source(
        "kg_assembly_log",
        "Paired KG assembly log",
        "results/cells1_4_exhaustive_20260805/logs/cells23-assemble-5c1e0dd.log",
        "Parses the no-Scallop and Scallop imports from the same frozen 9,681-fact candidate corpus.",
    )

    design_rows = [
        {"cell": "Cell 1", "answerer": "Flat", "evidence": "Raw 32k-character prefix", "validator": "n/a", "retrieval": "None"},
        {"cell": "Cell 2", "answerer": "Flat", "evidence": "Exhaustive KG", "validator": "None", "retrieval": "Hybrid, k=50"},
        {"cell": "Cell 3", "answerer": "Flat", "evidence": "Same exhaustive KG candidates", "validator": "Scallop", "retrieval": "Hybrid, k=50"},
        {"cell": "Cell 4", "answerer": "RLM", "evidence": "Raw full source via bounded subqueries", "validator": "n/a", "retrieval": "RLM subqueries"},
    ]
    kg_rows = [
        {"stage": "Unvalidated candidates", "facts": kg["candidate_facts"], "share": 1.0},
        {"stage": "Scallop accepted", "facts": kg["scallop_accepted"], "share": kg["scallop_retention"]},
        {"stage": "Scallop rejected", "facts": kg["scallop_rejected"], "share": kg["scallop_rejected"] / kg["candidate_facts"]},
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "NeuroSym Cells 1–4: Exhaustive LongBench Run",
            "description": "Technical report for the paired 50-example seed-0 run completed on Ynez.",
            "generatedAt": GENERATED_AT,
            "cards": [
                {
                    "id": f"cell{summary['cell_id']}_accuracy",
                    "description": f"{summary['correct']} correct of 50; {summary['n_answered']} answered; 95% Wilson interval {summary['ci95']}.",
                    "dataset": "cell_summary",
                    "sourceId": "four_cell_results",
                    "filter": {"cell_id": summary["cell_id"]},
                    "metrics": [{"label": f"Cell {summary['cell_id']} accuracy", "field": "accuracy", "format": "percent"}],
                }
                for summary in summaries
            ],
            "charts": [
                {
                    "id": "accuracy_by_cell",
                    "title": "Accuracy by experimental cell",
                    "subtitle": "Same 50-example seed-0 cohort; exact 95% Wilson intervals appear in the adjacent table.",
                    "type": "bar",
                    "dataset": "cell_summary",
                    "sourceId": "four_cell_results",
                    "encodings": {
                        "x": {"field": "cell", "type": "ordinal", "label": "Experimental cell"},
                        "y": {"field": "accuracy", "type": "quantitative", "label": "Accuracy", "format": "percent"},
                        "tooltip": [
                            {"field": "correct", "type": "quantitative", "label": "Correct"},
                            {"field": "n_examples", "type": "quantitative", "label": "Examples"},
                            {"field": "ci95", "type": "nominal", "label": "95% Wilson interval"},
                        ],
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "accuracy_detail",
                    "title": "Observed results and uncertainty",
                    "subtitle": "Accuracy uses all 50 examples, including errors or abstentions; this run had zero runtime errors and one Cell 4 blank.",
                    "dataset": "cell_summary",
                    "sourceId": "four_cell_results",
                    "defaultSort": {"field": "cell_id", "direction": "asc"},
                    "columns": [
                        {"field": "cell", "label": "Cell", "type": "text"},
                        {"field": "display", "label": "Condition", "type": "text"},
                        {"field": "correct", "label": "Correct", "format": "number"},
                        {"field": "n_answered", "label": "Answered", "format": "number"},
                        {"field": "accuracy", "label": "Accuracy", "format": "percent"},
                        {"field": "ci95", "label": "95% Wilson CI", "type": "text"},
                        {"field": "mean_latency_s", "label": "Mean latency (s)", "format": "number"},
                    ],
                },
                {
                    "id": "paired_cell23",
                    "title": "Paired Cell 2 versus Cell 3 outcomes",
                    "subtitle": "Each row counts the same question under the unvalidated and Scallop-validated KG conditions.",
                    "dataset": "paired_cell23",
                    "sourceId": "four_cell_results",
                    "defaultSort": {"field": "examples", "direction": "desc"},
                    "columns": [
                        {"field": "paired_outcome", "label": "Outcome", "type": "text"},
                        {"field": "examples", "label": "Examples", "format": "number"},
                    ],
                },
                {
                    "id": "design_matrix",
                    "title": "Cells included in this run",
                    "subtitle": "Cell 2 and Cell 3 form the cleanest paired treatment comparison.",
                    "dataset": "design",
                    "sourceId": "four_cell_results",
                    "defaultSort": {"field": "cell", "direction": "asc"},
                    "columns": [
                        {"field": "cell", "label": "Cell", "type": "text"},
                        {"field": "answerer", "label": "Answerer", "type": "text"},
                        {"field": "evidence", "label": "Evidence", "type": "text"},
                        {"field": "validator", "label": "Validator", "type": "text"},
                        {"field": "retrieval", "label": "Retrieval", "type": "text"},
                    ],
                },
                {
                    "id": "kg_filtering",
                    "title": "Scallop filtering of the frozen candidate corpus",
                    "subtitle": "Both KG arms began from the same 9,681 extracted candidates.",
                    "dataset": "kg_filtering",
                    "sourceId": "kg_assembly_log",
                    "defaultSort": {"field": "facts", "direction": "desc"},
                    "columns": [
                        {"field": "stage", "label": "Stage", "type": "text"},
                        {"field": "facts", "label": "Facts", "format": "number"},
                        {"field": "share", "label": "Share of candidates", "format": "percent"},
                    ],
                },
            ],
            "sources": [
                {"id": "four_cell_results", "label": result_source["label"], "path": result_source["path"]},
                {"id": "kg_assembly_log", "label": kg_source["label"], "path": kg_source["path"]},
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# NeuroSym Cells 1–4: Exhaustive LongBench Run"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "four_cell_results",
                    "body": "## Technical summary\n\n**All four cells completed the same ordered 50-example seed-0 cohort with zero runtime errors.** Observed accuracy was **22% for Cell 1**, **18% for Cell 2**, **30% for Cell 3**, and **34% for Cell 4**. Cell 4 is the observed leader, but all four confidence intervals overlap. The cleanest treatment comparison is Cell 2 versus Cell 3: Scallop filtering produced a **12 percentage-point observed gain**, but the paired exact test is suggestive rather than conclusive (**p = 0.070**). Treat this as a validated pilot result, not a final paper estimate.",
                },
                {"id": "accuracy_metrics", "type": "metric-strip", "cardIds": [f"cell{i}_accuracy" for i in range(1, 5)]},
                {
                    "id": "accuracy_finding",
                    "type": "markdown",
                    "sourceId": "four_cell_results",
                    "body": "## Cell 4 leads the observed ranking, with wide uncertainty\n\nThe chart compares observed multiple-choice accuracy on a common denominator of 50. Cell 4 answered 17 correctly, followed by Cell 3 with 15, Cell 1 with 11, and Cell 2 with 9. Because each arm has only 50 examples, the uncertainty remains large: the Wilson intervals overlap and the ranking should not be presented as definitive.",
                },
                {"id": "accuracy_chart", "type": "chart", "chartId": "accuracy_by_cell", "layout": "full"},
                {"id": "accuracy_table", "type": "table", "tableId": "accuracy_detail", "layout": "full"},
                {
                    "id": "scallop_finding",
                    "type": "markdown",
                    "sourceId": "four_cell_results",
                    "body": "## Scallop helped on this seed, but one more replicated run could change the inference\n\nCell 3 was correct where Cell 2 was wrong on seven questions; Cell 2 was uniquely correct on one. Eight questions were correct in both arms and 34 were wrong in both. The paired exact McNemar result is **p = 0.070**, just above the conventional 0.05 threshold. The direction is consistent with Scallop removing noisy evidence, but a single seed cannot establish a stable treatment effect.",
                },
                {"id": "paired_table", "type": "table", "tableId": "paired_cell23", "layout": "full"},
                {
                    "id": "kg_finding",
                    "type": "markdown",
                    "sourceId": "kg_assembly_log",
                    "body": "## Scallop removed 18.4% of candidate facts\n\nThe exhaustive build produced **9,681 unique candidates**. The no-Scallop graph retained all of them, while Scallop accepted **7,902** and rejected **1,779**. Cell 2 retrieved 41.1 triples per question on average, compared with 39.8 for Cell 3. The result is compatible with validation improving precision, but the run does not contain gold fact-level relevance labels, so that mechanism remains an inference.",
                },
                {"id": "kg_table", "type": "table", "tableId": "kg_filtering", "layout": "full"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": "## Scope and metric definitions\n\n**Population.** The frozen LongBench v2 seed-0 sample contains 50 unique examples, evaluated in identical order in every cell.\n\n**Accuracy.** Correct predictions divided by all 50 examples. Errors and abstentions remain in the denominator. This run had zero runtime errors; Cells 1–3 answered all 50 examples, while Cell 4 produced one blank parsed answer and therefore had 49 answered.\n\n**Uncertainty.** Reported intervals are 95% Wilson binomial intervals. The Cell 2/3 comparison uses an exact paired McNemar test over discordant outcomes. These are descriptive pilot statistics; no correction was applied for other exploratory comparisons.",
                },
                {"id": "design_intro", "type": "markdown", "body": "## Experimental design\n\nCells 2 and 3 use the same frozen extracted candidate corpus, the same flat answerer, the same question-only answer-time retrieval query, hybrid BGE/RRF retrieval, a 50-triple limit, and example-scoped memory. Their intended difference is Scallop validation. The raw-text cells use different orchestration and evidence exposure, so comparisons involving Cells 1 or 4 answer broader system questions rather than isolating a single component."},
                {"id": "design_table_block", "type": "table", "tableId": "design_matrix", "layout": "full"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## Limitations that materially affect interpretation\n\n- **Cell 1 and Cell 4 are not evidence-normalized.** Cell 1 receives a fixed 32,000-character prefix; Cell 4 can inspect the full source through bounded RLM subqueries. Their difference therefore combines recursion with broader evidence access.\n- **KG construction is question-conditioned.** Extraction uses the example question and choices. This is shared between Cells 2 and 3, so it does not invalidate that pair, but it prevents treating raw-versus-KG comparisons as a pure representation ablation.\n- **Hybrid often means dense-only here.** On 32 of 50 questions, the heuristic sparse branch produced zero candidates because it found no seed entity; dense retrieval still ran and no row degraded.\n- **One example produced no KG facts.** Both KG cells therefore answered that example without retrieved triples.\n- **Context fields are not cross-cell-equivalent.** Raw prefix characters, formatted triple characters, and full source characters available to RLM subqueries describe different quantities and should not be plotted as one metric.\n- **Single-seed uncertainty is substantial.** Cell 2's 18% is statistically compatible with four-choice chance, and the observed Cell 3 improvement is not yet conventionally significant.\n- **Historical top-level results are not comparable.** The earlier 52% Cell 2 artifact used pre-hybrid code and a different KG; this report replaces it rather than pooling it.",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": "## Recommended next steps\n\n1. Run the Cell 2/3 answer phase for at least two additional seeds. The Ynez run mirrors were removed during the requested cleanup, so they must be restored from an archive or rebuilt first.\n2. Run a paired retrieval-limit ablation at 10, 20, and 50 triples for both Cell 2 and Cell 3. This directly tests whether exhaustive unvalidated evidence is swamping the answerer.\n3. Persist selected fact IDs and formatted evidence excerpts per question so retrieval precision and Scallop's mechanism can be audited.\n4. Keep Cell 1/4 conclusions explicitly system-level unless both arms are redesigned to receive equivalent evidence.",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": "## Further questions\n\n- Does the Cell 3 advantage replicate across seeds and retrieval limits?\n- Which rejected facts account for the seven Cell 3-only successes?\n- Would a selector shared across the raw and KG arms yield a cleaner representation ablation without exceeding context limits?",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
            "status": "ready",
            "datasets": {
                "cell_summary": summaries,
                "paired_cell23": paired_rows,
                "design": design_rows,
                "kg_filtering": kg_rows,
            },
        },
        "sources": [result_source, kg_source],
        "package_info": {},
    }
    if args.write_portable_artifact:
        (output_dir / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")

    print(json.dumps({"cells": summaries, "paired": paired, "kg": kg}, indent=2))


if __name__ == "__main__":
    main()
