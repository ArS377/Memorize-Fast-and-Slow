from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from neurosym.reporting import read_jsonl


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(Path(path))


def recall_at_k(retrieved_fact_ids: Sequence[str], relevant_fact_ids: Iterable[str], k: int) -> float:
    relevant = {str(value) for value in relevant_fact_ids if str(value)}
    if not relevant:
        return 0.0
    found = relevant.intersection(str(value) for value in retrieved_fact_ids[:k])
    return len(found) / len(relevant)


def evaluate_retrieval_rows(rows: Sequence[Mapping[str, Any]], k_values: Sequence[int] = (1, 5, 10)) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    citation_proxy: Dict[tuple[str, str], set[str]] = defaultdict(set)
    has_ppr = any(
        str(row.get("mode") or row.get("retrieval_mode")) == "dense_ppr"
        or "ppr_fact_ids" in row
        or "ppr" in row.get("branch_counts", {})
        for row in rows
    )
    for row in rows:
        grouped[str(row.get("mode") or row.get("retrieval_mode") or "unknown")].append(row)
        if str(row.get("relevance_source", "")) == "cited_fact_ids":
            key = (str(row.get("cell_id", "")), str(row.get("example_id", "")))
            citation_proxy[key].update(str(value) for value in row.get("relevant_fact_ids", []))

    def relevant_ids(row: Mapping[str, Any]) -> List[str]:
        if str(row.get("relevance_source", "")) == "cited_fact_ids":
            key = (str(row.get("cell_id", "")), str(row.get("example_id", "")))
            return sorted(citation_proxy[key])
        return [str(value) for value in row.get("relevant_fact_ids", [])]

    modes: Dict[str, Any] = {}
    for mode, mode_rows in sorted(grouped.items()):
        labeled_rows = [row for row in mode_rows if relevant_ids(row)]
        recall = {
            f"recall_at_{k}": (
                round(
                    mean(
                        recall_at_k(
                            list(row.get("retrieved_fact_ids", [])),
                            relevant_ids(row),
                            k,
                        )
                        for row in labeled_rows
                    ),
                    6,
                )
                if labeled_rows
                else None
            )
            for k in k_values
        }
        branch_counts = {
            "sparse": sum(int(row.get("branch_counts", {}).get("sparse", 0)) for row in mode_rows),
            "dense": sum(int(row.get("branch_counts", {}).get("dense", 0)) for row in mode_rows),
        }
        if has_ppr:
            branch_counts["ppr"] = sum(
                int(row.get("branch_counts", {}).get("ppr", 0))
                for row in mode_rows
            )
        unique_retrieved = {
            str(fact_id)
            for row in mode_rows
            for fact_id in row.get("retrieved_fact_ids", [])
        }
        sparse_unique = {
            str(fact_id)
            for row in mode_rows
            for fact_id in row.get("sparse_fact_ids", [])
        }
        dense_unique = {
            str(fact_id)
            for row in mode_rows
            for fact_id in row.get("dense_fact_ids", [])
        }
        ppr_unique = {
            str(fact_id)
            for row in mode_rows
            for fact_id in row.get("ppr_fact_ids", [])
        }
        branch_overlap = (
            dense_unique.intersection(ppr_unique)
            if mode == "dense_ppr"
            else sparse_unique.intersection(dense_unique)
        )
        duplicate_count = sum(
            max(0, len(list(row.get("pre_fusion_fact_ids", []))) - len(set(row.get("pre_fusion_fact_ids", []))))
            for row in mode_rows
        )
        pre_fusion_count = sum(len(list(row.get("pre_fusion_fact_ids", []))) for row in mode_rows)
        mode_metrics = {
            "query_count": len(mode_rows),
            "labeled_query_count": len(labeled_rows),
            "relevance_source_counts": {
                source: sum(
                    str(row.get("relevance_source", "unlabeled")) == source
                    for row in mode_rows
                )
                for source in sorted(
                    {str(row.get("relevance_source", "unlabeled")) for row in mode_rows}
                )
            },
            **recall,
            "branch_candidate_counts": branch_counts,
            "unique_retrieved_fact_count": len(unique_retrieved),
            "sparse_unique_fact_count": len(sparse_unique),
            "dense_unique_fact_count": len(dense_unique),
            "branch_overlap_fact_count": len(branch_overlap),
            "fusion_duplicate_rate": round(duplicate_count / pre_fusion_count, 6) if pre_fusion_count else 0.0,
            "degraded_query_count": sum(bool(row.get("degraded")) for row in mode_rows),
        }
        if has_ppr:
            mode_metrics["ppr_unique_fact_count"] = len(ppr_unique)
        modes[mode] = mode_metrics
    warnings: List[Dict[str, Any]] = []
    sparse = modes.get("sparse")
    if sparse:
        for mode in ("dense", "hybrid", "dense_ppr"):
            candidate = modes.get(mode)
            if not candidate:
                continue
            for k in k_values:
                key = f"recall_at_{k}"
                baseline_value = sparse.get(key)
                candidate_value = candidate.get(key)
                if baseline_value is not None and candidate_value is not None and candidate_value < baseline_value:
                    warnings.append(
                        {
                            "code": "retrieval_recall_regression",
                            "mode": mode,
                            "metric": key,
                            "baseline_mode": "sparse",
                            "baseline_value": baseline_value,
                            "value": candidate_value,
                            "delta": round(candidate_value - baseline_value, 6),
                        }
                    )
    for mode, metrics in sorted(modes.items()):
        if metrics["degraded_query_count"]:
            warnings.append(
                {
                    "code": "retrieval_mode_degraded",
                    "mode": mode,
                    "query_count": metrics["degraded_query_count"],
                }
            )
    return {
        "schema_version": "retrieval_report.v1",
        "citation_proxy_method": "cross_mode_union_by_cell_and_example",
        "modes": modes,
        "warnings": warnings,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    def metric(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    lines = [
        "# Retrieval Quality Report",
        "",
    ]
    has_ppr = any(
        "ppr" in metrics.get("branch_candidate_counts", {})
        for metrics in report.get("modes", {}).values()
    )
    if has_ppr:
        lines.extend(
            [
                "| Mode | Queries | Recall@1 | Recall@5 | Recall@10 | Sparse candidates | Dense candidates | PPR candidates | Fusion duplicate rate | Degraded |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| Mode | Queries | Recall@1 | Recall@5 | Recall@10 | Sparse candidates | Dense candidates | Fusion duplicate rate | Degraded |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for mode, metrics in sorted(report.get("modes", {}).items()):
        counts = metrics.get("branch_candidate_counts", {})
        ppr_column = f"{counts.get('ppr', 0)} | " if has_ppr else ""
        lines.append(
            f"| {mode} | {metrics.get('query_count', 0)} | "
            f"{metric(metrics.get('recall_at_1'))} | "
            f"{metric(metrics.get('recall_at_5'))} | "
            f"{metric(metrics.get('recall_at_10'))} | "
            f"{counts.get('sparse', 0)} | {counts.get('dense', 0)} | "
            f"{ppr_column}"
            f"{metrics.get('fusion_duplicate_rate', 0.0):.4f} | "
            f"{metrics.get('degraded_query_count', 0)} |"
        )
    lines.extend(["", "## Regression Warnings", ""])
    warnings = list(report.get("warnings", []))
    if warnings:
        for warning in warnings:
            if warning.get("code") == "retrieval_recall_regression":
                lines.append(
                    f"- **{warning.get('mode')} {warning.get('metric')}**: "
                    f"{float(warning.get('value', 0.0)):.4f} versus sparse "
                    f"{float(warning.get('baseline_value', 0.0)):.4f} "
                    f"(delta {float(warning.get('delta', 0.0)):.4f})."
                )
            else:
                lines.append(
                    f"- **{warning.get('mode')} degraded**: "
                    f"{warning.get('query_count', 0)} queries used a fallback mode."
                )
    else:
        lines.append("- **None**: no measured regression or degradation was detected.")
    lines.extend(["", "## Fusion Diagnostics", ""])
    for mode, metrics in sorted(report.get("modes", {}).items()):
        detail = (
            f"; {metrics.get('ppr_unique_fact_count', 0)} facts were returned by PPR"
            if has_ppr
            else ""
        )
        lines.append(
            f"- **{mode}**: {metrics.get('unique_retrieved_fact_count', 0)} unique facts; "
            f"{metrics.get('branch_overlap_fact_count', 0)} facts occurred in both branches"
            f"{detail}."
        )
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = [row for path in args.input for row in _read_jsonl(path)]
    report = evaluate_retrieval_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "retrieval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "retrieval_report.md").write_text(
        render_markdown(report),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
