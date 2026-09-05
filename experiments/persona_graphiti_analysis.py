"""Render authenticated Graphiti benchmark artifacts into README and analysis.

Consumes a completed run directory (``metrics.json``, ``manifest.json``,
``predictions.jsonl``) and emits the two documents the persona benchmarks
publish: a headline ``README.md`` with matched effect sizes and clustered
bootstrap intervals, and an ``analysis.md`` with the per-family, per-phase, and
per-token-distance breakdown the pre-registered predictions are checked against.

It adds one analysis the generation metrics do not carry: an error taxonomy
separating a correct answer from a **stale intrusion** (a value that was this
account's preference at some point but not at the queried date), a
**cross-account intrusion** (a value belonging to a different account in the
interleaved stream), and an abstention. Stale intrusion is the error a
newest-wins revision policy is predicted to make, so it is reported per arm
rather than folded into a single accuracy number.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from experiments.answer_eval import exact_match
from neurosym.application.source_provenance import ensure_output_directory, paper_evidence_roots
from scripts.paper_artifacts import json_load, safe_path


ANALYSIS_VERSION = "persona_graphiti_analysis.v1"
_UNKNOWN = "unknown"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read one JSONL artifact."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def account_surface_values(events: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """Map each account to every surface preference value it ever asserts."""
    values: dict[str, set[str]] = {}
    for event in events:
        history_id = str(event.get("history_id", ""))
        fact = event.get("fact")
        if not history_id or not isinstance(fact, Mapping):
            continue
        surface = str(event.get("surface_object", "")).strip()
        if surface:
            values.setdefault(history_id, set()).add(surface)
    return values


def classify_answer(
    row: Mapping[str, Any], values_by_account: Mapping[str, set[str]]
) -> str:
    """Label one prediction as correct, stale, cross-account, abstained, or other.

    Stale intrusion is scoped to the queried account: the answer names a value
    this account genuinely held at some point, just not at the queried date.
    That is the newest-wins failure signature, and keeping it separate from a
    cross-account leak is what makes the per-arm comparison interpretable.
    """
    # Prefer the scored short answer the metrics used; the raw answer can
    # carry an "Answer:" prefix that the scorer strips, which would otherwise
    # make a correct row look like an intrusion.
    answer = str(row.get("short_answer") or row.get("answer", ""))
    gold = str(row.get("gold", ""))
    if exact_match(answer, gold):
        return "correct"
    if exact_match(answer, _UNKNOWN):
        return "abstained"
    history_id = str(row.get("history_id", ""))
    own = values_by_account.get(history_id, set())
    # Scoped to the account, but still a coarse label: it does not check the
    # predicate, scope qualifier, or validity interval of the named value, so
    # it counts "a value this account held that is not the gold" rather than a
    # proven temporal stale-read. Report it as such.
    if any(exact_match(answer, value) for value in own if value != gold):
        return "stale_intrusion"
    foreign = {
        value
        for account, account_values in values_by_account.items()
        if account != history_id
        for value in account_values
    }
    if any(exact_match(answer, value) for value in foreign):
        return "cross_account_intrusion"
    return "other"


def error_taxonomy(
    predictions: Sequence[Mapping[str, Any]],
    values_by_account: Mapping[str, set[str]],
) -> dict[str, dict[str, Any]]:
    """Count the error taxonomy per arm, and per arm and query family."""
    by_arm: dict[str, dict[str, Any]] = {}
    for row in predictions:
        arm = str(row["arm"])
        family = str(row.get("query_family", ""))
        label = classify_answer(row, values_by_account)
        arm_row = by_arm.setdefault(
            arm, {"total": 0, "labels": {}, "by_query_family": {}}
        )
        arm_row["total"] += 1
        arm_row["labels"][label] = arm_row["labels"].get(label, 0) + 1
        family_row = arm_row["by_query_family"].setdefault(family, {})
        family_row[label] = family_row.get(label, 0) + 1
    for arm_row in by_arm.values():
        total = arm_row["total"]
        arm_row["rates"] = {
            label: round(count / total, 4) for label, count in sorted(arm_row["labels"].items())
        }
    return by_arm


def constant_rule_baseline(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score memoryless rules that ignore the conversation entirely.

    If the gold values are concentrated, a rule that maps the query family to
    its most frequent answer can score well without consulting any memory. That
    number is the floor any memory arm must clear to have demonstrated
    anything, so it belongs beside the headline rather than in a footnote.
    """
    conditions: dict[str, Mapping[str, Any]] = {}
    for row in predictions:
        conditions.setdefault(str(row["evaluation_input_id"]), row)
    rows = list(conditions.values())
    golds = [str(row.get("gold", "")) for row in rows]
    distinct = sorted(set(golds))
    by_family: dict[str, list[str]] = {}
    for row in rows:
        by_family.setdefault(str(row.get("query_family", "")), []).append(
            str(row.get("gold", ""))
        )
    best_constant = max(
        (golds.count(value) for value in distinct), default=0
    )
    per_family_total = 0
    per_family_choice = {}
    for family, values in sorted(by_family.items()):
        best = max((values.count(v) for v in set(values)), default=0)
        pick = max(set(values), key=values.count) if values else ""
        per_family_total += best
        per_family_choice[family] = {"answer": pick, "correct": best, "n": len(values)}
    total = len(rows) or 1
    return {
        "condition_count": len(rows),
        "distinct_gold_values": distinct,
        "distinct_gold_count": len(distinct),
        "best_single_constant_em": round(best_constant / total, 4),
        "per_family_constant_em": round(per_family_total / total, 4),
        "per_family_choice": per_family_choice,
    }


def evidence_availability(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report how often the gold text actually reached the model's prompt.

    This replaces an earlier store-based "extraction ceiling" that was wrong in
    two ways: it searched only edges with no invalid_at, while point-in-time
    retrieval deliberately returns bounded edges whose window covers the query
    date, and it counted abstention golds (UNKNOWN) as facts that extraction
    had failed to capture. Measuring the rendered evidence instead is both
    correct and directly observable: ``retrieved_gold_occurrence_count`` counts
    the gold string in the retrieved rows that were fitted into the prompt.

    Abstention conditions are excluded, since no retrievable fact can carry
    "UNKNOWN" and including them understates availability.
    """
    report: dict[str, Any] = {}
    for row in predictions:
        arm = str(row["arm"])
        if str(row.get("arm_kind")) not in {"graphiti_memory", "hybrid_kg_memory"}:
            continue
        if str(row.get("gold", "")).strip().casefold() == _UNKNOWN:
            continue
        bucket = report.setdefault(
            arm,
            {"scored_rows": 0, "gold_in_prompt": 0, "correct_given_gold_in_prompt": 0,
             "correct_without_gold_in_prompt": 0},
        )
        bucket["scored_rows"] += 1
        shown = int(row.get("retrieved_gold_occurrence_count", 0)) > 0
        correct = float(row.get("exact_match", 0.0)) >= 1.0
        if shown:
            bucket["gold_in_prompt"] += 1
            bucket["correct_given_gold_in_prompt"] += correct
        elif correct:
            bucket["correct_without_gold_in_prompt"] += 1
    for bucket in report.values():
        n = bucket["scored_rows"] or 1
        shown = bucket["gold_in_prompt"] or 1
        bucket["retrieval_coverage"] = round(bucket["gold_in_prompt"] / n, 4)
        bucket["accuracy_given_gold_in_prompt"] = round(
            bucket["correct_given_gold_in_prompt"] / shown, 4
        )
    return report


def _percent(value: Any) -> str:
    """Render one proportion as a fixed-width percentage."""
    return f"{float(value) * 100:.2f}%"


def _points(value: Any) -> str:
    """Render one signed delta in percentage points."""
    return f"{float(value) * 100:+.2f}"


def _interval(row: Mapping[str, Any], field: str) -> str:
    """Render one bootstrap interval in percentage points."""
    interval = row.get(field) or []
    if len(interval) != 2:
        return "n/a"
    return f"[{float(interval[0]) * 100:.2f}, {float(interval[1]) * 100:.2f}]"


def _rows_for(aggregates: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    """Return one aggregate section, which the harness emits as a list of rows."""
    section = aggregates.get(name)
    if section is None:
        return []
    if not isinstance(section, list):
        raise ValueError(f"aggregate section {name} must be a list of rows")
    return [row for row in section if isinstance(row, Mapping)]


def _validity_block(diagnostics: Mapping[str, Any]) -> list[str]:
    """Render the memoryless floor first, as a blocking warning when it bites.

    This is generated rather than written by hand so that regenerating the
    analysis cannot quietly drop it.
    """
    baseline = diagnostics.get("constant_rule_baseline") or {}
    if not baseline:
        return []
    floor = float(baseline.get("per_family_constant_em", 0.0))
    lines = []
    if floor >= 0.5:
        lines += [
            "> **DO NOT PUBLISH THESE SCORES AS A MEMORY COMPARISON.** On this "
            f"corpus a memoryless rule that answers each query family with its "
            f"most frequent gold scores **{floor * 100:.2f}% exact match**, from "
            f"only {baseline.get('distinct_gold_count', '?')} distinct gold "
            "values across the conditions. Any arm at or below that floor has "
            "not demonstrated memory use. This is a property of the corpus and "
            "applies to every arm equally.",
            "",
        ]
    lines += [
        "## Memoryless floor",
        "",
        f"- distinct gold values: {baseline.get('distinct_gold_count', '?')} "
        f"({', '.join(repr(v) for v in baseline.get('distinct_gold_values', []))})",
        f"- best single constant answer: "
        f"{float(baseline.get('best_single_constant_em', 0.0)) * 100:.2f}% EM",
        f"- best per-query-family constant answer: {floor * 100:.2f}% EM",
        "",
        "Read every arm below against that floor, not against zero.",
        "",
    ]
    return lines


def _arm_table(aggregates: Mapping[str, Any]) -> list[str]:
    """Render the headline per-arm exact-match and F1 table."""
    lines = ["| Arm | Exact match | F1 | n | histories |", "|---|---:|---:|---:|---:|"]
    for row in sorted(_rows_for(aggregates, "by_arm"), key=lambda r: str(r["arm"])):
        lines.append(
            f"| `{row['arm']}` | {_percent(row['exact_match'])} | "
            f"{_percent(row['f1'])} | {row.get('row_count', '')} | "
            f"{row.get('history_cluster_count', '')} |"
        )
    return lines


def render_readme(
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None = None,
) -> str:
    """Render the headline artifact README in the persona benchmark house style."""
    aggregates = metrics["aggregates"]
    graphiti = manifest.get("graphiti_memory", {})
    model = manifest.get("model", {})
    arm_rows = _rows_for(aggregates, "by_arm")
    lines = [
        "# Qwen Persona Graphiti External Baseline",
        "",
        f"This run completed {metrics['generation_count']} deterministic generations "
        f"across {metrics['condition_count']} authenticated conditions and "
        f"{len(arm_rows)} matched arms.",
        "",
    ]
    lines += _validity_block(diagnostics or {})
    lines += _arm_table(aggregates)
    lines += ["", "## Matched effect sizes", ""]
    for row in _rows_for(aggregates, "paired_deltas"):
        lines.append(
            f"- `{row['left_arm']}` minus `{row['right_arm']}`: "
            f"{_points(row['exact_match_delta'])} exact-match points, "
            f"history-clustered 95% bootstrap interval "
            f"{_interval(row, 'exact_match_delta_ci_95')} over "
            f"{row.get('history_cluster_count', '?')} history clusters."
        )
    lines += [
        "",
        "## Error taxonomy",
        "",
        "Stale intrusion means the answer names a value the queried account "
        "genuinely held at some point but not at the queried date. It is the "
        "failure a newest-wins revision policy is predicted to make. The label "
        "is coarse: it does not verify predicate, scope, or validity interval, "
        "so treat it as an upper bound on temporal stale-reads rather than a "
        "proven count.",
        "",
        "| Arm | Correct | Stale intrusion | Cross-account | Abstained | Other |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in sorted(taxonomy):
        rates = taxonomy[arm]["rates"]
        lines.append(
            f"| `{arm}` | {_percent(rates.get('correct', 0))} | "
            f"{_percent(rates.get('stale_intrusion', 0))} | "
            f"{_percent(rates.get('cross_account_intrusion', 0))} | "
            f"{_percent(rates.get('abstained', 0))} | "
            f"{_percent(rates.get('other', 0))} |"
        )
    lines += [
        "",
        "## Mechanism",
        "",
        f"The Graphiti arms used `graphiti-core "
        f"{graphiti.get('graphiti_core_version', 'unknown')}` in "
        f"`{graphiti.get('ingestion_mode', 'unknown')}` mode with episode variant "
        f"`{graphiti.get('episode_variant', 'unknown')}` and retrieval state "
        f"`{graphiti.get('retrieval_state', 'unknown')}`, ingesting "
        f"{graphiti.get('episode_count', 0)} episodes across "
        f"{graphiti.get('history_count', 0)} accounts with "
        f"{graphiti.get('episode_failure_count', 0)} episode failures.",
        "",
        f"Extraction used `{graphiti.get('llm_model', 'unknown')}` and embeddings used "
        f"`{graphiti.get('embedding_model_id', 'unknown')}` revision "
        f"`{graphiti.get('embedding_revision', 'unknown')}`. The synthetic ingestion "
        f"clock was {json.dumps(graphiti.get('timestamp_mapping', {}), sort_keys=True)}.",
        "",
        "This arm is Graphiti ingestion with benchmark-matched retrieval, not "
        "off-the-shelf Graphiti. The deviations from stock defaults are recorded in "
        "`manifest.graphiti_memory.deviations_from_default`.",
        "",
        f"The model was `{model.get('model_id', 'unknown')}` revision "
        f"`{model.get('resolved_revision', 'unknown')}` under Transformers "
        f"{model.get('transformers_version', 'unknown')} and Torch "
        f"{model.get('torch_version', 'unknown')}. Generation was greedy, batch size "
        "one, on the configured device.",
        "",
        "## What this artifact does not establish",
        "",
        "- State fidelity is not reported. `graphiti_store_states.json` exports the "
        "valid-fact sets, but the alignment from Graphiti's extracted vocabulary to "
        "the corpus's canonical fact space is unimplemented, so only downstream "
        "generation is measured here.",
        "- Only the two delayed query families are probed directly. Scope exceptions, "
        "lineage retraction, and backdated correction are present in the stream as "
        "interference, so their effects are observable but not attributable.",
        "- No audit cost is measured: no latency, storage, or ledger-overhead "
        "comparison is made between arms.",
        "- The manifest hashes the configured Neo4j endpoint and database but does not "
        "authenticate a Neo4j server-instance identifier or server version.",
        "",
    ]
    return "\n".join(lines)


def render_analysis(
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    taxonomy: Mapping[str, Any],
) -> str:
    """Render the per-family, per-phase, and per-distance breakdown."""
    aggregates = metrics["aggregates"]
    lines = [
        "# Graphiti External Baseline Analysis",
        "",
        f"Analysis version `{ANALYSIS_VERSION}`.",
        "",
        "## Pre-registered predictions",
        "",
        "Graphiti was predicted to do well on plain preference change and plausibly "
        "on backdated correction, and to fail attributably on stale re-assertion "
        "(newest-wins invalidates the current correct fact), scope exceptions, and "
        "lineage retraction, while getting authority conflicts right by coincidence. "
        "The per-family and per-phase tables below are where those predictions are "
        "checked.",
        "",
        "## Exact match by arm and query family",
        "",
        "| Arm | Query family | Exact match | F1 | n |",
        "|---|---|---:|---:|---:|",
    ]
    for row in _rows_for(aggregates, "by_arm_query_family"):
        lines.append(
            f"| `{row['arm']}` | {row.get('query_family', '')} | "
            f"{_percent(row['exact_match'])} | {_percent(row['f1'])} | "
            f"{row.get('row_count', '')} |"
        )
    lines += [
        "",
        "## Exact match by arm and phase or token distance",
        "",
        "| Arm | Condition | Exact match | F1 | n |",
        "|---|---|---:|---:|---:|",
    ]
    for row in _rows_for(aggregates, "by_arm_condition"):
        lines.append(
            f"| `{row['arm']}` | {row.get('condition', '')} | "
            f"{_percent(row['exact_match'])} | {_percent(row['f1'])} | "
            f"{row.get('row_count', '')} |"
        )
    stratified = _rows_for(aggregates, "paired_deltas_by_query_family")
    if stratified:
        lines += [
            "",
            "## Matched effect sizes within each query family",
            "",
            "| Query family | Comparison | Delta (points) | 95% CI | histories |",
            "|---|---|---:|---|---:|",
        ]
        for row in stratified:
            lines.append(
                f"| {row.get('query_family', '')} | "
                f"`{row['left_arm']}` - `{row['right_arm']}` | "
                f"{_points(row['exact_match_delta'])} | "
                f"{_interval(row, 'exact_match_delta_ci_95')} | "
                f"{row.get('history_cluster_count', '')} |"
            )
    lines += ["", "## Error taxonomy by arm and query family", ""]
    labels = (
        "correct",
        "stale_intrusion",
        "cross_account_intrusion",
        "abstained",
        "other",
    )
    for arm in sorted(taxonomy):
        lines.append(f"### `{arm}`")
        lines.append("")
        lines.append("| Query family | " + " | ".join(labels) + " |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for family in sorted(taxonomy[arm]["by_query_family"]):
            counts = taxonomy[arm]["by_query_family"][family]
            lines.append(
                f"| {family} | "
                + " | ".join(str(counts.get(label, 0)) for label in labels)
                + " |"
            )
        lines.append("")
    graphiti = manifest.get("graphiti_memory", {})
    lines += [
        "## Ingestion provenance",
        "",
        f"- graphiti-core version: `{graphiti.get('graphiti_core_version', 'unknown')}`",
        f"- ingestion mode: `{graphiti.get('ingestion_mode', 'unknown')}`",
        f"- episode variant: `{graphiti.get('episode_variant', 'unknown')}`",
        f"- retrieval state: `{graphiti.get('retrieval_state', 'unknown')}`",
        f"- episodes ingested: {graphiti.get('episode_count', 0)}",
        f"- episode failures: {graphiti.get('episode_failure_count', 0)}",
        f"- valid facts committed: {graphiti.get('valid_fact_total', 'unknown')}",
        f"- timestamp mapping: {json.dumps(graphiti.get('timestamp_mapping', {}), sort_keys=True)}",
        f"- store state hash: `{graphiti.get('store_states_sha256', 'unknown')}`",
        "",
    ]
    return "\n".join(lines)


def _load_store_states(run_dir: Path) -> dict[str, Any]:
    """Load the per-checkpoint store snapshot the build queried, if present."""
    path = run_dir / "graphiti_store_states.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _verify_hash(root: Path, name: str, expected: Any) -> None:
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"invalid SHA-256 for {name}")
    path = safe_path(root, name)
    try:
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular artifact: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
    except OSError as error:
        raise ValueError(f"cannot verify artifact {path}: {error}") from error
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size, after.st_mtime_ns, after.st_ino
    ):
        raise ValueError(f"artifact changed while verifying: {path}")


def _verify_hash_map(root: Path, hashes: Any, required: Sequence[str] = ()) -> dict[str, str]:
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"artifact_sha256 must be a nonempty hash mapping: {root}")
    if not set(required).issubset(hashes):
        raise ValueError(f"artifact_sha256 missing required artifacts: {required}")
    for name, expected in hashes.items():
        _verify_hash(root, name, expected)
    return hashes


def _completed_manifest(root: Path, name: str) -> dict[str, Any]:
    manifest = json_load(safe_path(root, name))
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise ValueError(f"requires a completed manifest: {root / name}")
    return manifest


def _verify_declared_artifacts(root: Path, manifest: Mapping[str, Any], required: Sequence[str] = ()) -> dict[str, str]:
    hashes = _verify_hash_map(root, manifest.get("artifact_sha256"), required)
    if "artifacts" in manifest:
        artifacts = manifest["artifacts"]
        if not isinstance(artifacts, list) or any(
            not isinstance(name, str) or name not in hashes for name in artifacts
        ):
            raise ValueError("declared artifacts require artifact_sha256 entries")
    return hashes


def _verify_analysis_inputs(run_dir: Path, corpus_dir: Path) -> dict[str, Any]:
    manifest = _completed_manifest(run_dir, "manifest.json")
    run_hashes = _verify_declared_artifacts(
        run_dir, manifest, ("metrics.json", "predictions.jsonl")
    )
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("run manifest requires a primary dataset mapping")
    _verify_hash(corpus_dir, "generation_manifest.json", dataset.get("generation_manifest_sha256"))
    corpus_manifest = _completed_manifest(corpus_dir, "generation_manifest.json")
    corpus_hashes = _verify_declared_artifacts(corpus_dir, corpus_manifest, ("events.jsonl",))
    dataset_hashes = _verify_hash_map(corpus_dir, dataset.get("artifact_sha256"), ("events.jsonl",))
    for name, expected in dataset_hashes.items():
        if corpus_hashes.get(name) != expected:
            raise ValueError(f"primary dataset hash mapping differs from corpus manifest: {name}")
    for owner, key in ((dataset, "sha256"), (manifest, "dataset_sha256")):
        if key in owner:
            for name, expected in _verify_hash_map(corpus_dir, owner[key]).items():
                if corpus_hashes.get(name) != expected:
                    raise ValueError(f"dataset hash mapping differs from corpus manifest: {name}")
    if "generation_manifest_sha256" in manifest or "generation_manifest.json" in run_hashes:
        if "generation_manifest_sha256" in manifest:
            _verify_hash(run_dir, "generation_manifest.json", manifest["generation_manifest_sha256"])
        generation_manifest = _completed_manifest(run_dir, "generation_manifest.json")
        _verify_declared_artifacts(run_dir, generation_manifest)
        if "dataset" in generation_manifest and generation_manifest["dataset"] != dataset:
            raise ValueError("generation manifest primary dataset differs from run manifest")
    return manifest


def analyze_run(
    run_dir: Path, corpus_dir: Path, *, output_dir: Path | None = None,
    verify_inputs: bool = False,
) -> dict[str, Any]:
    """Render README and analysis documents for one completed run directory."""
    run_dir, corpus_dir = Path(run_dir), Path(corpus_dir)
    destination = ensure_output_directory(
        run_dir if output_dir is None else output_dir,
        input_dirs=(corpus_dir,) if output_dir is None else (run_dir, corpus_dir),
        frozen_roots=paper_evidence_roots(Path(__file__).resolve().parents[1]),
        allow_resume=output_dir is None,
    )
    names = ("diagnostics.json", "README.md", "analysis.md", "error_taxonomy.json")
    for name in names:
        path = safe_path(destination, name)
        if output_dir is not None and path.exists():
            raise ValueError(f"refusing to overwrite analysis artifact: {path}")
        if path.exists() and path.stat().st_nlink > 1:
            raise ValueError(f"refusing hard-linked analysis artifact: {path}")
    manifest = (
        _verify_analysis_inputs(run_dir, corpus_dir) if verify_inputs
        else json_load(run_dir / "manifest.json")
    )
    metrics = json_load(run_dir / "metrics.json")
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    events = _read_jsonl(corpus_dir / "events.jsonl")
    values_by_account = account_surface_values(events)
    taxonomy = error_taxonomy(predictions, values_by_account)
    diagnostics = {
        "constant_rule_baseline": constant_rule_baseline(predictions),
        "evidence_availability": evidence_availability(predictions),
    }
    documents = {
        "diagnostics.json": json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        "README.md": render_readme(metrics, manifest, taxonomy, diagnostics),
        "analysis.md": render_analysis(metrics, manifest, taxonomy),
        "error_taxonomy.json": json.dumps(taxonomy, indent=2, sort_keys=True) + "\n",
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in documents.items():
        with (destination / name).open("w" if output_dir is None else "x", encoding="utf-8") as stream:
            stream.write(content)
    return taxonomy


def main(argv: list[str] | None = None) -> int:
    """Render analysis artifacts for one Graphiti benchmark run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        help="Separate analysis destination; never overlaps inputs or overwrites analysis files. "
             "Omit only for legacy nonfrozen in-place rendering.",
    )
    parser.add_argument(
        "--verify-inputs", action="store_true",
        help="Required for paper workflows: verify completed manifests, every declared artifact "
             "SHA-256, and the primary dataset manifest pin and events before writing. "
             "Explicit opt-in, including with --output-dir; no metric rescoring or freeze-schema validation.",
    )
    args = parser.parse_args(argv)
    taxonomy = analyze_run(
        args.run_dir, args.corpus_dir, output_dir=args.output_dir,
        verify_inputs=args.verify_inputs,
    )
    print(json.dumps(taxonomy, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
