from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import contextlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

TOLERANCE = 1e-12
DELAYED_KINDS = {"preference-change-delayed", "preference-incongruity-delayed"}


class VerificationError(ValueError):
    pass


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise VerificationError(f"non-finite JSON number: {value}")


def _decode(text: str, location: str) -> dict[str, Any]:
    try:
        value = json.loads(text, object_pairs_hook=_object, parse_constant=_invalid_constant)
    except ValueError as exc:
        raise VerificationError(f"{location}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{location}: expected JSON object")
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return _decode(handle.read(), str(path))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            yield _decode(line, f"{path}:{number}")


def index_unique(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> dict:
    result = {}
    for number, row in enumerate(rows, 1):
        try:
            key = tuple(row[field] for field in fields)
            if any(not isinstance(value, str) or not value for value in key):
                raise VerificationError(f"row {number}: nonempty string key required for {fields}")
            if key in result:
                raise VerificationError(f"duplicate key {key}")
            result[key] = row
        except KeyError as exc:
            raise VerificationError(f"row {number}: missing key {exc}") from exc
    return result


def compare_values(actual: Any, recorded: Any, location: str = "value") -> None:
    if isinstance(actual, dict) and isinstance(recorded, dict):
        if actual.keys() != recorded.keys():
            raise VerificationError(f"{location}: object keys differ: {sorted(actual)} vs {sorted(recorded)}")
        for key in actual:
            compare_values(actual[key], recorded[key], f"{location}.{key}")
    elif isinstance(actual, (list, tuple)) and isinstance(recorded, (list, tuple)):
        if len(actual) != len(recorded):
            raise VerificationError(f"{location}: lengths differ: {len(actual)} vs {len(recorded)}")
        for index, (left, right) in enumerate(zip(actual, recorded)):
            compare_values(left, right, f"{location}[{index}]")
    elif isinstance(actual, bool) or isinstance(recorded, bool):
        if type(actual) is not type(recorded) or actual != recorded:
            raise VerificationError(f"{location}: {actual!r} != {recorded!r}")
    elif isinstance(actual, (int, float)) and isinstance(recorded, (int, float)):
        if not math.isfinite(actual) or not math.isfinite(recorded) or abs(actual - recorded) > TOLERANCE:
            raise VerificationError(f"{location}: {actual!r} != {recorded!r} (absolute tolerance {TOLERANCE})")
    elif type(actual) is not type(recorded) or actual != recorded:
        raise VerificationError(f"{location}: {actual!r} != {recorded!r}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def rescore_persona(generations: Iterable[Mapping[str, Any]], predictions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from experiments.persona_end_to_end_benchmark import _score_short_answer

    raw = index_unique(generations, ("evaluation_input_id", "arm"))
    scored = index_unique(predictions, ("evaluation_input_id", "arm"))
    _require(raw.keys() == scored.keys(), "persona: generation/prediction keys differ")
    result = []
    for key, generation in raw.items():
        prediction = scored[key]
        for field, value in generation.items():
            compare_values(value, prediction[field], f"persona.{key}.{field}")
        _require(isinstance(generation["answer"], str) and isinstance(generation["gold"], str), f"persona.{key}: answer/gold must be strings")
        scores = _score_short_answer(generation["answer"], generation["gold"])
        for field, value in scores.items():
            compare_values(value, prediction[field], f"persona.{key}.{field}")
        result.append({**generation, **scores})
    return result


def verify_persona(root: Path) -> dict[str, Any]:
    from experiments.persona_end_to_end_benchmark import _group_metrics, _paired_delta

    directory = root / "results/persona_joint_surface_a_build2"
    manifest = read_json(directory / "manifest.json")
    metrics = read_json(directory / "metrics.json")
    rows = rescore_persona(iter_jsonl(directory / "generations.jsonl"), iter_jsonl(directory / "predictions.jsonl"))
    arms = {row["arm"] for row in rows}
    conditions = {row["evaluation_input_id"] for row in rows}
    histories = {row["history_id"] for row in rows}
    counts = {"conditions": len(conditions), "histories": len(histories), "arms": len(arms), "rows": len(rows)}
    compare_values(counts, {"conditions": 120, "histories": 12, "arms": 8, "rows": 960}, "persona.counts")
    compare_values(sorted(arms), sorted(arm["name"] for arm in manifest["arms"]), "persona.manifest.arms")
    for recorded in (manifest, metrics):
        compare_values(len(conditions), recorded["condition_count"], "persona.condition_count")
        compare_values(len(rows), recorded["generation_count"], "persona.generation_count")
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["evaluation_input_id"]].append(row)
    for key, group in by_condition.items():
        _require({row["arm"] for row in group} == arms, f"persona.{key}: incomplete arms")
        for field in ("history_id", "condition", "query_id", "query_family", "gold"):
            _require(len({row[field] for row in group}) == 1, f"persona.{key}: inconsistent {field}")
    history_conditions = Counter(group[0]["history_id"] for group in by_condition.values())
    _require(all(count == 10 for count in history_conditions.values()), "persona: expected ten conditions per history")
    aggregates = metrics["aggregates"]
    by_arm = _group_metrics(rows, ("arm",))
    compare_values(by_arm, aggregates["by_arm"], "persona.by_arm")
    paired = []
    seen = set()
    _require(bool(aggregates["paired_deltas"]), "persona: no paired comparisons")
    for stored in aggregates["paired_deltas"]:
        key = (stored["left_arm"], stored["right_arm"])
        _require(key not in seen, f"persona: duplicate comparison {key}")
        seen.add(key)
        _require(isinstance(stored["bootstrap_samples"], int) and stored["bootstrap_samples"] > 0, "persona: invalid bootstrap samples")
        actual = _paired_delta(rows, *key, bootstrap_samples=stored["bootstrap_samples"], seed=stored["bootstrap_seed"])
        compare_values(actual, stored, f"persona.paired_deltas.{key}")
        paired.append(actual)
    return {"counts": counts, "score_values_checked": 16, "by_arm": by_arm, "paired_deltas": paired,
            "gaps": ["Full stratified bootstrap comparisons intentionally not recomputed.", "Saved answers are rescored, not regenerated; model, Graphiti, and ingestion services were not run."]}


def _evidence_scores(row: Mapping[str, Any], event_groups: Sequence, fact_groups: Sequence) -> tuple[bool, float]:
    _require(bool(event_groups) and bool(fact_groups), "empty evidence contract")
    events, facts = set(row["selected_event_ids"]), set(row["selected_fact_ids"])
    hits = [bool(events & set(group)) for group in event_groups] + [bool(facts & set(group)) for group in fact_groups]
    return all(hits), sum(hits) / len(hits)


def _check_flags(row: Mapping[str, Any], evidence_field: str, location: str) -> None:
    for field in ("answer_correct", "grounded_answer_correct", evidence_field):
        _require(type(row[field]) is bool, f"{location}.{field}: expected boolean")
    correct = row["prediction"] == row["gold"]
    compare_values(correct, row["answer_correct"], f"{location}.answer_correct")
    compare_values(correct and row[evidence_field], row["grounded_answer_correct"], f"{location}.grounded_answer_correct")


def verify_continual(root: Path) -> dict[str, Any]:
    directory = root / "results/continual_memory_benchmark_v3_stream"
    metrics = read_json(directory / "metrics.json")
    manifest = read_json(directory / "manifest.json")
    windows = [window["name"] for window in manifest["config"]["sliding_token_windows"]]
    compare_values(sorted(windows), sorted(["window_4k", "window_16k", "window_64k", "window_128k"]), "continual.windows")
    target_methods = {f"sliding_context:{window}{suffix}" for window in windows for suffix in ("", "+scallop_injection")}
    selected = []
    lineage = []
    seen = set()
    families = Counter()
    for row in iter_jsonl(directory / "predictions.jsonl"):
        key = (row["evaluation_family"], row["episode_id"], row["checkpoint_id"], row["method"])
        _require(key not in seen, f"continual: duplicate key {key}")
        seen.add(key)
        families[row["evaluation_family"]] += 1
        if row["evaluation_family"] == "scallop_reasoning_ablation":
            correct = row["prediction"] == row["gold"]
            compare_values(correct, row["answer_correct"], f"continual.{key}.answer_correct")
            lineage.append({**row, "answer_correct": correct})
        elif row["evaluation_family"] == "retrieval" and row["interference_tier"] == "deep_context_rot" and row["checkpoint_kind"] in DELAYED_KINDS and row["method"] in target_methods:
            _check_flags(row, "exact_evidence_hit", f"continual.{key}")
            selected.append(row)
    methods = {}
    for method in sorted({row["method"] for row in lineage}):
        group = [row for row in lineage if row["method"] == method]
        kinds = {}
        for kind in sorted({row["checkpoint_kind"] for row in group}):
            members = [row for row in group if row["checkpoint_kind"] == kind]
            kinds[kind] = {"count": len(members), "accuracy": sum(row["answer_correct"] for row in members) / len(members)}
        methods[method] = {"count": len(group), "accuracy": sum(row["answer_correct"] for row in group) / len(group), "by_checkpoint_kind": kinds}
    reasoning = metrics["scallop_reasoning_ablation"]
    compare_values(methods, reasoning["methods"], "continual.lineage.methods")
    compare_values(methods["scallop_recursive"]["by_checkpoint_kind"]["private-lineage"]["accuracy"], reasoning["recursive_accuracy"], "continual.recursive_accuracy")
    compare_values(methods["scallop_one_hop"]["by_checkpoint_kind"]["private-lineage"]["accuracy"], reasoning["one_hop_accuracy"], "continual.one_hop_accuracy")
    lineage_pairs = defaultdict(set)
    for row in lineage:
        lineage_pairs[row["checkpoint_id"]].add(row["method"])
    _require(all(value == set(methods) for value in lineage_pairs.values()), "continual: unmatched lineage methods")
    checkpoints = {}
    episode_count = checkpoint_count = max_tokens = 0
    for episode in iter_jsonl(directory / "episodes.jsonl"):
        episode_count += 1
        for checkpoint in episode["checkpoints"]:
            checkpoint_count += 1
            max_tokens = max(max_tokens, checkpoint["stream_token_count"])
            key = (episode["episode_id"], checkpoint["checkpoint_id"])
            _require(key not in checkpoints, f"continual: duplicate checkpoint {key}")
            checkpoints[key] = {field: checkpoint[field] for field in ("gold", "checkpoint_kind", "valid_alternative_event_id_groups", "valid_alternative_fact_id_groups")}
    compare_values(max_tokens, 408761, "continual.maximum_checkpoint_tokens.paper")
    compare_values(max_tokens, metrics["memory_growth"]["stored_tokens_max"], "continual.maximum_checkpoint_tokens.recorded")
    compare_values(episode_count, metrics["dataset"]["episode_count"], "continual.episode_count")
    compare_values(checkpoint_count, metrics["dataset"]["checkpoint_count"], "continual.checkpoint_count")
    for row in [*lineage, *selected]:
        checkpoint = checkpoints[(row["episode_id"], row["checkpoint_id"])]
        compare_values(row["gold"], checkpoint["gold"], "continual.checkpoint.gold")
        compare_values(row["checkpoint_kind"], checkpoint["checkpoint_kind"], "continual.checkpoint.kind")
    for row in selected:
        checkpoint = checkpoints[(row["episode_id"], row["checkpoint_id"])]
        complete, recall = _evidence_scores(row, checkpoint["valid_alternative_event_id_groups"], checkpoint["valid_alternative_fact_id_groups"])
        compare_values(complete, row["exact_evidence_hit"], "continual.saved_evidence.complete")
        compare_values(recall, row["evidence_recall"], "continual.saved_evidence.recall")
    slices = {}
    for window in windows:
        slices[window] = {}
        for kind in sorted(DELAYED_KINDS):
            raw_method = f"sliding_context:{window}"
            raw = index_unique((row for row in selected if row["method"] == raw_method and row["checkpoint_kind"] == kind), ("episode_id", "checkpoint_id"))
            injected = index_unique((row for row in selected if row["method"] == raw_method + "+scallop_injection" and row["checkpoint_kind"] == kind), ("episode_id", "checkpoint_id"))
            _require(bool(raw) and raw.keys() == injected.keys(), f"continual.{window}.{kind}: unmatched pairs")
            actual = {"matched_count": len(raw), "no_injection_grounded_accuracy": sum(row["grounded_answer_correct"] for row in raw.values()) / len(raw),
                      "scallop_injected_grounded_accuracy": sum(row["grounded_answer_correct"] for row in injected.values()) / len(injected),
                      "paired_delta": sum(int(injected[key]["grounded_answer_correct"]) - int(raw[key]["grounded_answer_correct"]) for key in raw) / len(raw)}
            recorded = metrics["scallop_stream_injection_ablation"]["by_window"][window]["deep_context_rot"][kind]
            compare_values(actual, {key: recorded[key] for key in actual}, f"continual.{window}.{kind}")
            slices[window][kind] = actual
    return {"prediction_rows_by_family": dict(families), "lineage_methods": methods, "deep_context_rot": slices,
            "episode_count": episode_count, "checkpoint_count": checkpoint_count, "maximum_saved_checkpoint_tokens": max_tokens,
            "gaps": ["Verifies saved answers, grounded flags and selected evidence IDs; Scallop and the oracle resolver were not rerun.", "Token maximum uses saved checkpoint counts, not retokenization.", "Other retrieval tiers, methods, and injection controls are outside this check."]}


class _OrderOnlyEncoder:
    def encode(self, text: str) -> range:
        return range(len(text))


def eligible_interleaved_queries(queries: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    from experiments.interleaved_memory_benchmark_v3 import _checkpoint_kind

    return [query for query in queries if query.get("split") == "test" and query.get("hardness_profile") == "anti_shortcut_interleaved_v3" and _checkpoint_kind(query) is not None]


def _root_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    _require(path.is_relative_to(root.resolve()), f"evidence path escapes root: {relative}")
    return path


def verify_interleaved(root: Path) -> dict[str, Any]:
    from experiments.interleaved_conversation import build_interleaved_schedule
    from experiments.interleaved_memory_benchmark import _aggregate_online_predictions
    from experiments.interleaved_memory_benchmark_v3 import _age_summary, _aggregate_by_checkpoint_kind, _checkpoint_kind, _paired_grounded_comparison

    directory = root / "results/interleaved_memory_benchmark_v3"
    metrics = read_json(directory / "metrics.json")
    manifest = read_json(directory / "manifest.json")
    dataset = _root_path(root, manifest["dataset"])
    queries = eligible_interleaved_queries(iter_jsonl(dataset / "queries.jsonl"))
    query_index = index_unique(queries, ("query_id",))
    events = [event for event in iter_jsonl(dataset / "events.jsonl") if event.get("split") == "test" and event.get("hardness_profile") == "anti_shortcut_interleaved_v3"]
    index_unique(events, ("event_id",))
    compare_values(len({event["history_id"] for event in events}), 1024, "interleaved.history_count.paper")
    schedule = build_interleaved_schedule(events, tokenizer=_OrderOnlyEncoder(), seed=metrics["schedule"]["seed"], concurrent_accounts=metrics["schedule"]["concurrent_accounts"], min_segment_events=4, max_segment_events=8)
    positions = {turn["stream_event"]["event_id"]: index + 1 for index, turn in enumerate(schedule["turns"])}
    total_turns = len(positions)
    _require(metrics["sweep_interval_tokens"] > 0, "interleaved: nonpositive sweep interval")
    expected = set()
    for query in queries:
        contract = query.get("checkpoint_contract")
        if not isinstance(contract, dict):
            continue
        trigger = positions[contract["trigger_event_id"]]
        if trigger == total_turns:
            continue
        groups = contract["required_event_id_groups"]
        _require(bool(groups) and all(any(event in positions and positions[event] <= trigger for event in group) for group in groups), f"interleaved.{query['query_id']}: unavailable evidence contract")
        expected.add(query["query_id"])
    rows = list(iter_jsonl(directory / "predictions.jsonl"))
    index_unique(rows, ("query_id", "method"))
    windows = metrics["windows"]
    compare_values(windows, [65536, 131072], "interleaved.windows")
    expected_methods = {f"{family}:{window}" for family in ("sliding_context", "structured_capacity_top1") for window in windows}
    actual_queries = {row["query_id"] for row in rows}
    compare_values(sorted(actual_queries), sorted(expected), "interleaved.eligible_query_ids")
    compare_values(len(queries), 2048, "interleaved.query_count.paper")
    compare_values(len(actual_queries), 2047, "interleaved.evaluated_count.paper")
    by_query = defaultdict(set)
    for row in rows:
        query = query_index[(row["query_id"],)]
        by_query[row["query_id"]].add(row["method"])
        compare_values(row["checkpoint_kind"], _checkpoint_kind(query), "interleaved.checkpoint_kind")
        for field in ("history_id", "gold"):
            compare_values(row[field], query[field], f"interleaved.{row['query_id']}.{field}")
        compare_values(row["trigger_turn_index"], positions[query["checkpoint_contract"]["trigger_event_id"]], "interleaved.trigger_turn_index")
        _require(row["trigger_turn_index"] < row["checkpoint_turn_index"] <= total_turns, "interleaved: invalid delayed checkpoint")
        compare_values(row["method"].split(":")[1], str(row["max_tokens"]), "interleaved.method_window")
        contract = query["checkpoint_contract"]
        complete, recall = _evidence_scores(row, contract["required_event_id_groups"], contract["required_fact_id_groups"])
        compare_values(complete, row["complete_provenance"], "interleaved.saved_evidence.complete")
        compare_values(recall, row["evidence_recall"], "interleaved.saved_evidence.recall")
        _check_flags(row, "complete_provenance", f"interleaved.{row['query_id']}.{row['method']}")
        _require(type(row["stale_memory_intrusion"]) is bool, "interleaved: stale flag must be boolean")
    _require(all(methods == expected_methods for methods in by_query.values()), "interleaved: missing/unexpected methods")
    dataset_counts = {"history_count": len({event["history_id"] for event in events}), "event_count": len(events), "query_count": len(queries), "evaluated_checkpoint_count": len(actual_queries), "excluded_query_ids": sorted(set(query["query_id"] for query in queries) - actual_queries)}
    compare_values(dataset_counts, metrics["dataset"], "interleaved.dataset")
    compare_values(total_turns, metrics["schedule"]["turn_count"], "interleaved.schedule.turn_count")
    methods = _aggregate_online_predictions(rows)
    compare_values(methods, metrics["methods"], "interleaved.methods")
    compare_values(_aggregate_by_checkpoint_kind(rows), metrics["by_checkpoint_kind"], "interleaved.by_checkpoint_kind")
    compare_values(_age_summary(rows), metrics["age_distribution"], "interleaved.age_distribution")
    paired = {}
    for window, stored in metrics["paired_grounded_comparison"].items():
        members = [row for row in rows if str(row["max_tokens"]) == window]
        paired.update(_paired_grounded_comparison(members, bootstrap_samples=stored["bootstrap_samples"], seed=stored["seed"]))
    compare_values(sorted(paired), sorted(str(window) for window in windows), "interleaved.paired_windows")
    compare_values(paired, metrics["paired_grounded_comparison"], "interleaved.paired_grounded_comparison")
    return {"dataset": dataset_counts, "prediction_row_count": len(rows), "methods": methods, "paired_grounded_comparison": paired,
            "eligibility": {"split": "test", "hardness_profile": "anti_shortcut_interleaved_v3", "checkpoint_kinds": sorted(DELAYED_KINDS), "schedule_seed": metrics["schedule"]["seed"], "concurrent_accounts": metrics["schedule"]["concurrent_accounts"], "segment_event_bounds": [4, 8], "rule": "Supported query with contract, available evidence, and trigger before terminal turn; every such trigger has a later periodic or terminal sweep for any positive token interval."},
            "gaps": ["Schedule order alone is reconstructed with a character-count placeholder; its counts are discarded. No token timing, sweep token placement, or model tokenizer is reproduced.", "Saved selected evidence IDs and flags are checked; Scallop, oracle answers, capsule selection and LLM generation are not rerun.", "Stale intrusion flags are reaggregated, not independently recomputed."]}


def verify(root: Path) -> dict[str, Any]:
    report = {"provenance_mode": "saved_record_reaggregation", "root": str(root.resolve()), "aggregation_code_root": str(SOURCE_ROOT), "absolute_float_tolerance": TOLERANCE,
              "integrity": {"authenticated": False, "external_freeze_verification": "not supplied or checked", "statement": "No file integrity authentication is performed. Numerical agreement is not hash verification; file integrity requires separately passed external freeze verification. Existing raw-file hash mismatches (including possible CRLF differences) are not resolved by this analysis."},
              "bundles": {}, "mismatches": []}
    for name, verifier in (("persona_joint_surface_a_build2", verify_persona), ("continual_memory_benchmark_v3_stream", verify_continual), ("interleaved_memory_benchmark_v3", verify_interleaved)):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                details = verifier(root)
            report["bundles"][name] = {"status": "passed", **details}
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            report["bundles"][name] = {"status": "failed", "error": message}
            report["mismatches"].append({"bundle": name, "error": message})
    report["status"] = "passed" if not report["mismatches"] else "failed"
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only saved-record scientific reaggregation; JSON stdout, no integrity authentication or live models/services.")
    parser.add_argument("--root", required=True, type=Path, help="Repo-shaped frozen evidence root")
    parser.add_argument("--expected-manifest-sha256", help="Verify the freeze against an externally retained manifest digest first")
    parser.add_argument("--output", type=Path, help="Save a new report outside the evidence root; never overwrite")
    args = parser.parse_args(argv)
    if args.output is not None:
        from neurosym.application.source_provenance import ensure_output_directory, paper_evidence_roots
        from scripts.paper_artifacts import safe_path

        if args.output.resolve().is_relative_to(args.root.resolve()):
            raise ValueError("verification report must be outside the evidence root")
        ensure_output_directory(args.output.parent, frozen_roots=paper_evidence_roots(SOURCE_ROOT))
        safe_path(args.output.parent, args.output.name)
        if args.output.exists():
            raise ValueError("verification report already exists")
    integrity = None
    if args.expected_manifest_sha256:
        from scripts.paper_artifacts import verify as verify_snapshot

        integrity = verify_snapshot(args.root, args.expected_manifest_sha256, require_trusted=True)
        if integrity["status"] != "verified":
            print(json.dumps({"status": "failed", "integrity": integrity}, indent=2, sort_keys=True))
            return 1
    report = verify(args.root)
    if integrity is not None:
        report["integrity"] = {
            "authenticated": True,
            "external_freeze_verification": integrity["trust"],
            "manifest_sha256": integrity["manifest_sha256"],
            "statement": "Frozen bytes and declared historical hashes verified against the supplied digest; not proof of original execution or a new model run.",
        }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    print(payload, end="")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
