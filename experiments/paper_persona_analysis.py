from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from experiments.persona_end_to_end_benchmark import _group_metrics, _paired_delta, _score_short_answer
from experiments.persona_graphiti_analysis import _read_jsonl, _verify_analysis_inputs
from experiments.persona_interference_schedule import _authenticate_dataset
from experiments.persona_surface_derivation import authenticate_pair_gate
from neurosym.application.source_provenance import ensure_output_directory, paper_evidence_roots
from scripts.paper_artifacts import json_load, safe_path


MODES = ("historical_a", "kimi_a", "kimi_ab")
PAIR_GATE_SHA256 = "d03540a6575c9d2967d8bfade7fef4e38a0951d98cb702420c06113dd228fb5b"
PARENT_SHA256 = "49247f1c361319cba951b21362ffa4920b2633eb394c6aeee59765dada252fc3"
HISTORICAL_SHA256 = "c64c5c4268d93691b9bdf12119fa31a91cc7e9a798182135593c85128fc89c89"
MODERN_SHA256 = {
    "A": "2cfdbf18705204a9ca7412e9e962ad3bb03bd08b2426f09270abbc387b047f65",
    "B": "986cbe8ce306e19169626d5b8729b92e120f45831a76b328cfa1e32401dfd121",
}
CORPORA = {
    "historical_a": "results/persona_conflict_conversations_surface_a_graphiti_build2",
    "A": "results/persona_conflict_conversations_surface_a",
    "B": "results/persona_conflict_conversations_surface_b",
}
ARMS = [
    {"name": f"{kind}_{cap}", "kind": kind, "prompt_token_cap": cap}
    for kind in ("sliding_context", "structured_memory", "hybrid_kg_memory", "graphiti_memory")
    for cap in (4096, 16384)
]
ARM_NAMES = {arm["name"] for arm in ARMS}
GROUPS = {
    "by_arm": ("arm",),
    "by_arm_condition": ("arm", "condition"),
    "by_arm_query_family": ("arm", "query_family"),
    "by_arm_condition_query_family": ("arm", "condition", "query_family"),
}
GRAPHITI_RESULTS = {
    "build_id", "memory_store_sha256", "memory_store_combined_sha256",
    "retrieval_map_sha256", "store_states_sha256", "valid_fact_total",
}
HYBRID_RESULTS = {
    "session_id", "admission_ledgers_sha256", "condition_facts_sha256",
    "retrieval_map_sha256", "committed_fact_count", "condition_fact_count",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch")


def _require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} requires a SHA-256 identity")


def normalize_scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(config))
    for key in ("surface_assignment", "dataset_dir_env", "output_dir_env"):
        result.get("runtime", {}).pop(key, None)
    result.get("schedule", {}).pop("source_manifest_sha256", None)
    return result


def _authenticate_surface(
    evidence_root: Path, mode: str, assignment: str, pair_gate_sha256: str,
) -> tuple[Path, dict[str, Any], dict[str, bytes]]:
    corpus = safe_path(evidence_root, CORPORA["historical_a" if mode == "historical_a" else assignment])
    expected = HISTORICAL_SHA256 if mode == "historical_a" else MODERN_SHA256[assignment]
    if mode == "historical_a":
        provenance, artifacts, _ = _authenticate_dataset(corpus, expected)
    else:
        gate = safe_path(evidence_root, "results/persona_surface_pair_gate.json")
        bindings = json_load(gate).get("paths", {})
        for label, name in (("A", CORPORA["A"]), ("B", CORPORA["B"]),
                            ("parent", "results/persona_conflict_conversations_v1")):
            if label not in bindings:
                raise ValueError("pair gate path bindings are incomplete")
            _require_equal((gate.parent / bindings[label]).resolve(), safe_path(evidence_root, name),
                           f"evidence {label} path")
        authenticated = authenticate_pair_gate(
            gate, pair_gate_sha256, dataset_dir=corpus, expected_assignment=assignment,
        )
        _require_equal(authenticated["assignment_manifest_sha256"], MODERN_SHA256, "fixed modern corpus hashes")
        _require_equal(authenticated["parent_generation_manifest_sha256"], PARENT_SHA256, "pair parent")
        _require_equal(Path(authenticated["parent_path"]).resolve(),
                       safe_path(evidence_root, "results/persona_conflict_conversations_v1"), "evidence parent path")
        provenance, artifacts, _ = _authenticate_dataset(
            corpus, expected, gate, pair_gate_sha256, assignment,
        )
    _require_equal(provenance.get("parent_generation_manifest_sha256"), PARENT_SHA256, "dataset parent")
    _require_equal(provenance.get("generation_manifest_sha256"), expected, "dataset manifest")
    return corpus, provenance, artifacts


def _validate_config(manifest: Mapping[str, Any], config: Mapping[str, Any], config_path: Path,
                     provenance: Mapping[str, Any], mode: str, assignment: str) -> None:
    _require_equal(manifest.get("config_sha256"), _digest(config_path), "config_sha256")
    for owner, label in ((manifest, "manifest"), (config, "config")):
        arms = owner.get("arms")
        if not isinstance(arms, list) or sorted(arms, key=lambda arm: arm["name"]) != sorted(ARMS, key=lambda arm: arm["name"]):
            raise ValueError(f"{label} requires exactly the eight matched arms; full context is excluded")
    schedule = config.get("schedule", {})
    _require_equal(schedule.get("source_manifest_sha256"), provenance["generation_manifest_sha256"], "config source")
    _require_equal(schedule.get("query_suffixes"), ["-preference-change-delayed", "-preference-incongruity-delayed"], "query families")
    _require_equal(schedule.get("token_distance_thresholds"), [4096, 8192], "condition thresholds")
    if mode != "historical_a":
        _require_equal(config.get("runtime", {}).get("surface_assignment"), assignment, "config assignment")
    dataset = manifest["dataset"]
    for key, value in provenance.items():
        if key not in {"path", "parent_path"}:
            _require_equal(dataset.get(key), value, f"run dataset {key}")
    _require_equal(manifest.get("condition_count"), 120, "manifest condition count")
    _require_equal(manifest.get("generation_count"), 960, "manifest generation count")
    instruction = config.get("prompt_instruction")
    if not isinstance(instruction, str) or not instruction:
        raise ValueError("config requires prompt_instruction")
    _require_equal(manifest.get("prompt_instruction_sha256"), hashlib.sha256(instruction.encode()).hexdigest(), "prompt instruction")
    _require_equal(manifest.get("decoding"), {
        "do_sample": False, "batch_size": 1, "enable_thinking": False,
        "max_new_tokens": config.get("generation", {}).get("max_new_tokens"),
    }, "decoding")
    for name in ("model", "tokenizer", "scallop", "hybrid_memory", "graphiti_memory"):
        if not isinstance(manifest.get(name), dict) or not manifest[name]:
            raise ValueError(f"manifest requires {name} identity")
    _require_digest(manifest.get("evaluator_script_sha256"), "evaluator")
    source = manifest.get("source_provenance") or manifest.get("git")
    if not isinstance(source, dict) or not source:
        raise ValueError("manifest requires source_provenance or git source identity")
    if "source_provenance" in manifest:
        if source.get("mode") not in {"git", "archive"}:
            raise ValueError("source_provenance requires git or archive identity")
        source = source.get(source["mode"], {})
    if "git_head" in source:
        if re.fullmatch(r"[0-9a-f]{40}", str(source["git_head"])) is None or type(source.get("dirty")) is not bool:
            raise ValueError("invalid git source identity")
        if source["dirty"]:
            _require_digest(source.get("dirty_diff_sha256"), "dirty source")
    else:
        _require_digest(source.get("manifest_sha256"), "archive source")
    if not all(manifest["scallop"].get(key) for key in ("engine", "rule_version", "version")):
        raise ValueError("Scallop requires engine, rule and version identity")
    model = manifest["model"]
    if not model.get("model_id") or not model.get("resolved_revision"):
        raise ValueError("model requires pinned identity")
    hashes = model.get("file_sha256", {})
    for name in ("tokenizer.json", "tokenizer_config.json"):
        _require_digest(hashes.get(name), name)
    if not any(name.endswith(".safetensors") for name in hashes):
        raise ValueError("model requires weight hashes")
    for name, value in hashes.items():
        _require_digest(value, f"model {name}")
    for key in ("model_id", "resolved_revision"):
        _require_equal(manifest["tokenizer"].get(key), model[key], f"tokenizer {key}")
    for key in ("dtype", "attention_implementation"):
        _require_equal(model.get(key), config.get("generation", {}).get(key), f"generation {key}")
    graphiti = manifest["graphiti_memory"]
    if graphiti.get("episode_failure_count") != 0 or graphiti.get("episode_failures") != []:
        raise ValueError("Graphiti ingestion must have zero failures")
    for key in ("graphiti_core_version", "episode_variant", "retrieval_state", "num_results", "llm_max_tokens", "max_concurrent_histories", "embedding_batch_size"):
        if key not in config.get("graphiti", {}) or key not in graphiti:
            raise ValueError(f"missing Graphiti control {key}")
        _require_equal(graphiti[key], config["graphiti"][key], f"Graphiti {key}")
    mapping = graphiti.get("timestamp_mapping", {})
    for key, config_key in (("base", "timestamp_base"), ("step_seconds", "timestamp_step_seconds")):
        if key not in mapping or config_key not in config["graphiti"]:
            raise ValueError("missing Graphiti timestamp control")
        _require_equal(mapping[key], config["graphiti"][config_key], f"Graphiti timestamp {key}")
    hybrid = manifest["hybrid_memory"]
    if hybrid.get("validator", {}).get("scallop_available") is not True:
        raise ValueError("hybrid retrieval requires authenticated Scallop validator identity")
    for key, config_key in (("top_k", "top_k"), ("hops_requested", "hops"), ("rrf_k", "rrf_k"), ("embedding_batch_size", "embedding_batch_size")):
        if key not in hybrid or config_key not in config.get("retrieval", {}):
            raise ValueError(f"missing retrieval control {key}")
        _require_equal(hybrid[key], config["retrieval"][config_key], f"retrieval {key}")
    for key in ("embedding_model_id", "embedding_revision"):
        if not hybrid.get(key):
            raise ValueError(f"missing retrieval {key}")
        _require_equal(graphiti.get(key), hybrid[key], f"matched embedding {key}")


def _expected_conditions(artifacts: Mapping[str, bytes], config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    schedule = config["schedule"]
    queries = [json.loads(line) for line in artifacts["queries.jsonl"].decode("utf-8").splitlines() if line.strip()]
    expected = {}
    for query in queries:
        if (query.get("split") != schedule.get("source_split")
                or query.get("hardness_profile") != schedule.get("source_profile")):
            continue
        suffixes = schedule["query_suffixes"]
        suffix = next((suffix for suffix in suffixes if query["query_id"].endswith(suffix)), None)
        if suffix is None:
            continue
        family = "preference_change" if suffix == "-preference-change-delayed" else "preference_incongruity"
        for condition, phase, distance in (
            ("pre_update", "pre_update", None), ("post_update", "post_update", None),
            ("delayed_probe", "delayed_probe", None),
            ("token_distance_4096", "token_distance", 4096),
            ("token_distance_8192", "token_distance", 8192),
        ):
            suffix = phase if distance is None else f"{phase}-{distance}"
            identifier = f"{query['query_id']}:{suffix}"
            if identifier in expected:
                raise ValueError("duplicate authenticated query condition")
            expected[identifier] = {
                "history_id": query["history_id"], "query_id": query["query_id"],
                "query_family": family, "condition": condition, "phase": phase,
                "requested_token_distance": distance,
            }
    if len(expected) != 120 or len({row["history_id"] for row in expected.values()}) != 12:
        raise ValueError("authenticated corpus must supply 120 conditions over 12 histories")
    return expected


def validate_predictions(rows: Sequence[Mapping[str, Any]], expected: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != 960:
        raise ValueError("surface requires 960 prediction rows")
    by_key = {}
    golds = {}
    checkpoints: dict[str, int] = {}
    rescored = []
    for row in rows:
        identifier, arm = row.get("evaluation_input_id"), row.get("arm")
        if identifier not in expected or arm not in ARM_NAMES:
            raise ValueError("unexpected condition or arm")
        key = identifier, arm
        if key in by_key:
            raise ValueError("duplicate (evaluation_input_id, arm)")
        by_key[key] = row
        for name, value in expected[identifier].items():
            _require_equal(row.get(name), value, f"prediction {identifier} {name}")
        arm_spec = next(spec for spec in ARMS if spec["name"] == arm)
        _require_equal(row.get("arm_kind"), arm_spec["kind"], "prediction arm kind")
        _require_equal(row.get("prompt_token_cap"), arm_spec["prompt_token_cap"], "prediction token cap")
        if not isinstance(row.get("gold"), str) or not isinstance(row.get("answer"), str):
            raise ValueError("prediction requires answer and gold strings")
        _require_equal(row["gold"], golds.setdefault(identifier, row["gold"]), "within-surface gold")
        if arm_spec["kind"] == "graphiti_memory":
            checkpoint = row.get("retrieval_metadata", {}).get("checkpoint_turn_index")
            if type(checkpoint) is not int or checkpoint < 1:
                raise ValueError("Graphiti prediction requires checkpoint identity")
            _require_equal(checkpoint, checkpoints.setdefault(identifier, checkpoint), "within-surface checkpoint")
        score = _score_short_answer(row["answer"], row["gold"])
        for name, value in score.items():
            _require_equal(row.get(name), value, f"stored score {name}")
        rescored.append({**row, **score})
    if set(by_key) != {(identifier, arm) for identifier in expected for arm in ARM_NAMES}:
        raise ValueError("missing arms or conditions")
    identity = {identifier: {**fields, "checkpoint_turn_index": checkpoints[identifier]} for identifier, fields in expected.items()}
    for row in rows:
        checkpoint = row.get("checkpoint_turn_index")
        if checkpoint is not None:
            _require_equal(checkpoint, checkpoints[row["evaluation_input_id"]], "prediction checkpoint")
    return rescored, identity


def summarize_predictions(rows: Sequence[Mapping[str, Any]], analysis: Mapping[str, Any]) -> dict[str, Any]:
    samples, seed = analysis.get("bootstrap_samples"), analysis.get("bootstrap_seed")
    if type(samples) is not int or samples < 1 or type(seed) is not int:
        raise ValueError("analysis requires positive bootstrap_samples and integer bootstrap_seed")
    comparisons = analysis.get("paired_comparisons")
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("analysis requires paired comparisons")
    if any(not isinstance(pair, (list, tuple)) or len(pair) != 2 or pair[0] == pair[1]
           or not set(pair) <= ARM_NAMES for pair in comparisons):
        raise ValueError("invalid paired comparison")
    return {
        "row_count": len(rows),
        "condition_count": len({row["evaluation_input_id"] for row in rows}),
        "history_cluster_count": len({row["history_id"] for row in rows}),
        "aggregates": {name: _group_metrics(rows, fields) for name, fields in GROUPS.items()},
        "paired_deltas": [_paired_delta(rows, left, right, bootstrap_samples=samples, seed=seed)
                          for left, right in comparisons],
    }


def pool_surfaces(surfaces: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    if set(surfaces) != {"A", "B"}:
        raise ValueError("pooling requires distinct A and B surfaces")
    return [{**row, "surface_assignment": assignment,
             "evaluation_input_id": f"{assignment}:{row['evaluation_input_id']}"}
            for assignment in ("A", "B") for row in surfaces[assignment]]


def _compare_runs(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    for name in ("model", "tokenizer", "scallop", "source_provenance", "git", "evaluator_script_sha256",
                 "evaluator_version", "schedule_version", "prompt_instruction_sha256", "decoding", "arms"):
        _require_equal(left.get(name), right.get(name), f"A/B {name}")
    for name, ignored in (("graphiti_memory", GRAPHITI_RESULTS), ("hybrid_memory", HYBRID_RESULTS)):
        controls = []
        for manifest in (left, right):
            values = deepcopy({key: value for key, value in manifest[name].items() if key not in ignored})
            if name == "graphiti_memory" and isinstance(values.get("session"), dict):
                values["session"].pop("server_filter_violations_dropped", None)
            controls.append(values)
        _require_equal(controls[0], controls[1], f"A/B {name} controls")


def analyze_paper_persona(*, mode: str = "historical_a", run_a: Path, config_a: Path,
                         evidence_root: Path, output_dir: Path, run_b: Path | None = None,
                         config_b: Path | None = None, pair_gate_sha256: str = PAIR_GATE_SHA256) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("unsupported analysis mode")
    if (mode == "kimi_ab" and (run_b is None or config_b is None)) or (
            mode != "kimi_ab" and (run_b is not None or config_b is not None)):
        raise ValueError("--run-b and --config-b are required only for kimi_ab")
    _require_digest(pair_gate_sha256, "pair gate")
    for directory, name in ((Path(evidence_root).absolute(), "results"),
                            (Path(run_a).absolute(), "manifest.json"),
                            (Path(output_dir).absolute(), "analysis.json")):
        safe_path(directory, name)
    for config_path in (config_a, config_b):
        if config_path is not None:
            config_path = Path(config_path).absolute()
            safe_path(config_path.parent, config_path.name)
    if run_b is not None:
        safe_path(Path(run_b).absolute(), "manifest.json")
    evidence_root = Path(evidence_root).resolve()
    runs = {"A": (Path(run_a).resolve(), Path(config_a).resolve())}
    if mode == "kimi_ab":
        runs["B"] = (Path(run_b).resolve(), Path(config_b).resolve())
        if runs["A"][0] == runs["B"][0]:
            raise ValueError("A/B require distinct run directories")
    source_root = Path(__file__).resolve().parents[1]
    freeze_roots = []
    for root in (source_root, evidence_root):
        selection = root / "configs/paper_artifacts.json"
        if selection.is_file():
            freeze_root = json_load(selection).get("freeze_root")
            if freeze_root:
                freeze_roots.append(Path(freeze_root))
    destination = ensure_output_directory(
        output_dir, input_dirs=(source_root, evidence_root, *(path for pair in runs.values() for path in pair)),
        frozen_roots=(*freeze_roots, *paper_evidence_roots(source_root), *paper_evidence_roots(evidence_root)),
    )
    if destination.exists():
        raise ValueError("output directory must be fresh and separate")
    artifact_path = safe_path(destination, "analysis.json")
    surfaces, manifests, configs, identities, summaries = {}, {}, {}, {}, {}
    for assignment, (run, config_path) in runs.items():
        corpus, provenance, artifacts = _authenticate_surface(evidence_root, mode, assignment, pair_gate_sha256)
        manifest = _verify_analysis_inputs(run, corpus)
        config = json_load(config_path)
        _validate_config(manifest, config, config_path, provenance, mode, assignment)
        if "generation_manifest.json" in manifest.get("artifact_sha256", {}):
            generation = json_load(safe_path(run, "generation_manifest.json"))
            for key in ("config_sha256", "dataset", "model", "tokenizer", "scallop", "source_provenance", "git",
                        "evaluator_script_sha256", "evaluator_version", "arms", "decoding", "condition_count",
                        "generation_count", "prompt_instruction_sha256", "schedule_sha256", "schedule_version",
                        "graphiti_memory", "hybrid_memory"):
                _require_equal(generation.get(key), manifest.get(key), f"generation manifest {key}")
        rows, identity = validate_predictions(_read_jsonl(safe_path(run, "predictions.jsonl")), _expected_conditions(artifacts, config))
        summary = summarize_predictions(rows, config.get("analysis", {}))
        metrics = json_load(safe_path(run, "metrics.json"))
        for key, count in (("condition_count", 120), ("generation_count", 960)):
            _require_equal(metrics.get(key), count, f"metrics {key}")
        for name, values in summary["aggregates"].items():
            _require_equal(metrics.get("aggregates", {}).get(name), values, f"stored aggregates {name}")
        _require_equal(metrics.get("aggregates", {}).get("paired_deltas"), summary["paired_deltas"], "stored paired deltas")
        surfaces[assignment], identities[assignment] = rows, identity
        manifests[assignment], configs[assignment] = manifest, config
        summaries[assignment] = {
            "assignment": assignment, "run_manifest_sha256": _digest(safe_path(run, "manifest.json")),
            "config_sha256": manifest["config_sha256"], "dataset": provenance,
            "run_path": str(run), "config_path": str(config_path), **summary,
        }
    result = {
        "version": "paper_persona_analysis.v1", "status": "completed", "mode": mode,
        "scope": ("Descriptive fixed pair-conditioned A/B surfaces, not independent replications; "
                  "history-cluster bootstrap keeps both surfaces in each of 12 sampled history clusters."
                  if mode == "kimi_ab" else
                  "Single-surface descriptive analysis only; no surface-robustness or replication claim."),
        "surfaces": summaries,
        "verification_scope": {
            "datasets": "Fixed corpus hashes and historical transformation or modern pair-gate authentication.",
            "results": "Declared result hashes and manifest/config consistency; not independent execution attestation.",
            "scores": "Saved answers rescored against saved, within-surface-consistent gold labels; causal gold resolution is not rerun.",
            "checkpoints": "Recorded checkpoint identities matched across surfaces; tokenizer distances are not recomputed.",
            "generation": "No model, ingestion, or Scallop execution is performed by this analysis.",
        },
        "historical_source_caveat": ("Artifact authentication does not recover the unavailable historical execution source."
                                     if mode == "historical_a" else None),
    }
    if mode == "kimi_ab":
        _require_equal(normalize_scientific_config(configs["A"]), normalize_scientific_config(configs["B"]), "A/B scientific config")
        _compare_runs(manifests["A"], manifests["B"])
        _require_equal(identities["A"], identities["B"], "A/B logical condition/history/query/phase/checkpoint identity")
        result["pooled"] = summarize_predictions(pool_surfaces(surfaces), configs["A"]["analysis"])
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    destination.mkdir(parents=True, exist_ok=False)
    with artifact_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="historical_a")
    for name in ("run-a", "config-a", "evidence-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-b", type=Path)
    parser.add_argument("--config-b", type=Path)
    parser.add_argument("--pair-gate-sha256", default=PAIR_GATE_SHA256)
    args = parser.parse_args(argv)
    try:
        result = analyze_paper_persona(**vars(args))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({"mode": result["mode"], "status": result["status"], "output_dir": str(args.output_dir),
                      "surfaces": {key: {field: value[field] for field in ("row_count", "condition_count", "history_cluster_count")}
                                   for key, value in result["surfaces"].items()},
                      "pooled": {field: result["pooled"][field] for field in ("row_count", "condition_count", "history_cluster_count")}
                      if "pooled" in result else None}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
