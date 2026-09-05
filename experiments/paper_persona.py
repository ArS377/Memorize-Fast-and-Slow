from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

from experiments.persona_interference_schedule import _authenticate_dataset
from experiments.persona_surface_derivation import authenticate_pair_gate
from neurosym.application.source_provenance import (
    _check_separate_path,
    _no_links,
    ensure_output_directory,
    paper_evidence_roots,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "configs/paper_persona_modes.json"
TEMPLATE_CONFIG = "configs/persona_end_to_end_joint_surface_a.json"
HISTORICAL_PATH = "results/persona_conflict_conversations_surface_a_graphiti_build2"
HISTORICAL_SHA256 = "c64c5c4268d93691b9bdf12119fa31a91cc7e9a798182135593c85128fc89c89"
PARENT_PATH = "results/persona_conflict_conversations_v1"
PARENT_SHA256 = "49247f1c361319cba951b21362ffa4920b2633eb394c6aeee59765dada252fc3"
PAIR_GATE_PATH = "results/persona_surface_pair_gate.json"
PAIR_GATE_SHA256 = "d03540a6575c9d2967d8bfade7fef4e38a0951d98cb702420c06113dd228fb5b"
MODERN_SURFACES = (
    ("A", "results/persona_conflict_conversations_surface_a", "2cfdbf18705204a9ca7412e9e962ad3bb03bd08b2426f09270abbc387b047f65"),
    ("B", "results/persona_conflict_conversations_surface_b", "986cbe8ce306e19169626d5b8729b92e120f45831a76b328cfa1e32401dfd121"),
)
MODES = ("historical_a", "kimi_a", "kimi_ab")


def _surface(assignment: str, path: str, digest: str) -> dict[str, str]:
    return {"assignment": assignment, "dataset_relative_path": path, "dataset_manifest_sha256": digest}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_mode_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    expected = {
        "version": 1,
        "default_mode": "historical_a",
        "template_config": TEMPLATE_CONFIG,
        "parent_dataset_relative_path": PARENT_PATH,
        "parent_generation_manifest_sha256": PARENT_SHA256,
        "pair_gate_relative_path": PAIR_GATE_PATH,
        "pair_gate_sha256": PAIR_GATE_SHA256,
        "modes": {
            "historical_a": {"authentication": "historical", "surfaces": [_surface("A", HISTORICAL_PATH, HISTORICAL_SHA256)]},
            "kimi_a": {"authentication": "pair_gate", "surfaces": [_surface(*MODERN_SURFACES[0])]},
            "kimi_ab": {"authentication": "pair_gate", "surfaces": [_surface(*row) for row in MODERN_SURFACES]},
        },
    }
    registry = _read_json(path)
    if type(registry.get("version")) is not int or registry != expected:
        raise ValueError("mode registry differs from the pinned paper dataset modes")
    return registry


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id) is None:
        raise ValueError("run_id must be 1-64 ASCII letters, digits, underscores or hyphens, starting with a letter or digit")
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])", run_id):
        raise ValueError("run_id cannot be a reserved Windows device name")


def _safe_output_root(output_root: Path, evidence_root: Path) -> Path:
    _no_links(output_root)
    source_roots = {ROOT, evidence_root}
    inputs = [root / name for root in source_roots for name in (HISTORICAL_PATH, PARENT_PATH, PAIR_GATE_PATH, *(row[1] for row in MODERN_SURFACES))]
    protected = [*paper_evidence_roots(ROOT), *paper_evidence_roots(evidence_root)]
    for root in source_roots:
        selection = root / "configs/paper_artifacts.json"
        if selection.exists():
            frozen = _read_json(selection).get("freeze_root")
            if isinstance(frozen, str) and frozen:
                path = Path(frozen).expanduser()
                protected.append(path if path.is_absolute() else root / path)
    output = ensure_output_directory(output_root, inputs, frozen_roots=protected)
    for source_root in source_roots:
        _check_separate_path(source_root, output, "paper output root")
        ensure_output_directory(output, [source_root / name for name in ("tests", "docs", ".git", ".venv", ".devin", ".planning", "__pycache__")])
    for ancestor in (output, *output.parents):
        if (ancestor / "plan.json").exists():
            raise ValueError("output root is inside an existing runset")
    return output


def _pinned_manifest(directory: Path, digest: str) -> dict[str, Any]:
    _no_links(directory)
    path = directory / "generation_manifest.json"
    _no_links(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"generation manifest hash mismatch: {directory}")
    return _read_json(path)


def _authenticate_surface(evidence_root: Path, surface: Mapping[str, str], modern: bool) -> dict[str, bytes]:
    directory = evidence_root / surface["dataset_relative_path"]
    manifest = _pinned_manifest(directory, surface["dataset_manifest_sha256"])
    parent = evidence_root / PARENT_PATH
    _pinned_manifest(parent, PARENT_SHA256)
    if modern:
        if manifest.get("model_identity") == "deterministic-derived-surface":
            raise ValueError("historical data cannot be used as a modern Kimi surface")
        gate_path = evidence_root / PAIR_GATE_PATH
        _no_links(gate_path)
        for assignment, relative, digest in MODERN_SURFACES:
            sibling = evidence_root / relative
            _pinned_manifest(sibling, digest)
        gate = _read_json(gate_path)
        expected_paths = {"parent": parent, **{row[0]: evidence_root / row[1] for row in MODERN_SURFACES}}
        if gate.get("paths") != {key: path.name for key, path in expected_paths.items()}:
            raise ValueError("pair gate paths differ from the pinned repo-shaped layout")
        authenticated = authenticate_pair_gate(
            gate_path, PAIR_GATE_SHA256, dataset_dir=directory,
            expected_assignment=surface["assignment"],
        )
        if (
            authenticated["parent_generation_manifest_sha256"] != PARENT_SHA256
            or Path(authenticated["parent_path"]).resolve() != parent.resolve()
            or authenticated["assignment_manifest_sha256"] != {row[0]: row[2] for row in MODERN_SURFACES}
        ):
            raise ValueError("pair gate bindings differ from the pinned paper datasets")
        _, artifacts, _ = _authenticate_dataset(
            directory, surface["dataset_manifest_sha256"], gate_path,
            PAIR_GATE_SHA256, surface["assignment"],
        )
    else:
        if manifest.get("model_identity") != "deterministic-derived-surface":
            raise ValueError("historical mode requires the pinned deterministic surface")
        _, artifacts, authenticated_parent = _authenticate_dataset(directory, surface["dataset_manifest_sha256"])
        if authenticated_parent is None or (
            authenticated_parent["generation_manifest_sha256"] != PARENT_SHA256
            or Path(authenticated_parent["path"]).resolve() != parent.resolve()
        ):
            raise ValueError("historical parent differs from the pinned paper dataset")
    return artifacts


def _controls(template: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(template))
    result.pop("runtime")
    result["schedule"].pop("source_manifest_sha256")
    for section in ("scallop", "retrieval", "graphiti"):
        result[section] = {key: value for key, value in result[section].items() if not key.endswith("_env")}
    return result


def build_plan(*, evidence_root: Path, output_root: Path, run_id: str, mode: str = "historical_a") -> dict[str, Any]:
    _validate_run_id(run_id)
    registry = load_mode_registry()
    if mode not in MODES:
        raise ValueError("unknown paper persona mode")
    evidence_root = Path(os.path.abspath(Path(evidence_root).expanduser()))
    _no_links(evidence_root)
    if not evidence_root.is_dir():
        raise ValueError("evidence_root must be an existing repo-shaped directory")
    evidence_root = evidence_root.resolve()
    output_root = _safe_output_root(Path(output_root).expanduser().absolute(), evidence_root)
    runset = output_root / mode / run_id
    _no_links(runset)
    ensure_output_directory(runset)
    if os.path.lexists(runset):
        raise ValueError("runset already exists; choose a new run_id (resume is not supported)")
    template = _read_json(ROOT / TEMPLATE_CONFIG)
    expected_arms = [
        {"name": f"{kind}_{cap}", "kind": kind, "prompt_token_cap": cap}
        for kind in ("sliding_context", "structured_memory", "hybrid_kg_memory", "graphiti_memory")
        for cap in (4096, 16384)
    ]
    if template["arms"] != expected_arms or template["schedule"]["source_manifest_sha256"] != HISTORICAL_SHA256:
        raise ValueError("paper template must retain the pinned historical eight-arm layout")
    modern = mode != "historical_a"
    surfaces = []
    for source in registry["modes"][mode]["surfaces"]:
        artifacts = _authenticate_surface(evidence_root, source, modern)
        schedule = template["schedule"]
        queries = [
            json.loads(line) for line in artifacts["queries.jsonl"].splitlines() if line.strip()
        ]
        count = sum(
            row.get("split") == schedule["source_split"]
            and row.get("hardness_profile") == schedule["source_profile"]
            and str(row.get("query_id", "")).endswith(tuple(schedule["query_suffixes"]))
            for row in queries
        )
        conditions = count * (3 + len(schedule["token_distance_thresholds"]))
        if conditions != 120:
            raise ValueError("pinned surface must select 120 planned evaluation conditions")
        name = "surface_" + source["assignment"].lower()
        surfaces.append({
            **source,
            "config_path": f"configs/{name}.json",
            "run_dir": name,
            "index_dir": f"{name}/embedding_indexes",
            "cache_dir": f"{name}/caches",
            "build_id": f"{mode}__{run_id}__{name}",
            "condition_count": conditions,
            "generation_count": conditions * len(template["arms"]),
        })
    plan = {
        "version": 1,
        "mode": mode,
        "run_id": run_id,
        "surfaces": surfaces,
        "controls": _controls(template),
        "generation_count": sum(row["generation_count"] for row in surfaces),
        "count_status": "planned; tokenizer checkpoint validation occurs in the evaluator",
    }
    if modern:
        plan.update({"pair_gate_sha256": PAIR_GATE_SHA256, "parent_generation_manifest_sha256": PARENT_SHA256})
    return plan


def generated_config(plan: Mapping[str, Any], surface: Mapping[str, Any]) -> dict[str, Any]:
    config = _read_json(ROOT / TEMPLATE_CONFIG)
    if _controls(config) != plan["controls"]:
        raise ValueError("template controls changed after planning")
    config["runtime"]["dataset_dir_env"] = "PAPER_DATASET_DIR"
    config["runtime"]["output_dir_env"] = "PAPER_OUTPUT_DIR"
    config["schedule"]["source_manifest_sha256"] = surface["dataset_manifest_sha256"]
    config["retrieval"]["index_root_env"] = "PAPER_RETRIEVAL_INDEX_ROOT"
    config["graphiti"]["build_id_env"] = "PAPER_GRAPHITI_BUILD_ID"
    if plan["mode"] != "historical_a":
        config["runtime"].update({
            "pair_gate_path_env": "PAPER_PAIR_GATE_PATH",
            "pair_gate_sha256_env": "PAPER_PAIR_GATE_SHA256",
            "surface_assignment": surface["assignment"],
        })
    return config


def surface_environment(plan: Mapping[str, Any], surface: Mapping[str, Any], *, evidence_root: Path, runset: Path, environ: Mapping[str, str]) -> dict[str, str]:
    env = dict(environ)
    env.update({
        "PAPER_DATASET_DIR": str(evidence_root.resolve() / surface["dataset_relative_path"]),
        "PAPER_OUTPUT_DIR": str(runset / surface["run_dir"]),
        "PAPER_RETRIEVAL_INDEX_ROOT": str(runset / surface["index_dir"]),
        "PAPER_GRAPHITI_BUILD_ID": surface["build_id"],
        "PERSONA_HYBRID_RETRIEVAL_CACHE": str(runset / surface["cache_dir"] / "hybrid_retrieval.json"),
        "PERSONA_GRAPHITI_RETRIEVAL_CACHE": str(runset / surface["cache_dir"] / "graphiti_retrieval.json"),
    })
    if plan["mode"] != "historical_a":
        env["PAPER_PAIR_GATE_PATH"] = str(evidence_root.resolve() / PAIR_GATE_PATH)
        env["PAPER_PAIR_GATE_SHA256"] = PAIR_GATE_SHA256
    else:
        env.pop("PAPER_PAIR_GATE_PATH", None)
        env.pop("PAPER_PAIR_GATE_SHA256", None)
    return env


def _write_json_fresh(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(*, evidence_root: Path, output_root: Path, run_id: str, mode: str = "historical_a", prepare: bool = False, execute: bool = False, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    if prepare and execute:
        raise ValueError("prepare and execute are mutually exclusive")
    plan = build_plan(evidence_root=evidence_root, output_root=output_root, run_id=run_id, mode=mode)
    runset = Path(output_root).expanduser().resolve() / mode / run_id
    base_env = os.environ if environ is None else environ
    model_inputs = [Path(base_env[name]).expanduser() for name in ("PERSONA_QWEN_MODEL_PATH", "PERSONA_EMBEDDING_MODEL_PATH") if base_env.get(name, "").strip()]
    ensure_output_directory(runset, model_inputs)
    if not prepare and not execute:
        return plan
    configs = [generated_config(plan, surface) for surface in plan["surfaces"]]
    _no_links(runset)
    runset.mkdir(parents=True, exist_ok=False)
    (runset / "configs").mkdir()
    for surface, config in zip(plan["surfaces"], configs, strict=True):
        _write_json_fresh(runset / surface["config_path"], config)
    _write_json_fresh(runset / "plan.json", plan)
    if not execute:
        return plan
    from experiments.persona_end_to_end_benchmark import load_benchmark_config

    environments = []
    for surface in plan["surfaces"]:
        env = surface_environment(plan, surface, evidence_root=evidence_root, runset=runset, environ=base_env)
        try:
            resolved = load_benchmark_config(runset / surface["config_path"], environ=env)
        except (ValueError, OSError):
            raise ValueError(f"runtime validation failed for surface {surface['assignment']}; check required environment variable names in the generated config") from None
        model_inputs = [resolved.model_path, resolved.hybrid_memory.embedding_model_path, resolved.graphiti_memory.embedding_model_path]
        ensure_output_directory(runset, model_inputs)
        environments.append(env)
    for surface in plan["surfaces"]:
        (runset / surface["index_dir"]).mkdir(parents=True)
        (runset / surface["cache_dir"]).mkdir()
    for surface, env in zip(plan["surfaces"], environments, strict=True):
        subprocess.run(
            [sys.executable, "-m", "experiments.persona_end_to_end_benchmark", "--config", str(runset / surface["config_path"])],
            cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan, prepare, or explicitly execute isolated paper persona runsets.")
    parser.add_argument("--mode", choices=MODES, default="historical_a")
    parser.add_argument("--evidence-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = run(**vars(args))
    except subprocess.CalledProcessError:
        parser.exit(1, "evaluator subprocess failed; no subsequent surface was started\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
