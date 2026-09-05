from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from experiments import persona_historical_surface as historical


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_COMMIT = "5b02fdc16b25f90f95df4814907b672db8d24e1b"
DATA_COMMIT = "ab3f8b3056a8d48f627f129cfc6bebcf239ef8a6"
CHILD_NAME = "persona_conflict_conversations_surface_a_graphiti_build2"
PARENT_NAME = "persona_conflict_conversations_v1"
CHILD_HASH = "c64c5c4268d93691b9bdf12119fa31a91cc7e9a798182135593c85128fc89c89"
PARENT_HASH = "49247f1c361319cba951b21362ffa4920b2633eb394c6aeee59765dada252fc3"
ARTIFACT_NAMES = (*historical.SOURCE_ARTIFACTS, "dialogue.jsonl", "requests.jsonl", "raw_responses.jsonl")


def _hash(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def _stable_hash(value):
    return _hash(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _save_manifest(directory, manifest):
    payload = _json(manifest)
    (directory / "generation_manifest.json").write_bytes(payload)
    return _hash(payload)


def _manifest(directory):
    return json.loads((directory / "generation_manifest.json").read_bytes())


def _write_corpus(directory, artifacts, manifest):
    directory.mkdir(parents=True)
    for name, payload in artifacts.items():
        (directory / name).write_bytes(payload)
    return _save_manifest(directory, manifest)


@pytest.fixture
def corpus(tmp_path):
    histories = [f"history-{index:03d}" for index in range(1, 17)]
    text = (
        "Cedar tea, mint tea, oolong tea, window seating, aisle seating, vegetable ramen, "
        "mushroom risotto, evening delivery, weekend delivery, digital receipts, cedar glass."
    )
    events = [
        {
            "history_id": history,
            "event_id": f"{history}-event",
            "model_text": text,
            "surface_object": "cedar tea",
            "fact": {"object": "latent-value", "support_text": "cedar tea"},
        }
        for history in histories
    ]
    queries = [
        {
            "history_id": history,
            "query_id": f"{history}-query",
            "query_text": "Which cedar tea?",
            "surface_query_text": "Which cedar tea?",
            "surface_gold": "cedar tea",
            "gold": "latent-value",
            "checkpoint_contract": {"trigger_event_id": f"{history}-event"},
        }
        for history in histories
    ]
    artifacts = {name: b'{"untouched": "cedar tea"}\r\n' for name in ARTIFACT_NAMES}
    artifacts.update({
        "events.jsonl": _jsonl(events),
        "dialogue.jsonl": _jsonl([{"history_id": event["history_id"], "text": text} for event in events]),
        "queries.jsonl": _jsonl(queries),
    })
    parent_manifest = {
        "status": "completed",
        "generator_role": "Kimi K3 generation-only surface realization",
        "model_identity": "kimi-k3",
        "prompt_schema_version": "persona-conversation.v3",
        "provider_identity_assurance": "endpoint-self-reported; artifact hashes are cryptographic",
        "artifact_sha256": {name: _hash(payload) for name, payload in artifacts.items()},
    }
    parent_dir = tmp_path / "parent corpus"
    parent_hash = _write_corpus(parent_dir, artifacts, parent_manifest)
    mapping = historical.build_surface_mapping(histories, 137)
    child_artifacts = historical.transform_surface_artifacts(artifacts, mapping)
    child_manifest = {
        "status": "completed",
        "completion_status": "completed",
        "model_identity": historical.HISTORICAL_MODEL_IDENTITY,
        "derivation": {"algorithm": historical.DERIVATION_ALGORITHM, "version": historical.DERIVATION_VERSION, "seed": 137},
        "parent": {
            "corpus_name": parent_dir.name,
            "generation_manifest_sha256": parent_hash,
            "artifact_sha256": parent_manifest["artifact_sha256"],
            "kimi_provenance": {key: value for key, value in parent_manifest.items() if key != "artifact_sha256"},
        },
        "surface_mapping": mapping,
        "surface_mapping_sha256": _stable_hash(mapping),
        "visible_replacement_contract": [
            {"source_phrase": source, "pattern": pattern, "mode": mode}
            for source, pattern, mode in historical.ROLE_VARIANT_RULES
        ],
        "visible_replacement_contract_sha256": _stable_hash(historical.ROLE_VARIANT_RULES),
        "artifact_sha256": {name: _hash(payload) for name, payload in child_artifacts.items()},
    }
    child_dir = tmp_path / "child corpus"
    child_hash = _write_corpus(child_dir, child_artifacts, child_manifest)
    return child_dir, child_hash, parent_dir


def test_authentication_contract_and_exact_parent_policy(corpus):
    child, digest, parent = corpus
    provenance, artifacts, authenticated_parent = historical.authenticate_historical_dataset(child, digest)
    assert provenance == {
        "path": str(child),
        "generation_model": "deterministic-derived-surface",
        "generation_manifest_sha256": digest,
        "artifact_sha256": {name: _hash(payload) for name, payload in artifacts.items()},
        "checkpoint_policy": "authenticated_parent_exact_indices",
        "derivation": _manifest(child)["derivation"],
        "parent_generation_manifest_sha256": _hash((parent / "generation_manifest.json").read_bytes()),
        "parent_path": str(parent.resolve()),
    }
    assert set(artifacts) == {*historical.SOURCE_ARTIFACTS, "dialogue.jsonl"}
    assert authenticated_parent["path"] == parent.resolve()
    assert authenticated_parent["generation_manifest_sha256"] == provenance["parent_generation_manifest_sha256"]
    assert authenticated_parent["artifact_bytes"] == {name: (parent / name).read_bytes() for name in ARTIFACT_NAMES}
    assert artifacts["facts.jsonl"] == b'{"untouched": "cedar tea"}\r\n'
    for before, after in zip(
        historical._load_jsonl(authenticated_parent["artifact_bytes"]["queries.jsonl"], "parent"),
        historical._load_jsonl(artifacts["queries.jsonl"], "child"),
        strict=True,
    ):
        assert before["gold"] == after["gold"]
        assert before["checkpoint_contract"] == after["checkpoint_contract"]
        assert before["surface_gold"] != after["surface_gold"]


def test_relocated_siblings_need_no_git(corpus, tmp_path):
    child, digest, parent = corpus
    relocated = tmp_path / "archive without git and with spaces"
    shutil.copytree(child, relocated / child.name)
    shutil.copytree(parent, relocated / parent.name)
    provenance, _, authenticated_parent = historical.authenticate_historical_dataset(relocated / child.name, digest)
    assert provenance["checkpoint_policy"] == "authenticated_parent_exact_indices"
    assert authenticated_parent["path"] == (relocated / parent.name).resolve()
    assert not (relocated / ".git").exists()


@pytest.mark.parametrize("name", ARTIFACT_NAMES)
@pytest.mark.parametrize("side", ["parent", "child"])
def test_every_artifact_hash_is_authenticated(corpus, name, side):
    child, digest, parent = corpus
    path = (parent if side == "parent" else child) / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=f"artifact hash mismatch for {name}"):
        historical.authenticate_historical_dataset(child, digest)


@pytest.mark.parametrize("name", ["events.jsonl", "dialogue.jsonl", "queries.jsonl", "facts.jsonl", "requests.jsonl"])
def test_rehashed_child_cannot_evade_canonical_transformation(corpus, name):
    child, _, _ = corpus
    path = child / name
    path.write_bytes(path.read_bytes() + b" ")
    manifest = _manifest(child)
    manifest["artifact_sha256"][name] = _hash(path.read_bytes())
    digest = _save_manifest(child, manifest)
    with pytest.raises(ValueError, match="canonical transformation mismatch"):
        historical.authenticate_historical_dataset(child, digest)


@pytest.mark.parametrize("field,value", [
    ("gold", "tampered-latent-gold"),
    ("surface_gold", "tampered surface gold"),
    ("checkpoint_contract", {"trigger_event_id": "other-history"}),
])
def test_rehashed_query_gold_and_causal_fields_are_rejected(corpus, field, value):
    child, _, _ = corpus
    rows = historical._load_jsonl((child / "queries.jsonl").read_bytes(), "queries")
    rows[0][field] = value
    payload = _jsonl(rows)
    (child / "queries.jsonl").write_bytes(payload)
    manifest = _manifest(child)
    manifest["artifact_sha256"]["queries.jsonl"] = _hash(payload)
    digest = _save_manifest(child, manifest)
    with pytest.raises(ValueError, match="query gold or causal contract"):
        historical.authenticate_historical_dataset(child, digest)


@pytest.mark.parametrize("seed", [True, False, "137", 137.0, None])
def test_seed_type_is_strict(corpus, seed):
    child, _, _ = corpus
    manifest = _manifest(child)
    manifest["derivation"]["seed"] = seed
    with pytest.raises(ValueError, match="invalid derivation contract"):
        historical.authenticate_historical_dataset(child, _save_manifest(child, manifest))


def test_changed_integer_seed_requires_canonical_mapping(corpus):
    child, _, _ = corpus
    manifest = _manifest(child)
    manifest["derivation"]["seed"] = 911
    with pytest.raises(ValueError, match="does not match its declared seed"):
        historical.authenticate_historical_dataset(child, _save_manifest(child, manifest))


@pytest.mark.parametrize("rehash", [False, True])
def test_tampered_mapping_is_rejected_even_with_new_mapping_hash(corpus, rehash):
    child, _, _ = corpus
    manifest = _manifest(child)
    manifest["surface_mapping"][0]["target_phrase"] = "invented delivery"
    if rehash:
        manifest["surface_mapping_sha256"] = _stable_hash(manifest["surface_mapping"])
    with pytest.raises(ValueError, match="mapping"):
        historical.authenticate_historical_dataset(child, _save_manifest(child, manifest))


@pytest.mark.parametrize("field,value", [
    ("model_identity", "kimi-k3"),
    ("model_identity", "other-derived-surface"),
    ("method", "kimi_surface_realization"),
    ("derivation", {"algorithm": "other", "version": historical.DERIVATION_VERSION, "seed": 137}),
    ("derivation", {"algorithm": historical.DERIVATION_ALGORITHM, "version": "persona-surface-derivation.v1", "seed": 137}),
    ("status", "running"),
    ("visible_replacement_contract", []),
    ("visible_replacement_contract_sha256", "0" * 64),
])
def test_wrong_identity_contract_and_replacement_rules(corpus, field, value):
    child, _, _ = corpus
    manifest = _manifest(child)
    manifest[field] = value
    with pytest.raises(ValueError):
        historical.authenticate_historical_dataset(child, _save_manifest(child, manifest))


def test_parent_manifest_bytes_and_copied_provenance_are_authenticated(corpus):
    child, digest, parent = corpus
    manifest = _manifest(parent)
    manifest["prompt_schema_version"] = "forged"
    parent_hash = _save_manifest(parent, manifest)
    with pytest.raises(ValueError, match="actual parent generation manifest hash mismatch"):
        historical.authenticate_historical_dataset(child, digest)
    child_manifest = _manifest(child)
    child_manifest["parent"]["generation_manifest_sha256"] = parent_hash
    with pytest.raises(ValueError, match="copied Kimi parent provenance"):
        historical.authenticate_historical_dataset(child, _save_manifest(child, child_manifest))


def test_rehashed_parent_artifacts_must_match_copied_hashes(corpus):
    child, _, parent = corpus
    payload = (parent / "facts.jsonl").read_bytes() + b" "
    (parent / "facts.jsonl").write_bytes(payload)
    manifest = _manifest(parent)
    manifest["artifact_sha256"]["facts.jsonl"] = _hash(payload)
    child_manifest = _manifest(child)
    child_manifest["parent"]["generation_manifest_sha256"] = _save_manifest(parent, manifest)
    with pytest.raises(ValueError, match="actual parent artifact hash manifest differs"):
        historical.authenticate_historical_dataset(child, _save_manifest(child, child_manifest))


@pytest.mark.parametrize("name", ["", ".", "..", "../parent", "..\\parent", "/parent", "C:parent", "C:\\parent", "parent/name", "parent.", "parent ", "NUL", "child corpus"])
def test_parent_must_be_a_safe_distinct_sibling(corpus, name):
    child, _, _ = corpus
    manifest = _manifest(child)
    manifest["parent"]["corpus_name"] = name
    with pytest.raises(ValueError, match="parent corpus name|sibling corpus"):
        historical.authenticate_historical_dataset(child, _save_manifest(child, manifest))


@pytest.mark.parametrize("name", ["../outside", "..\\outside", "C:outside", "generation_manifest.json"])
def test_artifact_paths_cannot_escape_corpus(corpus, name):
    child, _, _ = corpus
    manifest = _manifest(child)
    manifest["artifact_sha256"][name] = "0" * 64
    with pytest.raises(ValueError, match="invalid child artifact name"):
        historical.authenticate_historical_dataset(child, _save_manifest(child, manifest))


@pytest.mark.parametrize("name", ["requests.jsonl", "raw_responses.jsonl", "facts.jsonl"])
def test_missing_provenance_and_source_hashes_rejected(corpus, name):
    child, _, parent = corpus
    manifest = _manifest(child)
    parent_manifest = _manifest(parent)
    for hashes in (manifest["artifact_sha256"], manifest["parent"]["artifact_sha256"], parent_manifest["artifact_sha256"]):
        del hashes[name]
    manifest["parent"]["generation_manifest_sha256"] = _save_manifest(parent, parent_manifest)
    with pytest.raises(ValueError, match="omits required artifact hashes"):
        historical.authenticate_historical_dataset(child, _save_manifest(child, manifest))


def test_manifest_hash_pin_cannot_be_omitted_or_changed(corpus):
    child, digest, _ = corpus
    for bad in (None, "", "0" * 64, digest.upper()):
        with pytest.raises(ValueError, match="manifest hash"):
            historical.authenticate_historical_dataset(child, bad)
    path = child / "generation_manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        historical.authenticate_historical_dataset(child, digest)


@pytest.mark.parametrize("seed", [137, 911])
def test_mapping_category_blocks_uniqueness_and_seating_roles(seed):
    histories = [f"history-{index:03d}" for index in range(1, 17)]
    mapping = historical.build_surface_mapping(histories, seed)
    assert len(mapping) == 176
    assert mapping == historical.build_surface_mapping(histories, seed)
    for category, vocabulary in historical.CATEGORY_VOCABULARIES.items():
        rows = [row for row in mapping if row["category"] == category]
        assert {row["target_phrase"] for row in rows} == set(vocabulary)
    for row in mapping:
        if row["category"] == "seating":
            role = "positive" if row["source_phrase"] == "window seating" else "avoided"
            assert row["target_phrase"] in historical.SEATING_ROLE_VOCABULARIES[role]


@pytest.mark.parametrize("histories,seed", [(["h"] * 16, 137), ([""] * 16, 137), (["h"], 137), ([], 137), (["h"], True)])
def test_invalid_mapping_inputs(histories, seed):
    with pytest.raises(ValueError):
        historical.build_surface_mapping(histories, seed)


def test_role_replacement_order_case_articles_and_descriptors():
    replacements = {
        "cedar glass": "amber compass", "cedar tea": "thyme tea", "mint tea": "saffron tea",
        "oolong tea": "matcha tea", "window seating": "outward-facing seating",
        "aisle seating": "walkway-side seating", "evening delivery": "Tuesday delivery",
        "weekend delivery": "curbside delivery",
    }
    before = "CEDAR GLASS; cedar keepsake; Cedar tea; minty; oolong; a window seat; an aisle spot; evening option; weekend date."
    after = historical._replace_visible_text(before, replacements)
    assert after == "AMBER COMPASS; amber compass; Thyme tea; saffron tea; matcha tea; an outward-facing seat; a walkway-side spot; Tuesday option; curbside delivery date."
    historical._validate_visible_transform(before, after, replacements, "test")
    with pytest.raises(ValueError, match="contradiction"):
        historical._validate_visible_transform("cedar tea", "cedar tea and thyme tea", replacements, "test")
    with pytest.raises(ValueError, match="lost replacement targets"):
        historical._validate_visible_transform("cedar tea", "nothing", replacements, "test")


def _raw_mismatches(directory, manifest_hash):
    manifest_bytes = (directory / "generation_manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {"generation_manifest.json": manifest_hash, **manifest["artifact_sha256"]}
    mismatches = []
    for name, pinned in expected.items():
        payload = (directory / name).read_bytes()
        actual = _hash(payload)
        if actual != pinned:
            crlf = payload.count(b"\r\n")
            lf = payload.count(b"\n")
            mismatches.append(f"{directory.name}/{name}: expected {pinned}, got {actual}; CRLF={crlf}, LF={lf}")
    return mismatches


def test_as_found_historical_corpus_against_original_pins():
    child = ROOT / "results" / CHILD_NAME
    parent = ROOT / "results" / PARENT_NAME
    if not child.is_dir() or not parent.is_dir():
        pytest.skip("historical evidence directories are unavailable")
    mismatches = _raw_mismatches(child, CHILD_HASH) + _raw_mismatches(parent, PARENT_HASH)
    if mismatches:
        print("\nAS-FOUND RAW-BYTE MISMATCHES (no normalization or pin changes):\n" + "\n".join(mismatches))
        with pytest.raises(ValueError, match="hash mismatch"):
            historical.authenticate_historical_dataset(child, CHILD_HASH)
        pytest.xfail("as-found historical bytes differ from original pins; see raw-byte mismatch report")
    provenance, _, authenticated_parent = historical.authenticate_historical_dataset(child, CHILD_HASH)
    assert provenance["checkpoint_policy"] == "authenticated_parent_exact_indices"
    assert authenticated_parent["generation_manifest_sha256"] == PARENT_HASH


@pytest.fixture(scope="module")
def historical_git_corpus(tmp_path_factory):
    if not shutil.which("git"):
        pytest.skip("Git unavailable for read-only historical blob recovery")
    check = subprocess.run(["git", "cat-file", "-e", DATA_COMMIT], cwd=ROOT, capture_output=True)
    if check.returncode:
        pytest.skip("local recovery commit unavailable")
    destination = tmp_path_factory.mktemp("historical raw git blobs")
    for name in (CHILD_NAME, PARENT_NAME):
        target = destination / name
        target.mkdir()
        for artifact in ("generation_manifest.json", *ARTIFACT_NAMES):
            result = subprocess.run(
                ["git", "show", f"{DATA_COMMIT}:results/{name}/{artifact}"],
                cwd=ROOT, capture_output=True, check=True,
            )
            (target / artifact).write_bytes(result.stdout)
    return destination / CHILD_NAME, destination / PARENT_NAME


def test_actual_historical_git_blob_corpus_and_parent_validate(historical_git_corpus):
    child, parent = historical_git_corpus
    assert not _raw_mismatches(child, CHILD_HASH)
    assert not _raw_mismatches(parent, PARENT_HASH)
    provenance, artifacts, authenticated_parent = historical.authenticate_historical_dataset(child, CHILD_HASH)
    assert provenance["checkpoint_policy"] == "authenticated_parent_exact_indices"
    assert authenticated_parent["generation_manifest_sha256"] == PARENT_HASH
    assert len({row["history_id"] for row in historical._load_jsonl(artifacts["events.jsonl"], "events")}) == 16
    assert _manifest(child)["surface_mapping"] == historical.build_surface_mapping(
        [f"history-{index:03d}" for index in range(1, 17)], 137
    )


def test_scheduler_accepts_historical_protocol_and_preserves_parent_checkpoints(historical_git_corpus):
    from dataclasses import replace
    from experiments.persona_interference_schedule import ScheduleConfig, build_evaluation_schedule

    class Tokenizer:
        def encode(self, text):
            return text.split()

        def metadata(self):
            return {"name": "whitespace", "revision": "fixture"}

    child, parent = historical_git_corpus
    config = ScheduleConfig(
        source_split="test", source_profile="anti_shortcut_interleaved_v3",
        source_manifest_sha256=CHILD_HASH, seed=73, concurrent_accounts=8,
        min_segment_events=4, max_segment_events=8,
        query_suffixes=("-preference-change-delayed", "-preference-incongruity-delayed"),
        token_distance_thresholds=(100, 500), preserve_generated_dialogue=True,
    )
    scheduled = build_evaluation_schedule(child, Tokenizer(), config)
    original = build_evaluation_schedule(parent, Tokenizer(), replace(config, source_manifest_sha256=PARENT_HASH))
    assert len(scheduled["inputs"]) == 120
    assert scheduled["dataset"]["checkpoint_policy"] == "authenticated_parent_exact_indices"
    assert [(r["evaluation_input_id"], r["checkpoint_turn_index"]) for r in scheduled["inputs"]] == [
        (r["evaluation_input_id"], r["checkpoint_turn_index"]) for r in original["inputs"]
    ]


def test_historical_dispatch_refuses_pair_gate(historical_git_corpus):
    from experiments.persona_interference_schedule import _authenticate_dataset

    child, _ = historical_git_corpus
    with pytest.raises(ValueError, match="cannot use the Kimi pair-gate"):
        _authenticate_dataset(child, CHILD_HASH, Path("gate.json"), "0" * 64, "A")


def test_actual_historical_corpus_relocated_with_sibling_parent(historical_git_corpus, tmp_path):
    child, parent = historical_git_corpus
    destination = tmp_path / "relocated archive with spaces"
    shutil.copytree(child, destination / child.name)
    shutil.copytree(parent, destination / parent.name)
    provenance, _, authenticated_parent = historical.authenticate_historical_dataset(destination / child.name, CHILD_HASH)
    assert provenance["parent_path"] == str((destination / parent.name).resolve())
    assert authenticated_parent["generation_manifest_sha256"] == PARENT_HASH


def test_recovered_helpers_match_local_historical_source():
    if not shutil.which("git"):
        pytest.skip("Git unavailable for read-only source comparison")
    result = subprocess.run(
        ["git", "show", f"{RECOVERY_COMMIT}:experiments/persona_surface_derivation.py"],
        cwd=ROOT, capture_output=True,
    )
    if result.returncode:
        pytest.skip("local source recovery candidate unavailable")
    original = ast.parse(result.stdout)
    recovered = ast.parse(Path(historical.__file__).read_bytes())
    original_functions = {node.name: node for node in original.body if isinstance(node, ast.FunctionDef)}
    recovered_functions = {node.name: node for node in recovered.body if isinstance(node, ast.FunctionDef)}
    for name in (
        "_sha256_bytes", "_stable_hash", "_jsonl_bytes", "_load_jsonl", "_digest_int",
        "_replacement_index", "_preserve_capitalization", "_role_replacement",
        "_repair_indefinite_articles", "_replace_visible_text", "_stale_role_variants",
        "_validate_visible_transform", "transform_surface_artifacts",
    ):
        assert ast.dump(original_functions[name]) == ast.dump(recovered_functions[name]), name
    mapping = recovered_functions["build_surface_mapping"]
    mapping.body = mapping.body[:1] + mapping.body[4:]
    assert ast.dump(original_functions["build_surface_mapping"]) == ast.dump(mapping)
    assignments = lambda tree: {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    before, after = assignments(original), assignments(recovered)
    for name in (
        "DERIVATION_VERSION", "DERIVATION_ALGORITHM", "SURFACE_ASSIGNMENT_SEEDS",
        "CATEGORY_SOURCE_PHRASES", "CATEGORY_VOCABULARIES", "SEATING_ROLE_VOCABULARIES",
        "ROLE_VARIANT_RULES", "_COMPILED_ROLE_VARIANT_RULES",
    ):
        assert ast.dump(before[name]) == ast.dump(after[name]), name


@pytest.mark.parametrize("side", ["parent", "child"])
def test_missing_artifact_is_rejected(corpus, side):
    child, digest, parent = corpus
    ((parent if side == "parent" else child) / "events.jsonl").unlink()
    with pytest.raises(ValueError, match="cannot read corpus artifact"):
        historical.authenticate_historical_dataset(child, digest)


def test_missing_sibling_parent_is_rejected(corpus):
    child, digest, parent = corpus
    shutil.rmtree(parent)
    with pytest.raises(ValueError, match="cannot read corpus artifact"):
        historical.authenticate_historical_dataset(child, digest)


@pytest.mark.parametrize("payload", [b"[]", b"{", b"\xff", b'{"status": "running"}'])
def test_malformed_manifest_is_rejected(corpus, payload):
    child, _, _ = corpus
    (child / "generation_manifest.json").write_bytes(payload)
    with pytest.raises(ValueError):
        historical.authenticate_historical_dataset(child, _hash(payload))


@pytest.mark.parametrize("side", ["parent", "child"])
def test_symlink_artifacts_are_rejected(corpus, tmp_path, side):
    child, digest, parent = corpus
    path = (parent if side == "parent" else child) / "events.jsonl"
    target = tmp_path / "external events.jsonl"
    target.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink creation unavailable: {error}")
    with pytest.raises(ValueError, match="regular file inside its corpus"):
        historical.authenticate_historical_dataset(child, digest)


def test_import_does_not_load_optional_or_modern_generator_dependencies():
    code = (
        "import sys; import experiments.persona_historical_surface; "
        "assert not any(name in sys.modules for name in "
        "('torch', 'transformers', 'openai', 'experiments.persona_surface_derivation', "
        "'experiments.persona_conversation_generator', 'experiments.persona_interference_schedule'))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)
