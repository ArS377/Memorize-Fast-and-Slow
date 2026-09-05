import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


SPEC = importlib.util.spec_from_file_location("paper_artifacts", Path(__file__).parents[1] / "scripts/paper_artifacts.py")
artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(artifacts)


def test_original_evidence_still_matches_as_found_snapshot():
    import os

    snapshot_path = os.environ.get("NEUROSYM_AS_FOUND_SNAPSHOT")
    if not snapshot_path:
        pytest.skip("set NEUROSYM_AS_FOUND_SNAPSHOT for the original-byte integration check")
    digest = os.environ["NEUROSYM_AS_FOUND_SNAPSHOT_SHA256"]
    snapshot = Path(snapshot_path)
    verified = artifacts.verify(snapshot, digest, require_trusted=True)
    assert verified["trust"] == "external_digest_match"
    assert not verified["errors"]
    manifest = artifacts.json_load(snapshot / "freeze_manifest.json")
    assert manifest.get("evidence_source", "worktree") == "worktree"
    root = Path(__file__).resolve().parents[1]
    entries = [entry for entry in manifest["entries"] if entry["origin"]["kind"] == "worktree"]
    assert entries
    for entry in entries:
        assert artifacts.observe(root, entry["path"]) == entry["observed"], entry["path"]
    print(f"Verified {len(entries)} original evidence files against the as-found snapshot")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else artifacts.encode(value))
    return path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE).decode().strip()


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "repository with spaces"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Fixture")
    git(root, "config", "core.autocrlf", "false")
    put(root, "experiments/evaluator.py", b"print('baseline')\n")
    put(root, "requirements.txt", b"example==1.0\n")
    config = put(root, "configs/run.json", b'{"seed": 3}\n').read_bytes()
    data = b'{"id":1}\r\n{"id":2}\r\n'
    evidence = []
    for name, role, manifests in (
        ("primary", "result", ["manifest.json", "generation_manifest.json"]),
        ("continual", "result", ["manifest.json"]),
        ("interleaved", "result", ["manifest.json"]),
        ("child", "input", ["generation_manifest.json"]),
        ("parent", "input", ["generation_manifest.json"]),
        ("stream", "input", []),
        ("interleaved_input", "input", []),
    ):
        evidence.append({"path": "results/" + name, "role": role, "manifests": manifests})
        put(root, "results/" + name + "/events.jsonl", data)
    parent = {"status": "completed", "artifact_sha256": {"events.jsonl": sha(data)}}
    put(root, "results/parent/generation_manifest.json", parent)
    child = {"status": "completed", "artifact_sha256": {"events.jsonl": sha(data)},
             "parent": {"corpus_name": "old_parent", "artifact_sha256": {"events.jsonl": sha(data)},
                        "generation_manifest_sha256": sha(artifacts.encode(parent))}}
    put(root, "results/child/generation_manifest.json", child)
    primary = {"status": "completed", "artifact_sha256": {"generations.jsonl": sha(data)},
               "generation_count": 2, "config_sha256": sha(config),
               "evaluator_script_sha256": sha(b"print('baseline')\n"),
               "git": {"git_head": "1" * 40, "dirty": True, "untracked_file_count": 2, "dirty_diff_sha256": "2" * 64},
               "dataset": {"path": "/old/host/child_name", "artifact_sha256": {"events.jsonl": sha(data)},
                           "generation_manifest_sha256": sha(artifacts.encode(child)),
                           "parent_generation_manifest_sha256": sha(artifacts.encode(parent))}}
    put(root, "results/primary/generations.jsonl", data)
    put(root, "results/primary/generation_manifest.json", primary)
    primary_manifest = dict(primary, generation_manifest_sha256=sha(artifacts.encode(primary)))
    put(root, "results/primary/manifest.json", primary_manifest)
    put(root, "results/primary/ancillary.md", b"As found\r\n")
    put(root, "results/continual/manifest.json", {
        "status": "completed", "artifact_sha256": {"events.jsonl": sha(data)},
        "dataset": {"path": "results/stream", "sha256": {"events.jsonl": sha(data)}},
        "source": {"git_dirty": True}, "source_sha256": {"experiments/evaluator.py": sha(b"print('baseline')\n")},
        "config_sha256": sha(config), "config_path": "configs/run.json"})
    put(root, "results/interleaved/manifest.json", {
        "status": "completed", "artifacts": ["events.jsonl"],
        "dataset": "results/interleaved_input", "dataset_sha256": {"events.jsonl": sha(data)}})
    put(root, ".gitignore", b"results/**/*.jsonl\n")
    git(root, "add", "--force", ".")
    git(root, "commit", "-m", "fixture")
    revision = git(root, "rev-parse", "HEAD")
    binding = {"dataset": "results/child", "parent": "results/parent",
               "config": "configs/run.json", "evaluator": "experiments/evaluator.py"}
    selection = {"schema_version": 1, "baseline": {"revision": revision, "directories": ["experiments"],
                  "files": ["configs/run.json", "requirements.txt"]}, "evidence": evidence,
                 "relations": {"results/primary/manifest.json": binding,
                               "results/primary/generation_manifest.json": binding,
                               "results/child/generation_manifest.json": {"parent": "results/parent"},
                               "results/continual/manifest.json": {"dataset": "results/stream", "config": "configs/run.json"},
                               "results/interleaved/manifest.json": {"dataset": "results/interleaved_input", "candidate_source": "experiments/evaluator.py"}},
                 "source_candidates": [revision, "1" * 40]}
    return root, selection, tmp_path / "new frozen snapshot"


def test_inventory_copy_and_read_only_verify(bundle):
    root, selection, destination = bundle
    put(root, "experiments/evaluator.py", b"different working tree\r\n")
    put(root, "results/primary/untracked.log", b"not selected")
    inventory = artifacts.inventory(root, selection)
    assert inventory["assessment"]["status"] == "verified"
    entries = {e["path"]: e for e in inventory["entries"]}
    assert entries["results/primary/ancillary.md"]["status"] == "observed_only"
    assert entries["results/primary/generations.jsonl"]["observed"]["jsonl_rows"] == 2
    assert entries["baseline_source/experiments/evaluator.py"]["observed"]["sha256"] == sha(b"print('baseline')\n")
    refs = {(r["path"], r["field"]) for r in inventory["references"]}
    assert ("results/parent/events.jsonl", "parent.artifact_sha256.events.jsonl") in refs
    assert ("results/stream/events.jsonl", "dataset.sha256.events.jsonl") in refs
    assert ("results/interleaved_input/events.jsonl", "dataset_sha256.events.jsonl") in refs
    frozen = artifacts.freeze(root, selection, destination)
    assert frozen["status"] == "verified"
    assert not (destination / ".git").exists()
    assert not (destination / "results/primary/untracked.log").exists()
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in destination.rglob("*") if p.is_file()}
    report = artifacts.verify(destination, frozen["manifest_sha256"], require_trusted=True)
    assert report["status"] == "verified"
    assert report["trust"] == "external_digest_match"
    assert artifacts.verify(destination)["trust"] == "internal_consistency_only"
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert artifacts.verify(destination, require_trusted=True)["status"] != "verified"
    assert artifacts.verify(destination, "0" * 64)["status"] != "verified"
    source = root / "results/primary/generations.jsonl"
    copied = destination / "results/primary/generations.jsonl"
    assert source.read_bytes() == copied.read_bytes()
    assert source.stat().st_ino != copied.stat().st_ino
    root.rename(root.parent / "unavailable repository")
    assert artifacts.verify(destination, frozen["manifest_sha256"])["status"] == "verified"


@pytest.mark.parametrize("damage", ["tamper", "missing", "pointer", "malformed", "verification_status", "entry_status", "overall_status", "remove_reference", "extra"])
def test_verify_rejects_damage(bundle, damage):
    root, selection, destination = bundle
    frozen = artifacts.freeze(root, selection, destination)
    payload = destination / "results/primary/generations.jsonl"
    if damage == "tamper":
        payload.write_bytes(b'{"id":9}\n')
    elif damage == "missing":
        payload.unlink()
    elif damage == "pointer":
        payload.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 100\n")
    elif damage == "malformed":
        payload.write_bytes(b"not json\n")
    elif damage == "verification_status":
        record = artifacts.json_load(destination / "verification.json")
        record["status"] = "failed"
        put(destination, "verification.json", record)
    elif damage == "extra":
        put(destination, "results/primary/unlisted.txt", b"extra")
    else:
        manifest = artifacts.json_load(destination / "freeze_manifest.json")
        if damage == "entry_status":
            manifest["entries"][0]["status"] = "missing"
        elif damage == "overall_status":
            manifest["assessment"]["status"] = "incomplete"
        else:
            manifest["references"].pop()
        put(destination, "freeze_manifest.json", manifest)
    assert artifacts.verify(destination, frozen["manifest_sha256"])["status"] != "verified"
    assert artifacts.verify(destination)["status"] != "verified"


@pytest.mark.parametrize("damage", ["historical_mismatch", "malformed", "pointer", "missing", "missing_expected", "incomplete_run", "row_count", "bad_reference"])
def test_as_found_discrepancies_are_preserved(bundle, damage):
    root, selection, destination = bundle
    source = root / "results/primary/generations.jsonl"
    if damage == "historical_mismatch":
        source.write_bytes(b'{"id":99}\n')
    elif damage == "malformed":
        source.write_bytes(b"{}\nnot json\n")
    elif damage == "pointer":
        source.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 100\n")
    elif damage == "missing":
        source.unlink()
    else:
        manifest = artifacts.json_load(root / "results/primary/manifest.json")
        if damage == "missing_expected":
            manifest["artifact_sha256"]["absent.jsonl"] = "a" * 64
        elif damage == "incomplete_run":
            manifest["status"] = "running"
        elif damage == "row_count":
            manifest["generation_count"] = 999
        else:
            manifest["dataset"]["artifact_sha256"]["events.jsonl"] = "not-a-hash"
        put(root, "results/primary/manifest.json", manifest)
    original = source.read_bytes() if source.exists() else None
    frozen = artifacts.freeze(root, selection, destination)
    assert frozen["status"] != "verified"
    assert artifacts.verify(destination, frozen["manifest_sha256"])["status"] != "verified"
    if original is not None:
        assert source.read_bytes() == original
        assert (destination / "results/primary/generations.jsonl").read_bytes() == original


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/absolute", "a/../../b", "a\\b", "a//b", ".git/config", ".env", "secrets.json", "weights.safetensors", "NUL", "a./b"])
def test_unsafe_relative_paths(name, tmp_path):
    with pytest.raises(ValueError):
        artifacts.safe_path(tmp_path, name)


def test_refuse_existing_overlap_and_low_disk(bundle, monkeypatch):
    root, selection, destination = bundle
    destination.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        artifacts.freeze(root, selection, destination)
    with pytest.raises(ValueError, match="overlaps"):
        artifacts.freeze(root, selection, root / "frozen")
    destination.rmdir()
    monkeypatch.setattr(artifacts.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 0})())
    with pytest.raises(ValueError, match="insufficient disk"):
        artifacts.freeze(root, selection, destination)
    assert not destination.exists()


def test_symlink_escape(bundle):
    root, selection, destination = bundle
    source = root / "results/primary/generations.jsonl"
    outside = put(destination.parent, "outside.jsonl", b"{}\n")
    source.unlink()
    try:
        source.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="symlink/reparse"):
        artifacts.freeze(root, selection, destination)
    assert not destination.exists()


@pytest.mark.parametrize("damage", ["copy", "source", "interrupt"])
def test_interrupted_copy_and_source_mutation(bundle, monkeypatch, damage):
    root, selection, destination = bundle
    original = artifacts.shutil.copyfileobj

    def damaged(source, target, length):
        original(source, target, length)
        if str(source.name).endswith("generations.jsonl"):
            if damage == "copy":
                target.write(b"{}\n")
            elif damage == "source":
                Path(source.name).write_bytes(b"{}\n")
            else:
                raise OSError("simulated interruption")
    monkeypatch.setattr(artifacts.shutil, "copyfileobj", damaged)
    frozen = artifacts.freeze(root, selection, destination)
    assert frozen["status"] != "verified"
    manifest = artifacts.json_load(destination / "freeze_manifest.json")
    assert manifest["copy_issues"]
    assert artifacts.verify(destination, frozen["manifest_sha256"])["status"] != "verified"


def test_windows_junction_escape(bundle):
    if artifacts.os.name != "nt":
        pytest.skip("Windows junction test")
    root, selection, destination = bundle
    directory = root / "results/primary"
    outside = root.parent / "outside primary"
    directory.rename(outside)
    result = subprocess.run(["cmd", "/c", "mklink", "/J", str(directory), str(outside)], capture_output=True)
    if result.returncode:
        pytest.skip("junction creation unavailable")
    with pytest.raises(ValueError, match="symlink/reparse"):
        artifacts.freeze(root, selection, destination)
    assert not destination.exists()


def test_null_expected_hash_is_incomplete(bundle):
    root, selection, destination = bundle
    manifest = artifacts.json_load(root / "results/interleaved/manifest.json")
    manifest["dataset_sha256"]["events.jsonl"] = None
    put(root, "results/interleaved/manifest.json", manifest)
    result = artifacts.inventory(root, selection)
    assert result["assessment"]["status"] == "incomplete"
    assert any("malformed expected" in issue for issue in result["assessment"]["issues"])


def test_lfs_index_identity_is_checked(bundle):
    root, selection, _ = bundle
    data = b'{"id":1}\r\n{"id":2}\r\n'
    source = root / "results/stream/events.jsonl"
    source.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:" + sha(data).encode() + b"\nsize " + str(len(data)).encode() + b"\n")
    git(root, "add", "--force", "results/stream/events.jsonl")
    source.write_bytes(data)
    result = artifacts.inventory(root, selection)
    entry = next(e for e in result["entries"] if e["path"] == "results/stream/events.jsonl")
    assert entry["lfs_identity"] == {"oid_sha256": sha(data), "size": len(data)}
    assert entry["status"] == "historical_match"
    source.write_bytes(b"{}\n")
    result = artifacts.inventory(root, selection)
    entry = next(e for e in result["entries"] if e["path"] == "results/stream/events.jsonl")
    assert entry["status"] == "lfs_mismatch"


def test_source_audit_uses_blobs_and_exposes_gaps(bundle):
    root, selection, _ = bundle
    put(root, "experiments/evaluator.py", b"dirty checkout\r\n")
    report = artifacts.source_audit(root, selection, path_history=True)
    assert "1" * 40 in report["unavailable_candidates"]
    primary = next(r for r in report["runs"] if r["manifest"] == "results/primary/manifest.json")
    assert all(f["status"] == "hash_verified" for f in primary["files"])
    assert primary["exact_execution_source"] is False
    assert primary["status"] == "unresolved"
    assert any("unavailable" in gap for gap in primary["gaps"])
    assert any("untracked" in gap for gap in primary["gaps"])
    interleaved = next(r for r in report["runs"] if r["manifest"] == "results/interleaved/manifest.json")
    assert interleaved["files"][0]["status"] == "candidate_recovered"
    assert any("no recorded per-file" in gap for gap in interleaved["gaps"])


def test_git_evidence_uses_pinned_blobs_not_crlf_checkout_or_index(bundle):
    root, selection, destination = bundle
    put(root, "results/primary/ancillary.md", b"Pinned report\n")
    git(root, "add", "results/primary/ancillary.md")
    git(root, "commit", "-m", "canonical report")
    revision = git(root, "rev-parse", "HEAD")
    selection["baseline"]["revision"] = revision
    for group in selection["evidence"]:
        for name in group["manifests"]:
            path = root / group["path"] / name
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    put(root, "results/primary/ancillary.md", b"Pinned report\r\n")
    default = artifacts.inventory(root, selection)
    explicit = artifacts.inventory(root, selection, evidence_source="worktree")
    assert default == explicit
    assert default["assessment"]["status"] == "failed"
    asfound = artifacts.freeze(root, selection, destination)
    assert asfound["status"] == "failed"
    assert (destination / "results/primary/ancillary.md").read_bytes() == b"Pinned report\r\n"
    put(root, "results/primary/manifest.json", b"not a manifest\n")
    put(root, "results/primary/after-baseline.txt", b"new index file\n")
    git(root, "add", "results/primary/manifest.json", "results/primary/after-baseline.txt")
    original = (root / "results/primary/manifest.json").read_bytes()
    pinned = artifacts.inventory(root, selection, evidence_source="git")
    assert pinned["assessment"]["status"] == "verified"
    assert pinned["evidence_source"] == "git"
    assert "results/primary/after-baseline.txt" not in {e["path"] for e in pinned["entries"]}
    entry = next(e for e in pinned["entries"] if e["path"] == "results/primary/ancillary.md")
    assert entry["origin"]["kind"] == "git_blob"
    assert entry["origin"]["revision"] == revision
    assert entry["origin"]["oid"] == git(root, "rev-parse", revision + ":results/primary/ancillary.md")
    second = destination.parent / "second git snapshot"
    frozen = artifacts.freeze(root, selection, second, evidence_source="git")
    assert frozen["status"] == "verified"
    assert (second / "results/primary/ancillary.md").read_bytes() == b"Pinned report\n"
    assert (root / "results/primary/manifest.json").read_bytes() == original
    assert artifacts.verify(second, frozen["manifest_sha256"], require_trusted=True)["status"] == "verified"


def test_git_evidence_lfs_requires_pinned_materialized_identity(bundle):
    root, selection, destination = bundle
    name = "results/stream/events.jsonl"
    source = root / name
    data = source.read_bytes()
    lfs = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + sha(data).encode() + b"\nsize " + str(len(data)).encode() + b"\n"
    source.write_bytes(lfs)
    git(root, "add", "--force", name)
    git(root, "commit", "-m", "LFS pointer fixture")
    revision = git(root, "rev-parse", "HEAD")
    selection["baseline"]["revision"] = revision
    source.write_bytes(data)
    inventory = artifacts.inventory(root, selection, evidence_source="git")
    entry = next(e for e in inventory["entries"] if e["path"] == name)
    assert entry["origin"]["kind"] == "lfs_worktree"
    assert entry["origin"]["revision"] == revision
    assert entry["origin"]["materialized_path"] == name
    assert entry["lfs_identity"] == {"oid_sha256": sha(data), "size": len(data)}
    frozen = artifacts.freeze(root, selection, destination, evidence_source="git")
    assert frozen["status"] == "verified"
    assert (destination / name).read_bytes() == data
    for damaged in (data.replace(b"1", b"9"), data + b"{}\n", lfs, None):
        if damaged is None:
            source.unlink()
        else:
            source.write_bytes(damaged)
        refused = destination.parent / "refused LFS snapshot"
        with pytest.raises(ValueError, match="matching materialized LFS"):
            artifacts.freeze(root, selection, refused, evidence_source="git")
        assert not refused.exists()


def test_audit_report_and_source_exports_are_exclusive_and_labeled(bundle):
    root, selection, destination = bundle
    baseline = selection["baseline"]["revision"]
    put(root, "experiments/evaluator.py", b"print('candidate')\n")
    put(root, "experiments/helper.py", b"VALUE = 1\n")
    put(root, "legacy/credentials.txt", b"excluded fixture\n")
    git(root, "add", "experiments", "legacy")
    git(root, "commit", "-m", "candidate source")
    candidate = git(root, "rev-parse", "HEAD")
    selection["source_candidates"].append(candidate)
    put(root, "experiments/evaluator.py", b"dirty working tree\r\n")
    output = destination.parent / "source_recovery.json"
    recovery = destination.parent / "source recovery"
    report = artifacts.source_audit(root, selection, output=output, export_sources_directory=recovery)
    assert artifacts.json_load(output) == report
    exported = report["source_export"]
    manifest = artifacts.json_load(recovery / exported["manifest"])
    assert sha((recovery / exported["manifest"]).read_bytes()) == exported["manifest_sha256"]
    assert manifest["exact_execution_source"] is False
    snapshots = {s["revision"]: s for s in manifest["snapshots"]}
    assert snapshots[baseline]["classification"] == "baseline_snapshot"
    assert snapshots[candidate]["classification"] == "candidate_recovered"
    assert (recovery / "matches" / baseline / "experiments/evaluator.py").read_bytes() == b"print('baseline')\n"
    assert (recovery / "candidates" / candidate / "experiments/helper.py").read_bytes() == b"VALUE = 1\n"
    assert not any("legacy" in p.parts or ".git" in p.parts for p in recovery.rglob("*"))
    before = output.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        artifacts.source_audit(root, selection, output=output, export_sources_directory=destination)
    assert not destination.exists()
    with pytest.raises(ValueError, match="already exists"):
        artifacts.source_audit(root, selection, export_sources_directory=recovery)
    assert output.read_bytes() == before


@pytest.mark.parametrize("marker", ["freeze_manifest.json", "verification.json", "in_progress"])
def test_audit_output_cannot_enter_frozen_parent(bundle, marker):
    root, selection, destination = bundle
    if marker == "in_progress":
        (destination / "results").mkdir(parents=True)
        (destination / "baseline_source").mkdir()
    else:
        put(destination, marker, {})
    nested = destination / "nested"
    nested.mkdir()
    for argument in ("output", "export_sources_directory"):
        with pytest.raises(ValueError, match="inside a frozen snapshot"):
            artifacts.source_audit(root, selection, **{argument: nested / "new"})
        assert not (nested / "new").exists()
    with pytest.raises(ValueError, match="overlaps input"):
        artifacts.source_audit(root, selection, output=root / "new-report.json")


def test_audit_output_only_and_missing_candidate_files(bundle):
    root, selection, destination = bundle
    older = selection["baseline"]["revision"]
    put(root, "tests/test_new.py", b"def test_new():\n    assert True\n")
    git(root, "add", "tests/test_new.py")
    git(root, "commit", "-m", "new baseline test")
    selection["baseline"]["revision"] = git(root, "rev-parse", "HEAD")
    selection["baseline"]["files"].append("tests/test_new.py")
    output = destination.parent / "audit-only.json"
    report = artifacts.source_audit(root, selection, output=output)
    assert artifacts.json_load(output) == report
    assert "source_export" not in report
    report = artifacts.source_audit(root, selection, export_sources_directory=destination)
    snapshots = {s["revision"]: s for s in report["source_export"]["snapshots"]}
    assert snapshots[older]["missing_allowlisted_paths"] == ["tests/test_new.py"]
    assert snapshots[selection["baseline"]["revision"]]["missing_allowlisted_paths"] == []


def test_verify_hashes_the_same_manifest_bytes_it_parses(bundle, monkeypatch):
    root, selection, destination = bundle
    frozen = artifacts.freeze(root, selection, destination)
    original = artifacts.json_load

    def refuse_separate_manifest_read(path):
        if Path(path).name == "freeze_manifest.json":
            raise AssertionError("manifest must be parsed from the already-hashed bytes")
        return original(path)

    monkeypatch.setattr(artifacts, "json_load", refuse_separate_manifest_read)
    assert artifacts.verify(destination, frozen["manifest_sha256"], require_trusted=True)["status"] == "verified"


def test_new_cli_options(bundle, capsys):
    root, selection, destination = bundle
    config = put(destination.parent, "selection.json", selection)
    base = ["--repo", str(root), "--selection", str(config)]
    assert artifacts.main(["inventory", *base, "--evidence-source", "git"]) == 0
    assert json.loads(capsys.readouterr().out)["evidence_source"] == "git"
    assert artifacts.main(["freeze", *base, "--evidence-source", "git", "--destination", str(destination)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    output = destination.parent / "report.json"
    recovery = destination.parent / "recovery"
    assert artifacts.main(["source-audit", *base, "--output", str(output), "--export-sources", str(recovery)]) == 0
    assert json.loads(capsys.readouterr().out) == artifacts.json_load(output)


def test_cli_status_codes(bundle, capsys):
    root, selection, destination = bundle
    config = put(destination.parent, "selection.json", selection)
    assert artifacts.main(["inventory", "--repo", str(root), "--selection", str(config)]) == 0
    assert json.loads(capsys.readouterr().out)["assessment"]["status"] == "verified"
    assert artifacts.main(["freeze", "--repo", str(root), "--selection", str(config), "--destination", str(destination)]) == 0
    digest = json.loads(capsys.readouterr().out)["manifest_sha256"]
    assert artifacts.main(["verify", str(destination), "--require-trusted", "--expected-manifest-sha256", digest]) == 0
    capsys.readouterr()
    assert artifacts.main(["verify", str(destination), "--require-trusted"]) == 1
    capsys.readouterr()
    assert artifacts.main(["freeze", "--repo", str(root), "--selection", str(config), "--destination", str(destination)]) == 2
