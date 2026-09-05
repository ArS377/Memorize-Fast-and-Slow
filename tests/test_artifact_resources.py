import copy
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace

import pytest

from scripts import artifact_resources as resources
from scripts import paper_artifacts


ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "repo shaped root"
    directory = root / "results" / "input"
    directory.mkdir(parents=True)
    payload = b'{"id":1}\r\n'
    (directory / "data.jsonl").write_bytes(payload)
    manifest = json.dumps({"artifact_sha256": {"data.jsonl": sha(payload)}}).encode()
    (directory / "generation_manifest.json").write_bytes(manifest)
    gate = b'{"status":"completed"}\n'
    (root / "results" / "gate.json").write_bytes(gate)
    index = {
        "schema_version": 1,
        "groups": {"all": ["input", "gate"]},
        "resources": {
            "input": {
                "path": "results/input", "kind": "directory",
                "manifest": {"path": "generation_manifest.json", "sha256": sha(manifest), "hash_field": "artifact_sha256"},
                "required_members": ["data.jsonl"],
            },
            "gate": {"path": "results/gate.json", "kind": "file", "sha256": sha(gate)},
        },
    }
    return root, index


def test_bundled_exact_inputs_without_git_or_writes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("read-only resource verification must not dispatch subprocesses")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    report = resources.access(ROOT, "bundled-persona")
    assert report["status"] == "verified", report["errors"]
    assert report["file_count"] == 41
    assert sum(entry["bytes"] for entry in report["checked_files"].values()) == 14168913
    assert len(report["paths"]) == 5
    index = resources.load_index()
    assert index["groups"]["kimi_a"] == index["groups"]["kimi_ab"]
    assert "persona-kimi-b" in index["groups"]["kimi_a"]


def test_repo_shaped_access_and_standalone_gate(bundle):
    root, index = bundle
    report = resources.access(root, "all", index)
    assert report["status"] == "verified"
    assert report["file_count"] == 3
    assert report["paths"]["gate"] == str(root / "results/gate.json")
    assert not (root / ".git").exists()
    assert resources.verify(root / "results", "all", index)["status"] == "failed"


@pytest.mark.parametrize("mutation", ["tamper", "missing", "extra", "directory", "manifest", "gate"])
def test_fail_closed_without_access_paths(bundle, mutation):
    root, index = bundle
    directory = root / "results/input"
    if mutation == "tamper":
        (directory / "data.jsonl").write_bytes(b'{"id":2}\r\n')
    elif mutation == "missing":
        (directory / "data.jsonl").unlink()
    elif mutation == "extra":
        (directory / "unlisted.json").write_bytes(b"{}")
    elif mutation == "directory":
        (directory / "nested").mkdir()
    elif mutation == "manifest":
        (directory / "generation_manifest.json").write_bytes(b"{}")
    else:
        (root / "results/gate.json").write_bytes(b"{}")
    report = resources.access(root, "all", index)
    assert report["status"] == "failed"
    assert report["errors"]
    assert "paths" not in report


def test_external_members_need_no_freeze_or_git(bundle):
    root, index = bundle
    directory = root / "results/input"
    manifest = json.loads((directory / "generation_manifest.json").read_bytes())
    entry = index["resources"]["input"]
    entry["members"] = manifest["artifact_sha256"]
    entry["members"]["generation_manifest.json"] = entry.pop("manifest")["sha256"]
    entry.pop("required_members")
    assert resources.verify(root, "input", index)["status"] == "verified"


def test_lfs_pointer_is_never_a_payload(bundle):
    root, index = bundle
    pointer = b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 10\n"
    (root / "results/gate.json").write_bytes(pointer)
    index["resources"]["gate"]["sha256"] = sha(pointer)
    report = resources.access(root, "gate", index)
    assert report["status"] == "failed"
    assert "LFS pointer" in report["errors"][0]


@pytest.mark.parametrize("path", ["../outside", "C:/outside", "results/../escape", "results/con.json", "results/x:stream", "baseline_source/source.py", "results\\input"])
def test_unsafe_resource_paths_rejected(bundle, path):
    root, index = bundle
    index["resources"]["input"]["path"] = path
    with pytest.raises(ValueError):
        resources.access(root, "all", index)


def test_overlapping_resources_rejected(bundle):
    root, index = bundle
    index["resources"]["gate"]["path"] = "results/input/gate.json"
    with pytest.raises(ValueError, match="overlapping"):
        resources.access(root, "all", index)


def test_case_colliding_members_rejected(bundle):
    root, index = bundle
    index["resources"]["input"]["required_members"].append("DATA.jsonl")
    with pytest.raises(ValueError, match="case-colliding"):
        resources.verify(root, "input", index)


@pytest.mark.parametrize("link_kind", ["symlink", "reparse", "hardlink"])
def test_link_checks_without_windows_link_privileges(bundle, monkeypatch, link_kind):
    root, index = bundle
    target = root / "results/gate.json"
    original_lstat = Path.lstat
    original_stat = Path.stat
    if link_kind == "hardlink":
        def fake_stat(path, *args, **kwargs):
            if path == target:
                return SimpleNamespace(st_mode=stat.S_IFREG, st_nlink=2)
            return original_stat(path, *args, **kwargs)
        monkeypatch.setattr(Path, "stat", fake_stat)
    else:
        def fake_lstat(path, *args, **kwargs):
            if path == target:
                return SimpleNamespace(st_mode=stat.S_IFLNK if link_kind == "symlink" else stat.S_IFREG,
                                       st_file_attributes=0x400 if link_kind == "reparse" else 0)
            return original_lstat(path, *args, **kwargs)
        monkeypatch.setattr(Path, "lstat", fake_lstat)
    report = resources.access(root, "gate", index)
    assert report["status"] == "failed"
    assert "paths" not in report


def test_unknown_or_missing_root_has_no_fallback(bundle):
    root, index = bundle
    with pytest.raises(ValueError, match="unknown"):
        resources.verify(root, "unknown", index)
    with pytest.raises(ValueError, match="not a directory"):
        resources.verify(root / "absent", "all", index)


@pytest.mark.parametrize("command", ["inventory", "freeze", "source-audit"])
@pytest.mark.parametrize("flags", [[], ["--repo", "."], ["--selection", "configs/paper_artifacts.json"]])
def test_maintenance_requires_both_explicit_inputs(command, flags):
    extra = ["--destination", "unused"] if command == "freeze" else []
    with pytest.raises(SystemExit) as error:
        paper_artifacts.main([command, *flags, *extra])
    assert error.value.code == 2


def test_cli_list_and_access(capsys):
    assert resources.main(["list"]) == 0
    index = json.loads(capsys.readouterr().out)
    assert "not yet configured" in index["access_policy"]
    assert resources.main(["access", "persona-pair-gate", "--root", str(ROOT)]) == 0
    assert json.loads(capsys.readouterr().out)["paths"]["persona-pair-gate"] == str(ROOT / "results/persona_surface_pair_gate.json")
    with pytest.raises(SystemExit) as error:
        resources.main(["verify", "bundled-persona"])
    assert error.value.code == 2


def test_portable_historical_config():
    config = paper_artifacts.json_load(ROOT / "configs/paper_artifacts.json")
    assert config["freeze_root"] == "external-artifacts"
    index = resources.load_index()
    for snapshot in config["snapshots"]:
        assert snapshot["manifest_sha256"] == index["historical_provenance"][snapshot["directory"]]["manifest_sha256"]
    assert sum(len(entry["members"]) for entry in index["resources"].values() if "members" in entry) == 36


def test_invalid_digest_and_duplicate_json_rejected(bundle, tmp_path):
    root, index = bundle
    bad = copy.deepcopy(index)
    bad["resources"]["gate"]["sha256"] = "invalid"
    with pytest.raises(ValueError, match="SHA256"):
        resources.access(root, "gate", bad)
    path = tmp_path / "duplicate.json"
    path.write_bytes(b'{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate JSON"):
        resources.load_index(path)
