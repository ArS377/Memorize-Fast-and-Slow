from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from conftest import git_fixture_env

from neurosym.application.source_provenance import (
    ensure_output_directory,
    git_provenance,
    main,
    source_manifest,
    source_provenance,
    verify_source_manifest,
    write_source_manifest,
)


@pytest.fixture(autouse=True)
def clean_provenance_environment(monkeypatch):
    monkeypatch.delenv("NEUROSYM_SOURCE_MANIFEST", raising=False)
    monkeypatch.delenv("NEUROSYM_SOURCE_MANIFEST_SHA256", raising=False)
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)


def put(root, name, content=b"value = 1\n"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "portable archive with spaces"
    root.mkdir()
    put(root, "experiments/run.py")
    put(root, "neurosym/application/helper.py")
    put(root, "services/service.py")
    put(root, "scripts/tool.py")
    put(root, "configs/benchmark.json", b'{"seed": 1}\n')
    put(root, "experiments/scoring.py")
    put(root, "compiled_memory.py")
    return root


def signed_manifest(root):
    path = root / "source_manifest.json"
    return path, write_source_manifest(root, path)


def rewrite(path, manifest):
    payload = json.dumps(manifest).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", *args],
        cwd=root, check=True, capture_output=True, timeout=30, env=git_fixture_env()
    ).stdout


@pytest.fixture
def git_root(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("Git is not installed")
    root = tmp_path / "git fixture with spaces"
    root.mkdir()
    git(root, "init")
    put(root, "source.py")
    put(root, "results/tracked.json", b"{}\n")
    put(root, "custom output/tracked.json", b"{}\n")
    git(root, "add", ".")
    git(root, "commit", "-m", "fixture baseline")
    return root


def test_git_clean_identity_has_exact_legacy_fields(git_root):
    expected = {
        "git_head": git(git_root, "rev-parse", "HEAD").decode("ascii").strip(),
        "dirty": False,
        "dirty_diff_sha256": hashlib.sha256(b"tracked-diff\0").hexdigest(),
        "untracked_file_count": 0,
    }
    assert git_provenance(git_root / "results") == expected
    assert source_provenance(git_root) == {"mode": "git", "git": expected}


def test_git_preserves_tracked_binary_and_sorted_untracked_hash(git_root):
    put(git_root, "source.py", b"value = 2\n")
    git(git_root, "add", "source.py")
    put(git_root, "source.py", b"value = 3\n")
    put(git_root, "a new.py", b"a = 10\n")
    put(git_root, "z.bin", b"\0\xff\x10")
    put(git_root, "results/tracked.json", b"changed")
    put(git_root, "results/untracked.py")
    put(git_root, "custom output/tracked.json", b"changed")
    put(git_root, "custom output/untracked.py")
    diff = git(
        git_root, "diff", "--binary", "HEAD", "--", ".",
        ":(exclude)results", ":(exclude)results/**",
        ":(exclude)custom output", ":(exclude)custom output/**",
    )
    expected = hashlib.sha256(b"tracked-diff\0" + diff)
    for name in ("a new.py", "z.bin"):
        expected.update(b"\0untracked\0" + name.encode() + b"\0")
        expected.update((git_root / name).read_bytes())
    identity = git_provenance(git_root, excluded_untracked_dir=git_root / "custom output")
    assert identity == {
        "git_head": git(git_root, "rev-parse", "HEAD").decode().strip(),
        "dirty": True,
        "dirty_diff_sha256": expected.hexdigest(),
        "untracked_file_count": 2,
    }


def test_git_excluded_changes_do_not_dirty_identity(git_root):
    before = git_provenance(git_root, excluded_untracked_dir=git_root / "custom output")
    for folder in ("results", "custom output"):
        put(git_root, f"{folder}/tracked.json", b"modified")
        put(git_root, f"{folder}/new.py")
    assert git_provenance(git_root, excluded_untracked_dir=git_root / "custom output") == before


def test_no_git_requires_explicit_archive(archive):
    with pytest.raises(ValueError, match="cannot resolve git provenance"):
        source_provenance(archive)


def test_manifest_deterministic_sorted_and_portable(archive, tmp_path):
    path, digest = signed_manifest(archive)
    manifest = source_manifest(archive)
    assert json.loads(path.read_bytes()) == manifest
    names = [entry["path"] for entry in manifest["files"]]
    assert names == sorted(names)
    assert len(names) == 7
    assert all(not Path(name).is_absolute() and "\\" not in name for name in names)
    assert str(archive) not in path.read_text()
    relocated = tmp_path / "relocated source with spaces"
    shutil.copytree(archive, relocated)
    assert source_manifest(relocated) == manifest
    result = source_provenance(
        relocated, archive_manifest="source_manifest.json", archive_manifest_sha256=digest
    )
    assert result == {
        "mode": "archive",
        "archive": {"manifest_version": 1, "manifest_sha256": digest, "source_file_count": 7},
    }
    second = relocated / "second_manifest.json"
    assert write_source_manifest(relocated, second) == digest
    assert second.read_bytes() == path.read_bytes()
    assert "git" not in result and "git_head" not in result["archive"]


@pytest.mark.parametrize("name", ["scripts/run.sh", "requirements.txt", "requirements-graphiti-baseline.txt"])
def test_archive_binds_launchers_and_dependency_specs(archive, name):
    put(archive, name, b"original\n")
    path, digest = signed_manifest(archive)
    assert name in {entry["path"] for entry in json.loads(path.read_bytes())["files"]}
    put(archive, name, b"changed\n")
    with pytest.raises(ValueError, match="source file"):
        verify_source_manifest(archive, path, digest)


@pytest.mark.platform_specific
def test_output_hardlinks_cannot_modify_reference_bytes(tmp_path):
    reference = tmp_path / "reference.jsonl"
    reference.write_bytes(b"original\n")
    output = tmp_path / "output"
    output.mkdir()
    try:
        os.link(reference, output / "predictions.jsonl")
    except OSError as error:
        pytest.skip(f"hardlinks unavailable: {error}")
    with pytest.raises(ValueError, match="hard-linked"):
        ensure_output_directory(output, allow_resume=True)
    assert reference.read_bytes() == b"original\n"


def test_paper_selection_protects_original_reference_directories(tmp_path):
    from neurosym.application.source_provenance import paper_evidence_roots

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "paper_artifacts.json").write_text(
        json.dumps({"evidence": [{"path": "results/reference"}]}), encoding="utf-8"
    )
    protected = paper_evidence_roots(tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        ensure_output_directory(tmp_path / "results/reference", frozen_roots=protected, allow_resume=True)
    assert ensure_output_directory(tmp_path / "results/new-run", frozen_roots=protected)


@pytest.mark.parametrize("name", ["run_graphiti_baseline.sh", "supervise_joint_run.sh"])
@pytest.mark.platform_specific
def test_shell_driver_syntax_for_lf_export(name):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is unavailable for syntax-only verification")
    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / name).read_text(encoding="utf-8").encode("utf-8")
    result = subprocess.run([bash, "-n"], input=source, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert not result.stderr, result.stderr.decode(errors="replace")


def test_environment_archive_mode_never_executes_git(archive, monkeypatch):
    path, digest = signed_manifest(archive)
    monkeypatch.setenv("NEUROSYM_SOURCE_MANIFEST", path.name)
    monkeypatch.setenv("NEUROSYM_SOURCE_MANIFEST_SHA256", digest)

    def forbidden(*args, **kwargs):
        pytest.fail("archive verification must not invoke Git")

    monkeypatch.setattr(subprocess, "run", forbidden)
    assert source_provenance(archive)["mode"] == "archive"
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="external digest"):
        source_provenance(archive)


@pytest.mark.parametrize("env", [
    {"NEUROSYM_SOURCE_MANIFEST": "source_manifest.json"},
    {"NEUROSYM_SOURCE_MANIFEST_SHA256": "0" * 64},
    {"NEUROSYM_SOURCE_MANIFEST": "", "NEUROSYM_SOURCE_MANIFEST_SHA256": ""},
])
def test_incomplete_archive_configuration_does_not_fall_back(git_root, monkeypatch, env):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="requires both"):
        source_provenance(git_root)


@pytest.mark.parametrize("digest", [None, "", "invalid", "0" * 64])
def test_external_digest_is_required_and_checked(archive, digest):
    path, _ = signed_manifest(archive)
    with pytest.raises(ValueError, match="digest"):
        source_provenance(archive, archive_manifest=path, archive_manifest_sha256=digest)


@pytest.mark.parametrize("change", ["same_size", "different_size", "missing", "extra_python", "extra_config"])
def test_archive_source_tampering_is_rejected(archive, change):
    path, digest = signed_manifest(archive)
    if change == "same_size":
        put(archive, "experiments/run.py", b"value = 2\n")
    elif change == "different_size":
        put(archive, "experiments/run.py", b"value = 10000\n")
    elif change == "missing":
        (archive / "experiments/run.py").unlink()
    elif change == "extra_python":
        put(archive, "neurosym/new_package/new_module.py")
    else:
        put(archive, "configs/new.json", b"{}")
    with pytest.raises(ValueError, match="source (file|closure)"):
        verify_source_manifest(archive, path, digest)


@pytest.mark.parametrize("name", [
    "../outside.py", "/outside.py", "C:/outside.py", "C:outside.py",
    "experiments/../outside.py", "experiments//run.py", "experiments/./run.py",
    "experiments\\run.py", "//server/share/code.py", "experiments/run.py:stream",
    "experiments/nul.py", "experiments/run.py.", "experiments/run\x00.py",
    "results/run.py", "configs/credentials.json", "unknown.py",
])
def test_unsafe_or_undeclared_manifest_paths_rejected(archive, name):
    path, _ = signed_manifest(archive)
    manifest = json.loads(path.read_bytes())
    manifest["files"][0]["path"] = name
    digest = rewrite(path, manifest)
    with pytest.raises(ValueError, match="unsafe source path|undeclared source"):
        verify_source_manifest(archive, path, digest)


@pytest.mark.parametrize("change", ["version", "boolean_version", "scope", "unsorted", "duplicate", "size", "sha", "extra_field"])
def test_invalid_manifest_schema_rejected(archive, change):
    path, _ = signed_manifest(archive)
    manifest = json.loads(path.read_bytes())
    if change == "version":
        manifest["version"] = 2
    elif change == "boolean_version":
        manifest["version"] = True
    elif change == "scope":
        manifest["scope"]["directories"].remove("neurosym")
    elif change == "unsorted":
        manifest["files"].reverse()
    elif change == "duplicate":
        manifest["files"].insert(0, manifest["files"][0])
    elif change == "size":
        manifest["files"][0]["size_bytes"] = True
    elif change == "sha":
        manifest["files"][0]["sha256"] = "not a digest"
    else:
        manifest["unexpected"] = 1
    digest = rewrite(path, manifest)
    with pytest.raises(ValueError):
        verify_source_manifest(archive, path, digest)


@pytest.mark.parametrize("raw", [b'{"version": 1, "version": 1}', b"not JSON", b"\xff"])
def test_malformed_manifest_fails_with_valid_external_hash(archive, raw):
    path = archive / "source_manifest.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="invalid source manifest"):
        verify_source_manifest(archive, path, hashlib.sha256(raw).hexdigest())


def test_outputs_caches_tests_credentials_and_arbitrary_trees_not_read(archive, monkeypatch):
    before = source_manifest(archive)
    ignored_names = (
        "results/generated.py", "custom output/generated.py", "tests/test_generated.py",
        "neurosym/__pycache__/generated.py", "experiments/outputs/generated.py",
        "services/.venv/dependency.py", "services/cache/generated.json",
        "configs/credentials.json", "configs/secrets.json", "configs/credentials_dev.json", ".env",
        "unrelated/private/credentials.json", ".git/config",
    )
    ignored = {put(archive, name, b"DO NOT READ") for name in ignored_names}
    original = Path.open

    def checked_open(self, *args, **kwargs):
        assert self not in ignored, f"read excluded file: {self}"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", checked_open)
    assert source_manifest(archive, output_dir=archive / "custom output") == before
    path, digest = signed_manifest(archive)
    assert verify_source_manifest(archive, path, digest, output_dir="custom output")["source_file_count"] == 7


@pytest.mark.parametrize("output", [".", "experiments", "configs/generated", "neurosym/application"])
def test_output_exclusion_cannot_hide_source(archive, output):
    with pytest.raises(ValueError, match="overlaps"):
        source_manifest(archive, output_dir=output)


def test_credential_named_python_is_still_source(archive):
    path, digest = signed_manifest(archive)
    put(archive, "neurosym/secrets.py")
    with pytest.raises(ValueError, match="source closure differs"):
        verify_source_manifest(archive, path, digest)


def test_invalid_archive_manifest_never_falls_back_to_git(git_root):
    path = put(git_root, "source_manifest.json", b"{}")
    with pytest.raises(ValueError, match="external digest"):
        source_provenance(git_root, archive_manifest=path, archive_manifest_sha256="0" * 64)


def test_unlisted_root_python_is_rejected_without_reading(archive):
    put(archive, "unexpected.py", b"untracked root source")
    with pytest.raises(ValueError, match="undeclared source"):
        source_manifest(archive)


def test_manifest_cannot_overwrite_or_include_itself(archive):
    path, _ = signed_manifest(archive)
    with pytest.raises(FileExistsError):
        write_source_manifest(archive, path)
    with pytest.raises(ValueError, match="overlaps declared source"):
        write_source_manifest(archive, "configs/source_manifest.json")
    with pytest.raises(ValueError, match="root Python"):
        write_source_manifest(archive, "source_manifest.py")


@pytest.mark.parametrize("kind", ["file", "directory", "root", "manifest"])
@pytest.mark.platform_specific
def test_archive_rejects_symlinks(archive, tmp_path, kind):
    path, digest = signed_manifest(archive)
    target = tmp_path / "outside"
    target.mkdir()
    put(target, "run.py")
    link = archive / "experiments/linked.py"
    destination = target / "run.py"
    directory = False
    if kind == "directory":
        link = archive / "experiments/linked"
        destination = target
        directory = True
    elif kind == "root":
        link = tmp_path / "linked root"
        destination = archive
        directory = True
    elif kind == "manifest":
        link = archive / "linked_manifest.json"
        destination = path
    try:
        link.symlink_to(destination, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="symlink or junction"):
        verify_source_manifest(
            link if kind == "root" else archive,
            link if kind == "manifest" else path,
            digest,
        )


@pytest.mark.platform_specific
@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_archive_rejects_junction_escape(archive, tmp_path):
    path, digest = signed_manifest(archive)
    outside = tmp_path / "outside source"
    outside.mkdir()
    put(outside, "extra.py")
    junction = archive / "experiments/junction"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True, timeout=30,
    )
    if result.returncode:
        pytest.skip(f"junction creation unavailable: {result.stderr!r}")
    try:
        with pytest.raises(ValueError, match="symlink or junction"):
            verify_source_manifest(archive, path, digest)
        with pytest.raises(ValueError, match="overlaps input"):
            ensure_output_directory(junction / "output", [outside])
        put(outside, "freeze_manifest.json", b"{}")
        with pytest.raises(ValueError, match="frozen bundle|symlink or junction"):
            ensure_output_directory(junction / "output", allow_resume=True)
    finally:
        junction.rmdir()


@pytest.mark.parametrize("kind", ["same", "child", "parent"])
def test_output_input_overlap_rejected_in_both_directions(tmp_path, kind):
    source = tmp_path / "input"
    source.mkdir()
    output = {"same": source, "child": source / "new", "parent": tmp_path}[kind]
    with pytest.raises(ValueError, match="overlaps input"):
        ensure_output_directory(output, [source])


def test_output_validation_is_absolute_and_has_no_write_side_effect(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output = ensure_output_directory("fresh output", [tmp_path / "inputs"])
    assert output == tmp_path / "fresh output"
    assert not output.exists()
    with pytest.raises(ValueError, match="not a directory"):
        ensure_output_directory(put(tmp_path, "file"))


def test_frozen_sentinel_ancestor_cannot_be_bypassed_by_resume(tmp_path):
    frozen = tmp_path / "frozen"
    put(frozen, "freeze_manifest.json", b"{}")
    for allow_resume in (False, True):
        with pytest.raises(ValueError, match="frozen bundle"):
            ensure_output_directory(frozen / "run/nested/output", allow_resume=allow_resume)
    with pytest.raises(ValueError, match="frozen root"):
        ensure_output_directory(tmp_path, frozen_roots=[frozen])


def test_manifest_creation_refuses_frozen_bundle(archive):
    put(archive, "freeze_manifest.json", b"{}")
    with pytest.raises(ValueError, match="frozen bundle"):
        write_source_manifest(archive, "source_manifest.json")


def test_completed_output_refused_but_persona_resume_can_be_checked_by_caller(tmp_path):
    output = tmp_path / "run"
    put(output, "manifest.json", b'{"status": "completed"}')
    for path in (output, output / "nested"):
        with pytest.raises(ValueError, match="completed output"):
            ensure_output_directory(path)
        assert ensure_output_directory(path, allow_resume=True) == path
    put(output, "manifest.json", b"invalid")
    with pytest.raises(ValueError, match="cannot validate"):
        ensure_output_directory(output, allow_resume=True)


@pytest.mark.platform_specific
def test_output_symlink_cannot_hide_input_overlap(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    link = tmp_path / "alias"
    try:
        link.symlink_to(source, target_is_directory=True)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="overlaps input"):
        ensure_output_directory(link / "output", [source])


def test_cli_create_and_verify(archive, capsys):
    assert main(["--create", str(archive), "--output", "source_manifest.json"]) == 0
    digest = json.loads(capsys.readouterr().out)["manifest_sha256"]
    assert main([
        "--verify", str(archive), "--manifest", "source_manifest.json", "--sha256", digest,
    ]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "archive"
    put(archive, "experiments/run.py", b"changed")
    with pytest.raises(SystemExit) as error:
        main(["--verify", str(archive), "--manifest", "source_manifest.json", "--sha256", digest])
    assert error.value.code == 1


@pytest.mark.parametrize("invocation", ["module", "script"])
def test_module_cli_uses_only_standard_library(archive, invocation):
    project = Path(__file__).resolve().parents[1]
    module = project / "neurosym/application/source_provenance.py"
    entrypoint = ["-m", "neurosym.application.source_provenance"] if invocation == "module" else [str(module)]
    result = subprocess.run(
        [sys.executable, "-S", *entrypoint, "--create", str(archive), "--output", "source_manifest.json"],
        cwd=project if invocation == "module" else archive,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    digest = json.loads(result.stdout)["manifest_sha256"]
    assert verify_source_manifest(archive, "source_manifest.json", digest)["source_file_count"] == 7
