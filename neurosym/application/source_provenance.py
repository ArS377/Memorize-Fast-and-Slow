from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import subprocess
from typing import Any, Iterable


SOURCE_MANIFEST_VERSION = 1
SOURCE_DIRECTORIES = (
    "configs", "experiments", "neurosym", "scripts", "services",
)
ROOT_SOURCE_FILES = (
    "compiled_memory.py", "memory_artifacts.py", "neo4j_graph.py",
    "rejection_artifacts.py", "scallop_validator.py", "validator_backend.py",
    "requirements.txt", "requirements-qwen35-eval.txt",
    "requirements-graphiti-baseline.txt", "requirements-scallop.txt",
    "requirements-dense.txt", "requirements-generation.txt",
    "requirements-extraction.txt", "README.md", "pytest.ini",
    ".gitattributes", ".gitignore", "AGENTS.md",
)
SOURCE_SUFFIXES = (".json", ".py", ".sh")
EXCLUDED_DIRECTORIES = frozenset({
    "__pycache__", "cache", "caches", "env", "venv", "node_modules",
    "results", "outputs", "output", "build", "dist", "tests",
})
CREDENTIAL_STEMS = frozenset({
    "credential", "credentials", "secret", "secrets", "token", "tokens",
    "service_account", "service-account",
})


def git_provenance(
    path: Path | str, *, excluded_untracked_dir: Path | str | None = None
) -> dict[str, Any]:
    def run(cwd: Path, *args: str) -> bytes:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise ValueError(f"cannot resolve git provenance with {' '.join(args)}: {error}") from error

    root = Path(
        run(Path(path), "rev-parse", "--show-toplevel").decode("utf-8").strip()
    ).resolve()
    excluded_roots = [root / "results"]
    if excluded_untracked_dir is not None:
        excluded_roots.append(Path(excluded_untracked_dir).resolve())
    head = run(root, "rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff_args = ["diff", "--binary", "HEAD", "--", "."]
    for excluded in excluded_roots:
        if excluded == root or root in excluded.parents:
            relative_excluded = excluded.relative_to(root).as_posix()
            tracked_diff_args.extend(
                [
                    f":(exclude){relative_excluded}",
                    f":(exclude){relative_excluded}/**",
                ]
            )
    tracked_diff = run(root, *tracked_diff_args)
    untracked_names = [
        name.decode("utf-8")
        for name in run(
            root, "ls-files", "--others", "--exclude-standard", "-z"
        ).split(b"\0")
        if name
    ]
    digest = hashlib.sha256()
    digest.update(b"tracked-diff\0")
    digest.update(tracked_diff)
    included_untracked = []
    for name in sorted(untracked_names):
        source = (root / name).resolve()
        if any(
            source == excluded or excluded in source.parents
            for excluded in excluded_roots
        ):
            continue
        if not source.is_file():
            continue
        included_untracked.append(name)
        digest.update(b"\0untracked\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
    return {
        "git_head": head,
        "dirty": bool(tracked_diff or included_untracked),
        "dirty_diff_sha256": digest.hexdigest(),
        "untracked_file_count": len(included_untracked),
    }


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _absolute(path: Path | str, root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root / path


def _no_links(path: Path) -> None:
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise ValueError(f"symlink or junction is not allowed: {part}")


def _archive_root(root: Path | str) -> Path:
    path = Path(os.path.abspath(Path(root).expanduser()))
    _no_links(path)
    if not path.is_dir():
        raise ValueError(f"source root is not a directory: {path}")
    return path.resolve()


def _ignored(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith(".")
        or lowered in EXCLUDED_DIRECTORIES
        or (
            not lowered.endswith(".py")
            and any(
                lowered == stem or lowered.startswith((stem + ".", stem + "_", stem + "-"))
                for stem in CREDENTIAL_STEMS
            )
        )
    )


def _relative_name(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"unsafe source path: {value!r}")
    parts = value.split("/")
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or any(part in {"", ".", ".."} for part in parts)
        or any(re.search(r'[\x00-\x1f<>:"|?*]', part) or part.endswith((" ", ".")) for part in parts)
        or any(re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part) for part in parts)
    ):
        raise ValueError(f"unsafe source path: {value!r}")
    return value


def _source_name(name: str) -> bool:
    parts = name.split("/")
    if len(parts) == 1:
        return name in ROOT_SOURCE_FILES
    return (
        parts[0] in SOURCE_DIRECTORIES
        and not any(_ignored(part) for part in parts[1:])
        and PurePosixPath(name).suffix.lower() in SOURCE_SUFFIXES
    )


def _check_separate_path(root: Path, path: Path, label: str) -> None:
    path = path.resolve()
    if path == root or path in root.parents:
        raise ValueError(f"{label} overlaps source root: {path}")
    if any(_overlap(path, root / name) for name in SOURCE_DIRECTORIES + ROOT_SOURCE_FILES):
        raise ValueError(f"{label} overlaps declared source: {path}")


def _source_paths(root: Path, output_dir: Path | str | None) -> dict[str, Path]:
    if output_dir is not None:
        _check_separate_path(root, _absolute(output_dir, root), "output directory")
    paths: dict[str, Path] = {}

    def add(path: Path) -> None:
        _no_links(path)
        name = _relative_name(path.relative_to(root).as_posix())
        if not _source_name(name):
            raise ValueError(f"undeclared source file: {name}")
        if not path.is_file():
            raise ValueError(f"source is not a regular file: {name}")
        paths[name] = path

    def visit(directory: Path) -> None:
        _no_links(directory)
        if not directory.is_dir():
            raise ValueError(f"source directory is not a directory: {directory}")
        for path in sorted(directory.iterdir()):
            if _ignored(path.name):
                continue
            _no_links(path)
            if path.is_dir():
                visit(path)
            elif path.suffix.lower() in SOURCE_SUFFIXES:
                add(path)

    for name in SOURCE_DIRECTORIES:
        directory = root / name
        if os.path.lexists(directory):
            visit(directory)
    for path in root.iterdir():
        if path.suffix.lower() == ".py" or path.name in ROOT_SOURCE_FILES:
            add(path)
    if not paths:
        raise ValueError("source closure is empty")
    if len({name.casefold() for name in paths}) != len(paths):
        raise ValueError("source paths collide on case-insensitive filesystems")
    return dict(sorted(paths.items()))


def _file_identity(path: Path) -> tuple[int, str]:
    _no_links(path)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError(f"source is not a regular file: {path}")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _scope() -> dict[str, Any]:
    return {
        "directories": list(SOURCE_DIRECTORIES),
        "root_files": list(ROOT_SOURCE_FILES),
        "suffixes": list(SOURCE_SUFFIXES),
        "excluded_directories": sorted(EXCLUDED_DIRECTORIES),
        "excluded_credential_stems": sorted(CREDENTIAL_STEMS),
        "exclude_hidden": True,
    }


def source_manifest(
    root: Path | str, *, output_dir: Path | str | None = None
) -> dict[str, Any]:
    root = _archive_root(root)
    files = []
    for name, path in _source_paths(root, output_dir).items():
        size, digest = _file_identity(path)
        files.append({"path": name, "size_bytes": size, "sha256": digest})
    return {"version": SOURCE_MANIFEST_VERSION, "scope": _scope(), "files": files}


def write_source_manifest(
    root: Path | str,
    output: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> str:
    root = _archive_root(root)
    output = _absolute(output, root)
    _no_links(output)
    _check_separate_path(root, output, "source manifest")
    if output.suffix.lower() == ".py" and output.parent.resolve() == root:
        raise ValueError("source manifest cannot be a root Python source file")
    ensure_output_directory(output.parent)
    payload = (json.dumps(source_manifest(root, output_dir=output_dir), indent=2, sort_keys=True) + "\n").encode("utf-8")
    with output.open("xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def verify_source_manifest(
    root: Path | str,
    archive_manifest: Path | str,
    archive_manifest_sha256: str,
    *,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    if not isinstance(archive_manifest_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", archive_manifest_sha256):
        raise ValueError("archive source manifest requires an external SHA-256 digest")
    root = _archive_root(root)
    path = _absolute(archive_manifest, root)
    _no_links(path)
    _check_separate_path(root, path, "source manifest")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ValueError(f"cannot read source manifest {path}: {error}") from error
    digest = hashlib.sha256(raw).hexdigest()
    if digest != archive_manifest_sha256.lower():
        raise ValueError("source manifest SHA-256 differs from external digest")
    try:
        manifest = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"invalid source manifest: {error}") from error
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"version", "scope", "files"}
        or type(manifest["version"]) is not int
        or manifest["version"] != SOURCE_MANIFEST_VERSION
        or manifest["scope"] != _scope()
        or not isinstance(manifest["files"], list)
    ):
        raise ValueError("unsupported source manifest version or scope")
    entries = {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "size_bytes", "sha256"}:
            raise ValueError("invalid source manifest entry")
        name = _relative_name(entry["path"])
        if not _source_name(name):
            raise ValueError(f"undeclared source file: {name}")
        if name in entries:
            raise ValueError(f"duplicate source path: {name}")
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
            raise ValueError(f"invalid source size: {name}")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError(f"invalid source SHA-256: {name}")
        entries[name] = entry
    if list(entries) != sorted(entries):
        raise ValueError("source manifest files must be sorted by relative path")
    actual = _source_paths(root, output_dir)
    if set(actual) != set(entries):
        raise ValueError(
            f"source closure differs: missing={sorted(set(entries) - set(actual))}, "
            f"extra={sorted(set(actual) - set(entries))}"
        )
    for name, path in actual.items():
        size, file_digest = _file_identity(path)
        if size != entries[name]["size_bytes"] or file_digest != entries[name]["sha256"]:
            raise ValueError(f"source file size or SHA-256 differs: {name}")
    return {
        "manifest_version": SOURCE_MANIFEST_VERSION,
        "manifest_sha256": digest,
        "source_file_count": len(entries),
    }


def source_provenance(
    root: Path | str,
    *,
    output_dir: Path | str | None = None,
    archive_manifest: Path | str | None = None,
    archive_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    manifest = archive_manifest if archive_manifest is not None else os.environ.get("NEUROSYM_SOURCE_MANIFEST")
    digest = archive_manifest_sha256 if archive_manifest_sha256 is not None else os.environ.get("NEUROSYM_SOURCE_MANIFEST_SHA256")
    if manifest is not None or digest is not None:
        if not manifest or not digest:
            raise ValueError("archive provenance requires both source manifest and external SHA-256 digest")
        return {
            "mode": "archive",
            "archive": verify_source_manifest(root, manifest, digest, output_dir=output_dir),
        }
    return {
        "mode": "git",
        "git": git_provenance(root, excluded_untracked_dir=output_dir),
    }


def paper_evidence_roots(root: Path | str) -> tuple[Path, ...]:
    root = Path(root).resolve()
    selection = root / "configs" / "paper_artifacts.json"
    if not selection.exists():
        return ()
    payload = json.loads(selection.read_bytes(), object_pairs_hook=_unique_object)
    entries = payload.get("evidence")
    if not isinstance(entries, list):
        raise ValueError("paper artifact selection lacks evidence paths")
    resource_index = root / "configs" / "artifact_resources.json"
    if resource_index.exists():
        resources = json.loads(resource_index.read_bytes(), object_pairs_hook=_unique_object)
        entries = [*entries, *resources["resources"].values()]
    paths = []
    for entry in entries:
        name = _relative_name(entry["path"])
        if not name.startswith("results/"):
            raise ValueError("paper evidence path must be inside results")
        paths.append(root / name)
    frozen = payload.get("freeze_root")
    if frozen is not None:
        if not isinstance(frozen, str) or not frozen:
            raise ValueError("paper artifact freeze_root must be a nonempty path")
        paths.append(_absolute(frozen, root).resolve())
    return tuple(dict.fromkeys(paths))


def ensure_output_directory(
    output_dir: Path | str,
    input_dirs: Iterable[Path | str] = (),
    *,
    frozen_roots: Iterable[Path | str] = (),
    allow_resume: bool = False,
) -> Path:
    lexical = Path(os.path.abspath(Path(output_dir).expanduser()))
    output = lexical.resolve()
    for source in input_dirs:
        source = Path(source).expanduser().resolve()
        if _overlap(output, source):
            raise ValueError(f"output directory overlaps input: {source}")
    for frozen in frozen_roots:
        frozen = Path(frozen).expanduser().resolve()
        if _overlap(output, frozen):
            raise ValueError(f"output directory overlaps frozen root: {frozen}")
    _no_links(lexical)
    if output.is_dir():
        for directory, directories, files in os.walk(output, followlinks=False):
            for name in directories + files:
                path = Path(directory) / name
                _no_links(path)
                if name in files and path.stat().st_nlink > 1:
                    raise ValueError(f"refusing hard-linked output file: {path}")
    ancestors = set((lexical, *lexical.parents, output, *output.parents))
    for ancestor in sorted(ancestors):
        if os.path.lexists(ancestor / "freeze_manifest.json"):
            raise ValueError(f"output directory is inside a frozen bundle: {ancestor}")
        manifest_path = ancestor / "manifest.json"
        if os.path.lexists(manifest_path):
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
                )
            except (OSError, ValueError) as error:
                raise ValueError(f"cannot validate existing output manifest: {manifest_path}") from error
            if not isinstance(manifest, dict) or (
                "status" in manifest and not isinstance(manifest["status"], str)
            ):
                raise ValueError(f"invalid existing output manifest: {manifest_path}")
            if not allow_resume and (
                manifest.get("status") in {"completed", "complete"}
                or manifest.get("completed") is True
            ):
                raise ValueError(f"refusing completed output directory: {ancestor}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output directory is not a directory: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", metavar="ROOT", type=Path)
    mode.add_argument("--verify", metavar="ROOT", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.create is not None:
            if args.output is None:
                parser.error("--create requires --output")
            digest = write_source_manifest(args.create, args.output, output_dir=args.output_dir)
            result = {"manifest_sha256": digest}
        else:
            manifest = args.manifest or os.environ.get("NEUROSYM_SOURCE_MANIFEST")
            digest = args.sha256 or os.environ.get("NEUROSYM_SOURCE_MANIFEST_SHA256")
            if not manifest or not digest:
                parser.error("--verify requires --manifest and --sha256, or both environment variables")
            result = {"mode": "archive", "archive": verify_source_manifest(
                args.verify, manifest, digest, output_dir=args.output_dir
            )}
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
