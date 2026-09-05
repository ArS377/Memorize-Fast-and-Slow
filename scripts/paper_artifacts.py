import argparse
import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone


SCHEMA = 1
BLOCK = 1024 * 1024
HEX = re.compile(r"[0-9a-f]{64}\Z")
DEFAULT_SELECTION = Path(__file__).resolve().parents[1] / "configs/paper_artifacts.json"


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                            env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"})
    if result.returncode:
        raise ValueError("git " + " ".join(args) + ": " + result.stderr.decode("utf-8", "replace").strip())
    return result.stdout


@contextmanager
def blob_stream(root, oid):
    process = subprocess.Popen(["git", "-C", str(root), "cat-file", "blob", oid],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"})
    try:
        yield process.stdout
        error = process.stderr.read()
        if process.wait():
            raise ValueError("git cat-file: " + error.decode("utf-8", "replace"))
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        process.stdout.close()
        process.stderr.close()


def relative(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError(f"unsafe relative path: {name!r}")
    parts = name.split("/")
    if any(p in ("", ".", "..") or p.endswith((".", " ")) for p in parts):
        raise ValueError(f"unsafe relative path: {name!r}")
    for part in parts:
        low = part.lower()
        if (low in {".git", ".env", "credentials", "secrets", "__pycache__"}
                or low.startswith((".env.", "credentials.", "secrets."))
                or re.fullmatch(r"(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?", low)
                or any(ord(c) < 32 or c in '<>"|?*' for c in part)):
            raise ValueError(f"excluded or unsafe path: {name!r}")
    if PurePosixPath(name).is_absolute():
        raise ValueError(f"absolute path: {name!r}")
    if Path(name).suffix.lower() in {".pem", ".key", ".p12", ".safetensors", ".pt", ".pth", ".bin"}:
        raise ValueError(f"excluded payload: {name}")
    return name


def no_links(path):
    path = Path(os.path.abspath(path))
    for part in [*reversed(path.parents), path]:
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"symlink/reparse point refused: {part}")
    return path


def safe_path(root, name):
    root = no_links(root)
    path = no_links(root / relative(name))
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"path escapes root: {name}")
    return path


def json_decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)


def json_load(path):
    return json_decode(Path(path).read_bytes())


def bad_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def write_json(path, value):
    with path.open("xb") as stream:
        stream.write(encode(value))


def pointer(data):
    if not data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return None
    oid = re.search(rb"(?:^|\n)oid sha256:([0-9a-f]{64})(?:\r?\n|$)", data)
    size = re.search(rb"(?:^|\n)size ([0-9]+)(?:\r?\n|$)", data)
    return {"oid_sha256": oid[1].decode() if oid else None,
            "size": int(size[1]) if size else None}


def probe(stream, name):
    digest = hashlib.sha256()
    size = 0
    prefix = b""
    rows = 0 if name.endswith(".jsonl") else None
    malformed = 0
    first_bad = None
    chunks = iter(stream.readline, b"") if rows is not None else iter(lambda: stream.read(BLOCK), b"")
    for chunk in chunks:
        digest.update(chunk)
        size += len(chunk)
        prefix = (prefix + chunk[:1024])[:1024]
        if rows is not None:
            rows += 1
            try:
                value = json.loads(chunk.decode("utf-8"), parse_constant=bad_constant)
                if not isinstance(value, dict):
                    raise ValueError("JSONL rows must be objects")
            except (ValueError, UnicodeError):
                malformed += 1
                first_bad = first_bad or rows
    return {"bytes": size, "sha256": digest.hexdigest(), "jsonl_rows": rows,
            "malformed_rows": malformed, "first_malformed_row": first_bad,
            "lfs_pointer": pointer(prefix)}


def observe(root, name):
    path = safe_path(root, name)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular file: {name}")
    with path.open("rb") as stream:
        result = probe(stream, name)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"source changed while reading: {name}")
    return result


def selection_check(selection):
    if selection["schema_version"] != SCHEMA:
        raise ValueError("unsupported selection schema")
    seen = set()
    for group in selection["evidence"]:
        name = relative(group["path"])
        if group["role"] not in {"result", "input"} or not name.startswith("results/"):
            raise ValueError(f"invalid evidence group: {name}")
        if any(name == old or name.startswith(old + "/") or old.startswith(name + "/") for old in seen):
            raise ValueError("overlapping evidence groups")
        seen.add(name)
        for manifest in group["manifests"]:
            relative(manifest)
    if not seen:
        raise ValueError("empty evidence selection")
    baseline = selection["baseline"]
    for directory in baseline["directories"]:
        if directory not in {"experiments", "neurosym", "services"}:
            raise ValueError("baseline directory not in source allowlist")
    for name in baseline["files"]:
        relative(name)
        allowed = name in {"requirements.txt", "requirements-qwen35-eval.txt", "requirements-graphiti-baseline.txt"} or (
            name.startswith(("configs/", "tests/", "scripts/"))
            and Path(name).suffix in {".py", ".json", ".jsonl", ".sh", ".txt"})
        if not allowed:
            raise ValueError(f"baseline file not in allowlist: {name}")
    manifests = {g["path"] + "/" + m for g in selection["evidence"] for m in g["manifests"]}
    for manifest, bindings in selection.get("relations", {}).items():
        if manifest not in manifests:
            raise ValueError(f"unknown manifest relation: {manifest}")
        for key, target in bindings.items():
            relative(target)
            if key in {"dataset", "parent"} and not any(g["path"] == target and g["role"] == "input" for g in selection["evidence"]):
                raise ValueError(f"relation is not an explicit input: {target}")
    return selection


def baseline_tree(root, selection, allow_missing=False):
    baseline = selection["baseline"]
    revision = git(root, "rev-parse", "--verify", baseline["revision"] + "^{commit}").decode().strip()
    records = git(root, "ls-tree", "-rz", revision, "--", *baseline["directories"], *baseline["files"])
    entries = []
    found = set()
    for record in records.split(b"\0"):
        if not record:
            continue
        header, name = record.split(b"\t", 1)
        mode, kind, oid = header.decode().split()
        name = relative(name.decode("utf-8"))
        if name not in baseline["files"] and not (Path(name).suffix == ".py" and any(name.startswith(d + "/") for d in baseline["directories"])):
            continue
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"non-regular baseline blob: {name}")
        found.add(name)
        entries.append({"path": "baseline_source/" + name, "role": "baseline_source",
                        "origin": {"kind": "git_blob", "path": name, "revision": revision,
                                   "oid": oid, "mode": mode}})
    missing = set(baseline["files"]) - found
    if missing and not allow_missing:
        raise ValueError(f"baseline files missing: {sorted(missing)}")
    return revision, entries


def evidence_tree(root, selection, evidence_source="worktree", revision=None):
    directories = [g["path"] for g in selection["evidence"]]
    if evidence_source == "worktree":
        output = git(root, "ls-files", "--stage", "-z", "--", *directories)
    elif evidence_source == "git":
        revision = revision or git(root, "rev-parse", "--verify", selection["baseline"]["revision"] + "^{commit}").decode().strip()
        output = git(root, "ls-tree", "-rz", revision, "--", *directories)
    else:
        raise ValueError("evidence_source must be worktree or git")
    entries = []
    found = set()
    for record in output.split(b"\0"):
        if not record:
            continue
        header, name = record.split(b"\t", 1)
        if evidence_source == "git":
            mode, kind, oid = header.decode().split()
            regular = kind == "blob"
        else:
            mode, oid, stage = header.decode().split()
            regular = stage == "0"
        name = relative(name.decode("utf-8"))
        group = next((g for g in selection["evidence"] if name.startswith(g["path"] + "/")), None)
        if group is None or not regular or mode not in {"100644", "100755"}:
            raise ValueError(f"unsafe/unmerged evidence entry: {name}")
        found.add(group["path"])
        lfs = None
        if int(git(root, "cat-file", "-s", oid)) < 1024:
            lfs = pointer(git(root, "cat-file", "blob", oid))
        if evidence_source == "git":
            origin = {"kind": "lfs_worktree" if lfs else "git_blob", "path": name,
                      "revision": revision, "oid": oid, "mode": mode}
            if lfs:
                origin["materialized_path"] = name
        else:
            origin = {"kind": "worktree", "path": name, "index_oid": oid}
        entries.append({"path": name, "role": group["role"], "origin": origin, "lfs_identity": lfs})
    if found != set(directories):
        raise ValueError(f"selected directories have no tracked files: {sorted(set(directories) - found)}")
    return entries


def evidence_json_loader(root, entries):
    by_path = {entry["path"]: entry for entry in entries}

    def load(name):
        if name not in by_path:
            raise ValueError(f"manifest absent from selected Git revision: {name}")
        entry = by_path[name]
        if entry["origin"]["kind"] == "git_blob":
            return json_decode(git(root, "cat-file", "blob", entry["origin"]["oid"]))
        observed = observe(root, name)
        identity = entry["lfs_identity"]
        if observed["sha256"] != identity["oid_sha256"] or observed["bytes"] != identity["size"]:
            raise ValueError(f"materialized LFS manifest identity mismatch: {name}")
        return json_load(safe_path(root, name))

    return load


def references(root, selection, loader=None):
    refs = []
    issues = []
    required = set()
    documents = {}
    for group in selection["evidence"]:
        for filename in group["manifests"]:
            name = group["path"] + "/" + filename
            required.add(name)
            try:
                doc = loader(name) if loader else json_load(safe_path(root, name))
                if not isinstance(doc, dict):
                    raise ValueError("manifest must be an object")
                documents[name] = doc
                if doc.get("status", doc.get("completion_status")) != "completed":
                    issues.append(f"incomplete run/data manifest: {name}")
            except (OSError, ValueError) as exc:
                issues.append(f"manifest unreadable: {name}: {exc}")
                continue
            binding = selection.get("relations", {}).get(name, {})

            def add(target, expected, field, expected_rows=None):
                relative(target)
                if field not in {"artifacts", "generation_count"} and (not isinstance(expected, str) or not HEX.fullmatch(expected)):
                    raise ValueError(f"malformed expected SHA256 at {field}")
                required.add(target)
                refs.append({"path": target, "sha256": expected, "rows": expected_rows,
                             "manifest": name, "field": field})

            def hashes(values, directory, field):
                if not isinstance(values, dict):
                    raise ValueError(f"hash map must be an object: {field}")
                for child, expected in values.items():
                    add(directory + "/" + relative(child), expected, field + "." + child)

            try:
                hashes(doc.get("artifact_sha256", {}), group["path"], "artifact_sha256")
                for child in doc.get("artifacts", []):
                    add(group["path"] + "/" + relative(child), None, "artifacts")
                if "generation_manifest_sha256" in doc:
                    add(group["path"] + "/generation_manifest.json", doc["generation_manifest_sha256"], "generation_manifest_sha256")
                if "generation_count" in doc:
                    count = doc["generation_count"]
                    if type(count) is not int or count < 0:
                        raise ValueError("invalid generation_count")
                    add(group["path"] + "/generations.jsonl", None, "generation_count", count)
                dataset = doc.get("dataset")
                if dataset is not None:
                    directory = binding["dataset"]
                    if isinstance(dataset, dict):
                        for key in ("artifact_sha256", "sha256"):
                            hashes(dataset.get(key, {}), directory, "dataset." + key)
                        if "generation_manifest_sha256" in dataset:
                            add(directory + "/generation_manifest.json", dataset["generation_manifest_sha256"], "dataset.generation_manifest_sha256")
                        if "parent_generation_manifest_sha256" in dataset:
                            add(binding["parent"] + "/generation_manifest.json", dataset["parent_generation_manifest_sha256"], "dataset.parent_generation_manifest_sha256")
                    elif not isinstance(dataset, str):
                        raise ValueError("invalid dataset relation")
                    hashes(doc.get("dataset_sha256", {}), directory, "dataset_sha256")
                if "parent" in doc:
                    parent = doc["parent"]
                    hashes(parent.get("artifact_sha256", {}), binding["parent"], "parent.artifact_sha256")
                    if "generation_manifest_sha256" in parent:
                        add(binding["parent"] + "/generation_manifest.json", parent["generation_manifest_sha256"], "parent.generation_manifest_sha256")
                if "config_sha256" in doc:
                    add("baseline_source/" + binding["config"], doc["config_sha256"], "config_sha256")
            except (KeyError, TypeError, ValueError, AttributeError) as exc:
                issues.append(f"incomplete expected references: {name}: {exc}")
    refs.sort(key=lambda r: (r["path"], r["manifest"], r["field"]))
    return refs, sorted(required), sorted(issues), documents


def assess(entry, refs):
    observed = entry.get("observed")
    if observed is None:
        return "missing"
    if observed["lfs_pointer"] is not None:
        return "lfs_pointer"
    if observed["malformed_rows"]:
        return "malformed_jsonl"
    lfs = entry.get("lfs_identity")
    if lfs and (lfs["oid_sha256"] != observed["sha256"] or lfs["size"] != observed["bytes"]):
        return "lfs_mismatch"
    expected = [r for r in refs if r["path"] == entry["path"]]
    if any(r["sha256"] is not None and r["sha256"] != observed["sha256"] for r in expected):
        return "historical_mismatch"
    if any(r["rows"] is not None and r["rows"] != observed["jsonl_rows"] for r in expected):
        return "row_count_mismatch"
    return "historical_match" if any(r["sha256"] is not None for r in expected) else "observed_only"


def summary(entries, required, issues):
    paths = {e["path"] for e in entries}
    missing = sorted(set(required) - paths | {e["path"] for e in entries if e.get("observed") is None})
    failures = [{"path": e["path"], "status": e["status"]} for e in entries if e["status"] not in {"historical_match", "observed_only"}]
    counts = {}
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    return {"status": "incomplete" if missing or issues else "failed" if failures else "verified",
            "missing": missing, "failures": failures, "issues": issues, "counts": counts,
            "scope": "selected raw-byte evidence and baseline export; observed_only is not historical authentication; no exact execution-source claim"}


def inventory(root, selection, evidence_source="worktree"):
    root = no_links(root)
    selection_check(selection)
    revision, source = baseline_tree(root, selection)
    evidence = evidence_tree(root, selection, evidence_source, revision)
    entries = evidence + source
    loader = evidence_json_loader(root, evidence) if evidence_source == "git" else None
    refs, required, issues, _ = references(root, selection, loader)
    for entry in entries:
        if entry["origin"]["kind"] != "git_blob":
            safe_path(root, entry["path"])
        try:
            if entry["origin"]["kind"] == "git_blob":
                with blob_stream(root, entry["origin"]["oid"]) as stream:
                    observed = probe(stream, entry["path"])
            else:
                observed = observe(root, entry["path"])
            entry["observed"] = observed
        except (OSError, ValueError) as exc:
            entry["observed"] = None
            entry["error"] = str(exc)
        entry["status"] = assess(entry, refs)
    entries.sort(key=lambda e: e["path"])
    return {"schema_version": SCHEMA, "baseline_revision": revision, "selection": selection,
            "evidence_source": evidence_source, "entries": entries, "required_paths": required, "references": refs,
            "assessment": summary(entries, required, issues)}


def freeze(root, selection, destination, evidence_source="worktree"):
    root = no_links(root)
    destination = no_links(destination)
    if destination.exists():
        raise ValueError("destination already exists; snapshots are never overwritten")
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError("destination overlaps input repository")
    if not destination.parent.is_dir():
        raise ValueError("destination parent must already exist")
    manifest = inventory(root, selection, evidence_source)
    invalid_lfs = [e["path"] for e in manifest["entries"] if e["origin"]["kind"] == "lfs_worktree"
                   and (e["observed"] is None or e["observed"]["lfs_pointer"] is not None
                        or e["observed"]["sha256"] != e["lfs_identity"]["oid_sha256"]
                        or e["observed"]["bytes"] != e["lfs_identity"]["size"])]
    if invalid_lfs:
        raise ValueError(f"Git evidence requires matching materialized LFS payloads: {invalid_lfs}")
    total = sum(e["observed"]["bytes"] for e in manifest["entries"] if e["observed"])
    reserve = max(16 * BLOCK, total // 100)
    if shutil.disk_usage(destination.parent).free < total + reserve:
        raise ValueError(f"insufficient disk space: need {total + reserve} bytes")
    destination.mkdir()
    copy_issues = []
    for entry in manifest["entries"]:
        if entry["observed"] is None:
            continue
        try:
            target = safe_path(destination, entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as out:
                if entry["origin"]["kind"] == "git_blob":
                    with blob_stream(root, entry["origin"]["oid"]) as source:
                        shutil.copyfileobj(source, out, BLOCK)
                else:
                    with safe_path(root, entry["path"]).open("rb") as source:
                        shutil.copyfileobj(source, out, BLOCK)
            if observe(destination, entry["path"]) != entry["observed"]:
                copy_issues.append(f"copy differs from inventory: {entry['path']}")
            if entry["origin"]["kind"] != "git_blob" and observe(root, entry["path"]) != entry["observed"]:
                copy_issues.append(f"source changed after inventory: {entry['path']}")
        except (OSError, ValueError) as exc:
            copy_issues.append(f"copy failed: {entry['path']}: {exc}")
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["copy_issues"] = copy_issues
    if copy_issues:
        manifest["assessment"]["status"] = "incomplete"
    manifest_path = destination / "freeze_manifest.json"
    write_json(manifest_path, manifest)
    digest = hashlib.sha256(encode(manifest)).hexdigest()
    report = verify(destination, digest, check_record=False)
    write_json(destination / "verification.json", {k: v for k, v in report.items() if k != "trust"})
    return {"destination": str(destination), "manifest_sha256": digest,
            "status": report["status"], "verification": report}


def verify(snapshot, expected_manifest_sha256=None, require_trusted=False, check_record=True):
    snapshot = no_links(snapshot)
    manifest_path = safe_path(snapshot, "freeze_manifest.json")
    manifest_bytes = manifest_path.read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json_decode(manifest_bytes)
    errors = []
    if require_trusted and expected_manifest_sha256 is None:
        errors.append("trusted verification requires an externally retained manifest SHA256")
    if expected_manifest_sha256 is not None and (not HEX.fullmatch(expected_manifest_sha256) or digest != expected_manifest_sha256):
        errors.append("external manifest SHA256 mismatch")
    if manifest["schema_version"] != SCHEMA:
        raise ValueError("unsupported snapshot schema")
    selection = selection_check(manifest["selection"])
    refs, required, issues, _ = references(snapshot, selection)
    if refs != manifest["references"] or required != manifest["required_paths"]:
        errors.append("expected reference inventory mismatch")
    entries = manifest["entries"]
    paths = [relative(e["path"]) for e in entries]
    if len(paths) != len(set(p.casefold() for p in paths)):
        errors.append("duplicate/case-colliding snapshot paths")
    checked = []
    for entry in entries:
        current = dict(entry)
        try:
            current["observed"] = observe(snapshot, entry["path"])
        except (OSError, ValueError) as exc:
            current["observed"] = None
            errors.append(f"payload unreadable: {entry['path']}: {exc}")
        if current["observed"] != entry["observed"]:
            errors.append(f"observed payload mismatch: {entry['path']}")
        current["status"] = assess(current, refs)
        if current["status"] != entry["status"]:
            errors.append(f"entry status mismatch: {entry['path']}")
        checked.append(current)
    actual_paths = set()
    for directory, dirs, files in os.walk(snapshot, followlinks=False):
        for child in dirs + files:
            no_links(Path(directory) / child)
        for filename in files:
            actual_paths.add((Path(directory) / filename).relative_to(snapshot).as_posix())
    if actual_paths - {"freeze_manifest.json", "verification.json"} != set(paths):
        errors.append("snapshot file inventory mismatch")
    assessment = summary(checked, required, issues)
    if manifest.get("copy_issues"):
        assessment["status"] = "incomplete"
        errors.extend(manifest["copy_issues"])
    if assessment != manifest["assessment"]:
        errors.append("snapshot assessment/status mismatch")
    report = {"schema_version": SCHEMA, "manifest_sha256": digest,
              "status": "failed" if errors else assessment["status"],
              "assessment": assessment, "errors": sorted(errors)}
    if check_record:
        try:
            recorded = json_load(safe_path(snapshot, "verification.json"))
            if recorded != report:
                report["errors"].append("verification.json status/content mismatch")
                report["status"] = "failed"
        except (OSError, ValueError) as exc:
            report["errors"].append(f"verification.json unreadable: {exc}")
            report["status"] = "incomplete"
    report["trust"] = "external_digest_match" if expected_manifest_sha256 == digest else "internal_consistency_only"
    return report


def new_external_path(root, path):
    root = no_links(root)
    path = no_links(path)
    if path.exists():
        raise ValueError(f"output already exists: {path}")
    if path.is_relative_to(root) or root.is_relative_to(path):
        raise ValueError("output overlaps input repository")
    if not path.parent.is_dir():
        raise ValueError("output parent must already exist")
    for parent in path.parents:
        if ((parent / "freeze_manifest.json").exists() or (parent / "verification.json").exists()
                or ((parent / "baseline_source").is_dir() and (parent / "results").is_dir())):
            raise ValueError(f"output is inside a frozen snapshot: {parent}")
    return path


def export_sources(root, selection, report, destination):
    destination = new_external_path(root, destination)
    matches = {}
    for run in report["runs"]:
        for file in run["files"]:
            for comparison in file["comparisons"]:
                if comparison["status"] == "hash_verified":
                    matches[(comparison["commit"], file["path"])] = file["expected_sha256"]
    revisions = set(report["resolved_candidates"].values()) | {commit for commit, _ in matches}
    baseline = git(root, "rev-parse", "--verify", selection["baseline"]["revision"] + "^{commit}").decode().strip()
    planned = []
    snapshots = []
    matched_records = []
    for revision in sorted(revisions):
        extra = {path for commit, path in matches if commit == revision}
        chosen = dict(selection, baseline=dict(selection["baseline"], revision=revision,
                                               files=sorted(set(selection["baseline"]["files"]) | extra)))
        _, entries = baseline_tree(root, chosen, allow_missing=True)
        exported_paths = set()
        for entry in entries:
            source = entry["origin"]["path"]
            exported_paths.add(source)
            with blob_stream(root, entry["origin"]["oid"]) as stream:
                observed = probe(stream, source)
            if observed["lfs_pointer"] is not None:
                raise ValueError(f"source closure contains an unmaterialized LFS pointer: {revision}:{source}")
            record = {"path": f"candidates/{revision}/{source}", "source_path": source,
                      "revision": revision, "oid": entry["origin"]["oid"], "mode": entry["origin"]["mode"],
                      "sha256": observed["sha256"], "bytes": observed["bytes"],
                      "classification": "baseline_snapshot" if revision == baseline else "candidate_recovered"}
            planned.append(record)
            expected = matches.get((revision, source))
            if expected is not None:
                if observed["sha256"] != expected:
                    raise ValueError(f"matching source changed: {revision}:{source}")
                match = dict(record, path=f"matches/{revision}/{source}", classification="hash_verified", expected_sha256=expected)
                planned.append(match)
                matched_records.append(match)
        snapshots.append({"revision": revision, "path": f"candidates/{revision}",
                          "classification": "baseline_snapshot" if revision == baseline else "candidate_recovered",
                          "file_count": len(entries), "missing_allowlisted_paths": sorted(set(chosen["baseline"]["files"]) - exported_paths),
                          "closure": "allowlisted source snapshot; exact dependency/import closure not audited"})
        if extra - exported_paths:
            raise ValueError(f"matching sources absent from closure: {sorted(extra - exported_paths)}")
    total = sum(record["bytes"] for record in planned)
    if shutil.disk_usage(destination.parent).free < total + max(16 * BLOCK, total // 100):
        raise ValueError("insufficient disk space for source export")
    new_external_path(root, destination).mkdir()
    for record in planned:
        target = safe_path(destination, record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with blob_stream(root, record["oid"]) as source, target.open("xb") as out:
            shutil.copyfileobj(source, out, BLOCK)
        observed = observe(destination, record["path"])
        if observed["sha256"] != record["sha256"] or observed["bytes"] != record["bytes"]:
            raise ValueError(f"exported source verification failed: {record['path']}")
    recovery = {"schema_version": SCHEMA, "status": "complete", "exact_execution_source": False,
                "snapshots": snapshots, "files": planned,
                "scope": "per-file hash matches plus labeled allowlisted Git snapshots; unknown dirty files and environment remain unresolved"}
    write_json(destination / "source_export_manifest.json", recovery)
    return {"directory": str(destination), "manifest_sha256": hashlib.sha256(encode(recovery)).hexdigest(),
            "manifest": "source_export_manifest.json", "snapshots": snapshots,
            "matches": matched_records, "exact_execution_source": False}


def source_audit(root, selection, path_history=False, output=None, export_sources_directory=None):
    selection_check(selection)
    output = new_external_path(root, output) if output is not None else None
    export_directory = new_external_path(root, export_sources_directory) if export_sources_directory is not None else None
    if output is not None and export_directory is not None:
        if output.is_relative_to(export_directory) or export_directory.is_relative_to(output):
            raise ValueError("audit output and source export must be separate paths")
    refs, _, issues, documents = references(root, selection)
    candidates = [selection["baseline"]["revision"], *selection.get("source_candidates", [])]
    resolved = {}
    unavailable = []
    for candidate in candidates:
        try:
            if candidate.startswith("-"):
                raise ValueError("invalid revision")
            resolved[candidate] = git(root, "rev-parse", "--verify", candidate + "^{commit}").decode().strip()
        except ValueError:
            unavailable.append(candidate)
    runs = []
    cache = {}
    for name, doc in documents.items():
        binding = selection.get("relations", {}).get(name, {})
        expected = dict(doc.get("source_sha256", {}))
        if "evaluator_script_sha256" in doc:
            expected[binding["evaluator"]] = doc["evaluator_script_sha256"]
        if "config_sha256" in doc:
            expected[binding["config"]] = doc["config_sha256"]
        if "candidate_source" in binding:
            expected[binding["candidate_source"]] = None
        if not expected and not doc.get("git") and not doc.get("source"):
            continue
        outcomes = []
        for path, expected_hash in sorted(expected.items()):
            relative(path)
            if not path.startswith(("experiments/", "neurosym/", "services/", "configs/", "tests/", "scripts/")):
                raise ValueError(f"source audit path outside allowlist: {path}")
            if expected_hash is not None and not HEX.fullmatch(expected_hash):
                raise ValueError(f"invalid recorded source hash: {path}")
            commits = set(resolved.values())
            if path_history and commits:
                history = git(root, "log", "--format=%H", *sorted(commits), "--", path).decode().splitlines()
                commits.update(history)
            comparisons = []
            for commit in sorted(commits):
                key = (commit, path)
                if key not in cache:
                    try:
                        data = git(root, "show", commit + ":" + path)
                        cache[key] = hashlib.sha256(data).hexdigest()
                    except ValueError:
                        cache[key] = None
                observed = cache[key]
                comparisons.append({"commit": commit, "sha256": observed,
                                    "status": "unresolved" if observed is None else "hash_verified" if expected_hash == observed else "candidate_recovered"})
            matched = any(c["status"] == "hash_verified" for c in comparisons)
            outcomes.append({"path": path, "expected_sha256": expected_hash,
                             "status": "hash_verified" if matched else "candidate_recovered" if expected_hash is None and any(c["sha256"] for c in comparisons) else "unresolved",
                             "comparisons": comparisons})
        provenance = doc.get("git", doc.get("source", {}))
        base = provenance.get("git_head", provenance.get("git_sha"))
        gaps = ["dependency/environment closure and full historical execution tree are not authenticated"]
        if base:
            try:
                git(root, "rev-parse", "--verify", base + "^{commit}")
            except ValueError:
                gaps.append("recorded base commit unavailable locally: " + base)
        else:
            gaps.append("no recorded execution commit")
        if provenance.get("dirty", provenance.get("git_dirty")):
            gaps.append("dirty diff bytes and untracked file identities/bytes not recovered; combined dirty hash cannot be reconstructed")
        if not any(value is not None for value in expected.values()):
            gaps.append("no recorded per-file hashes; candidates cannot authenticate historical execution")
        runs.append({"manifest": name, "recorded_provenance": provenance, "files": outcomes,
                     "status": "unresolved", "exact_execution_source": False, "gaps": gaps})
    report = {"schema_version": SCHEMA, "baseline_revision": selection["baseline"]["revision"],
              "resolved_candidates": resolved, "unavailable_candidates": unavailable,
              "runs": runs, "issues": issues,
              "scope": "local Git blob bytes only; no checkout normalization or model/service execution"}
    if export_directory is not None:
        report["source_export"] = export_sources(root, selection, report, export_directory)
    if output is not None:
        report["output"] = str(output)
        new_external_path(root, output)
        write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preserve selected paper evidence without modifying originals.")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("inventory", "freeze", "source-audit"):
        child = sub.add_parser(command)
        child.add_argument("--repo", type=Path, default=Path.cwd())
        child.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
        if command in {"inventory", "freeze"}:
            child.add_argument("--evidence-source", choices=("worktree", "git"), default="worktree")
        if command == "freeze":
            child.add_argument("--destination", type=Path, required=True)
        if command == "source-audit":
            child.add_argument("--path-history", action="store_true")
            child.add_argument("--output", type=Path)
            child.add_argument("--export-sources", type=Path)
    child = sub.add_parser("verify")
    child.add_argument("snapshot", type=Path)
    child.add_argument("--expected-manifest-sha256")
    child.add_argument("--require-trusted", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify(args.snapshot, args.expected_manifest_sha256, args.require_trusted)
        else:
            selection = json_load(args.selection)
            if args.command == "inventory":
                result = inventory(args.repo, selection, args.evidence_source)
            elif args.command == "freeze":
                result = freeze(args.repo, selection, args.destination, args.evidence_source)
            else:
                result = source_audit(args.repo, selection, args.path_history, args.output, args.export_sources)
        print(encode(result).decode(), end="")
        status = result.get("status", result.get("assessment", {}).get("status"))
        return 0 if status in (None, "verified") else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
