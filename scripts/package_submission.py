"""Build a Git-free reviewer ZIP from source, tests, and verified bundled inputs.

No experiments run and no input bytes are rewritten. Identifier checks catch
common leaks; they do not replace manual review of the final archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurosym.application.source_provenance import _no_links, _relative_name, source_manifest
from scripts.artifact_resources import load_index, verify


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "supplementary_code"
TEST_SUFFIXES = {".py", ".json", ".jsonl"}


def _encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def identifier_findings(name: str, data: bytes) -> list[str]:
    """Report locations/categories only, never print potential secret values."""
    text = data.decode("utf-8")
    findings = []
    if re.search(r"(?:/Users|/home)/[A-Za-z0-9._-]+/", text):
        findings.append(f"{name}: personal home-directory path")
    owners = re.findall(r"https?://github\.com/([A-Za-z0-9_.-]+)/", text, re.I)
    if any(owner.lower() != "scallop-lang" for owner in owners):
        findings.append(f"{name}: GitHub repository link outside approved Scallop dependency")
    for commit in re.findall(r'"git_head"\s*:\s*"([0-9a-f]{40})"', text, re.I):
        if len(set(commit)) > 1:
            findings.append(f"{name}: stored Git commit identity")
    return findings


def submission_payload(root: Path) -> dict[str, bytes]:
    _no_links(root)
    root = root.resolve(strict=True)
    manifest = source_manifest(root)
    entries = {row["path"]: row for row in manifest["files"]}
    inputs = verify(root, "bundled-persona", load_index(root / "configs/artifact_resources.json"))
    if inputs["status"] != "verified":
        raise ValueError("Bundled inputs failed verification: " + "; ".join(inputs["errors"]))
    names = set(entries) | set(inputs["checked_files"]) | {"LICENSE"}
    tests = root / "tests"
    _no_links(tests)
    for path in tests.rglob("*"):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        _no_links(path)
        if path.is_file() and path.suffix in TEST_SUFFIXES:
            names.add(relative.as_posix())
    payload = {}
    findings = []
    for name in sorted(names):
        _relative_name(name)
        path = root / name
        _no_links(path)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        expected = entries.get(name) or inputs["checked_files"].get(name)
        if expected is not None and digest != expected["sha256"]:
            raise ValueError(f"File changed while packaging: {name}")
        findings.extend(identifier_findings(name, data))
        payload[name] = data
    if findings:
        raise ValueError("Anonymity checks failed:\n" + "\n".join(findings))
    # The existing runtime verifies this source manifest without Git. Data retain
    # their independently pinned manifests; tests are covered by package_manifest.
    payload["source_manifest.json"] = _encode(manifest)
    payload["source_manifest.sha256"] = (
        hashlib.sha256(payload["source_manifest.json"]).hexdigest() + "\n"
    ).encode("ascii")
    payload["package_manifest.json"] = _encode({
        "version": 1,
        "scope": "Packaged files except this manifest; not original execution authentication.",
        "files": [{"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
                  for name, data in sorted(payload.items())],
    })
    return payload


def build_archive(root: Path, output: Path) -> dict[str, object]:
    _no_links(root)
    _no_links(output)
    root = root.resolve(strict=True)
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Write the archive outside the source/evidence directory")
    if output.suffix.lower() != ".zip":
        raise ValueError("Output must end in .zip")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {output}")
    payload = submission_payload(root)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.comment = b""
        for name, data in sorted(payload.items()):
            item = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{name}", date_time=(1980, 1, 1, 0, 0, 0))
            item.create_system = 3
            item.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            item.compress_type = zipfile.ZIP_DEFLATED
            item.comment = item.extra = b""
            archive.writestr(item, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return {"archive": str(output), "file_count": len(payload),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_manifest_sha256": payload["source_manifest.sha256"].decode().strip(),
            "scope": "Anonymous packaging checks only; external artifacts and full experiment reruns not verified."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_archive(ROOT, args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
