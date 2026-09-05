"""Portable, read-only verification and access to explicitly located paper resources."""

import argparse
import hashlib
from pathlib import Path
import stat
import sys

if __package__:
    from . import paper_artifacts as artifacts
else:
    import paper_artifacts as artifacts


DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "configs/artifact_resources.json"


def digest_check(value):
    if not isinstance(value, str) or not artifacts.HEX.fullmatch(value):
        raise ValueError("invalid expected SHA256")
    return value


def member_check(name):
    artifacts.relative(name)
    if "/" in name:
        raise ValueError(f"resource members must be standalone filenames: {name}")
    return name


def load_index(path=DEFAULT_INDEX):
    index = artifacts.json_load(artifacts.no_links(path))
    validate_index(index)
    return index


def validate_index(index):
    if index["schema_version"] != 1 or not index["resources"]:
        raise ValueError("unsupported or empty resource index")
    paths = []
    for resource in index["resources"].values():
        path = artifacts.relative(resource["path"])
        if not path.startswith("results/"):
            raise ValueError("resources must be below results; historical source is never accessed here")
        paths.append(path.casefold())
        if resource["kind"] == "file":
            digest_check(resource["sha256"])
        elif resource["kind"] == "directory":
            if "manifest" in resource:
                manifest = resource["manifest"]
                member_check(manifest["path"])
                digest_check(manifest["sha256"])
                if manifest["hash_field"] != "artifact_sha256":
                    raise ValueError("unsupported manifest hash field")
                names = resource["required_members"]
                if "members" in resource or manifest["path"] in names:
                    raise ValueError("ambiguous manifest member inventory")
            else:
                names = list(resource["members"])
                for digest in resource["members"].values():
                    digest_check(digest)
            if not names or len(names) != len({name.casefold() for name in names}):
                raise ValueError("empty or case-colliding member inventory")
            for name in names:
                member_check(name)
        else:
            raise ValueError("unsupported resource kind")
    for i, path in enumerate(paths):
        if any(path == other or path.startswith(other + "/") or other.startswith(path + "/")
               for other in paths[:i]):
            raise ValueError("overlapping resource paths")
    for name, members in index["groups"].items():
        if name in index["resources"] or not members or len(members) != len(set(members)):
            raise ValueError("ambiguous or empty resource group")
        if any(member not in index["resources"] for member in members):
            raise ValueError("unknown resource in group")


def hash_file(path, expected):
    digest_check(expected)
    path = artifacts.no_links(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"resource is not a single-link regular file: {path}")
    digest = hashlib.sha256()
    prefix = b""
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(artifacts.BLOCK), b""):
            if not prefix:
                prefix = block[:1024]
            digest.update(block)
            size += len(block)
    if artifacts.pointer(prefix) is not None:
        raise ValueError(f"unmaterialized LFS pointer: {path}")
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return {"sha256": expected, "bytes": size}


def verify_resource(root, resource):
    path = artifacts.safe_path(root, resource["path"])
    if resource["kind"] == "file":
        return {resource["path"]: hash_file(path, resource["sha256"])}
    if not path.is_dir():
        raise ValueError(f"resource directory missing: {path}")
    checked = {}
    if "manifest" in resource:
        manifest = resource["manifest"]
        manifest_path = artifacts.safe_path(path, manifest["path"])
        checked[manifest["path"]] = hash_file(manifest_path, manifest["sha256"])
        document = artifacts.json_load(manifest_path)
        members = document[manifest["hash_field"]]
        if set(members) != set(resource["required_members"]):
            raise ValueError(f"manifest member inventory mismatch: {manifest_path}")
    else:
        members = resource["members"]
    expected = set(members) | set(checked)
    actual = set()
    for child in path.iterdir():
        artifacts.no_links(child)
        if not child.is_file():
            raise ValueError(f"unexpected non-file resource member: {child}")
        actual.add(child.name)
    if actual != expected:
        raise ValueError(f"resource inventory mismatch: {path}; missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}")
    for name, digest in members.items():
        member_check(name)
        checked[name] = hash_file(artifacts.safe_path(path, name), digest)
    return {resource["path"] + "/" + name: value for name, value in checked.items()}


def verify(root, selection, index=None):
    """Authenticate a resource/group without discovering roots or changing any files."""
    index = load_index() if index is None else index
    validate_index(index)
    root = artifacts.no_links(root)
    if not root.is_dir():
        raise ValueError(f"repo-shaped artifact root is not a directory: {root}")
    if selection in index["groups"]:
        names = index["groups"][selection]
    elif selection in index["resources"]:
        names = [selection]
    else:
        raise ValueError(f"unknown resource or group: {selection}")
    errors = []
    checked = {}
    for name in names:
        try:
            checked.update(verify_resource(root, index["resources"][name]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{name}: {exc}")
    return {"schema_version": 1, "status": "failed" if errors else "verified",
            "root": str(root), "selection": selection, "errors": errors,
            "checked_files": checked, "file_count": len(checked),
            "scope": "Selected exact-byte resources only; not a full snapshot, execution-source, or scientific-results verification."}


def access(root, selection, index=None):
    """Return paths only after every selected resource has verified successfully."""
    index = load_index() if index is None else index
    report = verify(root, selection, index)
    if report["status"] == "verified":
        names = index["groups"].get(selection, [selection])
        report["paths"] = {name: str(artifacts.safe_path(Path(report["root"]), index["resources"][name]["path"]))
                           for name in names}
        report["access_policy"] = "Read-only verified locations at check time; no copy/import or lock against later modification."
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Print roles, member inventories, trusted digests, and access limitations.")
    for command in ("verify", "access"):
        child = sub.add_parser(command)
        child.add_argument("resource", help="Resource or group ID from list")
        child.add_argument("--root", type=Path, required=True, help="Explicit repo-shaped root containing results/")
    args = parser.parse_args(argv)
    try:
        index = load_index()
        if args.command == "list":
            report = index
        else:
            operation = access if args.command == "access" else verify
            report = operation(args.root, args.resource, index)
        print(artifacts.encode(report).decode(), end="")
        return 0 if report.get("status", "verified") == "verified" else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(artifacts.encode({"status": "error", "error": str(exc)}).decode(), end="", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
