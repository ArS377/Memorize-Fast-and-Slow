import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from neurosym.application.source_provenance import verify_source_manifest
from scripts import package_submission as packaging


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reviewer_archive(tmp_path_factory):
    destination = tmp_path_factory.mktemp("anonymous-package") / "review.zip"
    report = packaging.build_archive(ROOT, destination)
    return destination, report


def test_zip_has_no_git_identity_or_owner_metadata(reviewer_archive):
    path, report = reviewer_archive
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as archive:
        assert archive.comment == b""
        assert len(archive.infolist()) == report["file_count"]
        for item in archive.infolist():
            assert item.filename.startswith("supplementary_code/")
            assert ".git" not in Path(item.filename).parts
            assert "__pycache__" not in Path(item.filename).parts
            assert not item.comment and not item.extra
            assert item.date_time == (1980, 1, 1, 0, 0, 0)
            assert packaging.identifier_findings(item.filename, archive.read(item)) == []


def test_packaged_inputs_and_sources_are_unchanged(reviewer_archive):
    path, _ = reviewer_archive
    with zipfile.ZipFile(path) as archive:
        prefix = packaging.ARCHIVE_ROOT + "/"
        manifest = json.loads(archive.read(prefix + "package_manifest.json"))
        assert {prefix + row["path"] for row in manifest["files"]} == set(archive.namelist()) - {prefix + "package_manifest.json"}
        data_count = 0
        for row in manifest["files"]:
            data = archive.read(prefix + row["path"])
            assert len(data) == row["size_bytes"]
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
            if row["path"] not in {"source_manifest.json", "source_manifest.sha256"}:
                assert data == (ROOT / row["path"]).read_bytes()
            data_count += row["path"].startswith("results/")
        assert data_count == 41


def test_extracted_package_verifies_without_git_and_repackages(reviewer_archive, tmp_path):
    path, _ = reviewer_archive
    with zipfile.ZipFile(path) as archive:
        archive.extractall(tmp_path)
    extracted = tmp_path / packaging.ARCHIVE_ROOT
    assert not (extracted / ".git").exists()
    digest = (extracted / "source_manifest.sha256").read_text().strip()
    verified = verify_source_manifest(extracted, "source_manifest.json", digest)
    assert verified["source_file_count"] > 0
    repacked = tmp_path / "repacked.zip"
    subprocess.run([sys.executable, "-B", "scripts/package_submission.py", "--output", str(repacked)],
                   cwd=extracted, check=True, capture_output=True)
    assert repacked.read_bytes() == path.read_bytes()


def test_existing_output_and_source_output_are_rejected(reviewer_archive, tmp_path):
    path, _ = reviewer_archive
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        packaging.build_archive(ROOT, path)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="outside"):
        packaging.build_archive(ROOT, ROOT / "review.zip")


def test_relative_output_outside_root(reviewer_archive, tmp_path, monkeypatch):
    path, _ = reviewer_archive
    with zipfile.ZipFile(path) as archive:
        archive.extractall(tmp_path)
    extracted = tmp_path / packaging.ARCHIVE_ROOT
    monkeypatch.chdir(extracted)
    packaging.build_archive(extracted, Path("../relative.zip"))
    assert (tmp_path / "relative.zip").is_file()


def test_common_identifier_checks_do_not_reject_dependency_pins():
    personal_path = "/" + "home" + "/" + "review-account" + "/project"
    own_link = "https://" + "github.com/" + "example-author/project"
    commit = json.dumps({"git_head": "0123456789" * 4})
    for sample in (personal_path, own_link, commit):
        assert packaging.identifier_findings("example.json", sample.encode())
    assert not packaging.identifier_findings("requirements.txt", (ROOT / "requirements-scallop.txt").read_bytes())
    assert not packaging.identifier_findings("model.json", json.dumps({"revision": "0123456789" * 4}).encode())


def test_fixture_is_marked_anonymized_without_fake_execution_identity():
    fixture = json.loads((ROOT / "tests/fixtures/historical_persona_manifest.json").read_bytes())
    assert fixture["anonymization"]["purpose"].startswith("Test fixture only;")
    assert "git" not in fixture and "source_provenance" not in fixture
    assert fixture["dataset"]["parent_path"].startswith("results/")
    assert fixture["dataset"]["path"].startswith("results/")
