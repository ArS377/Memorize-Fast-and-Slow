from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pytest

from scripts import verify_paper_results as verifier


def test_jsonl_stream_accepts_crlf(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"id": "first"}\r\n{"id": "second"}\r\n')
    rows = verifier.iter_jsonl(path)
    assert iter(rows) is rows
    assert list(rows) == [{"id": "first"}, {"id": "second"}]


@pytest.mark.parametrize("text", ['[]\n', '{broken}\n', '\n', '{"id": 1, "id": 2}\n', '{"score": NaN}\n', '{"score": Infinity}\n'])
def test_jsonl_rejects_malformed_records(tmp_path, text):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"valid": true}\n' + text, encoding="utf-8")
    rows = verifier.iter_jsonl(path)
    assert next(rows) == {"valid": True}
    with pytest.raises(verifier.VerificationError, match=r"rows.jsonl:2:"):
        next(rows)


def test_duplicate_composite_key_is_rejected():
    rows = [{"evaluation_input_id": "q", "arm": "a"}] * 2
    with pytest.raises(verifier.VerificationError, match="duplicate key"):
        verifier.index_unique(rows, ("evaluation_input_id", "arm"))
    assert len(verifier.index_unique([rows[0], {"evaluation_input_id": "q", "arm": "b"}], ("evaluation_input_id", "arm"))) == 2


@pytest.mark.parametrize("row", [{"id": ""}, {"id": None}, {"id": []}, {}])
def test_invalid_index_keys_are_rejected(row):
    with pytest.raises(verifier.VerificationError):
        verifier.index_unique([row], ("id",))


def test_absolute_float_tolerance():
    verifier.compare_values({"ci": [0.25, 0.5]}, {"ci": [0.25 + 5e-13, 0.5]})
    with pytest.raises(verifier.VerificationError, match=r"value.ci\[0\]"):
        verifier.compare_values({"ci": [0.25, 0.5]}, {"ci": [0.25 + 2e-12, 0.5]})
    with pytest.raises(verifier.VerificationError):
        verifier.compare_values(1000000.0, 1000000.000001)


@pytest.mark.parametrize("actual, recorded", [(True, 1), (False, 0.0), (float("nan"), float("nan")), (float("inf"), float("inf")), ({"a": 1}, {}), ([1], [1, 2])])
def test_comparison_rejects_type_shape_and_nonfinite_values(actual, recorded):
    with pytest.raises(verifier.VerificationError):
        verifier.compare_values(actual, recorded)


@pytest.fixture
def persona_pair():
    raw = {"evaluation_input_id": "q", "arm": "a", "history_id": "h", "answer": "Answer: rose tea\nextra text", "gold": "rose tea"}
    prediction = {**raw, "short_answer": "rose tea", "exact_match": 1.0, "f1": 1.0}
    return raw, prediction


def test_persona_rescores_raw_answer(persona_pair):
    raw, prediction = persona_pair
    assert verifier.rescore_persona([raw], [prediction]) == [prediction]


@pytest.mark.parametrize("field, value", [("exact_match", 0.0), ("f1", 0.9), ("short_answer", "other"), ("answer", "other"), ("gold", "other"), ("history_id", "other")])
def test_persona_detects_tampered_scores_and_metadata(persona_pair, field, value):
    raw, prediction = persona_pair
    tampered = {**prediction, field: value}
    with pytest.raises(verifier.VerificationError, match=field):
        verifier.rescore_persona([raw], [tampered])


def test_persona_rejects_unmatched_and_duplicate_rows(persona_pair):
    raw, prediction = persona_pair
    with pytest.raises(verifier.VerificationError, match="keys differ"):
        verifier.rescore_persona([raw], [])
    with pytest.raises(verifier.VerificationError, match="duplicate"):
        verifier.rescore_persona([raw, raw], [prediction])
    with pytest.raises(verifier.VerificationError, match="duplicate"):
        verifier.rescore_persona([raw], [prediction, prediction])


def test_persona_partial_f1_is_not_rounded(persona_pair):
    raw, prediction = persona_pair
    raw = {**raw, "answer": "rose", "gold": "rose tea"}
    prediction = {**raw, "short_answer": "rose", "exact_match": 0.0, "f1": 2 / 3}
    assert verifier.rescore_persona([raw], [prediction])[0]["f1"] == 2 / 3
    prediction["f1"] = 0.666667
    with pytest.raises(verifier.VerificationError, match="f1"):
        verifier.rescore_persona([raw], [prediction])


def test_evidence_flags_use_both_event_and_fact_contracts():
    row = {"selected_event_ids": ["alternative"], "selected_fact_ids": []}
    assert verifier._evidence_scores(row, [["original", "alternative"]], [["fact"]]) == (False, 0.5)
    row["selected_fact_ids"] = ["fact"]
    assert verifier._evidence_scores(row, [["original", "alternative"]], [["fact"]]) == (True, 1.0)
    with pytest.raises(verifier.VerificationError, match="empty evidence"):
        verifier._evidence_scores(row, [], [["fact"]])


def test_saved_grounded_flags_are_checked_without_resolving():
    row = {"prediction": "answer", "gold": "answer", "answer_correct": True, "exact_evidence_hit": False, "grounded_answer_correct": False}
    verifier._check_flags(row, "exact_evidence_hit", "test")
    with pytest.raises(verifier.VerificationError, match="grounded_answer_correct"):
        verifier._check_flags({**row, "grounded_answer_correct": True}, "exact_evidence_hit", "test")
    with pytest.raises(verifier.VerificationError, match="expected boolean"):
        verifier._check_flags({**row, "answer_correct": 1}, "exact_evidence_hit", "test")


def test_interleaved_eligibility_filters_exact_split_profile_and_suffix():
    valid = {"query_id": "h-preference-change-delayed", "split": "test", "hardness_profile": "anti_shortcut_interleaved_v3"}
    queries = [valid, {**valid, "split": "train"}, {**valid, "hardness_profile": "other"}, {**valid, "query_id": "h-preference-change-delayed-extra"}, {**valid, "query_id": "h-current"}]
    assert verifier.eligible_interleaved_queries(queries) == [valid]


def test_root_path_cannot_escape_evidence(tmp_path):
    assert verifier._root_path(tmp_path, "results/dataset") == tmp_path / "results/dataset"
    with pytest.raises(verifier.VerificationError, match="escapes root"):
        verifier._root_path(tmp_path, "../elsewhere")


def test_main_json_stdout_and_nonzero_failure(tmp_path, capsys):
    assert verifier.main(["--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "failed"
    assert len(report["mismatches"]) == 3
    assert report["provenance_mode"] == "saved_record_reaggregation"
    assert report["integrity"]["authenticated"] is False
    assert not list(tmp_path.iterdir())


def test_report_continues_after_a_bundle_mismatch(tmp_path, monkeypatch):
    def good(root):
        print("import diagnostic")
        return {"counts": {"rows": 1}}

    def bad(root):
        raise verifier.VerificationError("tampered score")

    monkeypatch.setattr(verifier, "verify_persona", good)
    monkeypatch.setattr(verifier, "verify_continual", bad)
    monkeypatch.setattr(verifier, "verify_interleaved", good)
    report = verifier.verify(tmp_path)
    assert report["status"] == "failed"
    assert len(report["mismatches"]) == 1
    assert report["bundles"]["interleaved_memory_benchmark_v3"]["status"] == "passed"


def test_report_success_remains_unauthenticated(tmp_path, monkeypatch, capsys):
    for name in ("verify_persona", "verify_continual", "verify_interleaved"):
        monkeypatch.setattr(verifier, name, lambda root: {"gaps": []})
    assert verifier.main(["--root", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "passed"
    assert not report["integrity"]["authenticated"]


def test_external_integrity_failure_prevents_reaggregation(tmp_path, monkeypatch, capsys):
    from scripts import paper_artifacts

    monkeypatch.setattr(paper_artifacts, "verify", lambda *a, **kw: {"status": "failed"})
    monkeypatch.setattr(verifier, "verify", lambda *a: pytest.fail("unverified input was scored"))
    assert verifier.main(["--root", str(tmp_path), "--expected-manifest-sha256", "0" * 64]) == 1
    assert json.loads(capsys.readouterr().out)["integrity"]["status"] == "failed"


def test_report_output_is_exclusive_and_outside_evidence(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    output = tmp_path / "report.json"
    monkeypatch.setattr(verifier, "verify", lambda *a: {"status": "passed"})
    assert verifier.main(["--root", str(evidence), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "passed"
    with pytest.raises(ValueError, match="already exists"):
        verifier.main(["--root", str(evidence), "--output", str(output)])
    with pytest.raises(ValueError, match="outside the evidence"):
        verifier.main(["--root", str(evidence), "--output", str(evidence / "report.json")])


@pytest.fixture
def table1_report():
    scores = {
        "sliding_context": (0.35, 0.37416666666666665, 0.6, 0.6458333333333334),
        "structured_memory": (0.6916666666666667, 0.7075, 0.7083333333333334, 0.725),
        "graphiti_memory": (0.48333333333333334, 0.5433333333333333, 0.6666666666666666, 0.6916666666666667),
        "hybrid_kg_memory": (0.7833333333333333, 0.7958333333333333, 0.7583333333333333, 0.7583333333333333),
    }
    return {
        "counts": {"conditions": 120, "histories": 12, "arms": 8, "rows": 960},
        "score_values_checked": 16,
        "by_arm": [
            {"arm": f"{method}_{budget}", "exact_match": values[offset], "f1": values[offset + 1],
             "row_count": 120, "history_cluster_count": 12}
            for method, values in scores.items() for budget, offset in ((4096, 0), (16384, 2))
        ][::-1],
    }


def test_table1_formats_primary_table_in_paper_order(table1_report):
    table = verifier.render_table1(table1_report)
    rows = [line for line in table.splitlines() if line.startswith("|")]
    assert rows == [
        "| Method | 4K EM | 4K F1 | 16K EM | 16K F1 |",
        "|---|---:|---:|---:|---:|",
        "| Sliding context | 35.00 | 37.42 | 60.00 | 64.58 |",
        "| Structured memory | 69.17 | 70.75 | 70.83 | 72.50 |",
        "| Graphiti | 48.33 | 54.33 | 66.67 | 69.17 |",
        "| Hybrid KG memory | **78.33** | **79.58** | **75.83** | **75.83** |",
    ]
    assert "historical Surface A" in table
    assert "120 conditions" in table and "12 histories" in table and "960 generations" in table
    assert "recorded gold" in table and "not rerun" in table


def test_table1_formats_computed_values_not_hardcoded_numbers(table1_report):
    for row in table1_report["by_arm"]:
        if row["arm"] == "sliding_context_4096":
            row["exact_match"] = 0.9
    table = verifier.render_table1(table1_report)
    assert "| Sliding context | **90.00** |" in table
    assert "| Hybrid KG memory | 78.33 |" in table


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "wrong_count", "nan", "inf", "negative", "above_one", "boolean"])
def test_table1_rejects_incomplete_or_invalid_summary(table1_report, mutation):
    if mutation == "missing":
        table1_report["by_arm"].pop()
    elif mutation == "duplicate":
        table1_report["by_arm"].append(deepcopy(table1_report["by_arm"][0]))
    elif mutation == "extra":
        table1_report["by_arm"].append({**table1_report["by_arm"][0], "arm": "full_qwen_context"})
    elif mutation == "wrong_count":
        table1_report["counts"]["rows"] = 959
    else:
        table1_report["by_arm"][0]["f1"] = {
            "nan": float("nan"), "inf": float("inf"), "negative": -0.01,
            "above_one": 1.01, "boolean": True,
        }[mutation]
    with pytest.raises(verifier.VerificationError):
        verifier.render_table1(table1_report)


@pytest.fixture
def primary_bundle(tmp_path, monkeypatch):
    from experiments.persona_end_to_end_benchmark import _group_metrics, _paired_delta, _score_short_answer
    from scripts import artifact_resources

    root = tmp_path / "isolated primary evidence"
    directory = root / "results/persona_joint_surface_a_build2"
    directory.mkdir(parents=True)
    arms = [f"{method}_{budget}" for method in (
        "sliding_context", "structured_memory", "graphiti_memory", "hybrid_kg_memory"
    ) for budget in (4096, 16384)]
    generations = [
        {"evaluation_input_id": f"h{history}-c{condition}", "history_id": f"h{history}",
         "condition": f"c{condition}", "query_id": f"h{history}-q{condition}",
         "query_family": "preference_change", "arm": arm,
         "answer": "rose tea" if condition < 3 else "rose" if condition < 6 else "wrong", "gold": "rose tea"}
        for history in range(12) for condition in range(10) for arm in arms
    ]
    predictions = [{**row, **_score_short_answer(row["answer"], row["gold"])} for row in generations]
    counts = {"condition_count": 120, "generation_count": 960}
    contents = {
        "generations.jsonl": "".join(json.dumps(row) + "\n" for row in generations),
        "predictions.jsonl": "".join(json.dumps(row) + "\n" for row in predictions),
        "manifest.json": json.dumps({**counts, "arms": [{"name": arm} for arm in arms]}),
        "metrics.json": json.dumps({**counts, "aggregates": {
            "by_arm": _group_metrics(predictions, ("arm",)),
            "paired_deltas": [_paired_delta(predictions, arms[-1], arms[1], bootstrap_samples=10, seed=73)],
        }}),
    }
    for name, text in contents.items():
        (directory / name).write_text(text, encoding="utf-8", newline="\n")
    members = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in contents}
    index = {"schema_version": 1, "groups": {}, "resources": {
        "reference-persona": {"path": "results/persona_joint_surface_a_build2", "kind": "directory", "members": members}
    }}
    monkeypatch.setattr(artifact_resources, "load_index", lambda: index)
    return root, directory, members


def test_table1_command_verifies_rescores_and_is_read_only(primary_bundle, monkeypatch, capsys):
    from scripts import artifact_resources

    root, directory, _ = primary_bundle
    before = {path: path.read_bytes() for path in directory.iterdir()}
    verified = []
    original_verify = artifact_resources.verify
    original_open = Path.open

    def verify(*args, **kwargs):
        verified.append(args[1])
        return original_verify(*args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Table 1 must not start models, services, processes, or write files")

    def read_only_open(path, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in ("w", "a", "x", "+"))
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(artifact_resources, "verify", verify)
    monkeypatch.setattr(verifier, "verify_continual", forbidden)
    monkeypatch.setattr(verifier, "verify_interleaved", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(Path, "open", read_only_open)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    assert verifier.main(["--table1", "--root", str(root)]) == 0
    captured = capsys.readouterr()
    assert "| Sliding context | **30.00** | **50.00** | **30.00** | **50.00** |" in captured.out
    assert "| Hybrid KG memory |" in captured.out
    assert "Verified" in captured.out and not captured.err
    assert verified == ["reference-persona", "reference-persona"]
    assert {path: path.read_bytes() for path in directory.iterdir()} == before
    assert not (root / ".git").exists()
    assert not (root / "results/persona_conflict_conversations_surface_b").exists()


@pytest.mark.parametrize("mutation", ["tamper", "missing", "extra"])
def test_table1_integrity_failure_prevents_scoring(primary_bundle, monkeypatch, capsys, mutation):
    root, directory, _ = primary_bundle
    if mutation == "tamper":
        (directory / "predictions.jsonl").write_bytes(b"{}\n")
    elif mutation == "missing":
        (directory / "generations.jsonl").unlink()
    else:
        (directory / "unexpected.json").write_bytes(b"{}")
    monkeypatch.setattr(verifier, "verify_persona", lambda *args: pytest.fail("unverified input was scored"))
    assert verifier.main(["--table1", "--root", str(root)]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "Table 1 verification failed" in captured.err


@pytest.mark.parametrize("mutation", ["score", "duplicate", "missing", "aggregate"])
def test_table1_scientific_failure_never_prints_a_table(primary_bundle, capsys, mutation):
    root, directory, members = primary_bundle
    name = "metrics.json" if mutation == "aggregate" else "predictions.jsonl"
    path = directory / name
    if mutation == "aggregate":
        payload = json.loads(path.read_bytes())
        payload["aggregates"]["by_arm"][0]["f1"] = 0.9
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        rows = [json.loads(line) for line in path.read_bytes().splitlines()]
        if mutation == "score":
            rows[0]["exact_match"] = 0.0
        elif mutation == "duplicate":
            rows.append(rows[0])
        else:
            rows.pop()
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    members[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verifier.main(["--table1", "--root", str(root)]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "Table 1 verification failed" in captured.err


def test_table1_rechecks_integrity_before_publishing(primary_bundle, monkeypatch, capsys):
    root, directory, _ = primary_bundle
    original_verify = verifier.verify_persona

    def changed_during_scoring(path):
        result = original_verify(path)
        (directory / "metrics.json").write_bytes(b"{}")
        return result

    monkeypatch.setattr(verifier, "verify_persona", changed_during_scoring)
    assert verifier.main(["--table1", "--root", str(root)]) == 1
    captured = capsys.readouterr()
    assert not captured.out and "SHA256 mismatch" in captured.err


def test_table1_missing_dependencies_fail_without_partial_output(primary_bundle, monkeypatch, capsys):
    root, _, _ = primary_bundle

    def unavailable(*args):
        raise ModuleNotFoundError("missing lightweight dependency")

    monkeypatch.setattr(verifier, "verify_persona", unavailable)
    assert verifier.main(["--table1", "--root", str(root)]) == 1
    captured = capsys.readouterr()
    assert not captured.out and "missing lightweight dependency" in captured.err


@pytest.mark.parametrize("flags", [[], ["--root", ".", "--output", "unused.md"], ["--root", ".", "--expected-manifest-sha256", "0" * 64]])
def test_table1_requires_explicit_root_and_stdout_only(flags):
    with pytest.raises(SystemExit) as error:
        verifier.main(["--table1", *flags])
    assert error.value.code == 2


@pytest.mark.parametrize("entrypoint", ["script", "module"])
def test_table1_cli_fails_cleanly_for_missing_reference_root(tmp_path, entrypoint):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-B"] + (
        [str(root / "scripts/verify_paper_results.py")] if entrypoint == "script"
        else ["-m", "scripts.verify_paper_results"]
    )
    result = subprocess.run(
        [*command, "--table1", "--root", str(tmp_path / "missing")],
        cwd=tmp_path if entrypoint == "script" else root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert not result.stdout
    assert "Table 1 verification failed" in result.stderr and "Traceback" not in result.stderr
    assert not list(tmp_path.iterdir())


def test_table1_runs_from_git_free_source_archive(primary_bundle, tmp_path):
    from scripts import artifact_resources

    evidence, _, _ = primary_bundle
    source = Path(__file__).resolve().parents[1]
    archive = tmp_path / "source archive with spaces"
    for directory in ("experiments", "neurosym"):
        shutil.copytree(source / directory, archive / directory, ignore=shutil.ignore_patterns("__pycache__"))
    (archive / "scripts").mkdir()
    for name in ("verify_paper_results.py", "artifact_resources.py", "paper_artifacts.py"):
        shutil.copyfile(source / "scripts" / name, archive / "scripts" / name)
    (archive / "configs").mkdir()
    (archive / "configs/artifact_resources.json").write_text(json.dumps(artifact_resources.load_index()), encoding="utf-8")
    before = {path.relative_to(archive): path.read_bytes() for path in archive.rglob("*") if path.is_file()}
    result = subprocess.run(
        [sys.executable, "-B", str(archive / "scripts/verify_paper_results.py"), "--table1", "--root", str(evidence)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "| Sliding context | **30.00** | **50.00** | **30.00** | **50.00** |" in result.stdout
    assert "Verified pinned reference-persona" in result.stdout
    assert not result.stderr
    assert not (archive / ".git").exists()
    assert {path.relative_to(archive): path.read_bytes() for path in archive.rglob("*") if path.is_file()} == before
