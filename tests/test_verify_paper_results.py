import json

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
