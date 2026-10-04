from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import experiments.synthetic_temporal_benchmark as temporal_benchmark
from experiments.synthetic_temporal_benchmark import replay_candidates, run_benchmark
from experiments.synthetic_temporal_preferences import DATASET_VERSION
from neurosym.adapters.validation_backend import HttpScallopValidatorBackend


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _ValidatorHandler(BaseHTTPRequestHandler):
    rule_version = DATASET_VERSION

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {
                "status": "ok",
                "engine": "scallopy",
                "scallop_available": True,
                "rule_version": self.rule_version,
            })
            return
        self._send(404, {"error": "not_found"})

    def do_POST(self) -> None:
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        fact = payload["new_fact"]
        if fact["fact_id"] == "replacement":
            self._send(200, {
                "decision": "replace",
                "reason": "synthetic replacement",
                "replace_fact_id": "old-fact",
                "rejection_label": None,
                "rule_params_version": DATASET_VERSION,
            })
            return
        if fact["fact_id"].endswith("-stale"):
            self._send(200, {
                "decision": "reject",
                "reason": "synthetic conflict",
                "replace_fact_id": None,
                "rejection_label": {
                    "code": "functional_conflict",
                    "category": "contradiction",
                    "rule_id": "synthetic.conflict",
                },
                "rule_params_version": DATASET_VERSION,
            })
            return
        if fact["fact_id"].endswith("-overlap-replacement"):
            self._send(200, {
                "decision": "replace",
                "reason": "synthetic higher-confidence replacement",
                "replace_fact_id": fact["fact_id"].replace("-overlap-replacement", "-replaceable"),
                "rejection_label": None,
                "rule_params_version": DATASET_VERSION,
            })
            return
        self._send(200, {
            "decision": "accept",
            "reason": "synthetic accept",
            "replace_fact_id": None,
            "rejection_label": None,
            "rule_params_version": DATASET_VERSION,
        })

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def validator_url() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ValidatorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_runner_writes_artifacts_for_both_conditions_and_actual_http_policy(
    tmp_path: Path, validator_url: str
) -> None:
    result = run_benchmark(tmp_path, scallop_validator_url=validator_url, history_count=1)

    assert result["manifest"]["status"] == "completed"
    assert result["manifest"]["validator_health"]["engine"] == "scallopy"
    assert (tmp_path / "datasets" / "lexical" / "events.jsonl").is_file()
    assert (tmp_path / "datasets" / "paraphrase" / "queries.jsonl").is_file()
    decisions = _rows(tmp_path / "decisions" / "lexical" / "scallop_fail_closed.jsonl")
    decisions_by_candidate = {record["candidate_id"]: record["decision"] for record in decisions}
    assert len(decisions_by_candidate) == 9
    assert decisions_by_candidate["history-001-stale"] == "reject"
    assert decisions_by_candidate["history-001-accepted"] == "accept"
    assert decisions_by_candidate["history-001-overlap-replacement"] == "replace"
    assert all(record["rule_params_version"] == DATASET_VERSION for record in decisions)
    assert (tmp_path / "predictions" / "paraphrase" / "accept_all" / "bm25.jsonl").is_file()
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["conditions"]
    assert "Synthetic Temporal Benchmark" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "dense" not in result["manifest"]
    assert not (tmp_path / "predictions" / "paraphrase" / "accept_all" / "dense.jsonl").exists()


def test_runner_writes_dense_artifacts_and_metadata_with_injected_evaluator(
    tmp_path: Path, validator_url: str
) -> None:
    calls: list[dict] = []

    def fake_dense_evaluator(
        replayed_by_history: dict[str, list[dict]],
        queries: list[dict],
        *,
        k: int,
        model: str,
        revision: str | None,
        batch_size: int,
    ) -> tuple[list[str | None], dict]:
        calls.append({
            "history_ids": sorted(replayed_by_history),
            "k": k,
            "model": model,
            "revision": revision,
            "batch_size": batch_size,
        })
        return [query["gold"] for query in queries], {
            "embedding": {"model_name": model, "resolved_revision": "fake-commit"},
            "retrieval_unit": "raw_event_support_text",
        }

    result = run_benchmark(
        tmp_path,
        scallop_validator_url=validator_url,
        include_dense=True,
        embedding_model="fake/bge",
        embedding_revision="test-revision",
        embedding_batch_size=7,
        dense_evaluator=fake_dense_evaluator,
    )

    dense_predictions = _rows(tmp_path / "predictions" / "paraphrase" / "accept_all" / "dense.jsonl")
    assert len(calls) == 4
    assert all(call["model"] == "fake/bge" for call in calls)
    assert all(call["revision"] == "test-revision" for call in calls)
    assert all(call["batch_size"] == 7 for call in calls)
    assert all(record["baseline"] == "dense" and record["correct"] for record in dense_predictions)
    assert result["metrics"]["conditions"]["paraphrase"]["accept_all"]["query_accuracy"]["dense"] == 1.0
    assert result["manifest"]["dense"] == {
        "enabled": True,
        "embedding_requested_model": "fake/bge",
        "embedding_requested_revision": "test-revision",
        "embedding_batch_size": 7,
        "retrieval_k": 5,
        "retrieval_unit": "raw_event_support_text",
        "metadata": {
            "embedding": {"model_name": "fake/bge", "resolved_revision": "fake-commit"},
            "retrieval_unit": "raw_event_support_text",
        },
    }
    assert "| Condition | Policy | Decision accuracy | no_history | recency | full_history | bm25 | dense |" in (
        tmp_path / "report.md"
    ).read_text(encoding="utf-8")


def test_runner_reports_query_accuracy_by_kind_with_fake_evaluator(
    tmp_path: Path, validator_url: str
) -> None:
    def fake_dense_evaluator(
        _replayed_by_history: dict[str, list[dict]],
        queries: list[dict],
        **_kwargs: object,
    ) -> tuple[list[str | None], dict]:
        return [
            query["gold"] if query["kind"] in {"preference", "private_recall"} else "wrong"
            for query in queries
        ], {}

    result = run_benchmark(
        tmp_path,
        scallop_validator_url=validator_url,
        include_dense=True,
        dense_evaluator=fake_dense_evaluator,
    )

    metrics = result["metrics"]["conditions"]["lexical"]["accept_all"]
    assert metrics["query_accuracy"]["dense"] == 7 / 9
    assert metrics["query_accuracy_by_kind"]["dense"] == {
        "preference": 1.0,
        "recommendation": 0.0,
        "private_recall": 1.0,
        "ambiguity": 0.0,
        "private_lineage": 0.0,
    }
    assert set(metrics["query_accuracy_by_kind"]) == {
        "no_history", "recency", "full_history", "bm25", "dense"
    }
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "private_lineage" in report
    assert "| lexical | accept_all | dense | 0.778 |" in report


def test_dense_evaluator_batches_replayed_documents_and_visible_queries(monkeypatch) -> None:
    class FakeEmbedder:
        document_calls: list[list[str]] = []
        query_calls: list[list[str]] = []

        def __init__(self, _config: object) -> None:
            return

        def encode_documents(self, texts: list[str]) -> list[list[float]]:
            self.document_calls.append(list(texts))
            return [[1.0] for _ in texts]

        def encode_queries(self, texts: list[str]) -> list[list[float]]:
            self.query_calls.append(list(texts))
            return [[1.0] for _ in texts]

        def metadata(self) -> dict[str, str]:
            return {"model_name": "fake/bge", "resolved_revision": "fake-commit"}

    from neurosym.adapters import dense_index

    monkeypatch.setattr(dense_index, "SentenceTransformerEmbedder", FakeEmbedder)
    replayed_by_history = {
        "history-1": [{
            "operation": "hard_constraint",
            "fact": {"fact_id": "f-1", "subject": "alex", "object": "peanuts", "support_text": "Alex cannot eat peanuts."},
        }],
        "history-2": [{
            "operation": "add",
            "fact": {"fact_id": "f-2", "subject": "blair", "object": "tea", "support_text": "Blair prefers tea."},
        }],
    }
    queries = [
        {"query_id": "q-1", "history_id": "history-2", "kind": "recommendation", "subject": "blair", "candidate": "tea", "query_text": "May tea be suggested to Blair?"},
        {"query_id": "q-2", "history_id": "history-1", "kind": "recommendation", "subject": "alex", "candidate": "peanuts", "query_text": "May peanuts be suggested to Alex?"},
        {"query_id": "q-3", "history_id": "history-2", "kind": "recommendation", "subject": "blair", "candidate": "coffee", "query_text": "May coffee be suggested to Blair?"},
    ]

    predictions, metadata = temporal_benchmark._evaluate_dense_raw_events(
        replayed_by_history,
        queries,
        k=1,
        model="fake/bge",
        revision="test-revision",
        batch_size=7,
    )

    assert FakeEmbedder.document_calls == [["Alex cannot eat peanuts.", "Blair prefers tea."]]
    assert len(FakeEmbedder.query_calls) == 1
    assert len(FakeEmbedder.query_calls[0]) == len(queries)
    assert predictions == ["ALLOWED", "INFEASIBLE", "ALLOWED"]
    assert metadata == {
        "embedding": {"model_name": "fake/bge", "resolved_revision": "fake-commit"},
        "embedding_device": "cpu",
        "embedding_batch_size": 7,
        "embedding_requested_revision": "test-revision",
        "embedding_requested_model": "fake/bge",
        "query_fields": ["query_text"],
        "query_template": "bge_query.v1",
        "retrieval_k": 1,
        "retrieval_unit": "raw_event_support_text",
    }


def test_cli_forwards_dense_embedding_options(tmp_path: Path, monkeypatch) -> None:
    received: dict = {}

    def fake_run_benchmark(output_dir: Path, **kwargs: object) -> dict:
        received["output_dir"] = output_dir
        received.update(kwargs)
        return {"metrics": {"conditions": {}}}

    monkeypatch.setattr(temporal_benchmark, "run_benchmark", fake_run_benchmark)

    temporal_benchmark.main([
        "--output-dir", str(tmp_path),
        "--scallop-validator-url", "http://validator.example",
        "--include-dense",
        "--embedding-model", "fake/bge",
        "--embedding-revision", "test-revision",
        "--embedding-batch-size", "7",
    ])

    assert received == {
        "output_dir": tmp_path,
        "scallop_validator_url": "http://validator.example",
        "history_count": 1,
        "timeout": 30.0,
        "bm25_k": 5,
        "include_dense": True,
        "embedding_model": "fake/bge",
        "embedding_revision": "test-revision",
        "embedding_batch_size": 7,
        "hardness_profile": "base",
    }


def test_runner_accepts_service_default_rule_version_when_responses_match(tmp_path: Path) -> None:
    _ValidatorHandler.rule_version = "wrong-rule-version"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ValidatorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = run_benchmark(tmp_path, scallop_validator_url=f"http://127.0.0.1:{server.server_port}")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
        _ValidatorHandler.rule_version = DATASET_VERSION

    assert result["manifest"]["status"] == "completed"


def test_http_replacement_decision_removes_the_replaced_fact(validator_url: str) -> None:
    backend = HttpScallopValidatorBackend(validator_url)
    events = [{"event_id": "old", "fact": {"fact_id": "old-fact"}}]
    candidates = [{
        "candidate_id": "replacement",
        "history_id": "history-001",
        "gold_decision": "accept",
        "fact": {"fact_id": "replacement"},
    }]

    replayed, decisions = replay_candidates(
        events, candidates, policy="scallop_fail_closed", backend=backend
    )

    assert [event["fact"]["fact_id"] for event in replayed] == ["replacement"]
    assert decisions[0]["decision"] == "replace"
    assert decisions[0]["admitted"] is True
