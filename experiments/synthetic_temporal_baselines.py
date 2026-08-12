"""Matched non-neural baselines for synthetic temporal preference events."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from rank_bm25 import BM25Okapi

from experiments.synthetic_temporal_preferences import preference_rule_parameters, resolve_query
from neurosym.domain.retrieval_config import DEFAULT_EMBEDDING_MODEL
from neurosym.adapters.scallop import validate_update_detailed


VISIBLE_QUERY_FIELDS = ("query_text",)


class DenseEventEmbedder(Protocol):
    """Encode raw event documents and visible query text for dense retrieval."""

    def encode_documents(self, texts: Sequence[str]) -> Any: ...

    def encode_query(self, text: str) -> Any: ...

    def metadata(self) -> dict[str, Any]: ...


def answer_no_history(_events: list[dict[str, Any]], _query: dict[str, str]) -> None:
    """Represent an agent that has no user history at query time."""
    return None


def answer_recency(events: list[dict[str, Any]], query: dict[str, str]) -> str | None:
    """Return the last observed preference, deliberately ignoring time and scope."""
    for event in reversed(events):
        fact = event["fact"]
        if event["operation"] in {"add", "supersede", "temporary_exception"} and fact["subject"] == query["subject"]:
            return fact["object"]
    return None


def answer_full_history(events: list[dict[str, Any]], query: dict[str, str]) -> str | None:
    """Use the complete event trace and deterministic safety policy."""
    return resolve_query(events, query)


def _tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens for BM25 matching."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _visible_query_tokens(query: dict[str, Any]) -> list[str]:
    """Build retrieval tokens from the structured fields visible at query time."""
    return _tokenize(_visible_query_text(query))


def _visible_query_text(query: dict[str, Any]) -> str:
    """Return only user-visible question text across the retrieval boundary."""
    text = query.get("query_text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("query_text must be a non-empty string")
    return text


def _event_text(event: dict[str, Any]) -> str:
    """Return natural model text when present, otherwise the source support span."""
    model_text = event.get("model_text")
    if isinstance(model_text, str) and model_text.strip():
        return model_text
    return str(event["fact"]["support_text"])


def answer_bm25_raw_events(
    events: list[dict[str, Any]], query: dict[str, Any], *, k: int = 5
) -> str | None:
    """Resolve a query from the top-``k`` BM25-ranked raw event support texts."""
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if not events:
        return resolve_query([], query)

    corpus = [_tokenize(_event_text(event)) for event in events]
    scores = BM25Okapi(corpus).get_scores(_visible_query_tokens(query))
    ranked_indices = sorted(range(len(events)), key=lambda index: (-float(scores[index]), index))
    retrieved_events = [events[index] for index in ranked_indices[:k]]
    return resolve_query(retrieved_events, query)


def evaluate_bm25_raw_events(
    events: list[dict[str, Any]], queries: list[dict[str, Any]], *, k: int = 5
) -> dict[str, float]:
    """Return exact-answer accuracy for BM25 retrieval over raw event support text."""
    return {
        "bm25_raw_event": sum(
            answer_bm25_raw_events(events, query, k=k) == query["gold"] for query in queries
        )
        / len(queries)
    }


def _default_dense_embedder(
    *, model: str, revision: str | None, batch_size: int
) -> DenseEventEmbedder:
    """Create the repository's CPU BGE embedder only when dense retrieval is requested."""
    from neurosym.adapters.dense_index import SentenceTransformerEmbedder
    from neurosym.domain.retrieval_config import EmbeddingConfig

    return SentenceTransformerEmbedder(
        EmbeddingConfig(
            model=model,
            requested_revision=revision,
            device="cpu",
            batch_size=batch_size,
        )
    )


def _rank_dense_raw_events(
    events: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    k: int,
    embedder: DenseEventEmbedder,
    document_vectors: Any | None = None,
    query_vector: Any | None = None,
) -> list[dict[str, Any]]:
    """Return deterministically ranked raw events using optional precomputed BGE vectors."""
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    if not events:
        return []

    from neurosym.adapters.dense_index import query_text_v1

    if document_vectors is None:
        document_vectors = embedder.encode_documents([_event_text(event) for event in events])
    if query_vector is None:
        query_vector = embedder.encode_query(query_text_v1(_visible_query_text(query)))
    scores = [
        sum(float(document_value) * float(query_value) for document_value, query_value in zip(document, query_vector))
        for document in document_vectors
    ]
    ranked_indices = sorted(range(len(events)), key=lambda index: (-scores[index], index))
    return [events[index] for index in ranked_indices[:k]]


def _encode_dense_query_vectors(
    queries: Sequence[dict[str, Any]], embedder: DenseEventEmbedder
) -> Sequence[Any]:
    """Encode all visible query texts at once when the embedder supports batching."""
    from neurosym.adapters.dense_index import query_text_v1

    texts = [query_text_v1(_visible_query_text(query)) for query in queries]
    batch_encoder = getattr(embedder, "encode_queries", None)
    if callable(batch_encoder):
        return batch_encoder(texts)
    return [embedder.encode_query(text) for text in texts]


def answer_dense_raw_events(
    events: list[dict[str, Any]],
    query: dict[str, Any],
    *,
    k: int = 5,
    embedder: DenseEventEmbedder | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    revision: str | None = None,
    batch_size: int = 32,
) -> str | None:
    """Resolve a query from top-``k`` CPU dense-retrieved raw event support texts."""
    provider = embedder or _default_dense_embedder(
        model=model, revision=revision, batch_size=batch_size
    )
    return resolve_query(_rank_dense_raw_events(events, query, k=k, embedder=provider), query)


def evaluate_batched_dense_raw_events(
    replayed_by_history: dict[str, list[dict[str, Any]]],
    queries: Sequence[dict[str, Any]],
    *,
    k: int,
    embedder: DenseEventEmbedder,
    model: str,
    revision: str | None,
    batch_size: int,
) -> tuple[list[str | None], dict[str, Any]]:
    """Answer ordered history-local queries after encoding documents and queries in batches."""
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    history_ids = sorted(replayed_by_history)
    replayed_events = [event for history_id in history_ids for event in replayed_by_history[history_id]]
    document_vectors = embedder.encode_documents(
        [_event_text(event) for event in replayed_events]
    )
    document_vectors_by_history: dict[str, Any] = {}
    vector_offset = 0
    for history_id in history_ids:
        history_size = len(replayed_by_history[history_id])
        document_vectors_by_history[history_id] = document_vectors[vector_offset:vector_offset + history_size]
        vector_offset += history_size
    query_vectors = _encode_dense_query_vectors(queries, embedder)
    predictions = [
        resolve_query(
            _rank_dense_raw_events(
                replayed_by_history[query["history_id"]],
                {field: value for field, value in query.items() if field != "gold"},
                k=k,
                embedder=embedder,
                document_vectors=document_vectors_by_history[query["history_id"]],
                query_vector=query_vector,
            ),
            {field: value for field, value in query.items() if field != "gold"},
        )
        for query, query_vector in zip(queries, query_vectors)
    ]
    return predictions, {
        "embedding": dict(embedder.metadata()),
        "embedding_device": "cpu",
        "embedding_batch_size": batch_size,
        "embedding_requested_revision": revision,
        "embedding_requested_model": model,
        "query_fields": list(VISIBLE_QUERY_FIELDS),
        "query_template": "bge_query.v1",
        "retrieval_k": k,
        "retrieval_unit": "raw_event_support_text",
    }


def evaluate_dense_raw_events(
    events: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    k: int = 5,
    embedder: DenseEventEmbedder | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    revision: str | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate CPU dense raw-event retrieval and return reproducibility metadata."""
    if not queries:
        raise ValueError("queries must not be empty")
    provider = embedder or _default_dense_embedder(
        model=model, revision=revision, batch_size=batch_size
    )
    routed_queries = [{**query, "history_id": "history"} for query in queries]
    answers, metadata = evaluate_batched_dense_raw_events(
        {"history": events},
        routed_queries,
        k=k,
        embedder=provider,
        model=model,
        revision=revision,
        batch_size=batch_size,
    )
    return {
        "dense_raw_event": sum(answer == query["gold"] for answer, query in zip(answers, queries))
        / len(queries),
        "metadata": metadata,
    }


def candidate_decision_accept_all(_candidate: dict[str, Any]) -> str:
    """Accept every syntactically supplied candidate without consistency checks."""
    return "accept"


def group_records_by_history(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group temporal records by history while preserving their original order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        history_id = record.get("history_id")
        if not isinstance(history_id, str) or not history_id:
            raise ValueError(f"record is missing a non-empty history_id: {record}")
        grouped.setdefault(history_id, []).append(record)
    return grouped


def replay_candidates_with_decisions(
    events: list[dict[str, Any]], candidates: list[dict[str, Any]], *, use_scallop: bool
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Replay candidates and return the resulting events plus their admission decisions."""
    replayed = list(events)
    facts_by_history = {
        history_id: [event["fact"] for event in history_events]
        for history_id, history_events in group_records_by_history(events).items()
    }
    decisions: list[dict[str, str]] = []
    for candidate in candidates:
        history_id = candidate["history_id"]
        history_facts = facts_by_history.setdefault(history_id, [])
        validation = (
            validate_update_detailed(history_facts, candidate["fact"], preference_rule_parameters())
            if use_scallop else None
        )
        decision = validation.decision if validation is not None else candidate_decision_accept_all(candidate)
        decisions.append({
            "candidate_id": candidate["candidate_id"],
            "history_id": history_id,
            "decision": decision,
        })
        if decision in {"accept", "replace"}:
            if decision == "replace":
                replacement = validation.replace_fact_id if validation is not None else None
                if replacement is None:
                    raise RuntimeError("validator returned replace without replace_fact_id")
                replayed = [event for event in replayed if event["fact"]["fact_id"] != replacement]
                history_facts[:] = [fact for fact in history_facts if fact["fact_id"] != replacement]
            replayed.append({
                "event_id": f"replay-{candidate['candidate_id']}",
                "history_id": history_id,
                "session_id": f"{history_id}-candidate-replay",
                "turn_index": 1,
                "operation": "add",
                "fact": candidate["fact"],
            })
            history_facts.append(candidate["fact"])
    return replayed, decisions


def replay_candidates(
    events: list[dict[str, Any]], candidates: list[dict[str, Any]], *, use_scallop: bool
) -> list[dict[str, Any]]:
    """Replay candidate writes with accept-all or Scallop admission control."""
    return replay_candidates_with_decisions(events, candidates, use_scallop=use_scallop)[0]


def evaluate_replayed_candidates(
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    use_scallop: bool,
    answer_query: Callable[[list[dict[str, Any]], dict[str, Any]], str | None],
) -> dict[str, float]:
    """Evaluate history-local candidate replay decisions and query answers.

    ``gold`` is retained only for scoring after retrieval so it cannot affect the
    query representation or retrieved raw events.
    """
    if not queries:
        raise ValueError("queries must not be empty")
    events_by_history = group_records_by_history(events)
    candidates_by_history = group_records_by_history(candidates)
    queries_by_history = group_records_by_history(queries)
    history_ids = events_by_history.keys() | candidates_by_history.keys() | queries_by_history.keys()
    decision_by_candidate: dict[str, str] = {}
    answers: list[tuple[str | None, dict[str, Any]]] = []
    for history_id in history_ids:
        replayed, decisions = replay_candidates_with_decisions(
            events_by_history.get(history_id, []),
            candidates_by_history.get(history_id, []),
            use_scallop=use_scallop,
        )
        decision_by_candidate.update({item["candidate_id"]: item["decision"] for item in decisions})
        for query in queries_by_history.get(history_id, []):
            visible_query = {field: value for field, value in query.items() if field != "gold"}
            answers.append((answer_query(replayed, visible_query), query))

    correct_decisions = sum(
        decision_by_candidate[candidate["candidate_id"]] == candidate["gold_decision"]
        for candidate in candidates
    )
    return {
        "candidate_decision_accuracy": correct_decisions / len(candidates) if candidates else 0.0,
        "candidate_accept_count": float(sum(decision in {"accept", "replace"} for decision in decision_by_candidate.values())),
        "candidate_reject_count": float(sum(decision == "reject" for decision in decision_by_candidate.values())),
        "query_accuracy": sum(answer == query["gold"] for answer, query in answers) / len(queries),
    }


def evaluate_bm25_replayed_candidates(
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    k: int = 5,
    use_scallop: bool,
) -> dict[str, float]:
    """Evaluate history-local BM25 raw-event retrieval after candidate replay."""
    return evaluate_replayed_candidates(
        events,
        candidates,
        queries,
        use_scallop=use_scallop,
        answer_query=lambda history_events, query: answer_bm25_raw_events(history_events, query, k=k),
    )


def evaluate_dense_replayed_candidates(
    events: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    *,
    k: int = 5,
    use_scallop: bool,
    embedder: DenseEventEmbedder | None = None,
    model: str = DEFAULT_EMBEDDING_MODEL,
    revision: str | None = None,
    batch_size: int = 32,
) -> dict[str, float]:
    """Evaluate history-local dense raw-event retrieval after candidate replay."""
    if not queries:
        raise ValueError("queries must not be empty")
    provider = embedder or _default_dense_embedder(
        model=model, revision=revision, batch_size=batch_size
    )
    events_by_history = group_records_by_history(events)
    candidates_by_history = group_records_by_history(candidates)
    queries_by_history = group_records_by_history(queries)
    history_ids = events_by_history.keys() | candidates_by_history.keys() | queries_by_history.keys()
    replayed_by_history: dict[str, list[dict[str, Any]]] = {}
    decision_by_candidate: dict[str, str] = {}
    for history_id in history_ids:
        replayed, decisions = replay_candidates_with_decisions(
            events_by_history.get(history_id, []),
            candidates_by_history.get(history_id, []),
            use_scallop=use_scallop,
        )
        replayed_by_history[history_id] = replayed
        decision_by_candidate.update({item["candidate_id"]: item["decision"] for item in decisions})

    dense_queries = [
        query for query in queries if replayed_by_history[query["history_id"]]
    ]
    queried_history_ids = queries_by_history.keys()
    document_vectors_by_history: dict[str, Any] = {}
    replayed_events = [
        event
        for history_id in queried_history_ids
        for event in replayed_by_history[history_id]
    ]
    document_vectors = (
        provider.encode_documents([_event_text(event) for event in replayed_events])
        if replayed_events
        else []
    )
    vector_offset = 0
    for history_id in queried_history_ids:
        history_size = len(replayed_by_history[history_id])
        document_vectors_by_history[history_id] = document_vectors[vector_offset:vector_offset + history_size]
        vector_offset += history_size
    query_vectors = _encode_dense_query_vectors(dense_queries, provider) if dense_queries else []
    query_vector_index = 0
    answers: list[tuple[str | None, dict[str, Any]]] = []
    for query in queries:
        visible_query = {field: value for field, value in query.items() if field != "gold"}
        history_events = replayed_by_history[query["history_id"]]
        query_vector = None
        if history_events:
            query_vector = query_vectors[query_vector_index]
            query_vector_index += 1
        answers.append((
            resolve_query(
                _rank_dense_raw_events(
                    history_events,
                    visible_query,
                    k=k,
                    embedder=provider,
                    document_vectors=document_vectors_by_history[query["history_id"]],
                    query_vector=query_vector,
                ),
                visible_query,
            ),
            query,
        ))

    correct_decisions = sum(
        decision_by_candidate[candidate["candidate_id"]] == candidate["gold_decision"]
        for candidate in candidates
    )
    return {
        "candidate_decision_accuracy": correct_decisions / len(candidates) if candidates else 0.0,
        "candidate_accept_count": float(sum(decision in {"accept", "replace"} for decision in decision_by_candidate.values())),
        "candidate_reject_count": float(sum(decision == "reject" for decision in decision_by_candidate.values())),
        "query_accuracy": sum(answer == query["gold"] for answer, query in answers) / len(queries),
    }


def render_full_transcript(events: list[dict[str, Any]]) -> str:
    """Render all observed event text for a raw-context reader baseline."""
    return "\n".join(_event_text(event) for event in events)


def evaluate(events: list[dict[str, Any]], queries: list[dict[str, str]]) -> dict[str, float]:
    """Return exact query accuracy for frozen local baselines."""
    methods = {
        "no_history": answer_no_history,
        "recency": answer_recency,
        "full_history": answer_full_history,
        "bm25_raw_event": answer_bm25_raw_events,
    }
    return {
        name: sum(method(events, query) == query["gold"] for query in queries) / len(queries)
        for name, method in methods.items()
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL evaluation records from ``path``."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Run the public CPU dense raw-event evaluator from JSONL inputs."""
    parser = argparse.ArgumentParser(description="Evaluate CPU dense retrieval over raw temporal events.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding-revision")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = evaluate_dense_raw_events(
        _read_jsonl(args.events),
        _read_jsonl(args.queries),
        k=args.k,
        model=args.embedding_model,
        revision=args.embedding_revision,
        batch_size=args.embedding_batch_size,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.report is not None:
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return result


if __name__ == "__main__":
    main()
