"""Structured reasoning state and order-gap termination for KG-backed RLMs.

The state is intentionally application-owned.  Qwen proposes searches, memory
updates, and answers, while this module records what is known, what remains
open, and whether another RLM iteration materially changes that state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


NODE_TYPES = ("claim", "partial_answer", "open_question")
EDGE_TYPES = ("supports", "requires", "contradicts")
TEXT_FEATURES = 64
MAX_EMBEDDED_NODES = 256


def _clamp_confidence(value: Any, default: float = 0.5) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if not math.isfinite(parsed):
        parsed = default
    return max(0.0, min(1.0, parsed))


def _stable_id(prefix: str, values: Iterable[Any]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def _normalise_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _text_features(text: str) -> np.ndarray:
    """Return a deterministic, dependency-free signed hashing embedding."""
    vector = np.zeros(TEXT_FEATURES, dtype=np.float64)
    for token in re.findall(r"[\w'-]+", _normalise_text(text), flags=re.UNICODE):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % TEXT_FEATURES
        vector[index] += 1.0 if digest[2] % 2 == 0 else -1.0
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector


def _temporal_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_start = str(left.get("valid_from") or "")
    left_end = str(left.get("valid_to") or "9999-12-31")
    right_start = str(right.get("valid_from") or "")
    right_end = str(right.get("valid_to") or "9999-12-31")
    return max(left_start, right_start) <= min(left_end, right_end)


@dataclass
class EpistemicNode:
    node_id: str
    node_type: str
    label: str
    confidence: float = 0.5
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "attributes": copy.deepcopy(self.attributes),
        }


@dataclass
class EpistemicEdge:
    source: str
    target: str
    edge_type: str
    confidence: float = 0.5

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.source, self.target, self.edge_type)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "confidence": round(self.confidence, 6),
        }


@dataclass
class EpistemicState:
    question: str
    choices: Dict[str, str] = field(default_factory=dict)
    nodes: Dict[str, EpistemicNode] = field(default_factory=dict)
    edges: Dict[Tuple[str, str, str], EpistemicEdge] = field(default_factory=dict)
    current_answer_id: Optional[str] = None
    successful_searches: int = 0
    empty_searches: int = 0
    failed_searches: int = 0
    committed_fact_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if "open:root" not in self.nodes:
            self.nodes["open:root"] = EpistemicNode(
                node_id="open:root",
                node_type="open_question",
                label=self.question,
                confidence=1.0,
                attributes={"status": "open", "queries": []},
            )

    def clone(self) -> "EpistemicState":
        return copy.deepcopy(self)

    def add_edge(self, edge: EpistemicEdge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            return
        current = self.edges.get(edge.key)
        if current is None or edge.confidence > current.confidence:
            self.edges[edge.key] = edge

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "choices": dict(self.choices),
            "current_answer_id": self.current_answer_id,
            "successful_searches": self.successful_searches,
            "empty_searches": self.empty_searches,
            "failed_searches": self.failed_searches,
            "committed_fact_ids": list(self.committed_fact_ids),
            "nodes": [self.nodes[key].to_dict() for key in sorted(self.nodes)],
            "edges": [self.edges[key].to_dict() for key in sorted(self.edges)],
        }

    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> Dict[str, Any]:
        counts = {
            node_type: sum(1 for node in self.nodes.values() if node.node_type == node_type)
            for node_type in NODE_TYPES
        }
        current = self.nodes.get(self.current_answer_id or "")
        return {
            "node_counts": counts,
            "edge_count": len(self.edges),
            "current_answer": (
                current.attributes.get("choice") if current is not None else None
            ),
            "committed_fact_ids": list(self.committed_fact_ids),
            "successful_searches": self.successful_searches,
            "empty_searches": self.empty_searches,
            "failed_searches": self.failed_searches,
            "root_status": self.nodes["open:root"].attributes.get("status", "open"),
        }

    def prompt_view(self, max_claims: int = 50) -> Dict[str, Any]:
        claims = [
            {
                "fact_id": node.attributes.get("fact_id"),
                "subject": node.attributes.get("subject"),
                "predicate": node.attributes.get("predicate"),
                "object": node.attributes.get("object"),
                "support_text": node.attributes.get("support_text"),
                "confidence": round(node.confidence, 6),
                "selected": bool(node.attributes.get("selected")),
            }
            for node in sorted(self.nodes.values(), key=lambda value: value.node_id)
            if node.node_type == "claim"
        ][:max_claims]
        return {**self.summary(), "claims": claims}


def _search_question_id(query: str) -> str:
    return _stable_id("open", [_normalise_text(query)])


def _claim_id(result: Mapping[str, Any]) -> str:
    fact_id = str(result.get("fact_id") or "").strip()
    if fact_id:
        return f"claim:{fact_id}"
    return _stable_id(
        "claim",
        [result.get("subject"), result.get("predicate"), result.get("object")],
    )


def _answer_id(choice: str) -> str:
    return f"answer:{choice}"


def expand(state: EpistemicState, evidence: Mapping[str, Any]) -> EpistemicState:
    """Apply the evidence-acquisition operator without consolidation."""
    kind = str(evidence.get("kind") or "")
    if kind == "search":
        response = evidence.get("response")
        response = response if isinstance(response, Mapping) else {}
        request = response.get("request")
        request = request if isinstance(request, Mapping) else {}
        query = str(request.get("query") or evidence.get("query") or state.question).strip()
        question_id = _search_question_id(query)
        existing = state.nodes.get(question_id)
        if existing is None:
            state.nodes[question_id] = EpistemicNode(
                node_id=question_id,
                node_type="open_question",
                label=query,
                confidence=0.75,
                attributes={"status": "open", "queries": [query]},
            )
        root_queries = state.nodes["open:root"].attributes.setdefault("queries", [])
        if query and query not in root_queries:
            root_queries.append(query)
        if state.current_answer_id:
            state.add_edge(
                EpistemicEdge(
                    source=state.current_answer_id,
                    target=question_id,
                    edge_type="requires",
                    confidence=0.75,
                )
            )

        status = str(response.get("status") or "")
        results = response.get("results")
        results = results if isinstance(results, list) else []
        if status == "ok":
            state.successful_searches += 1
            if not results:
                state.empty_searches += 1
                state.nodes[question_id].attributes["empty"] = True
        else:
            state.failed_searches += 1
            state.nodes[question_id].attributes["last_error"] = copy.deepcopy(
                response.get("error")
            )

        for rank, raw in enumerate(results, start=1):
            if not isinstance(raw, Mapping):
                continue
            result = dict(raw)
            claim_id = _claim_id(result)
            score = _clamp_confidence(result.get("score"), default=1.0 / rank)
            label = " | ".join(
                str(result.get(key) or "").strip()
                for key in ("subject", "predicate", "object", "support_text")
            ).strip(" |")
            prior = state.nodes.get(claim_id)
            attributes = {
                "fact_id": str(result.get("fact_id") or ""),
                "subject": str(result.get("subject") or ""),
                "predicate": str(result.get("predicate") or ""),
                "object": str(result.get("object") or ""),
                "support_text": str(result.get("support_text") or ""),
                "document_id": result.get("document_id"),
                "provenance": copy.deepcopy(result.get("provenance") or []),
                "valid_from": result.get("valid_from"),
                "valid_to": result.get("valid_to"),
                "retrieval_mode": result.get("retrieval_mode"),
                "rank": result.get("rank", rank),
                "base_confidence": score,
                "selected": bool(prior and prior.attributes.get("selected")),
            }
            state.nodes[claim_id] = EpistemicNode(
                node_id=claim_id,
                node_type="claim",
                label=label,
                confidence=max(score, prior.confidence if prior else 0.0),
                attributes=attributes,
            )
            state.add_edge(
                EpistemicEdge(
                    source=claim_id,
                    target=question_id,
                    edge_type="supports",
                    confidence=score,
                )
            )
        return state

    if kind == "memory":
        response = evidence.get("response")
        response = response if isinstance(response, Mapping) else {}
        artifact = response.get("artifact")
        artifact = artifact if isinstance(artifact, Mapping) else {}
        selected = [str(value) for value in artifact.get("selected_fact_ids", []) if value]
        excluded = [str(value) for value in artifact.get("excluded_fact_ids", []) if value]
        if response.get("status") == "ok":
            state.committed_fact_ids = list(dict.fromkeys(state.committed_fact_ids + selected))
        for fact_id in selected:
            claim = state.nodes.get(f"claim:{fact_id}")
            if claim is not None:
                claim.attributes["selected"] = True
                claim.confidence = max(claim.confidence, 0.75)
        for fact_id in excluded:
            claim = state.nodes.get(f"claim:{fact_id}")
            if claim is not None:
                claim.attributes["excluded"] = True
                claim.confidence = min(claim.confidence, 0.25)
        return state

    if kind == "model":
        choice = str(evidence.get("candidate") or "").strip().upper()
        if choice not in {"A", "B", "C", "D"}:
            return state
        answer_id = _answer_id(choice)
        answer_text = state.choices.get(choice, "")
        if answer_id not in state.nodes:
            state.nodes[answer_id] = EpistemicNode(
                node_id=answer_id,
                node_type="partial_answer",
                label=f"{choice}) {answer_text}".strip(),
                confidence=0.5,
                attributes={"choice": choice, "answer_text": answer_text},
            )
        state.current_answer_id = answer_id
        return state

    return state


def consolidate(state: EpistemicState) -> EpistemicState:
    """Apply deterministic deduplication, conflict, and answer consolidation."""
    claims = [node for node in state.nodes.values() if node.node_type == "claim"]
    canonical: Dict[Tuple[str, str, str], EpistemicNode] = {}
    replacements: Dict[str, str] = {}
    for claim in sorted(claims, key=lambda value: value.node_id):
        key = tuple(
            _normalise_text(claim.attributes.get(field))
            for field in ("subject", "predicate", "object")
        )
        prior = canonical.get(key)
        if prior is None:
            canonical[key] = claim
            continue
        winner, loser = (claim, prior) if claim.confidence > prior.confidence else (prior, claim)
        canonical[key] = winner
        replacements[loser.node_id] = winner.node_id
        provenance = list(winner.attributes.get("provenance") or [])
        for entry in loser.attributes.get("provenance") or []:
            if entry not in provenance:
                provenance.append(entry)
        winner.attributes["provenance"] = provenance

    for loser_id, winner_id in replacements.items():
        state.nodes.pop(loser_id, None)
        rewritten: Dict[Tuple[str, str, str], EpistemicEdge] = {}
        for edge in state.edges.values():
            source = winner_id if edge.source == loser_id else edge.source
            target = winner_id if edge.target == loser_id else edge.target
            updated = EpistemicEdge(source, target, edge.edge_type, edge.confidence)
            current = rewritten.get(updated.key)
            if current is None or updated.confidence > current.confidence:
                rewritten[updated.key] = updated
        state.edges = rewritten

    claims = [node for node in state.nodes.values() if node.node_type == "claim"]
    for index, left in enumerate(claims):
        for right in claims[index + 1 :]:
            same_subject = _normalise_text(left.attributes.get("subject")) == _normalise_text(
                right.attributes.get("subject")
            )
            same_predicate = _normalise_text(
                left.attributes.get("predicate")
            ) == _normalise_text(right.attributes.get("predicate"))
            different_object = _normalise_text(left.attributes.get("object")) != _normalise_text(
                right.attributes.get("object")
            )
            if same_subject and same_predicate and different_object and _temporal_overlap(
                left.attributes, right.attributes
            ):
                confidence = min(left.confidence, right.confidence)
                source, target = sorted((left.node_id, right.node_id))
                state.add_edge(EpistemicEdge(source, target, "contradicts", confidence))
                if left.confidence != right.confidence:
                    loser = left if left.confidence < right.confidence else right
                    base = _clamp_confidence(
                        loser.attributes.get("base_confidence"), loser.confidence
                    )
                    loser.confidence = min(loser.confidence, base * 0.5)

    current = state.nodes.get(state.current_answer_id or "")
    if current is not None:
        supported = 0
        for fact_id in state.committed_fact_ids:
            claim_id = f"claim:{fact_id}"
            claim = state.nodes.get(claim_id)
            if claim is None:
                continue
            state.add_edge(
                EpistemicEdge(
                    source=claim_id,
                    target=current.node_id,
                    edge_type="supports",
                    confidence=claim.confidence,
                )
            )
            supported += 1
        current.confidence = min(1.0, 0.5 + 0.1 * supported)
        if supported:
            state.nodes["open:root"].attributes["status"] = "resolved"
            state.nodes["open:root"].confidence = current.confidence

    for node in state.nodes.values():
        if node.node_type != "open_question" or node.node_id == "open:root":
            continue
        has_support = any(
            edge.target == node.node_id and edge.edge_type == "supports"
            for edge in state.edges.values()
        )
        if has_support:
            node.attributes["status"] = "resolved"
            node.confidence = 1.0
        elif node.attributes.get("empty"):
            node.attributes["status"] = "unresolved"
            node.confidence = 0.5
    return state


def _state_vector(state: EpistemicState, node_order: Sequence[str]) -> np.ndarray:
    node_width = len(NODE_TYPES) + 1 + TEXT_FEATURES
    node_block = np.zeros((len(node_order), node_width), dtype=np.float64)
    edge_block = np.zeros((len(EDGE_TYPES), len(node_order), len(node_order)), dtype=np.float64)
    index = {node_id: position for position, node_id in enumerate(node_order)}
    type_index = {value: position for position, value in enumerate(NODE_TYPES)}
    edge_index = {value: position for position, value in enumerate(EDGE_TYPES)}
    for node_id, position in index.items():
        node = state.nodes.get(node_id)
        if node is None:
            continue
        node_block[position, type_index[node.node_type]] = 1.0
        node_block[position, len(NODE_TYPES)] = node.confidence
        node_block[position, len(NODE_TYPES) + 1 :] = _text_features(node.label)
    for edge in state.edges.values():
        if edge.source in index and edge.target in index:
            edge_block[
                edge_index[edge.edge_type], index[edge.source], index[edge.target]
            ] = edge.confidence
    controls = np.array(
        [
            min(state.successful_searches, 10) / 10.0,
            min(state.empty_searches, 10) / 10.0,
            min(state.failed_searches, 10) / 10.0,
            min(len(state.committed_fact_ids), 50) / 50.0,
            1.0 if state.current_answer_id else 0.0,
        ],
        dtype=np.float64,
    )
    return np.concatenate((node_block.ravel(), edge_block.ravel(), controls))


def state_distance(left: EpistemicState, right: EpistemicState) -> float:
    node_order = sorted(set(left.nodes).union(right.nodes))[:MAX_EMBEDDED_NODES]
    left_vector = _state_vector(left, node_order)
    right_vector = _state_vector(right, node_order)
    denominator = max(float(np.linalg.norm(left_vector)), float(np.linalg.norm(right_vector)), 1.0)
    return float(np.linalg.norm(left_vector - right_vector) / denominator)


@dataclass(frozen=True)
class StateTransition:
    order_gap: float
    before_digest: str
    after_digest: str
    state_snapshot: Dict[str, Any]


class EpistemicStateTracker:
    """Own one per-example graph and its windowed order-gap history."""

    def __init__(
        self,
        *,
        question: str,
        choices: Optional[Mapping[str, Any]] = None,
        epsilon: float = 0.025,
        window: int = 2,
        min_iterations: int = 2,
    ) -> None:
        if epsilon < 0:
            raise ValueError("order-gap epsilon must not be negative")
        if window < 1:
            raise ValueError("order-gap window must be at least 1")
        if min_iterations < 1:
            raise ValueError("order-gap minimum iterations must be at least 1")
        self.epsilon = float(epsilon)
        self.window = int(window)
        self.min_iterations = int(min_iterations)
        self.state = EpistemicState(
            question=str(question or ""),
            choices={str(key): str(value or "") for key, value in (choices or {}).items()},
        )
        self.event_gaps: List[float] = []
        self.completion_gaps: List[float] = []
        self.completion_digests: List[str] = []

    def observe(self, evidence: Mapping[str, Any], *, completion_boundary: bool) -> StateTransition:
        before = self.state.clone()
        actual = consolidate(expand(before.clone(), evidence))
        alternate = expand(consolidate(before.clone()), evidence)
        gap = round(state_distance(actual, alternate), 8)
        self.state = actual
        self.event_gaps.append(gap)
        if completion_boundary:
            self.completion_gaps.append(gap)
            self.completion_digests.append(actual.digest())
        return StateTransition(
            order_gap=gap,
            before_digest=before.digest(),
            after_digest=actual.digest(),
            state_snapshot=actual.to_dict(),
        )

    @property
    def window_mean(self) -> Optional[float]:
        if len(self.completion_gaps) < self.window:
            return None
        values = self.completion_gaps[-self.window :]
        return round(sum(values) / len(values), 8)

    @property
    def stable(self) -> bool:
        return (
            len(self.completion_gaps) >= max(self.window, self.min_iterations)
            and self.window_mean is not None
            and self.window_mean <= self.epsilon
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "window": self.window,
            "min_iterations": self.min_iterations,
            "event_gaps": list(self.event_gaps),
            "completion_gaps": list(self.completion_gaps),
            "window_mean": self.window_mean,
            "stable": self.stable,
            "state_digest": self.state.digest(),
            "state": self.state.to_dict(),
        }
