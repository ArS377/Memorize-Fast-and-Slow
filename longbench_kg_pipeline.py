#!/usr/bin/env python3
"""
LongBench-v2 -> verified fact triples -> Neo4j pipeline.

What it does:
1. Loads LongBench-v2-style JSON/JSONL examples.
2. Splits `context` into sentence records with sent_id provenance.
3. Sends chunks to a vLLM OpenAI-compatible server for fact extraction.
4. Runs a second LLM self-check pass to verify facts.
5. Writes supported facts to JSONL and optionally inserts them into Neo4j.

Install:
    pip install openai neo4j

Example vLLM server:
    vllm serve Qwen/Qwen3-4B --host 0.0.0.0 --port 8000

Example run:
    python longbench_kg_pipeline.py \
        --input longbench_v2_sample.jsonl \
        --output verified_facts.jsonl \
        --model Qwen/Qwen3-4B \
        --vllm-base-url http://localhost:8000/v1

Example with Neo4j:
    python longbench_kg_pipeline.py \
        --input longbench_v2_sample.jsonl \
        --output verified_facts.jsonl \
        --model Qwen/Qwen3-4B \
        --neo4j-uri bolt://localhost:7687 \
        --neo4j-user neo4j \
        --neo4j-password password
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from openai import OpenAI

from neurosym.adapters.chunk_selection import (
    CHUNK_SELECTION_MODES,
    ChunkSelector,
    SelectedChunk,
    chunk_selection_query,
    source_evidence_snapshot,
)
from neurosym.domain.compiled_memory import fact_to_compiled_fact
from neurosym.domain.retrieval_config import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingConfig,
)
from neurosym.adapters.neo4j_graph import Neo4jGraph
from neurosym.reporting.rejections import append_rejection_jsonl, build_rejection_record


Fact = Dict[str, Any]
SentenceRecord = Dict[str, Any]


@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    model: str
    vllm_base_url: str
    api_key: str
    temperature: float
    max_tokens: int
    chunk_chars: int
    max_chunks_per_example: Optional[int]
    limit: Optional[int]
    sleep_seconds: float
    use_json_mode: bool
    neo4j_uri: Optional[str]
    neo4j_user: Optional[str]
    neo4j_password: Optional[str]
    neo4j_database: Optional[str] = None
    neo4j_session_id: Optional[str] = None
    neo4j_stateless: bool = False
    verify_batch_size: int = 20
    run_id: Optional[str] = None
    rejections_output_path: Optional[Path] = None
    chunk_selection: str = "first"
    chunk_selection_rrf_k: int = 60
    chunk_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_embedding_revision: Optional[str] = None
    chunk_embedding_device: str = "cpu"
    chunk_embedding_batch_size: int = 32


class LongBenchKGPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.client = OpenAI(base_url=config.vllm_base_url, api_key=config.api_key)
        self.graph: Optional[Neo4jGraph] = None
        self.run_id = config.run_id or make_run_id(config)
        self.chunk_selector = ChunkSelector(
            mode=config.chunk_selection,
            rrf_k=config.chunk_selection_rrf_k,
            embedding_config=EmbeddingConfig(
                model=config.chunk_embedding_model,
                requested_revision=config.chunk_embedding_revision,
                device=config.chunk_embedding_device,
                batch_size=config.chunk_embedding_batch_size,
            ),
        )

        if config.neo4j_uri:
            if not config.neo4j_user or not config.neo4j_password:
                raise ValueError("Neo4j URI provided, but user/password missing.")
            self.graph = Neo4jGraph(
                uri=config.neo4j_uri,
                user=config.neo4j_user,
                password=config.neo4j_password,
                database=config.neo4j_database,
                session_id=config.neo4j_session_id,
                stateless=config.neo4j_stateless,
            )

    @property
    def neo4j_driver(self):
        """Back-compat: legacy code/tests access pipeline.neo4j_driver directly.
        Reads/writes the underlying driver on self.graph.
        """
        return self.graph._driver if self.graph is not None else None

    @neo4j_driver.setter
    def neo4j_driver(self, value) -> None:
        if self.graph is not None:
            self.graph._driver = value

    def close(self) -> None:
        if self.graph is not None:
            self.graph.close()
            self.graph = None

    def select_chunks(
        self,
        example: Mapping[str, Any],
        chunks: Sequence[Sequence[Mapping[str, Any]]],
    ) -> List[SelectedChunk]:
        return self.chunk_selector.select(
            example,
            chunks,
            limit=self.config.max_chunks_per_example,
        )

    def chunk_selection_audit(
        self,
        example: Mapping[str, Any],
        selected: Sequence[SelectedChunk],
        *,
        session_id: str,
        total_chunks: int,
    ) -> List[Dict[str, Any]]:
        query = chunk_selection_query(example)
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        example_id = str(example.get("_id", "unknown"))
        return [
            item.audit_record(
                example_id=example_id,
                session_id=session_id,
                total_chunks=total_chunks,
                query_sha256=query_sha256,
                embedding=self.chunk_selector.embedding_metadata,
            )
            for item in selected
        ]

    def source_evidence_snapshot(
        self,
        example: Mapping[str, Any],
        chunks: Sequence[Sequence[Mapping[str, Any]]],
        selected: Sequence[SelectedChunk],
        *,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        return source_evidence_snapshot(
            example,
            chunks,
            selected,
            session_id=session_id,
            selection_mode=self.config.chunk_selection,
        )

    def run(self) -> None:
        examples = load_longbench_examples(self.config.input_path)
        if self.config.limit is not None:
            examples = examples[: self.config.limit]

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        rejections_path = (
            self.config.rejections_output_path
            or self.config.output_path.with_name(
                f"{self.config.output_path.stem}_rejections.jsonl"
            )
        )
        rejections_path.parent.mkdir(parents=True, exist_ok=True)
        rejections_path.write_text("", encoding="utf-8")
        total_supported = 0

        with self.config.output_path.open("w", encoding="utf-8") as out:
            for i, example in enumerate(examples, start=1):
                example_id = str(example.get("_id", f"example_{i}"))
                print(f"[{i}/{len(examples)}] Processing {example_id}", file=sys.stderr)

                sentence_records = flatten_context_to_sentence_records(example)
                chunks = chunk_sentence_records(sentence_records, self.config.chunk_chars)
                selected_chunks = self.select_chunks(example, chunks)

                extracted: List[Fact] = []
                for selected in selected_chunks:
                    facts = self.extract_facts(
                        example,
                        selected.records,
                        selected.chunk_index,
                    )
                    extracted.extend(facts)
                    polite_sleep(self.config.sleep_seconds)

                verified = self.verify_facts(example, extracted)
                for fact in verified:
                    fact["example_id"] = example_id
                    fact["fact_id"] = make_fact_id(example_id, fact)
                    fact["question"] = str(example.get("question", ""))

                supported = []
                verifier_rejected = []
                for fact in verified:
                    status = normalize_status(fact.get("status") or fact.get("confidence"))
                    if status == "supported":
                        supported.append(fact)
                    else:
                        verifier_rejected.append(fact)
                        append_rejection_jsonl(
                            rejections_path,
                            build_rejection_record(
                                candidate_fact=fact,
                                reason=fact.get("verification_reason") or status,
                                example_id=example_id,
                                session_id=self.config.neo4j_session_id or "default",
                                stage="llm_verification",
                                existing_conflicting_fact=None,
                                validator="llm_self_reflection",
                            ),
                        )

                for fact in supported:
                    fact.update(fact_to_compiled_fact(fact))

                accepted_for_output = list(supported)
                if self.graph is not None and supported:
                    result = self.graph.insert_facts(supported)
                    rejected_ids = {
                        str(rejected.get("candidate", {}).get("fact_id", ""))
                        for rejected in result.get("rejected", [])
                        if isinstance(rejected.get("candidate"), dict)
                    }
                    accepted_for_output = [
                        fact
                        for fact in supported
                        if str(fact.get("fact_id", "")) not in rejected_ids
                    ]
                    for rejected in result.get("rejected", []):
                        candidate = rejected.get("candidate", {})
                        if not isinstance(candidate, dict):
                            continue
                        append_rejection_jsonl(
                            rejections_path,
                            build_rejection_record(
                                candidate_fact=candidate,
                                reason=rejected.get("reason", ""),
                                example_id=str(candidate.get("example_id") or example_id),
                                session_id=self.graph.session_id,
                                stage="scallop_validation",
                                existing_conflicting_fact=rejected.get("existing_conflicting_fact"),
                                rule_fired=rejected.get("rule_fired"),
                                validator="scallop",
                            ),
                        )

                for fact in accepted_for_output:
                    out.write(json.dumps(fact, ensure_ascii=False) + "\n")

                total_supported += len(accepted_for_output)
                print(
                    f"    extracted={len(extracted)} supported={len(supported)} "
                    f"accepted={len(accepted_for_output)} "
                    f"rejected={len(verifier_rejected)} total_supported={total_supported}",
                    file=sys.stderr,
                )

        print(
            f"Done. Wrote {total_supported} supported facts to {self.config.output_path}; "
            f"rejections to {rejections_path}",
            file=sys.stderr,
        )

    def extract_facts(self, example: Dict[str, Any], chunk: List[SentenceRecord], chunk_index: int) -> List[Fact]:
        prompt = build_extraction_prompt(example, chunk, chunk_index)
        data = self.chat_json(prompt)
        raw_facts = data.get("facts", [])
        if not isinstance(raw_facts, list):
            return []

        clean_facts: List[Fact] = []
        for j, fact in enumerate(raw_facts):
            if not isinstance(fact, dict):
                continue
            cleaned = clean_fact(fact)
            if cleaned:
                cleaned["chunk_index"] = chunk_index
                cleaned["local_fact_index"] = j
                enrich_fact_provenance(
                    cleaned,
                    example=example,
                    chunk=chunk,
                    chunk_index=chunk_index,
                    extractor_model=self.config.model,
                    verifier_model="",
                    run_id=self.run_id,
                )
                clean_facts.append(cleaned)
        return clean_facts

    def verify_facts(self, example: Dict[str, Any], facts: List[Fact]) -> List[Fact]:
        if not facts:
            return []

        # Verify in small batches to keep prompts manageable.
        verified: List[Fact] = []
        batch_size = max(1, self.config.verify_batch_size)
        for start in range(0, len(facts), batch_size):
            batch = facts[start : start + batch_size]
            for idx, fact in enumerate(batch):
                fact["verification_id"] = f"f{start + idx}"

            prompt = build_verification_prompt(example, batch)
            data = self.chat_json(prompt)
            judgments = data.get("verified_facts", [])

            by_id = {f["verification_id"]: f for f in batch}
            seen_ids = set()
            if isinstance(judgments, list):
                for judgment in judgments:
                    if not isinstance(judgment, dict):
                        continue
                    vid = judgment.get("verification_id")
                    original = by_id.get(str(vid))
                    if not original:
                        continue
                    seen_ids.add(str(vid))
                    merged = dict(original)
                    merged["status"] = normalize_status(judgment.get("status"))
                    merged["verification_reason"] = str(judgment.get("verification_reason", ""))
                    merged["verifier_model"] = self.config.model
                    merged["run_id"] = self.run_id
                    merged["revised_subject"] = judgment.get("subject", original.get("subject"))
                    merged["revised_predicate"] = judgment.get("predicate", original.get("predicate"))
                    merged["revised_object"] = judgment.get("object", original.get("object"))

                    # Use revised fields only if the model still marks the fact supported.
                    if merged["status"] == "supported":
                        merged["subject"] = str(merged["revised_subject"]).strip()
                        merged["predicate"] = sanitize_predicate(str(merged["revised_predicate"]))
                        merged["object"] = str(merged["revised_object"]).strip()
                    verified.append(merged)

            for vid, original in by_id.items():
                if str(vid) in seen_ids:
                    continue
                missing = dict(original)
                missing["status"] = "rejected"
                missing["verification_reason"] = "No verifier judgment returned."
                missing["verifier_model"] = self.config.model
                missing["run_id"] = self.run_id
                verified.append(missing)

            polite_sleep(self.config.sleep_seconds)

        return verified

    def chat_json(self, user_prompt: str) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract and verify knowledge-graph facts. "
                    "Return only valid JSON. Do not include markdown, explanations, or code fences."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]

        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if self.config.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as first_error:
            # Some vLLM/model combinations do not support JSON mode.
            if self.config.use_json_mode:
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise first_error

        content = response.choices[0].message.content or "{}"
        return parse_json_object(content)

    def insert_facts_neo4j(self, facts: List[Fact]) -> None:
        """Delegate to Neo4jGraph.insert_facts. Preserved as a method for
        back-compat with existing tests and external callers.
        """
        if self.graph is None:
            raise RuntimeError("Neo4j is not configured for this pipeline.")
        self.graph.insert_facts(facts)


def load_longbench_examples(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        examples = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        return examples

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Common wrappers: {"data": [...]}, {"examples": [...]}, etc.
        for key in ("data", "examples", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    raise ValueError("Input must be a JSON object, JSON list, or JSONL file.")


def flatten_context_to_sentence_records(example: Dict[str, Any]) -> List[SentenceRecord]:
    """
    Handles two common forms:
    1. LongBench-v2 context as one long string.
    2. HotPotQA-like context as [[title, [sentences...]], ...].
    """
    context = example.get("context", "")
    example_id = str(example.get("_id", "unknown"))
    records: List[SentenceRecord] = []
    global_id = 0

    if isinstance(context, str):
        for sentence in split_text_to_sentence_like_units(context):
            records.append(
                {
                    "title": example_id,
                    "sent_id": global_id,
                    "text": sentence,
                }
            )
            global_id += 1
        return records

    if isinstance(context, list):
        for block_index, block in enumerate(context):
            title = f"{example_id}_block_{block_index}"
            sentences: Iterable[Any]

            if isinstance(block, list) and len(block) == 2 and isinstance(block[1], list):
                title = str(block[0])
                sentences = block[1]
            elif isinstance(block, dict):
                title = str(block.get("title", title))
                raw = block.get("sentences", block.get("text", ""))
                sentences = raw if isinstance(raw, list) else split_text_to_sentence_like_units(str(raw))
            else:
                sentences = split_text_to_sentence_like_units(str(block))

            for local_sent_id, sentence in enumerate(sentences):
                sentence_text = str(sentence).strip()
                if not sentence_text:
                    continue
                records.append(
                    {
                        "title": title,
                        "sent_id": global_id,
                        "local_sent_id": local_sent_id,
                        "text": sentence_text,
                    }
                )
                global_id += 1
        return records

    return [
        {
            "title": example_id,
            "sent_id": 0,
            "text": str(context),
        }
    ]


def split_text_to_sentence_like_units(text: str, max_unit_chars: int = 1200) -> List[str]:
    """
    Lightweight splitter with no external NLP dependency.
    For normal prose, it splits near sentence boundaries.
    For code/tables/very long paragraphs, it falls back to fixed-size chunks.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    rough = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'([{])", text)
    units: List[str] = []

    for piece in rough:
        piece = piece.strip()
        if not piece:
            continue
        if len(piece) <= max_unit_chars:
            units.append(piece)
        else:
            for start in range(0, len(piece), max_unit_chars):
                units.append(piece[start : start + max_unit_chars].strip())

    return units


def chunk_sentence_records(records: List[SentenceRecord], max_chars: int) -> List[List[SentenceRecord]]:
    chunks: List[List[SentenceRecord]] = []
    current: List[SentenceRecord] = []
    current_chars = 0

    for record in records:
        record_chars = len(record.get("text", "")) + 80
        if current and current_chars + record_chars > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(record)
        current_chars += record_chars

    if current:
        chunks.append(current)
    return chunks


def make_run_id(config: PipelineConfig) -> str:
    payload = json.dumps(
        {
            "input": str(config.input_path),
            "output": str(config.output_path),
            "model": config.model,
            "session": config.neo4j_session_id,
            "started_at": int(time.time()),
        },
        sort_keys=True,
    )
    return f"run_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def _maybe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _span_for_support(sentence_text: str, support_text: str) -> tuple[Optional[int], Optional[int]]:
    support = support_text.strip()
    if not sentence_text or not support:
        return None, None
    start = sentence_text.find(support)
    if start >= 0:
        return start, start + len(support)
    compact_sentence = re.sub(r"\s+", " ", sentence_text)
    compact_support = re.sub(r"\s+", " ", support)
    start = compact_sentence.find(compact_support)
    if start >= 0:
        return start, start + len(compact_support)
    return None, None


def _provenance_entries_from_support(
    *,
    support_text: str,
    chunk: List[SentenceRecord],
    example_id: str,
) -> List[Dict[str, Any]]:
    """Recover sentence provenance when the extractor cites text but no sent_id."""
    support = support_text.strip()
    if not support:
        return []
    recovered: List[Dict[str, Any]] = []
    for record in chunk:
        start, end = _span_for_support(str(record.get("text", "")), support)
        if start is None or end is None:
            continue
        document_id = str(record.get("document_id") or example_id)
        recovered.append(
            {
                "title": str(record.get("title") or example_id),
                "sent_id": record.get("sent_id"),
                "document_id": document_id,
                "sentence_id": f"{document_id}:{record.get('sent_id')}",
                "local_sent_id": record.get("local_sent_id"),
                "source_span_start": start,
                "source_span_end": end,
            }
        )
    return recovered[:3]


def enrich_fact_provenance(
    fact: Fact,
    *,
    example: Dict[str, Any],
    chunk: List[SentenceRecord],
    chunk_index: int,
    extractor_model: str,
    verifier_model: str,
    run_id: str,
) -> None:
    """Attach auditable document/sentence/span/model/run metadata in-place."""
    example_id = str(example.get("_id", "unknown"))
    fact["document_id"] = str(fact.get("document_id") or example_id)
    fact["extractor_model"] = extractor_model
    if verifier_model:
        fact["verifier_model"] = verifier_model
    fact["run_id"] = run_id

    records_by_sent = {str(record.get("sent_id")): record for record in chunk}
    provenance = fact.get("provenance", [])
    if not isinstance(provenance, list):
        provenance = []

    enriched: List[Dict[str, Any]] = []
    support_text = str(fact.get("support_text", "")).strip()
    if not provenance:
        provenance = _provenance_entries_from_support(
            support_text=support_text,
            chunk=chunk,
            example_id=example_id,
        )
    for entry in provenance:
        if not isinstance(entry, dict):
            continue
        out = dict(entry)
        sent_key = str(out.get("sent_id", ""))
        record = records_by_sent.get(sent_key)
        title = str(out.get("title") or (record or {}).get("title") or example_id)
        document_id = str(out.get("document_id") or example_id)
        out["title"] = title
        out["document_id"] = document_id
        out["sentence_id"] = str(
            out.get("sentence_id")
            or f"{document_id}:{out.get('sent_id', sent_key)}"
        )
        out["chunk_index"] = out.get("chunk_index", chunk_index)
        if record is not None:
            out["local_sent_id"] = out.get("local_sent_id", record.get("local_sent_id"))
            start, end = _span_for_support(str(record.get("text", "")), support_text)
            if start is not None and end is not None:
                existing_start = _maybe_int(out.get("source_span_start"))
                existing_end = _maybe_int(out.get("source_span_end"))
                out["source_span_start"] = existing_start if existing_start is not None else start
                out["source_span_end"] = existing_end if existing_end is not None else end
        out["extractor_model"] = str(out.get("extractor_model") or extractor_model)
        if verifier_model:
            out["verifier_model"] = str(out.get("verifier_model") or verifier_model)
        out["run_id"] = str(out.get("run_id") or run_id)
        enriched.append(out)

    fact["provenance"] = enriched


def build_extraction_prompt(example: Dict[str, Any], chunk: List[SentenceRecord], chunk_index: int) -> str:
    question_block = build_question_block(example)
    context_block = json.dumps(chunk, ensure_ascii=False)

    return f"""
You are extracting atomic facts from a LongBench-v2-style example.

Task metadata:
{question_block}

Context chunk index: {chunk_index}
Sentence records JSON:
{context_block}

Extract ONLY facts explicitly supported by the sentence records. Do not infer facts that are merely plausible.

Return this exact JSON shape:
{{
  "facts": [
    {{
      "subject": "canonical entity name",
      "predicate": "UPPER_SNAKE_CASE_RELATION",
      "object": "canonical entity/value name",
      "qualifiers": {{}},
      "temporal": {{"valid_from": null, "valid_to": null}},
      "provenance": [{{"title": "...", "sent_id": 0}}],
      "support_text": "exact supporting sentence(s)",
      "question_relevance": "why this fact could help answer the current multiple-choice question",
      "confidence": "supported",
      "normalization_notes": "alias/pronoun decisions, or empty string"
    }}
  ]
}}

Rules:
- One fact per subject-predicate-object claim.
- Keep claims atomic.
- Use provenance sent_id values from the sentence records.
- Use temporal.valid_from / temporal.valid_to only when the text gives an
  explicit time range, date, year, or event time; otherwise leave both null.
- If no useful facts are explicitly supported, return {{"facts": []}}.
""".strip()


def build_verification_prompt(example: Dict[str, Any], facts: List[Fact]) -> str:
    question_block = build_question_block(example)
    facts_json = json.dumps(facts, ensure_ascii=False, indent=2)

    return f"""
Verify extracted facts against their cited support_text/provenance.

Task metadata:
{question_block}

Facts to verify:
{facts_json}

For each fact, answer these self-reflection checks:
1. Is the fact explicitly supported by the support_text?
2. Is it atomic, not a bundle of multiple claims?
3. Are subject/object aliases and pronouns resolved correctly?
4. Is the predicate specific and meaningful?
5. Is the fact useful or potentially useful for answering the question?
6. Is there any unsupported inference?

Return this exact JSON shape:
{{
  "verified_facts": [
    {{
      "verification_id": "f0",
      "status": "supported | uncertain | rejected",
      "verification_reason": "brief reason",
      "subject": "possibly corrected subject",
      "predicate": "POSSIBLY_CORRECTED_PREDICATE",
      "object": "possibly corrected object"
    }}
  ]
}}

Mark as supported only if the claim is explicit in the cited support text.
""".strip()


def build_question_block(example: Dict[str, Any]) -> str:
    fields = {
        "_id": example.get("_id"),
        "domain": example.get("domain"),
        "sub_domain": example.get("sub_domain"),
        "difficulty": example.get("difficulty"),
        "length": example.get("length"),
        "question": example.get("question"),
        "choice_A": example.get("choice_A"),
        "choice_B": example.get("choice_B"),
        "choice_C": example.get("choice_C"),
        "choice_D": example.get("choice_D"),
    }
    return json.dumps(fields, ensure_ascii=False, indent=2)


def clean_fact(fact: Fact) -> Optional[Fact]:
    subject = str(fact.get("subject", "")).strip()
    predicate = sanitize_predicate(str(fact.get("predicate", "")))
    obj = str(fact.get("object", "")).strip()

    if not subject or not predicate or not obj:
        return None

    provenance = fact.get("provenance", [])
    if not isinstance(provenance, list):
        provenance = []

    qualifiers = fact.get("qualifiers", {})
    if not isinstance(qualifiers, dict):
        qualifiers = {}
    temporal = fact.get("temporal", {})
    if not isinstance(temporal, dict):
        temporal = {}
    clean_temporal = {
        "valid_from": temporal.get("valid_from"),
        "valid_to": temporal.get("valid_to"),
    }
    if not clean_temporal["valid_from"] and not clean_temporal["valid_to"]:
        clean_temporal = {}

    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "qualifiers": qualifiers,
        "temporal": clean_temporal,
        "provenance": provenance,
        "support_text": str(fact.get("support_text", "")).strip(),
        "question_relevance": str(fact.get("question_relevance", "")).strip(),
        "confidence": normalize_status(fact.get("confidence", "supported")),
        "normalization_notes": str(fact.get("normalization_notes", "")).strip(),
    }


def sanitize_predicate(predicate: str) -> str:
    pred = predicate.strip().upper()
    pred = re.sub(r"[^A-Z0-9_]+", "_", pred)
    pred = re.sub(r"_+", "_", pred).strip("_")
    return pred or "RELATED_TO"


def normalize_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    if "support" in status and "not" not in status:
        return "supported"
    if "reject" in status or "false" in status or "not" in status:
        return "rejected"
    if "uncertain" in status or "partial" in status or "maybe" in status:
        return "uncertain"
    return status if status in {"supported", "uncertain", "rejected"} else "uncertain"


def make_fact_id(example_id: str, fact: Fact) -> str:
    provenance_for_id = []
    for entry in fact.get("provenance", []) or []:
        if isinstance(entry, dict):
            provenance_for_id.append(
                {
                    "title": entry.get("title"),
                    "sent_id": entry.get("sent_id"),
                    "document_id": entry.get("document_id"),
                    "sentence_id": entry.get("sentence_id"),
                }
            )
    temporal = fact.get("temporal") if isinstance(fact.get("temporal"), dict) else {}
    stable = json.dumps(
        {
            "example_id": example_id,
            "subject": fact.get("subject"),
            "predicate": fact.get("predicate"),
            "object": fact.get("object"),
            "support_text": fact.get("support_text"),
            "provenance": provenance_for_id,
            "temporal": {
                "valid_from": temporal.get("valid_from") or fact.get("valid_from"),
                "valid_to": temporal.get("valid_to") or fact.get("valid_to"),
            },
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:24]


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    # Fallback: pull out the largest likely JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def polite_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Extract verified KG facts from LongBench-v2-style data.")
    parser.add_argument("--input", required=True, type=Path, help="Path to LongBench-v2 JSON or JSONL file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path for supported facts.")
    parser.add_argument("--rejections-output", type=Path, default=None,
                        help="Output JSONL path for rejected fact artifacts. Defaults to <output_stem>_rejections.jsonl.")
    parser.add_argument("--model", required=True, help="Model name served by vLLM, e.g. Qwen/Qwen3-4B.")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"), help="API key for vLLM server; often EMPTY locally.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--chunk-chars", type=int, default=12000, help="Approx max characters per context chunk.")
    parser.add_argument("--max-chunks-per-example", type=int, default=None, help="Debug option to cap chunks per example.")
    parser.add_argument(
        "--chunk-selection",
        choices=sorted(CHUNK_SELECTION_MODES),
        default="hybrid",
        help="How to choose chunks when --max-chunks-per-example truncates the context.",
    )
    parser.add_argument("--chunk-selection-rrf-k", type=int, default=60)
    parser.add_argument("--chunk-embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-embedding-revision", default=None)
    parser.add_argument("--chunk-embedding-device", default="cpu")
    parser.add_argument("--chunk-embedding-batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Debug option to cap number of examples.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Optional delay between model calls.")
    parser.add_argument("--use-json-mode", action="store_true", help="Try OpenAI JSON mode; fallback if unsupported.")
    parser.add_argument("--verify-batch-size", type=int, default=20, help="Number of facts verified per LLM call. Lower this when running with a small --max-model-len.")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI"), help="Optional, e.g. bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE"), help="Optional Neo4j database name.")
    parser.add_argument("--session-id", default=os.getenv("NEO4J_SESSION_ID"), help="Tag writes/queries with this session id (defaults to 'default').")
    parser.add_argument("--stateless", action="store_true", help="Wipe the session_id subgraph on close. Requires --session-id.")

    args = parser.parse_args()
    return PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        vllm_base_url=args.vllm_base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        chunk_chars=args.chunk_chars,
        max_chunks_per_example=args.max_chunks_per_example,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        use_json_mode=args.use_json_mode,
        verify_batch_size=args.verify_batch_size,
        run_id=None,
        rejections_output_path=args.rejections_output,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        neo4j_session_id=args.session_id,
        neo4j_stateless=args.stateless,
        chunk_selection=args.chunk_selection,
        chunk_selection_rrf_k=args.chunk_selection_rrf_k,
        chunk_embedding_model=args.chunk_embedding_model,
        chunk_embedding_revision=args.chunk_embedding_revision,
        chunk_embedding_device=args.chunk_embedding_device,
        chunk_embedding_batch_size=args.chunk_embedding_batch_size,
    )


def main() -> None:
    config = parse_args()
    pipeline = LongBenchKGPipeline(config)
    try:
        pipeline.run()
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
