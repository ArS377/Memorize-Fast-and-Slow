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
    vllm serve Qwen/Qwen3.5-4B --host 0.0.0.0 --port 8000

Example run:
    python longbench_kg_pipeline.py \
        --input longbench_v2_sample.jsonl \
        --output verified_facts.jsonl \
        --model Qwen/Qwen3.5-4B \
        --vllm-base-url http://localhost:8000/v1

Example with Neo4j:
    python longbench_kg_pipeline.py \
        --input longbench_v2_sample.jsonl \
        --output verified_facts.jsonl \
        --model Qwen/Qwen3.5-4B \
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

from neo4j_graph import Neo4jGraph


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


class LongBenchKGPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.client = OpenAI(base_url=config.vllm_base_url, api_key=config.api_key)
        self.graph: Optional[Neo4jGraph] = None

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

    def run(self) -> None:
        examples = load_longbench_examples(self.config.input_path)
        if self.config.limit is not None:
            examples = examples[: self.config.limit]

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        total_supported = 0

        with self.config.output_path.open("w", encoding="utf-8") as out:
            for i, example in enumerate(examples, start=1):
                example_id = str(example.get("_id", f"example_{i}"))
                print(f"[{i}/{len(examples)}] Processing {example_id}", file=sys.stderr)

                sentence_records = flatten_context_to_sentence_records(example)
                chunks = chunk_sentence_records(sentence_records, self.config.chunk_chars)

                if self.config.max_chunks_per_example is not None:
                    chunks = chunks[: self.config.max_chunks_per_example]

                extracted: List[Fact] = []
                for chunk_index, chunk in enumerate(chunks):
                    facts = self.extract_facts(example, chunk, chunk_index)
                    extracted.extend(facts)
                    polite_sleep(self.config.sleep_seconds)

                verified = self.verify_facts(example, extracted)
                supported = [f for f in verified if normalize_status(f.get("status") or f.get("confidence")) == "supported"]

                for fact in supported:
                    fact["example_id"] = example_id
                    fact["fact_id"] = make_fact_id(example_id, fact)
                    # FIX 3: propagate the question from the source example so
                    # insert_facts_neo4j can store it on the relationship.
                    # Without this, fact.get("question") is always "" in Neo4j.
                    fact["question"] = str(example.get("question", ""))
                    out.write(json.dumps(fact, ensure_ascii=False) + "\n")

                if self.graph is not None and supported:
                    self.graph.insert_facts(supported)

                total_supported += len(supported)
                print(
                    f"    extracted={len(extracted)} supported={len(supported)} total_supported={total_supported}",
                    file=sys.stderr,
                )

        print(f"Done. Wrote {total_supported} supported facts to {self.config.output_path}", file=sys.stderr)

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
            if isinstance(judgments, list):
                for judgment in judgments:
                    if not isinstance(judgment, dict):
                        continue
                    vid = judgment.get("verification_id")
                    original = by_id.get(str(vid))
                    if not original:
                        continue
                    merged = dict(original)
                    merged["status"] = normalize_status(judgment.get("status"))
                    merged["verification_reason"] = str(judgment.get("verification_reason", ""))
                    merged["revised_subject"] = judgment.get("subject", original.get("subject"))
                    merged["revised_predicate"] = judgment.get("predicate", original.get("predicate"))
                    merged["revised_object"] = judgment.get("object", original.get("object"))

                    # Use revised fields only if the model still marks the fact supported.
                    if merged["status"] == "supported":
                        merged["subject"] = str(merged["revised_subject"]).strip()
                        merged["predicate"] = sanitize_predicate(str(merged["revised_predicate"]))
                        merged["object"] = str(merged["revised_object"]).strip()
                    verified.append(merged)

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
        "answer": example.get("answer"),
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

    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "qualifiers": qualifiers,
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
    stable = json.dumps(
        {
            "example_id": example_id,
            "subject": fact.get("subject"),
            "predicate": fact.get("predicate"),
            "object": fact.get("object"),
            "support_text": fact.get("support_text"),
            "provenance": fact.get("provenance", []),
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
    parser.add_argument("--model", required=True, help="Model name served by vLLM, e.g. Qwen/Qwen3.5-4B.")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1", help="vLLM OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"), help="API key for vLLM server; often EMPTY locally.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--chunk-chars", type=int, default=12000, help="Approx max characters per context chunk.")
    parser.add_argument("--max-chunks-per-example", type=int, default=None, help="Debug option to cap chunks per example.")
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
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        neo4j_session_id=args.session_id,
        neo4j_stateless=args.stateless,
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
