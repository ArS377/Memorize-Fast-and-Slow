"""Build KG facts JSONL directly, without requiring Neo4j.

This is the setup path for KG-backed cells when a live Neo4j instance is not
available. It mirrors the extraction, verification, and optional Scallop
validation used by ``experiments.build_kg``, then writes the fallback facts file
consumed by cells 2/3/5/6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurosym.application.experiment_io import iter_pilot_examples


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_kg_facts_file(
    *,
    session_id: str,
    validate: bool,
    input_path: Path,
    model: str,
    vllm_base_url: str,
    api_key: str,
    limit: Optional[int] = None,
    chunk_chars: int = 12000,
    max_chunks_per_example: Optional[int] = 3,
    verify_batch_size: int = 20,
    max_tokens: int = 768,
    facts_out_dir: Path = Path("results/kg_builds"),
    scallop_validator_url: Optional[str] = None,
) -> Path:
    from neurosym.domain.compiled_memory import fact_to_compiled_fact
    from longbench_kg_pipeline import (
        LongBenchKGPipeline,
        PipelineConfig,
        chunk_sentence_records,
        flatten_context_to_sentence_records,
        make_fact_id,
        normalize_status,
    )
    from neurosym.reporting.rejections import append_rejection_jsonl, build_rejection_record
    from neurosym.domain.validation_rules import DEFAULT_RULE_PARAMETERS
    from neurosym.adapters.validation_backend import make_validator_backend

    out_path = facts_out_dir / f"{session_id}_facts.jsonl"
    rejections_path = facts_out_dir / f"{session_id}_rejections.jsonl"
    rejections_path.parent.mkdir(parents=True, exist_ok=True)
    rejections_path.write_text("", encoding="utf-8")

    cfg = PipelineConfig(
        input_path=Path(input_path),
        output_path=Path("/dev/null") if os.name != "nt" else Path("NUL"),
        model=model,
        vllm_base_url=vllm_base_url,
        api_key=api_key,
        temperature=0.0,
        max_tokens=max_tokens,
        chunk_chars=chunk_chars,
        max_chunks_per_example=None,
        limit=None,
        sleep_seconds=0.0,
        use_json_mode=False,
        neo4j_uri=None,
        neo4j_user=None,
        neo4j_password=None,
        verify_batch_size=verify_batch_size,
        run_id=f"build_{session_id}",
    )
    pipeline = LongBenchKGPipeline(cfg)
    examples = iter_pilot_examples(input_path, limit)
    print(
        f"[build_kg_facts_file] session={session_id} validate={validate} "
        f"examples={len(examples)} model={model}",
        file=sys.stderr,
    )

    committed: List[Dict[str, Any]] = []
    params = DEFAULT_RULE_PARAMETERS
    validator_backend = make_validator_backend(
        endpoint=scallop_validator_url,
        require_scallop=validate,
    )
    try:
        for i, example in enumerate(examples, start=1):
            example_id = str(example.get("_id", f"example_{i}"))
            print(
                f"[build_kg_facts_file] ({i}/{len(examples)}) extracting {example_id}",
                file=sys.stderr,
            )

            sentence_records = flatten_context_to_sentence_records(example)
            chunks = chunk_sentence_records(sentence_records, chunk_chars)
            if max_chunks_per_example is not None and max_chunks_per_example > 0:
                chunks = chunks[:max_chunks_per_example]

            extracted: List[Dict[str, Any]] = []
            for ci, chunk in enumerate(chunks):
                extracted.extend(pipeline.extract_facts(example, chunk, ci))

            verified = pipeline.verify_facts(example, extracted)
            for fact in verified:
                fact["example_id"] = example_id
                fact["fact_id"] = make_fact_id(example_id, fact)
                fact["question"] = str(example.get("question", ""))

            supported = []
            verifier_rejected = 0
            validator_rejected = 0
            for fact in verified:
                status = normalize_status(fact.get("status") or fact.get("confidence"))
                if status != "supported":
                    verifier_rejected += 1
                    append_rejection_jsonl(
                        rejections_path,
                        build_rejection_record(
                            candidate_fact=fact,
                            reason=fact.get("verification_reason") or status,
                            example_id=example_id,
                            session_id=session_id,
                            stage="llm_verification",
                            existing_conflicting_fact=None,
                            validator="llm_self_reflection",
                        ),
                    )
                    continue
                supported.append(fact_to_compiled_fact(fact))

            for fact in supported:
                if not validate:
                    committed.append(fact)
                    continue

                validation = validator_backend.validate(committed, fact, rule_params=params)
                if validation.decision == "accept":
                    committed.append(fact)
                elif validation.decision == "replace":
                    if validation.replace_fact_id:
                        committed = [
                            existing
                            for existing in committed
                            if existing.get("fact_id") != validation.replace_fact_id
                        ]
                    committed.append(fact)
                else:
                    validator_rejected += 1
                    label = (
                        validation.rejection_label.to_dict()
                        if validation.rejection_label
                        else None
                    )
                    append_rejection_jsonl(
                        rejections_path,
                        build_rejection_record(
                            candidate_fact=fact,
                            reason=validation.reason,
                            example_id=example_id,
                            session_id=session_id,
                            stage="scallop_validation",
                            existing_conflicting_fact=None,
                            rule_fired=(label or {}).get("code", "validator_reject"),
                            validator=validator_backend.info.name,
                        ),
                    )

            print(
                f"           extracted={len(extracted)} supported={len(supported)} "
                f"llm_rejected={verifier_rejected} "
                f"validator_rejected={validator_rejected} "
                f"committed_total={len(committed)}",
                file=sys.stderr,
            )
    finally:
        pipeline.close()

    _write_jsonl(out_path, committed)
    print(
        f"[build_kg_facts_file] done. session={session_id} mirrored={len(committed)} "
        f"-> {out_path}; rejections -> {rejections_path}",
        file=sys.stderr,
    )
    return out_path


def main(argv: Optional[List[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--input", type=Path, default=Path("data.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--chunk-chars", type=int, default=12000)
    parser.add_argument("--max-chunks-per-example", type=int, default=3)
    parser.add_argument("--verify-batch-size", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--facts-out-dir", type=Path, default=Path("results/kg_builds"))
    parser.add_argument("--scallop-validator-url", default=os.getenv("SCALLOP_VALIDATOR_URL"))
    args = parser.parse_args(argv)
    return build_kg_facts_file(
        session_id=args.session,
        validate=args.validate,
        input_path=args.input,
        model=args.model,
        vllm_base_url=args.vllm_base_url,
        api_key=args.api_key,
        limit=args.limit,
        chunk_chars=args.chunk_chars,
        max_chunks_per_example=args.max_chunks_per_example,
        verify_batch_size=args.verify_batch_size,
        max_tokens=args.max_tokens,
        facts_out_dir=args.facts_out_dir,
        scallop_validator_url=args.scallop_validator_url,
    )


if __name__ == "__main__":
    main()
