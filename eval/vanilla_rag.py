from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from rank_bm25 import BM25Okapi

from longbench_kg_pipeline import chunk_sentence_records, flatten_context_to_sentence_records


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def build_chunks(example: Dict[str, Any], chunk_chars: int) -> List[Dict[str, Any]]:
    records = flatten_context_to_sentence_records(example.get("raw_example", example))
    chunked = chunk_sentence_records(records, chunk_chars)
    chunks: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunked):
        text = " ".join(str(r.get("text", "")).strip() for r in chunk if str(r.get("text", "")).strip())
        if not text:
            continue
        sent_ids = [int(r.get("sent_id", -1)) for r in chunk if isinstance(r.get("sent_id"), int)]
        chunks.append(
            {
                "chunk_id": idx,
                "text": text,
                "sent_id_min": min(sent_ids) if sent_ids else None,
                "sent_id_max": max(sent_ids) if sent_ids else None,
            }
        )
    return chunks


def retrieve_top_k(question: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    corpus_tokens = [_tokenize(ch["text"]) for ch in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(_tokenize(question))
    ranked = sorted(zip(chunks, scores), key=lambda x: float(x[1]), reverse=True)
    out: List[Dict[str, Any]] = []
    for ch, score in ranked[:top_k]:
        row = dict(ch)
        row["score"] = float(score)
        out.append(row)
    return out


def build_qa_prompt(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(f"[{i}] {chunk['text']}")
    joined = "\n\n".join(context_blocks)
    return (
        "Use only the retrieved context to answer the question.\n"
        "If the context is insufficient, say: INSUFFICIENT_CONTEXT.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved context:\n{joined}\n\n"
        "Answer:"
    )


def generate_answer(
    prompt: str,
    *,
    model: str,
    vllm_base_url: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> str:
    client = OpenAI(base_url=vllm_base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise QA assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def run_vanilla_rag(
    examples: List[Dict[str, Any]],
    *,
    model: str,
    vllm_base_url: str,
    api_key: str,
    chunk_chars: int = 2500,
    top_k: int = 4,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, example in enumerate(examples, start=1):
        question = str(example.get("question", "")).strip()
        chunks = build_chunks(example, chunk_chars=chunk_chars)
        retrieved = retrieve_top_k(question, chunks, top_k=top_k)
        prompt = build_qa_prompt(question, retrieved)
        prediction = generate_answer(
            prompt,
            model=model,
            vllm_base_url=vllm_base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        rows.append(
            {
                "index": idx - 1,
                "id": example.get("id", ""),
                "dataset_name": example.get("dataset_name", ""),
                "question": question,
                "answers": example.get("answers", []),
                "prediction": prediction,
                "retrieved_chunks": retrieved,
            }
        )
    return rows


def write_predictions(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

