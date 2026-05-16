#!/usr/bin/env python3
"""
evaluate.py - Test the full pipeline on LongBench multiple-choice QA.

Modes:
  --mode kg       Use extracted facts from verified_facts.jsonl as context
  --mode raw      Use the raw document context (baseline, no KG)

Usage:
    python3 evaluate.py \
        --data data.jsonl \
        --facts verified_facts.jsonl \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --vllm-base-url http://localhost:8000/v1 \
        --mode kg \
        --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from experiments.common import extract_letter


def load_jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_facts_index(facts: List[Dict]) -> Dict[str, List[Dict]]:
    """Group facts by example_id for fast lookup."""
    index = defaultdict(list)
    for fact in facts:
        index[fact["example_id"]].append(fact)
    return dict(index)


def facts_to_context(facts: List[Dict], max_chars: int = 4000) -> str:
    """Format facts as a readable context block for the LLM."""
    lines = []
    used = 0
    for i, f in enumerate(facts, start=1):
        line = f"[F{i}] {f['subject']} -{f['predicate']}-> {f['object']}"
        if f.get("support_text"):
            line += f"\n     evidence: \"{f['support_text']}\""
        if used + len(line) > max_chars:
            lines.append(f"... [{len(facts) - i + 1} more facts truncated]")
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def build_prompt(example: Dict, context: str) -> str:
    question = example["question"]
    a = example["choice_A"]
    b = example["choice_B"]
    c = example["choice_C"]
    d = example["choice_D"]

    return f"""You are given the following information:

{context}

Based on the information above, answer the following multiple-choice question.
Respond with ONLY the letter A, B, C, or D.

Question: {question}

A) {a}
B) {b}
C) {c}
D) {d}

Answer:"""




def evaluate(
    data_path: Path,
    facts_path: Optional[Path],
    model: str,
    vllm_base_url: str,
    mode: str,
    limit: Optional[int],
    sleep: float,
    output_path: Optional[Path],
) -> None:
    examples = load_jsonl(data_path)
    if limit:
        examples = examples[:limit]

    facts_index = {}
    if mode == "kg" and facts_path:
        facts = load_jsonl(facts_path)
        facts_index = build_facts_index(facts)
        print(f"Loaded {len(facts)} facts for {len(facts_index)} examples", file=sys.stderr)

    client = OpenAI(base_url=vllm_base_url, api_key="EMPTY")

    results = []
    correct = 0
    total = 0
    skipped = 0

    for i, example in enumerate(examples, start=1):
        example_id = str(example.get("_id", f"example_{i}"))
        ground_truth = example.get("answer", "").strip().upper()

        if mode == "kg":
            example_facts = facts_index.get(example_id, [])
            if not example_facts:
                skipped += 1
                print(f"[{i}/{len(examples)}] {example_id} — no facts, skipping", file=sys.stderr)
                continue
            context = facts_to_context(example_facts)
        else:
            raw_context = example.get("context", "")
            context = raw_context[:4000] + ("..." if len(raw_context) > 4000 else "")

        prompt = build_prompt(example, context)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            raw_output = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[{i}/{len(examples)}] {example_id} — API error: {e}", file=sys.stderr)
            skipped += 1
            continue

        predicted = extract_letter(raw_output)
        is_correct = predicted == ground_truth
        if predicted:
            total += 1
            if is_correct:
                correct += 1

        results.append({
            "example_id": example_id,
            "question": example["question"],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "raw_output": raw_output,
            "correct": is_correct,
            "domain": example.get("domain", ""),
            "difficulty": example.get("difficulty", ""),
            "mode": mode,
        })

        accuracy = correct / total if total > 0 else 0.0
        print(
            f"[{i}/{len(examples)}] {example_id} — "
            f"gt={ground_truth} pred={predicted} "
            f"{'✓' if is_correct else '✗'} | "
            f"acc={accuracy:.1%} ({correct}/{total})",
            file=sys.stderr,
        )

        if sleep > 0:
            time.sleep(sleep)

    accuracy = correct / total if total > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"Mode:     {mode}")
    print(f"Model:    {model}")
    print(f"Examples: {len(examples)} (evaluated={total}, skipped={skipped})")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"{'='*50}")

    # Per-domain breakdown
    domain_stats: Dict[str, Dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if r["predicted"] is not None:
            domain_stats[r["domain"]]["total"] += 1
            if r["correct"]:
                domain_stats[r["domain"]]["correct"] += 1

    if domain_stats:
        print("\nPer-domain accuracy:")
        for domain, stats in sorted(domain_stats.items()):
            d_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {domain}: {d_acc:.1%} ({stats['correct']}/{stats['total']})")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data.jsonl"))
    parser.add_argument("--facts", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--mode", choices=["kg", "raw"], default="kg",
                        help="kg=use extracted facts, raw=use raw context")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    evaluate(
        data_path=args.data,
        facts_path=args.facts,
        model=args.model,
        vllm_base_url=args.vllm_base_url,
        mode=args.mode,
        limit=args.limit,
        sleep=args.sleep,
        output_path=args.output,
    )
