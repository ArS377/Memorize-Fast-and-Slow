#!/usr/bin/env python3
"""
Usage (ollama):
    python rlm_baseline.py \
        --input data.jsonl \
        --output rlm_baseline_output.jsonl \
        --backend vllm \
        --base-url http://localhost:11434/v1 \
        --model qwen2.5:1.5b \
        --limit 5

Usage (vllm-metal):
    python rlm_baseline.py \
        --input data.jsonl \
        --output rlm_baseline_output.jsonl \
        --backend vllm \
        --base-url http://localhost:8001/v1 \
        --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
        --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rlm.core.rlm import RLM
from rlm.logger.rlm_logger import RLMLogger


def load_examples(path: Path, limit: int | None) -> list[dict]:
    examples = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
            if limit and len(examples) >= limit:
                break
    return examples


def format_question(ex: dict) -> str:
    """Format LongBench multiple-choice question as the RLM root_prompt."""
    return (
        f"Question: {ex.get('question', '')}\n"
        f"A) {ex.get('choice_A', '')}\n"
        f"B) {ex.get('choice_B', '')}\n"
        f"C) {ex.get('choice_C', '')}\n"
        f"D) {ex.get('choice_D', '')}\n"
        f"\nAnswer with only the letter A, B, C, or D."
    )


def extract_letter(text: str) -> str:
    """Pull the first A/B/C/D out of whatever the RLM returns."""
    if not text:
        return ""
    for ch in text.upper():
        if ch in "ABCD":
            return ch
    return text.strip()[:1].upper()


def main():
    parser = argparse.ArgumentParser(description="RLM baseline on LongBench-v2.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="vllm", choices=[
        "openai", "vllm", "litellm", "anthropic", "azure_openai", "gemini"
    ])
    parser.add_argument("--base-url", default="http://localhost:11434/v1",
                        help="OpenAI-compatible base URL (ollama or vllm-metal)")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=2,
                        help="RLM recursion depth (1=no recursion, 2+=recursive)")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument("--log-dir", default="./rlm_logs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Set up logger
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = RLMLogger(log_dir=str(log_dir))

    # Backend kwargs — same pattern as your existing pipeline's OpenAI client
    backend_kwargs = {
        "model_name": args.model,   # <-- was "model"
        "base_url": args.base_url,
        "api_key": args.api_key,
    }

    print(f"Backend: {args.backend} @ {args.base_url}", file=sys.stderr)
    print(f"Model:   {args.model}", file=sys.stderr)
    print(f"Max depth: {args.max_depth}, Max iterations: {args.max_iterations}",
          file=sys.stderr)

    examples = load_examples(args.input, args.limit)
    print(f"Loaded {len(examples)} examples", file=sys.stderr)

    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as out:
        for i, ex in enumerate(examples, start=1):
            example_id = str(ex.get("_id", f"ex_{i}"))
            context = ex.get("context", "")
            question = format_question(ex)
            gold = str(ex.get("answer", "")).strip().upper()

            print(f"\n[{i}/{len(examples)}] {example_id}", file=sys.stderr)
            print(f"  context: {len(context)} chars | gold: {gold}", file=sys.stderr)

            # Instantiate a fresh RLM per example (stateless baseline)
            rlm = RLM(
                backend=args.backend,
                backend_kwargs=backend_kwargs,
                environment="local",
                max_depth=args.max_depth,
                max_iterations=args.max_iterations,
                max_tokens=args.max_tokens,
                logger=logger,
                verbose=args.verbose,
            )

            t0 = time.time()
            try:
                result = rlm.completion(
                    prompt=context,        # raw LongBench context as the RLM's working memory
                    root_prompt=question,  # the question the RLM must answer
                )
                # RLMChatCompletion — extract the answer string
                raw_answer = str(result) if result else ""
                predicted = extract_letter(raw_answer)
                error = None
            except Exception as e:
                error_str = str(e)
                raw_answer = error_str
                predicted = extract_letter(error_str)
                error = error_str
                print(f"  ERROR: {e}", file=sys.stderr)

            elapsed = time.time() - t0
            correct = (predicted == gold) if predicted and gold else False

            record = {
                "example_id": example_id,
                "predicted": predicted,
                "gold": gold,
                "correct": correct,
                "raw_answer": raw_answer[:200],  # truncate for readability
                "context_chars": len(context),
                "elapsed_seconds": round(elapsed, 2),
                "error": error,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)

            status = "✓" if correct else "✗"
            print(f"  {status} predicted={predicted!r} gold={gold!r} ({elapsed:.1f}s)",
                  file=sys.stderr)

    # Summary
    answered = [r for r in results if r["predicted"] and not r["error"]]
    correct = sum(r["correct"] for r in answered)
    total = len(results)
    errors = sum(1 for r in results if r["error"])

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"Results: {correct}/{len(answered)} correct "
          f"({correct/len(answered)*100:.1f}% of answered)" if answered else
          "Results: 0 answered", file=sys.stderr)
    print(f"Errors: {errors}/{total}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Logs:   {args.log_dir}/", file=sys.stderr)


if __name__ == "__main__":
    main()