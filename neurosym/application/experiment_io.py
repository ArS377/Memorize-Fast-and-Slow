"""Common helpers shared across all six ablation cells.

This module is intentionally dependency-light: it must be importable
without ``neo4j``, ``rlm``, ``openai``, or ``scallopy`` installed, so the
aggregator can run anywhere.
"""

from __future__ import annotations

import json
import re
from bisect import insort
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from neurosym.application.cells import serialize_experiment_result

# Single source of truth for the result-row schema.
CELL_SCHEMA: List[str] = [
    "cell_id",
    "label",
    "example_id",
    "predicted",
    "gold",
    "correct",
    "n_context_chars",
    "n_triples",
    "elapsed_seconds",
    "error",
    "run_id",
    "session_id",
    "memory_scope",
    "orchestration_mode",
    "validator_backend",
    "configured_retrieval_mode",
    "effective_retrieval_mode",
    "retrieval_degraded",
    "embedding_model",
    "embedding_revision",
    "embedding_device",
    "embedding_batch_size",
    "dense_index_root",
    "dense_failure_policy",
    "rrf_k",
    "dense_index_identity",
    "retrieval_branch_counts",
    "retrieval_branch_latency_seconds",
    "retrieval_rrf_settings",
    "termination_mode",
    "termination_reason",
    "order_gap_final",
    "order_gap_window_mean",
    "rlm_completion_count",
    "diagnostic_predicted",
    "diagnostic_correct",
]

# Cell metadata table (cell_id -> info). Used by the aggregator and the
# orchestrator to keep cell identity and ordering consistent.
CELLS: List[Dict[str, Any]] = [
    {"cell_id": 1, "label": "flat_raw", "retrieval": "raw", "validator": "n/a", "recursion": "flat"},
    {"cell_id": 2, "label": "flat_kg_noscallop", "retrieval": "kg", "validator": "none", "recursion": "flat"},
    {"cell_id": 3, "label": "flat_kg_scallop", "retrieval": "kg", "validator": "scallop", "recursion": "flat"},
    {"cell_id": 4, "label": "rlm_raw", "retrieval": "raw", "validator": "n/a", "recursion": "rlm"},
    {"cell_id": 5, "label": "rlm_kg_noscallop", "retrieval": "kg", "validator": "none", "recursion": "rlm"},
    {"cell_id": 6, "label": "rlm_kg_scallop", "retrieval": "kg", "validator": "scallop", "recursion": "rlm"},
]


def cell_output_path(cell_id: int, label: str, results_dir: Path = Path("results")) -> Path:
    return results_dir / f"cell{cell_id}_{label}" / "results.jsonl"


def load_examples(path: Path) -> List[Dict[str, Any]]:
    """Load all JSONL examples (no limit). Mirrors rlm_baseline.load_examples."""
    examples: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def iter_pilot_examples(
    path: Path,
    limit: Optional[int],
    seed: int = 0,  # reserved for future randomization; deterministic for now
) -> List[Dict[str, Any]]:
    """Stable pilot slice: sort all examples by ``_id`` and take the first
    ``limit``. Limited runs retain only the requested examples in memory, which
    matters for LongBench rows with very large contexts. This guarantees that
    every cell scores on identical example IDs without a side-channel.
    """
    if limit is None:
        examples = load_examples(path)
        examples.sort(key=lambda ex: str(ex.get("_id", "")))
        return examples
    if limit <= 0:
        return []

    selected: List[tuple[tuple[str, int], Dict[str, Any]]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            insort(selected, ((str(example.get("_id", "")), position), example))
            if len(selected) > limit:
                selected.pop()
    return [example for _key, example in selected]


def format_question(ex: Dict[str, Any]) -> str:
    """Format a LongBench-v2 multiple-choice example as a question prompt."""
    return (
        f"Question: {ex.get('question', '')}\n"
        f"A) {ex.get('choice_A', '')}\n"
        f"B) {ex.get('choice_B', '')}\n"
        f"C) {ex.get('choice_C', '')}\n"
        f"D) {ex.get('choice_D', '')}\n"
        f"\nAnswer with only the letter A, B, C, or D."
    )


def extract_letter(text: str) -> str:
    """Pull the answer letter (A/B/C/D) out of the model output.

    Order of preference, strongest signal first:
      1. Explicit pattern like ``Answer: X`` / ``Final answer: X`` (case-insensitive).
      2. Strip Qwen3 ``<think>...</think>`` block, then a *standalone* A/B/C/D
         token (e.g. line that is just ``A`` or ``A.``) -- the model's actual
         answer typically sits at the end of the response.
      3. Last standalone A/B/C/D anywhere in the (post-think) text.
      4. Last ``A``/``B``/``C``/``D`` character anywhere.

    Old behaviour returned the *first* A/B/C/D character in the entire string,
    which on Qwen3 caused phantom ``A`` predictions because the word
    ``Answer`` (or ``Analyze``, ``About``, ...) appears very early in the
    chain-of-thought.
    """
    if not text:
        return ""
    m = re.search(r"(?:final\s+answer|answer)\s*[:\-]?\s*\(?([ABCD])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Strip any <think>...</think> block (Qwen3) before further parsing.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip() or text  # if nothing left, fall back to original
    # Look for a standalone letter token (e.g. line "A" or "A." or "(A)").
    standalones = re.findall(r"(?:^|\W)([ABCD])(?:\W|$)", cleaned)
    if standalones:
        return standalones[-1].upper()
    # Fallback: last A/B/C/D character anywhere.
    for ch in reversed(cleaned.upper()):
        if ch in "ABCD":
            return ch
    return ""


def write_result_row(fp, **fields: Any) -> None:
    """Write one schema-conformant JSON line to ``fp``. Missing keys default to None."""
    row = serialize_experiment_result(fields, CELL_SCHEMA)
    fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_context(context: str, max_chars: Optional[int]) -> str:
    if max_chars is None or max_chars <= 0:
        return context
    if len(context) <= max_chars:
        return context
    return context[:max_chars]
