"""Common helpers shared across all six ablation cells.

This module is intentionally dependency-light: it must be importable
without ``neo4j``, ``rlm``, ``openai``, or ``scallopy`` installed, so the
aggregator can run anywhere.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    ``limit``. This guarantees that every cell scores on identical example
    IDs without needing to coordinate via a side-channel.
    """
    examples = load_examples(path)
    examples.sort(key=lambda ex: str(ex.get("_id", "")))
    if limit is not None:
        examples = examples[:limit]
    return examples


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
    row = {k: fields.get(k) for k in CELL_SCHEMA}
    fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def truncate_context(context: str, max_chars: Optional[int]) -> str:
    if max_chars is None or max_chars <= 0:
        return context
    if len(context) <= max_chars:
        return context
    return context[:max_chars]
