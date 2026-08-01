#!/usr/bin/env python3
"""Download HotpotQA and write a validated, normalized data JSONL.

Mirrors ``download_longbench.py``: atomic temp-file write, ASCII-safe lines,
a full round-trip validation pass, and per-field coverage. Output rows use the
shared multi-hop schema (see ``experiments/multihop_datasets``) so the KG build
and retrieval eval consume them unchanged and gold supporting facts map to
extracted fact IDs.

Usage
-----
    pip install datasets                     # in the env that has it
    python download_hotpotqa.py                                  # -> data_hotpotqa.jsonl
    python download_hotpotqa.py --split validation --limit 200
    python download_hotpotqa.py --validate-only --out data_hotpotqa.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from experiments.multihop_datasets import (
    NORMALIZED_FIELDS,
    normalize_hotpotqa_row,
    validate_row,
)

DEFAULT_DATASET = "hotpotqa/hotpot_qa"
DEFAULT_CONFIG = "distractor"


def download(out_path: Path, *, dataset: str, config: str, split: str, limit: int | None) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets package not found. Run: pip install datasets")

    print(f"Fetching {dataset} ({config}, split={split}) from HuggingFace …", flush=True)
    ds = load_dataset(dataset, config, split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"  (capped to {limit} rows via --limit)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=".tmp_hotpotqa_", suffix=".jsonl"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as f:
            for i, row in enumerate(ds):
                normalized = normalize_hotpotqa_row(dict(row))
                f.write(json.dumps(normalized, ensure_ascii=True) + "\n")
                if (i + 1) % 500 == 0:
                    print(f"  wrote {i + 1} / {len(ds)} rows …", flush=True)
        Path(tmp_name).replace(out_path)
        print(f"\nWrote {len(ds)} rows → {out_path}")
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def validate(path: Path) -> None:
    print(f"\nValidating {path} …", flush=True)
    field_missing = {f: 0 for f in NORMALIZED_FIELDS}
    n_rows = 0
    bad_rows: list[tuple[int, str]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.rstrip("\n")
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                bad_rows.append((line_no, str(exc)))
                continue
            n_rows += 1
            for field in NORMALIZED_FIELDS:
                if field not in row or row[field] in (None, "", [], {}):
                    field_missing[field] += 1
            problems = validate_row(row)
            if problems:
                bad_rows.append((line_no, "; ".join(problems)))

    if bad_rows:
        print(f"\n❌  {len(bad_rows)} problem line(s):")
        for lineno, msg in bad_rows[:10]:
            print(f"   line {lineno}: {msg}")
        if len(bad_rows) > 10:
            print(f"   … and {len(bad_rows) - 10} more")
        sys.exit(1)

    print(f"✅  {n_rows} rows — all lines parse and pass schema checks\n")
    col_w = max(len(f) for f in NORMALIZED_FIELDS)
    print(f"  {'Field':<{col_w}}   Missing / Total")
    print(f"  {'-' * col_w}   ---------------")
    for field in NORMALIZED_FIELDS:
        missing = field_missing[field]
        flag = "  ⚠️" if missing > 0 else ""
        print(f"  {field:<{col_w}}   {missing:>7} / {n_rows}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate HotpotQA.")
    parser.add_argument("--out", type=Path, default=Path("data_hotpotqa.jsonl"))
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset id")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="HF config (distractor/fullwiki)")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.validate_only:
        download(args.out, dataset=args.dataset, config=args.config, split=args.split, limit=args.limit)
    validate(args.out)


if __name__ == "__main__":
    main()
