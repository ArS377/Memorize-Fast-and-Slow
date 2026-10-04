#!/usr/bin/env python3
"""Normalize 2WikiMultihopQA into the shared multi-hop schema.

2WikiMultihopQA is the primary bed for verifying dense-seeded PPR: it is
multi-document, and every example ships gold ``supporting_facts`` (sentence
level) AND ``evidences`` as ``(subject, relation, object)`` triples that line up
with the KG fact schema. That gives independent Recall@k and multi-hop-coverage
labels that LongBench-v2 could not (see the dense_ppr sweep ANALYSIS).

The canonical release is distributed as JSON-array files (``train.json`` /
``dev.json`` / ``test.json``). This script reads one of those directly with
``--json``, or an equivalent HuggingFace mirror with ``--dataset``. Output
mirrors ``download_longbench.py``: atomic write, ASCII-safe, round-trip +
schema validation, per-field coverage.

Usage
-----
    # From the official release JSON array:
    python download_2wiki.py --json /path/to/2wiki/dev.json         # -> data_2wiki.jsonl
    python download_2wiki.py --json dev.json --limit 200

    # From a HuggingFace mirror that exposes the same columns:
    python download_2wiki.py --dataset <hf_id> --split validation

    python download_2wiki.py --validate-only --out data_2wiki.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, Dict, Any

from experiments.multihop_datasets import (
    NORMALIZED_FIELDS,
    normalize_2wiki_row,
    validate_row,
)


def _iter_json_array(path: Path) -> Iterator[Dict[str, Any]]:
    """The official 2Wiki files are a single JSON array; a JSONL fallback is
    also accepted."""
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        for row in json.loads(text):
            yield row
    else:  # tolerate JSONL
        for line in text.splitlines():
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_hf(dataset: str, split: str) -> Iterator[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets package not found. Run: pip install datasets")
    for row in load_dataset(dataset, split=split):
        yield dict(row)


def download(out_path: Path, rows: Iterable[Dict[str, Any]], limit: int | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=out_path.parent, prefix=".tmp_2wiki_", suffix=".jsonl")
    n = 0
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                if limit is not None and n >= limit:
                    break
                normalized = normalize_2wiki_row(dict(row))
                f.write(json.dumps(normalized, ensure_ascii=True) + "\n")
                n += 1
                if n % 500 == 0:
                    print(f"  wrote {n} rows …", flush=True)
        Path(tmp_name).replace(out_path)
        print(f"\nWrote {n} rows → {out_path}")
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
    n_with_triples = 0
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
            if row.get("gold_evidence_triples"):
                n_with_triples += 1
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

    print(f"✅  {n_rows} rows — all lines parse and pass schema checks")
    print(f"    {n_with_triples}/{n_rows} rows carry gold_evidence_triples\n")
    col_w = max(len(f) for f in NORMALIZED_FIELDS)
    print(f"  {'Field':<{col_w}}   Missing / Total")
    print(f"  {'-' * col_w}   ---------------")
    for field in NORMALIZED_FIELDS:
        missing = field_missing[field]
        flag = "  ⚠️" if missing > 0 else ""
        print(f"  {field:<{col_w}}   {missing:>7} / {n_rows}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and validate 2WikiMultihopQA.")
    parser.add_argument("--out", type=Path, default=Path("data_2wiki.jsonl"))
    parser.add_argument("--json", type=Path, default=None, help="Official release JSON array (e.g. dev.json)")
    parser.add_argument("--dataset", default=None, help="HuggingFace mirror id (alternative to --json)")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if not args.validate_only:
        if args.json is not None:
            rows = _iter_json_array(args.json)
        elif args.dataset is not None:
            rows = _iter_hf(args.dataset, args.split)
        else:
            sys.exit("provide --json <release file> or --dataset <hf id> (or --validate-only)")
        download(args.out, rows, limit=args.limit)
    validate(args.out)


if __name__ == "__main__":
    main()
