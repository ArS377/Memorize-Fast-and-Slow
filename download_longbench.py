#!/usr/bin/env python3
"""
Download LongBench-v2 from HuggingFace and write a validated data.jsonl.

Fixes vs the original script
------------------------------
1. Writes to a temp file first, then atomically renames — a crash mid-write
   will never leave a half-written data.jsonl behind.
2. Uses ensure_ascii=True so every line is pure ASCII; eliminates the
   "Unterminated string" JSONDecodeError caused by multi-byte UTF-8
   sequences getting split across a write buffer flush.
3. Runs a full round-trip validation pass after writing: reads every line
   back and calls json.loads() so bad rows are caught before you try to
   run the pipeline.
4. Reports per-field coverage so you can spot missing columns early.

Usage
-----
    pip install datasets
    python download_longbench.py                   # writes data.jsonl
    python download_longbench.py --out my_data.jsonl --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


EXPECTED_FIELDS = [
    "_id", "domain", "sub_domain", "difficulty", "length",
    "question", "choice_A", "choice_B", "choice_C", "choice_D",
    "answer", "context",
]


def download(out_path: Path, limit: int | None = None) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets package not found. Run: pip install datasets")

    print("Fetching zai-org/LongBench-v2 from HuggingFace …", flush=True)
    ds = load_dataset("zai-org/LongBench-v2", split="train")

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"  (capped to {limit} rows via --limit)")

    # Write to a temp file in the same directory so the final rename is
    # atomic on the same filesystem.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=".tmp_longbench_", suffix=".jsonl"
    )

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as f:
            for i, row in enumerate(ds):
                # ensure_ascii=True: every character is encoded as \uXXXX —
                # guarantees the line is pure ASCII and can never be split
                # across a buffer boundary.
                line = json.dumps(dict(row), ensure_ascii=True)
                f.write(line + "\n")

                if (i + 1) % 100 == 0:
                    print(f"  wrote {i + 1} / {len(ds)} rows …", flush=True)

        # Atomic replace
        Path(tmp_name).replace(out_path)
        print(f"\nWrote {len(ds)} rows → {out_path}")

    except Exception:
        # Clean up temp file on error so it doesn't clutter the directory.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def validate(path: Path) -> None:
    print(f"\nValidating {path} …", flush=True)

    field_missing: dict[str, int] = {f: 0 for f in EXPECTED_FIELDS}
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
            for field in EXPECTED_FIELDS:
                if field not in row or row[field] is None or row[field] == "":
                    field_missing[field] += 1

    if bad_rows:
        print(f"\n❌  {len(bad_rows)} INVALID line(s):")
        for lineno, msg in bad_rows[:10]:
            print(f"   line {lineno}: {msg}")
        if len(bad_rows) > 10:
            print(f"   … and {len(bad_rows) - 10} more")
        sys.exit(1)

    print(f"✅  {n_rows} rows — all lines parse as valid JSON\n")

    # Field coverage report
    col_w = max(len(f) for f in EXPECTED_FIELDS)
    print(f"  {'Field':<{col_w}}   Missing / Total")
    print(f"  {'-' * col_w}   ---------------")
    for field in EXPECTED_FIELDS:
        missing = field_missing[field]
        flag = "  ⚠️" if missing > 0 else ""
        print(f"  {field:<{col_w}}   {missing:>7} / {n_rows}{flag}")

    any_missing = any(v > 0 for v in field_missing.values())
    if any_missing:
        print("\n⚠️  Some rows have empty/missing fields (see above).")
    else:
        print("\n✅  All expected fields present in every row.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and validate LongBench-v2.")
    parser.add_argument("--out", type=Path, default=Path("data.jsonl"),
                        help="Output path (default: data.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only download the first N rows (for quick tests).")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip download; just validate an existing file.")
    args = parser.parse_args()

    if not args.validate_only:
        download(args.out, limit=args.limit)

    validate(args.out)


if __name__ == "__main__":
    main()
