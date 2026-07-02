#!/usr/bin/env python3
"""Convert RLM trajectory logs (``rlm_logs_ablation/*.jsonl``) into a readable
Markdown report.

Each log file is one example's RLM trajectory:
  * line 0:  ``{"type": "metadata", "root_model", "max_depth", ...}`` run config
  * line k:  ``{"type": "iteration", "iteration", "prompt", "response",
               "code_blocks": [{"code", "result": {"stdout", ...}}],
               "final_answer", "iteration_time"}``

By default this emits *reasoning only* -- each iteration's ``response`` plus the
code actions it executed and their sub-LM output -- and excludes the bulky
repeated context ``prompt`` field.

Log files are matched to ``results/cell4_rlm_raw/results.jsonl`` rows by run
order (both are ordered by example), so each section is annotated with the
example id and correct/predicted/gold outcome when the counts line up.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

FNAME_RE = re.compile(r"rlm_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_[0-9a-f]+\.jsonl$")


def parse_dt(path: Path) -> Optional[datetime]:
    m = FNAME_RE.search(path.name)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y-%m-%d_%H-%M-%S")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_logs(log_dir: Path, since: Optional[datetime], until: Optional[datetime]) -> List[Path]:
    dated = []
    for p in log_dir.glob("rlm_*.jsonl"):
        dt = parse_dt(p)
        if dt is None:
            continue
        if since and dt < since:
            continue
        if until and dt > until:
            continue
        dated.append((dt, p))
    dated.sort(key=lambda t: t[0])
    return [p for _, p in dated]


def indent_block(text: str) -> str:
    """Render as an indented (4-space) Markdown code block -- robust against
    backticks/fences that may appear inside model output."""
    return "\n".join("    " + ln for ln in str(text).splitlines()) or "    "


def render(files: List[Path], results: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    meta: Dict[str, Any] = {}
    if files:
        first_rows = load_jsonl(files[0])
        if first_rows and first_rows[0].get("type") == "metadata":
            meta = first_rows[0]

    out.append("# RLM Reasoning -- cell 4 (rlm_raw)\n")
    if files:
        out.append(f"- **Run window:** {parse_dt(files[0])} to {parse_dt(files[-1])}")
    out.append(
        f"- **Model:** {meta.get('root_model', '?')} | "
        f"**backend:** {meta.get('backend', '?')} | "
        f"**max_depth:** {meta.get('max_depth', '?')} | "
        f"**max_iterations:** {meta.get('max_iterations', '?')}"
    )
    out.append(f"- **Examples:** {len(files)}")
    out.append("")
    out.append("---\n")

    mapped = len(results) == len(files) and len(files) > 0

    for i, fp in enumerate(files):
        rows = load_jsonl(fp)
        iters = [r for r in rows if r.get("type") == "iteration"]
        res = results[i] if mapped else {}
        example_id = res.get("example_id", f"(log {i + 1})")

        out.append(f"## Example {i + 1} -- {example_id}\n")
        if res:
            out.append(
                f"- **Result:** correct={res.get('correct')} | "
                f"predicted=`{res.get('predicted')}` | gold=`{res.get('gold')}` | "
                f"{res.get('elapsed_seconds')}s"
            )
            if res.get("error"):
                out.append(f"- **Error:** {res.get('error')}")
        out.append(f"- **Log:** `{fp.name}` | iterations: {len(iters)}\n")

        for it in iters:
            out.append(f"### Iteration {it.get('iteration')}\n")
            resp = it.get("response")
            if resp:
                out.append(str(resp).strip())
                out.append("")
            for j, cb in enumerate(it.get("code_blocks") or [], start=1):
                if not isinstance(cb, dict):
                    continue
                out.append(f"**Action {j} -- code executed:**\n")
                out.append(indent_block(cb.get("code", "")))
                out.append("")
                result = cb.get("result") or {}
                if isinstance(result, dict):
                    if result.get("stdout"):
                        out.append("**Output (sub-LM):**\n")
                        out.append(indent_block(result["stdout"]))
                        out.append("")
                    if result.get("stderr"):
                        out.append("**stderr:**\n")
                        out.append(indent_block(result["stderr"]))
                        out.append("")
                    if result.get("final_answer") is not None:
                        out.append(f"**Action final_answer:** {result['final_answer']}\n")
            if it.get("final_answer") is not None:
                out.append(f"**Final answer:** {it['final_answer']}\n")
        out.append("---\n")

    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", type=Path, default=Path("rlm_logs_ablation"))
    ap.add_argument("--results", type=Path, default=Path("results/cell4_rlm_raw/results.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/rlm_reasoning.md"))
    ap.add_argument("--since", default="2026-07-01_21-56-00",
                    help="Only include logs at/after this YYYY-MM-DD_HH-MM-SS (empty = all)")
    ap.add_argument("--until", default="",
                    help="Only include logs at/before this YYYY-MM-DD_HH-MM-SS (empty = no upper bound)")
    args = ap.parse_args(argv)

    fmt = "%Y-%m-%d_%H-%M-%S"
    since = datetime.strptime(args.since, fmt) if args.since else None
    until = datetime.strptime(args.until, fmt) if args.until else None

    files = select_logs(args.log_dir, since, until)
    results = load_jsonl(args.results) if args.results.exists() else []
    md = render(files, results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(
        f"Wrote {args.out} ({len(md):,} chars) from {len(files)} log files; "
        f"results_mapped={'yes' if len(results) == len(files) and files else 'no'} "
        f"(logs={len(files)}, results_rows={len(results)})"
    )


if __name__ == "__main__":
    main()
