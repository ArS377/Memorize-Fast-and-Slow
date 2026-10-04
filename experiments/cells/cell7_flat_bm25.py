"""Cell 7: Flat LLM over BM25-only chunk retrieval (no KG, no graph)."""

from __future__ import annotations

import sys

from experiments._cli import run_cell

CELL_ID = 7
LABEL = "flat_bm25"


def main(argv=None):
    argv = list(argv) if argv is not None else sys.argv[1:]
    if "--chunk-retrieval-mode" not in argv:
        argv = ["--chunk-retrieval-mode", "bm25", *argv]
    return run_cell(
        cell_id=CELL_ID, label=LABEL, kind="flat", retrieval="chunk",
        session_id=None, argv=argv,
    )


if __name__ == "__main__":
    main()
