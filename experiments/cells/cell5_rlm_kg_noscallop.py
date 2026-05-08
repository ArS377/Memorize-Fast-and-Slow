"""Cell 5: RLM over KG context (Scallop validator OFF)."""

from __future__ import annotations

from experiments._cli import run_cell

CELL_ID = 5
LABEL = "rlm_kg_noscallop"
SESSION = "pilot_noscallop"


def main(argv=None):
    return run_cell(
        cell_id=CELL_ID, label=LABEL, kind="rlm", retrieval="kg",
        session_id=SESSION, argv=argv,
    )


if __name__ == "__main__":
    main()
