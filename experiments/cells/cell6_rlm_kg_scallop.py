"""Cell 6: RLM over Scallop-validated KG context."""

from __future__ import annotations

from experiments._cli import run_cell

CELL_ID = 6
LABEL = "rlm_kg_scallop"
SESSION = "pilot_scallop"


def main(argv=None):
    return run_cell(
        cell_id=CELL_ID, label=LABEL, kind="rlm", retrieval="kg",
        session_id=SESSION, argv=argv,
    )


if __name__ == "__main__":
    main()
