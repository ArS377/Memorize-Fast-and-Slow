#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.official_scorer_runner import run_official_scorer, write_scorer_summary


def main() -> int:
    scorer_script = Path("fake_scorer.py")
    pred_path = Path("fake_preds.jsonl")
    out_json = Path("fake_result.json")
    scorer_script.write_text("print('noop')\n", encoding="utf-8")
    pred_path.write_text('{"pred":"x","answers":["x"],"all_classes":[]}\n', encoding="utf-8")

    fake_proc = mock.MagicMock()
    fake_proc.returncode = 0
    fake_proc.stdout = '{"score": 0.5}\n'
    fake_proc.stderr = ""

    with mock.patch("eval.official_scorer_runner.subprocess.run", return_value=fake_proc) as run_mock:
        payload = run_official_scorer(scorer_script, pred_path, output_json_path=out_json)

    assert payload["returncode"] == 0, "runner return code mismatch"
    assert payload["result_json_path"] == str(out_json), "result path mismatch"
    run_mock.assert_called_once()
    called_command = run_mock.call_args[0][0]
    assert "--pred" in called_command, "missing --pred flag"
    assert "--out" in called_command, "missing --out flag"

    summary_path = Path("official_runner_summary.json")
    write_scorer_summary(summary_path, payload)
    parsed = json.loads(summary_path.read_text(encoding="utf-8"))
    assert parsed["returncode"] == 0, "summary write failed"
    print("Official scorer runner smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

