from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def run_official_scorer(
    scorer_script: Path,
    pred_path: Path,
    *,
    output_json_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    if not scorer_script.exists():
        raise FileNotFoundError(f"Scorer script not found: {scorer_script}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    command = [sys.executable, str(scorer_script), "--pred", str(pred_path)]
    if output_json_path is not None:
        command.extend(["--out", str(output_json_path)])

    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "result_json_path": str(output_json_path) if output_json_path else None,
    }


def write_scorer_summary(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

