#!/usr/bin/env bash
# Re-enter the benchmark until it reports completion.
#
# Something on this host SIGTERMs the run roughly an hour in; the sender has
# not been identified (earlyoom is active but its dual threshold is not met,
# there is no kernel OOM record, and the sibling tmux sessions survive). With
# both retrieval caches populated a re-entry replays the scored memory and
# appends to generations.jsonl, so each attempt strictly advances. Supervising
# the loop turns an unexplained external kill into a slowdown.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-graphiti/bin/python}"
CONFIG="${CONFIG:-$ROOT/configs/persona_end_to_end_joint_surface_a.json}"
DATASET_ENV="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["dataset_dir_env"])' "$CONFIG")" || exit 1
CORPUS="${CORPUS:-${!DATASET_ENV:-$ROOT/results/persona_conflict_conversations_surface_a_graphiti_build2}}"
OUTPUT_ENV="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["output_dir_env"])' "$CONFIG")" || exit 1
OUT="${OUT:-${!OUTPUT_ENV:-}}"
: "${OUT:?set OUT or the configured benchmark output environment variable}"
: "${PERSONA_GRAPHITI_BUILD_ID:?export the build ID for this run}"
: "${PERSONA_GRAPHITI_RETRIEVAL_CACHE:?export a run-specific Graphiti retrieval cache path}"
: "${PERSONA_HYBRID_RETRIEVAL_CACHE:?export a run-specific hybrid retrieval cache path}"
: "${PERSONA_QWEN_DEVICE:?export the model device for this host}"
export "$DATASET_ENV=$CORPUS" "$OUTPUT_ENV=$OUT"
export PERSONA_GRAPHITI_BUILD_ID PERSONA_GRAPHITI_RETRIEVAL_CACHE PERSONA_HYBRID_RETRIEVAL_CACHE PERSONA_QWEN_DEVICE
LOG_ROOT="${LOG_ROOT:-$ROOT/results/joint_supervisor_logs}"
"$PYTHON_BIN" - "$OUT" "$CORPUS" "$LOG_ROOT" <<'PY' || exit 1
import os, sys
from pathlib import Path
from neurosym.application.source_provenance import ensure_output_directory, paper_evidence_roots
from scripts.paper_artifacts import safe_path
output, corpus, logs = map(Path, sys.argv[1:])
protected = paper_evidence_roots(Path.cwd())
ensure_output_directory(output, input_dirs=(corpus,), frozen_roots=protected, allow_resume=True)
ensure_output_directory(logs, input_dirs=(corpus, output), frozen_roots=protected)
for key in ("PERSONA_GRAPHITI_RETRIEVAL_CACHE", "PERSONA_HYBRID_RETRIEVAL_CACHE"):
    cache = Path(os.environ[key])
    safe_path(cache.parent, cache.name)
    ensure_output_directory(cache.parent, input_dirs=(corpus,), frozen_roots=protected, allow_resume=True)
    if cache.exists() and cache.stat().st_nlink > 1:
        raise ValueError(f"refusing hard-linked retrieval cache: {cache}")
PY
SESSION_LOG_DIR=""

for attempt in $(seq 1 12); do
  status="$("$PYTHON_BIN" - "$OUT" <<'PY'
import sys
from pathlib import Path
from neurosym.application.source_provenance import ensure_output_directory, paper_evidence_roots
from scripts.paper_artifacts import json_load
output = ensure_output_directory(sys.argv[1], frozen_roots=paper_evidence_roots(Path.cwd()), allow_resume=True)
manifest = output / "manifest.json"
print(json_load(manifest).get("status", "") if manifest.is_file() else "")
PY
)" || exit 1
  if [ "$status" = "completed" ]; then
    echo "[supervisor] run already complete"; break
  fi
  if [ -z "$SESSION_LOG_DIR" ]; then
    mkdir -p "$LOG_ROOT" || exit 1
    SESSION_LOG_DIR="$(mktemp -d "$LOG_ROOT/joint.XXXXXXXX")" || exit 1
  fi
  done_rows="$(wc -l < "$OUT/generations.jsonl" 2>/dev/null || echo 0)"
  echo "[supervisor $(date -u +%H:%M:%S)] attempt $attempt, $done_rows/960 generations done"
  # Per-attempt log: scanning an accumulated log would match a refusal from
  # an earlier attempt and abort a run that is actually progressing.
  attempt_log="$SESSION_LOG_DIR/attempt_${attempt}.log"
  stdbuf -oL -eL "$PYTHON_BIN" \
      -m experiments.persona_end_to_end_benchmark --config "$CONFIG" \
      > "$attempt_log" 2>&1
  code=$?
  cat "$attempt_log" >> "$SESSION_LOG_DIR/supervised.log"
  echo "[supervisor $(date -u +%H:%M:%S)] attempt $attempt exited $code" | tee -a "$SESSION_LOG_DIR/supervised.log"
  if [ "$code" = "0" ]; then
    echo "[supervisor] completed"; break
  fi
  # A refused resume means the replayed memory no longer matches; re-entering
  # would loop forever, so stop and leave it for inspection.
  if grep -q "cannot resume" "$attempt_log"; then
    echo "[supervisor] resume refused - stopping for inspection"; break
  fi
  sleep 20
done
echo "[supervisor] final: $(wc -l < "$OUT/generations.jsonl" 2>/dev/null || echo 0)/960"
