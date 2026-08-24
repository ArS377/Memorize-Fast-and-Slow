#!/usr/bin/env bash
# Re-enter the benchmark until it reports completion.
#
# Something on this host SIGTERMs the run roughly an hour in; the sender has
# not been identified (earlyoom is active but its dual threshold is not met,
# there is no kernel OOM record, and the sibling tmux sessions survive). With
# both retrieval caches populated a re-entry replays the scored memory and
# appends to generations.jsonl, so each attempt strictly advances. Supervising
# the loop turns an unexplained external kill into a slowdown.
set -u
cd "$HOME/neurosym-graphiti"
source graphiti_env.sh
export PERSONA_GRAPHITI_BUILD_ID=sabuild1
export PERSONA_GRAPHITI_RETRIEVAL_CACHE=/tmp/joint_cache.json
export PERSONA_HYBRID_RETRIEVAL_CACHE=/tmp/joint_hybrid_cache.json
export PERSONA_QWEN_DEVICE=cuda:2
CONFIG=configs/persona_end_to_end_joint_surface_a.json
OUT="$PERSONA_SURFACE_A_BENCHMARK_OUTPUT_DIR"

for attempt in $(seq 1 12); do
  if [ -f "$OUT/manifest.json" ] && \
     grep -q '"status": "completed"' "$OUT/manifest.json" 2>/dev/null; then
    echo "[supervisor] run already complete"; break
  fi
  done_rows="$(wc -l < "$OUT/generations.jsonl" 2>/dev/null || echo 0)"
  echo "[supervisor $(date -u +%H:%M:%S)] attempt $attempt, $done_rows/960 generations done"
  # Per-attempt log: scanning an accumulated log would match a refusal from
  # an earlier attempt and abort a run that is actually progressing.
  attempt_log="/tmp/joint_attempt_${attempt}.log"
  stdbuf -oL -eL .venv-graphiti/bin/python \
      -m experiments.persona_end_to_end_benchmark --config "$CONFIG" \
      > "$attempt_log" 2>&1
  code=$?
  cat "$attempt_log" >> /tmp/joint_supervised.log
  echo "[supervisor $(date -u +%H:%M:%S)] attempt $attempt exited $code" | tee -a /tmp/joint_supervised.log
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
