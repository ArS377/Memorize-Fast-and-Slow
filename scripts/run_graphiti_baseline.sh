#!/usr/bin/env bash
# Drive the Graphiti external baseline end to end with per-stage gates.
#
# Each stage refuses to start until the previous one's gate passes, so a
# misconfigured extractor or a missing service fails in minutes rather than
# after hours of ingestion. Intended to run detached under tmux:
#
#   tmux new-session -d -s graphiti "scripts/run_graphiti_baseline.sh 2>&1 | tee /tmp/graphiti_run.log"
#
# Stages: preflight smoke (1 account) -> full preflight (12) -> builds -> analysis.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PYTHON_BIN:?set PYTHON_BIN to the absolute path of the prepared Graphiti interpreter}"
CONFIG="${CONFIG:-$ROOT/configs/persona_end_to_end_joint_surface_a.json}"
# v1 is shortcut-solvable (a per-family constant rule scores ~100% on the
# delayed probes), so the account-unique surface corpus is the default.
CORPUS="${CORPUS:-$ROOT/results/persona_conflict_conversations_surface_a_graphiti_build2}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/outputs}"
BUILDS="${BUILDS:-build1 build2 build3}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
RESUME="${RESUME:-0}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$RESULTS_ROOT/graphiti_analysis}"
cd "$ROOT"

# The config declares which environment variables carry the dataset and output
# paths, so read them from it rather than assuming the v1 names.
DATASET_ENV="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["dataset_dir_env"])' "$CONFIG")"
OUTPUT_ENV="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"]["output_dir_env"])' "$CONFIG")"
export "$DATASET_ENV"="$CORPUS"

say() { printf '\n=== [%s] %s ===\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }

validate_output() {
  "$PYTHON_BIN" - "$1" "$CORPUS" "${2:-0}" <<'PY'
import sys
from pathlib import Path
from neurosym.application.source_provenance import ensure_output_directory, paper_evidence_roots
output = ensure_output_directory(sys.argv[1], input_dirs=(sys.argv[2],), frozen_roots=paper_evidence_roots(Path.cwd()), allow_resume=sys.argv[3] == "1")
if output.exists() and sys.argv[3] != "1":
    raise SystemExit(f"refusing existing output {output}; use a fresh path or RESUME=1 for scored builds")
PY
}

is_completed() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]) / "manifest.json"
raise SystemExit(0 if path.is_file() and json.loads(path.read_text())["status"] == "completed" else 1)
PY
}

# --- Stage 0: services -------------------------------------------------------
say "stage 0: service preflight"
[ -x "$PYTHON_BIN" ] || die "python not found at $PYTHON_BIN"
[ -f "$CONFIG" ] || die "config not found at $CONFIG"
[ -d "$CORPUS" ] || die "corpus not found at $CORPUS"

: "${PERSONA_GRAPHITI_LLM_BASE_URL:?export the Graphiti service environment before running}"
curl -sf -m 10 "${PERSONA_GRAPHITI_LLM_BASE_URL%/}/models" >/dev/null \
  || die "extraction endpoint not reachable at $PERSONA_GRAPHITI_LLM_BASE_URL"
echo "extraction endpoint: OK"

"$PYTHON_BIN" - <<'PY' || die "graphiti Neo4j not reachable"
import os
from neo4j import GraphDatabase
uri = os.environ["PERSONA_GRAPHITI_NEO4J_URI"]
auth = (os.environ["PERSONA_GRAPHITI_NEO4J_USER"], os.environ["PERSONA_GRAPHITI_NEO4J_PASSWORD"])
driver = GraphDatabase.driver(uri, auth=auth)
with driver.session(database=os.environ["PERSONA_GRAPHITI_NEO4J_DATABASE"]) as session:
    session.run("RETURN 1").consume()
driver.close()
print("graphiti neo4j: OK")
PY

# Scallop is only needed when an arm of kind structured_memory is configured.
# Check it now rather than an hour into ingestion, but only when it matters.
NEEDS_SCALLOP="$("$PYTHON_BIN" -c 'import json,sys; arms=json.load(open(sys.argv[1]))["arms"]; print("1" if any(a.get("kind")=="structured_memory" for a in arms) else "0")' "$CONFIG")"
if [ "$NEEDS_SCALLOP" = "1" ]; then
  if curl -sf -m 10 "${PERSONA_SCALLOP_ENDPOINT%/}/health" >/dev/null 2>&1; then
    echo "scallop validator: OK (required by structured_memory arms)"
    SCALLOP_UP=1
  else
    echo "scallop validator: NOT REACHABLE at ${PERSONA_SCALLOP_ENDPOINT:-unset}"
    echo "  this config declares structured_memory arms, so builds will be skipped."
    SCALLOP_UP=0
  fi
else
  echo "scallop validator: not required by this config"
  SCALLOP_UP=1
fi

gate_preflight() {
  # Read the manifest rather than trusting the exit code: a well-formed but
  # empty extraction raises nothing and would otherwise look like a result.
  local manifest="$1" min_facts="$2"
  "$PYTHON_BIN" - "$manifest" "$min_facts" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
minimum = int(sys.argv[2])
health = manifest["extraction_health"]
summary = manifest["extraction_summary"]
print(json.dumps({
    "status": health["status"],
    "fatal": health["fatal"],
    "warnings": health["warnings"],
    "valid_fact_rows_total": summary["valid_fact_rows_total"],
    "empty_retrievals": f"{summary['empty_retrieval_count']}/{summary['condition_count']}",
    "valid_at_resolved": summary["valid_at_resolved_fraction"],
}, indent=2))
if health["status"] != "passed":
    raise SystemExit("extraction health failed")
if summary["valid_fact_rows_total"] < minimum:
    raise SystemExit(f"only {summary['valid_fact_rows_total']} facts, expected >= {minimum}")
empty = summary["empty_retrieval_count"]
if summary["condition_count"] and empty > summary["condition_count"] // 2:
    raise SystemExit(f"{empty} of {summary['condition_count']} conditions retrieved nothing")
PY
}

# --- Stage 1: one-account smoke ---------------------------------------------
if [ "$SKIP_SMOKE" != "1" ]; then
  say "stage 1: one-account extraction smoke test"
  SMOKE_DIR="${SMOKE_DIR:-$RESULTS_ROOT/persona_graphiti_preflight_smoke}"
  validate_output "$SMOKE_DIR"
  env PERSONA_GRAPHITI_BUILD_ID=smoke "$OUTPUT_ENV=$SMOKE_DIR" \
    "$PYTHON_BIN" -m experiments.persona_graphiti_preflight \
      --config "$CONFIG" --histories 1 >/dev/null \
    || die "smoke preflight raised; inspect $SMOKE_DIR and the log"
  gate_preflight "$SMOKE_DIR/preflight_manifest.json" 20 \
    || die "smoke gate failed: extraction is not usable, do not proceed"
  echo "smoke gate: PASSED"
fi

# --- Stage 2: full ingestion preflight ---------------------------------------
PRE_DIR="${PRE_DIR:-$RESULTS_ROOT/persona_graphiti_preflight_v1}"
if [ "${SKIP_PREFLIGHT:-0}" = "1" ] && [ -f "$PRE_DIR/preflight_manifest.json" ]; then
  say "stage 2: reusing the completed preflight at $PRE_DIR"
else
  say "stage 2: full 12-account ingestion preflight"
  validate_output "$PRE_DIR"
  env PERSONA_GRAPHITI_BUILD_ID=preflight "$OUTPUT_ENV=$PRE_DIR" \
    "$PYTHON_BIN" -m experiments.persona_graphiti_preflight --config "$CONFIG" >/dev/null \
    || die "full preflight raised; inspect $PRE_DIR"
fi
gate_preflight "$PRE_DIR/preflight_manifest.json" 200 \
  || die "full preflight gate failed"
"$PYTHON_BIN" - "$PRE_DIR/preflight_manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["condition_count"] == 120, m["condition_count"]
assert m["graphiti_prompt_count"] == 240, m["graphiti_prompt_count"]
episodes = m["graphiti_memory"]["episode_count"]
assert episodes == 312, f"expected 312 incremental episodes, got {episodes}"
print("coverage gate: 120 conditions, 240 prompts, 312 episodes")
PY
echo "full preflight gate: PASSED"

# --- Stage 3: scored builds ---------------------------------------------------
if [ "$SCALLOP_UP" != "1" ]; then
  say "stopping before builds: the Scallop validator is required and is down"
  exit 0
fi

for build in $BUILDS; do
  say "stage 3: benchmark $build"
  OUT="$RESULTS_ROOT/persona_graphiti_e2e_$build"
  validate_output "$OUT" "$RESUME"
  if [ "$RESUME" = "1" ] && is_completed "$OUT"; then
    say "reusing completed build at $OUT without writing it"
  else
    env PERSONA_GRAPHITI_BUILD_ID="$build" "$OUTPUT_ENV=$OUT" \
      "$PYTHON_BIN" -m experiments.persona_end_to_end_benchmark --config "$CONFIG" >/dev/null \
      || die "benchmark $build failed; resume with RESUME=1 SKIP_SMOKE=1 SKIP_PREFLIGHT=1"
  fi
  # The scored build ingests its own store, independently and stochastically,
  # so a healthy preflight says nothing about it. Re-run the same extraction
  # health checks against the store this build actually queried.
  "$PYTHON_BIN" - "$OUT/manifest.json" "$OUT/graphiti_store_states.json" "$OUT/predictions.jsonl" <<'PY'
import json, sys
sys.path.insert(0, ".")
from experiments.persona_graphiti_preflight import (
    assess_extraction_health,
    summarize_extraction,
)

manifest = json.load(open(sys.argv[1]))
assert manifest["status"] == "completed", manifest["status"]
failures = manifest["graphiti_memory"]["episode_failure_count"]
assert failures == 0, f"{failures} episode failures"

states = json.load(open(sys.argv[2]))
# Read the retrievals the model actually saw. Deriving them from store fact
# counts would certify a build whose stores are populated but whose
# query-time retrievals are empty, which is the failure that matters.
predictions = [json.loads(line) for line in open(sys.argv[3]) if line.strip()]
retrievals = {
    str(row["evaluation_input_id"]): {"rows": [None] * int(row.get("retrieved_fact_count", 0))}
    for row in predictions
    if str(row.get("arm_kind")) == "graphiti_memory"
}
if not retrievals:
    raise SystemExit("no graphiti rows in predictions; cannot gate the scored build")
summary = summarize_extraction(states, retrievals)
health = assess_extraction_health(summary)
print(json.dumps({
    "scored_store_valid_facts": summary["valid_fact_rows_total"],
    "checkpoints_without_facts": summary["checkpoints_with_zero_valid_facts"],
    "health": health["status"],
    "fatal": health["fatal"],
}, indent=2))
if health["status"] != "passed":
    raise SystemExit(f"scored build extraction health failed: {health['fatal']}")
print("build gate: completed, zero episode failures, scored store healthy")
PY

  say "stage 4: analysis for $build"
  ANALYSIS_DIR="$ANALYSIS_ROOT/persona_graphiti_e2e_$build"
  validate_output "$ANALYSIS_DIR"
  "$PYTHON_BIN" -m experiments.persona_graphiti_analysis \
    --run-dir "$OUT" --corpus-dir "$CORPUS" \
    --output-dir "$ANALYSIS_DIR" --verify-inputs >/dev/null
  echo "wrote $ANALYSIS_DIR/README.md, analysis.md, error_taxonomy.json"
done

say "all stages complete"
for build in $BUILDS; do
  echo "--- $build headline ---"
  sed -n '/^| Arm /,/^$/p' "$ANALYSIS_ROOT/persona_graphiti_e2e_$build/README.md" 2>/dev/null || true
done
