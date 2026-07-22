#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/vllm-env/bin/python}"
KG_PYTHON_BIN="${KG_PYTHON_BIN:-$HOME/.conda/envs/kg-env/bin/python}"
DATA_PATH="${DATA_PATH:-$ROOT/data.jsonl}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
SCALLOP_VALIDATOR_URL="${SCALLOP_VALIDATOR_URL:-http://127.0.0.1:8765}"
NEO4J_URI="${NEO4J_URI:-bolt://127.0.0.1:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
SESSION_ID="${SESSION_ID:-pilot_scallop}"
LIMIT="${LIMIT:-50}"
MODEL_TIMEOUT="${MODEL_TIMEOUT:-90}"
MODEL_MAX_RETRIES="${MODEL_MAX_RETRIES:-0}"
NATIVE_TOOL_PREFLIGHT_TIMEOUT="${NATIVE_TOOL_PREFLIGHT_TIMEOUT:-20}"
RUN_ID="${RUN_ID:-cell6_hybrid_$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/$RUN_ID}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Cell 6 Python environment is unavailable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: dataset is unavailable: $DATA_PATH" >&2
  exit 2
fi
if ! curl -fsS --max-time 5 "$VLLM_BASE_URL/models" >/dev/null; then
  echo "ERROR: Qwen/vLLM is unavailable at $VLLM_BASE_URL" >&2
  exit 2
fi
if ! PYTHONPATH="$ROOT" "$PYTHON_BIN" - "$VLLM_BASE_URL" "$MODEL" "$NATIVE_TOOL_PREFLIGHT_TIMEOUT" <<'PY'
from __future__ import annotations

import sys

from openai import OpenAI

from experiments.kg_search_tool import SEARCH_KNOWLEDGE_GRAPH_TOOL
from experiments.working_memory_tool import UPDATE_WORKING_MEMORY_TOOL

base_url, model, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout, max_retries=0)
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Briefly acknowledge this request."}],
    tools=[SEARCH_KNOWLEDGE_GRAPH_TOOL, UPDATE_WORKING_MEMORY_TOOL],
    tool_choice="required",
    temperature=0.0,
    max_tokens=32,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
if not response.choices or not response.choices[0].message.tool_calls:
    raise RuntimeError("vLLM did not return a native tool call during preflight")
print("Native tool-call preflight passed.")
PY
then
  echo "ERROR: Qwen/vLLM failed the native tool-call preflight; restart or repair vLLM before Cell 6." >&2
  exit 2
fi

if ! curl -fsS --max-time 5 "$SCALLOP_VALIDATOR_URL/health" >/dev/null; then
  if [[ ! -x "$KG_PYTHON_BIN" ]]; then
    echo "ERROR: Scallop is down and its Python environment is unavailable." >&2
    exit 2
  fi
  mkdir -p "$ROOT/results/services"
  nohup env PYTHONPATH="$ROOT" "$KG_PYTHON_BIN" \
    -m services.scallop_validator_service --host 127.0.0.1 --port 8765 \
    >"$ROOT/results/services/scallop_validator.log" 2>&1 &
  echo $! >"$ROOT/results/services/scallop_validator.pid"
  for _ in {1..10}; do
    curl -fsS --max-time 2 "$SCALLOP_VALIDATOR_URL/health" >/dev/null && break
    sleep 1
  done
fi
curl -fsS --max-time 5 "$SCALLOP_VALIDATOR_URL/health" >/dev/null || {
  echo "ERROR: the actual Scallop validator did not become healthy." >&2
  exit 2
}

if [[ -z "${NEO4J_PASSWORD:-}" ]] && command -v docker >/dev/null 2>&1; then
  neo4j_auth="$({
    docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' cell6-neo4j 2>/dev/null || true
  } | grep '^NEO4J_AUTH=' | cut -d= -f2-)"
  if [[ -n "$neo4j_auth" && "$neo4j_auth" != "none" ]]; then
    NEO4J_USER="${neo4j_auth%%/*}"
    NEO4J_PASSWORD="${neo4j_auth#*/}"
  fi
fi
if [[ -z "${NEO4J_PASSWORD:-}" ]]; then
  echo "ERROR: NEO4J_PASSWORD is unavailable." >&2
  exit 2
fi
export NEO4J_PASSWORD

mkdir -p "$RESULTS_DIR"
echo "Cell 6 hybrid run: $RUN_ID"
echo "Results: $RESULTS_DIR"
echo "Attach/detach safely; the run continues inside tmux."

cd "$ROOT"
set +e
PYTHONPATH="$ROOT" "$PYTHON_BIN" -m experiments.cells.cell6_rlm_kg_scallop \
  --input "$DATA_PATH" \
  --limit "$LIMIT" \
  --run-id "$RUN_ID" \
  --results-dir "$RESULTS_DIR" \
  --model "$MODEL" \
  --vllm-base-url "$VLLM_BASE_URL" \
  --api-key EMPTY \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --session-id "$SESSION_ID" \
  --scallop-validator-url "$SCALLOP_VALIDATOR_URL" \
  --retrieval-mode hybrid \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --embedding-device cpu \
  --dense-failure-policy error \
  --rrf-k 60 \
  --hops 2 \
  --limit-triples 50 \
  --max-depth 2 \
  --max-iterations 10 \
  --max-tokens 64000 \
  --max-tool-calls 3 \
  --termination-mode order_gap \
  --order-gap-epsilon 0.025 \
  --order-gap-window 2 \
  --order-gap-min-iterations 2 \
  --tool-choice required \
  --tool-timeout 30 \
  --tool-max-tokens 2048 \
  --model-timeout "$MODEL_TIMEOUT" \
  --model-max-retries "$MODEL_MAX_RETRIES" \
  --verbose \
  2>&1 | tee "$RESULTS_DIR/run.log"
status=${PIPESTATUS[0]}
set -e

echo "Cell 6 exit status: $status"
exit "$status"
