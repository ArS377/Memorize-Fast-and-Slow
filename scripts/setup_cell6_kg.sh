#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
LIMIT="${LIMIT:-50}"
KG_ENV="${KG_ENV:-kg-env}"
KG_ENV_DIR="${KG_ENV_DIR:-$HOME/.conda/envs/$KG_ENV}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-$HOME/.conda/envs/vllm-env}"
KG_PY="$KG_ENV_DIR/bin/python3"
VLLM_PY="$VLLM_ENV_DIR/bin/python3"
FACTS_FILE="results/kg_builds/pilot_scallop_facts.jsonl"
SCALLOP_VALIDATOR_URL="${SCALLOP_VALIDATOR_URL:-http://127.0.0.1:8765}"
SCALLOP_VALIDATOR_PORT="${SCALLOP_VALIDATOR_PORT:-8765}"
NEO4J_URI="${NEO4J_URI:-bolt://127.0.0.1:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"

if [[ -z "${NEO4J_PASSWORD:-}" ]]; then
  echo "[setup-cell6] ERROR: NEO4J_PASSWORD is required for persistent Cell 6 updates." >&2
  exit 2
fi

echo "[setup-cell6] repo: $ROOT"
echo "[setup-cell6] model: $MODEL"
echo "[setup-cell6] vLLM:  $VLLM_BASE_URL"
echo "[setup-cell6] KG extraction: exhaustive (all source chunks)"

if ! curl -fsS "$VLLM_BASE_URL/models" >/dev/null; then
  echo "[setup-cell6] ERROR: vLLM is not reachable at $VLLM_BASE_URL" >&2
  echo "Start vLLM first, then rerun this script." >&2
  exit 2
fi

echo "[setup-cell6] installing RLM runtime in the active env..."
if [[ -x "$VLLM_PY" ]]; then
  "$VLLM_PY" -m pip install -q "rlms>=0.1.1" openai neo4j
else
  python3 -m pip install -q "rlms>=0.1.1" openai neo4j
fi

if [[ ! -x "$KG_PY" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "[setup-cell6] ERROR: $KG_ENV_DIR does not exist and conda is not on PATH." >&2
    exit 2
  fi
  echo "[setup-cell6] creating $KG_ENV for Scallop KG build..."
  conda create -n "$KG_ENV" python=3.10 -y
fi

echo "[setup-cell6] installing KG build dependencies in $KG_ENV..."
"$KG_PY" -m pip install -q --upgrade pip setuptools wheel
"$KG_PY" -m pip install -q \
  openai neo4j pydantic tqdm httpx numpy "sentence-transformers==3.4.1" \
  "scallopy @ https://github.com/scallop-lang/scallop/releases/download/0.2.4/scallopy-0.2.4-cp310-cp310-manylinux_2_27_x86_64.whl"

if ! curl -fsS "$SCALLOP_VALIDATOR_URL/health" >/dev/null; then
  echo "[setup-cell6] starting actual-scallopy validator service..."
  mkdir -p results/services
  nohup env PYTHONPATH="$ROOT" "$KG_PY" -m services.scallop_validator_service \
    --host 127.0.0.1 --port "$SCALLOP_VALIDATOR_PORT" \
    >results/services/scallop_validator.log 2>&1 &
  echo $! > results/services/scallop_validator.pid
  for _ in 1 2 3 4 5; do
    curl -fsS "$SCALLOP_VALIDATOR_URL/health" >/dev/null && break
    sleep 1
  done
fi
curl -fsS "$SCALLOP_VALIDATOR_URL/health" >/dev/null || {
  echo "[setup-cell6] ERROR: actual Scallop validator did not become healthy." >&2
  exit 2
}

echo "[setup-cell6] building Scallop-validated persistent KG..."
PYTHONPATH="$ROOT" "$KG_PY" -m experiments.build_kg \
  --input data.jsonl \
  --limit "$LIMIT" \
  --session pilot_scallop \
  --validate \
  --model "$MODEL" \
  --vllm-base-url "$VLLM_BASE_URL" \
  --api-key EMPTY \
  --neo4j-uri "$NEO4J_URI" \
  --neo4j-user "$NEO4J_USER" \
  --neo4j-password "$NEO4J_PASSWORD" \
  --scallop-validator-url "$SCALLOP_VALIDATOR_URL" \
  --max-tokens 2048

if [[ ! -s "$FACTS_FILE" ]]; then
  echo "[setup-cell6] ERROR: expected facts file was not created: $FACTS_FILE" >&2
  exit 2
fi

echo "[setup-cell6] ready: persistent session pilot_scallop; mirror: $FACTS_FILE"
echo
echo "Run Cell 6 with:"
cat <<EOF
python3 -m experiments.cells.cell6_rlm_kg_scallop \\
  --input data.jsonl \\
  --limit 50 \\
  --model $MODEL \\
  --vllm-base-url $VLLM_BASE_URL \\
  --api-key EMPTY \\
  --neo4j-uri $NEO4J_URI \\
  --neo4j-user $NEO4J_USER \\
  --neo4j-password "\$NEO4J_PASSWORD" \\
  --scallop-validator-url $SCALLOP_VALIDATOR_URL \\
  --hops 2 \\
  --limit-triples 50 \\
  --context-max-chars 4000 \\
  --max-depth 2 \\
  --max-iterations 10 \\
  --max-tokens 64000 \\
  --verbose
EOF
