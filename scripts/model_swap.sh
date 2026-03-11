#!/usr/bin/env bash
# =============================================================================
# model_swap.sh — Load a model into vLLM, replacing whatever is currently loaded
#
# Usage: bash scripts/model_swap.sh <model-id>
#   model-id must match an entry in models/registry.yaml
#
# Examples:
#   bash scripts/model_swap.sh qwen2.5-coder-7b
#   bash scripts/model_swap.sh qwen2.5-coder-72b
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

MODEL_ID="${1:-}"
[[ -z "$MODEL_ID" ]] && error "Usage: bash scripts/model_swap.sh <model-id>"

# ── Parse registry ────────────────────────────────────────────────────────────
REGISTRY="models/registry.yaml"
[[ -f "$REGISTRY" ]] || error "models/registry.yaml not found — run from repo root"

# Extract model config using Python (avoids yq dependency)
MODEL_CONFIG=$(python3 - <<EOF
import yaml, sys
with open("$REGISTRY") as f:
    reg = yaml.safe_load(f)
models = {m['id']: m for m in reg['models'] if 'hf_repo' in m}
m = models.get("$MODEL_ID")
if not m:
    print("NOT_FOUND")
    sys.exit(0)
print(f"HF_REPO='{m['hf_repo']}'")
print(f"REVISION='{m.get('revision') or 'UNSET'}'")
print(f"PRECISION='{m.get('precision','fp16')}'")
print(f"TIER='{m.get('tier','mid')}'")
print(f"VRAM_GB={m.get('vram_gb_estimate',0)}")
print(f"EXTRA_ARGS='{m.get('vllm_extra_args','')}'")
print(f"QUANTIZED_PATH='{m.get('quantized_path','NONE')}'")
EOF
)

[[ "$MODEL_CONFIG" == "NOT_FOUND" ]] && error "Model '$MODEL_ID' not found in registry.yaml"
eval "$MODEL_CONFIG"

# ── Validation ────────────────────────────────────────────────────────────────
[[ "$REVISION" == "UNSET" ]] && error "Revision SHA not set for $MODEL_ID in registry.yaml — pin it first!"

info "Model:     $MODEL_ID"
info "HF Repo:   $HF_REPO"
info "Revision:  $REVISION"
info "Precision: $PRECISION"
info "Tier:      $TIER"
info "VRAM Est:  ${VRAM_GB} GB"

# XL tier safety check
if [[ "$TIER" == "xl" ]]; then
  warn "XL tier model — verifying no other models are loaded..."
  # Check vLLM for active models
  ACTIVE=$(curl -s http://localhost:8000/v1/models 2>/dev/null | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    models=[m['id'] for m in d.get('data',[])]
    print(','.join(models) if models else 'none')
except: print('unreachable')
")
  info "Currently loaded: $ACTIVE"
fi

# ── Determine model path ──────────────────────────────────────────────────────
if [[ "$QUANTIZED_PATH" != "NONE" && -d "$QUANTIZED_PATH" ]]; then
  MODEL_PATH="$QUANTIZED_PATH"
  info "Using quantized model at: $MODEL_PATH"
elif [[ "$QUANTIZED_PATH" != "NONE" && ! -d "$QUANTIZED_PATH" ]]; then
  warn "Quantized path not found: $QUANTIZED_PATH"
  warn "Either run models/quantize.sh first, or set precision to fp16"
  read -p "Continue with HF repo (fp16 download)? [y/N]: " CONT
  [[ "$CONT" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }
  MODEL_PATH="$HF_REPO"
else
  MODEL_PATH="$HF_REPO"
fi

# ── Stop existing vLLM container ──────────────────────────────────────────────
info "Stopping existing vLLM container..."
docker stop vllm 2>/dev/null || true
docker rm -f vllm 2>/dev/null || true

# ── Start vLLM with new model ─────────────────────────────────────────────────
info "Starting vLLM with $MODEL_ID..."

SERVE_CMD="$MODEL_PATH --host 0.0.0.0 --port 8000 $EXTRA_ARGS"
[[ "$REVISION" != "UNSET" && "$MODEL_PATH" == "$HF_REPO" ]] && \
  SERVE_CMD="$SERVE_CMD --revision $REVISION"

echo -e "${CYAN}Command: vllm serve $SERVE_CMD${NC}"

docker run -d \
  --name vllm \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e HF_HOME=/data/models \
  -v /data/models:/data/models \
  -p 8000:8000 \
  --network benchmark-lab \
  vllm/vllm-openai:latest \
  $SERVE_CMD

# ── Wait for readiness ────────────────────────────────────────────────────────
info "Waiting for vLLM to be ready (model loading can take 2–10 min for large models)..."
ATTEMPTS=0
MAX_ATTEMPTS=60
until curl -sf http://localhost:8000/health &>/dev/null; do
  sleep 10
  ATTEMPTS=$((ATTEMPTS + 1))
  echo -n "."
  [[ $ATTEMPTS -ge $MAX_ATTEMPTS ]] && error "vLLM failed to start after $((MAX_ATTEMPTS * 10))s — check docker logs vllm"
done
echo ""

info "vLLM ready. Verifying model..."
LOADED=$(curl -s http://localhost:8000/v1/models | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(d['data'][0]['id'] if d.get('data') else 'unknown')
")
info "Loaded model: $LOADED"

# ── Log to MLflow ─────────────────────────────────────────────────────────────
python3 - <<EOF
import mlflow, os, datetime
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI","http://localhost:5000"))
with mlflow.start_run(run_name=f"model_swap_{datetime.date.today()}"):
    mlflow.log_params({
        "model_id": "$MODEL_ID",
        "hf_repo": "$HF_REPO",
        "revision": "$REVISION",
        "precision": "$PRECISION",
        "tier": "$TIER",
        "model_path": "$MODEL_PATH"
    })
    mlflow.log_metric("vram_gb_estimated", $VRAM_GB)
print("Model swap logged to MLflow")
EOF

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  $MODEL_ID is ready${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  API: http://localhost:8000/v1"
echo "  Test: curl http://localhost:8000/v1/models"
