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
print(f"CUDA_SM_MIN='{m.get('cuda_sm_min','')}'")
EOF
)

[[ "$MODEL_CONFIG" == "NOT_FOUND" ]] && error "Model '$MODEL_ID' not found in registry.yaml"
eval "$MODEL_CONFIG"

# ── Validation ────────────────────────────────────────────────────────────────
[[ "$REVISION" == "UNSET" ]] && error "Revision SHA not set for $MODEL_ID in registry.yaml — pin it first!"

# ── GPU Compute Capability Check ──────────────────────────────────────────────
GPU_SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || echo "unknown")
if [[ "$GPU_SM" == "unknown" ]]; then
  warn "Could not detect GPU compute capability — skipping compatibility check"
else
  info "GPU compute capability: SM $GPU_SM"
  if [[ -n "$CUDA_SM_MIN" ]]; then
    COMPAT=$(python3 -c "
gpu = tuple(int(x) for x in '$GPU_SM'.split('.'))
req = tuple(int(x) for x in '$CUDA_SM_MIN'.split('.'))
print('ok' if gpu >= req else 'incompatible')
")
    if [[ "$COMPAT" == "incompatible" ]]; then
      warn "Model '$MODEL_ID' requires CUDA SM >= $CUDA_SM_MIN (GPU is SM $GPU_SM)"
      warn "Model may fail to load or produce incorrect results"
      read -p "Continue anyway? [y/N]: " CONT
      [[ "$CONT" =~ ^[Yy]$ ]] || { info "Aborted."; exit 0; }
    else
      info "Compute capability SM $GPU_SM satisfies model requirement SM >= $CUDA_SM_MIN"
    fi
  fi
fi

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
  ACTIVE=$(curl -sf http://localhost:8000/v1/models 2>/dev/null | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    models=[m['id'] for m in d.get('data',[])]
    print(','.join(models) if models else 'none')
except: print('none')
" 2>/dev/null || echo "none")
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

# ── Pre-download model weights ────────────────────────────────────────────────
# Download before starting vLLM so startup timeout covers loading only, not downloading.
SNAPSHOT_DIR="/data/models/snapshots/$MODEL_ID"
if [[ -d "$SNAPSHOT_DIR" ]]; then
  info "Snapshot already exists at $SNAPSHOT_DIR — skipping download"
else
  info "Downloading model weights to $SNAPSHOT_DIR (this may take a while for large models)..."
  python3 - <<PYEOF
from huggingface_hub import snapshot_download
import sys
try:
    path = snapshot_download(
        repo_id="$HF_REPO",
        revision="$REVISION" if "$REVISION" != "UNSET" else None,
        local_dir="$SNAPSHOT_DIR",
        ignore_patterns=["*.pt", "*.bin"],  # prefer safetensors
    )
    print(f"Downloaded to: {path}")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
  info "Download complete."
fi
MODEL_PATH="$SNAPSHOT_DIR"

# ── Stop existing vLLM container ──────────────────────────────────────────────
info "Stopping existing vLLM container..."
docker stop vllm 2>/dev/null || true
docker rm -f vllm 2>/dev/null || true

# ── Start vLLM with new model ─────────────────────────────────────────────────
info "Starting vLLM with $MODEL_ID..."

SERVE_CMD="$MODEL_PATH --host 0.0.0.0 --port 8000 $EXTRA_ARGS"
# Revision is baked into the snapshot dir — no --revision flag needed

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
# XL models loading from disk can take 15–30 min; others typically 2–5 min.
[[ "$TIER" == "xl" ]] && MAX_ATTEMPTS=180 || MAX_ATTEMPTS=60   # 30 min vs 10 min
info "Waiting for vLLM to be ready (timeout: $((MAX_ATTEMPTS * 10))s)..."
ATTEMPTS=0
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
