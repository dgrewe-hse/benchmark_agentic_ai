#!/usr/bin/env bash
# =============================================================================
# start.sh — Start the full benchmark infrastructure stack
#
# Prerequisites: bash scripts/setup.sh must have been run first.
#
# Usage: bash scripts/start.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

COMPOSE_FILE="infrastructure/docker-compose.yml"

# ── Preflight checks ──────────────────────────────────────────────────────────
info "Running preflight checks..."

[[ -d .venv ]] || error ".venv not found — run 'bash scripts/setup.sh' first"
[[ -f "$COMPOSE_FILE" ]] || error "$COMPOSE_FILE not found"

command -v docker &>/dev/null || error "Docker not found"
docker info &>/dev/null       || error "Docker daemon is not running"

# Verify GPU access for Docker
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi \
  &>/dev/null || error "Docker GPU access failed — check nvidia-container-toolkit"
info "Docker GPU access: OK"

# Activate virtual environment
source .venv/bin/activate

# ── Pull latest images ────────────────────────────────────────────────────────
info "Pulling latest Docker images (this may take several minutes)..."
docker compose -f "$COMPOSE_FILE" pull

# ── Evict stray containers that would conflict with compose ───────────────────
for NAME in vllm; do
  if docker inspect "$NAME" &>/dev/null; then
    warn "Removing stray container '$NAME' (not managed by compose)..."
    docker stop "$NAME" 2>/dev/null || true
    docker rm -f "$NAME" 2>/dev/null || true
  fi
done

# ── Start infrastructure stack ────────────────────────────────────────────────
info "Starting infrastructure stack..."
docker compose -f "$COMPOSE_FILE" up -d

# ── Wait for services ─────────────────────────────────────────────────────────
info "Waiting for services to become healthy (30s)..."
sleep 30

# ── Initialise results database ───────────────────────────────────────────────
info "Initialising results database..."
if [[ -f scoring/results_db.py ]]; then
  python3 scoring/results_db.py --init
else
  warn "scoring/results_db.py not found — skipping DB init (commit scoring/ and re-run to fix)"
fi

# ── Healthcheck ───────────────────────────────────────────────────────────────
info "Running healthcheck..."
bash scripts/healthcheck.sh

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Stack is up!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Grafana:   http://localhost:3000  (admin / changeme)"
echo "  MLflow:    http://localhost:5000"
echo "  vLLM API:  http://localhost:8000/v1"
echo "  OpenHands: http://localhost:3001"
echo ""
echo "  Next steps:"
echo "  1. Load a model:        bash scripts/model_swap.sh qwen2.5-coder-7b"
echo "  2. Run academic bench:  bash benchmarks/academic/run_lm_eval.sh qwen2.5-coder-7b"
echo "  3. Run infra bench:     bash benchmarks/infrastructure/run_genai_perf.sh qwen2.5-coder-7b"
echo ""
