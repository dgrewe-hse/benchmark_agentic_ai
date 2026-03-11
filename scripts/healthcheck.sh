#!/usr/bin/env bash
# healthcheck.sh — Verify all services are running correctly

set -euo pipefail
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
PASS=0; FAIL=0

check() {
  local name="$1"; local cmd="$2"
  if eval "$cmd" &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} $name"
    PASS=$((PASS+1))
  else
    echo -e "  ${RED}✗${NC} $name"
    FAIL=$((FAIL+1))
  fi
}

echo ""
echo "Benchmark Lab Healthcheck"
echo "========================="
echo ""
echo "Infrastructure:"
check "Docker daemon"          "docker info"
check "GPU visible to Docker"  "docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi"
check "vLLM API"               "curl -sf http://localhost:8000/health"
check "MLflow server"          "curl -sf http://localhost:5000/health"
check "Prometheus"             "curl -sf http://localhost:9090/-/healthy"
check "Grafana"                "curl -sf http://localhost:3000/api/health"
check "DCGM Exporter"          "curl -sf http://localhost:9400/metrics"
check "OpenHands"              "curl -sf http://localhost:3001"

echo ""
echo "Python Environment:"
check "venv active"            "[[ -n \${VIRTUAL_ENV:-} ]]"
check "mlflow importable"      "python3 -c 'import mlflow'"
check "openai importable"      "python3 -c 'import openai'"
check "vllm importable"        "python3 -c 'import vllm'"
check "lm_eval importable"     "python3 -c 'import lm_eval'"

echo ""
echo "Data Directories:"
check "/data/models/hub"        "[[ -d /data/models/hub ]]"
check "/data/benchmark-results" "[[ -d /data/benchmark-results ]]"
check "/data/mlflow"            "[[ -d /data/mlflow ]]"

echo ""
echo "Environment Variables:"
check "HF_HOME set"            "[[ -n \${HF_HOME:-} ]]"
check "MLFLOW_TRACKING_URI"    "[[ -n \${MLFLOW_TRACKING_URI:-} ]]"
check "OPENAI_API_KEY set"     "[[ -n \${OPENAI_API_KEY:-} ]]"
check "ANTHROPIC_API_KEY set"  "[[ -n \${ANTHROPIC_API_KEY:-} ]]"

echo ""
echo "─────────────────────────────────────"
echo -e "  Passed: ${GREEN}$PASS${NC}   Failed: ${RED}$FAIL${NC}"
echo "─────────────────────────────────────"
[[ $FAIL -eq 0 ]] && echo -e "${GREEN}All checks passed — lab is ready.${NC}" \
                  || echo -e "${YELLOW}Fix failing checks before running benchmarks.${NC}"
echo ""
