#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-time environment bootstrap for DGX OS
#
# Installs: system packages, Python venv, core libs, benchmark tools
# Does NOT start services — run scripts/start.sh for that.
#
# Usage: bash scripts/setup.sh
# =============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Preflight checks ──────────────────────────────────────────────────────────
info "Running preflight checks..."

command -v nvidia-smi &>/dev/null || error "nvidia-smi not found — is this DGX OS?"
command -v docker     &>/dev/null || error "Docker not found"
command -v python3    &>/dev/null || error "python3 not found"
command -v git        &>/dev/null || error "git not found"

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
CUDA=$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',' || echo "unknown")
info "NVIDIA Driver: $DRIVER | CUDA: $CUDA"

# ── System packages ───────────────────────────────────────────────────────────
info "Installing system packages..."
sudo apt-get install -y --quiet \
  git-lfs \
  build-essential \
  curl \
  python3-venv \
  python3-pip

git lfs install
info "Git LFS ready"

# ── Data directories ──────────────────────────────────────────────────────────
info "Creating data directories..."
sudo mkdir -p \
  /data/models/hub \
  /data/models/quantized \
  /data/benchmark-results/academic \
  /data/benchmark-results/challenges \
  /data/benchmark-results/infrastructure \
  /data/mlflow/artifacts

sudo chown -R "$USER:$USER" /data
info "Data directories ready at /data/"

# ── Environment variables ─────────────────────────────────────────────────────
PROFILE_FILE="$HOME/.bashrc"
if ! grep -q "BENCHMARK_LAB" "$PROFILE_FILE" 2>/dev/null; then
  cat >> "$PROFILE_FILE" << 'EOF'

# ── Benchmark Lab ──────────────────────────────────────────
export HF_HOME=/data/models/hub
export MLFLOW_TRACKING_URI=http://localhost:5000
export VLLM_BASE_URL=http://localhost:8000/v1
export BENCHMARK_LAB=1
# Add your API keys:
# export OPENAI_API_KEY=sk-...
# export ANTHROPIC_API_KEY=sk-ant-...
EOF
  info "Environment variables added to $PROFILE_FILE"
  warn "Run: source ~/.bashrc (or open a new terminal) to apply"
fi

# ── Python virtual environment ────────────────────────────────────────────────
info "Setting up Python virtual environment..."
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  info ".venv created"
else
  info ".venv already exists — activating existing environment"
fi
source .venv/bin/activate
info "Virtual environment active: $VIRTUAL_ENV"
pip install --upgrade pip --quiet

# ── Core Python packages ──────────────────────────────────────────────────────
info "Installing core Python packages..."
pip install \
  vllm \
  transformers \
  datasets \
  huggingface_hub \
  mlflow \
  openai \
  anthropic \
  langchain \
  langgraph \
  langchain-openai \
  pyyaml \
  pandas \
  numpy \
  scipy \
  matplotlib \
  plotly \
  jupyter \
  pytest \
  pytest-cov \
  pylint \
  radon \
  sqlalchemy \
  flask \
  tqdm \
  rich

info "Installing llm-compressor (quantization support)..."
pip install --quiet llm-compressor \
  || warn "llm-compressor not available for this platform — skipping (only needed for fp8/AWQ quantization)"

# ── Benchmark tools ───────────────────────────────────────────────────────────
TOOLS_DIR="$(pwd)/tools"
mkdir -p "$TOOLS_DIR"

clone_if_missing() {
  local name="$1"; local url="$2"; local dir="$TOOLS_DIR/$name"
  if [[ -d "$dir/.git" ]]; then
    info "$name already cloned — skipping"
  else
    info "Cloning $name..."
    git clone --depth 1 "$url" "$dir"
  fi
}

install_from_dir() {
  local name="$1"; local dir="$TOOLS_DIR/$name"
  if [[ -f "$dir/pyproject.toml" ]]; then
    info "Installing $name (pyproject.toml)..."
    pip install --quiet -e "$dir" \
      || warn "$name install failed — skipping (manual install may be required)"
  elif [[ -f "$dir/requirements.txt" ]]; then
    info "Installing $name (requirements.txt)..."
    pip install --quiet -r "$dir/requirements.txt" \
      || warn "$name install failed — skipping (manual install may be required)"
  else
    warn "$name: no pyproject.toml or requirements.txt found — skipping install"
  fi
}

# EleutherAI LM Evaluation Harness
clone_if_missing "lm-evaluation-harness" "https://github.com/EleutherAI/lm-evaluation-harness"
info "Installing lm-evaluation-harness with vLLM extras..."
pip install --quiet -e "$TOOLS_DIR/lm-evaluation-harness[vllm]"

# LiveCodeBench
clone_if_missing "LiveCodeBench" "https://github.com/LiveCodeBench/LiveCodeBench"
install_from_dir "LiveCodeBench"

# AgentBench (multi-task agentic loop evaluation)
# Repo is cloned for reference only — pip install is skipped because AgentBench
# pins numpy~=1.23.5 which uses numpy.distutils, removed in Python 3.12.
# To run AgentBench, use its built-in Docker environment (see tools/AgentBench/docs/).
clone_if_missing "AgentBench" "https://github.com/THUDM/AgentBench"
info "AgentBench cloned — skipping pip install (requires Python ≤3.10; use Docker instead)"

# GenAI-Perf (NVIDIA throughput & latency profiling — replaces archived LLMPerf)
info "Installing GenAI-Perf..."
pip install --quiet genai-perf \
  || warn "genai-perf install failed — skipping (install manually: pip install genai-perf)"

# ── HuggingFace CLI ───────────────────────────────────────────────────────────
info "HuggingFace setup..."
echo ""
echo "  A HuggingFace token is required to download gated models (Llama, Phi)."
echo "  Get one at: https://huggingface.co/settings/tokens"
read -r -p "  Enter HF token (or press Enter to skip): " HF_TOKEN
if [[ -n "$HF_TOKEN" ]]; then
  hf auth login --token "$HF_TOKEN"
  info "HuggingFace CLI authenticated"
else
  warn "Skipped HF auth — run 'huggingface-cli login' before downloading gated models"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Next steps:"
echo "  1. source ~/.bashrc"
echo "  2. Add API keys to ~/.bashrc — used only as judge/evaluator models (GPT-4o, Claude), NOT for benchmarking (OPENAI_API_KEY, ANTHROPIC_API_KEY)"
echo "  3. Fill in revision SHAs in models/registry.yaml"
echo "  4. Start the full stack: bash scripts/start.sh"
echo ""
