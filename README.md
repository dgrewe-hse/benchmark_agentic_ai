# 🧪 Local Agentic AI Coding Benchmark

This repository contains the code and the documentation for local agentic AI coding benchmarks on NVIDIA DGX Spark (GB200 · 128 GB Unified Memory · DGX OS). It serves the purpose of systematically evaluating open-weight LLMs for agentic coding — academic benchmarks, real-world prototype challenges, and infrastructure profiling — feeding into consulting insights and scientific publication.


---

## Research Question

The main research question this repository aims to answer is:

> *How good are local coding LLMs for agentic setups? How do they compare to each other? How do they perform on real-world tasks using multi-agent architectures and evaluation metrics such as HumanEval, LiveCodeBench, SWE-bench, and Composite Agent Score (CAS)?*

---

## Repository Structure

The repository is structured as follows:

```
.
├── README.md                        # This file
├── ROADMAP.md                       # Phased project timeline
├── CONTRIBUTING.md                  # How to add models, challenges, results
│
├── docs/
│   ├── architecture.md              # System architecture & design decisions
│   ├── hardware-profile.md          # DGX Spark specs & resource strategy
│   ├── model-matrix.md              # All models, tiers, quantization strategy
│   ├── scoring-framework.md         # CAS formula, rubrics, judge protocol
│   └── publication-notes.md         # Research framing, target venues, methodology
│
├── infrastructure/
│   ├── docker-compose.yml           # Full stack: vLLM + monitoring + MLflow
│   ├── vllm/
│   │   ├── serve.sh                 # Model-aware launch script
│   │   └── configs/                 # Per-model vLLM config files
│   ├── monitoring/
│   │   ├── prometheus.yml           # Scrape config (DCGM + vLLM + MLflow)
│   │   └── grafana/
│   │       └── dashboards/          # Pre-built Grafana dashboard JSONs
│   └── sandbox/
│       └── Dockerfile               # Agent execution sandbox (Docker-in-Docker)
│
├── models/
│   ├── registry.yaml                # All model IDs, HF revision hashes, quant method
│   └── quantize.sh                  # llm-compressor fp8/AWQ batch script
│
├── benchmarks/
│   ├── academic/
│   │   ├── run_lm_eval.sh           # LM Evaluation Harness runner
│   │   ├── run_livecodebench.sh     # LiveCodeBench runner
│   │   └── run_swebench.sh          # SWE-bench Lite runner
│   ├── infrastructure/
│   │   └── run_genai_perf.sh        # Throughput & latency profiling
│   └── results/                     # Raw results (gitignored large files, schema here)
│       └── .gitkeep
│
├── challenges/
│   ├── README.md                    # Challenge overview & scoring protocol
│   ├── C1-invoice-extractor/        # PDF invoice extraction API
│   ├── C2-code-review-bot/          # GitHub PR review automation
│   ├── C3-anomaly-detector/         # Time-series anomaly microservice
│   └── C4-job-scheduler/            # Multi-tenant scheduling dashboard
│
├── scoring/
│   ├── cas_scorer.py                # Composite Agent Score calculator
│   ├── llm_judge.py                 # Frontier model judge (GPT-4o / Claude API)
│   ├── rubrics/                     # Per-challenge JSON rubrics
│   └── results_db.py               # SQLite result storage schema
│
├── analysis/
│   ├── correlation_study.ipynb      # Academic vs CAS correlation (publication core)
│   ├── model_comparison.ipynb       # Cross-model radar charts, heatmaps
│   └── agent_topology.ipynb         # Single vs multi-agent analysis
│
└── scripts/
    ├── setup.sh                     # Full environment bootstrap (DGX OS aware)
    ├── model_swap.sh                # Hot-swap model in vLLM between runs
    └── run_full_benchmark.sh        # End-to-end orchestration script
```

---

## Quick Start

> **Prerequisites:** DGX OS (NVIDIA drivers, Docker, nvidia-container-toolkit all pre-installed ✓)

```bash
# 1. Clone the repo
git clone https://github.com/dgrewe-hse/benchmark_agentic_ai
cd benchmark_agentic_ai

# 2. Install Python environment, system packages & benchmark tools (one-time)
bash scripts/setup.sh

# 3. Register your models
nano models/registry.yaml   # Add HF repo IDs + revision hashes

# 4. Start the full infrastructure stack (Docker Compose, DB init, healthcheck)
bash scripts/start.sh

# 5. Run your first academic benchmark
bash benchmarks/academic/run_lm_eval.sh qwen2.5-coder-7b
```

---

## Model Tiers at a Glance

| Tier | Models | Precision | Concurrent Runs |
|------|--------|-----------|-----------------|
| Small (≤8B) | Qwen2.5-Coder-7B, GLM-4-Flash, Phi-4-mini, Llama-3.2-3B | fp16 | Up to 3 |
| Mid (8–22B) | Qwen2.5-Coder-14B, GLM-4-Air, Llama-3.1-8B | fp16 | Up to 2 |
| Large (22–35B) | Qwen2.5-Coder-32B, DeepSeek-Coder-V2-Lite, GLM-4-Plus | fp8/AWQ | Up to 2 |
| XL (70B+) | Qwen2.5-Coder-72B, DeepSeek-R1-70B, Llama-3.3-70B | fp8 | Exclusive (1) |
| Reference | GPT-4o, Claude 3.5 Sonnet | API | Judge only |

→ Full details: [`docs/model-matrix.md`](docs/model-matrix.md)

---

## Benchmark Tracks

| Track | Tools | Output Metric |
|-------|-------|---------------|
| Academic | LM Eval Harness, LiveCodeBench, SWE-bench Lite | pass@1, pass@10, resolve rate |
| Real-World Agentic | OpenHands + LangGraph + Docker sandbox | Composite Agent Score (CAS) |
| Infrastructure | GenAI-Perf, DCGM, Nsight Systems | tokens/s, TTFT, VRAM, power |

→ Full scoring formula: [`docs/scoring-framework.md`](docs/scoring-framework.md)

---

## Agent Topologies Tested

Every real-world challenge is run under **all three** multi-agent patterns:

1. **Single-Agent** — one model, full context, no collaboration
2. **Planner + Executor** — 2-agent: reasoning/decomposition separated from coding
3. **Orchestrator + Coder + Reviewer** — 3-agent: full review loop

→ Architecture details: [`docs/architecture.md`](docs/architecture.md)

---

## Publication Target

This study targets an **arXiv preprint** as primary vehicle, with subsequent submission to MLSys, an ICLR workshop, or an applied NLP venue. The novel contribution is the systematic cross-correlation between academic benchmark scores and real-world agentic task performance, with hardware profiling on a GB200 unified-memory system.

→ Methodology & framing: [`docs/publication-notes.md`](docs/publication-notes.md)

--- 

## HuggingFace Account

Make sure to create a HuggingFace account and get an access token to various models: https://huggingface.co/settings/tokens. The installation script will prompt you to enter your token.

In order to be able to download the models, you need to be logged in to HuggingFace and to accept the license terms of the models you want to download.

---

## License

GPL-3.0 License: see [`LICENSE`](LICENSE)
