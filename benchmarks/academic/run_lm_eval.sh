#!/usr/bin/env bash
# =============================================================================
# run_lm_eval.sh — Run LM Evaluation Harness academic benchmarks
#
# Usage: bash benchmarks/academic/run_lm_eval.sh <model-id> [tasks]
#
# Examples:
#   bash benchmarks/academic/run_lm_eval.sh qwen2.5-coder-7b
#   bash benchmarks/academic/run_lm_eval.sh qwen2.5-coder-7b humaneval,mbpp
# =============================================================================

set -euo pipefail
source .venv/bin/activate

MODEL_ID="${1:-}"
TASKS="${2:-humaneval,mbpp}"
RESULTS_DIR="/data/benchmark-results/academic"

[[ -z "$MODEL_ID" ]] && { echo "Usage: $0 <model-id> [tasks]"; exit 1; }

# Load model config from registry
MODEL_INFO=$(python3 - <<EOF
import yaml
with open("models/registry.yaml") as f:
    reg = yaml.safe_load(f)
m = next((m for m in reg['models'] if m.get('id') == "$MODEL_ID"), None)
if not m or 'hf_repo' not in m:
    print("NOT_FOUND"); exit()
print(m['hf_repo'])
EOF
)
[[ "$MODEL_INFO" == "NOT_FOUND" ]] && { echo "Model $MODEL_ID not found in registry"; exit 1; }

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="$RESULTS_DIR/${MODEL_ID}_${TIMESTAMP}"
mkdir -p "$OUTPUT_PATH"

echo "Running LM Eval Harness"
echo "  Model:  $MODEL_ID ($MODEL_INFO)"
echo "  Tasks:  $TASKS"
echo "  Output: $OUTPUT_PATH"
echo ""

# Run via vLLM backend (model must already be loaded via model_swap.sh)
# --confirm_run_unsafe_code required for HumanEval (executes generated code locally)
lm_eval \
  --model vllm \
  --model_args "pretrained=$MODEL_INFO,base_url=http://localhost:8000/v1,tokenizer_backend=huggingface" \
  --tasks "$TASKS" \
  --batch_size auto \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --seed 42 \
  --confirm_run_unsafe_code

# Log results to MLflow
python3 - <<EOF
import mlflow, json, os, glob
from pathlib import Path

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
result_files = glob.glob("$OUTPUT_PATH/**/*.json", recursive=True)

with mlflow.start_run(run_name="lmeval_${MODEL_ID}_${TIMESTAMP}",
                      tags={"model_id": "$MODEL_ID", "benchmark": "lm_eval", "tasks": "$TASKS"}):
    mlflow.log_param("model_id", "$MODEL_ID")
    mlflow.log_param("tasks", "$TASKS")
    mlflow.log_param("timestamp", "$TIMESTAMP")

    for rf in result_files:
        with open(rf) as f:
            data = json.load(f)
        results = data.get("results", {})
        for task, metrics in results.items():
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{task}_{metric.replace(',','_')}", value)

    mlflow.log_artifacts("$OUTPUT_PATH", artifact_path="lm_eval_results")

print("Results logged to MLflow")
EOF

echo ""
echo "Done. Results at: $OUTPUT_PATH"
echo "MLflow run logged."
