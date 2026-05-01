#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Model and Inference Hyperparameters
export DATASET="${DATASET:-extracted_questions.jsonl}"
export OUTPUT_PATH="${OUTPUT_PATH:-inference_results/ref}"
export TASK_LOG_DIR="${TASK_LOG_DIR:-log_dir/ref.log}"
export ROLLOUT_COUNT="${ROLLOUT_COUNT:-1}"
export TEMPERATURE="${TEMPERATURE:-1}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.1}"
export MAX_WORKERS="${MAX_WORKERS:-10}"
export MODEL_NAME="${MODEL_NAME:-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"

# API Keys and External Services
# Serper API for web search and Google Scholar
export SERPER_KEY_ID="${SERPER_KEY_ID:-your_serper_api_key}"

# Jina API for web page reading
export JINA_API_KEYS="${JINA_API_KEYS:-your_jina_api_key}"

# Summary model configuration
export SUMMARY_AZURE_API_KEY="${SUMMARY_AZURE_API_KEY:-your_azure_api_key_here}"
export SUMMARY_AZURE_API_BASE="${SUMMARY_AZURE_API_BASE:-https://your-azure-endpoint.openai.azure.com}"
export SUMMARY_AZURE_API_VERSION="${SUMMARY_AZURE_API_VERSION:-2025-01-01-preview}"
export SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-azure/gpt-5-mini}"

# inference model configuration
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-}"
export AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-2}"


# Cache configuration
export VISIT_CACHE_ENABLED="${VISIT_CACHE_ENABLED:-true}"
export VISIT_CACHE_FILE="${VISIT_CACHE_FILE:-${REPO_ROOT}/database/visit_cache.db}"
export VISIT_CACHE_RESUME="${VISIT_CACHE_RESUME:-true}"
export SEARCH_CACHE_ENABLED="${SEARCH_CACHE_ENABLED:-true}"
export SEARCH_CACHE_FILE="${SEARCH_CACHE_FILE:-${REPO_ROOT}/database/search_cache.db}"
export SEARCH_CACHE_RESUME="${SEARCH_CACHE_RESUME:-true}"

mkdir -p "${REPO_ROOT}/database"
python -u "$SCRIPT_DIR/run_multi_react_ref_gen.py" --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers "$MAX_WORKERS" --model "$MODEL_NAME" --temperature "$TEMPERATURE" --presence_penalty "$PRESENCE_PENALTY" --total_splits "${WORLD_SIZE:-1}" --worker_split "$((${RANK:-0} + 1))" --roll_out_count "$ROLLOUT_COUNT"
