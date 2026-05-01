#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Activate the DeepResearch environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepresearch

# Common API configuration
export SERPER_KEY_ID="${SERPER_KEY_ID:-your_serper_api_key}"

# LiteLLM model naming convention:
# - openai/<model-name>
# - azure/<deployment-name>
# - bedrock/<model-id>
# - vllm/<model-name>

# Summary model configuration
export SUMMARY_AZURE_API_KEY="${SUMMARY_AZURE_API_KEY:-your_azure_api_key}"
export SUMMARY_AZURE_API_BASE="${SUMMARY_AZURE_API_BASE:-https://your-azure-endpoint.openai.azure.com}"
export SUMMARY_AZURE_API_VERSION="${SUMMARY_AZURE_API_VERSION:-2025-01-01-preview}"
export SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-azure/gpt-5-mini}"

# DeepResearch model configuration
export DEEPRESEARCH_AWS_CREDENTIALS="${DEEPRESEARCH_AWS_CREDENTIALS:-[{\"access_key_id\":\"your_aws_access_key_id\",\"secret_access_key\":\"your_aws_secret_access_key\",\"region\":\"us-east-2\"}]}"
export DEEPRESEARCH_MODEL_NAME="${DEEPRESEARCH_MODEL_NAME:-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"
export DEEPRESEARCH_OPENAI_API_KEY="${DEEPRESEARCH_OPENAI_API_KEY:-}"

export SAVE_TRAJ="${SAVE_TRAJ:-true}"
export TRAJ_DIR="${TRAJ_DIR:-./outputs/objective_trajectories}"

# Memory configuration
export MEMORY_ENABLED="${MEMORY_ENABLED:-false}"
export MEMORY_CONTEXT_THRESHOLD="${MEMORY_CONTEXT_THRESHOLD:-32000}"
export TASK_LOG_DIR="${TASK_LOG_DIR:-./outputs/objective_task_logs}"

export MEMORY_TOKENIZER_PATH="${MEMORY_TOKENIZER_PATH:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"

# Visit service configuration
export VISIT_SERVICE="${VISIT_SERVICE:-jina}"
export JINA_API_KEYS="${JINA_API_KEYS:-your_jina_api_key}"

# Cache configuration
export VISIT_CACHE_ENABLED="${VISIT_CACHE_ENABLED:-true}"
export VISIT_CACHE_FILE="${VISIT_CACHE_FILE:-${REPO_ROOT}/database/visit_cache.db}"
export VISIT_CACHE_RESUME="${VISIT_CACHE_RESUME:-true}"
export SEARCH_CACHE_ENABLED="${SEARCH_CACHE_ENABLED:-true}"
export SEARCH_CACHE_FILE="${SEARCH_CACHE_FILE:-${REPO_ROOT}/database/search_cache.db}"
export SEARCH_CACHE_RESUME="${SEARCH_CACHE_RESUME:-true}"

mkdir -p "${REPO_ROOT}/database"
python generate_tasks.py
