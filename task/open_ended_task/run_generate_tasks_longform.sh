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
export SUMMARY_AZURE_API_KEY="${SUMMARY_AZURE_API_KEY:-your_azure_api_key_here}"
export SUMMARY_AZURE_API_BASE="${SUMMARY_AZURE_API_BASE:-https://your-azure-endpoint.openai.azure.com}"
export SUMMARY_AZURE_API_VERSION="${SUMMARY_AZURE_API_VERSION:-2025-01-01-preview}"
export SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-azure/gpt-5-mini}"

# DeepResearch model configuration
export DEEPRESEARCH_AWS_ACCESS_KEY_ID="${DEEPRESEARCH_AWS_ACCESS_KEY_ID:-your_aws_access_key_id_here}"
export DEEPRESEARCH_AWS_SECRET_ACCESS_KEY="${DEEPRESEARCH_AWS_SECRET_ACCESS_KEY:-your_aws_secret_access_key_here}"
export DEEPRESEARCH_AWS_REGION_NAME="${DEEPRESEARCH_AWS_REGION_NAME:-us-east-2}"
export DEEPRESEARCH_MODEL_NAME="${DEEPRESEARCH_MODEL_NAME:-bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0}"

export SAVE_TRAJ="${SAVE_TRAJ:-true}"
export TRAJ_DIR="${TRAJ_DIR:-./outputs/openended_trajectories}"

# Visit service configuration
export JINA_API_KEYS="${JINA_API_KEYS:-your_jina_api_key}"

# Cache configuration
export VISIT_CACHE_ENABLED="${VISIT_CACHE_ENABLED:-true}"
export VISIT_CACHE_FILE="${VISIT_CACHE_FILE:-${REPO_ROOT}/database/visit_cache.db}"
export VISIT_CACHE_RESUME="${VISIT_CACHE_RESUME:-true}"
export SEARCH_CACHE_ENABLED="${SEARCH_CACHE_ENABLED:-true}"
export SEARCH_CACHE_FILE="${SEARCH_CACHE_FILE:-${REPO_ROOT}/database/search_cache.db}"
export SEARCH_CACHE_RESUME="${SEARCH_CACHE_RESUME:-true}"

# ========== run script ==========
mkdir -p "${REPO_ROOT}/database"
python generate_longform_tasks.py
