# Model and Inference Hyperparameters
export DATASET=extracted_questions.jsonl
export OUTPUT_PATH=inference_results/ref.jsonl
export TASK_LOG_DIR=log_dir/ref.log
export ROLLOUT_COUNT=1
export TEMPERATURE=1
export PRESENCE_PENALTY=1.1
export MAX_WORKERS=10

# API Keys and External Services
# Serper API for web search and Google Scholar
export SERPER_KEY_ID="${SERPER_KEY_ID:-your_serper_api_key}"

# Jina API for web page reading
export JINA_API_KEYS="${JINA_API_KEYS:-your_jina_api_key}"

# Summary model configuration
export SUMMARY_AZURE_API_KEY="${SUMMARY_AZURE_API_KEY:-your_azure_api_key_here}"
export SUMMARY_AZURE_API_BASE="${SUMMARY_AZURE_API_BASE:-https://your-azure-endpoint.openai.azure.com}"
export SUMMARY_AZURE_API_VERSION="2025-01-01-preview"
export SUMMARY_MODEL_NAME="azure/gpt-5-mini"

# inference model configuration
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}"
export AWS_REGION_NAME="${AWS_REGION_NAME}"


# Cache configuration
export VISIT_CACHE_ENABLED="${VISIT_CACHE_ENABLED:-true}"
export VISIT_CACHE_FILE="${VISIT_CACHE_FILE:-/fs/scratch/PAS1576/jianxie/DeepResearch/verl/recipe/deepresearch/database/visit_cache.db}"
export VISIT_CACHE_RESUME="${VISIT_CACHE_RESUME:-true}"
export SEARCH_CACHE_ENABLED="${SEARCH_CACHE_ENABLED:-true}"
export SEARCH_CACHE_FILE="${SEARCH_CACHE_FILE:-/fs/scratch/PAS1576/jianxie/DeepResearch/verl/recipe/deepresearch/database/search_cache.db}"
export SEARCH_CACHE_RESUME="${SEARCH_CACHE_RESUME:-true}"

python -u run_multi_react_ref_gen.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_NAME  --temperature $TEMPERATURE --presence_penalty $PRESENCE_PENALTY --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT