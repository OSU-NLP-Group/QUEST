#!/bin/bash
# Target model name list - Add multiple models to test them sequentially
# Each model should be a separate array element
TARGET_MODELS=("qwen3-moe-rl-45steps-16k-output-80k-memory-200turns-iter1" "qwen3-moe-rl-45steps-16k-output-80k-memory-200turns-iter2" "qwen3-moe-rl-45steps-16k-output-80k-memory-200turns-iter3")

# Example: Test multiple models
# TARGET_MODELS=("vanilla-iter2" "vanilla-iter3" "model-v1" "model-v2")

# Common parameters for both RACE and Citation evaluations
RAW_DATA_DIR="data/test_data/raw_data"
OUTPUT_DIR="results"
N_TOTAL_PROCESS=50
QUERY_DATA_PATH="data/prompt_data/query.jsonl"

# LLM model selection (litellm format)
# For evaluation (scoring), you can use:
# - Gemini: "gemini/gemini-2.5-pro-preview-06-05", "gemini/gemini-2.0-flash-exp"
# - OpenAI: "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"
# - Anthropic: "claude-3-sonnet-20240229", "claude-3-opus-20240229"
# - DeepSeek: "deepseek/deepseek-chat", "deepseek/deepseek-reasoner"
# - Azure OpenAI: "azure/<deployment_name>"
#   Requires: export AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION
# - AWS Bedrock: "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
#   Requires: export AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME
export OPENAI_API_KEY="${OPENAI_API_KEY:-your_openai_api_key}"

export JINA_API_KEY="${JINA_API_KEY:-your_jina_api_key}"

# Model configuration - export as environment variables for api.py
export EVAL_MODEL="vertexai/gemini-2.5-pro"

export AZURE_API_KEY="${AZURE_API_KEY:-your_azure_api_key}"
export AZURE_API_BASE="${AZURE_API_BASE:-your_azure_api_base}"
export AZURE_API_VERSION="${AZURE_API_VERSION:-your_azure_api_version}"

# For article cleaning, you can use a faster/cheaper model:
export CLEAN_MODEL="vertexai/gemini-2.5-flash"

# For citation extraction (FACT evaluation), you can use a faster model:
export FACT_MODEL="vertexai/gemini-2.5-flash"

# Set default model to avoid fallback to gemini
export DEFAULT_MODEL="vertexai/gemini-2.5-flash"

# Example: Using Azure OpenAI
# export EVAL_MODEL="azure/gpt-4-deployment"
# export CLEAN_MODEL="azure/gpt-35-turbo-deployment"
# export FACT_MODEL="azure/gpt-35-turbo-deployment"
# export DEFAULT_MODEL="azure/gpt-4-deployment"

# Example: Using AWS Bedrock
# export AWS_ACCESS_KEY_ID="your-aws-access-key-id"
# export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
# export AWS_REGION_NAME="us-east-1"
# export EVAL_MODEL="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
# export CLEAN_MODEL="bedrock/anthropic.claude-3-haiku-20240307-v1:0"
# export FACT_MODEL="bedrock/anthropic.claude-3-haiku-20240307-v1:0"
# export DEFAULT_MODEL="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"


# Limit on number of prompts to process (for testing). Uncomment to enable
# LIMIT="--limit 2"

# Skip article cleaning step. Uncomment to enable
# SKIP_CLEANING="--skip_cleaning"

# Only process specific language data. Uncomment to enable
# ONLY_ZH="--only_zh"  # Only process Chinese data
# ONLY_EN="--only_en"  # Only process English data

# Force re-evaluation even if results exist. Uncomment to enable
# FORCE="--force"

# Specify log output file
OUTPUT_LOG_FILE="output.log"

# Clear log file
echo "Starting benchmark tests, log output to: $OUTPUT_LOG_FILE" > "$OUTPUT_LOG_FILE"

# Loop through each model in the target models list
for TARGET_MODEL in "${TARGET_MODELS[@]}"; do
  echo "Running benchmark for target model: $TARGET_MODEL"
  echo -e "\n\n========== Starting evaluation for $TARGET_MODEL ==========\n" >> "$OUTPUT_LOG_FILE"

  # --- Phase 1: RACE Evaluation ---
  echo "==== Phase 1: Running RACE Evaluation for $TARGET_MODEL ====" | tee -a "$OUTPUT_LOG_FILE"
  RACE_OUTPUT="$OUTPUT_DIR/race/$TARGET_MODEL"
  mkdir -p $RACE_OUTPUT

  # Base command for current target model
  PYTHON_CMD="python -u deepresearch_bench_race.py \"$TARGET_MODEL\" --raw_data_dir $RAW_DATA_DIR --max_workers $N_TOTAL_PROCESS --query_file $QUERY_DATA_PATH --output_dir $RACE_OUTPUT --eval_model $EVAL_MODEL --clean_model $CLEAN_MODEL"

  # Add optional parameters
  if [[ -n "$LIMIT" ]]; then
    PYTHON_CMD="$PYTHON_CMD $LIMIT"
  fi

  if [[ -n "$SKIP_CLEANING" ]]; then
    PYTHON_CMD="$PYTHON_CMD $SKIP_CLEANING"
  fi
  
  if [[ -n "$ONLY_ZH" ]]; then
    PYTHON_CMD="$PYTHON_CMD $ONLY_ZH"
  fi
  
  if [[ -n "$ONLY_EN" ]]; then
    PYTHON_CMD="$PYTHON_CMD $ONLY_EN"
  fi
  
  if [[ -n "$FORCE" ]]; then
    PYTHON_CMD="$PYTHON_CMD $FORCE"
  fi

  # Execute command and append stdout and stderr to single log file
  echo "Executing command: $PYTHON_CMD" | tee -a "$OUTPUT_LOG_FILE"
  eval $PYTHON_CMD >> "$OUTPUT_LOG_FILE" 2>&1

  echo "Completed RACE benchmark test for target model: $TARGET_MODEL"
  echo -e "\n========== RACE test completed for $TARGET_MODEL ==========\n" >> "$OUTPUT_LOG_FILE"
  
  # --- Phase 2: Citation Evaluation ---
  # echo "==== Phase 2: Running FACT Evaluation for $TARGET_MODEL ====" | tee -a "$OUTPUT_LOG_FILE"

  # # Create citation output directory if it doesn't exist
  # CITATION_OUTPUT="$OUTPUT_DIR/fact/$TARGET_MODEL"
  # RAW_DATA_PATH="$RAW_DATA_DIR/$TARGET_MODEL.jsonl"
  # mkdir -p $CITATION_OUTPUT

  # # Run citation extraction, deduplication, scraping, and validation
  # echo "Extracting citations for $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
  # python -u -m utils.extract --raw_data_path $RAW_DATA_PATH --output_path $CITATION_OUTPUT/extracted.jsonl --query_data_path $QUERY_DATA_PATH --n_total_process $N_TOTAL_PROCESS --model $FACT_MODEL

  # echo "Deduplicate citations for $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
  # python -u -m utils.deduplicate --raw_data_path $CITATION_OUTPUT/extracted.jsonl --output_path $CITATION_OUTPUT/deduplicated.jsonl --query_data_path $QUERY_DATA_PATH --n_total_process $N_TOTAL_PROCESS

  # echo "Scrape webpages for $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
  # python -u -m utils.scrape --raw_data_path $CITATION_OUTPUT/deduplicated.jsonl --output_path $CITATION_OUTPUT/scraped.jsonl --n_total_process $N_TOTAL_PROCESS

  # echo "Validate citations for $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
  # python -u -m utils.validate --raw_data_path $CITATION_OUTPUT/scraped.jsonl --output_path $CITATION_OUTPUT/validated.jsonl --query_data_path $QUERY_DATA_PATH --n_total_process $N_TOTAL_PROCESS --model $FACT_MODEL

  # echo "Collecting statistics for $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
  # python -u -m utils.stat --input_path $CITATION_OUTPUT/validated.jsonl --output_path $CITATION_OUTPUT/fact_result.txt

  # echo "Completed FACT benchmark test for target model: $TARGET_MODEL"
  # echo -e "\n========== FACT test completed for $TARGET_MODEL ==========\n" >> "$OUTPUT_LOG_FILE"
  # echo "--------------------------------------------------"
done

echo "All benchmark tests completed. Logs saved in $OUTPUT_LOG_FILE"
