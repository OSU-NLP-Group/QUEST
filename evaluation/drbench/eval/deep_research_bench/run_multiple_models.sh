#!/bin/bash

# =============================================================================
# Run Benchmark with Multiple Model Configurations
# =============================================================================
# This script runs the benchmark multiple times with different model configs
# Useful for comparing different LLM providers
# =============================================================================

# Target models to evaluate (articles to score)
TARGET_MODELS=("claude-3-7-sonnet-latest")

# Common parameters
RAW_DATA_DIR="data/test_data/raw_data"
CLEANED_DATA_DIR="data/test_data/cleaned_data"
QUERY_DATA_PATH="data/prompt_data/query.jsonl"
MAX_WORKERS=5

# Optional flags (uncomment to enable)
# LIMIT="--limit 2"
# SKIP_CLEANING="--skip_cleaning"
# ONLY_ZH="--only_zh"
# ONLY_EN="--only_en"
# FORCE="--force"

# Log file
OUTPUT_LOG_FILE="multi_model_output.log"
echo "Starting multi-model benchmark - $(date)" > "$OUTPUT_LOG_FILE"

# =============================================================================
# Define Model Configurations
# Add or remove configurations as needed
# =============================================================================

declare -A MODEL_CONFIGS

# Configuration format: "name|eval_model|clean_model|env_setup"

MODEL_CONFIGS["gemini"]="Gemini|gemini/gemini-2.5-pro-preview-06-05|gemini/gemini-2.5-flash-preview-05-20|export GEMINI_API_KEY=\"\${GEMINI_API_KEY}\""

MODEL_CONFIGS["openai"]="OpenAI GPT-4|gpt-4|gpt-3.5-turbo|export OPENAI_API_KEY=\"\${OPENAI_API_KEY}\""

MODEL_CONFIGS["claude"]="Anthropic Claude|claude-3-opus-20240229|claude-3-sonnet-20240229|export ANTHROPIC_API_KEY=\"\${ANTHROPIC_API_KEY}\""

MODEL_CONFIGS["deepseek"]="DeepSeek|deepseek/deepseek-chat|deepseek/deepseek-chat|export DEEPSEEK_API_KEY=\"\${DEEPSEEK_API_KEY}\""

# Uncomment to add Azure OpenAI (replace with your actual values)
# MODEL_CONFIGS["azure"]="Azure OpenAI|azure/gpt-4-deployment|azure/gpt-35-turbo-deployment|export AZURE_API_KEY=\"your-azure-api-key\"; export AZURE_API_BASE=\"https://your-resource.openai.azure.com/\"; export AZURE_API_VERSION=\"2024-02-15-preview\""

# Uncomment to add AWS Bedrock (replace with your actual values)
# MODEL_CONFIGS["bedrock"]="AWS Bedrock|bedrock/anthropic.claude-3-sonnet-20240229-v1:0|bedrock/anthropic.claude-3-haiku-20240307-v1:0|export AWS_ACCESS_KEY_ID=\"your-aws-access-key-id\"; export AWS_SECRET_ACCESS_KEY=\"your-aws-secret-key\"; export AWS_REGION_NAME=\"us-east-1\""

# =============================================================================
# Select which configurations to run
# Comment out any you don't want to run
# =============================================================================

CONFIGS_TO_RUN=(
    "gemini"
    # "openai"
    # "claude"
    # "deepseek"
    # "azure"
    # "bedrock"
)

# =============================================================================
# Run benchmarks
# =============================================================================

echo "========================================="
echo "Multi-Model Benchmark Runner"
echo "========================================="
echo "Target Models: ${TARGET_MODELS[@]}"
echo "Configurations: ${CONFIGS_TO_RUN[@]}"
echo "========================================="
echo ""

TOTAL_RUNS=$((${#TARGET_MODELS[@]} * ${#CONFIGS_TO_RUN[@]}))
CURRENT_RUN=0

for TARGET_MODEL in "${TARGET_MODELS[@]}"; do
    for CONFIG_NAME in "${CONFIGS_TO_RUN[@]}"; do
        ((CURRENT_RUN++))

        # Get configuration
        CONFIG="${MODEL_CONFIGS[$CONFIG_NAME]}"
        if [ -z "$CONFIG" ]; then
            echo "Error: Configuration '$CONFIG_NAME' not found!"
            continue
        fi

        # Parse configuration
        IFS='|' read -r DISPLAY_NAME EVAL_MODEL CLEAN_MODEL ENV_SETUP <<< "$CONFIG"

        echo ""
        echo "========================================="
        echo "Run $CURRENT_RUN of $TOTAL_RUNS"
        echo "========================================="
        echo "Target:     $TARGET_MODEL"
        echo "Config:     $DISPLAY_NAME"
        echo "Eval Model: $EVAL_MODEL"
        echo "Clean Model: $CLEAN_MODEL"
        echo "========================================="
        echo ""

        # Set up environment
        eval "$ENV_SETUP"

        # Create output directory
        OUTPUT_DIR="results/multi_model/${CONFIG_NAME}/${TARGET_MODEL}"
        mkdir -p "$OUTPUT_DIR"

        # Build command
        PYTHON_CMD="python -u deepresearch_bench_race.py \"$TARGET_MODEL\" \
            --raw_data_dir $RAW_DATA_DIR \
            --cleaned_data_dir $CLEANED_DATA_DIR \
            --max_workers $MAX_WORKERS \
            --query_file $QUERY_DATA_PATH \
            --output_dir $OUTPUT_DIR \
            --eval_model $EVAL_MODEL \
            --clean_model $CLEAN_MODEL"

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

        # Log command
        echo "Command: $PYTHON_CMD" | tee -a "$OUTPUT_LOG_FILE"
        echo "" | tee -a "$OUTPUT_LOG_FILE"

        # Execute
        if eval $PYTHON_CMD >> "$OUTPUT_LOG_FILE" 2>&1; then
            echo "✓ Completed: $DISPLAY_NAME for $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
            echo "  Results: $OUTPUT_DIR/race_result.txt" | tee -a "$OUTPUT_LOG_FILE"
        else
            echo "✗ Failed: $DISPLAY_NAME for $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
        fi

        echo "" | tee -a "$OUTPUT_LOG_FILE"
    done
done

# =============================================================================
# Generate comparison summary
# =============================================================================

echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "=========================================" | tee -a "$OUTPUT_LOG_FILE"
echo "Benchmark Summary" | tee -a "$OUTPUT_LOG_FILE"
echo "=========================================" | tee -a "$OUTPUT_LOG_FILE"

for TARGET_MODEL in "${TARGET_MODELS[@]}"; do
    echo "" | tee -a "$OUTPUT_LOG_FILE"
    echo "Target Model: $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
    echo "-----------------------------------------" | tee -a "$OUTPUT_LOG_FILE"

    for CONFIG_NAME in "${CONFIGS_TO_RUN[@]}"; do
        CONFIG="${MODEL_CONFIGS[$CONFIG_NAME]}"
        IFS='|' read -r DISPLAY_NAME EVAL_MODEL CLEAN_MODEL ENV_SETUP <<< "$CONFIG"

        RESULT_FILE="results/multi_model/${CONFIG_NAME}/${TARGET_MODEL}/race_result.txt"

        echo "" | tee -a "$OUTPUT_LOG_FILE"
        echo "Configuration: $DISPLAY_NAME" | tee -a "$OUTPUT_LOG_FILE"

        if [ -f "$RESULT_FILE" ]; then
            cat "$RESULT_FILE" | tee -a "$OUTPUT_LOG_FILE"
        else
            echo "  No results found" | tee -a "$OUTPUT_LOG_FILE"
        fi
    done

    echo "" | tee -a "$OUTPUT_LOG_FILE"
done

echo "=========================================" | tee -a "$OUTPUT_LOG_FILE"
echo "All benchmarks completed!" | tee -a "$OUTPUT_LOG_FILE"
echo "Detailed logs: $OUTPUT_LOG_FILE" | tee -a "$OUTPUT_LOG_FILE"
echo "=========================================" | tee -a "$OUTPUT_LOG_FILE"
