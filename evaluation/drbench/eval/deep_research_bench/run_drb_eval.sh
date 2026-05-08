#!/bin/bash
# End-to-end DeepResearch Bench evaluation runner.
#
# Steps:
#   1. Convert QUEST inference iter*.jsonl outputs into DRB raw_data format.
#   2. Run RACE evaluation (report quality) for each iter.
#   3. Run FACT evaluation (citation accuracy) for each iter — extract,
#      deduplicate, scrape, validate, stat.
#
# Usage:
#   bash run_drb_eval.sh                      # full RACE + FACT
#   RUN_FACT=false bash run_drb_eval.sh       # RACE only
#   LIMIT=5 SMOKE_ITERS=1 bash run_drb_eval.sh  # quick smoke test
#
# Override the variables in the CONFIG section below from the shell, or edit
# them in place for each new run.

set -euo pipefail

SCRIPT_DIR="$(cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
DRB_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
QUEST_PUBLIC_ROOT="$(cd "$DRB_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Load shared API config (same yaml/json file used by inference)
# -----------------------------------------------------------------------------
API_CONFIG_FILE="${API_CONFIG_FILE:-${QUEST_PUBLIC_ROOT}/inference/api_config.yaml}"

load_api_config() {
    local config_file="$1"
    if [ ! -f "$config_file" ]; then
        echo "Error: API config file not found: ${config_file}"
        exit 1
    fi
    eval "$(
        python3 - "$config_file" <<'PY'
import json, shlex, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = json.load(f)
for key, value in config.get("common", {}).items():
    if value is None:
        value = ""
    elif isinstance(value, bool):
        value = "true" if value else "false"
    else:
        value = str(value)
    print(f"export {key}={shlex.quote(value)}")
PY
    )"
}

load_api_config "$API_CONFIG_FILE"
echo "Loaded API config from ${API_CONFIG_FILE}"

# Map keys from api_config.yaml to the names DRB / litellm expect:
# - JINA_API_KEYS (plural in inference config) -> JINA_API_KEY (singular, scrape.py)
# - API_KEY (OpenAI key in inference config)   -> OPENAI_API_KEY
if [ -n "${JINA_API_KEYS:-}" ] && [ -z "${JINA_API_KEY:-}" ]; then
    # JINA_API_KEYS may be a comma-separated list; take the first one
    export JINA_API_KEY="${JINA_API_KEYS%%,*}"
fi
if [ -n "${API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    export OPENAI_API_KEY="$API_KEY"
fi
if [ -n "${AZURE_OPENAI_ENDPOINT:-}" ]; then
    export AZURE_API_BASE="${AZURE_API_BASE:-${AZURE_OPENAI_ENDPOINT}}"
    export AZURE_API_KEY="${AZURE_API_KEY:-${API_KEY:-}}"
    export AZURE_API_VERSION="${AZURE_API_VERSION:-${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}}"
    DEFAULT_JUDGE_MODEL="${DEFAULT_JUDGE_MODEL:-azure/${AZURE_OPENAI_DEPLOYMENT:-gpt-5-mini}}"
else
    DEFAULT_JUDGE_MODEL="${DEFAULT_JUDGE_MODEL:-gpt-5-mini}"
fi

# =============================================================================
# CONFIG — override from the shell or edit these for each run
# =============================================================================

# Path to the directory containing iter1.jsonl / iter2.jsonl / iter3.jsonl
# (produced by inference/scripts/run_react_infer_drb.sh).
# Default points at the inference output layout used by scripts/run_react_infer_drb.sh:
#   ${OUTPUT_PATH}/${MODEL_NAME}/${DATASET_BASENAME}/iter*.jsonl
INFER_BASE_DIR="${INFER_BASE_DIR:-${QUEST_PUBLIC_ROOT}/inference/outputs/drbench/results/deepresearch/deepresearch_bench_questions}"

# Path to the original DRB question jsonl (provides question -> id mapping).
QUESTIONS_FILE="${QUESTIONS_FILE:-${DRB_DIR}/deepresearch_bench_questions.jsonl}"

# Identifier prefix used for converted file names AND TARGET_MODELS in eval.
# Final raw_data files: data/test_data/raw_data/${IDENTIFIER}-iter{1,2,3}.jsonl
IDENTIFIER="${IDENTIFIER:-quest-run}"

# Which rollout iters to evaluate
ITERS=(${ITERS:-1 2 3})

# DRB internal paths (usually no need to change)
RAW_DATA_DIR="${RAW_DATA_DIR:-data/test_data/raw_data}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
QUERY_DATA_PATH="${QUERY_DATA_PATH:-data/prompt_data/query.jsonl}"
N_TOTAL_PROCESS="${N_TOTAL_PROCESS:-100}"

# Smoke-test / phase switches:
#   LIMIT=N         -> only evaluate first N tasks per iter (RACE --limit)
#   SMOKE_ITERS=1   -> override ITERS to just (1) for a quick run
#   RUN_FACT=false  -> skip Phase 2 (FACT) entirely
LIMIT="${LIMIT:-}"
RUN_FACT="${RUN_FACT:-true}"
if [ -n "${SMOKE_ITERS:-}" ]; then
    ITERS=($SMOKE_ITERS)
fi

# -------- Judge models --------
# API keys (OPENAI_API_KEY / JINA_API_KEY) come from api_config.yaml above.
# Only the model names are configured here.
export EVAL_MODEL="${EVAL_MODEL:-$DEFAULT_JUDGE_MODEL}"
export CLEAN_MODEL="${CLEAN_MODEL:-$DEFAULT_JUDGE_MODEL}"
export FACT_MODEL="${FACT_MODEL:-$DEFAULT_JUDGE_MODEL}"
export DEFAULT_MODEL="${DEFAULT_MODEL:-$DEFAULT_JUDGE_MODEL}"

# -------- Vertex AI (optional, for vertexai/* models) --------
# Requires: `gcloud auth application-default login` already done.
# Set VERTEXAI_PROJECT in your environment or api_config.yaml.
if [ -n "${VERTEXAI_PROJECT:-}" ]; then
    export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-central1}"
    export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$VERTEXAI_PROJECT}"
    export GOOGLE_CLOUD_QUOTA_PROJECT="${GOOGLE_CLOUD_QUOTA_PROJECT:-$VERTEXAI_PROJECT}"
fi

# =============================================================================
# Step 1 — Convert inference outputs to DRB raw_data format
# =============================================================================
echo "==== Step 1: Converting inference outputs ===="
echo "  base-dir:       $INFER_BASE_DIR"
echo "  questions-file: $QUESTIONS_FILE"
echo "  identifier:     $IDENTIFIER"
echo "  iters:          ${ITERS[*]}"

python convert_to_eval_format.py \
    --base-dir "$INFER_BASE_DIR" \
    --questions-file "$QUESTIONS_FILE" \
    --output-dir "$RAW_DATA_DIR" \
    --identifier "$IDENTIFIER" \
    --iters "${ITERS[@]}"

# Verify the converted files actually exist
TARGET_MODELS=()
for i in "${ITERS[@]}"; do
    f="$RAW_DATA_DIR/${IDENTIFIER}-iter${i}.jsonl"
    if [ ! -f "$f" ]; then
        echo "Error: expected converted file not found: $f"
        exit 1
    fi
    TARGET_MODELS+=("${IDENTIFIER}-iter${i}")
done
echo "  -> generated ${#TARGET_MODELS[@]} raw_data files"

# =============================================================================
# Step 2 — Show resolved config (sanity check before running judge)
# =============================================================================
echo ""
echo "==== Step 2: Eval config ===="
echo "  EVAL_MODEL:    $EVAL_MODEL"
echo "  CLEAN_MODEL:   $CLEAN_MODEL"
echo "  FACT_MODEL:    $FACT_MODEL"
echo "  DEFAULT_MODEL: $DEFAULT_MODEL"
echo "  TARGET_MODELS: ${TARGET_MODELS[*]}"
echo "  N_PROCESS:     $N_TOTAL_PROCESS"

if [ -z "${JINA_API_KEY:-}" ]; then
    echo "Error: JINA_API_KEY is empty. Check that api_config.yaml has JINA_API_KEYS set."
    exit 1
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Warning: OPENAI_API_KEY is empty. RACE/FACT will fail unless judge models use a different provider."
fi

# =============================================================================
# Step 3 — Run RACE evaluation for each target model
# =============================================================================
OUTPUT_LOG_FILE="$SCRIPT_DIR/output.log"
echo "Starting benchmark tests, log output to: $OUTPUT_LOG_FILE" > "$OUTPUT_LOG_FILE"

for TARGET_MODEL in "${TARGET_MODELS[@]}"; do
    echo ""
    echo "==== Step 3: RACE eval for $TARGET_MODEL ===="
    echo -e "\n\n========== Starting evaluation for $TARGET_MODEL ==========\n" >> "$OUTPUT_LOG_FILE"

    RACE_OUTPUT="$OUTPUT_DIR/race/$TARGET_MODEL"
    mkdir -p "$RACE_OUTPUT"

    PYTHON_CMD="python -u deepresearch_bench_race.py \"$TARGET_MODEL\" \
        --raw_data_dir $RAW_DATA_DIR \
        --max_workers $N_TOTAL_PROCESS \
        --query_file $QUERY_DATA_PATH \
        --output_dir $RACE_OUTPUT \
        --eval_model $EVAL_MODEL \
        --clean_model $CLEAN_MODEL"
    if [ -n "$LIMIT" ]; then
        PYTHON_CMD="$PYTHON_CMD --limit $LIMIT"
    fi

    echo "Executing: $PYTHON_CMD" | tee -a "$OUTPUT_LOG_FILE"
    eval $PYTHON_CMD >> "$OUTPUT_LOG_FILE" 2>&1

    echo "Completed RACE eval: $TARGET_MODEL"
    echo -e "\n========== RACE test completed for $TARGET_MODEL ==========\n" >> "$OUTPUT_LOG_FILE"
done

# =============================================================================
# Step 4 — Run FACT (citation) evaluation for each target model
# =============================================================================
if [ "$RUN_FACT" != "true" ]; then
    echo ""
    echo "Skipping FACT phase for all models (RUN_FACT=$RUN_FACT)"
else
    for TARGET_MODEL in "${TARGET_MODELS[@]}"; do
        echo ""
        echo "==== Step 4: FACT eval for $TARGET_MODEL ===="
        echo -e "\n==== Phase 2: FACT eval for $TARGET_MODEL ====\n" >> "$OUTPUT_LOG_FILE"

        CITATION_OUTPUT="$OUTPUT_DIR/fact/$TARGET_MODEL"
        RAW_DATA_PATH="$RAW_DATA_DIR/$TARGET_MODEL.jsonl"
        mkdir -p "$CITATION_OUTPUT"

        echo "  [extract]     $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
        python -u -m utils.extract \
            --raw_data_path "$RAW_DATA_PATH" \
            --output_path "$CITATION_OUTPUT/extracted.jsonl" \
            --query_data_path "$QUERY_DATA_PATH" \
            --n_total_process "$N_TOTAL_PROCESS" \
            --model "$FACT_MODEL" >> "$OUTPUT_LOG_FILE" 2>&1

        echo "  [deduplicate] $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
        python -u -m utils.deduplicate \
            --raw_data_path "$CITATION_OUTPUT/extracted.jsonl" \
            --output_path "$CITATION_OUTPUT/deduplicated.jsonl" \
            --query_data_path "$QUERY_DATA_PATH" \
            --n_total_process "$N_TOTAL_PROCESS" >> "$OUTPUT_LOG_FILE" 2>&1

        echo "  [scrape]      $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
        python -u -m utils.scrape \
            --raw_data_path "$CITATION_OUTPUT/deduplicated.jsonl" \
            --output_path "$CITATION_OUTPUT/scraped.jsonl" \
            --n_total_process "$N_TOTAL_PROCESS" >> "$OUTPUT_LOG_FILE" 2>&1

        echo "  [validate]    $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
        python -u -m utils.validate \
            --raw_data_path "$CITATION_OUTPUT/scraped.jsonl" \
            --output_path "$CITATION_OUTPUT/validated.jsonl" \
            --query_data_path "$QUERY_DATA_PATH" \
            --n_total_process "$N_TOTAL_PROCESS" \
            --model "$FACT_MODEL" >> "$OUTPUT_LOG_FILE" 2>&1

        echo "  [stat]        $TARGET_MODEL" | tee -a "$OUTPUT_LOG_FILE"
        python -u -m utils.stat \
            --input_path "$CITATION_OUTPUT/validated.jsonl" \
            --output_path "$CITATION_OUTPUT/fact_result.txt" >> "$OUTPUT_LOG_FILE" 2>&1

        echo "Completed FACT eval: $TARGET_MODEL"
        echo -e "\n========== FACT test completed for $TARGET_MODEL ==========\n" >> "$OUTPUT_LOG_FILE"
    done
fi

echo ""
echo "All done. Logs: $OUTPUT_LOG_FILE"
echo "RACE results:  $SCRIPT_DIR/$OUTPUT_DIR/race/<model>/race_result.txt"
echo "FACT results:  $SCRIPT_DIR/$OUTPUT_DIR/fact/<model>/fact_result.txt"
