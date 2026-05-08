#!/bin/bash
# End-to-end LiveResearchBench (DeepEval) evaluation runner.
#
# Steps:
#   1. Convert QUEST inference iter*.jsonl outputs into the
#      <model_outputs>/<model_name>/qid_<qid>_report.md layout.
#   2. Run preprocess.py to build the reports JSON index.
#   3. Run main.py (DeepEval) on the chosen criteria + judge providers.
#
# Usage:
#   bash run_lrb_eval.sh            # first run (convert + preprocess + eval)
#   bash run_lrb_eval.sh            # re-run resumes from where it left off
#   bash run_lrb_eval.sh --fresh    # force re-convert + re-preprocess
#
# Override the variables in the CONFIG section below from the shell for each
# new run, or edit them in place.

set -euo pipefail

SCRIPT_DIR="$(cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
LRB_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
QUEST_PUBLIC_ROOT="$(cd "$LRB_EVAL_DIR/../.." && pwd)"
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

# Map keys from api_config.yaml to the names DeepEval expects.
if [ -n "${API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    export OPENAI_API_KEY="$API_KEY"
fi
export AZURE_OPENAI_JUDGE_DEPLOYMENT="${AZURE_OPENAI_JUDGE_DEPLOYMENT:-${AZURE_OPENAI_DEPLOYMENT:-gpt-5-mini}}"
# GEMINI_API_KEY / HF_TOKEN are expected to already be in api_config.yaml if used.
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN not set; preprocess.py loads the gated LRB dataset and will fail."
fi

# -------- Vertex AI (optional, for vertexai-based Gemini calls) --------
# Requires: `gcloud auth application-default login` already done.
# Set VERTEXAI_PROJECT in your environment or api_config.yaml.
if [ -n "${VERTEXAI_PROJECT:-}" ]; then
    export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-us-central1}"
    export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$VERTEXAI_PROJECT}"
    export GOOGLE_CLOUD_QUOTA_PROJECT="${GOOGLE_CLOUD_QUOTA_PROJECT:-$VERTEXAI_PROJECT}"
fi

# =============================================================================
# CONFIG — override from the shell or edit these for each run
# =============================================================================

# Path to the directory containing iter1.jsonl / iter2.jsonl / iter3.jsonl
# (produced by inference/scripts/run_react_infer_lrb.sh).
# Default points at the inference output layout used by scripts/run_react_infer_lrb.sh:
#   ${OUTPUT_PATH}/${MODEL_NAME}/${DATASET_BASENAME}/iter*.jsonl
INFER_BASE_DIR="${INFER_BASE_DIR:-${QUEST_PUBLIC_ROOT}/inference/outputs/lrb/results/deepresearch/liveresearchbench_questions}"

# Identifier prefix used for model_outputs subdir names.
# Final per-rollout dirs: <MODEL_OUTPUTS_DIR>/<IDENTIFIER>-iter{1,2,3}/qid_*.md
IDENTIFIER="${IDENTIFIER:-quest-run}"

# Which rollout iters to evaluate
ITERS=(${ITERS:-1 2 3})

# Working directories
MODEL_OUTPUTS_DIR="${MODEL_OUTPUTS_DIR:-$SCRIPT_DIR/model_outputs}"
EXTRACTED_DIR="${EXTRACTED_DIR:-$SCRIPT_DIR/extracted_reports}"

# Eval config
CRITERIA="${CRITERIA:-presentation,consistency,citation,coverage,depth}"

# Parse --fresh flag: force re-convert + re-preprocess
FRESH=false
for arg in "$@"; do
    if [ "$arg" = "--fresh" ]; then
        FRESH=true
    fi
done

QUESTIONS_FILE="${QUESTIONS_FILE:-${LRB_EVAL_DIR}/liveresearchbench_questions.jsonl}"

# Auto-prepare the LiveResearchBench question jsonl from HuggingFace if missing.
if [ ! -f "$QUESTIONS_FILE" ]; then
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "Error: HF_TOKEN not set; cannot download gated LiveResearchBench dataset."
        echo "       Add HF_TOKEN to api_config.yaml (under common:) or export it."
        exit 1
    fi
    echo "LRB question file not found, fetching from HuggingFace -> $QUESTIONS_FILE"
    python3 "${LRB_EVAL_DIR}/prepare_lrb_questions.py" \
        --output "$QUESTIONS_FILE" --use-realtime
fi

# Fixed-name reports JSON — eval always uses this path so that incremental
# results stay associated even after re-preprocessing with new tasks.
REPORTS_JSON="$EXTRACTED_DIR/reports_${IDENTIFIER}.json"

if [ "$FRESH" = "true" ] || [ ! -f "$REPORTS_JSON" ]; then
    # =============================================================================
    # Step 1 — Convert QUEST inference outputs to LRB model_outputs layout
    # =============================================================================
    echo "==== Step 1: Converting inference outputs ===="
    echo "  base-dir:           $INFER_BASE_DIR"
    echo "  model_outputs dir:  $MODEL_OUTPUTS_DIR"
    echo "  identifier:         $IDENTIFIER"
    echo "  iters:              ${ITERS[*]}"

    python convert_to_lrb_format.py \
        --base-dir "$INFER_BASE_DIR" \
        --output-dir "$MODEL_OUTPUTS_DIR" \
        --identifier "$IDENTIFIER" \
        --questions-file "$QUESTIONS_FILE" \
        --iters "${ITERS[@]}"

    # Sanity check that at least one report dir exists
    MODELS=()
    for i in "${ITERS[@]}"; do
        d="$MODEL_OUTPUTS_DIR/${IDENTIFIER}-iter${i}"
        if [ ! -d "$d" ]; then
            echo "Error: expected converted dir not found: $d"
            exit 1
        fi
        MODELS+=("${IDENTIFIER}-iter${i}")
    done

    # =============================================================================
    # Step 2 — Build the reports JSON index via preprocess.py
    # =============================================================================
    echo ""
    echo "==== Step 2: preprocess.py ===="
    mkdir -p "$EXTRACTED_DIR"
    python preprocess.py "$MODEL_OUTPUTS_DIR" \
        -m "${MODELS[@]}" \
        -o "$EXTRACTED_DIR" \
        --use-realtime \
        --questions-file "$QUESTIONS_FILE"

    # preprocess.py writes a timestamped file; copy it to the fixed name
    LATEST_REPORTS="$(ls -t "$EXTRACTED_DIR"/reports_*.json 2>/dev/null | head -n 1 || true)"
    if [ -z "$LATEST_REPORTS" ]; then
        echo "Error: no reports_*.json produced in $EXTRACTED_DIR"
        exit 1
    fi
    cp "$LATEST_REPORTS" "$REPORTS_JSON"
    echo "  reports json (fixed): $REPORTS_JSON"
    echo "  (copied from: $LATEST_REPORTS)"
else
    # Resume mode: reuse existing fixed-name reports JSON
    echo "==== Resuming with existing reports JSON (skip step 1+2) ===="
    echo "  reports json: $REPORTS_JSON"
    echo "  (use --fresh to force re-convert + re-preprocess)"
fi

# Judge models. OpenAI/Azure gpt-5-mini is the default; Gemini can be enabled
# explicitly with RUN_GEMINI=true when that provider is configured.
RUN_GEMINI="${RUN_GEMINI:-false}"
GEMINI_JUDGE="${GEMINI_JUDGE:-gemini-2.5-pro}"
OPENAI_JUDGE="${OPENAI_JUDGE:-gpt-5-mini}"

REPORTS_BASENAME="$(basename "${REPORTS_JSON%.json}")"
OUTPUT_LOG_FILE="$SCRIPT_DIR/eval_output.log"
echo "Logging eval output to: $OUTPUT_LOG_FILE"

GEMINI_RESULTS_DIR=""
if [ "$RUN_GEMINI" = "true" ]; then
    # =============================================================================
    # Step 3a — Grade with Gemini
    # =============================================================================
    echo ""
    echo "==== Step 3a: DeepEval grading (Gemini: $GEMINI_JUDGE) ===="
    echo "  criteria: $CRITERIA"

    echo -e "\n========== Step 3a: Gemini ($GEMINI_JUDGE) ==========" >> "$OUTPUT_LOG_FILE"
    python main.py \
        --input "$REPORTS_JSON" \
        --criteria "$CRITERIA" \
        --provider gemini \
        --model "$GEMINI_JUDGE" \
        --max-concurrent 10 \
        --verbose 2>&1 | tee -a "$OUTPUT_LOG_FILE"

    GEMINI_RESULTS_DIR="$SCRIPT_DIR/results/${REPORTS_BASENAME}_graded_gemini_${GEMINI_JUDGE}"
    echo "Gemini results: $GEMINI_RESULTS_DIR" | tee -a "$OUTPUT_LOG_FILE"
else
    echo ""
    echo "==== Step 3a: Skipping Gemini judge (RUN_GEMINI=$RUN_GEMINI) ===="
fi

# =============================================================================
# Step 3b — Grade with OpenAI
# =============================================================================
echo ""
echo "==== Step 3b: DeepEval grading (OpenAI: $OPENAI_JUDGE) ===="

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY not set." | tee -a "$OUTPUT_LOG_FILE" && exit 1
fi

echo -e "\n========== Step 3b: OpenAI ($OPENAI_JUDGE) ==========" >> "$OUTPUT_LOG_FILE"
python main.py \
    --input "$REPORTS_JSON" \
    --criteria "$CRITERIA" \
    --provider openai \
    --model "$OPENAI_JUDGE" \
    --max-concurrent 10 \
    --verbose 2>&1 | tee -a "$OUTPUT_LOG_FILE"

OPENAI_RESULTS_DIR="$SCRIPT_DIR/results/${REPORTS_BASENAME}_graded_openai_${OPENAI_JUDGE}"
echo "OpenAI results: $OPENAI_RESULTS_DIR" | tee -a "$OUTPUT_LOG_FILE"

# =============================================================================
# Step 4 — Average the two judges
# =============================================================================
echo ""
echo "==== Step 4: Averaging dual-judge results ===="
echo -e "\n========== Step 4: Averaging ==========" >> "$OUTPUT_LOG_FILE"

OPENAI_SUMMARY="$(ls -t "$OPENAI_RESULTS_DIR"/summary_*.json 2>/dev/null | head -n 1 || true)"
if [ "$RUN_GEMINI" = "true" ]; then
    GEMINI_SUMMARY="$(ls -t "$GEMINI_RESULTS_DIR"/summary_*.json 2>/dev/null | head -n 1 || true)"
else
    GEMINI_SUMMARY=""
fi

if [ "$RUN_GEMINI" != "true" ]; then
    echo "Skipping averaging because RUN_GEMINI=$RUN_GEMINI." | tee -a "$OUTPUT_LOG_FILE"
elif [ -z "$GEMINI_SUMMARY" ] || [ -z "$OPENAI_SUMMARY" ]; then
    echo "Warning: could not find both summary files for averaging." | tee -a "$OUTPUT_LOG_FILE"
    echo "  Gemini: ${GEMINI_SUMMARY:-NOT FOUND}" | tee -a "$OUTPUT_LOG_FILE"
    echo "  OpenAI: ${OPENAI_SUMMARY:-NOT FOUND}" | tee -a "$OUTPUT_LOG_FILE"
else
    AVERAGED_DIR="$SCRIPT_DIR/results/${REPORTS_BASENAME}_averaged"
    mkdir -p "$AVERAGED_DIR"

    python average_results.py \
        --input-a "$GEMINI_SUMMARY" \
        --input-b "$OPENAI_SUMMARY" \
        --output "$AVERAGED_DIR/summary_multi_judge.json" 2>&1 | tee -a "$OUTPUT_LOG_FILE"

    echo "Averaged results: $AVERAGED_DIR/summary_multi_judge.json" | tee -a "$OUTPUT_LOG_FILE"
fi

echo "" | tee -a "$OUTPUT_LOG_FILE"
echo "All done." | tee -a "$OUTPUT_LOG_FILE"
echo "Gemini results:  $GEMINI_RESULTS_DIR" | tee -a "$OUTPUT_LOG_FILE"
echo "OpenAI results:  $OPENAI_RESULTS_DIR" | tee -a "$OUTPUT_LOG_FILE"
echo "Averaged:        ${AVERAGED_DIR:-N/A}" | tee -a "$OUTPUT_LOG_FILE"
echo "Log file:        $OUTPUT_LOG_FILE"
