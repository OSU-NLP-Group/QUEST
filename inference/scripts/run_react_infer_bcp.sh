#!/bin/bash
# BrowseComp-Plus fair comparison:
# QUEST-35B-SFT+Mid+RL + Qwen3-Embedding-8B (FAISS), k=5, 512-token snippets,
# no visit, mem=80k, out=16k. Credentials are loaded from api_config.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
INFERENCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${INFERENCE_DIR}/.." && pwd)"
QUEST_INFER_DIR="${QUEST_INFER_DIR:-${INFERENCE_DIR}}"
BROWSECOMP_PLUS_DIR="${BROWSECOMP_PLUS_DIR:-${REPO_ROOT}/evaluation/browsecomp_plus}"

load_api_config() {
    local config_file="$1"
    if [ ! -f "$config_file" ]; then
        echo "Error: API config file not found: ${config_file}" >&2
        exit 1
    fi

    eval "$(
        python3 - "$config_file" <<'PY'
import json
import shlex
import sys

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

export API_CONFIG_FILE="${API_CONFIG_FILE:-${INFERENCE_DIR}/api_config.yaml}"
load_api_config "$API_CONFIG_FILE"
echo "Loaded API config from ${API_CONFIG_FILE}"

# --- FAISS search config ---
export FAISS_INDEX_PATH="${FAISS_INDEX_PATH:-${REPO_ROOT}/data/browsecomp_plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl}"
export FAISS_MODEL_NAME="${FAISS_MODEL_NAME:-Qwen/Qwen3-Embedding-8B}"
export FAISS_TOP_K="${FAISS_TOP_K:-5}"
export FAISS_SNIPPET_MAX_TOKENS="${FAISS_SNIPPET_MAX_TOKENS:-512}"
export FAISS_CUDA_DEVICE="${FAISS_CUDA_DEVICE:-1}"

# --- Search-only fair comparison ---
export SERPER_KEY_ID=""
export JINA_API_KEYS=""
export DISABLE_VISIT_TOOL=true

# --- QUEST model server endpoints ---
export SERVER_ENDPOINTS_FILE="${SERVER_ENDPOINTS_FILE:-${INFERENCE_DIR}/server_endpoints.conf}"
export HOSTNAME_LIST="${HOSTNAME_LIST:-c0808,c0809,c0810,c0812,c0817,c0818,c0819,c0820,c0807,c0814}"
export PORTS="${PORTS:-6000,6001,6002,6003}"

# --- Model config ---
export MODEL_NAME="${MODEL_NAME:-deepresearch}"
export MODEL_PATH="${MODEL_PATH:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
export MEMORY_TOKENIZER_PATH="${MEMORY_TOKENIZER_PATH:-${MODEL_PATH}}"

# --- Dataset/output ---
export DATASET="${DATASET:-${BROWSECOMP_PLUS_DIR}/browsecomp_plus_quest_130.jsonl}"
export OUTPUT_PATH="${OUTPUT_PATH:-${INFERENCE_DIR}/outputs/browsecomp_plus/quest35b_rl_qwen3embed8b_mem80k}"
export TASK_LOG_DIR="${TASK_LOG_DIR:-${OUTPUT_PATH}/logs}"

# --- Inference hyperparams ---
export ROLLOUT_COUNT="${ROLLOUT_COUNT:-3}"
export TEMPERATURE="${TEMPERATURE:-1}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.1}"
export MAX_WORKERS="${MAX_WORKERS:-400}"
export MAX_TURN="${MAX_TURN:-400}"
export MAX_LLM_CALL_PER_RUN="${MAX_LLM_CALL_PER_RUN:-${MAX_TURN}}"
export MAX_RUNTIME_MINUTES="${MAX_RUNTIME_MINUTES:-1440}"

# --- Resume/distributed config ---
export RESUME_FROM_MESSAGES="${RESUME_FROM_MESSAGES:-false}"
export RESUME_TERMINATIONS="${RESUME_TERMINATIONS:-No answer found after 1440min,temporal save}"
export RESUME_OVERWRITE_EXISTING="${RESUME_OVERWRITE_EXISTING:-true}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export WORKER_START_BATCH_SIZE="${WORKER_START_BATCH_SIZE:-0}"
export WORKER_START_BATCH_DELAY="${WORKER_START_BATCH_DELAY:-0.0}"
export WORKER_START_STAGGER="${WORKER_START_STAGGER:-0.0}"

# --- Memory config: keep gpt-5-mini ---
export MEMORY_ENABLED=true
export MEMORY_STRATEGY="${MEMORY_STRATEGY:-discard_all}"
export MEMORY_CONTEXT_THRESHOLD="${MEMORY_CONTEXT_THRESHOLD:-120000}"
export MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-120000}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-16000}"
export AZURE_OPENAI_DEPLOYMENT="gpt-5-mini"
export SUMMARY_MODEL_NAME="gpt-5-mini"
export MEMORY_MODEL_NAME="gpt-5-mini"
export MEMORY_OPENAI_API_KEY="${MEMORY_OPENAI_API_KEY:-${MEMORY_API_KEY:-${API_KEY:-}}}"

# --- Cache/tool config ---
export CACHE_DIR="${CACHE_DIR:-${OUTPUT_PATH}/cache}"
export SEARCH_CACHE_ENABLED="${SEARCH_CACHE_ENABLED:-false}"
export VISIT_CACHE_ENABLED="${VISIT_CACHE_ENABLED:-false}"
export ENABLE_PYTHON_TOOL="${ENABLE_PYTHON_TOOL:-false}"
export ENABLE_SCHOLAR_TOOL="${ENABLE_SCHOLAR_TOOL:-false}"

mkdir -p "${OUTPUT_PATH}" "${TASK_LOG_DIR}" "${CACHE_DIR}"

echo "=========================================="
echo " QUEST-35B+RL + Qwen3-Embed-8B | mem=80k"
echo " k=5, 512 tokens, no visit, gpt-5-mini memory/summary"
echo " Dataset: ${DATASET}"
echo " Output: ${OUTPUT_PATH}"
echo " Endpoints conf: ${SERVER_ENDPOINTS_FILE}"
echo " Workers: ${MAX_WORKERS}"
echo "=========================================="

resume_args=()
case "$(echo "$RESUME_FROM_MESSAGES" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
        resume_args+=(--resume_from_messages --resume_terminations "$RESUME_TERMINATIONS")
        case "$(echo "$RESUME_OVERWRITE_EXISTING" | tr '[:upper:]' '[:lower:]')" in
            1|true|yes|on)
                resume_args+=(--resume_overwrite_existing)
                ;;
        esac
        ;;
esac

cd "${QUEST_INFER_DIR}"
python -u run_multi_react.py \
    --dataset "${DATASET}" \
    --output "${OUTPUT_PATH}" \
    --max_workers "${MAX_WORKERS}" \
    --model "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --temperature "${TEMPERATURE}" \
    --presence_penalty "${PRESENCE_PENALTY}" \
    --roll_out_count "${ROLLOUT_COUNT}" \
    --total_splits "${WORLD_SIZE}" \
    --worker_split "$((RANK + 1))" \
    --worker_start_batch_size "${WORKER_START_BATCH_SIZE}" \
    --worker_start_batch_delay "${WORKER_START_BATCH_DELAY}" \
    --worker_start_stagger "${WORKER_START_STAGGER}" \
    "${resume_args[@]}"
