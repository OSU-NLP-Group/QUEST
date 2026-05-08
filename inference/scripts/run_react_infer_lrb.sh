#!/bin/bash

SCRIPT_DIR="$(cd "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd)"
INFERENCE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${INFERENCE_DIR}/.." && pwd)"

load_api_config() {
    local config_file="$1"

    if [ ! -f "$config_file" ]; then
        echo "Error: API config file not found: ${config_file}"
        exit 1
    fi

    eval "$(
        python3 - "$config_file" <<'PY'
import json
import shlex
import sys

config_file = sys.argv[1]
with open(config_file, "r", encoding="utf-8") as f:
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

# =============================================================================
# Environment Variables Configuration
# =============================================================================

# TORCH/NCCL Configuration (for multi-GPU setups)
export TORCHDYNAMO_VERBOSE=1
export TORCHDYNAMO_DISABLE=1
export NCCL_IB_TC=16
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth
export NCCL_DEBUG=INFO
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_MIN_NCHANNELS=4
export NCCL_NET_PLUGIN=none
# export GLOO_SOCKET_IFNAME=eth0

# DeepResearch Configuration
export NLP_WEB_SEARCH_ONLY_CACHE=false
export NLP_WEB_SEARCH_ENABLE_READPAGE=false
export NLP_WEB_SEARCH_ENABLE_SFILTER=false
export QWEN_SEARCH_ENABLE_CSI=false
export SPECIAL_CODE_MODE=false
export PYTHONDONTWRITEBYTECODE=1
export CACHE_DIR="${CACHE_DIR:-${INFERENCE_DIR}/cache/${USER:-default}}"

# Model and Inference Hyperparameters
export MODEL_NAME="${MODEL_NAME:-deepresearch}"
export MODEL_PATH="${MODEL_PATH:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}"
export DATASET="${DATASET:-${REPO_ROOT}/evaluation/liveresearchbench/liveresearchbench_questions.jsonl}"
export OUTPUT_PATH="${OUTPUT_PATH:-${INFERENCE_DIR}/outputs/lrb/results}"
export TASK_LOG_DIR="${TASK_LOG_DIR:-${INFERENCE_DIR}/outputs/lrb/memory_logs}"

# Auto-prepare the LiveResearchBench question jsonl from HuggingFace if missing.
# LRB is a gated dataset — needs HF_TOKEN (will be loaded from api_config.yaml below
# if defined there, otherwise must be set in the environment beforehand).
if [ ! -f "$DATASET" ]; then
    export API_CONFIG_FILE="${API_CONFIG_FILE:-${INFERENCE_DIR}/api_config.yaml}"
    load_api_config "$API_CONFIG_FILE"
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "Error: HF_TOKEN not set; cannot download gated LiveResearchBench dataset."
        echo "       Add HF_TOKEN to api_config.yaml (under common:) or export it."
        exit 1
    fi
    echo "LRB question file not found, fetching from HuggingFace -> $DATASET"
    python3 "${REPO_ROOT}/evaluation/liveresearchbench/prepare_lrb_questions.py" \
        --output "$DATASET" --use-realtime
fi

export ROLLOUT_COUNT=3
export TEMPERATURE=1
export PRESENCE_PENALTY=1.1
export MAX_WORKERS="${MAX_WORKERS:-70}"
export MAX_TURN="${MAX_TURN:-400}"
export MAX_LLM_CALL_PER_RUN=$MAX_TURN

# Use the LRB-tuned system prompt (adds inline-citation requirements).
# Set to "SYSTEM_PROMPT" to use the default prompt instead.
export SYSTEM_PROMPT_NAME="${SYSTEM_PROMPT_NAME:-SYSTEM_PROMPT_FOR_LRB}"

# API and external service configuration
export API_CONFIG_FILE="${API_CONFIG_FILE:-${INFERENCE_DIR}/api_config.yaml}"
load_api_config "$API_CONFIG_FILE"
echo "Loaded API config from ${API_CONFIG_FILE}"

export MEMORY_THRESHOLD="${MEMORY_THRESHOLD:-80000}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-10000}"
export SANDBOX_FUSION_ENDPOINT="${SANDBOX_FUSION_ENDPOINT:-your_sandbox_endpoint}"

# Multi-Worker Configuration (Optional)
# These are typically set by distributed training frameworks
# export WORLD_SIZE=1
# export RANK=0

# Server endpoint configuration (supports runtime edits via SERVER_ENDPOINTS_FILE)
# react_agent.py will hot-reload SERVER_ENDPOINTS_FILE on every service call.
export SERVER_ENDPOINTS_FILE="${SERVER_ENDPOINTS_FILE:-${INFERENCE_DIR}/server_endpoints.conf}"

normalize_list_value() {
    local value="$1"
    value="$(echo "$value" | sed -E 's/^[[:space:]]+|[[:space:]]+$//g')"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
        value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
        value="${value:1:${#value}-2}"
    fi
    echo "$value"
}

load_server_endpoints_from_file() {
    local config_file="$1"

    if [ ! -f "$config_file" ]; then
        return 0
    fi

    while IFS= read -r raw_line || [ -n "$raw_line" ]; do
        local line="$raw_line"
        line="${line#"${line%%[![:space:]]*}"}"

        if [ -z "$line" ] || [[ "$line" == \#* ]] || [[ "$line" != *=* ]]; then
            continue
        fi

        local key="${line%%=*}"
        local value="${line#*=}"

        key="$(echo "$key" | sed -E 's/[[:space:]]+//g')"
        value="$(normalize_list_value "$value")"

        if [ "$key" = "HOSTNAME_LIST" ] && [ -n "$value" ]; then
            HOSTNAME_LIST="$value"
        elif [ "$key" = "PORTS" ] && [ -n "$value" ]; then
            PORTS="$value"
        fi
    done < "$config_file"
}

# Hostname List Configuration (multi-node access settings)
# Default values; can be overridden by server_endpoints.conf or external environment variables
# Example: export HOSTNAME_LIST="node1,node2,node3"
export HOSTNAME_LIST="${HOSTNAME_LIST:-localhost}"

# Main model port list, comma-separated
export PORTS="${PORTS:-6000,6001,6002,6003}"

# Load from the config file (file values override the defaults above and support hot-swapping)
load_server_endpoints_from_file "$SERVER_ENDPOINTS_FILE"
if [ -f "$SERVER_ENDPOINTS_FILE" ]; then
    echo "Endpoint config file: ${SERVER_ENDPOINTS_FILE}"
else
    echo "Warning: endpoint config file not found, using env vars only: ${SERVER_ENDPOINTS_FILE}"
fi
######################################
### 1. start server           ###
######################################

# Parse the host list
if [ -z "$HOSTNAME_LIST" ]; then
    HOSTNAME_LIST="localhost"
fi

# Convert the host list to an array
OLD_IFS=$IFS
IFS=',' read -ra HOSTS <<< "$HOSTNAME_LIST"
# Convert the port list to an array
IFS=',' read -ra RAW_PORTS <<< "$PORTS"
IFS=$OLD_IFS
main_ports=()
for port in "${RAW_PORTS[@]}"; do
    port=$(echo "$port" | xargs)  # Trim whitespace
    if [ -n "$port" ]; then
        main_ports+=("$port")
    fi
done

echo "Checking if VLLM servers are already running..."
echo "Hosts to check: ${HOSTNAME_LIST}"
echo "Ports to check: ${main_ports[*]}"

all_servers_running=true
for host in "${HOSTS[@]}"; do
    host=$(echo "$host" | xargs)  # Trim whitespace
    for port in "${main_ports[@]}"; do
        if ! curl -s -f --max-time 5 http://$host:$port/v1/models > /dev/null 2>&1; then
            echo "Server not running: $host:$port"
            all_servers_running=false
        else
            echo "Server running: $host:$port"
        fi
    done
done

if [ "$all_servers_running" = "true" ]; then
    echo "Detected running VLLM servers on all hosts and ports, skip starting new servers."
else
    echo "Missing VLLM servers, stop now."
    exit 1
fi
# CUDA_VISIBLE_DEVICES=2 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6003 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=3 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6004 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=4 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6005 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=5 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6006 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=6 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6007 --disable-log-requests &
# CUDA_VISIBLE_DEVICES=7 vllm serve $MODEL_PATH --host 0.0.0.0 --port 6008 --disable-log-requests &

#######################################################
### 2. Waiting for the server port to be ready  ###
######################################################

timeout=6000
start_time=$(date +%s)

echo "Mode: All ports used as main model"
echo "Waiting for servers to start on hosts: ${HOSTNAME_LIST}"

# Initialize the server status map (using host:port as the key)
declare -A server_status
for host in "${HOSTS[@]}"; do
    host=$(echo "$host" | xargs)  # Trim whitespace
    for port in "${main_ports[@]}"; do
        server_status["$host:$port"]=false
    done
done

while true; do
    all_ready=true

    for host in "${HOSTS[@]}"; do
        host=$(echo "$host" | xargs)  # Trim whitespace
        for port in "${main_ports[@]}"; do
            if [ "${server_status[$host:$port]}" = "false" ]; then
                if curl -s -f --max-time 5 http://$host:$port/v1/models > /dev/null 2>&1; then
                    echo "Main model server ($host:$port) is ready!"
                    server_status["$host:$port"]=true
                else
                    all_ready=false
                fi
            fi
        done
    done

    if [ "$all_ready" = "true" ]; then
        echo "All servers are ready for inference!"
        break
    fi

    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout ]; then
        echo -e "\nError: Server startup timeout after ${timeout} seconds"

        for host in "${HOSTS[@]}"; do
            host=$(echo "$host" | xargs)  # Trim whitespace
            for port in "${main_ports[@]}"; do
                if [ "${server_status[$host:$port]}" = "false" ]; then
                    echo "Main model server ($host:$port) failed to start"
                fi
            done
        done

        exit 1
    fi

    printf 'Waiting for servers to start .....'
    sleep 10
done

failed_servers=()
for host in "${HOSTS[@]}"; do
    host=$(echo "$host" | xargs)  # Trim whitespace
    for port in "${main_ports[@]}"; do
        if [ "${server_status[$host:$port]}" = "false" ]; then
            failed_servers+=("$host:$port")
        fi
    done
done

if [ ${#failed_servers[@]} -gt 0 ]; then
    echo "Error: The following servers failed to start: ${failed_servers[*]}"
    exit 1
else
    echo "All required servers are running successfully!"
fi

#####################################
### 3. start infer               ####
#####################################

echo "==== start infer... ===="

# ========== Cache Configuration (Cache configuration) ==========
# Visit Cache and Search Cache use SQLite databases to reduce duplicate requests
#
# Cache environment variables(can be configured via a .env file or set here):
#
# Visit Cache (webpage access cache):
#   VISIT_CACHE_ENABLED: whether to enable caching(default: "true")
#   VISIT_CACHE_FILE: cache database file path(default: "visit_cache.db")
#   VISIT_CACHE_RESUME: whether to resume from an existing cache(default: "true")
#
# Search Cache (search query cache):
#   SEARCH_CACHE_ENABLED: whether to enable caching(default: "true")
#   SEARCH_CACHE_FILE: cache database file path(default: "search_cache.db")
#   SEARCH_CACHE_RESUME: whether to resume from an existing cache(default: "true")
#
# Note:if these environment variables are not set, caching is enabled by default and uses the default paths
# Cache files will be created in the current working directory(usually the script run directory)

cd "${INFERENCE_DIR}"

python -u run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --model $MODEL_NAME --model_path $MODEL_PATH --temperature $TEMPERATURE --presence_penalty $PRESENCE_PENALTY --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT
