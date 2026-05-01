#!/bin/bash

# =============================================================================
# Evaluation Run Script
# =============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# =============================================================================
# Configuration
# =============================================================================

# Evaluation model (model name exposed by the judge service)
export MODEL_NAME="${MODEL_NAME:-eval_model}"

# Judge server endpoints (hot-swap: modifying endpoints.conf takes effect at runtime)
export SERVER_ENDPOINTS_FILE="${SERVER_ENDPOINTS_FILE:-${SCRIPT_DIR}/endpoints.conf}"

if [ ! -f "$SERVER_ENDPOINTS_FILE" ]; then
    echo "ERROR: endpoint config not found: ${SERVER_ENDPOINTS_FILE}"
    exit 1
fi

export HOSTNAME_LIST=$(grep '^HOSTNAME_LIST=' "$SERVER_ENDPOINTS_FILE" | cut -d'=' -f2-)
export PORTS=$(grep '^PORTS=' "$SERVER_ENDPOINTS_FILE" | cut -d'=' -f2-)

# Data file paths
PROMPT_TO_EVAL="${PROMPT_TO_EVAL:-/path/to/polished_criteria.jsonl}"
ANSWER_TO_EVAL="${ANSWER_TO_EVAL:-/path/to/final_answers.jsonl}"
REF_TO_EVAL="${REF_TO_EVAL:-/path/to/reference_answers.jsonl}"

# Document-level concurrency (how many documents to evaluate simultaneously)
MAX_WORKERS="${MAX_WORKERS:-300}"

# Extract iter info from answer filename (e.g., iter1) and write to output path
_answer_basename="$(basename "$ANSWER_TO_EVAL" .jsonl)"
_iter_tag="$(echo "$_answer_basename" | grep -oE 'iter[0-9]+')"
_iter_tag="${_iter_tag:-iter_unknown}"

OUTPUT_FILE="${OUTPUT_FILE:-${SCRIPT_DIR}/results/all_iters_eval.jsonl}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

# =============================================================================
# 1. Check if all port services are running properly
# =============================================================================

# Parse host/port strings into arrays
OLD_IFS=$IFS
IFS=',' read -ra HOSTS <<< "$HOSTNAME_LIST"
IFS=',' read -ra MAIN_PORTS <<< "$PORTS"
IFS=$OLD_IFS

echo "=============================================="
echo "Checking vLLM server endpoints..."
echo "Hosts : ${HOSTNAME_LIST}"
echo "Ports : ${PORTS}"
echo "=============================================="

all_servers_running=true
for host in "${HOSTS[@]}"; do
    host=$(echo "$host" | xargs)
    for port in "${MAIN_PORTS[@]}"; do
        port=$(echo "$port" | xargs)
        if curl -s -f --max-time 5 "http://${host}:${port}/v1/models" > /dev/null 2>&1; then
            echo "  [OK]   ${host}:${port}"
        else
            echo "  [FAIL] ${host}:${port}"
            all_servers_running=false
        fi
    done
done

if [ "$all_servers_running" != "true" ]; then
    echo ""
    echo "ERROR: Some servers are not reachable. Please start them before running eval."
    exit 1
fi

echo ""
echo "All servers are reachable. Proceeding..."

# =============================================================================
# 2. Wait for all ports to be ready (prevent case where just started and not fully ready)
# =============================================================================

timeout=600
start_time=$(date +%s)

declare -A server_ready
for host in "${HOSTS[@]}"; do
    host=$(echo "$host" | xargs)
    for port in "${MAIN_PORTS[@]}"; do
        port=$(echo "$port" | xargs)
        server_ready["${host}:${port}"]=false
    done
done

echo "Waiting for all servers to be fully ready..."
while true; do
    all_ready=true

    for host in "${HOSTS[@]}"; do
        host=$(echo "$host" | xargs)
        for port in "${MAIN_PORTS[@]}"; do
            port=$(echo "$port" | xargs)
            key="${host}:${port}"
            if [ "${server_ready[$key]}" = "false" ]; then
                if curl -s -f --max-time 5 "http://${host}:${port}/v1/models" > /dev/null 2>&1; then
                    echo "  [Ready] ${key}"
                    server_ready[$key]=true
                else
                    all_ready=false
                fi
            fi
        done
    done

    [ "$all_ready" = "true" ] && break

    elapsed=$(( $(date +%s) - start_time ))
    if [ $elapsed -gt $timeout ]; then
        echo "ERROR: Timeout after ${timeout}s waiting for servers."
        for key in "${!server_ready[@]}"; do
            [ "${server_ready[$key]}" = "false" ] && echo "  [TIMEOUT] ${key}"
        done
        exit 1
    fi

    printf "  Still waiting... (%ds elapsed)\n" "$elapsed"
    sleep 10
done

echo ""
echo "All servers ready!"

# =============================================================================
# 3. Start evaluation
# =============================================================================

echo ""
echo "=============================================="
echo "Starting open-ended evaluation"
echo "  iter tag    : ${_iter_tag}"
echo "  model       : ${MODEL_NAME}"
echo "  answer file : ${ANSWER_TO_EVAL}"
echo "  output file : ${OUTPUT_FILE}"
echo "  endpoints   : ${SERVER_ENDPOINTS_FILE}"
echo "=============================================="
echo ""

cd "$SCRIPT_DIR"

python -u evaluate_criteria_args_parallel_open_ended.py \
    --model         "$MODEL_NAME" \
    --prompt_to_eval "$PROMPT_TO_EVAL" \
    --answer_to_eval "$ANSWER_TO_EVAL" \
    --ref_to_eval   "$REF_TO_EVAL" \
    --output_file   "$OUTPUT_FILE" \
    --hostname_list "$HOSTNAME_LIST" \
    --ports         "$PORTS" \
    --endpoints_file "$SERVER_ENDPOINTS_FILE" \
    --max_workers   "$MAX_WORKERS"
