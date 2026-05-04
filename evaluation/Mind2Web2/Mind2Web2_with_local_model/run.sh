#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default multi-node hosts (fallback when NODE_ENDPOINTS_FILE is absent).
export HOSTNAME_LIST="a0012, a0013, a0016, a0020, a0007, a0008"
export VLLM_PORTS="${VLLM_PORTS:-6000,6001,6002,6003}"
# Hot-reload endpoint file (recommended one node per line, e.g. "a0012";
# host-only entries auto-expand with VLLM_PORTS).
export NODE_ENDPOINTS_FILE="${NODE_ENDPOINTS_FILE:-${SCRIPT_DIR}/config/vllm_nodes.txt}"
export LOCAL_OPENAI_ENDPOINTS_FILE="${LOCAL_OPENAI_ENDPOINTS_FILE:-${NODE_ENDPOINTS_FILE}}"
export LOCAL_OPENAI_ENDPOINTS_RELOAD_SECONDS="${LOCAL_OPENAI_ENDPOINTS_RELOAD_SECONDS:-15}"

export GOOGLE_MAPS_API_KEY="${GOOGLE_MAPS_API_KEY:-your_google_maps_api_key}"
export JINA_API_KEYS="${JINA_API_KEYS:-your_jina_api_key}"



export MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/model/Qwen3-4B-Instruct-2507}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-eval_model}"
# vLLM bind vs advertise:
# - VLLM_BIND_HOST: what vLLM listens on (use 0.0.0.0 to expose to other nodes)
# - VLLM_ADVERTISE_HOST: what clients should use to reach this node (e.g. node IP/hostname)
VLLM_BIND_HOST="${VLLM_BIND_HOST:-127.0.0.1}"
VLLM_ADVERTISE_HOST="${VLLM_ADVERTISE_HOST:-${VLLM_BIND_HOST}}"
# Launch one vLLM server per GPU (replicated model).
VLLM_GPUS="${VLLM_GPUS:-0,1,2,3}"
IFS=',' read -r -a GPUS <<<"${VLLM_GPUS}"
IFS=',' read -r -a PORTS <<<"${VLLM_PORTS}"

ENDPOINTS=()
HOSTS=()
_append_host_unique() {
  local host="$1"
  [ -z "${host}" ] && return
  for h in "${HOSTS[@]:-}"; do
    if [ "${h}" = "${host}" ]; then
      return
    fi
  done
  HOSTS+=("${host}")
}

_append_endpoint() {
  local host="$1"
  local port="$2"
  local endpoint="http://${host}:${port}/v1"
  ENDPOINTS+=("${endpoint}")
  _append_host_unique "${host}"
}

_parse_endpoint_token() {
  local token="$1"
  token="$(echo "${token}" | xargs)"
  [ -z "${token}" ] && return

  if [[ "${token}" =~ ^https?:// ]]; then
    local endpoint="${token%/}"
    if [[ ! "${endpoint}" =~ /v1$ ]]; then
      endpoint="${endpoint}/v1"
    fi
    ENDPOINTS+=("${endpoint}")
    local host_only
    host_only="$(echo "${endpoint}" | sed -E 's#^https?://([^/:]+).*$#\1#')"
    _append_host_unique "${host_only}"
    return
  fi

  # host:port
  if [[ "${token}" == *:* ]]; then
    local host="${token%%:*}"
    local port="${token##*:}"
    host="$(echo "${host}" | xargs)"
    port="$(echo "${port}" | xargs)"
    if [ -n "${host}" ] && [[ "${port}" =~ ^[0-9]+$ ]]; then
      _append_endpoint "${host}" "${port}"
    fi
    return
  fi

  # host only -> combine with VLLM_PORTS
  local host="${token}"
  if [ -n "${host}" ]; then
    for port in "${PORTS[@]}"; do
      _append_endpoint "${host}" "${port}"
    done
  fi
}

if [ -f "${NODE_ENDPOINTS_FILE}" ]; then
  while IFS= read -r raw_line || [ -n "${raw_line}" ]; do
    line="${raw_line%%#*}"
    line="$(echo "${line}" | xargs)"
    [ -z "${line}" ] && continue
    OLD_IFS=$IFS
    IFS=',' read -r -a TOKENS <<<"${line}"
    IFS=$OLD_IFS
    for token in "${TOKENS[@]}"; do
      _parse_endpoint_token "${token}"
    done
  done <"${NODE_ENDPOINTS_FILE}"
fi

if [ "${#ENDPOINTS[@]}" -gt 0 ]; then
  AUTO_START_VLLM="${AUTO_START_VLLM:-0}"
elif [ -n "${HOSTNAME_LIST+x}" ] && [ -n "${HOSTNAME_LIST}" ]; then
  RAW_HOSTNAME_LIST="${HOSTNAME_LIST}"
  AUTO_START_VLLM="${AUTO_START_VLLM:-0}"
  OLD_IFS=$IFS
  IFS=',' read -r -a HOSTS_RAW <<<"${RAW_HOSTNAME_LIST}"
  IFS=$OLD_IFS
  for host in "${HOSTS_RAW[@]}"; do
    host="$(echo "${host}" | xargs)"
    [ -z "${host}" ] && continue
    _append_host_unique "${host}"
    for port in "${PORTS[@]}"; do
      _append_endpoint "${host}" "${port}"
    done
  done
else
  AUTO_START_VLLM="${AUTO_START_VLLM:-1}"
  _append_host_unique "${VLLM_ADVERTISE_HOST}"
  for port in "${PORTS[@]}"; do
    _append_endpoint "${VLLM_ADVERTISE_HOST}" "${port}"
  done
fi

if [ "${#ENDPOINTS[@]}" -eq 0 ]; then
  echo "Error: no valid endpoints found. Check NODE_ENDPOINTS_FILE='${NODE_ENDPOINTS_FILE}' or HOSTNAME_LIST."
  exit 1
fi


export LOCAL_OPENAI_API_KEY="dummy"
# Build endpoint pool and export to client.
LOCAL_OPENAI_BASE_URLS="$(IFS=','; echo "${ENDPOINTS[*]}")"
if [ -n "${EXTRA_LOCAL_OPENAI_BASE_URLS:-}" ]; then
  if [ -z "${LOCAL_OPENAI_BASE_URLS}" ]; then
    LOCAL_OPENAI_BASE_URLS="${EXTRA_LOCAL_OPENAI_BASE_URLS}"
  else
    LOCAL_OPENAI_BASE_URLS="${LOCAL_OPENAI_BASE_URLS},${EXTRA_LOCAL_OPENAI_BASE_URLS}"
  fi
fi
export LOCAL_OPENAI_BASE_URLS

# Back-compat: a single base url (some scripts expect this).
export LOCAL_OPENAI_BASE_URL="${ENDPOINTS[0]}"
# Ensure localhost requests don't go through a proxy (common cause of connection errors).
NO_PROXY_HOSTS="127.0.0.1,localhost"
for host in "${HOSTS[@]}"; do
  if [ -n "${host}" ]; then
    NO_PROXY_HOSTS="${NO_PROXY_HOSTS},${host}"
  fi
done
export NO_PROXY="${NO_PROXY:-},${NO_PROXY_HOSTS}"

# ----------------------------
# Logging
# ----------------------------
LOG_DIR="${LOG_DIR:-./log}"
mkdir -p "${LOG_DIR}"
# Use nanoseconds + PID to avoid collisions across rapid restarts.
TS="$(date +%Y%m%d_%H%M%S_%N)"
RUN_ID="${TS}_pid$$"
RUN_DIR="${LOG_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
RUN_LOG="${RUN_DIR}/run.log"

# Redirect all script output to both terminal and log file.
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Logging to: ${RUN_LOG}"

MODELS_URL="${LOCAL_OPENAI_BASE_URL%/}/models"

_link_vllm_log() {
  # Best effort: link the vLLM server log into the per-run folder.
  local meta_file="$1"
  local out_file="$2"

  local server_log_path=""
  if [ -f "${meta_file}" ]; then
    server_log_path="$(sed -n 's/^log=//p' "${meta_file}" | head -n 1)"
  fi
  if [ -n "${server_log_path}" ] && [ -f "${server_log_path}" ]; then
    if ln -sf "${server_log_path}" "${out_file}" 2>/dev/null; then
      echo "vLLM log (linked): ${out_file} -> ${server_log_path}"
      return 0
    fi
    echo "vLLM log: ${server_log_path}" >"${out_file}"
    echo "vLLM log (path saved): ${out_file}"
    return 0
  fi
  echo "vLLM log: (unknown; no ${meta_file} yet)" >"${out_file}"
}

echo "Checking/starting vLLM servers: ${LOCAL_OPENAI_BASE_URLS}"
if [ "${AUTO_START_VLLM}" -eq 1 ]; then
  if [ "${#GPUS[@]}" -ne "${#PORTS[@]}" ]; then
    echo "Error: VLLM_GPUS count (${#GPUS[@]}) must equal VLLM_PORTS count (${#PORTS[@]}) when AUTO_START_VLLM=1"
    exit 1
  fi

  for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    gpu="${GPUS[$i]}"
    base_url="http://${VLLM_ADVERTISE_HOST}:${port}/v1"
    models_url="${base_url%/}/models"

    VLLM_STATE_DIR="${LOG_DIR}/vllm_server_${port}"
    VLLM_PID_FILE="${VLLM_STATE_DIR}/vllm.pid"
    VLLM_META_FILE="${VLLM_STATE_DIR}/vllm.meta"
    mkdir -p "${VLLM_STATE_DIR}"
    VLLM_LOG_PORT="${RUN_DIR}/vllm_${port}.log"

    echo "Checking local vLLM at ${models_url} ..."
    if curl -sSf "${models_url}" >/dev/null 2>&1; then
      echo "Detected running local vLLM server on port ${port}; skipping launch."
      _link_vllm_log "${VLLM_META_FILE}" "${VLLM_LOG_PORT}"
      continue
    fi

    VLLM_PID=""
    if [ -f "${VLLM_PID_FILE}" ]; then
      VLLM_PID="$(cat "${VLLM_PID_FILE}" 2>/dev/null || true)"
    fi

    if [ -n "${VLLM_PID}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "Detected existing local vLLM process (pid=${VLLM_PID}) for port ${port} but it isn't ready yet; will wait."
      _link_vllm_log "${VLLM_META_FILE}" "${VLLM_LOG_PORT}"
      continue
    fi

    echo "Starting local vLLM (daemonized) at ${base_url} on GPU ${gpu} ..."
    SERVER_TS="$(date +%Y%m%d_%H%M%S_%N)"
    SERVER_LOG="${VLLM_STATE_DIR}/vllm_${SERVER_TS}.log"
    printf "base_url=%s\nlog=%s\n" "${base_url}" "${SERVER_LOG}" >"${VLLM_META_FILE}"

    nohup env CUDA_VISIBLE_DEVICES="${gpu}" vllm serve "${MODEL_DIR}" \
      --host "${VLLM_BIND_HOST}" \
      --port "${port}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --tensor-parallel-size 1 \
      --trust-remote-code \
      >"${SERVER_LOG}" 2>&1 </dev/null &
    VLLM_PID=$!
    echo "${VLLM_PID}" >"${VLLM_PID_FILE}"

    echo "vLLM PID (port ${port}): ${VLLM_PID}"
    echo "vLLM server log (port ${port}): ${SERVER_LOG}"
    _link_vllm_log "${VLLM_META_FILE}" "${VLLM_LOG_PORT}"
  done
else
  echo "AUTO_START_VLLM=0, skip starting local vLLM. Will only wait/check configured hosts."
fi

echo "Waiting for vLLM servers to be ready ..."
VLLM_READY_TIMEOUT_SECONDS="${VLLM_READY_TIMEOUT_SECONDS:-6000}"
for endpoint in "${ENDPOINTS[@]}"; do
  models_url="${endpoint%/}/models"
  echo "Waiting: ${models_url}"
  for ((i = 0; i < VLLM_READY_TIMEOUT_SECONDS; i++)); do
    if curl -sSf "${models_url}" >/dev/null 2>&1; then
      echo "vLLM ready: ${models_url}"
      break
    fi
    sleep 1
  done

  if ! curl -sSf "${models_url}" >/dev/null 2>&1; then
    echo "vLLM not ready after timeout: ${models_url}"
    echo "Configured endpoint pool: ${LOCAL_OPENAI_BASE_URLS}"
    exit 1
  fi
done

# Run evaluation (set RUN_EVAL=0 on worker-only nodes).
RUN_EVAL="${RUN_EVAL:-1}"
if [ "${RUN_EVAL}" -eq 1 ]; then
  python run_eval.py --agent_name Last_sythetic_data_all_verification_data_with_summaries_20260325  --eval_version 2025_10_23  --eval_results_root ./eval_results/Last_sythetic_data_all_verification_data_with_summaries_20260325 --llm_provider local_openai --judge_model "${SERVED_MODEL_NAME}"
fi
