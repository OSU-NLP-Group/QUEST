#!/usr/bin/env bash
# 在单独机器上启动 Search HTTP 服务（建议 2×A100：embedding 用 GPU，FAISS 用 CPU）
# 直接运行即可，默认会读同 repo 的 config/tools.yaml（含 faiss_similarity_threshold、serper 等）
# 覆盖配置：export SEARCH_SERVICE_CONFIG=/other/path/tools.yaml
# 覆盖端口：export SEARCH_SERVICE_PORT=8001

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 从 script 所在目录到 verl：scripts -> deepresearch -> recipe -> verl
VERL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
QUEST_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$VERL_ROOT"

DEEPRESEARCH_SECRETS_ENV="${DEEPRESEARCH_SECRETS_ENV:-$QUEST_ROOT/.secrets/deepresearch_api_keys.env}"
if [[ -f "$DEEPRESEARCH_SECRETS_ENV" ]]; then
  set -a
  source "$DEEPRESEARCH_SECRETS_ENV"
  set +a
fi

# 默认使用全部可见 GPU：cuda:0/1/2 给读（search_top1），cuda:3 给写线程
# 若只想用 1 张卡：启动前 export CUDA_VISIBLE_DEVICES=0
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

# 默认 3 卡读 + 1 卡写；写线程按 30s 窗口批量落盘，批大小上限 512
export SEARCH_FAISS_READ_GPUS="${SEARCH_FAISS_READ_GPUS:-0,1,2}"
export SEARCH_FAISS_WRITE_GPUS="${SEARCH_FAISS_WRITE_GPUS:-3}"
export SEARCH_FAISS_WRITE_FLUSH_MS="${SEARCH_FAISS_WRITE_FLUSH_MS:-30000}"
export SEARCH_FAISS_WRITE_BATCH_SIZE="${SEARCH_FAISS_WRITE_BATCH_SIZE:-512}"

# 默认使用本 repo 的 tools.yaml，保证 threshold/cache_dir 等与训练侧一致
CONFIG_PATH="${SEARCH_SERVICE_CONFIG:-$SCRIPT_DIR/../config/tools.yaml}"
PORT="${SEARCH_SERVICE_PORT:-8000}"

exec python -m recipe.deepresearch.tools.search_service \
  --host 0.0.0.0 \
  --port "$PORT" \
  --config "$CONFIG_PATH" \
  "$@"
