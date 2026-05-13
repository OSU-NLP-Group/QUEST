#!/usr/bin/env bash
# Start the Search HTTP service on a separate machine. Recommended setup: 2x A100, with embeddings on GPU and FAISS on CPU.
# Run directly. By default, it reads config/tools.yaml in this repo, including faiss_similarity_threshold and Serper settings.
# Override config: export SEARCH_SERVICE_CONFIG=/other/path/tools.yaml
# Override port: export SEARCH_SERVICE_PORT=8001

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Path from this script directory to the RL root: scripts -> deepresearch -> recipe -> rl
VERL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
QUEST_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$VERL_ROOT"

DEEPRESEARCH_SECRETS_ENV="${DEEPRESEARCH_SECRETS_ENV:-$QUEST_ROOT/.secrets/deepresearch_api_keys.env}"
if [[ -f "$DEEPRESEARCH_SECRETS_ENV" ]]; then
  set -a
  source "$DEEPRESEARCH_SECRETS_ENV"
  set +a
fi

# By default, use all visible GPUs: cuda:0/1/2 for reads (search_top1), cuda:3 for the writer thread.
# To use only one GPU, export CUDA_VISIBLE_DEVICES=0 before launch.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

# Default: 3 GPUs for reads and 1 GPU for writes. The writer flushes batches every 30s, up to 512 items per batch.
export SEARCH_FAISS_READ_GPUS="${SEARCH_FAISS_READ_GPUS:-0,1,2}"
export SEARCH_FAISS_WRITE_GPUS="${SEARCH_FAISS_WRITE_GPUS:-3}"
export SEARCH_FAISS_WRITE_FLUSH_MS="${SEARCH_FAISS_WRITE_FLUSH_MS:-30000}"
export SEARCH_FAISS_WRITE_BATCH_SIZE="${SEARCH_FAISS_WRITE_BATCH_SIZE:-512}"

# By default, use this repo tools.yaml so threshold/cache_dir match the training side.
CONFIG_PATH="${SEARCH_SERVICE_CONFIG:-$SCRIPT_DIR/../config/tools.yaml}"
PORT="${SEARCH_SERVICE_PORT:-8000}"

exec python -m recipe.deepresearch.tools.search_service \
  --host 0.0.0.0 \
  --port "$PORT" \
  --config "$CONFIG_PATH" \
  "$@"
