#!/usr/bin/env bash
# 一键初始化 FAISS 搜索库：合并 shard → 建 FAISS 索引
# 从 config/tools.yaml 读路径；建索引用 GPU 加速（可选）
# 运行前请安装: pip install faiss-cpu sentence-transformers pyyaml

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$VERL_ROOT"

# 可选：指定用哪张卡做 embedding（建索引时建议用 GPU）
# export CUDA_VISIBLE_DEVICES=0

# 默认从 recipe/deepresearch/config/tools.yaml 读 cache_dir / cache_file / shards / faiss_embedding_model
# 建索引用 GPU（有则自动选 cuda）
python -m recipe.deepresearch.scripts.build_search_faiss --device cuda "$@"

# 若只建 FAISS、不合并（例如已合并过）:
# python -m recipe.deepresearch.scripts.build_search_faiss --skip-merge --device cuda
