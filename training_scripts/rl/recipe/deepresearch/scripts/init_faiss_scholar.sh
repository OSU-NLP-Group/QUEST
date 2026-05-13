#!/usr/bin/env bash
# Initialize the FAISS scholar store: merge shards -> build FAISS index.
# Read paths from config/tools.yaml; optionally use GPU to accelerate index building.
# Before running, install: pip install faiss-cpu sentence-transformers pyyaml

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$VERL_ROOT"

# Optional: choose which GPU to use for embeddings. GPU is recommended for index building.
# export CUDA_VISIBLE_DEVICES=0

# By default, read cache_dir / cache_file / shards / faiss_embedding_model from recipe/deepresearch/config/tools.yaml.
# Use GPU for index building, automatically selecting cuda when available.
python -m recipe.deepresearch.scripts.build_scholar_faiss --device cuda "$@"

# To build only FAISS without merging, for example after a previous merge:
# python -m recipe.deepresearch.scripts.build_scholar_faiss --skip-merge --device cuda
