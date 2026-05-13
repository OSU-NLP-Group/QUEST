#!/usr/bin/env python3
"""
Merge search SQLite shards and build the FAISS index in one pass.
After that, online serving only needs append writes.

Usage:
  # Read paths from config/tools.yaml (recommended)
  python -m recipe.deepresearch.scripts.build_search_faiss

  # Or pass explicit arguments
  python -m recipe.deepresearch.scripts.build_search_faiss \\
    --cache-dir /path/to/database \\
    --cache-file /path/to/database/search.db \\
    --shards 16 \\
    --embedding-model /path/to/Qwen-3-8B-Embedding \\
    --device cuda

Efficiency notes:
  - FAISS retrieval: IndexFlatIP over 200k entries on CPU is about 5-20 ms/query, so GPU is usually unnecessary.
  - Embedding: index building must vectorize about 200k queries, which is the main cost.
    - With Qwen-3-8B, GPU can process roughly thousands of entries per minute; CPU is over an order of magnitude slower, so GPU is recommended for index building.
  - Conclusion: use GPU to reduce index build time; CPU is sufficient for online retrieval, and FAISS plus a small model can also run fully on CPU.
"""

import argparse
import os
import sys
import time

# Ensure recipe.deepresearch can be imported when running from the RL root or repo root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# .../verl/recipe/deepresearch/scripts -> the path must include the RL root
RECIPE_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
if RECIPE_PARENT not in sys.path:
    sys.path.insert(0, RECIPE_PARENT)

from recipe.deepresearch.tools.search_tool import (
    merge_shards_into_master,
    build_faiss_from_search_cache,
    incremental_update_faiss,
)


def load_config_paths():
    """Read search cache_dir, cache_file, shards, and faiss_embedding_model from config/tools.yaml."""
    import os
    config_path = os.path.join(
        os.path.dirname(SCRIPT_DIR), "config", "tools.yaml"
    )
    if not os.path.exists(config_path):
        return None
    try:
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        for t in (data or {}).get("tools") or []:
            c = (t or {}).get("config") or {}
            if "cache_dir" in c and "cache_file" in c:
                return {
                    "cache_dir": c.get("cache_dir", ""),
                    "cache_file": c.get("cache_file", ""),
                    "shards": int(c.get("cache_shards", 1)),
                    "embedding_model": c.get("faiss_embedding_model") or "sentence-transformers/all-MiniLM-L6-v2",
                }
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Merge search cache shards into master and build FAISS index."
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Cache directory (default: from config or database under recipe)",
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="Master cache DB path (default: from config)",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=None,
        help="Number of shards (default: from config or 16)",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model path or name (default: from config or MiniLM)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
        help="Device for embedding: cuda=fast for one-time build, cpu=no GPU (default: auto)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Max entries to index (default: all)",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip merge step; build FAISS from existing master only",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Incremental update: after merging shards, embed only new entries and append them to the existing FAISS index instead of rebuilding it.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Encode batch size (default 2048; try 4096 if GPU memory is sufficient)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Use N GPUs in parallel for encoding on CUDA (e.g. 4); full build and incremental are both supported",
    )
    args = parser.parse_args()

    cfg = load_config_paths()
    cache_dir = args.cache_dir or (cfg and cfg["cache_dir"])
    cache_file = args.cache_file or (cfg and cfg["cache_file"])
    shards = args.shards if args.shards is not None else (cfg and cfg["shards"] or 16)
    embedding_model = args.embedding_model or (cfg and cfg["embedding_model"]) or "sentence-transformers/all-MiniLM-L6-v2"

    if not cache_dir or not cache_file:
        print("Error: need --cache-dir and --cache-file (or config in config/tools.yaml)", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()

    if not args.skip_merge and shards > 1:
        print("Step 1: Merging shards into master ...")
        # Check the existing master row count before merging.
        import sqlite3 as _sq
        _mc = _sq.connect(cache_file, timeout=30.0)
        _before = _mc.execute("SELECT COUNT(*) FROM search_cache").fetchone()[0] if os.path.exists(cache_file) else 0
        _mc.close()
        n = merge_shards_into_master(cache_dir, cache_file, shards)
        print(f"  Master: {_before} -> {n} (new: {n - _before})")
    else:
        if args.skip_merge:
            print("Step 1: Skip merge (--skip-merge)")
        else:
            print("Step 1: Single DB, no merge")

    if args.incremental:
        print("Step 2: Incremental update (only new entries) ...")
        n = incremental_update_faiss(
            cache_dir=cache_dir,
            cache_file=cache_file,
            embedding_model=embedding_model,
            device=args.device,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus,
        )
        print(f"  Added {n} new entries to FAISS")
    else:
        print("Step 2: Building FAISS index (full) ...")
        n = build_faiss_from_search_cache(
            cache_dir=cache_dir,
            cache_file=cache_file,
            embedding_model=embedding_model,
            index_path=None,
            meta_path=None,
            max_entries=args.max_entries,
            device=args.device,
            batch_size=args.batch_size,
            num_gpus=args.num_gpus,
        )
        print(f"  Indexed {n} entries")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")
    if n and elapsed:
        print(f"  (~{n / elapsed:.0f} entries/s encoding)")


if __name__ == "__main__":
    main()
