# Copyright 2025 DeepResearch authors
# Worker for multi-GPU FAISS build.
# Usage: python _faiss_build_worker.py <cache_file> <start> <end> <embedding_model> <output_npy> [batch_size]
# CUDA_VISIBLE_DEVICES is set by the parent process before the child starts; this module does not import torch at top level.

import os
import sys
import sqlite3


def main():
    cache_file = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    embedding_model = sys.argv[4]
    output_npy = sys.argv[5]
    batch_size = int(sys.argv[6]) if len(sys.argv) > 6 else 2048

    import numpy as np
    from sentence_transformers import SentenceTransformer

    conn = sqlite3.connect(cache_file, timeout=60.0)
    cur = conn.cursor()
    cur.execute(
        "SELECT query FROM search_cache ORDER BY timestamp ASC LIMIT ? OFFSET ?",
        (end - start, start),
    )
    queries = [row[0] for row in cur.fetchall()]
    conn.close()
    if not queries:
        np.save(output_npy, np.zeros((0, 1), dtype=np.float32))
        return
    encoder = SentenceTransformer(embedding_model, device="cuda")
    embs = encoder.encode(
        queries,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(output_npy, embs)
    print(f"[worker gpu={os.environ.get('CUDA_VISIBLE_DEVICES')}] "
          f"Encoded {len(queries)} queries (batch_size={batch_size}) -> {output_npy}")


if __name__ == "__main__":
    main()
