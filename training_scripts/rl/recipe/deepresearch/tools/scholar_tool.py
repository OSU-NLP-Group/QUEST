# Copyright 2025 DeepResearch authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Search tool for DeepResearch using Serper API with SQLite cache.
Supports optional sharded cache and optional FAISS semantic retrieval:
- Given a query, retrieve top-1 from FAISS; if similarity >= threshold, return cached result.
- Otherwise call real Serper API. New results are written to both SQLite and FAISS (real-time write + query).
"""

import asyncio
import http.client
import json
import logging
import os
import re
import sys
import sqlite3
import threading
import time
import atexit
import hashlib
from typing import Any, List, Optional, Tuple

from uuid import uuid4

try:
    import requests
except ImportError:
    requests = None

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
_RECIPE_DIR = os.path.dirname(_TOOL_DIR)
_DEFAULT_CACHE_DIR = os.path.join(_RECIPE_DIR, "database")

# Optional FAISS + embedding (tool works without them)
_FAISS_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    import faiss
    import numpy as np
    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    np = None
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None


def contains_chinese(text: str) -> bool:
    return any("\u4E00" <= char <= "\u9FFF" for char in text)


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y"}


def _get_leader_flag(config: dict) -> bool:
    if config.get("cache_shard_leader") is not None:
        return bool(config.get("cache_shard_leader"))
    if os.getenv("SCHOLAR_CACHE_SHARD_LEADER") is not None:
        return _bool_env("SCHOLAR_CACHE_SHARD_LEADER", False)
    # Auto-detect common rank envs: only rank 0 is leader
    for env in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if os.getenv(env) is not None:
            try:
                return int(os.getenv(env, "-1")) == 0
            except ValueError:
                return False
    return False


def _is_search_error_result(result: str) -> bool:
    if not result:
        return True
    error_prefixes = (
        "[Search Error]",
        "Google scholar Timeout",
        "Google scholar failed:",
    )
    return result.startswith(error_prefixes)


def _is_empty_search_result(result: str) -> bool:
    return bool(result) and result.startswith("No results found")


def rewrite_faiss_result_header(result_text: str, current_query: str) -> str:
    """On a FAISS hit, rewrite 'A Google scholar for ''...'' found N results' to the current query to avoid exposing the original query."""
    if not result_text or not current_query:
        return result_text
    # Replace only the first occurrence and keep the remaining text.
    pattern = re.compile(r"^A Google scholar for '[^']*' found (\d+) results:\n\n", re.MULTILINE)
    def repl(m):
        return f"A Google scholar for '{current_query}' found {m.group(1)} results:\n\n"
    out = pattern.sub(repl, result_text, count=1)
    # If there is "No results found for '...'" also rewrite it to the current query.
    out = re.sub(r"^No results found for '[^']*'\.", f"No results found for '{current_query}'.", out, count=1)
    return out


class _RWLock:
    """Simple read-write lock: many readers or one writer. FAISS IndexFlatIP allows concurrent reads."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writers = 0

    def read_acquire(self) -> None:
        with self._cond:
            while self._writers > 0:
                self._cond.wait()
            self._readers += 1

    def read_release(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def write_acquire(self) -> None:
        self._cond.acquire()
        while self._readers > 0 or self._writers > 0:
            self._cond.wait()
        self._writers += 1

    def write_release(self) -> None:
        self._writers -= 1
        self._cond.notify_all()
        self._cond.release()


class FaissRetriever:
    """
    FAISS-based semantic cache: embed queries, search top-1, return cached result if score >= threshold.
    Supports real-time add + search (IndexFlatIP); suitable for ~200k vectors on CPU.
    """

    def __init__(
        self,
        cache_dir: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        index_path: Optional[str] = None,
        meta_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if not _FAISS_AVAILABLE or not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "FAISS retriever requires faiss-cpu and sentence-transformers. "
                "Install with: pip install faiss-cpu sentence-transformers"
            )
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.index_path = index_path or os.path.join(cache_dir, "scholar_faiss.index")
        self.meta_path = meta_path or os.path.join(cache_dir, "scholar_faiss_meta.db")
        self.threshold = float(similarity_threshold)
        self._rwlock = _RWLock()
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self._encoder = SentenceTransformer(embedding_model, device=device)
        self._dim = self._encoder.get_sentence_embedding_dimension()
        self._index: Optional["faiss.IndexFlatIP"] = None
        self._meta_conn: Optional[sqlite3.Connection] = None
        self._meta_read_local = threading.local()
        self._next_id = 0
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self._index = faiss.read_index(self.index_path)
                self._meta_conn = sqlite3.connect(self.meta_path, timeout=30.0, check_same_thread=False)
                self._meta_conn.row_factory = sqlite3.Row
                cur = self._meta_conn.cursor()
                cur.execute("SELECT MAX(id) FROM faiss_search_meta")
                row = cur.fetchone()
                self._next_id = (row[0] or 0) + 1
                logger.info("[scholar] FAISS loaded: %s vectors, next_id=%s", self._index.ntotal, self._next_id)
            except Exception as e:
                logger.warning("[scholar] FAISS load failed: %s, starting empty", e)
                self._index = None
                self._meta_conn = None
        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dim)
            self._meta_conn = sqlite3.connect(self.meta_path, timeout=30.0, check_same_thread=False)
            self._meta_conn.row_factory = sqlite3.Row
            cur = self._meta_conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS faiss_search_meta (
                    id INTEGER PRIMARY KEY,
                    query TEXT NOT NULL,
                    result TEXT NOT NULL
                )
                """
            )
            self._meta_conn.commit()
            self._next_id = 0

    def _embed(self, texts: list[str]) -> "np.ndarray":
        emb = self._encoder.encode(texts, normalize_embeddings=True)
        return emb.astype(np.float32)

    def _get_meta_read_conn(self) -> sqlite3.Connection:
        """Per-thread read connection to reduce cursor contention on shared sqlite connection."""
        conn = getattr(self._meta_read_local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.meta_path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._meta_read_local.conn = conn
        return conn

    def _search_top1_id_with_vec(self, vec: "np.ndarray") -> Tuple[Optional[int], float]:
        """Return (faiss_id, score) if top-1 similarity >= threshold else (None, score)."""
        if self._index.ntotal == 0:
            return None, 0.0
        self._rwlock.read_acquire()
        try:
            if self._index.ntotal == 0:
                return None, 0.0
            scores, ids = self._index.search(vec, 1)
            score = float(scores[0][0])
            if score < self.threshold:
                return None, score
            idx = int(ids[0][0])
            if idx < 0:
                return None, score
            return idx, score
        finally:
            self._rwlock.read_release()

    def search_top1(self, query: str) -> Tuple[Optional[str], float]:
        """Return (cached_result, score) if top-1 similarity >= threshold else (None, score). Concurrent reads allowed."""
        if self._index.ntotal == 0:
            return None, 0.0
        q = self._embed([query])
        return self.search_top1_with_vec(q)

    def search_top1_with_vec(self, vec: "np.ndarray") -> Tuple[Optional[str], float]:
        """Same as search_top1, but accepts a precomputed embedding vector with shape (1, dim). Used when an external encoder computes vectors for multi-GPU reads."""
        idx, score = self._search_top1_id_with_vec(vec)
        if idx is None:
            return None, score
        try:
            cur = self._get_meta_read_conn().cursor()
            cur.execute("SELECT result FROM faiss_search_meta WHERE id = ?", (idx,))
            row = cur.fetchone()
            return (row["result"], score) if row else (None, score)
        except Exception as e:
            logger.warning("[scholar] FAISS meta read failed id=%s: %s", idx, e)
            return None, score

    def search_top1_with_vecs(self, vecs: "np.ndarray") -> list[Tuple[Optional[str], float]]:
        """Batch version of search_top1_with_vec. vecs has shape (N, dim); returns N (result_or_none, score) pairs."""
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        n = int(vecs.shape[0])
        if n == 0:
            return []
        if self._index.ntotal == 0:
            return [(None, 0.0)] * n

        self._rwlock.read_acquire()
        try:
            if self._index.ntotal == 0:
                return [(None, 0.0)] * n

            scores, ids = self._index.search(vecs, 1)
            out: list[Tuple[Optional[str], float]] = [(None, float(scores[i][0])) for i in range(n)]
            pos_to_id: dict[int, int] = {}
            for i in range(n):
                score = float(scores[i][0])
                idx = int(ids[i][0])
                if score < self.threshold or idx < 0:
                    continue
                pos_to_id[i] = idx
            if not pos_to_id:
                return out

            id_to_result: dict[int, str] = {}
            try:
                cur = self._get_meta_read_conn().cursor()
                unique_ids = sorted(set(pos_to_id.values()))
                chunk_size = 500
                for start in range(0, len(unique_ids), chunk_size):
                    chunk = unique_ids[start : start + chunk_size]
                    placeholders = ",".join("?" for _ in chunk)
                    cur.execute(
                        f"SELECT id, result FROM faiss_search_meta WHERE id IN ({placeholders})",
                        chunk,
                    )
                    for row in cur.fetchall():
                        id_to_result[int(row["id"])] = row["result"]
            except Exception as e:
                logger.warning("[scholar] FAISS batch meta read failed: %s", e)
                return out

            for pos, idx in pos_to_id.items():
                result = id_to_result.get(idx)
                if result is not None:
                    out[pos] = (result, float(scores[pos][0]))
            return out
        finally:
            self._rwlock.read_release()

    def search_topk(
        self, query: str, k: int = 10
    ) -> list[Tuple[str, str, float]]:
        """Return top-k (cached_query, result, similarity_score). Used for testing/debugging."""
        if self._index.ntotal == 0:
            return []
        q = self._embed([query])
        self._rwlock.read_acquire()
        try:
            k = min(k, self._index.ntotal)
            scores, ids = self._index.search(q, k)
            out = []
            cur = self._meta_conn.cursor()
            for i in range(k):
                idx = int(ids[0][i])
                score = float(scores[0][i])
                cur.execute(
                    "SELECT query, result FROM faiss_search_meta WHERE id = ?",
                    (idx,),
                )
                row = cur.fetchone()
                if row:
                    out.append((row["query"], row["result"], score))
            return out
        finally:
            self._rwlock.read_release()

    def add(self, query: str, result: str) -> None:
        """Add one (query, result) and its embedding to the index (real-time write). Exclusive with other writers and readers."""
        vec = self._embed([query])
        self.add_vectors(vec, [query], [result])

    def add_vector(self, vec: "np.ndarray", query: str, result: str) -> None:
        """Append one (query, result) with a precomputed embedding. vec has shape (1, dim) or (dim,) float32 and is used for multi-GPU parallel writes."""
        self.add_vectors(vec, [query], [result])

    def add_vectors(self, vecs: "np.ndarray", queries: list[str], results: list[str]) -> None:
        """Batch append multiple entries with precomputed embeddings to reduce write locks and sqlite commits."""
        if len(queries) != len(results):
            raise ValueError(f"queries/results length mismatch: {len(queries)} vs {len(results)}")
        if len(queries) == 0:
            return
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[0] != len(queries):
            raise ValueError(f"vec rows must match queries: {vecs.shape[0]} vs {len(queries)}")
        if vecs.shape[1] != self._dim:
            raise ValueError(f"vec dim must be {self._dim}, got {vecs.shape[1]}")
        self._rwlock.write_acquire()
        try:
            start_idx = self._next_id
            # IndexFlatIP does not support add_with_ids，use add() to append in order,
            # idx == self._index.ntotal(before append)，and matches the id in the meta table one-to-one.
            self._index.add(vecs)
            self._next_id += len(queries)
            cur = self._meta_conn.cursor()
            rows = [
                (start_idx + i, queries[i], results[i])
                for i in range(len(queries))
            ]
            cur.executemany(
                "INSERT INTO faiss_search_meta (id, query, result) VALUES (?, ?, ?)",
                rows,
            )
            self._meta_conn.commit()
            logger.debug(
                "[FAISS] batch added start_id=%s batch=%s (ntotal=%s)",
                start_idx, len(queries), self._index.ntotal,
            )
        finally:
            self._rwlock.write_release()

    def save(self) -> None:
        self._rwlock.write_acquire()
        try:
            if self._index is not None and self._index.ntotal > 0:
                faiss.write_index(self._index, self.index_path)
                logger.debug("[scholar] FAISS index saved to %s", self.index_path)
        finally:
            self._rwlock.write_release()

    def close(self) -> None:
        self.save()
        if self._meta_conn is not None:
            try:
                self._meta_conn.close()
            except Exception:
                pass
            self._meta_conn = None


def merge_shards_into_master(
    cache_dir: str,
    cache_file: str,
    shards: int,
    clear_shards: bool = True,
) -> int:
    """
    Merge all shard DBs (scholar_cache table) into the master DB file.
    Master path is cache_file; shards are cache_dir/{base}_shard{0..shards-1}.db.
    clear_shards: Clear merged shard tables after merging, so the next run only processes new data.
    Returns number of rows in master after merge.
    """
    os.makedirs(cache_dir, exist_ok=True)
    base, _ = os.path.splitext(os.path.basename(cache_file))
    shard_files = [
        os.path.join(cache_dir, f"{base}_shard{idx}.db") for idx in range(shards)
    ]
    master = sqlite3.connect(cache_file, timeout=120.0)
    master.row_factory = sqlite3.Row
    cur = master.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS scholar_cache (
            query TEXT PRIMARY KEY,
            result TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """
    )
    master.commit()
    # Record the master row count before merging.
    cur.execute("SELECT COUNT(*) FROM scholar_cache")
    before = cur.fetchone()[0]
    shard_total = 0
    for path in shard_files:
        if not os.path.exists(path):
            continue
        conn = sqlite3.connect(path, timeout=60.0)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT query, result, timestamp FROM scholar_cache")
        rows = c.fetchall()
        shard_total += len(rows)
        for row in rows:
            cur.execute(
                """
                INSERT OR IGNORE INTO scholar_cache (query, result, timestamp)
                VALUES (?, ?, ?)
                """,
                (row["query"], row["result"], row["timestamp"]),
            )
        master.commit()
        # Clear this shard after merging so it is not processed again next time.
        if clear_shards and rows:
            c.execute("DELETE FROM scholar_cache")
            conn.commit()
        conn.close()
    cur.execute("SELECT COUNT(*) FROM scholar_cache")
    after = cur.fetchone()[0]
    master.close()
    new_rows = after - before
    logger.info(
        "[scholar] Merge: shards had %s rows, master %s -> %s (new: %s)",
        shard_total, before, after, new_rows,
    )
    return after


def build_faiss_from_scholar_cache(
    cache_dir: str,
    cache_file: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.85,
    index_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    max_entries: Optional[int] = None,
    device: Optional[str] = None,
    batch_size: int = 2048,
    num_gpus: Optional[int] = None,
) -> int:
    """
    One-time build of FAISS index from existing SQLite scholar_cache (e.g. master DB).
    device: "cuda" or "cpu". batch_size: increase this when GPU memory is sufficient (for example 2048/4096) for speedup.
    num_gpus: Set to the GPU count for multi-GPU encoding, which can approach linear speedup.
    """
    if not _FAISS_AVAILABLE or not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Need faiss-cpu and sentence-transformers installed")
    if not os.path.exists(cache_file):
        return 0
    conn = sqlite3.connect(cache_file, timeout=60.0)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT query, result FROM scholar_cache ORDER BY timestamp ASC")
    rows = cur.fetchall()
    if max_entries is not None:
        rows = rows[: max_entries]
    conn.close()
    if not rows:
        return 0
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    n = len(rows)
    os.makedirs(cache_dir, exist_ok=True)
    ipath = index_path or os.path.join(cache_dir, "scholar_faiss.index")
    mpath = meta_path or os.path.join(cache_dir, "scholar_faiss_meta.db")

    if device == "cuda" and num_gpus is not None and num_gpus > 1:
        try:
            import torch
            ngpu = min(num_gpus, torch.cuda.device_count(), n)
        except Exception:
            ngpu = 1
        if ngpu <= 1:
            num_gpus = None
        else:
            import subprocess
            import tempfile
            # Use subprocesses to start isolated child processes. Set CUDA_VISIBLE_DEVICES before each child starts,
            # fully isolating the CUDA environment so each process sees only its assigned GPU.
            worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_faiss_build_worker.py")
            chunk_size = (n + ngpu - 1) // ngpu
            temp_dir = tempfile.mkdtemp(prefix="faiss_build_")
            procs = []
            for i in range(ngpu):
                s, e = i * chunk_size, min((i + 1) * chunk_size, n)
                if s >= e:
                    continue
                out = os.path.join(temp_dir, f"embs_{i}.npy")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(i)
                p = subprocess.Popen(
                    [sys.executable, worker_script, cache_file, str(s), str(e), embedding_model, out, str(batch_size)],
                    env=env,
                )
                procs.append((p, s, e, out))
            for p, s, e, out in procs:
                p.wait()
                if p.returncode != 0:
                    raise RuntimeError(f"Worker (GPU chunk {s}-{e}) exited with {p.returncode}")
            dim = None
            for _, _, _, out in sorted(procs, key=lambda x: x[1]):
                embs = np.load(out)
                if embs.size > 0:
                    dim = embs.shape[1]
                    break
            if dim is None:
                raise RuntimeError("No embeddings produced by workers")
            index = faiss.IndexFlatIP(dim)
            meta_conn = sqlite3.connect(mpath, timeout=30.0)
            meta_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faiss_search_meta (
                    id INTEGER PRIMARY KEY,
                    query TEXT NOT NULL,
                    result TEXT NOT NULL
                )
                """
            )
            meta_conn.commit()
            for _, s, e, out in sorted(procs, key=lambda x: x[1]):
                embs = np.load(out)
                if embs.size == 0:
                    continue
                index.add(embs)
                cur = meta_conn.cursor()
                for i in range(s, e):
                    r = rows[i]
                    cur.execute(
                        "INSERT OR REPLACE INTO faiss_search_meta (id, query, result) VALUES (?, ?, ?)",
                        (i, r["query"], r["result"]),
                    )
                meta_conn.commit()
                try:
                    os.remove(out)
                except Exception:
                    pass
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            meta_conn.close()
            faiss.write_index(index, ipath)
            logger.info("[scholar] FAISS built from cache (multi-GPU): %s entries -> %s", n, ipath)
            return n

    encoder = SentenceTransformer(embedding_model, device=device)
    dim = encoder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)
    meta_conn = sqlite3.connect(mpath, timeout=30.0)
    meta_conn.execute(
        """
        CREATE TABLE IF NOT EXISTS faiss_search_meta (
            id INTEGER PRIMARY KEY,
            query TEXT NOT NULL,
            result TEXT NOT NULL
        )
        """
    )
    meta_conn.commit()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        queries = [r["query"] for r in batch]
        embs = encoder.encode(queries, normalize_embeddings=True).astype(np.float32)
        index.add(embs)
        cur = meta_conn.cursor()
        for i, r in enumerate(batch):
            cur.execute(
                "INSERT OR REPLACE INTO faiss_search_meta (id, query, result) VALUES (?, ?, ?)",
                (start + i, r["query"], r["result"]),
            )
        meta_conn.commit()
    faiss.write_index(index, ipath)
    meta_conn.close()
    logger.info("[scholar] FAISS built from cache: %s entries -> %s", len(rows), ipath)
    return len(rows)


def incremental_update_faiss(
    cache_dir: str,
    cache_file: str,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    index_path: Optional[str] = None,
    meta_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_size: int = 2048,
    num_gpus: Optional[int] = None,
) -> int:
    """
    Incremental update: merge shards into master, then embed and append only entries present in master but missing from FAISS metadata.
    This is much faster than a full rebuild because it only processes new entries.
    Returns: number of entries added in this run.
    """
    if not _FAISS_AVAILABLE or not _SENTENCE_TRANSFORMERS_AVAILABLE:
        raise RuntimeError("Need faiss-cpu and sentence-transformers installed")
    if not os.path.exists(cache_file):
        return 0

    os.makedirs(cache_dir, exist_ok=True)
    ipath = index_path or os.path.join(cache_dir, "scholar_faiss.index")
    mpath = meta_path or os.path.join(cache_dir, "scholar_faiss_meta.db")

    if not os.path.exists(ipath) or not os.path.exists(mpath):
        raise FileNotFoundError(
            f"FAISS index or meta not found at {ipath} / {mpath}. "
            "Run full build first (without --incremental)."
        )

    # 1) Read all queries from master.
    master_conn = sqlite3.connect(cache_file, timeout=60.0)
    master_conn.row_factory = sqlite3.Row
    cur = master_conn.cursor()
    cur.execute("SELECT query, result FROM scholar_cache ORDER BY timestamp ASC")
    all_rows = cur.fetchall()
    master_conn.close()

    # 2) Read the existing query set from FAISS metadata.
    meta_conn = sqlite3.connect(mpath, timeout=30.0)
    meta_conn.row_factory = sqlite3.Row
    mcur = meta_conn.cursor()
    mcur.execute("SELECT query FROM faiss_search_meta")
    existing_queries = {row["query"] for row in mcur.fetchall()}
    mcur.execute("SELECT MAX(id) FROM faiss_search_meta")
    row = mcur.fetchone()
    next_id = (row[0] or 0) + 1 if row and row[0] is not None else 0

    # 3) Find new entries.
    new_rows = [r for r in all_rows if r["query"] not in existing_queries]
    if not new_rows:
        meta_conn.close()
        logger.info("[scholar] FAISS incremental: no new entries")
        return 0
    total_new = len(new_rows)
    print(f"[incremental] total new entries: {total_new}", flush=True)

    # 4) Load the existing index.
    index = faiss.read_index(ipath)

    # 5) Encode and append new entries.
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    def _print_overall_progress(stage: str, done: int, total: int, stage_t0: float) -> None:
        elapsed = max(1e-6, time.perf_counter() - stage_t0)
        pct = 100.0 * done / max(1, total)
        speed = done / elapsed
        remain = max(total - done, 0)
        eta = remain / speed if speed > 1e-9 else float("inf")
        eta_str = f"{eta:.1f}s" if eta != float("inf") else "inf"
        print(
            f"[incremental:{stage}] overall {done}/{total} ({pct:.1f}%), {speed:.1f} q/s, eta {eta_str}",
            flush=True,
        )

    added = 0
    if device == "cuda" and num_gpus is not None and num_gpus > 1:
        try:
            import torch
            ngpu = min(num_gpus, torch.cuda.device_count(), len(new_rows))
        except Exception:
            ngpu = 1
        if ngpu > 1:
            import shutil
            import subprocess
            import tempfile

            # Reuse the existing multi-GPU embedding worker by writing incremental rows into a temp cache DB.
            worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_faiss_build_worker.py")
            visible_devices = [x.strip() for x in os.getenv("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
            if visible_devices:
                worker_devices = visible_devices[:ngpu]
                ngpu = min(ngpu, len(worker_devices))
            else:
                worker_devices = [str(i) for i in range(ngpu)]
            if ngpu > 1:
                chunk_size = (len(new_rows) + ngpu - 1) // ngpu
                temp_dir = tempfile.mkdtemp(prefix="faiss_incremental_")
                temp_cache = os.path.join(temp_dir, "incremental_cache.db")
                tmp_conn = sqlite3.connect(temp_cache, timeout=60.0)
                tmp_cur = tmp_conn.cursor()
                tmp_cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scholar_cache (
                        query TEXT PRIMARY KEY,
                        result TEXT NOT NULL,
                        timestamp REAL NOT NULL
                    )
                    """
                )
                tmp_cur.executemany(
                    "INSERT OR REPLACE INTO scholar_cache (query, result, timestamp) VALUES (?, ?, ?)",
                    [(r["query"], r["result"], float(i)) for i, r in enumerate(new_rows)],
                )
                tmp_conn.commit()
                tmp_conn.close()

                procs = []
                try:
                    for i in range(ngpu):
                        s, e = i * chunk_size, min((i + 1) * chunk_size, len(new_rows))
                        if s >= e:
                            continue
                        out = os.path.join(temp_dir, f"embs_{i}.npy")
                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = worker_devices[i]
                        p = subprocess.Popen(
                            [sys.executable, worker_script, temp_cache, str(s), str(e), embedding_model, out, str(batch_size)],
                            env=env,
                        )
                        procs.append((p, s, e, out))
                    encode_t0 = time.perf_counter()
                    encoded_done = 0
                    failed_workers = []
                    pending = {p: (s, e, out) for p, s, e, out in procs}
                    while pending:
                        progressed = False
                        for p in list(pending.keys()):
                            rc = p.poll()
                            if rc is None:
                                continue
                            s, e, _ = pending.pop(p)
                            if rc != 0:
                                failed_workers.append((s, e, rc))
                            else:
                                encoded_done += (e - s)
                            _print_overall_progress("encode", encoded_done, total_new, encode_t0)
                            progressed = True
                        if pending and not progressed:
                            time.sleep(0.2)
                    if failed_workers:
                        detail = ", ".join([f"{s}-{e}:{rc}" for s, e, rc in failed_workers])
                        raise RuntimeError(f"Incremental workers failed (chunk:code): {detail}")

                    append_t0 = time.perf_counter()
                    for _, s, e, out in sorted(procs, key=lambda x: x[1]):
                        embs = np.load(out)
                        if embs.size == 0:
                            continue
                        index.add(embs)
                        mcur = meta_conn.cursor()
                        for r in new_rows[s:e]:
                            mcur.execute(
                                "INSERT OR REPLACE INTO faiss_search_meta (id, query, result) VALUES (?, ?, ?)",
                                (next_id, r["query"], r["result"]),
                            )
                            next_id += 1
                        meta_conn.commit()
                        added += (e - s)
                        _print_overall_progress("append", added, total_new, append_t0)
                finally:
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass

                faiss.write_index(index, ipath)
                meta_conn.close()
                logger.info(
                    "[scholar] FAISS incremental (multi-GPU): added %s new entries (total now %s)",
                    added,
                    index.ntotal,
                )
                return added

    single_t0 = time.perf_counter()
    encoder = SentenceTransformer(embedding_model, device=device)
    for start in range(0, len(new_rows), batch_size):
        batch = new_rows[start: start + batch_size]
        queries = [r["query"] for r in batch]
        embs = encoder.encode(queries, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        index.add(embs)
        mcur = meta_conn.cursor()
        for r in batch:
            mcur.execute(
                "INSERT OR REPLACE INTO faiss_search_meta (id, query, result) VALUES (?, ?, ?)",
                (next_id, r["query"], r["result"]),
            )
            next_id += 1
        meta_conn.commit()
        added += len(batch)
        _print_overall_progress("single", added, total_new, single_t0)

    faiss.write_index(index, ipath)
    meta_conn.close()
    logger.info("[scholar] FAISS incremental: added %s new entries (total now %s)", added, index.ntotal)
    return added


class SearchCache:
    """SQLite cache mapping query -> result. Supports sharded DB files."""

    def __init__(
        self,
        cache_dir: str,
        cache_file: str,
        shards: int = 1,
        resume: bool = True,
        auto_shard: bool = False,
        auto_merge: bool = False,
        is_leader: bool = False,
    ) -> None:
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.master_file = cache_file
        self.shards = max(1, int(shards))
        self.resume = resume
        self.auto_shard = auto_shard
        self.auto_merge = auto_merge
        self.is_leader = is_leader

        base, _ = os.path.splitext(os.path.basename(self.master_file))
        self._shard_files = [
            os.path.join(self.cache_dir, f"{base}_shard{idx}.db") for idx in range(self.shards)
        ]

        self._conns: dict[int, sqlite3.Connection] = {}
        self._locks: dict[int, threading.Lock] = {}

        if self.shards == 1:
            self._get_conn(0)

        if self.shards > 1 and self.auto_shard and self.is_leader:
            self._split_master()

        if self.shards > 1 and self.auto_merge and self.is_leader:
            atexit.register(self.merge_shards)

        atexit.register(self.close)

    def close(self) -> None:
        """Close all SQLite connections."""
        for conn in self._conns.values():
            try:
                conn.close()
            except Exception:
                pass

    def _open_conn(self, path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-8000")
        conn.commit()
        self._ensure_table(conn)
        return conn

    def _ensure_table(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scholar_cache (
                query TEXT PRIMARY KEY,
                result TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        conn.commit()

    def _get_conn(self, shard_id: int) -> sqlite3.Connection:
        if shard_id not in self._conns:
            if shard_id == 0 and self.shards == 1:
                path = self.master_file
            else:
                path = self._shard_files[shard_id]
            self._conns[shard_id] = self._open_conn(path)
            self._locks[shard_id] = threading.Lock()
        return self._conns[shard_id]

    def _shard_id(self, query: str) -> int:
        if self.shards == 1:
            return 0
        digest = hashlib.md5(query.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.shards

    def _split_master(self) -> None:
        if not os.path.exists(self.master_file):
            return

        if not self.resume:
            for shard_file in self._shard_files:
                if os.path.exists(shard_file):
                    os.remove(shard_file)

        master_conn = self._open_conn(self.master_file)
        master_cursor = master_conn.cursor()
        master_cursor.execute("SELECT query, result, timestamp FROM scholar_cache")

        for row in master_cursor.fetchall():
            shard_id = self._shard_id(row["query"])
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO scholar_cache (query, result, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (row["query"], row["result"], row["timestamp"]),
                )
                conn.commit()

    def merge_shards(self) -> None:
        if self.shards <= 1:
            return
        master_conn = self._open_conn(self.master_file)
        master_cursor = master_conn.cursor()
        for shard_id, shard_file in enumerate(self._shard_files):
            if not os.path.exists(shard_file):
                continue
            conn = self._open_conn(shard_file)
            cursor = conn.cursor()
            cursor.execute("SELECT query, result, timestamp FROM scholar_cache")
            rows = cursor.fetchall()
            for row in rows:
                master_cursor.execute(
                    """
                    INSERT OR REPLACE INTO scholar_cache (query, result, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (row["query"], row["result"], row["timestamp"]),
                )
            master_conn.commit()

    def get(self, query: str) -> Optional[str]:
        shard_id = self._shard_id(query)
        conn = self._get_conn(shard_id)
        with self._locks[shard_id]:
            cursor = conn.cursor()
            cursor.execute("SELECT result FROM scholar_cache WHERE query = ?", (query,))
            row = cursor.fetchone()
            return row["result"] if row else None

    def set(self, query: str, result: str) -> None:
        shard_id = self._shard_id(query)
        conn = self._get_conn(shard_id)
        with self._locks[shard_id]:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO scholar_cache (query, result, timestamp)
                VALUES (?, ?, ?)
                """,
                (query, result, time.time()),
            )
            conn.commit()


def google_scholar_sync(query: str, topk: int = 10, serper_key: str = "") -> str:
    if not serper_key:
        return "[Search Error] SERPER_KEY_ID environment variable not set."

    # Use a 30s timeout for each Serper call to avoid slowing the whole /scholar request; client timeout is 120s.
    # If Serper is slow, fail fast and retry instead of waiting the full 60s.
    serper_timeout = int(os.environ.get("SERPER_TIMEOUT", "30"))
    max_retries = int(os.environ.get("SERPER_MAX_RETRIES", "2"))  # Reduce retry count to avoid accumulated latency.
    
    if contains_chinese(query):
        payload = {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn", "num": topk}
    else:
        payload = {"q": query, "location": "United States", "gl": "us", "hl": "en", "num": topk}

    headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}
    url = "https://google.serper.dev/scholar"

    # Use requests for more reliable timeout control.
    if not requests:
        return "[Search Error] requests library not installed."
    
    last_error = None
    results = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=serper_timeout)
            r.raise_for_status()
            results = r.json()
            last_error = None  # Retry succeeded; clear the previous error.
            break
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries:
                wait = 1.0 * (attempt + 1)  # Backoff: 1s, 2s.
                logger.debug("[serper] timeout (attempt %s/%s), retry in %.1fs", attempt + 1, max_retries + 1, wait)
                time.sleep(wait)
            else:
                logger.warning("[serper] timeout after %s attempts: %s", max_retries + 1, e)
                return f"Google scholar Timeout after {max_retries + 1} attempts. Please try again later."
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries:
                wait = 1.0 * (attempt + 1)
                logger.debug("[serper] error (attempt %s/%s), retry in %.1fs: %s", attempt + 1, max_retries + 1, wait, e)
                time.sleep(wait)
            else:
                logger.warning("[serper] failed after %s attempts: %s", max_retries + 1, e)
                return f"Google scholar failed: {e}"
        except Exception as e:
            logger.warning("[serper] unexpected error: %s", e)
            return f"Google scholar error: {e}"
    
    if last_error or results is None:
        return f"Google scholar failed: {last_error}"

    try:
        if "organic" not in results:
            raise ValueError(f"No results found for query: '{query}'. Use a less specific query.")

        web_snippets = []
        idx = 0
        for page in results.get("organic", [])[:topk]:
            idx += 1
            date_published = f"\nDate published: {page['year']}" if "year" in page else ""
            publication_info = f"\npublicationInfo: {page['publicationInfo']}" if "publicationInfo" in page else ""
            snippet = f"\n{page['snippet']}" if "snippet" in page else ""
            cited_by = f"\ncitedBy: {page['citedBy']}" if "citedBy" in page else ""
            # Use a real URL in markdown. Prefer direct PDF when available, else fall back to the result link.
            link_url = page.get("pdfUrl") or page.get("link") or ""
            title = page.get("title", "")
            if link_url:
                title_part = f"[{title}]({link_url})"
            else:
                title_part = title

            formatted = (
                f"{idx}. {title_part}{publication_info}{date_published}{cited_by}\n{snippet}"
            )
            formatted = formatted.replace("Your browser can't play this video.", "")
            web_snippets.append(formatted)

        content = (
            f"A Google scholar for '{query}' found {len(web_snippets)} results:\n\n## Scholar Results\n"
            + "\n\n".join(web_snippets)
        )
        return content
    except Exception:
        return f"No results found for '{query}'. Try with a more general query."


class DeepResearchScholarTool(BaseTool):
    """Scholar tool with optional SQLite cache (sharded), or forward to HTTP scholar service."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # If scholar_service_url or scholar_nodes_conf is configured, only forward HTTP requests and do not build local cache/FAISS.
        # scholar_nodes_conf: path to a hot-reloadable conf file (re-read every call, like eval_llm_nodes.conf)
        self._scholar_nodes_conf = (
            config.get("scholar_nodes_conf") or os.environ.get("SCHOLAR_NODES_CONF") or ""
        )
        # Static fallback URL (used when scholar_nodes_conf is absent or fails to parse)
        self._scholar_service_url_static = (
            (config.get("scholar_service_url") or os.environ.get("SCHOLAR_SERVICE_URL") or "").rstrip("/")
        )
        self._use_scholar_service = bool(self._scholar_nodes_conf or self._scholar_service_url_static)
        if self._use_scholar_service:
            self.cache = None
            self.faiss_retriever = None
            self.serper_key = ""
            self.scholar_service_max_retries = int(config.get("scholar_service_max_retries") or os.environ.get("SCHOLAR_SERVICE_MAX_RETRIES", "3"))
            self.scholar_service_timeout = int(config.get("scholar_service_timeout") or os.environ.get("SCHOLAR_SERVICE_TIMEOUT", "120"))
            logger.info(
                "[scholar] Using HTTP scholar service (nodes_conf=%s, static_url=%s, max_retries=%s, timeout=%ss)",
                self._scholar_nodes_conf or "(none)",
                self._scholar_service_url_static or "(none)",
                self.scholar_service_max_retries,
                self.scholar_service_timeout,
            )
            return

        self.serper_key = config.get("serper_key") or os.environ.get("SERPER_KEY_ID", "")
        self.topk = config.get("topk", 10)
        self.num_workers = config.get("num_workers", 120)

        cache_enabled = config.get("cache_enabled")
        if cache_enabled is None:
            cache_enabled = _bool_env("SCHOLAR_CACHE_ENABLED", True)
        cache_resume = config.get("cache_resume")
        if cache_resume is None:
            cache_resume = _bool_env("SCHOLAR_CACHE_RESUME", True)

        cache_dir = config.get("cache_dir") or os.getenv("SCHOLAR_CACHE_DIR", "")
        if not cache_dir:
            cache_dir = _DEFAULT_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = config.get("cache_file") or os.getenv("SCHOLAR_CACHE_FILE", "")
        if not cache_file:
            cache_file = os.path.join(cache_dir, "scholar_cache.db")

        cache_shards = config.get("cache_shards")
        if cache_shards is None:
            cache_shards = int(os.getenv("SCHOLAR_CACHE_SHARDS", "1"))

        cache_auto_shard = config.get("cache_auto_shard")
        if cache_auto_shard is None:
            cache_auto_shard = _bool_env("SCHOLAR_CACHE_AUTO_SHARD", False)

        cache_auto_merge = config.get("cache_auto_merge")
        if cache_auto_merge is None:
            cache_auto_merge = _bool_env("SCHOLAR_CACHE_AUTO_MERGE", False)

        is_leader = _get_leader_flag(config)

        self.cache = (
            SearchCache(
                cache_dir=cache_dir,
                cache_file=cache_file,
                shards=cache_shards,
                resume=cache_resume,
                auto_shard=cache_auto_shard,
                auto_merge=cache_auto_merge,
                is_leader=is_leader,
            )
            if cache_enabled
            else None
        )

        # Optional FAISS semantic retrieval: top-1 by similarity, fallback to Serper if below threshold
        self.faiss_retriever: Optional[FaissRetriever] = None
        faiss_enabled = config.get("faiss_enabled")
        if faiss_enabled is None:
            faiss_enabled = _bool_env("SCHOLAR_FAISS_ENABLED", False)
        if faiss_enabled and cache_enabled:
            faiss_threshold = config.get("faiss_similarity_threshold")
            if faiss_threshold is None:
                try:
                    faiss_threshold = float(os.getenv("SCHOLAR_FAISS_SIMILARITY_THRESHOLD", "0.85"))
                except ValueError:
                    faiss_threshold = 0.85
            faiss_model = config.get("faiss_embedding_model") or os.getenv("SCHOLAR_FAISS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            try:
                self.faiss_retriever = FaissRetriever(
                    cache_dir=cache_dir,
                    embedding_model=faiss_model,
                    similarity_threshold=faiss_threshold,
                    index_path=config.get("faiss_index_path") or os.getenv("SCHOLAR_FAISS_INDEX_PATH"),
                    meta_path=config.get("faiss_meta_path") or os.getenv("SCHOLAR_FAISS_META_PATH"),
                )
                atexit.register(self._faiss_close)
                logger.info(
                    "[scholar] FAISS semantic retrieval enabled (threshold=%.2f, model=%s)",
                    self.faiss_retriever.threshold,
                    faiss_model,
                )
            except Exception as e:
                logger.warning("[scholar] FAISS retriever init failed: %s; using cache+Serper only", e)

        if not self.serper_key:
            logger.warning("SERPER_KEY_ID not set, scholar tool will not work")

    def _faiss_close(self) -> None:
        if self.faiss_retriever is not None:
            try:
                self.faiss_retriever.close()
            except Exception:
                pass

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"queries": [], "results": []}
        return instance_id, ToolResponse()

    def _scholar_with_cache(self, query: str) -> Tuple[str, str, Optional[float]]:
        """Return (result_text, channel, similarity). The stripped query is used as the cache/FAISS key."""
        q = (query or "").strip()
        if not q:
            return "[Tool Error] Scholar query cannot be empty.", "serper", None
        # 1) Exact match in SQLite
        if self.cache:
            cached = self.cache.get(q)
            if cached:
                logger.info("[scholar] cache hit (exact) for query: %s", q)
                return cached, "exact", None

        # 2) FAISS top-1: if similarity >= threshold, return cached result
        if self.faiss_retriever is not None:
            faiss_result, score = self.faiss_retriever.search_top1(q)
            if faiss_result is not None:
                faiss_result = rewrite_faiss_result_header(faiss_result, q)
                logger.info("[scholar] cache hit (FAISS) for query: %s (score=%.3f)", q, score)
                return faiss_result, "faiss", round(score, 6)

        # 3) Call real Serper API
        result = google_scholar_sync(q, self.topk, self.serper_key)

        # 4) Write to SQLite and FAISS when result is good
        if self.cache and result:
            if not _is_empty_search_result(result) and not _is_search_error_result(result):
                self.cache.set(q, result)
                if self.faiss_retriever is not None:
                    try:
                        self.faiss_retriever.add(q, result)
                        logger.info("[scholar] cached (exact+faiss) for query=%r", q[:80])
                    except Exception as e:
                        logger.warning("[scholar] FAISS add failed: %s", e)
            elif _is_empty_search_result(result) or _is_search_error_result(result):
                logger.info("[scholar] not cached (empty/error result) for query=%r", q[:80])

        return result, "serper", None

    def _get_scholar_service_url(self) -> str:
        """Hot-reload URL from scholar_nodes_conf on every call; fall back to static URL."""
        if self._scholar_nodes_conf:
            try:
                with open(self._scholar_nodes_conf, encoding="utf-8") as f:
                    for raw_line in f:
                        line = raw_line.split("#", 1)[0].strip()
                        if "=" in line:
                            k, v = line.split("=", 1)
                            if k.strip().lower() == "url":
                                return v.strip().rstrip("/")
            except Exception as e:
                logger.warning("[scholar] Failed to read scholar_nodes_conf %s: %s", self._scholar_nodes_conf, e)
        return self._scholar_service_url_static

    def _scholar_via_service(self, queries: list[str]) -> tuple[str, dict]:
        """Sync: POST to scholar service, return (text, metrics). Automatically retry on timeout, connection errors, or 5xx responses."""
        if not requests:
            return "[Search Error] requests not installed; cannot call scholar service.", {"error_count": len(queries), "status": "error"}
        url = f"{self._get_scholar_service_url()}/scholar"
        max_retries = self.scholar_service_max_retries
        timeout_sec = self.scholar_service_timeout
        last_error = None
        for attempt in range(max_retries):
            try:
                r = requests.post(url, json={"query": queries}, timeout=timeout_sec)
                r.raise_for_status()
                data = r.json()
                metrics = data.get("metrics", {})
                metrics["sources"] = data.get("sources", [])
                return data.get("text", ""), metrics
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("[scholar] service timeout/connection (attempt %s/%s), retry in %ss: %s", attempt + 1, max_retries, wait, e)
                    time.sleep(wait)
                else:
                    logger.warning("[scholar] service call failed after %s retries: %s", max_retries, e)
                    return f"[Search Error] Service unavailable: {e}", {"error_count": len(queries), "status": "error"}
            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response is not None and e.response.status_code >= 500 and attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("[scholar] service 5xx (attempt %s/%s), retry in %ss: %s", attempt + 1, max_retries, wait, e)
                    time.sleep(wait)
                else:
                    logger.warning("[scholar] service call failed: %s", e)
                    return f"[Search Error] Service error: {e}", {"error_count": len(queries), "status": "error"}
            except Exception as e:
                last_error = e
                logger.warning("[scholar] service call failed: %s", e)
                return f"[Search Error] Service unavailable: {e}", {"error_count": len(queries), "status": "error"}
        return f"[Search Error] Service unavailable: {last_error}", {"error_count": len(queries), "status": "error"}

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        query = parameters.get("query", "")

        if not query:
            return ToolResponse(text="[Tool Error] Scholar query cannot be empty."), 0.0, {"error": "empty_query"}

        queries = query if isinstance(query, list) else [query]

        if self._use_scholar_service:
            loop = asyncio.get_running_loop()
            text, metrics = await loop.run_in_executor(None, self._scholar_via_service, queries)
            self._instance_dict[instance_id]["queries"].extend(queries)
            self._instance_dict[instance_id]["results"].append(text)
            return ToolResponse(text=text), 0.0, metrics

        loop = asyncio.get_running_loop()
        out: List[Tuple[str, str, Optional[float]]] = []
        for q in queries:
            out.append(await loop.run_in_executor(None, self._scholar_with_cache, q))
        results = [t[0] for t in out]
        sources = [{"source": t[1], "similarity": t[2]} for t in out]

        self._instance_dict[instance_id]["queries"].extend(queries)
        self._instance_dict[instance_id]["results"].extend(results)

        combined_result = "\n=======\n".join(results)
        error_count = sum(1 for result in results if _is_search_error_result(result))
        empty_count = sum(1 for result in results if _is_empty_search_result(result))
        if error_count == len(results):
            status = "error"
        elif error_count > 0:
            status = "partial_success"
        else:
            status = "success"
        metrics = {
            "query_count": len(queries),
            "error_count": error_count,
            "empty_count": empty_count,
            "status": status,
            "sources": sources,
        }

        return ToolResponse(text=combined_result), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
