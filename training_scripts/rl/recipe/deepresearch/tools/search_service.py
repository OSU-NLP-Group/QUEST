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
HTTP service for DeepResearch search: SQLite cache + FAISS + Serper.
Deploy on a dedicated machine (e.g. 2×A100) for high concurrent read/write.
"""

import asyncio
import collections
import logging
import os
import sys
import time
from typing import Any, List, Literal, Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None

# Ensure `recipe` can be imported from any cwd (`verl` is `recipe`'s parent directory).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VERL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))
_RECIPE_DIR = os.path.dirname(_SCRIPT_DIR)
_DEFAULT_CACHE_DIR = os.path.join(_RECIPE_DIR, "database")
if _VERL_ROOT not in sys.path:
    sys.path.insert(0, _VERL_ROOT)

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recipe.deepresearch.tools.search_tool import (
    FaissRetriever,
    SearchCache,
    _is_empty_search_result,
    _is_search_error_result,
    google_search_sync,
    rewrite_faiss_result_header,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y"}


def _resolve_oc_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if not (value.startswith("${oc.env:") and value.endswith("}")):
        return value
    inner = value[len("${oc.env:") : -1]
    name, _, default = inner.partition(",")
    return os.getenv(name.strip(), default)


def _load_config() -> dict:
    """Load from env vars; override from SEARCH_SERVICE_CONFIG YAML if provided."""
    config = {
        "serper_key": os.getenv("SERPER_KEY_ID", ""),
        "topk": int(os.getenv("SEARCH_TOP_K", "10")),
        "serper_timeout": int(os.getenv("SERPER_TIMEOUT", "20")),
        "serper_max_retries": int(os.getenv("SERPER_MAX_RETRIES", "1")),
        "cache_enabled": _bool_env("SEARCH_CACHE_ENABLED", True),
        "cache_resume": _bool_env("SEARCH_CACHE_RESUME", True),
        "cache_dir": os.getenv("SEARCH_CACHE_DIR", ""),
        "cache_file": os.getenv("SEARCH_CACHE_FILE", ""),
        "cache_shards": int(os.getenv("SEARCH_CACHE_SHARDS", "1")),
        "cache_auto_shard": _bool_env("SEARCH_CACHE_AUTO_SHARD", False),
        "cache_auto_merge": _bool_env("SEARCH_CACHE_AUTO_MERGE", False),
        "faiss_enabled": _bool_env("SEARCH_FAISS_ENABLED", True),
        "faiss_similarity_threshold": float(os.getenv("SEARCH_FAISS_SIMILARITY_THRESHOLD", "0.85")),
        "faiss_embedding_model": os.getenv("SEARCH_FAISS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "faiss_device": os.getenv("SEARCH_FAISS_DEVICE", "cuda"),
        "max_workers": int(os.getenv("SEARCH_SERVICE_MAX_WORKERS", "1000")),
        "max_inflight_queries": int(os.getenv("SEARCH_SERVICE_MAX_INFLIGHT_QUERIES", "2000")),
        "serper_reserved_workers": int(os.getenv("SEARCH_SERVICE_SERPER_RESERVED_WORKERS", "32")),
        "queue_wait_timeout_sec": float(os.getenv("SEARCH_SERVICE_QUEUE_WAIT_TIMEOUT_SEC", "90")),
        "request_batch_size": int(os.getenv("SEARCH_SERVICE_REQUEST_BATCH_SIZE", "128")),
        "batch_window_sec": float(os.getenv("SEARCH_SERVICE_BATCH_WINDOW_SEC", "0.05")),
        "max_batch_queue_size": int(os.getenv("SEARCH_SERVICE_MAX_BATCH_QUEUE_SIZE", "0")),
        "faiss_write_queue_size": int(os.getenv("SEARCH_FAISS_WRITE_QUEUE_SIZE", "5000")),  # FAISS main queue capacity; when full, buffer in overflow
        "faiss_write_batch_size": int(os.getenv("SEARCH_FAISS_WRITE_BATCH_SIZE", "512")),  # FAISS batch write size (per worker; internally capped at 512)
        "faiss_write_flush_ms": int(os.getenv("SEARCH_FAISS_WRITE_FLUSH_MS", "30000")),  # Max wait time for FAISS batch writes (ms)
        "faiss_write_retry_max": int(os.getenv("SEARCH_FAISS_WRITE_RETRY_MAX", "-1")),  # -1 means unlimited retries
        "faiss_write_retry_backoff_ms": int(os.getenv("SEARCH_FAISS_WRITE_RETRY_BACKOFF_MS", "100")),
        "faiss_write_overflow_log_every": int(os.getenv("SEARCH_FAISS_WRITE_OVERFLOW_LOG_EVERY", "200")),
        "faiss_write_drain_timeout_sec": float(os.getenv("SEARCH_FAISS_WRITE_DRAIN_TIMEOUT_SEC", "120")),
        "faiss_read_gpus": os.getenv("SEARCH_FAISS_READ_GPUS", "0,1,2"),    # GPUs used by the read path (comma-separated)
        "faiss_read_threads_per_gpu": int(os.getenv("SEARCH_FAISS_READ_THREADS_PER_GPU", "16")),  # Number of read encoding threads per GPU
        "faiss_write_gpus": os.getenv("SEARCH_FAISS_WRITE_GPUS", "3"),  # GPUs used by the write path (comma-separated)
    }
    if not config["cache_dir"]:
        config["cache_dir"] = _DEFAULT_CACHE_DIR
    if not config["cache_file"]:
        config["cache_file"] = os.path.join(config["cache_dir"], "search.db")
    config_path = os.getenv("SEARCH_SERVICE_CONFIG")
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f)
            for t in (data or {}).get("tools") or []:
                c = (t or {}).get("config") or {}
                if "serper_key" in c or "cache_dir" in c:
                    config.update({k: _resolve_oc_env(v) for k, v in c.items() if v is not None})
                    break
        except Exception as e:
            logger.warning("Failed to load SEARCH_SERVICE_CONFIG %s: %s", config_path, e)
    return config


class SearchRequest(BaseModel):
    query: Any = Field(..., description="Single query string or list of query strings")


class SearchSourceItem(BaseModel):
    source: Literal["exact", "faiss", "serper"] = Field(
        ..., description="exact=SQLite exact hit, faiss=semantic similarity hit, serper=live search API"
    )
    similarity: Optional[float] = Field(None, description="Only set when source=faiss; similarity score")


class SearchResponse(BaseModel):
    text: str = Field(..., description="Combined search result text")
    metrics: dict = Field(default_factory=dict, description="query_count, error_count, empty_count, status")
    sources: List[SearchSourceItem] = Field(
        default_factory=list,
        description="Per-query source and similarity (aligned with results order)",
    )


class FaissTopKItem(BaseModel):
    rank: int = Field(..., description="1-based rank")
    cached_query: str = Field(..., description="Query string in cache")
    result: str = Field(..., description="Cached result text")
    similarity: float = Field(..., description="Cosine similarity (IndexFlatIP score)")


class FaissTopKResponse(BaseModel):
    query: str = Field(..., description="Your input query")
    k: int = Field(..., description="Requested k")
    results: List[FaissTopKItem] = Field(default_factory=list, description="Top-k with similarity")


def create_search_app(config: Optional[dict] = None) -> FastAPI:
    _t0 = time.perf_counter()
    if config is None:
        config = _load_config()
    logger.info("[startup] config loaded in %.1fs", time.perf_counter() - _t0)

    # Centralize Serper network params in search_service to avoid long-tail timeouts from defaults.
    os.environ["SERPER_TIMEOUT"] = str(max(1, int(config.get("serper_timeout", 20))))
    os.environ["SERPER_MAX_RETRIES"] = str(max(0, int(config.get("serper_max_retries", 1))))

    cache: Optional[SearchCache] = None
    faiss_retriever: Optional[FaissRetriever] = None
    serper_key = config.get("serper_key", "")
    topk = config.get("topk", 10)

    if config.get("cache_enabled", True):
        _t1 = time.perf_counter()
        cache = SearchCache(
            cache_dir=config["cache_dir"],
            cache_file=config["cache_file"],
            shards=config.get("cache_shards", 1),
            resume=config.get("cache_resume", True),
            auto_shard=config.get("cache_auto_shard", False),
            auto_merge=config.get("cache_auto_merge", False),
            is_leader=True,
        )
        logger.info("[startup] SearchCache in %.1fs", time.perf_counter() - _t1)
        if config.get("faiss_enabled", True) and cache:
            try:
                _t2 = time.perf_counter()
                faiss_retriever = FaissRetriever(
                    cache_dir=config["cache_dir"],
                    embedding_model=config.get("faiss_embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                    similarity_threshold=config.get("faiss_similarity_threshold", 0.85),
                    index_path=config.get("faiss_index_path"),
                    meta_path=config.get("faiss_meta_path"),
                    device=config.get("faiss_device"),
                )
                logger.info("[startup] FaissRetriever (embedding model + index) in %.1fs; threshold=%.2f", time.perf_counter() - _t2, faiss_retriever.threshold)
            except Exception as e:
                logger.warning("FAISS init failed: %s", e)

    def _exact_cache_key(text: str) -> str:
        """For exact cache: remove all whitespace so "OSU NLP" and "OSUNLP" map to the same key."""
        s = (text or "").strip()
        return "".join(s.split()) or s

    # ---- FAISS read path: one encoder + isolated thread pool per GPU, fully isolating read encoding from Serper network threads ----
    # Core design: encoding (GPU-bound), Serper (IO-bound), and FAISS search (CPU-bound) each use separate thread pools,
    # preventing starvation/deadlock under high concurrency when encoding and Serper share one pool.
    import threading as _threading_mod
    import itertools as _itertools_mod
    from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor

    _read_encoders: list = []  # One encoder per GPU
    _read_gpu_executors: list = []  # One isolated thread pool per GPU (isolates encoding threads from Serper)
    _read_encoder_cycle = None  # Round-robin via itertools.cycle
    _read_encoder_lock = _threading_mod.Lock()
    _read_gpu_ids: list = []
    _READ_THREADS_PER_GPU = max(1, int(config.get("faiss_read_threads_per_gpu", 4)))

    # Shared small pool for fast tasks (FAISS index search/cache ops), without consuming Serper/encoding threads
    _misc_executor = _ThreadPoolExecutor(max_workers=16, thread_name_prefix="misc")

    if faiss_retriever is not None:
        _read_gpu_str = str(config.get("faiss_read_gpus", "0,1,2"))
        _read_gpu_ids = [int(x.strip()) for x in _read_gpu_str.split(",") if x.strip()]
        if not _read_gpu_ids:
            _read_gpu_ids = [0]
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            _embed_model = config.get("faiss_embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            # Reuse faiss_retriever's built-in encoder on the first GPU to avoid duplicate loading
            _read_encoders.append(faiss_retriever._encoder)
            _t3 = time.perf_counter()
            for gpu_id in _read_gpu_ids[1:]:
                device = f"cuda:{gpu_id}"
                enc = SentenceTransformer(_embed_model, device=device)
                _read_encoders.append(enc)
            if _read_gpu_ids[1:]:
                logger.info("[startup] read-path extra encoders (GPUs %s) in %.1fs", _read_gpu_ids[1:], time.perf_counter() - _t3)
            _read_encoder_cycle = _itertools_mod.cycle(range(len(_read_encoders)))
            # Create one isolated thread pool per GPU
            for i, gpu_id in enumerate(_read_gpu_ids[:len(_read_encoders)]):
                exe = _ThreadPoolExecutor(
                    max_workers=_READ_THREADS_PER_GPU,
                    thread_name_prefix=f"faiss-read-{gpu_id}",
                )
                _read_gpu_executors.append(exe)
            logger.info(
                "FAISS read encoder pool: %s encoders on GPUs %s, %s threads/GPU (isolated executor per GPU)",
                len(_read_encoders), _read_gpu_ids, _READ_THREADS_PER_GPU,
            )
        except Exception as e:
            logger.warning("FAISS read encoder pool init failed, using single encoder: %s", e)
            _read_encoders = [faiss_retriever._encoder] if faiss_retriever else []
            _read_encoder_cycle = _itertools_mod.cycle(range(max(1, len(_read_encoders))))
            if _read_encoders and not _read_gpu_executors:
                _read_gpu_executors.append(
                    _ThreadPoolExecutor(max_workers=_READ_THREADS_PER_GPU, thread_name_prefix="faiss-read-0")
                )

    def _get_read_gpu_idx() -> int:
        """Get a read-GPU index via round-robin, used for paired access to _read_encoders/_read_gpu_executors."""
        if not _read_encoders:
            return -1
        with _read_encoder_lock:
            idx = next(_read_encoder_cycle)
        return idx

    # ---- FAISS async write path: multi-thread consumers, each pinned to one GPU for embedding; index writes are serialized (short critical section) ----
    import queue as _queue_mod

    _faiss_write_queue_size = config.get("faiss_write_queue_size", 5000)
    _faiss_write_batch_size = min(512, max(1, int(config.get("faiss_write_batch_size", 512))))
    _faiss_write_flush_s = max(0.001, float(config.get("faiss_write_flush_ms", 30000)) / 1000.0)
    _faiss_write_retry_max = int(config.get("faiss_write_retry_max", -1))
    _faiss_write_retry_backoff_s = max(0.0, float(config.get("faiss_write_retry_backoff_ms", 100)) / 1000.0)
    _faiss_write_overflow_log_every = max(1, int(config.get("faiss_write_overflow_log_every", 200)))
    _faiss_write_drain_timeout_sec = max(1.0, float(config.get("faiss_write_drain_timeout_sec", 120.0)))
    _faiss_write_queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=_faiss_write_queue_size)
    _faiss_write_overflow_queue: collections.deque = collections.deque()
    _faiss_write_seen: set = set()
    _faiss_write_seen_lock = _threading_mod.Lock()
    _faiss_write_overflow_lock = _threading_mod.Lock()
    _faiss_write_overflow_event = _threading_mod.Event()
    _faiss_write_overflow_count = 0
    _faiss_write_pump_stop = _threading_mod.Event()
    _faiss_write_pump_thread: Optional[_threading_mod.Thread] = None

    _faiss_write_encoders: list = []
    _faiss_write_threads: list = []
    _faiss_write_active = False

    class SerperClient:
        """Standalone Serper client with aiohttp session, throttling, retries, and background task management."""
        def __init__(self, key: str, max_inflight: int, timeout: int, max_retries: int):
            self.key = key
            self.timeout = timeout
            self.max_retries = max_retries
            self.sem = asyncio.Semaphore(max_inflight)
            self.session: Optional[aiohttp.ClientSession] = None
            self.pending_tasks: set[asyncio.Task] = set()
            self.executor = None # Fallback executor

        async def startup(self):
            if aiohttp:
                conn = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300) # limit=0 means no cap here; semaphore controls concurrency
                timeout_obj = aiohttp.ClientTimeout(total=60)
                self.session = aiohttp.ClientSession(connector=conn, timeout=timeout_obj)
                logger.info("[SerperClient] Async session started (sem=%s)", self.sem._value)
            else:
                logger.warning("[SerperClient] aiohttp not installed, fallback to sync requests")

        async def shutdown(self):
            if self.session:
                await self.session.close()
            if self.pending_tasks:
                logger.info("[SerperClient] Cancelling %s pending tasks...", len(self.pending_tasks))
                for t in self.pending_tasks:
                    t.cancel()
                await asyncio.gather(*list(self.pending_tasks), return_exceptions=True)

        def _track_task(self, task: asyncio.Task):
            self.pending_tasks.add(task)
            task.add_done_callback(self.pending_tasks.discard)

        async def _do_search(self, query: str, topk: int) -> str:
            # Prefer async first
            if self.session:
                return await self._search_async(query, topk)
            # Fallback to sync
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, _search_serper_only, query)

        async def _search_async(self, query: str, topk: int) -> str:
            is_cn = any("\u4E00" <= char <= "\u9FFF" for char in query)
            payload = {
                "q": query,
                "num": topk,
                "hl": "zh-cn" if is_cn else "en",
                "gl": "cn" if is_cn else "us",
                "location": "China" if is_cn else "United States",
            }
            headers = {"X-API-KEY": self.key, "Content-Type": "application/json"}
            url = "https://google.serper.dev/search"

            for attempt in range(self.max_retries + 1):
                try:
                    async with self.session.post(url, json=payload, headers=headers, timeout=self.timeout) as resp:
                        if resp.status >= 400:
                            text = await resp.text()
                            if resp.status >= 500 and attempt < self.max_retries:
                                wait = 0.5 * (attempt + 1)
                                await asyncio.sleep(wait)
                                continue
                            return f"Google search failed: {resp.status} {text}"
                        
                        data = await resp.json()
                        if "organic" not in data:
                             return f"No results found for query: '{query}'"
                        
                        snippets = []
                        for i, item in enumerate(data.get("organic", [])[:topk]):
                            snippet = item.get("snippet", "")
                            title = item.get("title", "")
                            link = item.get("link", "")
                            snippets.append(f"{i+1}. [{title}]({link})\n{snippet}")
                        
                        return f"A Google search for '{query}' found {len(snippets)} results:\n\n## Web Results\n" + "\n\n".join(snippets)
                except Exception as e:
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.5 * (attempt + 1))
                    else:
                        return f"Google search failed: {e}"
            return "Google search failed: Timeout"

        async def submit(self, query: str, fut: asyncio.Future, topk: int = 10):
            """Submit a Serper request task to run in the background and set the future when done."""
            async def _worker():
                if not query:
                     _set_future_result_safe(fut, ("[Tool Error] Empty query", "serper", None, None))
                     return
                
                # Acquire concurrency semaphore
                acquired = False
                try:
                    await asyncio.wait_for(self.sem.acquire(), timeout=60.0)
                    acquired = True
                    result = await self._do_search(query, topk)
                    _set_future_result_safe(fut, (result, "serper", None, None))
                except asyncio.TimeoutError:
                    _set_future_result_safe(fut, ("[Search Error] Serper queue full", "serper", None, None))
                except Exception as e:
                    logger.error("[SerperClient] Internal error: %s", e)
                    _set_future_result_safe(fut, (f"[Search Error] {e}", "serper", None, None))
                finally:
                    if acquired:
                        self.sem.release()

            t = asyncio.create_task(_worker(), name="serper-worker")
            self._track_task(t)

    # Old _aiohttp_session and _google_search_async were removed and replaced by SerperClient
    # _aiohttp_session: Optional["aiohttp.ClientSession"] = None
    # async def _google_search_async... (Deleted)

    def _enqueue_faiss_task(item: tuple[str, str, int]) -> None:
        """Fast path: try in-memory queue; fallback to overflow buffer to avoid drop/block in request path."""
        nonlocal _faiss_write_overflow_count
        try:
            _faiss_write_queue.put_nowait(item)
            return
        except _queue_mod.Full:
            pass
        with _faiss_write_overflow_lock:
            _faiss_write_overflow_queue.append(item)
            _faiss_write_overflow_count += 1
            overflow_size = len(_faiss_write_overflow_queue)
        _faiss_write_overflow_event.set()
        if (
            _faiss_write_overflow_count % _faiss_write_overflow_log_every == 0
            or overflow_size == 1
        ):
            logger.warning(
                "[search] FAISS write queue full, buffering overflow=%s (no drop)",
                overflow_size,
            )

    def _requeue_failed_faiss_item(query: str, result: str, retry: int) -> None:
        next_retry = retry + 1
        if _faiss_write_retry_max >= 0 and next_retry > _faiss_write_retry_max:
            with _faiss_write_seen_lock:
                _faiss_write_seen.discard(query)
            logger.error(
                "[search] FAISS write dropped after retries=%s query=%r",
                _faiss_write_retry_max, query[:80],
            )
            return
        if _faiss_write_retry_backoff_s > 0:
            time.sleep(min(_faiss_write_retry_backoff_s * max(1, next_retry), 1.0))
        _enqueue_faiss_task((query, result, next_retry))

    def _faiss_write_overflow_pump() -> None:
        """Move buffered overflow tasks back into main queue when capacity is available."""
        while not _faiss_write_pump_stop.is_set():
            moved = 0
            while True:
                with _faiss_write_overflow_lock:
                    if not _faiss_write_overflow_queue:
                        _faiss_write_overflow_event.clear()
                        break
                    item = _faiss_write_overflow_queue.popleft()
                    if not _faiss_write_overflow_queue:
                        _faiss_write_overflow_event.clear()
                try:
                    _faiss_write_queue.put_nowait(item)
                    moved += 1
                except _queue_mod.Full:
                    with _faiss_write_overflow_lock:
                        _faiss_write_overflow_queue.appendleft(item)
                        _faiss_write_overflow_event.set()
                    break
            if moved == 0:
                _faiss_write_overflow_event.wait(timeout=0.05)

    def _flush_faiss_batch(
        worker_id: int,
        batch_vecs: list,
        batch_queries: list[str],
        batch_results: list[str],
        batch_retries: list[int],
    ) -> None:
        if not batch_queries:
            return
        import numpy as np

        n = len(batch_queries)
        try:
            vecs = np.vstack(batch_vecs)
            faiss_retriever.add_vectors(vecs, batch_queries, batch_results)
            logger.debug("[search] async cached (FAISS) worker=%s batch=%s", worker_id, n)
            with _faiss_write_seen_lock:
                for q in batch_queries:
                    _faiss_write_seen.discard(q)
        except Exception as e:
            logger.warning(
                "[search] async FAISS batch add failed worker=%s batch=%s: %s",
                worker_id, n, e,
            )
            for i in range(n):
                _requeue_failed_faiss_item(batch_queries[i], batch_results[i], batch_retries[i])
        finally:
            for _ in range(n):
                _faiss_write_queue.task_done()
            batch_vecs.clear()
            batch_queries.clear()
            batch_results.clear()
            batch_retries.clear()

    def _faiss_write_worker(worker_id: int, encoder) -> None:
        """Single write worker: compute embeddings on this thread's bound GPU and batch-submit to reduce lock contention."""
        import numpy as np
        batch_vecs: list = []
        batch_queries: list[str] = []
        batch_results: list[str] = []
        batch_retries: list[int] = []
        while True:
            try:
                item = _faiss_write_queue.get(timeout=_faiss_write_flush_s)
            except _queue_mod.Empty:
                _flush_faiss_batch(worker_id, batch_vecs, batch_queries, batch_results, batch_retries)
                continue
            if item is None:
                _faiss_write_queue.task_done()
                _flush_faiss_batch(worker_id, batch_vecs, batch_queries, batch_results, batch_retries)
                break
            if len(item) == 3:
                q_write, result_write, retry = item
            else:
                q_write, result_write = item
                retry = 0
            try:
                emb = encoder.encode([q_write], normalize_embeddings=True)
                vec = np.asarray(emb, dtype=np.float32)
                if vec.ndim == 1:
                    vec = vec.reshape(1, -1)
                batch_vecs.append(vec)
                batch_queries.append(q_write)
                batch_results.append(result_write)
                batch_retries.append(retry)
                if len(batch_queries) >= _faiss_write_batch_size:
                    _flush_faiss_batch(worker_id, batch_vecs, batch_queries, batch_results, batch_retries)
            except Exception as e:
                logger.warning("[search] async FAISS add failed worker=%s query=%r: %s", worker_id, q_write[:80], e)
                _requeue_failed_faiss_item(q_write, result_write, retry)
                _faiss_write_queue.task_done()

    if faiss_retriever is not None:
        _write_gpu_str = str(config.get("faiss_write_gpus", "3"))
        _write_gpu_ids = [int(x.strip()) for x in _write_gpu_str.split(",") if x.strip()]
        try:
            from sentence_transformers import SentenceTransformer
            _embed_model = config.get("faiss_embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            _t4 = time.perf_counter()
            if not _write_gpu_ids:
                # No write GPU configured -> single worker reuses read encoder
                enc = faiss_retriever._encoder
                _faiss_write_encoders.append(enc)
                t = _threading_mod.Thread(
                    target=_faiss_write_worker, args=(0, enc), daemon=True, name="faiss-write-0",
                )
                t.start()
                _faiss_write_threads.append(t)
                logger.info(
                    "FAISS async write: 1 worker (reuse read encoder), queue maxsize=%s, batch_size=%s, flush_ms=%s retry_max=%s",
                    _faiss_write_queue_size, _faiss_write_batch_size, int(_faiss_write_flush_s * 1000),
                    _faiss_write_retry_max,
                )
            else:
                # Use read GPU list to determine which write GPUs can reuse read encoders
                _read_gpu_set = set(_read_gpu_ids) if _read_gpu_ids else {0}
                for i, gpu_id in enumerate(_write_gpu_ids):
                    if gpu_id in _read_gpu_set and _read_encoders:
                        # If write GPU matches a read GPU, reuse read encoder and avoid reloading
                        enc = _read_encoders[_read_gpu_ids.index(gpu_id)]
                        logger.info("[search] write worker %s reuses read encoder on cuda:%s", i, gpu_id)
                    else:
                        device = f"cuda:{gpu_id}"
                        enc = SentenceTransformer(_embed_model, device=device)
                    _faiss_write_encoders.append(enc)
                    t = _threading_mod.Thread(
                        target=_faiss_write_worker, args=(i, enc), daemon=True, name="faiss-write-%s" % i,
                    )
                    t.start()
                    _faiss_write_threads.append(t)
                logger.info(
                    "FAISS async write: %s workers on GPUs %s, queue maxsize=%s, batch_size=%s, flush_ms=%s retry_max=%s",
                    len(_write_gpu_ids), _write_gpu_ids, _faiss_write_queue_size,
                    _faiss_write_batch_size, int(_faiss_write_flush_s * 1000), _faiss_write_retry_max,
                )
            logger.info("[startup] FAISS write workers in %.1fs", time.perf_counter() - _t4)
            _faiss_write_active = True
            _faiss_write_pump_thread = _threading_mod.Thread(
                target=_faiss_write_overflow_pump,
                daemon=True,
                name="faiss-write-pump",
            )
            _faiss_write_pump_thread.start()
        except Exception as e:
            logger.warning("[search] FAISS write init failed, no async write: %s", e)
            _faiss_write_threads = []
            _faiss_write_encoders = []
            _faiss_write_active = False

    def _submit_faiss_write(query: str, result: str) -> None:
        """Submit one FAISS write task with dedup; if queue is full, send to overflow (no drop). Skip if workers are not active."""
        if not _faiss_write_active:
            return  # Write workers did not start successfully; skip
        with _faiss_write_seen_lock:
            if query in _faiss_write_seen:
                return  # Already queued; skip
            _faiss_write_seen.add(query)
        _enqueue_faiss_task((query, result, 0))

    def _faiss_pending_tasks() -> int:
        unfinished = int(getattr(_faiss_write_queue, "unfinished_tasks", 0))
        with _faiss_write_overflow_lock:
            overflow = len(_faiss_write_overflow_queue)
        return unfinished + overflow

    def search_one_with_source(query: str) -> tuple:
        """Return (result_text, source, similarity or None). Exact cache uses normalized key; FAISS keeps original query for semantics.
        Note: this is a synchronous compatibility path; use _search_one_async in high-concurrency scenarios."""
        q = (query or "").strip()
        if not q:
            return "[Tool Error] Search query cannot be empty.", "serper", None, None
        exact_key = _exact_cache_key(q)
        if cache:
            cached = cache.get(exact_key)
            if cached:
                logger.debug("[search] cache hit (exact) %s", q[:80])
                return cached, "exact", None, None
        if faiss_retriever is not None:
            gpu_idx = _get_read_gpu_idx()
            if gpu_idx >= 0:
                enc = _read_encoders[gpu_idx]
                import numpy as np
                emb = enc.encode([q], normalize_embeddings=True)
                vec = np.asarray(emb, dtype=np.float32)
                if vec.ndim == 1:
                    vec = vec.reshape(1, -1)
                faiss_result, score = faiss_retriever.search_top1_with_vec(vec)
            else:
                faiss_result, score = faiss_retriever.search_top1(q)
            if faiss_result is not None:
                faiss_result = rewrite_faiss_result_header(faiss_result, q)
                logger.debug("[search] cache hit (FAISS) %s score=%.3f", q[:80], score)
                return faiss_result, "faiss", round(score, 6), None
        result = google_search_sync(q, topk, serper_key)
        if cache and result and not _is_empty_search_result(result) and not _is_search_error_result(result):
            cache.set(exact_key, result)
            if faiss_retriever is not None:
                return result, "serper", None, (q, result)
        else:
            if result and (_is_empty_search_result(result) or _is_search_error_result(result)):
                logger.info("[search] not cached (empty/error result) for query=%r", q[:80])
        return result, "serper", None, None

    from concurrent.futures import ThreadPoolExecutor
    max_workers = config.get("max_workers", 300)
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="search")
    max_inflight_queries = max(1, int(config.get("max_inflight_queries", 96)))
    serper_reserved_workers = max(1, int(config.get("serper_reserved_workers", 16)))
    max_serper_inflight = max(1, max_workers - serper_reserved_workers)
    if max_inflight_queries > max_serper_inflight:
        logger.warning(
            "[search] reduce serper inflight %s -> %s to reserve %s worker threads",
            max_inflight_queries, max_serper_inflight, serper_reserved_workers,
        )
    max_inflight_queries = min(max_inflight_queries, max_serper_inflight)
    serper_client = SerperClient(
        key=serper_key,
        max_inflight=max_inflight_queries,
        timeout=int(config.get("serper_timeout", 20)),
        max_retries=int(config.get("serper_max_retries", 1)),
    )
    serper_client.executor = executor # For sync fallback

    queue_wait_timeout_sec = max(0.1, float(config.get("queue_wait_timeout_sec", 5.0)))
    request_batch_size = max(1, int(config.get("request_batch_size", 8)))
    batch_window_sec = max(0.0, float(config.get("batch_window_sec", 0.0)))
    max_batch_queue_size = int(config.get("max_batch_queue_size", 0))
    if max_batch_queue_size <= 0:
        max_batch_queue_size = max_inflight_queries * 10
    max_batch_queue_size = max(max_inflight_queries, max_batch_queue_size)
    
    # Old global semaphore removed; now managed inside SerperClient
    # inflight_gate = asyncio.Semaphore(max_inflight_queries)
    
    batch_queue: asyncio.Queue = asyncio.Queue(maxsize=max_batch_queue_size)
    batch_processing_sem = asyncio.Semaphore(max(1, int(os.getenv("SEARCH_SERVICE_BATCH_CONCURRENCY", "64"))))
    batch_dispatcher_task: Optional[asyncio.Task] = None
    
    # Old pending_serper_tasks removed; now managed inside SerperClient
    # pending_serper_tasks: set[asyncio.Task] = set()

    def _search_serper_only(query: str) -> tuple:
        q = (query or "").strip()
        if not q:
            return "[Tool Error] Search query cannot be empty.", "serper", None, None
        exact_key = _exact_cache_key(q)
        result = google_search_sync(q, topk, serper_key)
        if cache and result and not _is_empty_search_result(result) and not _is_search_error_result(result):
            cache.set(exact_key, result)
            if faiss_retriever is not None:
                return result, "serper", None, (q, result)
        elif result and (_is_empty_search_result(result) or _is_search_error_result(result)):
            logger.info("[search] not cached (empty/error result) for query=%r", q[:80])
        return result, "serper", None, None

    async def _run_serper_with_gate(query: str):
        q = (query or "").strip()
        wait_t0 = time.perf_counter()
        acquired = False
        try:
            await asyncio.wait_for(serper_client.sem.acquire(), timeout=queue_wait_timeout_sec)
            acquired = True
        except asyncio.TimeoutError:
            logger.warning(
                "[search] overloaded: serper gate wait > %.1fs query=%r",
                queue_wait_timeout_sec, q[:80],
            )
            return (
                "[Search Error] Search service overloaded (queue wait timeout), please retry.",
                "serper",
                None,
                None,
            )
        waited = time.perf_counter() - wait_t0
        if waited > 1.0:
            logger.warning("[search] serper wait %.2fs query=%r", waited, q[:80])
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, _search_serper_only, q)
        finally:
            if acquired:
                serper_client.sem.release()

    def _set_future_result_safe(fut: asyncio.Future, result: tuple) -> None:
        if fut.cancelled() or fut.done():
            return
        fut.set_result(result)

    async def _resolve_serper_for_future(query: str, fut: asyncio.Future) -> None:
        try:
            result = await _run_serper_with_gate(query)
        except Exception as e:
            logger.warning("[search] batch serper failed query=%r: %s", (query or "")[:80], e)
            result = (
                "[Search Error] Search service internal error, please retry.",
                "serper",
                None,
                None,
            )
        _set_future_result_safe(fut, result)

    def _encode_shard_with_encoder(encoder, shard_indices: list[int], shard_queries: list[str]):
        import numpy as np

        emb = encoder.encode(shard_queries, normalize_embeddings=True)
        vecs = np.asarray(emb, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        return shard_indices, vecs

    def _encode_single_query(encoder, query: str):
        """Encode a single query, used by per-GPU executors."""
        import numpy as np
        emb = encoder.encode([query], normalize_embeddings=True)
        vec = np.asarray(emb, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        return vec

    async def _encode_queries_multi_gpu(queries: list[str]):
        """Multi-GPU parallel encoding: each GPU uses an isolated thread pool, without competing with Serper threads."""
        if not queries:
            return None
        if faiss_retriever is None:
            return None

        encoder_pool = _read_encoders if _read_encoders else [faiss_retriever._encoder]
        if not encoder_pool:
            return None

        loop = asyncio.get_running_loop()
        if len(encoder_pool) == 1 or len(queries) == 1:
            gpu_exe = _read_gpu_executors[0] if _read_gpu_executors else None
            _, vecs = await loop.run_in_executor(
                gpu_exe,
                _encode_shard_with_encoder,
                encoder_pool[0],
                list(range(len(queries))),
                queries,
            )
            return vecs

        shard_count = len(encoder_pool)
        shards: list[list[tuple[int, str]]] = [[] for _ in range(shard_count)]
        for i, q in enumerate(queries):
            shards[i % shard_count].append((i, q))

        jobs = []
        for enc_idx, shard in enumerate(shards):
            if not shard:
                continue
            shard_indices = [idx for idx, _ in shard]
            shard_queries = [q for _, q in shard]
            gpu_exe = _read_gpu_executors[enc_idx] if enc_idx < len(_read_gpu_executors) else None
            jobs.append(
                loop.run_in_executor(
                    gpu_exe,
                    _encode_shard_with_encoder,
                    encoder_pool[enc_idx],
                    shard_indices,
                    shard_queries,
                )
            )

        parts = await asyncio.gather(*jobs)
        import numpy as np

        ordered_vecs = [None] * len(queries)
        for idxs, vecs in parts:
            for row_idx, orig_idx in enumerate(idxs):
                ordered_vecs[orig_idx] = vecs[row_idx]
        return np.vstack(ordered_vecs)

    async def _process_query_batch(items: list[tuple[str, asyncio.Future]]) -> None:
        if not items:
            return

        pending_queries: list[str] = []
        pending_futures: list[asyncio.Future] = []

        for query, fut in items:
            q = (query or "").strip()
            if not q:
                _set_future_result_safe(fut, ("[Tool Error] Search query cannot be empty.", "serper", None, None))
                continue
            exact_key = _exact_cache_key(q)
            if cache:
                cached = cache.get(exact_key)
                if cached:
                    _set_future_result_safe(fut, (cached, "exact", None, None))
                    continue
            pending_queries.append(q)
            pending_futures.append(fut)

        if not pending_queries:
            return

        serper_queries: list[str] = []
        serper_futures: list[asyncio.Future] = []

        if pending_queries and faiss_retriever is not None:
            faiss_hits: list[tuple[Optional[str], float]]
            try:
                vecs = await _encode_queries_multi_gpu(pending_queries)
                if vecs is None:
                    faiss_hits = [(None, 0.0)] * len(pending_queries)
                else:
                    loop = asyncio.get_running_loop()
                    faiss_hits = await loop.run_in_executor(_misc_executor, faiss_retriever.search_top1_with_vecs, vecs)
            except Exception as e:
                logger.warning("[search] batch FAISS lookup failed (%s queries): %s", len(pending_queries), e)
                faiss_hits = [(None, 0.0)] * len(pending_queries)

            for idx, (q, fut) in enumerate(zip(pending_queries, pending_futures, strict=True)):
                faiss_result, score = faiss_hits[idx]
                if faiss_result is not None:
                    faiss_result = rewrite_faiss_result_header(faiss_result, q)
                    _set_future_result_safe(fut, (faiss_result, "faiss", round(score, 6), None))
                else:
                    serper_queries.append(q)
                    serper_futures.append(fut)
        else:
            serper_queries.extend(pending_queries)
            serper_futures.extend(pending_futures)

        # Key point: Serper misses are moved to SerperClient's background queue for full decoupling
        for q, fut in zip(serper_queries, serper_futures, strict=True):
            await serper_client.submit(q, fut, topk)

    async def _batch_dispatcher() -> None:
        stop = False
        while not stop:
            item = await batch_queue.get()
            if item is None:
                batch_queue.task_done()
                break

            batch: list[tuple[str, asyncio.Future]] = [item]
            deadline = time.perf_counter() + batch_window_sec
            while len(batch) < request_batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(batch_queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                if next_item is None:
                    batch_queue.task_done()
                    stop = True
                    break
                batch.append(next_item)

            # Submit batch processing asynchronously so dispatcher can collect the next batch immediately (pipelined concurrency)
            # Use semaphore to cap concurrent batches and avoid unbounded GPU task buildup (OOM/cascading failure), while preserving batch_queue backpressure
            await batch_processing_sem.acquire()
            
            async def _process_and_ack(items: list):
                try:
                    await _process_query_batch(items)
                except Exception as e:
                    logger.exception("[search] batch processing task failed")
                    # Safety net: if processing crashes, ensure futures never hang forever
                    for _, fut in items:
                        if not fut.done():
                            _set_future_result_safe(fut, ("[Search Error] Internal batch error", "serper", None, None))
                finally:
                    batch_processing_sem.release()
                    for _ in items:
                        batch_queue.task_done()

            asyncio.create_task(_process_and_ack(batch))

    logger.info(
        "Search service pool: max_workers=%s serper_inflight=%s reserved_workers=%s queue_wait_timeout=%.1fs batch_size=%s batch_window=%.1fs batch_queue_max=%s",
        max_workers, max_inflight_queries, serper_reserved_workers, queue_wait_timeout_sec, request_batch_size, batch_window_sec, max_batch_queue_size,
    )

    app = FastAPI(title="DeepResearch Search Service", version="0.1.0")
    app.state.executor = executor

    @app.on_event("startup")
    async def startup():
        await serper_client.startup()
        nonlocal batch_dispatcher_task
        if batch_window_sec > 0:
            batch_dispatcher_task = asyncio.create_task(_batch_dispatcher(), name="search-batch-dispatcher")
            logger.info("[search] request batching enabled: window=%.1fs", batch_window_sec)

    @app.on_event("shutdown")
    async def shutdown():
        await serper_client.shutdown()
        
        if batch_dispatcher_task is not None:
            await batch_queue.put(None)
            await batch_queue.join()
            await batch_dispatcher_task
        executor.shutdown(wait=False)
        _misc_executor.shutdown(wait=False)
        for _gpu_exe in _read_gpu_executors:
            _gpu_exe.shutdown(wait=False)
        # On normal shutdown, try to drain FAISS async write queues to avoid write loss.
        if _faiss_write_active:
            deadline = time.perf_counter() + _faiss_write_drain_timeout_sec
            while time.perf_counter() < deadline:
                pending = _faiss_pending_tasks()
                if pending == 0:
                    break
                await asyncio.sleep(0.2)
            pending = _faiss_pending_tasks()
            if pending > 0:
                logger.warning(
                    "[search] FAISS pending writes remain after %.1fs drain timeout: %s",
                    _faiss_write_drain_timeout_sec, pending,
                )
            _faiss_write_pump_stop.set()
            _faiss_write_overflow_event.set()
            if _faiss_write_pump_thread is not None:
                _faiss_write_pump_thread.join(timeout=5)
        # Stop FAISS write threads: one sentinel per worker (with timeout to avoid blocking), then join
        for _ in _faiss_write_threads:
            try:
                _faiss_write_queue.put(None, timeout=5)
            except _queue_mod.Full:
                logger.warning("[search] FAISS write queue full during shutdown, skipping sentinel")
        for t in _faiss_write_threads:
            t.join(timeout=30)
        logger.info("FAISS write threads stopped (queue remaining: %s)", _faiss_write_queue.qsize())

    @app.get("/health")
    async def health():
        return {"status": "ok", "cache": cache is not None, "faiss": faiss_retriever is not None}

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest):
        t0 = time.perf_counter()
        q = req.query
        queries: List[str] = [q] if isinstance(q, str) else list(q)
        if not queries:
            raise HTTPException(status_code=400, detail="query cannot be empty")
        loop = asyncio.get_running_loop()
        executor = app.state.executor

        async def _search_one_async(query: str) -> tuple:
            """Async search: encoding runs in isolated per-GPU pools, Serper uses the main pool, avoiding cross-starvation/deadlock."""
            s = (query or "").strip()
            if not s:
                return "[Tool Error] Search query cannot be empty.", "serper", None, None
            exact_key = _exact_cache_key(s)
            if cache:
                cached = cache.get(exact_key)
                if cached:
                    return cached, "exact", None, None
            if faiss_retriever is not None and _read_gpu_executors:
                gpu_idx = _get_read_gpu_idx()
                if gpu_idx >= 0:
                    vec = await loop.run_in_executor(
                        _read_gpu_executors[gpu_idx],
                        _encode_single_query, _read_encoders[gpu_idx], s,
                    )
                    faiss_result, score = faiss_retriever.search_top1_with_vec(vec)
                    if faiss_result is not None:
                        faiss_result = rewrite_faiss_result_header(faiss_result, s)
                        return faiss_result, "faiss", round(score, 6), None
            # FAISS miss -> Serper (main thread pool + inflight_gate throttling)
            return await _run_serper_with_gate(s)

        async def one(query: str):
            s = (query or "").strip()
            if not s:
                return "[Tool Error] Search query cannot be empty.", "serper", None, None
            if batch_window_sec > 0:
                fut = loop.create_future()
                try:
                    await asyncio.wait_for(batch_queue.put((s, fut)), timeout=queue_wait_timeout_sec)
                except asyncio.TimeoutError:
                    logger.warning("[search] overloaded: batch queue wait > %.1fs query=%r", queue_wait_timeout_sec, s[:80])
                    return (
                        "[Search Error] Search service overloaded (queue wait timeout), please retry.",
                        "serper",
                        None,
                        None,
                    )
                return await fut

            # Non-batch mode: use async path; encoding runs in isolated per-GPU pools, not competing with Serper
            return await _search_one_async(s)

        # Note: do NOT wrap run_in_executor with asyncio.wait_for on the server side.
        # Python ThreadPoolExecutor threads cannot be cancelled; wait_for timeout only returns 504 to the client,
        # but worker threads keep running -> client retries -> more submissions -> thread-pool meltdown.
        # Keep timeout protection on the client side via search_service_timeout.
        if batch_window_sec > 0:
            # In batch mode, submit all queries in this request at once
            # to avoid head-of-line blocking caused by request_batch_size sequential waits.
            out = await asyncio.gather(*[one(x) for x in queries])
        else:
            out = []
            for i in range(0, len(queries), request_batch_size):
                batch = queries[i:i + request_batch_size]
                out.extend(await asyncio.gather(*[one(x) for x in batch]))
        
        # Execute FAISS writes asynchronously (multi-thread encoding + batched commits) without blocking response
        for t in out:
            if len(t) > 3 and t[3] is not None:
                q_write, result_write = t[3]
                _submit_faiss_write(q_write, result_write)
        
        results = [t[0] for t in out]
        sources = [SearchSourceItem(source=t[1], similarity=t[2]) for t in out]
        combined = "\n=======\n".join(results)
        error_count = sum(1 for r in results if _is_search_error_result(r))
        empty_count = sum(1 for r in results if _is_empty_search_result(r))
        overload_count = sum(1 for r in results if "overloaded (queue wait timeout)" in r)
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
            "overload_count": overload_count,
            "status": status,
        }
        elapsed = time.perf_counter() - t0
        if elapsed > 60:
            logger.warning("[search] slow request: %.1fs for %s queries", elapsed, len(queries))
        else:
            logger.info("[search] request: %.1fs for %s queries", elapsed, len(queries))
        return SearchResponse(text=combined, metrics=metrics, sources=sources)

    @app.get("/search_faiss_topk", response_model=FaissTopKResponse)
    async def search_faiss_topk(query: str = "", k: int = 10):
        """Test endpoint: return top-k and similarity from FAISS semantic cache only."""
        if not query or not query.strip():
            raise HTTPException(status_code=400, detail="query cannot be empty")
        if faiss_retriever is None:
            raise HTTPException(status_code=503, detail="FAISS retriever not enabled")
        k = max(1, min(k, 100))
        loop = asyncio.get_running_loop()
        rows = await loop.run_in_executor(
            _misc_executor,
            lambda: faiss_retriever.search_topk(query.strip(), k=k),
        )
        items = [
            FaissTopKItem(rank=i + 1, cached_query=q, result=r, similarity=round(s, 6))
            for i, (q, r, s) in enumerate(rows)
        ]
        return FaissTopKResponse(query=query.strip(), k=k, results=items)

    logger.info("[startup] create_search_app total %.1fs", time.perf_counter() - _t0)
    return app


def main():
    import argparse
    parser = argparse.ArgumentParser(description="DeepResearch Search HTTP Service")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--config", default=None, help="Path to tools.yaml (overrides env)")
    args = parser.parse_args()
    if args.config:
        os.environ["SEARCH_SERVICE_CONFIG"] = args.config
    app = create_search_app()
    logger.info("Starting search service on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
