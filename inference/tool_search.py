import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import requests
from qwen_agent.tools.base import BaseTool, register_tool
import asyncio
from typing import Dict, List, Optional, Union
import uuid
import http.client
import json
import sqlite3
import time
import os
from contextlib import contextmanager
import threading
import atexit
import hashlib
import fcntl

SERPER_KEY = os.environ.get('SERPER_KEY_ID')

# Cache configuration
_shared_cache_dir = "/fs/ess/PAA0201/jianxie/database_only_for_eval"
_default_cache_dir = os.getenv("CACHE_DIR", _shared_cache_dir)
os.makedirs(_default_cache_dir, exist_ok=True)
_default_search_cache_file = os.path.join(_default_cache_dir, "search_cache_merged.db")
SEARCH_CACHE_FILE = os.getenv("SEARCH_CACHE_FILE", _default_search_cache_file)
SEARCH_CACHE_SHARD_DIR = os.getenv("SEARCH_CACHE_SHARD_DIR", _default_cache_dir)
SEARCH_CACHE_SHARED_FILE = os.getenv("SEARCH_CACHE_SHARED_FILE", os.path.join(_shared_cache_dir, "search_cache_merged.db"))
SEARCH_CACHE_ENABLED = os.getenv("SEARCH_CACHE_ENABLED", "true").lower() == "true"
SEARCH_CACHE_RESUME = os.getenv("SEARCH_CACHE_RESUME", "true").lower() == "true"
SEARCH_CACHE_SHARDS = max(1, int(os.getenv("SEARCH_CACHE_SHARDS", "32")))
SEARCH_CACHE_AUTO_MERGE = os.getenv("SEARCH_CACHE_AUTO_MERGE", "true").lower() == "true"


def _is_cache_merge_leader() -> bool:
    for env in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        val = os.getenv(env)
        if val is not None:
            try:
                return int(val) == 0
            except ValueError:
                return False
    return True


class SearchCache:
    """Search tool cache backed by sharded SQLite databases."""

    def __init__(self, cache_file: str = SEARCH_CACHE_FILE, resume: bool = True, shards: int = SEARCH_CACHE_SHARDS):
        self.cache_file = cache_file
        self.resume = resume
        self.shards = max(1, int(shards))
        self.auto_merge = SEARCH_CACHE_AUTO_MERGE
        self.is_merge_leader = _is_cache_merge_leader()
        self.shard_dir = SEARCH_CACHE_SHARD_DIR
        os.makedirs(self.shard_dir, exist_ok=True)
        self._master_lock = threading.Lock()
        self._conns: dict[int, sqlite3.Connection] = {}
        self._locks: dict[int, threading.Lock] = {}

        base_name = os.path.splitext(os.path.basename(self.cache_file))[0]
        self._shard_files = [os.path.join(self.shard_dir, f"{base_name}_shard{idx}.db") for idx in range(self.shards)]
        self._master_read_conn = None
        self._user_read_conn = None
        if self.resume:
            shared = SEARCH_CACHE_SHARED_FILE
            if shared and os.path.exists(shared):
                self._master_read_conn = self._open_readonly_conn(shared)
            if os.path.exists(self.cache_file) and self.cache_file != SEARCH_CACHE_SHARED_FILE:
                self._user_read_conn = self._open_readonly_conn(self.cache_file)

        if self.shards == 1:
            self._get_conn(0)

        if self.shards > 1 and self.auto_merge and self.is_merge_leader:
            atexit.register(self.merge_shards)
        atexit.register(self.close)

    def close(self):
        if self._master_read_conn:
            try:
                self._master_read_conn.close()
            except Exception:
                pass
        if self._user_read_conn:
            try:
                self._user_read_conn.close()
            except Exception:
                pass
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

    def _open_readonly_conn(self, path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                query TEXT PRIMARY KEY,
                result TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        conn.commit()

    def _get_conn(self, shard_id: int) -> sqlite3.Connection:
        if shard_id not in self._conns:
            path = self.cache_file if self.shards == 1 else self._shard_files[shard_id]
            self._conns[shard_id] = self._open_conn(path)
            self._locks[shard_id] = threading.Lock()
        return self._conns[shard_id]

    def _shard_id(self, query: str) -> int:
        if self.shards == 1:
            return 0
        digest = hashlib.md5(query.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.shards

    def merge_shards(self) -> None:
        if self.shards <= 1:
            return
        try:
            lock_path = f"{self.cache_file}.merge.lock"
            os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                master_conn = self._open_conn(self.cache_file)
                master_cursor = master_conn.cursor()
                for shard_file in self._shard_files:
                    if not os.path.exists(shard_file):
                        continue
                    shard_conn = sqlite3.connect(shard_file, timeout=60.0, check_same_thread=False)
                    shard_conn.row_factory = sqlite3.Row
                    shard_cursor = shard_conn.cursor()
                    shard_cursor.execute("SELECT query, result, timestamp FROM search_cache")
                    for row in shard_cursor.fetchall():
                        master_cursor.execute("""
                            INSERT OR REPLACE INTO search_cache
                            (query, result, timestamp)
                            VALUES (?, ?, ?)
                        """, (row["query"], row["result"], row["timestamp"]))
                    master_conn.commit()
                    shard_conn.close()
                master_conn.close()
        except Exception as e:
            print(f"[SearchCache] Error merging shard cache into '{self.cache_file}': {e}")

    def get(self, query: str) -> Optional[str]:
        if not SEARCH_CACHE_ENABLED:
            return None

        try:
            shard_id = self._shard_id(query)
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT result FROM search_cache
                    WHERE query = ?
                """, (query,))
                row = cursor.fetchone()
            if row:
                return row['result']
            if not row and self._user_read_conn and self.shards > 1:
                with self._master_lock:
                    cursor = self._user_read_conn.cursor()
                    cursor.execute("""
                        SELECT result FROM search_cache
                        WHERE query = ?
                    """, (query,))
                    row = cursor.fetchone()
            if not row and self._master_read_conn and self.shards > 1:
                with self._master_lock:
                    cursor = self._master_read_conn.cursor()
                    cursor.execute("""
                        SELECT result FROM search_cache
                        WHERE query = ?
                    """, (query,))
                    row = cursor.fetchone()
            return row['result'] if row else None
        except Exception as e:
            print(f"[SearchCache] Error getting result for query '{query}': {e}")
            return None

    def set(self, query: str, result: str):
        if not SEARCH_CACHE_ENABLED:
            return

        try:
            current_time = time.time()
            shard_id = self._shard_id(query)
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO search_cache
                    (query, result, timestamp)
                    VALUES (?, ?, ?)
                """, (query, result, current_time))
                conn.commit()
        except Exception as e:
            print(f"[SearchCache] Error writing cache for query '{query}': {e}")


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.cache = SearchCache(resume=SEARCH_CACHE_RESUME, shards=SEARCH_CACHE_SHARDS) if SEARCH_CACHE_ENABLED else None

    def google_search_with_serp(self, query: str):
        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)
        conn = http.client.HTTPSConnection("google.serper.dev")
        if contains_chinese_basic(query):
            gl = "cn"
            hl = "zh-cn"
        else:
            gl = "us"
            hl = "en"
        payload = json.dumps({
        "q": query,
        "gl": gl,
        "hl": hl
        })
        headers = {
        'X-API-KEY': SERPER_KEY,
        'Content-Type': 'application/json'
        }
        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return f"Google search Timeout, return None, Please try again later."
                continue
        data = res.read()
        results = json.loads(data.decode("utf-8"))
        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\nSnipptes: " + page["snippet"]

                    site_links = ""
                    if "sitelinks" in page:
                        site_links = "\nSitelinks:\n"
                        for sitelink in page["sitelinks"]:
                            if "title" in sitelink and "link" in sitelink:
                                site_links += f"- {sitelink['title']}: {sitelink['link']}\n"

                    web_snippets.append(
                        "Title: " + page["title"] + "\n"
                        "Link: " + page["link"] + date_published + snippet + site_links)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
            return content
        except Exception as e:
            return f"No results found for query: '{query}'. Error: {str(e)}"

    def search_with_serp(self, query: str):
        if self.cache:
            cached_result = self.cache.get(query)
            if cached_result:
                print(f"[search] Cache hit for query: {query}")
                return cached_result

        result = self.google_search_with_serp(query)

        if self.cache and result and not result.startswith("No results found") and not result.startswith("Google search Timeout"):
            self.cache.set(query, result)

        return result

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        query = params['query']
        if not query:
            return "[Tool Error] Search query cannot be empty."

        if isinstance(query, str):
            response = self.search_with_serp(query)
        else:
            queries = query if isinstance(query, List) else [query]
            responses = []
            for q in queries:
                responses.append(self.search_with_serp(q))
            response = "\n\n".join(responses)

        return response
