import json
import re
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
BLOCK_HUGGINGFACE = os.getenv("BLOCK_HUGGINGFACE", "false").lower() == "true"
if BLOCK_HUGGINGFACE:
    print("[Warining] BLOCK_HUGGINGFACE is enabled.")

# BrowseComp-Plus offline retrieval configuration.
BM25_INDEX_PATH = os.environ.get('BM25_INDEX_PATH', '')
BM25_TOP_K = int(os.environ.get('BM25_TOP_K', '10'))
FAISS_INDEX_PATH = os.environ.get('FAISS_INDEX_PATH', '')
_bm25_searcher = None
_bm25_lock = threading.Lock()


def _get_bm25_searcher():
    """Lazy-init a shared LuceneSearcher for BM25 retrieval."""
    global _bm25_searcher
    if _bm25_searcher is None:
        with _bm25_lock:
            if _bm25_searcher is None:
                from pyserini.search.lucene import LuceneSearcher
                _bm25_searcher = LuceneSearcher(BM25_INDEX_PATH)
                print(f"[search] BM25 searcher initialized: {BM25_INDEX_PATH}")
    return _bm25_searcher

# Cache configuration
_shared_cache_dir = os.getenv(
    "SHARED_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"),
)
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

    def _filter_huggingface_from_result(self, result: str) -> str:
        """Filter out huggingface.co URLs from the formatted search result string."""
        if not result or result.startswith("No results found") or result.startswith("Google search Timeout"):
            return result

        # Detect old format (contains "## Web Results")
        if "## Web Results" in result:
            return self._filter_huggingface_from_result_old_format(result)

        # New format handling
        # Split header and entries
        parts = result.split("\n\n", 1)
        if len(parts) < 2:
            return result

        header = parts[0]
        entries_text = parts[1]

        # Split into individual entries (each entry starts with "Title:")
        entries = entries_text.split("\n\nTitle: ")
        filtered_entries = []

        for i, entry in enumerate(entries):
            # Add back "Title: " prefix for entries after the first one
            if i > 0:
                entry = "Title: " + entry

            # Check if the Link contains huggingface.co
            lines = entry.split("\n")
            has_huggingface = False
            filtered_lines = []
            in_sitelinks = False

            for line in lines:
                if line.startswith("Link: "):
                    if "huggingface.co" in line.lower():
                        has_huggingface = True
                        print("[search] filter huggingface.co")
                        break
                if line.startswith("Sitelinks:"):
                    in_sitelinks = True
                    filtered_lines.append(line)
                    continue
                if in_sitelinks and line.startswith("- "):
                    # Filter sitelinks containing huggingface.co
                    if "huggingface.co" not in line.lower():
                        filtered_lines.append(line)
                else:
                    in_sitelinks = False
                    filtered_lines.append(line)

            if not has_huggingface:
                filtered_entries.append("\n".join(filtered_lines))

        if not filtered_entries:
            # Extract query from header
            match = re.search(r"A Google search for '(.+?)' found", header)
            query = match.group(1) if match else "unknown"
            return f"A Google search for '{query}' found 0 results:\n\n"

        # Rebuild the result with updated count
        match = re.search(r"A Google search for '(.+?)' found", header)
        query = match.group(1) if match else "unknown"
        new_header = f"A Google search for '{query}' found {len(filtered_entries)} results:"

        return new_header + "\n\n" + "\n\n".join(filtered_entries)

    def _filter_huggingface_from_result_old_format(self, result: str) -> str:
        """Filter out huggingface.co URLs from the old format search result string.

        Old format example:
        A Google search for 'query' found N results:

        ## Web Results
        1. [Title](Link)
        Date published: ...
        Source: ...
        snippet text

        2. [Title](Link)
        ...
        """
        # Split by "## Web Results"
        parts = result.split("## Web Results\n", 1)
        if len(parts) < 2:
            return result

        header_part = parts[0].strip()
        entries_text = parts[1]

        # Split entries by the pattern of numbered items (e.g., "1. [", "2. [")
        # Each entry starts with a number followed by ". ["
        entry_pattern = re.compile(r'\n(?=\d+\. \[)')
        entries = entry_pattern.split(entries_text)
        filtered_entries = []
        new_idx = 0

        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue

            # Extract URL from markdown link format: [Title](URL)
            url_match = re.search(r'\]\((https?://[^\)]+)\)', entry)
            if url_match:
                url = url_match.group(1).lower()
                if "huggingface.co" in url:
                    print("[search] filter huggingface.co (old format)")
                    continue

            # Re-number the entry
            new_idx += 1
            # Replace the original number with new number
            renumbered_entry = re.sub(r'^\d+\.', f'{new_idx}.', entry)
            filtered_entries.append(renumbered_entry)

        if not filtered_entries:
            # Extract query from header
            match = re.search(r"A Google search for '(.+?)' found", header_part)
            query = match.group(1) if match else "unknown"
            return f"A Google search for '{query}' found 0 results:\n\n## Web Results\n"

        # Rebuild the result with updated count
        match = re.search(r"A Google search for '(.+?)' found", header_part)
        query = match.group(1) if match else "unknown"
        new_header = f"A Google search for '{query}' found {len(filtered_entries)} results:\n\n## Web Results"

        return new_header + "\n" + "\n\n".join(filtered_entries)

    def search_with_serp(self, query: str):
        if self.cache:
            cached_result = self.cache.get(query)
            if cached_result:
                print(f"[search] Cache hit for query: {query}")
                # Filter huggingface.co URLs from cached result if blocking is enabled
                if BLOCK_HUGGINGFACE:
                    return self._filter_huggingface_from_result(cached_result)
                return cached_result

        result = self.google_search_with_serp(query)

        # Cache the unfiltered result
        if self.cache and result and not result.startswith("No results found") and not result.startswith("Google search Timeout"):
            self.cache.set(query, result)

        # Filter huggingface.co URLs from result if blocking is enabled
        if BLOCK_HUGGINGFACE:
            return self._filter_huggingface_from_result(result)

        return result

    def search_with_bm25(self, query: str):
        """Search using a local BM25 index for BrowseComp-Plus static corpus."""
        import json as _json
        searcher = _get_bm25_searcher()
        hits = searcher.search(query, BM25_TOP_K)

        if not hits:
            return f"No results found for query: '{query}'. Use a less specific query."

        if not hasattr(self, '_snippet_tokenizer'):
            from transformers import AutoTokenizer
            self._snippet_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

        snippet_max_tokens = int(os.environ.get('BM25_SNIPPET_MAX_TOKENS', '512'))

        web_snippets = []
        for hit in hits:
            raw = _json.loads(hit.lucene_document.get("raw"))
            text = raw.get("contents", "")
            tokens = self._snippet_tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > snippet_max_tokens:
                text = self._snippet_tokenizer.decode(tokens[:snippet_max_tokens], skip_special_tokens=True)
            web_snippets.append(
                f"Title: Document {hit.docid}\n"
                f"Link: bm25://{hit.docid}\n"
                f"Score: {hit.score:.4f}\n"
                f"Snipptes: {text}"
            )

        return f"A search for '{query}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        query = params['query']
        if not query:
            return "[Tool Error] Search query cannot be empty."

        if FAISS_INDEX_PATH:
            from search_faiss_bridge import faiss_search
            search_fn = faiss_search
        elif BM25_INDEX_PATH:
            search_fn = self.search_with_bm25
        else:
            search_fn = self.search_with_serp

        if isinstance(query, str):
            response = search_fn(query)
        else:
            queries = query if isinstance(query, List) else [query]
            responses = []
            for q in queries:
                responses.append(search_fn(q))
            response = "\n\n".join(responses)
        # print("[DEBUG]", response)
        return response
