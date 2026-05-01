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

SERPER_KEY=os.environ.get('SERPER_KEY_ID')
SEARCH_CACHE_FILE = os.getenv("SEARCH_CACHE_FILE", "search_cache.db")
SEARCH_CACHE_ENABLED = os.getenv("SEARCH_CACHE_ENABLED", "true").lower() == "true"
SEARCH_CACHE_RESUME = os.getenv("SEARCH_CACHE_RESUME", "true").lower() == "true"


class SearchCache:
    """Documentation omitted."""
    
    def __init__(self, cache_file: str = SEARCH_CACHE_FILE, resume: bool = True):
        """Documentation omitted."""
        self.cache_file = cache_file
        self.resume = resume
        self._lock = threading.Lock()
        db_exists = os.path.exists(cache_file)
        self._conn = sqlite3.connect(cache_file, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-8000")
        self._conn.commit()
        if resume and db_exists:
            try:
                cursor = self._conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='search_cache'
                """)
                if cursor.fetchone() is None:
                    print(f"[SearchCache] Table missing, initializing...")
                    self._init_database()
            except Exception as e:
                print(f"[SearchCache] Database validation failed: {e}, reinitializing...")
                self._init_database()
        else:
            self._init_database()
        atexit.register(self.close)
    
    def close(self):
        """Documentation omitted."""
        if hasattr(self, '_conn') and self._conn:
            try:
                self._conn.close()
            except:
                pass
    
    def _execute_read(self, query: str, params: tuple = ()):
        """Documentation omitted."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()
    
    def _execute_write(self, query: str, params: tuple = ()):
        """Documentation omitted."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(query, params)
            self._conn.commit()
    
    def _init_database(self):
        """Documentation omitted."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    query TEXT PRIMARY KEY,
                    result TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            self._conn.commit()
    
    def get(self, query: str) -> Optional[str]:
        """Documentation omitted."""
        if not SEARCH_CACHE_ENABLED:
            return None
        
        try:
            row = self._execute_read("""
                SELECT result FROM search_cache 
                WHERE query = ?
            """, (query,))
            return row['result'] if row else None
        except Exception as e:
            print(f"[SearchCache] Error getting result for query '{query}': {e}")
            return None
    
    def set(self, query: str, result: str):
        """Documentation omitted."""
        if not SEARCH_CACHE_ENABLED:
            return
        
        try:
            current_time = time.time()
            self._execute_write("""
                INSERT OR REPLACE INTO search_cache 
                (query, result, timestamp)
                VALUES (?, ?, ?)
            """, (query, result, current_time))
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
        """Documentation omitted."""
        super().__init__(cfg)
        self.cache = SearchCache(resume=SEARCH_CACHE_RESUME) if SEARCH_CACHE_ENABLED else None
    def google_search_with_serp(self, query: str):
        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)
        conn = http.client.HTTPSConnection("google.serper.dev")
        if contains_chinese_basic(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            })
            
        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
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

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content
        except:
            return f"No results found for '{query}'. Try with a more general query."


    
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
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        
        if isinstance(query, str):
            response = self.search_with_serp(query)
        else:
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.search_with_serp(q))
            response = "\n=======\n".join(responses)
            
        return response

