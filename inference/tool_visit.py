import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Tuple, Optional, Dict
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from prompt import (
    build_visit_extractor_messages,
    choose_local_openai_base_url,
    get_local_served_model_name,
    use_visit_local_prompt,
)
from openai import OpenAI, AzureOpenAI
import random
from urllib.parse import urlparse, unquote
import time 
from transformers import AutoTokenizer
import tiktoken
import hashlib
import sqlite3
from contextlib import contextmanager
import atexit
import fcntl

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))
BLOCK_HUGGINGFACE = os.getenv("BLOCK_HUGGINGFACE", "false").lower() == "true"

JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")

# Cache configuration
_shared_cache_dir = os.getenv(
    "SHARED_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"),
)
_default_cache_dir = os.getenv("CACHE_DIR", _shared_cache_dir)
os.makedirs(_default_cache_dir, exist_ok=True)
_default_visit_cache_file = os.path.join(_default_cache_dir, "visit_cache_merged.db")
VISIT_CACHE_FILE = os.getenv("VISIT_CACHE_FILE", _default_visit_cache_file)
VISIT_CACHE_SHARD_DIR = os.getenv("VISIT_CACHE_SHARD_DIR", _default_cache_dir)
VISIT_CACHE_SHARED_FILE = os.getenv("VISIT_CACHE_SHARED_FILE", os.path.join(_shared_cache_dir, "visit_cache_merged.db"))
VISIT_CACHE_ENABLED = os.getenv("VISIT_CACHE_ENABLED", "true").lower() == "true"
VISIT_CACHE_RESUME = os.getenv("VISIT_CACHE_RESUME", "true").lower() == "true"
VISIT_CACHE_SHARDS = max(1, int(os.getenv("VISIT_CACHE_SHARDS", "32")))
VISIT_CACHE_AUTO_MERGE = os.getenv("VISIT_CACHE_AUTO_MERGE", "true").lower() == "true"


def _is_cache_merge_leader() -> bool:
    for env in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        val = os.getenv(env)
        if val is not None:
            try:
                return int(val) == 0
            except ValueError:
                return False
    return True


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    """Truncate text to the specified token count"""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most relevant information** from the content, never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal."}}}}"""


class VisitCache:
    """Visit tool cache backed by sharded SQLite databases."""

    def __init__(self, cache_file: str = VISIT_CACHE_FILE, resume: bool = True, shards: int = VISIT_CACHE_SHARDS):
        self.cache_file = cache_file
        self.resume = resume
        self.shards = max(1, int(shards))
        self.auto_merge = VISIT_CACHE_AUTO_MERGE
        self.is_merge_leader = _is_cache_merge_leader()
        self.shard_dir = VISIT_CACHE_SHARD_DIR
        os.makedirs(self.shard_dir, exist_ok=True)
        self._master_lock = threading.Lock()
        self._conns: dict[int, sqlite3.Connection] = {}
        self._locks: dict[int, threading.Lock] = {}

        base_name = os.path.splitext(os.path.basename(self.cache_file))[0]
        self._shard_files = [os.path.join(self.shard_dir, f"{base_name}_shard{idx}.db") for idx in range(self.shards)]
        self._master_read_conn = None
        self._user_read_conn = None
        if self.resume:
            shared = VISIT_CACHE_SHARED_FILE
            if shared and os.path.exists(shared):
                self._master_read_conn = self._open_readonly_conn(shared)
            if os.path.exists(self.cache_file) and self.cache_file != VISIT_CACHE_SHARED_FILE:
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
        self._ensure_tables(conn)
        return conn

    def _open_readonly_conn(self, path: str) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS url_content (
                url TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS url_goal_info (
                url TEXT NOT NULL,
                goal TEXT NOT NULL,
                useful_information TEXT NOT NULL,
                timestamp REAL NOT NULL,
                PRIMARY KEY (url, goal)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_url_goal_info_url
            ON url_goal_info(url)
        """)
        conn.commit()

    def _get_conn(self, shard_id: int) -> sqlite3.Connection:
        if shard_id not in self._conns:
            path = self.cache_file if self.shards == 1 else self._shard_files[shard_id]
            self._conns[shard_id] = self._open_conn(path)
            self._locks[shard_id] = threading.Lock()
        return self._conns[shard_id]

    def _shard_id(self, url: str) -> int:
        if self.shards == 1:
            return 0
        digest = hashlib.md5(url.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.shards

    def _init_database(self):
        if self.shards == 1:
            self._get_conn(0)

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
                    shard_cursor.execute("SELECT url, content, timestamp FROM url_content")
                    for row in shard_cursor.fetchall():
                        master_cursor.execute("""
                            INSERT OR REPLACE INTO url_content
                            (url, content, timestamp)
                            VALUES (?, ?, ?)
                        """, (row["url"], row["content"], row["timestamp"]))
                    shard_cursor.execute("SELECT url, goal, useful_information, timestamp FROM url_goal_info")
                    for row in shard_cursor.fetchall():
                        master_cursor.execute("""
                            INSERT OR REPLACE INTO url_goal_info
                            (url, goal, useful_information, timestamp)
                            VALUES (?, ?, ?, ?)
                        """, (row["url"], row["goal"], row["useful_information"], row["timestamp"]))
                    master_conn.commit()
                    shard_conn.close()
                master_conn.close()
        except Exception as e:
            print(f"[VisitCache] Error merging shard cache into '{self.cache_file}': {e}")

    def _migrate_from_old_structure(self):
        print("[VisitCache] Legacy inline migration is disabled in sharded mode. Keep visit_cache_merged.db as read-only fallback or migrate offline.")
    
    def get_content_by_url(self, url: str) -> Optional[str]:
        """
        Quickly fetch content by URL (goal is ignored)
        
        Args:
            url: webpage URL
            
        Returns:
            content, or None if it does not exist
        """
        if not VISIT_CACHE_ENABLED:
            return None
        
        try:
            shard_id = self._shard_id(url)
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT content FROM url_content
                    WHERE url = ?
                """, (url,))
                row = cursor.fetchone()
            if not row and self._user_read_conn and self.shards > 1:
                with self._master_lock:
                    cursor = self._user_read_conn.cursor()
                    cursor.execute("""
                        SELECT content FROM url_content
                        WHERE url = ?
                    """, (url,))
                    row = cursor.fetchone()
            if not row and self._master_read_conn and self.shards > 1:
                with self._master_lock:
                    cursor = self._master_read_conn.cursor()
                    cursor.execute("""
                        SELECT content FROM url_content
                        WHERE url = ?
                    """, (url,))
                    row = cursor.fetchone()
            return row['content'] if row else None
        except Exception as e:
            print(f"[VisitCache] Error getting content by URL {url}: {e}")
            return None
    
    def get_useful_information(self, url: str, goal: str) -> Optional[str]:
        """
        Quickly fetch useful_information by URL + goal
        
        Args:
            url: webpage URL
            goal: visit objective
            
        Returns:
            useful_information, or None if it does not exist
        """
        if not VISIT_CACHE_ENABLED:
            return None
        
        try:
            shard_id = self._shard_id(url)
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT useful_information FROM url_goal_info
                    WHERE url = ? AND goal = ?
                """, (url, goal))
                row = cursor.fetchone()
            if not row and self._user_read_conn and self.shards > 1:
                with self._master_lock:
                    cursor = self._user_read_conn.cursor()
                    cursor.execute("""
                        SELECT useful_information FROM url_goal_info
                        WHERE url = ? AND goal = ?
                    """, (url, goal))
                    row = cursor.fetchone()
            if not row and self._master_read_conn and self.shards > 1:
                with self._master_lock:
                    cursor = self._master_read_conn.cursor()
                    cursor.execute("""
                        SELECT useful_information FROM url_goal_info
                        WHERE url = ? AND goal = ?
                    """, (url, goal))
                    row = cursor.fetchone()
            return row['useful_information'] if row else None
        except Exception as e:
            print(f"[VisitCache] Error getting useful_information for url={url}, goal={goal}: {e}")
            return None
    
    def get(self, url: str, goal: str) -> Optional[Dict]:
        """
        Get full data from the cache (compatible with the old interface)
        
        Args:
            url: webpage URL
            goal: visit objective
            
        Returns:
            cached data dict containing url, goal, content, useful_information, and timestamp
            None if it does not exist
        """
        if not VISIT_CACHE_ENABLED:
            return None
        
        try:
            shard_id = self._shard_id(url)
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT
                        uc.url,
                        ugi.goal,
                        uc.content,
                        ugi.useful_information,
                        uc.timestamp as content_timestamp,
                        ugi.timestamp as info_timestamp
                    FROM url_content uc
                    LEFT JOIN url_goal_info ugi ON uc.url = ugi.url AND ugi.goal = ?
                    WHERE uc.url = ?
                """, (goal, url))
                row = cursor.fetchone()
            if (not row or not row['useful_information']) and self._user_read_conn and self.shards > 1:
                with self._master_lock:
                    cursor = self._user_read_conn.cursor()
                    cursor.execute("""
                        SELECT
                            uc.url,
                            ugi.goal,
                            uc.content,
                            ugi.useful_information,
                            uc.timestamp as content_timestamp,
                            ugi.timestamp as info_timestamp
                        FROM url_content uc
                        LEFT JOIN url_goal_info ugi ON uc.url = ugi.url AND ugi.goal = ?
                        WHERE uc.url = ?
                    """, (goal, url))
                    row = cursor.fetchone()
            if (not row or not row['useful_information']) and self._master_read_conn and self.shards > 1:
                with self._master_lock:
                    cursor = self._master_read_conn.cursor()
                    cursor.execute("""
                        SELECT
                            uc.url,
                            ugi.goal,
                            uc.content,
                            ugi.useful_information,
                            uc.timestamp as content_timestamp,
                            ugi.timestamp as info_timestamp
                        FROM url_content uc
                        LEFT JOIN url_goal_info ugi ON uc.url = ugi.url AND ugi.goal = ?
                        WHERE uc.url = ?
                    """, (goal, url))
                    row = cursor.fetchone()
            if row and row['useful_information']:
                return {
                    'url': row['url'],
                    'goal': row['goal'],
                    'content': row['content'],
                    'useful_information': row['useful_information'],
                    'timestamp': row['info_timestamp'] or row['content_timestamp']
                }
            return None
        except Exception as e:
            print(f"[VisitCache] Error getting cache for url={url}, goal={goal}: {e}")
            return None
    
    def set(self, url: str, goal: str, content: str, useful_information: str):
        """
        Write data to the cache (uses INSERT OR REPLACE to support updates)
        
        Args:
            url: webpage URL
            goal: visit objective
            content: webpage content (from html_readpage_jina)
            useful_information: useful information summary
        """
        if not VISIT_CACHE_ENABLED:
            return
        
        try:
            current_time = time.time()
            shard_id = self._shard_id(url)
            conn = self._get_conn(shard_id)
            with self._locks[shard_id]:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO url_content
                    (url, content, timestamp)
                    VALUES (?, ?, ?)
                """, (url, content, current_time))
                cursor.execute("""
                    INSERT OR REPLACE INTO url_goal_info
                    (url, goal, useful_information, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (url, goal, useful_information, current_time))
                conn.commit()
        except Exception as e:
            print(f"[VisitCache] Error writing cache for url={url}, goal={goal}: {e}")


@register_tool('visit', allow_overwrite=True)
class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                    },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
        },
        "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
        }
        },
        "required": ["url", "goal"]
    }
    
    def __init__(self, *args, **kwargs):
        """Initialize the Visit tool and create the cache instance"""
        super().__init__(*args, **kwargs)
        self.cache = VisitCache(resume=VISIT_CACHE_RESUME, shards=VISIT_CACHE_SHARDS) if VISIT_CACHE_ENABLED else None
    
    def _validate_url(self, url: str) -> Tuple[bool, str]:
        """
        Validate whether the URL is valid by checking invalid prefixes

        Args:
            url: URL to validate

        Returns:
            (is_valid, error_message): if invalid, return False and an error message; otherwise return True and an empty string
        """
        if not isinstance(url, str):
            return False, "[Visit] URL must be a string."

        url = url.strip()
        if not url:
            return False, "[Visit] URL cannot be empty."

        lowered = url.lower()
        if lowered.startswith("view-source:"):
            return False, "[Visit] Invalid URL protocol: 'view-source:' is not allowed."
        if lowered.startswith("javascript:"):
            return False, "[Visit] Invalid URL protocol: 'javascript:' is not allowed."
        if lowered.startswith("data:"):
            return False, "[Visit] Invalid URL protocol: 'data:' is not allowed."
        if lowered.startswith("file:"):
            return False, "[Visit] Invalid URL protocol: 'file:' is not allowed."
        # Block huggingface.co if enabled
        if BLOCK_HUGGINGFACE and "huggingface.co" in lowered:
            print("[Visit] Access to huggingface.co is forbidden.")
            return False, "[Visit] Access to huggingface.co is forbidden."
        return True, ""

    def _normalize_summary_output(self, raw) -> Tuple[str, str]:
        """
        Normalize summary model output so missing keys do not fail the visit tool.

        Returns:
            (evidence_text, summary_text)
        """
        if isinstance(raw, dict):
            evidence = (
                raw.get("evidence")
                or raw.get("excerpt")
                or raw.get("excerpts")
                or raw.get("content")
                or raw.get("rational")
                or raw.get("rationale")
                or ""
            )
            summary = (
                raw.get("summary")
                or raw.get("answer")
                or raw.get("conclusion")
                or raw.get("final")
                or ""
            )

            if not evidence and not summary:
                fallback = json.dumps(raw, ensure_ascii=False)
                print(f"[visit] Warning: summary JSON missing expected fields, using raw JSON fallback. Keys: {list(raw.keys())}")
                return fallback, fallback
            if not evidence:
                print(f"[visit] Warning: summary JSON missing 'evidence', falling back to summary/other fields. Keys: {list(raw.keys())}")
                evidence = summary or json.dumps(raw, ensure_ascii=False)
            if not summary:
                print(f"[visit] Warning: summary JSON missing 'summary', falling back to evidence/other fields. Keys: {list(raw.keys())}")
                summary = evidence or json.dumps(raw, ensure_ascii=False)
            return str(evidence), str(summary)

        fallback = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        print(f"[visit] Warning: summary output is not a dict, using text fallback. Type: {type(raw).__name__}")
        return str(fallback), str(fallback)
    
    def _process_content_to_summary(self, url: str, goal: str, content: str) -> str:
        """
        Generate useful_information from cached content
        
        Args:
            url: webpage URL
            goal: visit objective
            content: webpage content (from cache)
            
        Returns:
            useful_information: processed useful information summary
        """
        summary_page_func = self.call_server
        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))
        
        # Keep the original content for caching
        original_content = content
        
        if content and not content.startswith("[visit] Failed to read page after all retries.") and content != "[visit] Empty content." and not content.startswith("[document_parser]") and not content.startswith("[visit] Error"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
            parse_retry_times = 0
            raw = summary_page_func(messages, max_retries=max_retries)
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                status_msg = (
                    f"[visit] Summary url[{url}] from cache " 
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[visit] Summary url[{url}] from cache failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                print(status_msg)
                content = content[:truncate_length]
                extraction_prompt = EXTRACTOR_PROMPT.format(
                    webpage_content=content,
                    goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = summary_page_func(messages, max_retries=max_retries)
                summary_retries -= 1

            parse_retry_times = 2
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = summary_page_func(messages, max_retries=max_retries)
                    parse_retry_times += 1
            
            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            else:
                evidence_text, summary_text = self._normalize_summary_output(raw)
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + evidence_text + "\n\n"
                useful_information += "Summary: \n" + summary_text + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                print("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"
            
            # Write to cache only on success(normal status code and successfully parsed result)
            # Check conditions:1) the original content is not an error 2) JSON was parsed successfully 3) useful_information is not a failure message
            if (self.cache and original_content and 
                not original_content.startswith("[visit] Failed to read page after all retries.") and 
                not original_content.startswith("[visit] Error") and
                parse_retry_times < 3 and  # JSON was parsed successfully
                not useful_information.startswith("[visit] Failed to read page") and
                len(useful_information) >= 10):  # useful_information is valid
                self.cache.set(url, goal, original_content, useful_information)
            
            return useful_information
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            
            # Do not cache failure cases(cache only responses with normal status codes)
            
            return useful_information
    
    # The `call` method is the main function of the tool.
    def call(self, params: Union[str, dict], **kwargs) -> str:
        if not isinstance(params, dict):
            return "[Visit Error] URL cannot be empty."

        url = params.get("url", "")
        goal = params.get("goal", "Extract relevant information")
        if not url:
            return "[Visit Error] URL cannot be empty."

        start_time = time.time()
        
        # Create log folder if it doesn't exist
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        # Validate the URL; if invalid, return the error directly
        if isinstance(url, str):
            is_valid, error_msg = self._validate_url(url)
            if not is_valid:
                return error_msg
            
            # Cache lookup logic
            if self.cache:
                # 1. Check url+goal first; if present, return useful_information directly
                useful_information = self.cache.get_useful_information(url, goal)
                if useful_information:
                    print(f"[visit] Cache hit (url+goal) for url: {url}, goal: {goal}")
                    return useful_information
                
                # 2. If not found, query by url; if present, return the content and process it
                cached_content = self.cache.get_content_by_url(url)
                if cached_content:
                    print(f"[visit] Cache hit (url) for url: {url}, processing content with goal: {goal}")
                    useful_information = self._process_content_to_summary(url, goal, cached_content)
                    return useful_information
            
            # 3. If neither exists, run the original pipeline
            response = self.readpage_jina(url, goal)
        else:
            response = []
            urls = url if isinstance(url, List) else [url]
            start_time = time.time()
            for u in urls:
                # Validate each URL
                is_valid, error_msg = self._validate_url(u)
                if not is_valid:
                    response.append(error_msg)
                    continue
                
                if time.time() - start_time > 900:
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    try:
                        # Apply the cache lookup logic to each URL as well
                        if self.cache:
                            # 1. Check url+goal first
                            useful_information = self.cache.get_useful_information(u, goal)
                            if useful_information:
                                print(f"[visit] Cache hit (url+goal) for url: {u}, goal: {goal}")
                                cur_response = useful_information
                            else:
                                # 2. Query by url
                                cached_content = self.cache.get_content_by_url(u)
                                if cached_content:
                                    print(f"[visit] Cache hit (url) for url: {u}, processing content with goal: {goal}")
                                    cur_response = self._process_content_to_summary(u, goal, cached_content)
                                else:
                                    # 3. Run the original pipeline
                                    cur_response = self.readpage_jina(u, goal)
                        else:
                            cur_response = self.readpage_jina(u, goal)
                    except Exception as e:
                        cur_response = f"[visit] Error: Unexpected error: {str(e)}"
                response.append(cur_response)
            response = "\n=======\n".join(response)
        
        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()
        
    def call_server(self, msgs, max_retries=2):
        if use_visit_local_prompt():
            model_name = get_local_served_model_name()
            last_error = None
            for attempt in range(max_retries):
                try:
                    client = OpenAI(
                        api_key="EMPTY",
                        base_url=choose_local_openai_base_url(),
                        timeout=600.0,
                    )
                    chat_response = client.chat.completions.create(
                        model=model_name,
                        messages=msgs,
                        temperature=1
                    )
                    content = chat_response.choices[0].message.content
                    if content:
                        try:
                            json.loads(content)
                        except Exception:
                            left = content.find('{')
                            right = content.rfind('}')
                            if left != -1 and right != -1 and left <= right:
                                content = content[left:right + 1]
                        print("[visit] local server call success")
                        return content
                except Exception as e:
                    last_error = e
                    print(f"[visit] local server call error: {e}")
                    continue
            if last_error:
                print(f"[visit] local server exhausted retries: {last_error}")
            return ""

        api_key = os.environ.get("API_KEY")
        url_llm = os.environ.get("API_BASE")
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = (
            os.getenv("AZURE_OPENAI_API_VERSION")
            or "2024-08-01-preview"
        )
        if azure_endpoint:
            model_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or model_name
            client = AzureOpenAI(
                api_key=api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint,
            )
        else:
            client = OpenAI(
                api_key=api_key,
                base_url=url_llm,
            )
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=model_name,
                    messages=msgs,
                    temperature=1
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    print("[visit] call server success")
                    return content
            except Exception as e:
                print(e)
                print("[Visit] call server error")
                if attempt == (max_retries - 1):
                    return ""
                continue


    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        if not JINA_API_KEYS:
            return "[visit] Error: JINA_API_KEYS environment variable not set."

        max_retries = 3
        timeout = 50
        headers = {"Authorization": f"Bearer {JINA_API_KEYS}"}

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout,
                )
                if response.status_code == 200:
                    return response.text

                error_info = response.text
                try:
                    error_json = json.loads(response.text)
                    error_message = error_json.get("readableMessage") or error_json.get("message", response.text)
                    error_info = f"[visit] Error (HTTP {response.status_code}): {error_message}"
                except Exception:
                    error_info = f"[visit] Error (HTTP {response.status_code}): {response.text[:500]}"

                if attempt == max_retries - 1:
                    return error_info
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    return f"[visit] Error: Request timed out after {timeout}s"
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return f"[visit] Error: Network error: {str(e)}"
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"[visit] Error: Unexpected error: {str(e)}"

        return "[visit] Failed to read page after all retries."

    def html_readpage_jina(self, url: str) -> str:
        max_attempts = 8
        last_error = None
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            service = "jina"     
            print(service)
            # Check whether this is an error message(starting with [visit] Error or [visit] Failed)
            if content and content.startswith("[visit] Error"):
                last_error = content
                # Continue retrying, but record the last error
                continue
            if content and not content.startswith("[visit] Failed") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
        # If all attempts fail, return the last error message(if any), otherwise return a generic error
        if last_error:
            return last_error
        return "[visit] Failed to read page after all retries."

    def readpage_jina(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
   
        summary_page_func = self.call_server
        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))

        content = self.html_readpage_jina(url)
        # Keep the original content for caching
        original_content = content

        # Check whether this is an error message; if so, return it directly to the model
        if content and (content.startswith("[visit] Error") or content.startswith("[visit] Failed")):
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + content + "\n\n"
            useful_information += "Summary: \n" + "The webpage could not be accessed. " + content.replace("[visit] ", "") + "\n\n"
            
            # Do not cache failure cases(cache only responses with normal status codes)
            
            return useful_information

        if content and not content.startswith("[visit] Failed") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = build_visit_extractor_messages(content, goal)
            parse_retry_times = 0
            raw = summary_page_func(messages, max_retries=max_retries)
            summary_retries = 3
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                status_msg = (
                    f"[visit] Summary url[{url}] " 
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[visit] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                print(status_msg)
                content = content[:truncate_length]
                messages = build_visit_extractor_messages(content, goal)
                raw = summary_page_func(messages, max_retries=max_retries)
                summary_retries -= 1

            parse_retry_times = 2
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = summary_page_func(messages, max_retries=max_retries)
                    parse_retry_times += 1
            
            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            else:
                evidence_text, summary_text = self._normalize_summary_output(raw)
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + evidence_text + "\n\n"
                useful_information += "Summary: \n" + summary_text + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                print("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"
            
            # Write to cache only on success(normal status code and successfully parsed result)
            # Check conditions:1) the original content is not an error 2) JSON was parsed successfully 3) useful_information is not a failure message
            if (self.cache and original_content and 
                not original_content.startswith("[visit] Failed to read page after all retries.") and 
                not original_content.startswith("[visit] Error") and
                parse_retry_times < 3 and  # JSON was parsed successfully
                not useful_information.startswith("[visit] Failed to read page") and
                len(useful_information) >= 10):  # useful_information is valid
                self.cache.set(url, goal, original_content, useful_information)
            
            return useful_information

        # If no valid content was obtained after all retries
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            
            # Do not cache failure cases(cache only responses with normal status codes)
            
            return useful_information

    
