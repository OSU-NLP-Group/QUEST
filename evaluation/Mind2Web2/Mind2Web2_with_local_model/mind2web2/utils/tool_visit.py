import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Tuple, Optional, Dict
import requests
from qwen_agent.tools.base import BaseTool, register_tool
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
from pathlib import Path

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")

_repo_root = Path(__file__).resolve().parents[5]
_default_cache_dir = _repo_root / "database"
_default_cache_dir.mkdir(parents=True, exist_ok=True)
_default_visit_cache_file = str(_default_cache_dir / "visit_cache_merged.db")
VISIT_CACHE_FILE = os.getenv("VISIT_CACHE_FILE", _default_visit_cache_file)
VISIT_CACHE_ENABLED = os.getenv("VISIT_CACHE_ENABLED", "true").lower() == "true"
VISIT_CACHE_RESUME = os.getenv("VISIT_CACHE_RESUME", "true").lower() == "true"


class VisitCache:
    """
    Cache implementation for the Visit tool, backed by SQLite for fast indexed queries.

    Data model:
    - url_content table: stores URL -> content (URL is the primary key because content depends only on URL)
    - url_goal_info table: stores (url, goal) -> useful_information (composite primary key)

    Features:
    - Uses SQLite with indexes for fast lookups
    - Fast content lookup by URL primary key
    - Fast useful_information lookup by (URL, goal) composite key
    - Supports concurrent access (SQLite locking + WAL mode)
    - Supports upsert updates
    - Reduces duplication: each URL's content is stored only once

    Example usage:
        # Default resume=True: connect to existing DB if present, otherwise create one
        cache = VisitCache()

        # Force re-initialization even if the DB already exists
        cache = VisitCache(resume=False)

        # Fast content lookup by URL
        content = cache.get_content_by_url("https://example.com")

        # Fast useful_information lookup by URL + goal
        info = cache.get_useful_information("https://example.com", "find pricing information")

        # Get full cached record
        data = cache.get("https://example.com", "find pricing information")

        # Write cache data
        cache.set("https://example.com", "find pricing information", "raw content", "useful information")
    """
    
    def __init__(self, cache_file: str = VISIT_CACHE_FILE, resume: bool = True):
        """
        Initialize cache and create a persistent DB connection.

        Args:
            cache_file: Path to the cache database file.
            resume: If True and DB exists, connect without re-initializing.
                If False or DB does not exist, create/initialize a new database.
        """
        self.cache_file = cache_file
        self.resume = resume
        self._lock = threading.Lock()  # Thread lock for thread safety.
        
        # Check whether database file exists.
        db_exists = os.path.exists(cache_file)
        
        # Create persistent connection.
        self._conn = sqlite3.connect(cache_file, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        
        # Initialize DB pragmas (one-time setup).
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-8000")
        self._conn.commit()
        
        # If resume=True and DB exists, skip full initialization.
        if resume and db_exists:
            # Validate DB integrity (simple table check).
            try:
                cursor = self._conn.cursor()
                # List all tables.
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table'
                """)
                all_tables = {row['name'] for row in cursor.fetchall()}
                
                # Check whether new schema exists.
                has_new_tables = 'url_content' in all_tables and 'url_goal_info' in all_tables
                # Check whether old schema exists.
                has_old_table = 'visit_cache' in all_tables
                
                if has_new_tables:
                    # New schema already exists.
                    pass
                elif has_old_table:
                    # Old schema detected, migrate data.
                    print(f"[VisitCache] Detected old table structure, migrating data...")
                    self._migrate_from_old_structure()
                else:
                    # Missing/incomplete schema, initialize tables.
                    print(f"[VisitCache] Tables missing or incomplete, initializing...")
                    self._init_database()
            except Exception as e:
                # DB invalid/corrupted, re-initialize.
                print(f"[VisitCache] Database validation failed: {e}, reinitializing...")
                self._init_database()
        else:
            # resume=False or DB missing: initialize schema.
            self._init_database()
        
        # Ensure DB connection is closed on process exit.
        atexit.register(self.close)
    
    def close(self):
        """Close database connection."""
        if hasattr(self, '_conn') and self._conn:
            try:
                self._conn.close()
            except:
                pass
    
    def _execute_read(self, query: str, params: tuple = ()):
        """Execute read-only query (thread-safe)."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()
    
    def _execute_write(self, query: str, params: tuple = ()):
        """Execute write query (thread-safe)."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(query, params)
            self._conn.commit()
    
    def _init_database(self):
        """Initialize database schema and indexes."""
        with self._lock:
            cursor = self._conn.cursor()
            
            # Create url_content table: URL -> content.
            # URL is primary key since content depends only on URL.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS url_content (
                    url TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            
            # Create url_goal_info table: (url, goal) -> useful_information.
            # (url, goal) is a composite primary key.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS url_goal_info (
                    url TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    useful_information TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    PRIMARY KEY (url, goal)
                )
            """)
            
            # Create index for faster joins/lookups on url_goal_info.url.
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_url_goal_info_url 
                ON url_goal_info(url)
            """)
            
            self._conn.commit()
    
    def _migrate_from_old_structure(self):
        """
        Migrate data from old schema (visit_cache) to new schema
        (url_content + url_goal_info).

        Migration steps:
        1. Read all rows from old visit_cache table.
        2. For each (url, goal):
           - Write content into url_content (once per URL, newest wins)
           - Write useful_information into url_goal_info
        3. Keep old table as backup (optional manual cleanup).
        """
        try:
            with self._lock:
                cursor = self._conn.cursor()
                
                # 1. Ensure new schema exists.
                self._init_database()
                
                # 2. Check if old table exists.
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='visit_cache'
                """)
                if cursor.fetchone() is None:
                    print(f"[VisitCache] Old table 'visit_cache' not found, skipping migration")
                    return
                
                # 3. Read all old records.
                cursor.execute("""
                    SELECT url, goal, content, useful_information, timestamp 
                    FROM visit_cache
                    ORDER BY timestamp DESC
                """)
                old_records = cursor.fetchall()
                
                if not old_records:
                    print(f"[VisitCache] No data to migrate from old table")
                    return
                
                print(f"[VisitCache] Migrating {len(old_records)} records from old table structure...")
                
                # 4. Migrate records into new tables.
                migrated_urls = set()  # Track migrated URLs to avoid duplicate content insert.
                migrated_count = 0
                
                for row in old_records:
                    url = row['url']
                    goal = row['goal']
                    content = row['content']
                    useful_information = row['useful_information']
                    timestamp = row['timestamp']
                    
                    # 4.1 Migrate content into url_content (once per URL, newest first).
                    if url not in migrated_urls and content:
                        cursor.execute("""
                            INSERT OR REPLACE INTO url_content 
                            (url, content, timestamp)
                            VALUES (?, ?, ?)
                        """, (url, content, timestamp))
                        migrated_urls.add(url)
                    
                    # 4.2 Migrate useful_information into url_goal_info.
                    if useful_information:
                        cursor.execute("""
                            INSERT OR REPLACE INTO url_goal_info 
                            (url, goal, useful_information, timestamp)
                            VALUES (?, ?, ?, ?)
                        """, (url, goal, useful_information, timestamp))
                        migrated_count += 1
                
                self._conn.commit()
                print(f"[VisitCache] Migration completed: {len(migrated_urls)} URLs and {migrated_count} (url, goal) pairs migrated")
                print(f"[VisitCache] Note: Old table 'visit_cache' is kept as backup. You can delete it manually if needed.")
                
        except Exception as e:
            print(f"[VisitCache] Error during migration: {e}")
            # If migration fails, still initialize new schema.
            print(f"[VisitCache] Creating new table structure anyway...")
            self._init_database()
            raise
    
    def get_content_by_url(self, url: str) -> Optional[str]:
        """
        Quickly get content by URL (goal-independent).

        Args:
            url: Webpage URL.

        Returns:
            Content string, or None if not found.
        """
        if not VISIT_CACHE_ENABLED:
            return None
        
        try:
            row = self._execute_read("""
                SELECT content FROM url_content 
                WHERE url = ?
            """, (url,))
            return row['content'] if row else None
        except Exception as e:
            print(f"[VisitCache] Error getting content by URL {url}: {e}")
            return None
    
    
    def get(self, url: str, goal: str) -> Optional[Dict]:
        """
        Get full cached record (legacy-compatible API).

        Args:
            url: Webpage URL.
            goal: Visit goal.

        Returns:
            Dict with url, goal, content, useful_information, timestamp,
            or None if not found.
        """
        if not VISIT_CACHE_ENABLED:
            return None
        
        try:
            row = self._execute_read("""
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
            if row and row['useful_information']:  # Ensure useful_information exists.
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
        Write data to cache (upsert via INSERT OR REPLACE).

        Args:
            url: Webpage URL.
            goal: Visit goal.
            content: Webpage content (typically from html_readpage_jina).
            useful_information: Extracted useful information.
        """
        if not VISIT_CACHE_ENABLED:
            return
        
        try:
            current_time = time.time()
            
            # 1. Upsert into url_content table (URL is primary key).
            self._execute_write("""
                INSERT OR REPLACE INTO url_content 
                (url, content, timestamp)
                VALUES (?, ?, ?)
            """, (url, content, current_time))
            
            # 2. Upsert into url_goal_info table ((url, goal) is composite key).
            self._execute_write("""
                INSERT OR REPLACE INTO url_goal_info 
                (url, goal, useful_information, timestamp)
                VALUES (?, ?, ?, ?)
            """, (url, goal, useful_information, current_time))
        except Exception as e:
            print(f"[VisitCache] Error writing cache for url={url}, goal={goal}: {e}")

    def set_content_only(self, url: str, content: str):
        """
        Upsert only into url_content table (no url_goal_info write).

        Used when only raw webpage content needs to be cached, without
        generating goal-specific useful_information yet.
        """
        if not VISIT_CACHE_ENABLED:
            return

        try:
            current_time = time.time()
            self._execute_write("""
                INSERT OR REPLACE INTO url_content
                (url, content, timestamp)
                VALUES (?, ?, ?)
            """, (url, content, current_time))
        except Exception as e:
            print(f"[VisitCache] Error writing content for url={url}: {e}")


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
        """Initialize Visit tool and cache instance."""
        super().__init__(*args, **kwargs)
        self.cache = VisitCache(resume=VISIT_CACHE_RESUME) if VISIT_CACHE_ENABLED else None
    
    def _validate_url(self, url: str) -> Tuple[bool, str]:
        """
        Validate URL and reject invalid prefixes.

        Args:
            url: URL to validate.

        Returns:
            (is_valid, error_message): returns False and error text for invalid
            input; returns True and empty text for valid input.
        """
        if not isinstance(url, str):
            return True, ""  # Non-string inputs are handled by later logic.
        
        url = url.strip()
        
        # Check invalid prefixes.
        if url.startswith("view-source:"):
            return False, f"[Visit] Invalid URL protocol: 'view-source:' is not a valid protocol. Please use the actual URL without the 'view-source:' prefix. For example, use 'https://example.com' instead of 'view-source:https://example.com'."
        
        if url.startswith("javascript:"):
            return False, f"[Visit] Invalid URL protocol: 'javascript:' is not a valid protocol for visiting webpages. Please provide a valid HTTP/HTTPS URL."
        
        return True, ""
    

    # The `newcall` method is the main function of the tool.
    def newcall(self,url: str) -> str:
        try:
            url = url
            goal = ""
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()
        
        # Create log folder if it doesn't exist
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)

        # Validate URL and return early on invalid input.
        if isinstance(url, str):
            is_valid, error_msg = self._validate_url(url)
            if not is_valid:
                return error_msg
            
            # Cache lookup logic.
            if self.cache:
                cached_content = self.cache.get_content_by_url(url)
                if cached_content:
                    return cached_content
            response = self.html_readpage_jina(url)
            original_content = response
            if (
                self.cache
                and original_content
                and not original_content.startswith("[visit] Failed")
                and not original_content.startswith("[visit] Error")
            ):
                self.cache.set_content_only(url, original_content)
        else:
            response = []
            if time.time() - start_time > 900:
                cur_response = "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format."
                response.append(cur_response)      
        return response.strip()
    
    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
        max_retries = 3
        timeout = 50
        
        for attempt in range(max_retries):
            headers = {
                "Authorization": f"Bearer {JINA_API_KEYS}",
            }
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    # Try parsing JSON error response for clearer diagnostics.
                    error_info = response.text
                    try:
                        error_json = json.loads(response.text)
                        error_message = error_json.get("readableMessage") or error_json.get("message", response.text)
                        error_code = error_json.get("code", response.status_code)
                        error_name = error_json.get("name", "UnknownError")
                        error_info = f"[visit] Error ({error_name}, code {error_code}): {error_message}"
                    except:
                        # Fallback to raw response text when not JSON.
                        error_info = f"[visit] Error (HTTP {response.status_code}): {response.text[:500]}"
                    
                    print(f"[visit] Jina API error: {error_info}")
                    # Return detailed error on the final attempt.
                    if attempt == max_retries - 1:
                        return error_info
                    raise ValueError(error_info)
            except requests.exceptions.RequestException as e:
                error_info = f"[visit] Network error: {str(e)}"
                print(f"[visit] Request exception: {error_info}")
                if attempt == max_retries - 1:
                    return error_info
                time.sleep(0.5)
            except Exception as e:
                error_info = f"[visit] Error: {str(e)}"
                print(f"[visit] Exception: {error_info}")
                if attempt == max_retries - 1:
                    return error_info
                time.sleep(0.5)
                
        return "[visit] Failed to read page after all retries."

    def html_readpage_jina(self, url: str) -> str:
        max_attempts = 8
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            # service = "jina"     
            # print(service)
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
        return "[visit] Failed to read page."

    
