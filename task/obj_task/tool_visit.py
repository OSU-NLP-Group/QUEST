import json
import os
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Tuple, Optional, Dict
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from generation_prompts import EXTRACTOR_PROMPT 
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

VISIT_SERVER_TIMEOUT = int(os.getenv("VISIT_SERVER_TIMEOUT", 200))
WEBCONTENT_MAXLENGTH = int(os.getenv("WEBCONTENT_MAXLENGTH", 150000))

JINA_API_KEYS = os.getenv("JINA_API_KEYS", "")
VISIT_CACHE_FILE = os.getenv("VISIT_CACHE_FILE", "visit_cache.db")
VISIT_CACHE_ENABLED = os.getenv("VISIT_CACHE_ENABLED", "true").lower() == "true"
VISIT_CACHE_RESUME = os.getenv("VISIT_CACHE_RESUME", "true").lower() == "true"


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    """Documentation omitted."""
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
    """Documentation omitted."""
    
    def __init__(self, cache_file: str = VISIT_CACHE_FILE, resume: bool = True):
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
                    WHERE type='table'
                """)
                all_tables = {row['name'] for row in cursor.fetchall()}
                has_new_tables = 'url_content' in all_tables and 'url_goal_info' in all_tables
                has_old_table = 'visit_cache' in all_tables
                
                if has_new_tables:
                    pass
                elif has_old_table:
                    print(f"[VisitCache] Detected old table structure, migrating data...")
                    self._migrate_from_old_structure()
                else:
                    print(f"[VisitCache] Tables missing or incomplete, initializing...")
                    self._init_database()
            except Exception as e:
                print(f"[VisitCache] Database validation failed: {e}, reinitializing...")
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
            
            self._conn.commit()
    
    def _migrate_from_old_structure(self):
        """Documentation omitted."""
        try:
            with self._lock:
                cursor = self._conn.cursor()
                self._init_database()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='visit_cache'
                """)
                if cursor.fetchone() is None:
                    print(f"[VisitCache] Old table 'visit_cache' not found, skipping migration")
                    return
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
                migrated_urls = set()
                migrated_count = 0
                
                for row in old_records:
                    url = row['url']
                    goal = row['goal']
                    content = row['content']
                    useful_information = row['useful_information']
                    timestamp = row['timestamp']
                    if url not in migrated_urls and content:
                        cursor.execute("""
                            INSERT OR REPLACE INTO url_content 
                            (url, content, timestamp)
                            VALUES (?, ?, ?)
                        """, (url, content, timestamp))
                        migrated_urls.add(url)
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
            print(f"[VisitCache] Creating new table structure anyway...")
            self._init_database()
            raise
    
    def get_content_by_url(self, url: str) -> Optional[str]:
        """Documentation omitted."""
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
    
    def get_useful_information(self, url: str, goal: str) -> Optional[str]:
        """Documentation omitted."""
        if not VISIT_CACHE_ENABLED:
            return None
        
        try:
            row = self._execute_read("""
                SELECT useful_information FROM url_goal_info 
                WHERE url = ? AND goal = ?
            """, (url, goal))
            return row['useful_information'] if row else None
        except Exception as e:
            print(f"[VisitCache] Error getting useful_information for url={url}, goal={goal}: {e}")
            return None
    
    def get(self, url: str, goal: str) -> Optional[Dict]:
        """Documentation omitted."""
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
        """Documentation omitted."""
        if not VISIT_CACHE_ENABLED:
            return
        
        try:
            current_time = time.time()
            self._execute_write("""
                INSERT OR REPLACE INTO url_content 
                (url, content, timestamp)
                VALUES (?, ?, ?)
            """, (url, content, current_time))
            self._execute_write("""
                INSERT OR REPLACE INTO url_goal_info 
                (url, goal, useful_information, timestamp)
                VALUES (?, ?, ?, ?)
            """, (url, goal, useful_information, current_time))
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
        """Documentation omitted."""
        super().__init__(*args, **kwargs)
        self.cache = VisitCache(resume=VISIT_CACHE_RESUME) if VISIT_CACHE_ENABLED else None
    
    def _validate_url(self, url: str) -> Tuple[bool, str]:
        """Documentation omitted."""
        if not isinstance(url, str):
            return True, ""
        
        url = url.strip()
        if url.startswith("view-source:"):
            return False, f"[Visit] Invalid URL protocol: 'view-source:' is not a valid protocol. Please use the actual URL without the 'view-source:' prefix. For example, use 'https://example.com' instead of 'view-source:https://example.com'."
        
        if url.startswith("javascript:"):
            return False, f"[Visit] Invalid URL protocol: 'javascript:' is not a valid protocol for visiting webpages. Please provide a valid HTTP/HTTPS URL."
        
        return True, ""
    
    def _process_content_to_summary(self, url: str, goal: str, content: str) -> str:
        """Documentation omitted."""
        summary_page_func = self.call_server
        max_retries = int(os.getenv('VISIT_SERVER_MAX_RETRIES', 1))
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
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                print("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"
            if (self.cache and original_content and 
                not original_content.startswith("[visit] Failed to read page after all retries.") and 
                not original_content.startswith("[visit] Error") and
                parse_retry_times < 3 and
                not useful_information.startswith("[visit] Failed to read page") and
                len(useful_information) >= 10):
                self.cache.set(url, goal, original_content, useful_information)
            
            return useful_information
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            
            return useful_information
    
    # The `call` method is the main function of the tool.
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        start_time = time.time()
        
        # Create log folder if it doesn't exist
        log_folder = "log"
        os.makedirs(log_folder, exist_ok=True)
        if isinstance(url, str):
            is_valid, error_msg = self._validate_url(url)
            if not is_valid:
                return error_msg
            if self.cache:
                useful_information = self.cache.get_useful_information(url, goal)
                if useful_information:
                    print(f"[visit] Cache hit (url+goal) for url: {url}, goal: {goal}")
                    return useful_information
                cached_content = self.cache.get_content_by_url(url)
                if cached_content:
                    print(f"[visit] Cache hit (url) for url: {url}, processing content with goal: {goal}")
                    useful_information = self._process_content_to_summary(url, goal, cached_content)
                    return useful_information
            response = self.readpage_jina(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()
            for u in url:
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
                        if self.cache:
                            useful_information = self.cache.get_useful_information(u, goal)
                            if useful_information:
                                print(f"[visit] Cache hit (url+goal) for url: {u}, goal: {goal}")
                                cur_response = useful_information
                            else:
                                cached_content = self.cache.get_content_by_url(u)
                                if cached_content:
                                    print(f"[visit] Cache hit (url) for url: {u}, processing content with goal: {goal}")
                                    cur_response = self._process_content_to_summary(u, goal, cached_content)
                                else:
                                    cur_response = self.readpage_jina(u, goal)
                        else:
                            cur_response = self.readpage_jina(u, goal)
                    except Exception as e:
                        cur_response = f"Error fetching {u}: {str(e)}"
                response.append(cur_response)
            response = "\n=======\n".join(response)
        
        print(f'Summary Length {len(response)}; Summary Content {response}')
        return response.strip()
        
    def call_server(self, msgs, max_retries=2):
        api_key = os.environ.get("SUMMARY_AZURE_API_KEY") or os.environ.get("SUMMARY_OPENAI_API_KEY") or os.environ.get("API_KEY")
        url_llm = os.environ.get("SUMMARY_AZURE_API_BASE") or os.environ.get("API_BASE")
        model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
        azure_endpoint = (
            os.getenv("SUMMARY_AZURE_API_BASE")
            or os.getenv("AZURE_OPENAI_ENDPOINT")
            or os.getenv("AZURE_ENDPOINT")
        )
        azure_api_version = (
            os.getenv("SUMMARY_AZURE_API_VERSION")
            or os.getenv("AZURE_OPENAI_API_VERSION")
            or os.getenv("AZURE_API_VERSION")
            or "2024-08-01-preview"
        )
        
        if azure_endpoint:
            if model_name.startswith("azure/"):
                model_name = model_name.split("/", 1)[1]
            else:
                model_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT") or model_name
            
            if not api_key:
                raise ValueError("SUMMARY_AZURE_API_KEY or API_KEY must be set for Azure OpenAI")
            
            client = AzureOpenAI(
                api_key=api_key,
                api_version=azure_api_version,
                azure_endpoint=azure_endpoint,
            )
        else:
            if not api_key:
                raise ValueError("SUMMARY_OPENAI_API_KEY or API_KEY must be set for OpenAI API")
            
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
                    return content
            except Exception as e:
                # print(e)
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
                    error_info = response.text
                    try:
                        error_json = json.loads(response.text)
                        error_message = error_json.get("readableMessage") or error_json.get("message", response.text)
                        error_code = error_json.get("code", response.status_code)
                        error_name = error_json.get("name", "UnknownError")
                        error_info = f"[visit] Error ({error_name}, code {error_code}): {error_message}"
                    except:
                        error_info = f"[visit] Error (HTTP {response.status_code}): {response.text[:500]}"
                    
                    print(f"[visit] Jina API error: {error_info}")
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
        last_error = None
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            service = "jina"     
            print(service)
            if content and content.startswith("[visit] Error"):
                last_error = content
                continue
            if content and not content.startswith("[visit] Failed") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
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
        original_content = content
        if content and (content.startswith("[visit] Error") or content.startswith("[visit] Failed")):
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + content + "\n\n"
            useful_information += "Summary: \n" + "The webpage could not be accessed. " + content.replace("[visit] ", "") + "\n\n"
            
            return useful_information

        if content and not content.startswith("[visit] Failed") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            content = truncate_to_tokens(content, max_tokens=95000)
            messages = [{"role":"user","content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
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
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                print("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"
            if (self.cache and original_content and 
                not original_content.startswith("[visit] Failed to read page after all retries.") and 
                not original_content.startswith("[visit] Error") and
                parse_retry_times < 3 and
                not useful_information.startswith("[visit] Failed to read page") and
                len(useful_information) >= 10):
                self.cache.set(url, goal, original_content, useful_information)
            
            return useful_information

        # If no valid content was obtained after all retries
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            
            return useful_information

    
