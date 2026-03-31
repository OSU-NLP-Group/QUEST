"""
Batch crawler using CacheFileSys (v2) - file-based cache with single-task design.

Key changes from the old batch_cache.py:
- Uses CacheFileSys instead of CacheClass (one cache instance per task)
- Stores content in task directories instead of PKL files
- Uses put_web(url, text, screenshot) instead of separate put_text/put_screenshot
- Removes MHTML storage (not supported in CacheFileSys)
- Memory efficient: only indexes in memory, content loaded on-demand

Depends on unified path management (`PathConfig`), which auto-detects the
project root and subdirectories like dataset/workspace. No manual path
concatenation needed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from tqdm import tqdm

# -------------------------------------------------------------------- #
# Mind2Web2 imports
# -------------------------------------------------------------------- #
from mind2web2.llm_client.azure_openai_client import AsyncAzureOpenAIClient
from mind2web2.llm_client.openai_client import AsyncOpenAIClient
from mind2web2.utils.page_info_retrieval import BatchBrowserManager
from mind2web2.api_tools.tool_pdf import is_pdf, PDFParser
from mind2web2.utils.cache_filesys import CacheFileSys
from mind2web2.utils.logging_setup import create_logger
from mind2web2.utils.path_config import PathConfig
from mind2web2.prompts.cache_prompts import llm_extraction_prompts
from mind2web2.utils.url_tools import remove_utm_parameters, normalize_url_simple, regex_find_urls, URLs

# -------------------------------------------------------------------- #
# Constants
# -------------------------------------------------------------------- #
MAX_LLM_CONCURRENCY = 30  # Concurrent LLM calls for URL extraction


# -------------------------------------------------------------------- #
# Helpers for URL extraction
# -------------------------------------------------------------------- #

async def llm_extract_urls_with_model(
    client: AsyncAzureOpenAIClient | AsyncOpenAIClient,
    answer_text: str,
    model: str,
    llm_semaphore: asyncio.Semaphore,
    logger: Logger,
) -> List[str]:
    """Extract URLs using specified LLM model with enhanced prompt."""
    try:
        async with llm_semaphore:
            result: URLs = await client.response(
                model=model,
                messages=[{"role": "system", "content": llm_extraction_prompts}, {"role": "user", "content": answer_text}],
                response_format=URLs,
            )
        return result.urls or []
    except Exception as e:
        logger.warning(f"LLM extraction failed with model {model}: {e}")
        return []


async def llm_extract_urls_multi_model(
    client: AsyncAzureOpenAIClient | AsyncOpenAIClient,
    answer_text: str,
    llm_semaphore: asyncio.Semaphore,
    logger: Logger,
    models: List[str] = None,
) -> List[str]:
    """Extract URLs using multiple LLM models concurrently and merge results."""
    if models is None:
        models = ["o4-mini", "gpt-4.1"]

    # Run all models concurrently
    tasks = [
        llm_extract_urls_with_model(client, answer_text, model, llm_semaphore, logger)
        for model in models
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge all results
    all_urls = set()
    for result in results:
        if isinstance(result, list):
            all_urls.update(result)
        elif isinstance(result, Exception):
            logger.warning(f"Model extraction failed: {result}")

    return list(all_urls)


def filter_url_variants(urls: List[str], priorities: Dict[str, int] | None = None) -> List[str]:
    """Filter out URL variants to keep only unique URLs.

    Args:
        urls: URL candidates (duplicates allowed).
        priorities: Optional map assigning lower scores to preferred originals.
    """
    if not urls:
        return []

    # Group URLs by normalized form
    url_groups: Dict[str, List[str]] = {}
    for url in urls:
        normalized = normalize_url_simple(url)
        if normalized not in url_groups:
            url_groups[normalized] = []
        url_groups[normalized].append(url)

    # Select representative URL from each group
    unique_urls = []
    priority_lookup = priorities or {}
    default_priority = 1 if priorities else 0
    for group in url_groups.values():
        # Prefer https over http, then prefer shorter URLs
        group.sort(key=lambda u: (
            priority_lookup.get(u, default_priority),
            0 if u.startswith('https://') else 1,  # https first
            len(u),  # shorter first
            u.lower()  # alphabetical
        ))
        unique_urls.append(group[0])

    return unique_urls


async def extract_from_file(
    client: AsyncAzureOpenAIClient | AsyncOpenAIClient | None,
    ans_path: Path,
    rel_source: str,
    llm_semaphore: asyncio.Semaphore,
    logger: Logger,
    llm_models: List[str] = None,
) -> Tuple[Dict[str, List[str]], int]:
    """Enhanced URL extraction with multi-model LLM and comprehensive regex + variant filtering."""
    text = ans_path.read_text(encoding="utf-8")

    # --- Enhanced regex extraction ---
    urls_regex = regex_find_urls(text)

    # --- Multi-model LLM extraction ---
    urls_llm: List[str] = []
    if client is not None:
        urls_llm = await llm_extract_urls_multi_model(client, text, llm_semaphore, logger, llm_models)

    # --- Merge all results ---
    priorities: Dict[str, int] = {}
    for url in urls_regex:
        priorities[url] = 0
    for url in urls_llm:
        priorities.setdefault(url, 1)

    all_urls = urls_regex + urls_llm

    # --- Filter variants to avoid duplicates ---
    unique_urls = filter_url_variants(all_urls, priorities if priorities else None)

    mapping = {u: [rel_source] for u in unique_urls}
    return mapping, len(unique_urls)


# -------------------------------------------------------------------- #
# Crawling helpers
# -------------------------------------------------------------------- #

async def crawl_one_page(
    url: str,
    cache: CacheFileSys,
    pdf_parser: PDFParser,
    browser_manager: BatchBrowserManager,
    logger: Logger,
) -> None:
    """Crawl a single page using a shared browser instance."""
    try:
        # Already cached? Skip
        if cache.has(url):
            return
        url = remove_utm_parameters(url)
        logger.info(f"Crawling {url}")
        # ---------- PDF ----------
        is_pdf_or_not = await is_pdf(url)
        if is_pdf_or_not:
            try:
                await asyncio.sleep(0.2 * random.random())
                buf = await pdf_parser._fetch_pdf_bytes(url)
                if buf is not None:
                    cache.put_pdf(url, buf)
                    return
            except Exception as e:
                logger.info(f"Fail to extract PDF from {url} : {e}")

        # ---------- Web page capture (using shared browser) ----------
        if is_pdf_or_not:
            logger.info(f"⚠Try to load the Seemingly PDF file by loading online: {url}")

        shot, text = await browser_manager.capture_page(url, logger)

        # ---------- Persist ----------
        if shot and text:
            cache.put_web(url, text, shot)

    except Exception:
        logger.error(f"Error crawling {url}", exc_info=True)


# -------------------------------------------------------------------- #
# Safe wrapper with timeout
# -------------------------------------------------------------------- #
async def crawl_one_page_safe(
    url: str,
    cache: CacheFileSys,
    pdf_parser: PDFParser,
    browser_manager: BatchBrowserManager,
    logger: Logger,
    overall_timeout: int = 300,
) -> None:
    """
    Wrap `crawl_one_page()` with an overall timeout to prevent hanging.

    Args:
        overall_timeout: Maximum time in seconds for the entire page capture process.
                        This prevents a single page from hanging the entire program.
                        Playwright's internal timeouts (30s) handle navigation issues.
    """
    try:
        await asyncio.wait_for(
            crawl_one_page(url, cache, pdf_parser, browser_manager, logger),
            timeout=overall_timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Overall timeout: abandoned {url} after {overall_timeout}s to prevent program hanging")
    except Exception:
        logger.error(f"Unexpected error crawling {url}", exc_info=True)


# -------------------------------------------------------------------- #
# Utilities
# -------------------------------------------------------------------- #

def sort_ci(iterable):
    """Case-insensitive sorting."""
    return sorted(iterable, key=lambda s: s.lower())


# -------------------------------------------------------------------- #
# Main pipeline per task
# -------------------------------------------------------------------- #

async def process_cache(
    agent_name: str,
    task_id: str,
    llm_provider: str = "openai",
    max_concurrent_pages: int = 30,
    max_retries: int = 1,
    overall_timeout: int = 300,
    headless: bool = False,
    answers_root: Optional[Path] = None,
    cache_root: Optional[Path] = None,
    logger: Optional[Logger] = None,
) -> None:
    """
    1) Discover and aggregate all URLs in answers; write to <cache_root>/<agent_name>/<task_id>.json
    2) Crawl web/PDF content by unique URL; write to <cache_root>/<agent_name>/<task_id>/ directory

    Args:
        answers_root: Base directory containing answers. If None, uses PathConfig defaults.
        cache_root: Base directory for cache storage. If None, uses PathConfig defaults.
        logger: Logger instance. If None, creates a default one.
    """
    # Resolve defaults lazily
    if answers_root is None or cache_root is None:
        paths = PathConfig(Path(__file__).resolve().parent)
        if answers_root is None:
            answers_root = paths.answers_root
        if cache_root is None:
            cache_root = paths.cache_root

    if logger is None:
        logger, _ = create_logger(__name__, "tmp_logs")

    llm_semaphore = asyncio.Semaphore(MAX_LLM_CONCURRENCY)

    answer_root = answers_root / agent_name / task_id
    agent_cache_root = cache_root / agent_name
    agent_cache_root.mkdir(parents=True, exist_ok=True)

    meta_json = agent_cache_root / f"{task_id}.json"
    cache_task_dir = agent_cache_root / task_id

    # ------------------------------------------------- #
    # Step 1: URL discovery
    # ------------------------------------------------- #
    meta_data: Dict[str, Any]

    if meta_json.exists():
        logger.info(f"[{agent_name}/{task_id}] Found existing {meta_json.name}, skipping extraction ...")
        data = json.loads(meta_json.read_text("utf-8"))
        url_meta: Dict[str, List[str]] = data["urls"]
        all_unique_urls: List[str] = data["all_unique_urls"]
        meta_data = data
    else:
        # Initialize LLM client based on provider
        if llm_provider == "openai":
            client = AsyncOpenAIClient()
        elif llm_provider == "azure_openai":
            client = AsyncAzureOpenAIClient()
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
        url_meta: Dict[str, List[str]] = {}

        # All .md answer files
        answer_files = [p for p in answer_root.rglob("*.md") if p.is_file()]
        logger.info(f"[{agent_name}/{task_id}] Extracting URLs from {len(answer_files)} .md answer files ...")

        async def handle_file(p: Path):
            rel_path = p.relative_to(answer_root)
            rel_source = str(rel_path)
            mapping, _ = await extract_from_file(client, p, rel_source, llm_semaphore, logger)
            return mapping

        # Progress bar: extraction
        with tqdm(total=len(answer_files), desc="Extracting", unit="file", ncols=80) as bar:
            coros = [handle_file(p) for p in answer_files]
            for coro in asyncio.as_completed(coros):
                mapping = await coro
                for u, srcs in mapping.items():
                    url_meta.setdefault(u, []).extend(srcs)
                bar.update(1)

        # Deduplicate + sort
        url_meta = {u: sort_ci(list(set(srcs))) for u, srcs in url_meta.items()}
        ordered_items = sorted(url_meta.items(), key=lambda kv: (-len(kv[1]), kv[0].lower()))
        url_meta_ordered = {u: srcs for u, srcs in ordered_items}
        all_unique_urls = sort_ci(url_meta_ordered.keys())

        payload = {
            "agent_name": agent_name,
            "task_id": task_id,
            "total_unique_urls": len(all_unique_urls),
            "all_unique_urls": all_unique_urls,
            "urls": url_meta_ordered,
            "url_types": {},
        }
        meta_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        logger.info(f"[{agent_name}/{task_id}] Wrote URL metadata -> {meta_json}")
        url_meta = url_meta_ordered
        meta_data = payload

    # ------------------------------------------------- #
    # Step 2: Crawl & cache (using shared browser instance)
    # ------------------------------------------------- #
    logger.info(f"[{agent_name}/{task_id}] Total unique URLs to crawl: {len(all_unique_urls)}")

    pdf_parser = PDFParser()
    cache = CacheFileSys(str(cache_task_dir))

    # Use BatchBrowserManager to share browser instance; supports high concurrency
    logger.info(f"[{agent_name}/{task_id}] Headless mode: {headless}")

    async with BatchBrowserManager(
        headless=headless,
        max_concurrent_pages=max_concurrent_pages,
        max_retries=max_retries
    ) as browser_manager:
        logger.info(f"[{agent_name}/{task_id}] Browser manager initialized")

        tasks = [
            crawl_one_page_safe(u, cache, pdf_parser, browser_manager, logger, overall_timeout=overall_timeout)
            for u in all_unique_urls
        ]
        with tqdm(total=len(tasks), desc="Crawling", unit="url", ncols=80) as bar:
            for coro in asyncio.as_completed(tasks):
                await coro
                bar.update(1)

        logger.info(f"[{agent_name}/{task_id}] Browser manager will be cleaned up automatically")

    cache.save()

    # Update metadata with cached content types
    try:
        url_types: Dict[str, str] = {}
        for url in all_unique_urls:
            content_type = cache.has(url)
            if content_type:
                url_types[url] = content_type

        meta_data.update({
            "agent_name": agent_name,
            "task_id": task_id,
            "total_unique_urls": len(all_unique_urls),
            "all_unique_urls": all_unique_urls,
            "urls": url_meta,
            "url_types": url_types,
            "cached_url_count": len(url_types),
        })
        meta_json.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2), "utf-8")
        logger.info(f"[{agent_name}/{task_id}] Updated metadata with cache types -> {meta_json}")
    except Exception:
        logger.error(f"[{agent_name}/{task_id}] Failed to update metadata with cache types", exc_info=True)

    logger.info(f"[{agent_name}/{task_id}] Saved updated cache -> {cache_task_dir}")


# -------------------------------------------------------------------- #
# Entry point
# -------------------------------------------------------------------- #

def _strip_suffixes(task_id: str) -> str:
    """If CLI argument mistakenly includes .json/.pkl, strip it automatically."""
    suffixes = (".json", ".pkl")
    for s in suffixes:
        if task_id.endswith(s):
            return task_id[: -len(s)]
    return task_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch crawl pages and cache results using CacheFileSys (v2)")
    parser.add_argument("agent_name", help="Agent name (e.g., chatgpt_agent)")
    parser.add_argument("task_id", help="Task ID")
    parser.add_argument(
        "--llm_provider",
        choices=["openai", "azure_openai"],
        default="openai",
        help="LLM provider (openai or azure_openai, default: openai)"
    )
    parser.add_argument(
        "--max_concurrent_pages",
        type=int,
        default=5,
        help="Maximum number of concurrent pages to process (default: 5)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="Maximum number of retries per page (default: 1)"
    )
    parser.add_argument(
        "--overall_timeout",
        type=int,
        default=120,
        help="Overall timeout in seconds for each page capture to prevent hanging (default: 120s)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (default: headful)"
    )

    args = parser.parse_args()

    task_id = _strip_suffixes(args.task_id)
    asyncio.run(process_cache(
        agent_name=args.agent_name,
        task_id=task_id,
        llm_provider=args.llm_provider,
        max_concurrent_pages=args.max_concurrent_pages,
        max_retries=args.max_retries,
        overall_timeout=args.overall_timeout,
        headless=args.headless,
    ))
