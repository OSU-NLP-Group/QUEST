# utils — Shared Utilities

## Modules

### cache_filesys.py — File-Based Webpage Cache
`CacheFileSys`: One instance per task, stores cached webpage content on disk.

**Directory layout:**
```
task_dir/
├── index.json      # {url: "web"|"pdf"}
├── <md5_hash>.txt  # page text content
├── <md5_hash>.jpg  # page screenshot
├── <md5_hash>.pdf  # PDF content
```

**Key methods:**
- `put_web(url, text, screenshot)`: Store webpage (converts screenshot to JPG)
- `put_pdf(url, pdf_bytes)`: Store PDF
- `get_web(url)` → `(text, screenshot_bytes)`
- `get_pdf(url)` → `pdf_bytes`
- `has(url)` → `"web"` | `"pdf"` | `None`
- `save()`: Persist index.json to disk

**URL matching** is the most complex part. `_find_url()` tries:
1. Direct lookup
2. Normalized form (`normalize_url_simple`)
3. Reverse normalized comparison against all stored URLs
4. Full variant expansion (`_get_url_variants`) — generates 100+ variants per URL by combining scheme swaps, encoding variants, www prefix, utm params, trailing slashes

### page_info_retrieval.py — Browser-Based Web Capture
**`BatchBrowserManager`**: Manages a shared Chromium browser (via patchright) for concurrent page capture.
- `capture_page(url, logger)` → `(screenshot_b64, text_content)`
- Uses CDP (Chrome DevTools Protocol) for efficient screenshot + HTML capture
- Converts HTML to markdown via `html2text`
- Scrolls pages to trigger lazy-loaded content
- Auto-restarts browser on crashes
- Concurrency controlled by internal semaphore (`max_concurrent_pages`)

**`PageManager`**: Manages active pages within a browser context, handles page close/crash/navigation events.

### path_config.py — Centralized Path Management
`PathConfig` dataclass holding all project-relative directories:
- `project_root`, `answers_root`, `eval_scripts_root`, `eval_results_root`, `cache_root`
- `default_script_for(task_id)` → `eval_scripts/<version>/<task_id>.py`
- `apply_overrides()`: Override any path via CLI args

### logging_setup.py — Structured Logging
`create_logger(name, log_folder)` creates loggers with multiple handlers:
- **JSONL file**: Machine-readable structured logs
- **Readable file**: Human-readable format with timestamps
- **Console**: Colored structured output (optional)
- **Shared error handler**: Cross-logger error display for concurrent evaluation

Custom formatters:
- `ColoredStructuredFormatter`: Colored console output with op_id/node context
- `HumanReadableFormatter`: File logs with structured field display
- `CompactJsonFormatter`: Compact JSONL for machine parsing

### url_tools.py — URL Normalization & Extraction
- `normalize_url_simple(url)`: Normalize for comparison (lowercase, remove www/utm/fragments/trailing slash, force https)
- `remove_utm_parameters(url)`: Strip all `utm_*` query params
- `normalize_url_for_browser(url)`: Ensure URL has protocol for navigation
- `regex_find_urls(text)`: Extract URLs from markdown text using multiple regex patterns
- `URLs` Pydantic model: For LLM structured output of URL lists

### load_eval_script.py — Dynamic Script Loading
`load_eval_script(path)`: Dynamically imports a Python file and returns its `evaluate_answer` coroutine.
- Validates the function exists, is async, and has required parameters
- Uses unique module names to avoid namespace collisions

### misc.py — Small Helpers
- `normalize_url_markdown(url)`: Remove markdown escape chars from URLs
- `text_dedent(str)`: `textwrap.dedent().strip()`
- `encode_image(path)` / `encode_image_buffer(bytes)`: Base64 encoding
- `extract_doc_description(docstring)`: Extract description portion of a docstring
