# Cache Manager Web

Web-based tool for reviewing and fixing cached web pages used by Mind2Web agents. Uses a browser-based UI + Chrome Extension.

## Architecture

```
cache_manager_web/
├── run.py                     # Entry point: starts FastAPI server, auto-opens browser
├── backend/
│   ├── app.py                 # FastAPI app, lifespan, CORS, static file serving
│   ├── config.py              # Constants (paths, limits)
│   ├── api/routes.py          # ALL API endpoints + SSE + MHTML parsing
│   └── models/
│       ├── cache_manager.py   # CacheManager — reads/writes cache directory structure
│       └── keyword_detector.py # KeywordDetector — scans text for issue keywords
├── frontend/
│   ├── index.html             # Single-page app shell (no framework, no build step)
│   ├── css/style.css          # Complete design system with CSS custom properties
│   └── js/
│       ├── main.js            # Init, toolbar, keyboard shortcuts, SSE, drag-drop
│       ├── actions.js         # Shared actions (selectTask, selectUrl, toast, etc.)
│       ├── store.js           # Reactive state store with selective subscriptions
│       ├── api.js             # Fetch-based API client
│       └── components/
│           ├── task-panel.js  # Task list with search/filter
│           ├── url-list.js    # URL list with filters, progress bar
│           └── preview.js     # Screenshot/text/answer preview with zoom
└── extension/
    ├── manifest.json          # Chrome Extension Manifest V3
    ├── background.js          # Service worker: capture, batch mode, CAPTCHA detection
    └── popup.html/js          # Extension popup UI with batch progress display
```

## Key Design Decisions

- **No build step**: Vanilla JS with ES modules. Files are served directly by FastAPI's StaticFiles.
- **No circular imports**: Components import shared actions from `actions.js`, NOT from `main.js`. This is critical — `main.js` imports components, so components must not import from `main.js`.
- **Selective state subscriptions**: `subscribe(fn, ['key1', 'key2'])` — components only re-render when their relevant keys change.
- **Chrome Extension for capture**: Uses a real browser session (not Playwright/Selenium) so it works on Cloudflare-protected and anti-bot pages.
- **SSE for real-time updates**: When the extension captures a page, the frontend updates instantly.
- **contentVersion cache busting**: Screenshot URLs include `&v={contentVersion}` to force browser to re-fetch after capture.
- **MHTML parsing without Qt**: Uses Python's `email` module to parse MHTML (MIME format).

## Running

```bash
uv run python3 cache_manager_web/run.py zhoukai              # Agent name
uv run python3 cache_manager_web/run.py /path/to/cache/folder # Full path
# Options: --port 8000  --host 127.0.0.1  --no-browser
```

## Package Management

This project uses `uv`, not pip. Use `uv run`, `uv sync`, `uv add`.

## API Endpoints (routes.py)

| Method | Path | Purpose |
|--------|------|---------|
| POST | /api/load | Load cache folder, run issue scan |
| GET | /api/status | Current load status |
| GET | /api/tasks | Task list with summaries |
| GET | /api/tasks/{id}/urls | URLs with issue detection, reviewed status |
| GET | /api/content/{id}/text | Text content + issues |
| GET | /api/content/{id}/screenshot | Screenshot JPEG |
| GET | /api/content/{id}/pdf | PDF content |
| POST/GET | /api/capture/target | Active capture target for extension |
| POST | /api/capture/batch/start | Start batch capture with URL queue |
| GET | /api/capture/batch/status | Batch progress (polled by extension) |
| POST | /api/capture/batch/skip | Skip current URL (on failure) |
| POST | /api/capture/batch/stop | Stop batch capture |
| POST | /api/capture/batch/captcha | CAPTCHA detected notification |
| POST | /api/capture | Receive capture from extension |
| POST | /api/flag/{id} | Flag URL as issue (web: replace text; PDF: flags.json) |
| POST | /api/reset/{id} | Reset URL cache (clear content + auto-flag) |
| GET | /api/review/{id} | Get review statuses for a task |
| POST | /api/review/{id} | Set review status |
| GET | /api/review-progress | Overall progress |
| GET | /api/answers/{id} | Answer markdown files |
| POST | /api/urls/{id} | Add URL to task (auto_flag, PDF suffix detection) |
| POST | /api/urls/{id}/rename | Rename/edit URL link (moves content) |
| POST | /api/urls/{id}/pdf | Add PDF URL to task |
| DELETE | /api/urls/{id} | Delete URL |
| POST | /api/upload-mhtml/{id} | Upload MHTML |
| POST | /api/upload-pdf/{id} | Upload PDF (replaces content, switches type) |
| POST | /api/scan | Re-scan all tasks for issues |
| GET | /api/events | SSE stream |

## State Store (store.js)

Key state fields:
- `loaded`, `agentName`, `agentPath`, `stats` — cache status
- `tasks`, `taskIssues`, `selectedTaskId` — task list
- `urls`, `selectedUrl`, `urlTotal`, `urlReviewedCount` — URL list
- `previewMode`, `currentText`, `currentIssues`, `answers` — preview
- `issueIndex`, `issueCursor` — cross-task issue navigation
- `contentVersion` — incremented on capture to bust screenshot cache
- `batchActive`, `batchTotal`, `batchCompleted` — batch capture state
- `fitToWidth`, `zoomLevel` — screenshot zoom

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `j` / `↓` | Next URL |
| `k` / `↑` | Previous URL |
| `n` | Next issue (cross-task) |
| `N` | Previous issue (cross-task) |
| `r` / `Ctrl+Enter` | Mark as reviewed |
| `f` | Flag as issue (red) |
| `d` / `Backspace` | Delete URL |
| `x` | Reset URL cache & flag |
| `e` | Edit URL link |
| `a` | Add new URL |
| `o` | Open in browser |
| `u` / `Ctrl+U` | Recapture live |
| `1` / `2` / `3` | Screenshot / Text / Answer view |
| `Space` | Toggle Screenshot / Text |
| `Ctrl+O` | Open cache folder |
| `Ctrl+Wheel` | Zoom screenshot |
| `?` | Show shortcuts help |
| `Escape` | Close shortcuts modal |

## Common Tasks for Contributors

**Adding a new API endpoint**: Add to `backend/api/routes.py`, add client function in `frontend/js/api.js`.

**Adding a new UI action**: Add to `frontend/js/actions.js` (NOT main.js) if components need it. Components import from actions.js.

**Adding state**: Add default in `store.js`, subscribe in the relevant component with key list.

**Changing the extension**: Edit files in `extension/`, then reload in `chrome://extensions/`.

## Review Statuses

| Status | Meaning | Border Color | Counts as Fixed? |
|--------|---------|-------------|-----------------|
| `""` | Not reviewed | grey/yellow/red | No |
| `"ok"` | Reviewed OK | green | Yes |
| `"fixed"` | Single-capture fixed | green | Yes |
| `"skip"` | Skipped | green | Yes |
| `"recaptured"` | Batch-recaptured, needs human review | blue | No |

## Batch Capture Features

- **Two modes**: Auto (captures CAPTCHA pages and moves on) and Pause-on-CAPTCHA (waits for manual solving)
- **Retry logic**: Pages with body < 200 chars are auto-retried up to 2 times
- **15s timeout**: Force-captures after 15s if page hasn't loaded
- **URL redirect handling**: `actual_url` field saves content for both original and redirected URLs
- **CAPTCHA detection**: Cloudflare, Turnstile, reCAPTCHA, hCaptcha, generic blocked pages
- **Rich popup UI**: Live progress bar, status badge, current URL, scrollable log
- **Skip on failure**: Failed captures skip and advance to prevent infinite loops

## Gotchas

- `Ctrl+R` conflicts with browser refresh — don't use it as a shortcut.
- The extension needs `activeTab` + `scripting` + `tabs` + `<all_urls>` permissions for batch mode.
- Screenshot browser caching: always use `contentVersion` in screenshot URLs.
- MHTML upload uses Python's `email` module parser.
- `"recaptured"` status is NOT counted in progress — these URLs still need human review.
- `captureVisibleTab` captures the active tab, not a specific tab — must activate the target tab first.
