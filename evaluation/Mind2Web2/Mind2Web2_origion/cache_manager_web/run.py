#!/usr/bin/env python3
"""Entry point for the web-based Cache Manager.

Usage:
    python run.py [agent_name_or_path] [--port 8000]

Examples:
    python run.py zhoukai                          # Agent name (resolves to cache/zhoukai)
    python run.py /path/to/agent/cache/folder      # Full path

Opens the Cache Manager web UI in your default browser.
"""

import sys
import argparse
import webbrowser
from pathlib import Path

# Ensure project root is on the path so we can import mind2web2
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

CACHE_ROOT = project_root / "cache"


def resolve_cache_folder(arg: str) -> Path:
    """Resolve an agent name or path to a cache folder.

    If the argument is an existing directory, use it directly.
    Otherwise, treat it as an agent name under <project_root>/cache/.
    """
    path = Path(arg)

    # If it's already a valid directory path, use it as-is
    if path.is_dir():
        return path.resolve()

    # Treat as agent name: look under cache root
    agent_path = CACHE_ROOT / arg
    if agent_path.is_dir():
        return agent_path.resolve()

    # Not found — list available agents and exit
    available = sorted(d.name for d in CACHE_ROOT.iterdir() if d.is_dir()) if CACHE_ROOT.is_dir() else []
    msg = f"Cache folder not found: '{arg}'"
    if available:
        msg += f"\nAvailable agents: {', '.join(available)}"
    else:
        msg += f"\nNo agent folders found in {CACHE_ROOT}"
    print(msg, file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Cache Manager Web UI")
    parser.add_argument("agent", nargs="?", default=None,
                        help="Agent name (e.g. 'zhoukai') or full path to cache folder")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    # Store startup cache folder in environment so the app can read it
    import os
    if args.agent:
        cache_folder = resolve_cache_folder(args.agent)
        os.environ["CM_INITIAL_CACHE_FOLDER"] = str(cache_folder)

    # Open browser after a short delay
    if not args.no_browser:
        import threading
        def open_browser():
            import time
            time.sleep(1.0)
            webbrowser.open(f"http://{args.host}:{args.port}")
        threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(
        "cache_manager_web.backend.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
