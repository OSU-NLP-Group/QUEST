"""Configuration for the web-based cache manager."""

from __future__ import annotations
from pathlib import Path

# Paths
PACKAGE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PACKAGE_DIR / "frontend"

# Server
DEFAULT_PORT = 8000
CORS_ORIGINS = ["*"]  # localhost only, safe for local tool

# Capture
MAX_SCREENSHOT_SIZE = 20 * 1024 * 1024  # 20MB max screenshot
MAX_TEXT_SIZE = 50 * 1024 * 1024  # 50MB max text
