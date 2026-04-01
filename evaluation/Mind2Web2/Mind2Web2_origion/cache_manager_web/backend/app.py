"""FastAPI application for the web-based Cache Manager."""

from __future__ import annotations
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from .models import CacheManager, KeywordDetector
from .config import FRONTEND_DIR, CORS_ORIGINS
from .api.routes import router, set_app_state

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    cm = CacheManager()
    kd = KeywordDetector()
    set_app_state(cm, kd)

    # Auto-load cache folder if specified via env
    initial_folder = os.environ.get("CM_INITIAL_CACHE_FOLDER")
    if initial_folder and Path(initial_folder).is_dir():
        try:
            cm.load_agent_cache(initial_folder)
            logger.info(f"Auto-loaded cache from {initial_folder}")
        except Exception as e:
            logger.warning(f"Failed to auto-load cache: {e}")

    yield


app = FastAPI(title="Cache Manager", lifespan=lifespan)

# CORS — allow the Chrome extension (and any localhost origin) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api")

# Serve frontend static files
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serve the main SPA page."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/favicon.ico")
async def favicon():
    """Return an empty favicon to avoid 404 in browser console."""
    return Response(content=b"", media_type="image/x-icon")
