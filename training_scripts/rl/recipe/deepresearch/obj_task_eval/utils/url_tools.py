from urllib.parse import urldefrag, unquote, urlparse, parse_qs, urlencode, urlunparse

import argparse
import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from urllib.parse import urlparse
from urllib.parse import urldefrag, unquote
from pydantic import BaseModel
from tqdm import tqdm
import validators
from urllib.parse import urldefrag, unquote, urlparse, parse_qs, urlencode, urlunparse

class URLs(BaseModel):
    urls: List[str]

def _is_valid_url(u: str) -> bool:
    return validators.url(u, public=True) is True

def remove_utm_parameters(url: str) -> str:
    """Remove all UTM tracking parameters from URL."""
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    parsed = urlparse(url)

    # If there are no query parameters, return original URL
    if not parsed.query:
        return url

    # Parse query parameters
    params = parse_qs(parsed.query, keep_blank_values=True)

    # Filter out all utm_* parameters
    filtered_params = {k: v for k, v in params.items() if not k.startswith('utm_')}

    # Reconstruct query string
    new_query = urlencode(filtered_params, doseq=True)

    # Reconstruct URL
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment
    ))



def normalize_url_simple(url: str) -> str:
    """Simple URL normalization for variant detection."""

    url=remove_utm_parameters(url)
    # Remove fragment
    url_no_frag, _ = urldefrag(url)

    # Decode URL encoding
    decoded = unquote(url_no_frag)

    # Remove trailing slash (except for root)
    if decoded.endswith('/') and len(decoded) > 1 and not decoded.endswith('://'):
        decoded = decoded[:-1]

    # # Remove common tracking parameters
    # if decoded.endswith('?utm_source=chatgpt.com'):
    #     decoded = decoded[:-len('?utm_source=chatgpt.com')]


    # Remove all UTM parameters
    parsed = urlparse(decoded)
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=True)
        # Filter out all utm_* parameters
        filtered_params = {k: v for k, v in params.items() if not k.startswith('utm_')}
        # Reconstruct query string
        new_query = urlencode(filtered_params, doseq=True)
        decoded = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment
        ))

    # Normalize scheme
    if decoded.startswith('http://'):
        decoded = 'https://' + decoded[7:]

    # Remove www prefix for comparison
    if '://www.' in decoded:
        decoded = decoded.replace('://www.', '://')

    return decoded.lower()



def normalize_url_for_browser(url: str) -> str:
    """Simple URL normalization for variant detection."""


    url=remove_utm_parameters(url)
    # Remove fragment
    if not url.startswith(('http://', 'https://', 'ftp://')):
        return f'https://{url}'
    return url

def regex_find_urls(text: str) -> List[str]:
    """Enhanced regex extraction for comprehensive URL discovery."""
    urls = set()

    # 1. Standard markdown links: [text](url)
    urls.update(
        m for m in re.findall(r"\[.*?\]\((https?://[^\s)]+)\)", text)
        if _is_valid_url(m)
    )

    # 2. Standard full URLs with protocol
    urls.update(
        m for m in re.findall(
            r"\bhttps?://[A-Za-z0-9\-.]+\.[A-Za-z]{2,}(?:/[^\s<>\"'`{}|\\^\[\]]*)?\b",
            text
        )
        if _is_valid_url(m)
    )

    # 3. URLs without protocol (www.example.com)
    www_matches = re.findall(
        r"\bwww\.[A-Za-z0-9\-.]+\.[A-Za-z]{2,}(?:/[^\s<>\"'`{}|\\^\[\]]*)?\b",
        text
    )
    for match in www_matches:
        # Always prefer https for www domains
        urls.add(f"https://{match}")


    # 4. URLs in quotes or parentheses
    quote_patterns = [
        r'"(https?://[^"\s]+)"',
        r"'(https?://[^'\s]+)'",
        r"\((https?://[^)\s]+)\)",
        r"<(https?://[^>\s]+)>"
    ]
    for pattern in quote_patterns:
        urls.update(
            m for m in re.findall(pattern, text)
            if _is_valid_url(m)
        )

    # Clean URLs by removing trailing punctuation
    cleaned_urls = set()
    for url in urls:
        # Remove trailing punctuation that might be captured accidentally
        cleaned_url = re.sub(r'[.,;:!?\)\]}>"\'\u201d\u201c]*$', '', url)
        if cleaned_url and _is_valid_url(cleaned_url):
            cleaned_urls.add(cleaned_url)

    return list(cleaned_urls)