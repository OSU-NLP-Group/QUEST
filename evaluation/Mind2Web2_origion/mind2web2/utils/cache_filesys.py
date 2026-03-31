from __future__ import annotations

import logging
import os
import json
import hashlib
import base64
from typing import Literal, List, Dict, Any, Optional, Tuple
from urllib.parse import urldefrag, quote, unquote, quote_plus
from PIL import Image
import io
from .url_tools import normalize_url_simple, remove_utm_parameters

ContentType = Literal["web", "pdf"]


class CacheFileSys:
    """Single-task file system cache with lazy loading.
    
    Each instance handles one task's cached content. URLs are stored as either
    'web' (text + screenshot) or 'pdf'. Files are named using URL hashes.
    
    Directory structure:
    task_dir/
    ├── index.json          # {"url1": "web", "url2": "pdf"}
    ├── <hash1>.txt         # text content
    ├── <hash1>.jpg         # screenshot
    ├── <hash2>.pdf         # pdf content
    └── ...
    """

    def __init__(self, task_dir: str):
        """Initialize cache for a single task.
        
        Args:
            task_dir: Directory path for this specific task's cache
        """
        self.task_dir = os.path.abspath(task_dir)
        self.index_file = os.path.join(self.task_dir, "index.json")
        self.urls: Dict[str, ContentType] = {}  # url -> "web"/"pdf"
        self._variant_cache: Dict[str, List[str]] = {}  # url -> variants

        # Create task directory if it doesn't exist
        os.makedirs(self.task_dir, exist_ok=True)
        
        # Load index immediately
        self._load_index()
    
    def _get_url_hash(self, url: str) -> str:
        """Generate consistent hash for URL to use as filename."""
        normalized_url = self._remove_frag_and_slash(url)
        return hashlib.md5(normalized_url.encode('utf-8')).hexdigest()
    
    def _remove_frag_and_slash(self, url: str) -> str:
        """Normalize URL to a consistent format for storage"""
        url_no_frag, _ = urldefrag(url)
        decoded = unquote(url_no_frag)
        if decoded.endswith('/') and len(decoded) > 1 and not decoded.endswith('://'):
            decoded = decoded[:-1]
        return decoded

    def _get_url_variants(self, url: str) -> List[str]:
        """Generate all possible variants of URL for matching."""
        if url in self._variant_cache:
            return self._variant_cache[url]

        def swap_scheme(u: str):
            if u.startswith("http://"):
                return "https://" + u[7:]
            if u.startswith("https://"):
                return "http://" + u[8:]
            return None

        url_no_frag, _ = urldefrag(url)
        base_urls: set[str] = {
            url, url_no_frag, remove_utm_parameters(url), remove_utm_parameters(url_no_frag),
            f"{url}?utm_source=chatgpt.com", f"{url_no_frag}?utm_source=chatgpt.com",
            f"{url}?utm_source=openai.com", f"{url_no_frag}?utm_source=openai.com",
        }

        if not url.endswith("/"):
            base_urls.add(f"{url}/?utm_source=chatgpt.com")
        if not url_no_frag.endswith("/"):
            base_urls.add(f"{url_no_frag}/?utm_source=chatgpt.com")

        if not url.endswith("/"):
            base_urls.add(f"{url}/?utm_source=openai.com")
        if not url_no_frag.endswith("/"):
            base_urls.add(f"{url_no_frag}/?utm_source=openai.com")

        if url.startswith("http://www."):
            base_urls.add("http://" + url[11:])
        elif url.startswith("https://www."):
            base_urls.add("https://" + url[12:])
        else: #TODO: how do we handle this?
            pass

        for u in list(base_urls):
            swapped = swap_scheme(u)
            if swapped:
                base_urls.add(swapped)

        variants = []
        for base_url in base_urls:
            try:
                original = base_url
                encoded_default = quote(base_url)
                encoded_basic = quote(base_url, safe=':/?#')
                encoded_common = quote(base_url, safe=':/?#@!$&\'*+,;=')
                encoded_brackets = quote(base_url, safe=':/?#[]@!$&\'*+,;=')
                encoded_rfc = quote(base_url, safe=':/?#[]@!$&\'()*+,;=')
                encoded_minimal = quote(base_url, safe=':/')
                encoded_plus = quote_plus(base_url, safe=':/?#[]@!$&\'()*+,;=')
                decoded_url = unquote(base_url)

                encoding_variants = [
                    original, encoded_default, encoded_basic, encoded_common,
                    encoded_brackets, encoded_rfc, encoded_minimal, encoded_plus, decoded_url
                ]

                for url_variant in encoding_variants:
                    variants.append(url_variant)
                    if url_variant.endswith("/") and len(url_variant) > 1 and not url_variant.endswith('://'):
                        variants.append(url_variant[:-1])
                    elif not url_variant.endswith('/'):
                        variants.append(url_variant + "/")
            except Exception:
                variants.append(base_url)
                if base_url.endswith("/") and len(base_url) > 1 and not base_url.endswith('://'):
                    variants.append(base_url[:-1])
                elif not base_url.endswith('/'):
                    variants.append(base_url + "/")

        # Deduplicate while maintaining order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)

        self._variant_cache[url] = unique_variants
        return unique_variants

    def _load_index(self):
        """Load the index file and verify file integrity."""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    loaded_urls = json.load(f)  # Direct load: {url: type}
            except (IOError, json.JSONDecodeError) as e:
                logging.getLogger(__name__).warning(f"Failed to load index: {e}. Starting with empty index.")
                loaded_urls = {}
        else:
            loaded_urls = {}
        
        # Verify file integrity and keep only URLs with existing files
        self.urls = {}
        for url, content_type in loaded_urls.items():
            url_hash = self._get_url_hash(url)
            files_exist = True
            
            if content_type == "web":
                text_file = os.path.join(self.task_dir, f"{url_hash}.txt")
                screenshot_file = os.path.join(self.task_dir, f"{url_hash}.jpg")
                if not (os.path.exists(text_file) and os.path.exists(screenshot_file)):
                    files_exist = False
            elif content_type == "pdf":
                pdf_file = os.path.join(self.task_dir, f"{url_hash}.pdf")
                if not os.path.exists(pdf_file):
                    files_exist = False
            
            if files_exist:
                self.urls[url] = content_type
            else:
                logging.getLogger(__name__).warning(f"Missing files for URL {url}, removing from index")

    def _find_url(self, url: str) -> Optional[str]:
        """Find stored URL that matches input URL (handling variants)."""
        
        # Direct lookup
        if url in self.urls:
            return url
        
        # Try normalized
        normalized = normalize_url_simple(url)
        if normalized in self.urls:
            return normalized
        

        # Reverse search - check if any stored URL normalizes to same as input
        normalized_input = normalize_url_simple(url)
        for stored_url in self.urls:
            try:
                if normalize_url_simple(stored_url) == normalized_input:
                    return stored_url
            except Exception:
                continue

        # Try all variants
        variants = self._get_url_variants(url)
        for variant in variants:
            if variant in self.urls:
                return variant
        
        return None

    def _convert_image_to_jpg(self, image_data: str | bytes, quality: int = 85) -> bytes:
        """Convert image data to JPG format for storage efficiency."""
        try:
            if isinstance(image_data, str):
                if image_data.startswith('data:image/'):
                    image_data = image_data.split(',', 1)[1]
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
                
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                if image.mode in ('RGBA', 'LA'):
                    background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
                
            jpg_buffer = io.BytesIO()
            image.save(jpg_buffer, format='JPEG', quality=quality, optimize=True)
            return jpg_buffer.getvalue()
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Error converting image to JPG: {e}")
            if isinstance(image_data, str):
                if image_data.startswith('data:image/'):
                    image_data = image_data.split(',', 1)[1]
                return base64.b64decode(image_data)
            return image_data

    # Public API methods
    def put_web(self, url: str, text: str, screenshot: str | bytes):
        """Store web page content (text + screenshot)."""
        url_hash = self._get_url_hash(url)
        
        # Save text file
        text_file = os.path.join(self.task_dir, f"{url_hash}.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Convert and save screenshot as JPG
        jpg_data = self._convert_image_to_jpg(screenshot)
        screenshot_file = os.path.join(self.task_dir, f"{url_hash}.jpg")
        with open(screenshot_file, 'wb') as f:
            f.write(jpg_data)
        
        # Update index (safe because each async handles different URLs)
        self.urls[self._remove_frag_and_slash(url)] = "web"

    def put_pdf(self, url: str, pdf_bytes: bytes):
        """Store PDF content."""
        url_hash = self._get_url_hash(url)
        
        # Save PDF file
        pdf_file = os.path.join(self.task_dir, f"{url_hash}.pdf")
        with open(pdf_file, 'wb') as f:
            f.write(pdf_bytes)
        
        # Update index (safe because each async handles different URLs)
        self.urls[self._remove_frag_and_slash(url)] = "pdf"

    def get_web(self, url: str, get_screenshot=True) -> Tuple[str, bytes]:
        """Get web page content (text, screenshot_bytes). Raises error if not found."""
        stored_url = self._find_url(url)
        if not stored_url or self.urls[stored_url] != "web":
            raise KeyError(f"No web content found for URL: {url}")
        
        url_hash = self._get_url_hash(stored_url)
        
        # Load text (files are guaranteed to exist due to integrity check)
        text_file = os.path.join(self.task_dir, f"{url_hash}.txt")
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Load screenshot
        if get_screenshot:
            screenshot_file = os.path.join(self.task_dir, f"{url_hash}.jpg")
            with open(screenshot_file, 'rb') as f:
                screenshot_bytes = f.read()
        else:
            screenshot_bytes = None
        
        return text, screenshot_bytes

    def get_pdf(self, url: str) -> bytes:
        """Get PDF content. Raises error if not found."""
        stored_url = self._find_url(url)
        if not stored_url or self.urls[stored_url] != "pdf":
            raise KeyError(f"No PDF content found for URL: {url}")
        
        url_hash = self._get_url_hash(stored_url)
        
        # Load PDF (file is guaranteed to exist due to integrity check)
        pdf_file = os.path.join(self.task_dir, f"{url_hash}.pdf")
        with open(pdf_file, 'rb') as f:
            return f.read()

    def has(self, url: str) -> ContentType | None:
        """Check what type of content exists for URL.
        
        Returns:
            "web" if web content exists
            "pdf" if PDF content exists  
            None if no content exists
        """
        stored_url = self._find_url(url)
        if stored_url is not None:
            return self.urls[stored_url]
        return None

    def has_web(self, url: str) -> bool:
        """Check if web content exists for URL."""
        return self.has(url) == "web"

    def has_pdf(self, url: str) -> bool:
        """Check if PDF content exists for URL."""
        return self.has(url) == "pdf"

    def get_all_urls(self) -> List[str]:
        """Get all stored URLs."""
        return list(self.urls.keys())

    def summary(self) -> Dict[str, Any]:
        """Get cache summary."""
        web_count = sum(1 for content_type in self.urls.values() if content_type == "web")
        pdf_count = sum(1 for content_type in self.urls.values() if content_type == "pdf")
        
        return {
            "total_urls": len(self.urls),
            "web_pages": web_count,
            "pdf_pages": pdf_count,
        }

    def save(self):
        """Save the index to disk."""
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.urls, f, indent=2, ensure_ascii=False)  # Direct save: {url: type}

    def clear(self):
        """Clear all cached content."""
        if os.path.exists(self.task_dir):
            import shutil
            shutil.rmtree(self.task_dir)
            os.makedirs(self.task_dir, exist_ok=True)
        
        self.urls.clear()
        self._variant_cache.clear()