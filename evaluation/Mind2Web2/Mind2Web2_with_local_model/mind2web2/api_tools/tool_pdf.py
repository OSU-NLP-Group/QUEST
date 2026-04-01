# pdf_parser.py  ---------------------------------------------------------
"""
Lightweight PDF parser:
    * extract()   - Pass URL / local path / bytes, asynchronously returns (imgs, text)
    * If download or parsing fails, always returns (None, None)
    * imgs: screenshot of each page (JPEG, base64), up to 50 pages
    * text: all plain text, up to 100 pages
Dependencies:
    pip install aiohttp pymupdf pillow
"""

import asyncio
import base64
import random
import ssl
from io import BytesIO
from logging import Logger
from typing import List, Tuple, Union, Optional
from urllib.parse import urlparse, unquote
import certifi

import aiohttp
import certifi
import fitz  # PyMuPDF
import httpx
import requests
from PIL import Image
from ..utils.url_tools import remove_utm_parameters,normalize_url_for_browser

def make_blank_png_b64() -> str:
    # Create 1×1 RGBA fully transparent pixel
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    # Convert to base64 and remove line breaks
    return base64.b64encode(buf.getvalue()).decode()


# ------------------ Constants ------------------
PDF_MAGIC = b"%PDF-"  # PDF file header
UA_CHROME = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# User-agent strings for PDF detection
USER_AGENT_STRINGS = [
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0',
]


# ================================ PDF Detection Functions ================================


def is_pdf_by_suffix(url: str) -> bool:
    """Check if URL likely points to PDF based on path/query patterns."""
    parsed = urlparse(url.lower())
    path = unquote(parsed.path)

    # Direct .pdf extension
    if path.endswith('.pdf'):
        return True

    # Common PDF URL patterns
    pdf_patterns = [
        'arxiv.org/pdf/',
        '/download/pdf',
        '/fulltext.pdf',
        '/article/pdf',
        '/content/pdf',
        'type=pdf',
        'format=pdf',
        'download=pdf',
        '.pdf?',
        '/pdf/',
        'pdfviewer',
    ]

    url_lower = url.lower()
    return any(pattern in url_lower for pattern in pdf_patterns)



def is_pdf_by_requests_head(url: str) -> bool:
    """Check via HEAD request whether URL is a PDF, with strict certificate verification."""
    try:
        r = requests.head(
            url,
            allow_redirects=True,
            timeout=10,
            verify=certifi.where()  # Use certifi's root CA bundle
        )
        ct = r.headers.get("content-type", "").lower()
        return "pdf" in ct
    except requests.RequestException as e:
        # If some sites have certificate issues, you can log it
        # print(f"HEAD request failed for {url}: {e}")
        return False

async def is_pdf_by_httpx_get_range(url: str, timeout: int = 10) -> bool:
    """Check PDF via partial GET request to read file header."""
    try:
        # Configure httpx with custom SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                verify=False
        ) as client:

            headers = {
                "User-Agent": random.choice(USER_AGENT_STRINGS),
                "Range": "bytes=0-1023",  # Get first 1KB to check magic number
                "Accept": "*/*",
            }

            r = await client.get(url, headers=headers)

            # First check Content-Type
            ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
            if "pdf" in ctype:
                return True

            # If we got content, check PDF magic number
            if r.content:
                # PDF files start with %PDF-
                return r.content.startswith(b'%PDF-')

    except httpx.TimeoutException:
        print(f"[is_pdf_httpx_get_range] Timeout for {url}")
        return False
    except httpx.ConnectError:
        print(f"[is_pdf_httpx_get_range] Connection error for {url}")
        return False
    except Exception as e:
        print(f"[is_pdf_httpx_get_range] Error for {url}: {type(e).__name__}: {e}")
        return False


async def is_pdf_by_full_get(url: str, timeout: int = 15) -> bool:
    """Last resort: download beginning of file to check magic number."""
    try:
        async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=timeout,
                verify=False
        ) as client:

            headers = {
                "User-Agent": random.choice(USER_AGENT_STRINGS),
                "Accept": "*/*",
            }

            # Stream the response to avoid downloading large files
            async with client.stream('GET', url, headers=headers) as response:
                # Read first chunk to check PDF magic number
                chunk_data = b""
                async for chunk in response.aiter_bytes(chunk_size=5):
                    chunk_data += chunk
                    if len(chunk_data) >= 5:
                        break

                if chunk_data and chunk_data.startswith(b'%PDF-'):
                    return True

                # Also check Content-Type from response
                ctype = response.headers.get("content-type", "").split(";")[0].strip().lower()
                return "pdf" in ctype

    except Exception as e:
        print(f"[is_pdf_by_full_get] Error for {url}: {type(e).__name__}: {e}")
        return False


async def is_pdf(url: str, logger: Logger = None) -> bool:
    """
    Robustly detect if a URL points to a PDF file using multiple strategies.
    
    Args:
        url: The URL to check
        logger: Optional logger instance
        
    Returns:
        bool: True if URL points to a PDF, False otherwise
    """
    url = normalize_url_for_browser(url)

    if logger:
        logger.debug(f"Checking if URL is PDF: {url}")

    # 1. Fast URL pattern check
    if is_pdf_by_suffix(url):
        if logger:
            logger.info(f"URL pattern indicates PDF: {url}")
        else:
            print(f"{url} IS a PDF (by URL pattern)")
        return True

    # 2. Try HEAD request first (fastest network check)
    if is_pdf_by_requests_head(url):
        if logger:
            logger.info(f"HEAD request confirms PDF: {url}")
        else:
            print(f"{url} IS a PDF (by HEAD request)")
        return True

    # 3. Try partial GET with magic number check
    if await is_pdf_by_httpx_get_range(url):
        if logger:
            logger.info(f"Partial GET confirms PDF: {url}")
        else:
            print(f"{url} IS a PDF (by partial GET)")
        return True

    # 4. Last resort: stream beginning of file
    if await is_pdf_by_full_get(url):
        if logger:
            logger.info(f"Full GET confirms PDF: {url}")
        else:
            print(f"{url} IS a PDF (by full GET)")
        return True

    # # Not a PDF
    # if logger:
    #     logger.debug(f"URL is not a PDF: {url}")
    # else:
    #     print(f"{url} IS NOT a PDF")
    return False


class PDFParser:
    """
    Download and parse PDF. Returns (None, None) on failure.
    """

    # Default limits
    MAX_PAGES: int = 100
    MAX_IMAGE_PAGES: int = 50
    RENDER_DPI: int = 144
    JPEG_QUALITY: int = 70

    # ------------------ Public API ------------------
    async def extract(
            self,
            source: Union[str, bytes, BytesIO],
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Parameters
        ----------
        source : str | bytes | BytesIO
            URL / local file path / PDF byte stream

        Returns
        -------
        imgs : list[str] | None
        text : str | None
        """
        try:
            # 1) Obtain PDF bytes
            if isinstance(source, (bytes, BytesIO)):
                data = source.getvalue() if isinstance(source, BytesIO) else source
            elif isinstance(source, str) and source.lower().startswith(("http://", "https://")):
                data = await self._fetch_pdf_bytes(source)
            else:  # Local file
                data = await asyncio.to_thread(lambda p: open(p, "rb").read(), str(source))

            # 2) Magic number check
            if not data.lstrip().startswith(PDF_MAGIC):
                return [make_blank_png_b64()], "PDF extraction failed: Invalid PDF format"

            # 3) Parsing (CPU-intensive, synchronous), run in thread
            return await asyncio.to_thread(self._extract_from_bytes, data)

        except Exception as e:
            print(f"PDF extraction failed: {e}")
            return [make_blank_png_b64()], "PDF extraction failed: Download or parsing error"

    # ------------------ Internal Implementation ------------------
    async def _fetch_pdf_bytes(self, url: str) -> bytes:
        """
        Fetch PDF with browser User-Agent; if necessary, switch to export.arxiv.org as backup.
        """
        headers = {
            "User-Agent": UA_CHROME,
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
        }

        async def _download(u: str) -> bytes:
            async with aiohttp.ClientSession(headers=headers) as s:
                async with s.get(u, allow_redirects=True, timeout=30) as r:
                    r.raise_for_status()
                    return await r.read()

        data = await _download(url)
        # print(data)

        # If returned HTML, try backup domain for arxiv
        if not data.lstrip().startswith(PDF_MAGIC) and "arxiv.org" in url:
            backup = url.replace("://arxiv.org", "://export.arxiv.org")
            try:
                data = await _download(backup)
            except Exception as e:
                print(f"failed to download from {url} with backup export arxiv: {e}")

        # print(data)

        return data

    def _extract_from_bytes(
            self, data: bytes
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Actual parsing logic. Returns (None, None) on failure.
        """
        # Double-check magic number (in case called directly by other modules)
        if not data.lstrip().startswith(PDF_MAGIC):
            return [make_blank_png_b64()], "PDF extraction failed: Invalid PDF format"

        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except (fitz.FileDataError, RuntimeError):
            return [make_blank_png_b64()], "PDF extraction failed: Unable to parse PDF file"

        imgs: List[str] = []
        texts: List[str] = []
        zoom = self.RENDER_DPI / 72

        max_pages = min(self.MAX_PAGES, doc.page_count)
        max_img_pages = min(self.MAX_IMAGE_PAGES, doc.page_count)

        for i in range(max_pages):
            page = doc.load_page(i)
            texts.append(page.get_text("text"))

            if i < max_img_pages:
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                buf = BytesIO()
                img.save(buf, "JPEG", quality=self.JPEG_QUALITY,
                         optimize=True, progressive=True)
                imgs.append(base64.b64encode(buf.getvalue()).decode())
        # print(texts)
        return imgs, "\n".join(texts)


# ------------------ PDF Testing Functions ------------------

async def test_pdf_detection():
    """Test PDF detection functionality."""
    # Test URLs
    test_urls = [
        "https://www.fhwa.dot.gov/policyinformation/statistics/2023/pdf/mv1.pdf",  # Should be PDF
        "https://arxiv.org/pdf/2301.00001.pdf",  # Should be PDF (arxiv)
        "https://www.google.com",  # Should NOT be PDF
        "https://example.com/document.pdf",  # Should be PDF by suffix
    ]

    print("🧪 Testing PDF detection functionality...")
    print("=" * 50)

    for url in test_urls:
        print(f"\n🔍 Testing: {url}")
        try:
            result = await is_pdf(url)
            status = "✅ IS PDF" if result else "❌ NOT PDF"
            print(f"   Result: {status}")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "=" * 50)
    print("✅ PDF detection test completed!")


# ------------------ Local Quick Test ------------------
if __name__ == "__main__":
    async def _demo() -> None:
        # Test PDF detection
        # await test_pdf_detection()
        #
        # print("\n" + "=" * 50)
        # print("🧪 Testing PDF parsing functionality...")

        parser = PDFParser()

        # # ✅ Normal PDF
        # ok_imgs, ok_txt = await parser.extract(
        #     "https://arxiv.org/pdf/2505.07880.pdf"
        # )
        # print("Normal PDF:", "Success" if ok_txt else "Failed")

        # ❌ Fake PDF
        bad_imgs, bad_txt = await parser.extract(
            "https://arxiv.org/pdf/2408.XXXXXv1.pdf"
        )
        # print(bad_txt)
        # print("Fake PDF:", "Success" if bad_txt else "Failed")


    asyncio.run(_demo())
