# api_tools — External API Integrations

Tools for interacting with external services during evaluation. These are used by eval scripts that need data beyond what's in the cached webpages.

## Modules

### tool_arxiv.py — ArXiv Paper Lookup
`ArxivTool` wraps the `arxiv` Python library:
- `is_arxiv_pdf_link(link)`: Check if URL is an arXiv PDF
- `get_arxiv_id_from_pdf_link(link)`: Extract arXiv ID from URL
- `search_arxiv_by_id(id)`: Async search by arXiv ID
- `search_arxiv_by_title(title)`: Async search by title
- Uses `asyncio.to_thread()` for the synchronous arxiv library

### tool_googlemap.py — Google Maps Geocoding & Routing
`GoogleMapsTool` wraps the `googlemaps` Python client:
- `get_city_name(address, level)`: Geocode address to city/sublocality name
- `get_address_information(address)`: Full geocoding result
- `calculate_distance(addr1, addr2, mode)`: Driving/walking/transit distance in meters
- `calculate_travel_time(addr1, addr2, mode)`: Travel time in seconds
- Requires `GOOGLE_MAPS_API_KEY` env var

### tool_pdf.py — PDF Detection & Parsing
Two main components:

**`is_pdf(url)`**: Multi-strategy PDF detection:
1. URL suffix pattern matching (fast, no network)
2. HEAD request content-type check
3. Partial GET with magic number check (`%PDF-`)
4. Full GET stream with magic number check

**`PDFParser`**: Download and parse PDFs:
- `extract(source)`: Accept URL, file path, or bytes → returns `(images_b64_list, text)`
- Renders pages as JPEG images (up to 50 pages)
- Extracts plain text (up to 100 pages)
- Uses PyMuPDF (fitz) for parsing, aiohttp for downloading
- Special handling for arXiv URLs (fallback to export.arxiv.org)

## Usage Context
These tools are primarily used in:
- `eval_toolkit.py`'s `BaseEvaluator.get_page_info()` for PDF detection + parsing
- `batch_answer_cache.py` for pre-caching PDF content
- Individual eval scripts that need Google Maps or arXiv data
