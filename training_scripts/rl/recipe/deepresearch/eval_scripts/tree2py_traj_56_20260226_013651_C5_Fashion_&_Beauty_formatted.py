import asyncio
import logging
import re
from typing import Optional, List, Dict, Any, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_fashion_beauty_2025"
TASK_DESCRIPTION = (
    "Identify four distinct celebrity developments in the fashion and beauty industry that were announced during 2025. "
    "These developments should include brand partnerships (such as brand ambassadorships), creative directorial appointments "
    "at fashion or beauty companies, or celebrity-founded product launches. For each development, provide: "
    "(1) The celebrity's full name, (2) The brand or company name, (3) The type of development (brand partnership/ambassadorship, "
    "directorial appointment, or product launch), (4) The specific role, position, or product name, "
    "(5) The month and date when the development was announced or launched (must be within 2025), "
    "and (6) A supporting URL reference from a credible source. "
    "The four developments must be from four different months in 2025, and they should represent a diverse range of activity types "
    "within the fashion and beauty industry."
)

TARGET_YEAR = 2025

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Development(BaseModel):
    celebrity_name: Optional[str] = None
    brand_company: Optional[str] = None
    development_type: Optional[str] = None  # raw text as stated in the answer
    specific_detail: Optional[str] = None   # role/position/product name
    announcement_date: Optional[str] = None # month and day string, ideally includes year 2025
    source_url: Optional[str] = None        # primary supporting URL
    extra_urls: List[str] = Field(default_factory=list)  # any additional URLs mentioned


class DevelopmentsExtraction(BaseModel):
    developments: List[Development] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_developments() -> str:
    return """
    Extract up to 6 celebrity developments in the fashion and beauty industry as presented in the answer text.
    Each development should be one of: a brand partnership/ambassadorship, a creative directorial appointment at a fashion/beauty company, or a celebrity-founded product launch.
    For each development, extract the following fields exactly as they appear in the answer:
    - celebrity_name: The celebrity's full name as stated.
    - brand_company: The brand or company involved.
    - development_type: The type of development as described in the answer (e.g., "brand ambassador", "creative director", "launched fragrance", etc.).
    - specific_detail: The specific role/position or product name (e.g., "Global Brand Ambassador", "Artistic Director", "skin-care line 'GlowX'").
    - announcement_date: The month and day (and possibly year) when the development was announced/launched. If the answer does not provide a month and day, return the best available date string from the answer; otherwise return null.
    - source_url: A single primary supporting URL explicitly mentioned in the answer. If multiple URLs are listed, choose the one that most directly supports the development (e.g., brand press release, reputable media article). If none provided, return null.
    - extra_urls: Any additional URLs cited for the same development. If none, return [].

    Return a JSON object with a top-level "developments" array of objects with the fields above.
    Do not invent information not present in the answer text. Use null/[] if any field is missing.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12
}

ALLOWED_TYPE_CATEGORIES = {
    "partnership/ambassadorship",
    "directorial_appointment",
    "product_launch",
}

def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False

def normalize_text(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s.strip()) if s else ""

def parse_month_year_from_date_str(date_str: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """
    Heuristically extract month number and year from a free-form date string.
    Returns (month_num, year) where each can be None if not found.
    """
    if not date_str:
        return None, None
    s = date_str.strip().lower()

    # Try Month name first
    for name, num in MONTH_MAP.items():
        if re.search(rf"\b{name}\b", s):
            # Find year nearby
            y = None
            m = num
            m = num
            year_match = re.search(r"\b(20\d{2})\b", s)
            if year_match:
                try:
                    y = int(year_match.group(1))
                except Exception:
                    y = None
            return m, y

    # Try ISO or numeric formats: YYYY-MM-DD, MM/DD/YYYY, MM-DD-YYYY
    # ISO
    m = None
    y = None
    m1 = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", s)
    if m1:
        try:
            y = int(m1.group(1))
            m = int(m1.group(2))
            return m, y
        except Exception:
            pass

    # US style with slashes
    m2 = re.search(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b", s)
    if m2:
        try:
            m = int(m2.group(1))
            y = int(m2.group(3))
            return m, y
        except Exception:
            pass

    # US style with dashes
    m3 = re.search(r"\b(\d{1,2})-(\d{1,2})-(20\d{2})\b", s)
    if m3:
        try:
            m = int(m3.group(1))
            y = int(m3.group(3))
            return m, y
        except Exception:
            pass

    # Day Month Year like "12 Jan 2025" or "12 January 2025"
    m4 = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(20\d{2})\b", s)
    if m4:
        month_name = m4.group(2).lower()
        y = int(m4.group(3))
        m = MONTH_MAP.get(month_name)
        return m, y

    # Month Day, Year like "January 12, 2025" or "Jan 12, 2025"
    m5 = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),\s*(20\d{2})\b", s)
    if m5:
        month_name = m5.group(1).lower()
        y = int(m5.group(3))
        m = MONTH_MAP.get(month_name)
        return m, y

    # If only year is present
    y_match = re.search(r"\b(20\d{2})\b", s)
    y = int(y_match.group(1)) if y_match else None
    return None, y

def has_month_and_day(date_str: Optional[str]) -> bool:
    """Check whether the string contains an explicit month+day (name or numeric)."""
    if not date_str:
        return False
    s = date_str.strip()
    # Month name + day
    if re.search(r"\b([A-Za-z]{3,9})\s+\d{1,2}\b", s):
        return True
    # Day Month
    if re.search(r"\b\d{1,2}\s+([A-Za-z]{3,9})\b", s):
        return True
    # Numeric MM/DD or MM-DD
    if re.search(r"\b\d{1,2}[/-]\d{1,2}\b", s):
        return True
    # ISO YYYY-MM-DD
    if re.search(r"\b20\d{2}-\d{1,2}-\d{1,2}\b", s):
        return True
    return False

def categorize_type(raw: Optional[str]) -> str:
    """
    Normalize the development type into one of:
    - 'partnership/ambassadorship'
    - 'directorial_appointment'
    - 'product_launch'
    - 'other'
    """
    if not raw:
        return "other"
    s = raw.lower()

    # Partnerships / Ambassadorships
    if any(k in s for k in [
        "ambassador", "ambassadorship", "spokesperson", "spokes-person", "face of", "brand partnership",
        "partners with", "campaign star", "campaign face"
    ]):
        return "partnership/ambassadorship"

    # Directorial appointments
    if any(k in s for k in [
        "creative director", "co-creative director", "artistic director", "beauty director", "fashion director",
        "director of", "design director", "editorial director"
    ]):
        return "directorial_appointment"

    # Product launch / brand launch
    if any(k in s for k in [
        "launch", "launched", "debuted", "debut", "introduced", "unveiled", "rolled out", "released", "founder",
        "co-founded", "founded", "new brand", "product line", "fragrance", "perfume", "skincare", "skin-care",
        "makeup line", "cosmetics line", "haircare", "collection", "capsule collection"
    ]):
        return "product_launch"

    return "other"


# --------------------------------------------------------------------------- #
# Verification for one development                                            #
# --------------------------------------------------------------------------- #
async def verify_development(
    evaluator: Evaluator,
    parent_node,
    dev: Development,
    idx: int,
) -> None:
    """
    Build and execute verification nodes for a single development (index 1..4).
    """
    dev_idx = idx + 1

    # Create parent parallel node for this development (non-critical to allow partial credit across developments)
    dev_node = evaluator.add_parallel(
        id=f"development_{dev_idx}",
        desc=f"Development #{dev_idx}: celebrity fashion/beauty development with all required details",
        parent=parent_node,
        critical=False
    )

    # Reference URL existence and basic validity (Critical gate)
    url_ok = is_valid_url(dev.source_url)
    evaluator.add_custom_node(
        result=url_ok,
        id=f"reference_url_{dev_idx}",
        desc=f"A valid supporting URL reference from a credible source is provided for development #{dev_idx}",
        parent=dev_node,
        critical=True
    )

    sources = dev.source_url if url_ok else None

    # Celebrity name supported
    celeb_leaf = evaluator.add_leaf(
        id=f"celebrity_name_{dev_idx}",
        desc="Celebrity's full name is accurately provided",
        parent=dev_node,
        critical=True
    )
    celeb_claim = (
        f"The webpage supports that the development prominently involves the celebrity named '{normalize_text(dev.celebrity_name)}' "
        f"in the context of a fashion or beauty industry announcement or launch."
    )
    await evaluator.verify(
        claim=celeb_claim,
        node=celeb_leaf,
        sources=sources,
        additional_instruction=(
            "Judge Incorrect if the provided name is missing or blank. "
            "Allow minor variations (e.g., middle initials, diacritics, stage names vs. legal names) if clearly the same person."
        )
    )

    # Brand or company name supported
    brand_leaf = evaluator.add_leaf(
        id=f"brand_company_{dev_idx}",
        desc="Brand or company name is accurately provided",
        parent=dev_node,
        critical=True
    )
    brand_claim = (
        f"The webpage supports that the brand/company involved is '{normalize_text(dev.brand_company)}' for this development."
    )
    await evaluator.verify(
        claim=brand_claim,
        node=brand_leaf,
        sources=sources,
        additional_instruction=(
            "Judge Incorrect if the provided brand/company is missing or blank. "
            "Accept minor lexical variants (e.g., Inc., Ltd., SA, NV) and abbreviated forms if unambiguous."
        )
    )

    # Development type supported (categorized)
    raw_type = normalize_text(dev.development_type)
    norm_type = categorize_type(raw_type)
    type_leaf = evaluator.add_leaf(
        id=f"development_type_{dev_idx}",
        desc="Type of development (brand partnership/ambassadorship, directorial appointment, or product launch) is accurately described",
        parent=dev_node,
        critical=True
    )
    type_claim = (
        f"The webpage indicates that the nature of this development aligns with the category '{norm_type}', "
        f"consistent with the answer's description '{raw_type}'."
    )
    await evaluator.verify(
        claim=type_claim,
        node=type_leaf,
        sources=sources,
        additional_instruction=(
            "Map the page's description into one of: "
            "partnership/ambassadorship (e.g., ambassador, spokesperson, face of a campaign), "
            "directorial_appointment (e.g., creative/artistic/fashion/beauty director), "
            "product_launch (e.g., launched/debuted a product or brand). "
            "Judge Incorrect if the provided type is missing/blank or does not match the page's substance."
        )
    )

    # Specific role/position/product name supported
    detail_leaf = evaluator.add_leaf(
        id=f"specific_detail_{dev_idx}",
        desc="Specific role, position, or product name with key characteristics is accurately mentioned",
        parent=dev_node,
        critical=True
    )
    detail_claim = (
        f"The webpage explicitly mentions the specific role/position/product detail stated as '{normalize_text(dev.specific_detail)}'."
    )
    await evaluator.verify(
        claim=detail_claim,
        node=detail_leaf,
        sources=sources,
        additional_instruction=(
            "Judge Incorrect if the provided detail is missing or blank. "
            "Allow minor paraphrase or formatting differences (e.g., capitalization, punctuation), "
            "but it must be clearly the same role/position/product."
        )
    )

    # Announcement date supported and within 2025
    date_leaf = evaluator.add_leaf(
        id=f"announcement_date_{dev_idx}",
        desc="Announcement or launch date with month and day is provided and falls within 2025",
        parent=dev_node,
        critical=True
    )
    date_claim = (
        f"The webpage shows that the announcement/launch occurred in the calendar year 2025 and corresponds to the provided date '{normalize_text(dev.announcement_date)}'."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_leaf,
        sources=sources,
        additional_instruction=(
            "Judge Incorrect if the provided date is missing/blank OR does not include both a month and day. "
            "Use the article publish date or press release date if it clearly corresponds to the announcement/launch. "
            "Slight timezone-related ±1 day differences are acceptable if clearly the same event. "
            "Ultimately, the event must be in 2025."
        )
    )


# --------------------------------------------------------------------------- #
# Diversity checks (month and type)                                           #
# --------------------------------------------------------------------------- #
def compute_month_diversity(extracted_four: List[Development]) -> Tuple[bool, List[Optional[int]]]:
    months: List[Optional[int]] = []
    for dev in extracted_four:
        m, y = parse_month_year_from_date_str(dev.announcement_date)
        if y is not None and y != TARGET_YEAR:
            # Provided year is not 2025; consider month invalid for diversity
            months.append(None)
            continue
        months.append(m)
    # Count unique non-None months
    unique_months = {m for m in months if m is not None}
    return (len(unique_months) >= 4), months

def compute_type_diversity(extracted_four: List[Development]) -> Tuple[bool, List[str]]:
    cats = []
    for dev in extracted_four:
        cats.append(categorize_type(dev.development_type))
    # Only count recognized categories from the required set
    recognized = {c for c in cats if c in ALLOWED_TYPE_CATEGORIES}
    return (len(recognized) >= 2), cats


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 2025 celebrity fashion/beauty developments task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract developments from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_developments(),
        template_class=DevelopmentsExtraction,
        extraction_name="developments_extraction"
    )

    # Ensure we have exactly 4 items for evaluation: take first 4 or pad with empty placeholders
    devs = list(extracted.developments) if extracted and extracted.developments else []
    if len(devs) > 4:
        devs = devs[:4]
    while len(devs) < 4:
        devs.append(Development())

    # Build verification subtrees for the four developments
    for i, dev in enumerate(devs[:4]):
        await verify_development(evaluator, root, dev, i)

    # Month diversity check (critical)
    month_ok, months_list = compute_month_diversity(devs[:4])
    evaluator.add_custom_node(
        result=month_ok,
        id="month_diversity",
        desc=f"The four developments are announced/launched in four different months within 2025. Parsed months: {months_list}",
        parent=root,
        critical=True
    )

    # Type diversity check (critical)
    type_ok, cats_list = compute_type_diversity(devs[:4])
    evaluator.add_custom_node(
        result=type_ok,
        id="type_diversity",
        desc=f"The four developments represent at least two different categories among partnership/ambassadorship, directorial_appointment, product_launch. Parsed categories: {cats_list}",
        parent=root,
        critical=True
    )

    # Record some helpful debug info
    months_info = []
    for idx, dev in enumerate(devs[:4], start=1):
        m, y = parse_month_year_from_date_str(dev.announcement_date)
        months_info.append({"idx": idx, "provided_date": dev.announcement_date, "parsed_month": m, "parsed_year": y})

    evaluator.add_custom_info(
        info={
            "parsed_months": months_info,
            "parsed_types": cats_list,
            "target_year": TARGET_YEAR
        },
        info_type="debug_parsed",
        info_name="parsed_overview"
    )

    # Produce final structured summary
    return evaluator.get_summary()