import asyncio
import logging
import re
from datetime import datetime, date
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "celebrity_brand_partnerships_2025_2026"
TASK_DESCRIPTION = (
    "Identify 4 celebrities who announced new fashion or beauty brand partnerships between January 2025 and "
    "February 2026. For each celebrity, provide: (1) The celebrity's name, (2) The fashion or beauty brand name, "
    "(3) The specific role or type of partnership (e.g., brand ambassador, campaign star, capsule collection designer), "
    "and (4) The month and year when the partnership was publicly announced. The partnerships must be in the fashion or "
    "beauty industry and must have been officially announced during the specified time period. For each celebrity, "
    "include a reference URL from a major fashion/beauty publication or official brand source that verifies the "
    "partnership details."
)

MAX_ITEMS = 4
TIME_WINDOW_START = date(2025, 1, 1)
TIME_WINDOW_END = date(2026, 2, 28)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PartnershipItem(BaseModel):
    celebrity_name: Optional[str] = None
    brand_name: Optional[str] = None
    partnership_role: Optional[str] = None
    announcement_month_year: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class PartnershipsExtraction(BaseModel):
    items: List[PartnershipItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_partnerships() -> str:
    return """
    Extract up to the first 4 celebrity-brand partnership entries from the answer. Each entry should include:
    - celebrity_name: The celebrity's full name exactly as written in the answer.
    - brand_name: The fashion or beauty brand involved (e.g., Dior, Fenty Beauty, Nike, Tiffany & Co., Sephora).
    - partnership_role: The specific role or type of partnership (e.g., brand ambassador, campaign star, global face, capsule collection designer, collaborator).
    - announcement_month_year: The month and year when the partnership was publicly announced, as stated or implied by the answer (e.g., "January 2025", "Feb 2026"). If a full date is given, convert it to "Month YYYY".
    - reference_urls: An array of URLs the answer cites for this partnership (major fashion/beauty publications or official brand sources). Extract actual URLs only (plain or markdown).
    
    Rules:
    - Only extract information explicitly present in the answer.
    - If any field is missing, set it to null (for strings) or an empty array for reference_urls.
    - Do not infer URLs or add extra sources not in the answer.
    - Keep at most 4 entries in the 'items' array (use the first 4 mentioned).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _try_parse_date_formats(text: str) -> Optional[date]:
    text = text.strip()
    fmts = [
        "%B %Y",      # January 2025
        "%b %Y",      # Jan 2025
        "%B %d, %Y",  # January 10, 2025
        "%b %d, %Y",  # Jan 10, 2025
        "%Y-%m-%d",
        "%Y-%m",
        "%Y/%m/%d",
        "%Y/%m",
        "%m/%Y",
        "%m-%Y",
        "%d %B %Y",
        "%d %b %Y",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(text, fmt).date()
            # Normalize to mid-month for month/year formats lacking a day
            if "%d" not in fmt and ("%" in fmt):
                return date(dt.year, dt.month, 15)
            return dt
        except Exception:
            continue
    return None


def parse_month_year_to_date(text: Optional[str]) -> Optional[date]:
    if not text:
        return None
    # First, try direct formats
    parsed = _try_parse_date_formats(text)
    if parsed:
        # If only month/year info likely, standardize to mid-month
        if re.fullmatch(r"([A-Za-z]{3,9})\s+\d{4}", text.strip()) or re.fullmatch(r"\d{2}/\d{4}", text.strip()):
            return date(parsed.year, parsed.month, 15)
        return parsed

    # Try extracting month name and year from free text
    lower = text.lower()
    month_map = {
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
        "december": 12, "dec": 12,
    }
    month = None
    for k, v in month_map.items():
        if re.search(rf"\b{k}\b", lower):
            month = v
            break
    year_match = re.search(r"\b(20\d{2})\b", lower)
    if month and year_match:
        try:
            return date(int(year_match.group(1)), month, 15)
        except Exception:
            return None
    return None


def in_time_window(text: Optional[str]) -> bool:
    dt = parse_month_year_to_date(text)
    if not dt:
        return False
    return TIME_WINDOW_START <= dt <= TIME_WINDOW_END


def ordinal(n: int) -> str:
    mapping = {1: "First", 2: "Second", 3: "Third", 4: "Fourth"}
    return mapping.get(n, f"Item {n}")


# --------------------------------------------------------------------------- #
# Verification per item                                                       #
# --------------------------------------------------------------------------- #
async def verify_partnership_item(
    evaluator: Evaluator,
    parent_node,
    item: PartnershipItem,
    index: int,
) -> None:
    """
    Build the verification subtree for one celebrity partnership item.
    """
    ord_word = ordinal(index + 1)

    # Parent node for this celebrity (non-critical parallel to allow partial across items)
    celeb_node = evaluator.add_parallel(
        id=f"Celebrity_{index + 1}",
        desc=f"{ord_word} celebrity with complete partnership information",
        parent=parent_node,
        critical=False,
    )

    name_val = (item.celebrity_name or "").strip()
    brand_val = (item.brand_name or "").strip()
    role_val = (item.partnership_role or "").strip()
    date_val = (item.announcement_month_year or "").strip()
    sources_list = item.reference_urls or []

    # 1) Name provided (existence check)
    evaluator.add_custom_node(
        result=bool(name_val),
        id=f"Celebrity_{index + 1}_Name",
        desc=f"{ord_word} celebrity - Name is provided",
        parent=celeb_node,
        critical=True,
    )

    # 2) URL from credible source is provided and verifies the partnership details (major publication or official brand)
    url_node = evaluator.add_leaf(
        id=f"Celebrity_{index + 1}_URL",
        desc=f"{ord_word} celebrity - Reference URL from credible fashion/beauty publication or official source is provided",
        parent=celeb_node,
        critical=True,
    )
    url_claim = (
        f"At least one of the provided webpages is from either a major fashion/beauty publication or an official brand "
        f"site/press release and it reports that {name_val} has a partnership with {brand_val}."
    )
    await evaluator.verify(
        claim=url_claim,
        node=url_node,
        sources=sources_list,
        additional_instruction=(
            "Major fashion/beauty publications include outlets like Vogue, Harper's Bazaar, Elle, WWD, Business of Fashion, "
            "Allure, InStyle, Glamour, Cosmopolitan, Byrdie, Refinery29, GQ, The Cut, Nylon, etc. "
            "An official brand source includes the brand's official website newsroom/press release/blog or a corporate press page. "
            "Passing requires the page to be from one of these credible types AND to mention/announce the partnership."
        ),
    )

    # 3) Brand correctly identified (supported by sources)
    brand_node = evaluator.add_leaf(
        id=f"Celebrity_{index + 1}_Brand",
        desc=f"{ord_word} celebrity - Fashion or beauty brand name is correctly identified",
        parent=celeb_node,
        critical=True,
    )
    brand_claim = (
        f"The source reports that {name_val} announced a partnership with the brand '{brand_val}'."
    )
    await evaluator.verify(
        claim=brand_claim,
        node=brand_node,
        sources=sources_list,
        additional_instruction=(
            "Verify that the page explicitly connects the celebrity to the named brand as a partnership or official role. "
            "Allow reasonable variants like sub-brands or brand divisions (e.g., 'Dior Beauty' vs 'Dior')."
        ),
    )

    # 4) Role/type correctly identified (supported by sources)
    role_node = evaluator.add_leaf(
        id=f"Celebrity_{index + 1}_Role",
        desc=f"{ord_word} celebrity - Specific partnership role/type is correctly identified",
        parent=celeb_node,
        critical=True,
    )
    role_claim = (
        f"The source indicates that in this partnership, {name_val} serves as '{role_val}' for {brand_val}."
    )
    await evaluator.verify(
        claim=role_claim,
        node=role_node,
        sources=sources_list,
        additional_instruction=(
            "Check that the page describes the same role/type (e.g., brand ambassador, global face, campaign star, "
            "capsule collection designer/collaborator). Allow minor wording variations that mean the same thing."
        ),
    )

    # 5) Announcement month and year correctly identified (supported by sources)
    date_node = evaluator.add_leaf(
        id=f"Celebrity_{index + 1}_Date",
        desc=f"{ord_word} celebrity - Announcement month and year are correctly identified",
        parent=celeb_node,
        critical=True,
    )
    date_claim = (
        f"The partnership was publicly announced in {date_val}."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=sources_list,
        additional_instruction=(
            "Accept if the page explicitly states the announcement timing or if the article is the initial announcement "
            "and its publication date matches the month and year. Allow abbrev. months (e.g., Feb vs February) and minor "
            "formatting variants. If multiple dates appear, focus on the announcement/publication date."
        ),
    )

    # 6) Time window constraint (Jan 1, 2025 – Feb 28, 2026 inclusive)
    evaluator.add_custom_node(
        result=in_time_window(date_val),
        id=f"Celebrity_{index + 1}_TimeConstraint",
        desc=f"{ord_word} celebrity - Partnership was announced between January 1, 2025 and February 28, 2026",
        parent=celeb_node,
        critical=True,
    )

    # 7) Industry constraint (must be fashion or beauty)
    industry_node = evaluator.add_leaf(
        id=f"Celebrity_{index + 1}_IndustryConstraint",
        desc=f"{ord_word} celebrity - Partnership is in the fashion or beauty industry",
        parent=celeb_node,
        critical=True,
    )
    industry_claim = (
        f"The partnership between {name_val} and {brand_val} is in the fashion or beauty industry."
    )
    await evaluator.verify(
        claim=industry_claim,
        node=industry_node,
        sources=sources_list,
        additional_instruction=(
            "Confirm that the brand is clearly within fashion (apparel, luxury houses, accessories, footwear, jewelry) "
            "or beauty (cosmetics, skincare, fragrance, haircare). Pages describing tech, gaming, food, or unrelated "
            "industries should not pass unless the partnership explicitly involves the brand's fashion/beauty line."
        ),
    )


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the celebrity-brand partnerships task.
    """
    # Initialize evaluator
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
        default_model=model,
    )

    # Record time window info
    evaluator.add_custom_info(
        info={
            "time_window_start": TIME_WINDOW_START.isoformat(),
            "time_window_end": TIME_WINDOW_END.isoformat(),
            "max_items": MAX_ITEMS,
        },
        info_type="config",
        info_name="evaluation_config",
    )

    # Extract structured items from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_partnerships(),
        template_class=PartnershipsExtraction,
        extraction_name="extracted_partnerships",
    )

    # Normalize to exactly MAX_ITEMS items
    items: List[PartnershipItem] = list(extracted.items or [])
    if len(items) > MAX_ITEMS:
        items = items[:MAX_ITEMS]
    while len(items) < MAX_ITEMS:
        items.append(PartnershipItem())

    # Build verification subtrees for each celebrity item
    for idx in range(MAX_ITEMS):
        await verify_partnership_item(
            evaluator=evaluator,
            parent_node=root,
            item=items[idx],
            index=idx,
        )

    # Return final structured summary
    return evaluator.get_summary()