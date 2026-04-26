import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "music_festivals_2026"
TASK_DESCRIPTION = """
Identify four major U.S. music festivals scheduled for 2026 that meet the following criteria:

1. Temporal Distribution: The four festivals must be distributed across different time periods:
   - One festival occurring in April 2026
   - One festival occurring in May or June 2026
   - One festival occurring in July 2026
   - One festival occurring in August 2026

2. Duration: Each festival must span 3 to 4 consecutive days.

3. Geographic Requirement: Each festival must be held at a different location within the United States.

4. Ticket Pricing:
   - General Admission (GA) passes must be priced between $450 and $700 for the full festival duration
   - VIP passes must be priced between $1,100 and $1,400 for the full festival duration

5. Venue Features:
   - Each festival must have at least 3 distinct performance stages or areas
   - VIP ticket holders must have access to designated VIP areas that include enhanced amenities such as specialty food and drink vendors, upgraded restroom facilities, and dedicated viewing or seating areas

For each of the four festivals, provide:
- Festival name
- Specific location (city and state)
- Exact dates (start and end dates)
- GA pass price
- VIP pass price
- Description of the multiple stages
- Description of VIP amenities
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FestivalItem(BaseModel):
    name: Optional[str] = None
    location_city: Optional[str] = None
    location_state: Optional[str] = None
    location_country: Optional[str] = None  # Should be United States / USA
    start_date: Optional[str] = None        # Free-form date string from the answer
    end_date: Optional[str] = None          # Free-form date string from the answer
    ga_price: Optional[str] = None          # Free-form price string (e.g., "$499", "about $500", "$450-$700")
    vip_price: Optional[str] = None         # Free-form price string (e.g., "$1,299", "$1,100-$1,400")
    stages_description: Optional[str] = None
    vip_amenities_description: Optional[str] = None

    # Source URLs cited in the answer for each aspect
    location_urls: List[str] = Field(default_factory=list)
    date_urls: List[str] = Field(default_factory=list)
    ga_price_urls: List[str] = Field(default_factory=list)
    vip_price_urls: List[str] = Field(default_factory=list)
    stages_urls: List[str] = Field(default_factory=list)
    vip_amenities_urls: List[str] = Field(default_factory=list)


class FestivalsExtraction(BaseModel):
    festivals: List[FestivalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_festivals() -> str:
    return """
    Extract up to the first four U.S. music festivals mentioned in the answer text that the answer claims satisfy the task. For each festival, extract the following fields exactly as presented:

    Required fields:
    - name: Festival name
    - location_city: City of the festival
    - location_state: State of the festival (2-letter or full name; extract as-is)
    - location_country: Country (should be United States / USA if present)
    - start_date: Start date string (e.g., "April 11, 2026" or "Apr 11, 2026")
    - end_date: End date string (e.g., "April 14, 2026")
    - ga_price: General Admission pass price string (for full festival duration as given by the answer)
    - vip_price: VIP pass price string (for full festival duration as given by the answer)
    - stages_description: Description text indicating multiple stages (extract as-is; can be a sentence or list)
    - vip_amenities_description: Description text indicating VIP amenities (extract as-is)

    Source URLs (explicitly mentioned in the answer; extract all URLs for each category as arrays; only include valid URLs):
    - location_urls: URLs that confirm the festival location
    - date_urls: URLs that confirm the festival dates
    - ga_price_urls: URLs that confirm GA pricing
    - vip_price_urls: URLs that confirm VIP pricing
    - stages_urls: URLs that confirm multiple stages (at least 3)
    - vip_amenities_urls: URLs that confirm VIP amenities (specialty F&B, upgraded restrooms, dedicated viewing/seating)

    Special rules:
    - Only extract data explicitly present in the answer; do not invent or infer missing details.
    - If a URL is missing from the answer for a category, return an empty array for that category.
    - If any required text field is missing, set it to null.
    - Preserve all textual formatting in extracted fields as-is (e.g., "$1,299+", "Aug 2–5, 2026").
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
MONTH_PATTERN = re.compile(r"(april|may|june|july|august)", re.IGNORECASE)


def detect_month_group(date_text: Optional[str]) -> Optional[str]:
    """
    Map a date string to one of the month groups:
    - 'apr' for April
    - 'mayjun' for May or June
    - 'jul' for July
    - 'aug' for August
    Returns None if month cannot be detected.
    """
    if not date_text:
        return None
    m = MONTH_PATTERN.search(date_text)
    if not m:
        return None
    month = m.group(1).lower()
    if month == "april":
        return "apr"
    if month in ("may", "june"):
        return "mayjun"
    if month == "july":
        return "jul"
    if month == "august":
        return "aug"
    return None


def price_numbers(price_text: Optional[str]) -> List[float]:
    """
    Extract plausible numeric amounts from a price text (e.g., "$1,299", "450-700", "about 499").
    Returns a list of floats (commas removed). Does not distinguish currency.
    """
    if not price_text:
        return []
    # Find numbers with optional thousands separators and decimals
    raw_nums = re.findall(r"\d{2,4}(?:,\d{3})*(?:\.\d{1,2})?", price_text)
    vals = []
    for s in raw_nums:
        try:
            vals.append(float(s.replace(",", "")))
        except Exception:
            pass
    return vals


def prices_in_range(price_text: Optional[str], low: float, high: float) -> bool:
    """
    Return True if any numeric price extracted from the text lies in [low, high].
    Intended as a soft check based on the answer text itself.
    """
    nums = price_numbers(price_text)
    if not nums:
        return False
    return any(low <= v <= high for v in nums)


def normalize_str(s: Optional[str]) -> Optional[str]:
    return s.strip().lower() if isinstance(s, str) else None


def distinct_us_locations(fests: List[FestivalItem]) -> bool:
    """
    Check that all festivals have distinct (city, state) pairs and are within the US.
    """
    pairs = []
    for f in fests:
        city = normalize_str(f.location_city)
        state = normalize_str(f.location_state)
        country = normalize_str(f.location_country)
        if not city or not state:
            return False
        # If country provided, require it to be United States/US/USA
        if country and ("united states" not in country and "usa" not in country and country != "us"):
            return False
        pairs.append((city, state))
    return len(pairs) == len(set(pairs))


def temporal_distribution_ok(fests: List[FestivalItem]) -> bool:
    """
    Check that the set of detected start months across festivals covers exactly:
    {apr, mayjun, jul, aug}.
    """
    groups = []
    for f in fests:
        g = detect_month_group(f.start_date)
        if not g:
            return False
        groups.append(g)
    return set(groups) == {"apr", "mayjun", "jul", "aug"} and len(groups) == 4


# --------------------------------------------------------------------------- #
# Verification per festival                                                   #
# --------------------------------------------------------------------------- #
async def verify_single_festival(
    evaluator: Evaluator,
    parent_node,
    fest: FestivalItem,
    index: int,
    expected_group: str,  # one of 'apr', 'mayjun', 'jul', 'aug'
) -> None:
    """
    Build verification subtree for a single festival.
    Each major requirement is a critical parallel sub-node under the festival.
    """
    festival_node = evaluator.add_parallel(
        id=f"festival_{index + 1}",
        desc=f"Festival #{index + 1} verification ({'April' if expected_group=='apr' else 'May/June' if expected_group=='mayjun' else 'July' if expected_group=='jul' else 'August'} 2026)",
        parent=parent_node,
        critical=False,  # Allow partial scoring across different festivals
    )

    # --------------------- Location ---------------------
    location_node = evaluator.add_parallel(
        id=f"F{index + 1}_Location",
        desc="Festival is held in the United States at a specified location",
        parent=festival_node,
        critical=True,
    )

    # URL presence check (critical)
    evaluator.add_custom_node(
        result=bool(fest.location_urls),
        id=f"F{index + 1}_Location_URL_present",
        desc="At least one URL is provided to confirm the festival location",
        parent=location_node,
        critical=True,
    )

    # Verify location by URLs (critical)
    loc_leaf = evaluator.add_leaf(
        id=f"F{index + 1}_Location_URL",
        desc="URL reference confirming the festival location",
        parent=location_node,
        critical=True,
    )
    claim_location = f"The festival '{fest.name or 'N/A'}' takes place in {fest.location_city or 'N/A'}, {fest.location_state or 'N/A'}, United States."
    await evaluator.verify(
        claim=claim_location,
        node=loc_leaf,
        sources=fest.location_urls,
        additional_instruction="Confirm the stated city and state in the U.S. using the provided URLs. If URLs are irrelevant or missing, judge as not supported."
    )

    # --------------------- Dates ---------------------
    dates_node = evaluator.add_parallel(
        id=f"F{index + 1}_Dates",
        desc="Festival dates meet the month window and span 3-4 consecutive days",
        parent=festival_node,
        critical=True,
    )

    # URL presence check (critical)
    evaluator.add_custom_node(
        result=bool(fest.date_urls),
        id=f"F{index + 1}_Dates_URL_present",
        desc="At least one URL is provided to confirm the festival dates",
        parent=dates_node,
        critical=True,
    )

    # Verify exact date strings are supported by URLs (critical)
    dates_exact_leaf = evaluator.add_leaf(
        id=f"F{index + 1}_Dates_URL",
        desc="URL reference confirming the festival dates",
        parent=dates_node,
        critical=True,
    )
    claim_dates_exact = f"The festival '{fest.name or 'N/A'}' runs from {fest.start_date or 'N/A'} to {fest.end_date or 'N/A'} in 2026."
    await evaluator.verify(
        claim=claim_dates_exact,
        node=dates_exact_leaf,
        sources=fest.date_urls,
        additional_instruction="Confirm the start and end dates for the 2026 edition using the provided URLs."
    )

    # Verify month window (critical)
    dates_window_leaf = evaluator.add_leaf(
        id=f"F{index + 1}_Dates_Window",
        desc="Festival dates fall within the required month window for this slot",
        parent=dates_node,
        critical=True,
    )
    if expected_group == "apr":
        claim_window = "This festival occurs in April 2026."
    elif expected_group == "mayjun":
        claim_window = "This festival occurs in May or June 2026."
    elif expected_group == "jul":
        claim_window = "This festival occurs in July 2026."
    else:
        claim_window = "This festival occurs in August 2026."
    await evaluator.verify(
        claim=claim_window,
        node=dates_window_leaf,
        sources=fest.date_urls,
        additional_instruction="Use the provided URLs to confirm the month of the festival dates. If dates span months, accept if any day is within the required window, as long as total duration is still 3–4 consecutive days."
    )

    # Verify 3–4 consecutive days (critical)
    duration_leaf = evaluator.add_leaf(
        id=f"F{index + 1}_Dates_Duration",
        desc="Festival spans 3 to 4 consecutive days",
        parent=dates_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This festival spans 3 to 4 consecutive days.",
        node=duration_leaf,
        sources=fest.date_urls,
        additional_instruction="Infer duration from the listed start and end dates on the provided URLs. Confirm that the event covers 3 or 4 consecutive calendar days."
    )

    # --------------------- Ticket Pricing ---------------------
    pricing_node = evaluator.add_parallel(
        id=f"F{index + 1}_Ticket_Pricing",
        desc="Festival offers GA and VIP passes with specified pricing",
        parent=festival_node,
        critical=True,
    )

    # GA: URL presence (critical)
    evaluator.add_custom_node(
        result=bool(fest.ga_price_urls),
        id=f"F{index + 1}_GA_Price_URL_present",
        desc="At least one URL is provided to confirm GA ticket pricing",
        parent=pricing_node,
        critical=True,
    )

    # GA: Range supported by URLs (critical)
    ga_range_leaf = evaluator.add_leaf(
        id=f"F{index + 1}_GA_Price",
        desc="General Admission pass price is between $450 and $700",
        parent=pricing_node,
        critical=True,
    )
    await evaluator.verify(
        claim="General Admission (GA) full festival pass price is between $450 and $700.",
        node=ga_range_leaf,
        sources=fest.ga_price_urls,
        additional_instruction="Use the pricing page(s) to confirm that a GA full festival pass (not single-day) falls within $450–$700. If multiple tiers exist, accept if the typical full GA pass fits the range."
    )

    # VIP: URL presence (critical)
    evaluator.add_custom_node(
        result=bool(fest.vip_price_urls),
        id=f"F{index + 1}_VIP_Price_URL_present",
        desc="At least one URL is provided to confirm VIP ticket pricing",
        parent=pricing_node,
        critical=True,
    )

    # VIP: Range supported by URLs (critical)
    vip_range_leaf = evaluator.add_leaf(
        id=f"F{index + 1}_VIP_Price",
        desc="VIP pass price is between $1,100 and $1,400",
        parent=pricing_node,
        critical=True,
    )
    await evaluator.verify(
        claim="VIP full festival pass price is between $1,100 and $1,400.",
        node=vip_range_leaf,
        sources=fest.vip_price_urls,
        additional_instruction="Use pricing page(s) to confirm the VIP full festival pass (not single-day or add-ons) lies within $1,100–$1,400. Consider typical VIP tiers that cover the full festival duration."
    )

    # --------------------- Venue Features ---------------------
    venue_node = evaluator.add_parallel(
        id=f"F{index + 1}_Venue_Features",
        desc="Festival venue includes required multi-stage and VIP amenities",
        parent=festival_node,
        critical=True,
    )

    # Multiple stages: URL presence (critical)
    evaluator.add_custom_node(
        result=bool(fest.stages_urls),
        id=f"F{index + 1}_Stages_URL_present",
        desc="At least one URL is provided to confirm multiple stages",
        parent=venue_node,
        critical=True,
    )

    # Multiple stages: supported by URLs (critical)
    stages_leaf = evaluator.add_leaf(
        id=f"F{index + 1}_Multiple_Stages",
        desc="Festival has at least 3 distinct performance stages or areas",
        parent=venue_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This festival has at least three distinct performance stages or areas.",
        node=stages_leaf,
        sources=fest.stages_urls,
        additional_instruction="Confirm via lineup/production pages or site maps. Accept if evidence clearly shows three or more distinct stages/areas."
    )

    # VIP amenities: URL presence (critical)
    evaluator.add_custom_node(
        result=bool(fest.vip_amenities_urls),
        id=f"F{index + 1}_VIP_Amenities_URL_present",
        desc="At least one URL is provided to confirm VIP amenities",
        parent=venue_node,
        critical=True,
    )

    # VIP amenities: supported by URLs (critical)
    vip_amenities_leaf = evaluator.add_leaf(
        id=f"F{index + 1}_VIP_Amenities",
        desc="VIP areas include enhanced amenities: specialty food/drink, upgraded restrooms, dedicated viewing/seating",
        parent=venue_node,
        critical=True,
    )
    await evaluator.verify(
        claim="VIP areas include enhanced amenities such as specialty food and drink vendors, upgraded restroom facilities, and dedicated viewing or seating areas.",
        node=vip_amenities_leaf,
        sources=fest.vip_amenities_urls,
        additional_instruction="Confirm at least these three categories are included. Equivalent wording is acceptable if it clearly matches these amenities."
    )


# --------------------------------------------------------------------------- #
# Global constraints verification                                             #
# --------------------------------------------------------------------------- #
def build_global_constraints(
    evaluator: Evaluator,
    parent_node,
    festivals: List[FestivalItem],
) -> None:
    """
    Add global constraints nodes: unique US locations, and month distribution coverage.
    These are critical checks under a critical container.
    """
    global_node = evaluator.add_parallel(
        id="Global_Constraints",
        desc="Global constraints across festivals",
        parent=parent_node,
        critical=True,
    )

    # Distinct US locations (critical)
    distinct_locations_leaf = evaluator.add_custom_node(
        result=distinct_us_locations(festivals),
        id="Global_Distinct_US_Locations",
        desc="All four festivals are in distinct city/state pairs within the United States",
        parent=global_node,
        critical=True,
    )

    # Temporal distribution coverage (critical)
    temporal_dist_leaf = evaluator.add_custom_node(
        result=temporal_distribution_ok(festivals),
        id="Global_Temporal_Distribution",
        desc="The four festivals cover April, May/June, July, and August 2026 (one per window)",
        parent=global_node,
        critical=True,
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
    Evaluate an answer for the 2026 U.S. music festivals task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent verification trees per festival + global constraints
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

    # Extract structured festival data
    extracted = await evaluator.extract(
        prompt=prompt_extract_festivals(),
        template_class=FestivalsExtraction,
        extraction_name="festivals_extraction",
    )

    # Keep exactly 4 festivals (pad with empty placeholders if fewer)
    fests: List[FestivalItem] = list(extracted.festivals[:4])
    while len(fests) < 4:
        fests.append(FestivalItem())

    # Build per-festival verification
    # Assign expected time windows per slot: [April, May/June, July, August]
    expected_groups = ["apr", "mayjun", "jul", "aug"]
    for i in range(4):
        await verify_single_festival(
            evaluator=evaluator,
            parent_node=root,
            fest=fests[i],
            index=i,
            expected_group=expected_groups[i],
        )

    # Build global constraints checks
    build_global_constraints(evaluator, root, fests)

    # Add some custom info for debugging (optional)
    evaluator.add_custom_info(
        info={
            "festival_count_extracted": len(extracted.festivals),
            "processed_count": len(fests),
            "expected_month_windows": ["April", "May/June", "July", "August"],
        },
        info_type="meta",
        info_name="processing_info",
    )

    # Return summary
    return evaluator.get_summary()