import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "conf2026_planning"
TASK_DESCRIPTION = """
An early-career researcher specializing in computer vision and machine learning is planning their 2026 conference attendance and travel schedule. Identify three major international conferences scheduled for 2026 that meet the following requirements:

1. One conference in North America (USA or Canada)
2. One conference in Europe
3. One conference in Asia-Pacific region (Asia or Oceania)
4. All three conferences must focus on computer vision, machine learning, or artificial intelligence
5. All three conferences must have confirmed dates for 2026

For each conference, provide:
- Conference name
- Specific location (city and country/state)
- Conference dates in 2026
- Official website or reference URL
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ConferenceItem(BaseModel):
    """Single conference information as extracted from the answer."""
    name: Optional[str] = None
    city: Optional[str] = None
    country_or_state: Optional[str] = None
    dates_text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ConferencesExtraction(BaseModel):
    """Three-region conference selection extracted from the answer."""
    north_america: Optional[ConferenceItem] = None
    europe: Optional[ConferenceItem] = None
    asia_pacific: Optional[ConferenceItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
    Extract exactly three (3) conferences from the answer, mapped to these regions:
    - north_america: Must be located in the USA or Canada
    - europe: Must be located in Europe
    - asia_pacific: Must be located in Asia or Oceania

    For EACH region, extract the following fields from the answer exactly as written:
    1) name: The conference name
    2) city: The city of the 2026 conference venue
    3) country_or_state: The country (for non-US/Canada) OR the state/province with country if applicable (e.g., "California, USA" or "Ontario, Canada"). If only a country is provided, use that country.
    4) dates_text: The conference dates in 2026, exactly as given in the answer (e.g., "June 15–20, 2026")
    5) urls: An array of all URLs provided for that conference (official website or other reference links). Only include actual URLs present in the answer.

    Rules:
    - Do NOT invent or infer any fields not explicitly present in the answer text.
    - If any field for a region is missing, set it to null (for strings) or [] (for urls).
    - If the answer lists more than three conferences, choose the first appropriate one for each region based on the provided location information.
    - The URLs must be actual URLs (plain links or markdown links). If none are provided, use an empty list.

    Return a JSON object with the keys: north_america, europe, asia_pacific. Each key maps to one object with the specified fields (or null if not provided).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_text(value: Optional[str]) -> str:
    return value.strip() if value else ""


def _format_location(item: Optional[ConferenceItem]) -> str:
    if not item:
        return ""
    city = _safe_text(item.city)
    cos = _safe_text(item.country_or_state)
    if city and cos:
        return f"{city}, {cos}"
    return city or cos


def _urls_or_empty(item: Optional[ConferenceItem]) -> List[str]:
    return item.urls if (item and item.urls) else []


# --------------------------------------------------------------------------- #
# Region verification logic                                                   #
# --------------------------------------------------------------------------- #
async def verify_region_conference(
    evaluator: Evaluator,
    parent_node,
    item: Optional[ConferenceItem],
    region_parent_id: str,
    region_parent_desc: str,
    id_prefix: str,
    region_expectation_text: str,
    region_short: str
) -> None:
    """
    Build the verification subtree for a single regional conference.

    Args:
        evaluator: Evaluator instance
        parent_node: Root or parent node to attach this region node
        item: Extracted conference item for this region
        region_parent_id: Node ID for this region parent (e.g., "Conference_1_North_America")
        region_parent_desc: Description for this region parent
        id_prefix: Prefix used in leaf IDs (NA/EU/AP)
        region_expectation_text: Human-readable region description for verification claims
        region_short: Short region tag for instructions (e.g., "USA or Canada", "Europe", "Asia-Pacific")
    """
    # Create the region parent node
    region_node = evaluator.add_parallel(
        id=region_parent_id,
        desc=region_parent_desc,
        parent=parent_node,
        critical=False
    )

    # Data unpack
    conf_name = _safe_text(item.name if item else None)
    loc_text = _format_location(item)
    urls = _urls_or_empty(item)
    dates_text = _safe_text(item.dates_text if item else None)

    # 1) Existence: Conference name
    evaluator.add_custom_node(
        result=bool(conf_name),
        id=f"{id_prefix}_Conference_Name",
        desc="Conference name is provided.",
        parent=region_node,
        critical=True
    )

    # 2) Existence: Reference URL provided
    evaluator.add_custom_node(
        result=bool(urls),
        id=f"{id_prefix}_Reference_URL",
        desc="Official conference website or reference URL is provided.",
        parent=region_node,
        critical=True
    )

    # 3) Existence: Specific venue (city and country/state)
    evaluator.add_custom_node(
        result=bool(_safe_text(item.city if item else None)) and bool(_safe_text(item.country_or_state if item else None)),
        id=f"{id_prefix}_Specific_Venue",
        desc="Specific city and country/state of the conference venue is provided.",
        parent=region_node,
        critical=True
    )

    # Prepare verification leaf nodes that rely on URLs or logical checks
    # 4) Geographic location in required region (verify with URL to confirm location; reason about region)
    geo_node = evaluator.add_leaf(
        id=f"{id_prefix}_Geographic_Location",
        desc=f"Conference location is in {region_expectation_text}.",
        parent=region_node,
        critical=True
    )
    geo_claim = (
        f"The official website or provided reference indicates that the conference '{conf_name}' takes place in {loc_text}. "
        f"Given that {loc_text} is in {region_expectation_text}, the conference location satisfies the regional requirement."
    )
    geo_instruction = (
        f"First, confirm from the webpage(s) that the conference location is listed as '{loc_text}' (or an equivalent phrasing). "
        f"Then, using common world knowledge, confirm that this location is in {region_short}. "
        f"If the page does not provide the location or it's ambiguous, mark as not supported."
    )

    # 5) Dates in 2026 are confirmed and provided (verify with URL)
    dates_node = evaluator.add_leaf(
        id=f"{id_prefix}_Conference_Dates",
        desc="Conference dates in 2026 are confirmed and provided.",
        parent=region_node,
        critical=True
    )
    dates_claim = (
        f"The 2026 edition of the conference '{conf_name}' has confirmed dates: {dates_text}. "
        f"These dates are explicitly stated on the official website or the provided reference, and they are in the year 2026."
    )
    dates_instruction = (
        "Only pass if the webpage clearly specifies the dates for the 2026 conference. "
        "If the page says 'TBA', 'To be announced', or does not show 2026 dates, this should fail. "
        "Allow reasonable formatting variations (e.g., 'June 15–20, 2026' vs '15-20 June 2026')."
    )

    # 6) Field relevance (verify with URL)
    field_node = evaluator.add_leaf(
        id=f"{id_prefix}_Field_Relevance",
        desc="Conference focuses on computer vision, machine learning, or artificial intelligence.",
        parent=region_node,
        critical=True
    )
    field_claim = (
        f"The conference '{conf_name}' is focused on computer vision, machine learning, or artificial intelligence "
        f"(e.g., in its scope, title, or call for papers)."
    )
    field_instruction = (
        "Check the conference name, scope, about page, or call for papers. "
        "Accept synonyms such as 'computer vision', 'pattern recognition', 'machine learning', 'deep learning', 'artificial intelligence'. "
        "If the focus is unrelated (e.g., purely databases without ML/AI focus), mark as not supported."
    )

    # 7) Major international conference (verify with URL)
    major_node = evaluator.add_leaf(
        id=f"{id_prefix}_Major_International",
        desc="Conference is a major international conference (e.g., widely recognized flagship/top-tier venue or explicitly international in scope).",
        parent=region_node,
        critical=True
    )
    major_claim = (
        f"The conference '{conf_name}' is a major international conference in the field (widely recognized/top-tier/flagship or explicitly international in scope)."
    )
    major_instruction = (
        "Rely primarily on the webpage(s). Indicators include: the name includes 'International', it's a well-known flagship venue, "
        "or the site describes global participation/recognition. If the site provides no evidence of international scope or major status, do not support."
    )

    # Perform verifications (batch with URL evidence where possible)
    claims_and_sources = [
        (geo_claim, urls, geo_node, geo_instruction),
        (dates_claim, urls, dates_node, dates_instruction),
        (field_claim, urls, field_node, field_instruction),
        (major_claim, urls, major_node, major_instruction),
    ]
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the 2026 conference identification and verification task.
    """
    # Initialize evaluator with PARALLEL aggregation at root
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

    # Extract structured conference info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferencesExtraction,
        extraction_name="conference_selection"
    )

    # Build three regional subtrees
    await verify_region_conference(
        evaluator=evaluator,
        parent_node=root,
        item=extracted.north_america,
        region_parent_id="Conference_1_North_America",
        region_parent_desc="A major international conference located in North America (USA or Canada) that meets all specified requirements.",
        id_prefix="NA",
        region_expectation_text="North America (United States or Canada)",
        region_short="the USA or Canada"
    )

    await verify_region_conference(
        evaluator=evaluator,
        parent_node=root,
        item=extracted.europe,
        region_parent_id="Conference_2_Europe",
        region_parent_desc="A major international conference located in Europe that meets all specified requirements.",
        id_prefix="EU",
        region_expectation_text="Europe",
        region_short="Europe"
    )

    await verify_region_conference(
        evaluator=evaluator,
        parent_node=root,
        item=extracted.asia_pacific,
        region_parent_id="Conference_3_Asia_Pacific",
        region_parent_desc="A major international conference located in Asia-Pacific region (Asia or Oceania) that meets all specified requirements.",
        id_prefix="AP",
        region_expectation_text="the Asia-Pacific region (Asia or Oceania)",
        region_short="the Asia-Pacific region (Asia or Oceania)"
    )

    # Return structured summary
    return evaluator.get_summary()