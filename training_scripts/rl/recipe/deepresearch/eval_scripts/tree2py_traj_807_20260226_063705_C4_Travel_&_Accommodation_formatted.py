import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wi_dells_largest_indoor_waterpark"
TASK_DESCRIPTION = (
    "Which resort in Wisconsin Dells, Wisconsin, has the largest single indoor waterpark facility under one roof "
    "(measuring at least 100,000 square feet), offers on-site hotel accommodations, operates year-round, and "
    "advertises itself as having 'Wisconsin's Largest Indoor Waterpark'? Provide the resort name and specify the "
    "exact size of its indoor waterpark in square feet."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ResortExtraction(BaseModel):
    """
    Structured extraction of the resort information from the answer text.
    """
    resort_name: Optional[str] = None
    indoor_waterpark_size_sqft: Optional[str] = None
    location_text: Optional[str] = None
    year_round_text: Optional[str] = None
    largest_claim_text: Optional[str] = None
    official_urls: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_resort_info() -> str:
    return """
    Extract the resort identification details presented in the answer. Return a JSON object with the following fields:

    - resort_name: The exact name of the resort identified in the answer.
    - indoor_waterpark_size_sqft: The exact size of the resort’s largest single indoor waterpark facility under one roof, written in square feet (e.g., "125,000 square feet", "100000 sq ft"). Preserve formatting from the answer.
    - location_text: Any explicit location text mentioned in the answer (e.g., "Wisconsin Dells, Wisconsin" or a street address).
    - year_round_text: Any phrase or sentence in the answer that claims the indoor waterpark operates year-round (e.g., "open year-round", "indoor waterpark is open 365 days").
    - largest_claim_text: Any phrase or sentence in the answer that claims the resort advertises "Wisconsin's Largest Indoor Waterpark" or an equivalent phrase.
    - official_urls: All URLs in the answer that appear to be official resort website pages (e.g., domains like kalahariresorts.com, wildernessresort.com, greatwolf.com/dells). Include only URLs explicitly present in the answer.
    - source_urls: All other URLs in the answer that serve as supporting references (e.g., travel guides, reputable news, Wikipedia). Include only URLs explicitly present in the answer.

    SPECIAL RULES FOR URL EXTRACTION:
    - Extract only URLs explicitly present in the answer text. Do not infer or fabricate URLs.
    - Include full URLs. If a URL is missing protocol (http/https), prepend http://.
    - Return empty arrays for any URL lists if none are provided.

    If any field is not mentioned, return null for that field (or an empty array for URLs).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _parse_size_sqft(size_text: Optional[str]) -> Optional[int]:
    """
    Attempt to parse an integer square feet number from a free-form size text.
    Examples: "125,000 square feet", "100000 sq ft", "150,000+ sq. ft."
    Returns None if parsing fails.
    """
    if not size_text:
        return None
    # Find the largest integer-like token (strip commas)
    numbers = re.findall(r"\d[\d,]*", size_text)
    if not numbers:
        return None
    try:
        values = [int(n.replace(",", "")) for n in numbers if n]
        return max(values) if values else None
    except Exception:
        return None


def _mentions_sqft_unit(size_text: Optional[str]) -> bool:
    """
    Check if the size text mentions square feet units in common variants.
    """
    if not size_text:
        return False
    t = size_text.lower()
    variants = ["square feet", "sq ft", "sq. ft", "sqft", "sf", "ft²"]
    return any(v in t for v in variants)


def _collect_all_sources(info: ResortExtraction) -> List[str]:
    """
    Combine official and other source URLs, deduplicate, preserve order.
    """
    seen = set()
    merged: List[str] = []
    for url in (info.official_urls or []) + (info.source_urls or []):
        u = (url or "").strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            merged.append(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_resort_verification_tree(evaluator: Evaluator, parent_node, info: ResortExtraction) -> None:
    """
    Build the verification tree under the main 'Resort_Identification' node and run verifications.
    All child checks are critical to satisfy the rubric’s mandatory criteria.
    """
    # Create the main node for resort identification (critical, parallel aggregation)
    resort_node = evaluator.add_parallel(
        id="Resort_Identification",
        desc="Identify a Wisconsin Dells resort that meets all specified criteria for indoor waterpark size, facilities, and accommodations, and provide both the resort name and exact waterpark size",
        parent=parent_node,
        critical=True
    )

    # Precompute helpers
    resort_name = (info.resort_name or "").strip()
    size_text = (info.indoor_waterpark_size_sqft or "").strip()
    size_value = _parse_size_sqft(size_text)
    has_sqft_unit = _mentions_sqft_unit(size_text)
    all_sources = _collect_all_sources(info)
    official_sources = info.official_urls or []

    # 1) Resort_Name_Provided (existence in the answer)
    evaluator.add_custom_node(
        result=bool(resort_name),
        id="Resort_Name_Provided",
        desc="The answer provides the name of the resort",
        parent=resort_node,
        critical=True
    )

    # 2) Exact_Size_Provided (must include a parseable integer and a square-feet unit)
    evaluator.add_custom_node(
        result=bool(size_value) and has_sqft_unit,
        id="Exact_Size_Provided",
        desc="The answer specifies the exact size of the indoor waterpark in square feet",
        parent=resort_node,
        critical=True
    )

    # 3) Location_Requirement (verify via provided URLs)
    loc_node = evaluator.add_leaf(
        id="Location_Requirement",
        desc="The identified resort is located in Wisconsin Dells, Wisconsin, United States",
        parent=resort_node,
        critical=True
    )
    loc_claim = f"The resort '{resort_name}' is located in Wisconsin Dells, Wisconsin, United States."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=all_sources,
        additional_instruction=(
            "Check the address or location details on the provided sources. "
            "Allow minor naming variants (e.g., 'Wisconsin Dells, WI')."
        ),
    )

    # 4) Indoor_Waterpark_Facility (verify year-round operation)
    yr_node = evaluator.add_leaf(
        id="Indoor_Waterpark_Facility",
        desc="The resort has at least one indoor waterpark facility that operates year-round",
        parent=resort_node,
        critical=True
    )
    yr_claim = f"The indoor waterpark at '{resort_name}' operates year-round."
    await evaluator.verify(
        claim=yr_claim,
        node=yr_node,
        sources=all_sources,
        additional_instruction=(
            "Look for phrases like 'open year-round', 'open 365 days', or indications that the indoor waterpark operates throughout the year. "
            "General resort pages indicating year-round indoor waterpark access also count."
        ),
    )

    # 5) Waterpark_Size_Requirement (≥ 100,000 sq ft under one roof)
    size_req_node = evaluator.add_leaf(
        id="Waterpark_Size_Requirement",
        desc="The resort's single largest indoor waterpark facility under one roof is at least 100,000 square feet in size",
        parent=resort_node,
        critical=True
    )
    size_req_claim = (
        f"The largest single indoor waterpark facility under one roof at '{resort_name}' "
        f"is at least 100,000 square feet in size."
    )
    await evaluator.verify(
        claim=size_req_claim,
        node=size_req_node,
        sources=all_sources,
        additional_instruction=(
            "Verify explicit square footage statements from the sources. "
            "Allow minor rounding differences (e.g., 125,000 vs 125000). "
            "If multiple waterpark areas are mentioned, focus on the largest single indoor waterpark facility under one roof."
        ),
    )

    # 6) Onsite_Accommodations (verify hotel/lodging on-site)
    acc_node = evaluator.add_leaf(
        id="Onsite_Accommodations",
        desc="The resort offers on-site hotel accommodations for overnight guests",
        parent=resort_node,
        critical=True
    )
    acc_claim = f"The resort '{resort_name}' offers on-site hotel accommodations for overnight guests."
    await evaluator.verify(
        claim=acc_claim,
        node=acc_node,
        sources=all_sources,
        additional_instruction=(
            "Check for hotel rooms, suites, lodging pages, or any explicit mentions that overnight accommodations are available on-site."
        ),
    )

    # 7) Largest_Indoor_Waterpark_Claim (verify advertised claim 'Wisconsin's Largest Indoor Waterpark')
    largest_claim_node = evaluator.add_leaf(
        id="Largest_Indoor_Waterpark_Claim",
        desc="The resort advertises or claims the title of having 'Wisconsin's Largest Indoor Waterpark' based on its single largest indoor waterpark facility under one roof",
        parent=resort_node,
        critical=True
    )
    largest_claim = (
        f"The resort '{resort_name}' advertises or claims the title 'Wisconsin's Largest Indoor Waterpark'."
    )
    await evaluator.verify(
        claim=largest_claim,
        node=largest_claim_node,
        sources=official_sources if official_sources else all_sources,
        additional_instruction=(
            "Look for the explicit phrase 'Wisconsin's Largest Indoor Waterpark' or a very close variant on official resort pages. "
            "If official pages are not provided, reputable third-party sources clearly attributing this claim to the resort are acceptable."
        ),
    )

    # 8) Information_Verification (sub-checks ensuring name and specs are supported by sources)
    info_node = evaluator.add_parallel(
        id="Information_Verification",
        desc="The resort name and waterpark specifications are verifiable through official resort website or reputable third-party travel sources with provided URL references",
        parent=resort_node,
        critical=True
    )

    # 8a) Sources are provided at all
    evaluator.add_custom_node(
        result=len(all_sources) > 0,
        id="Info_Sources_Present",
        desc="At least one official or reputable third-party source URL is provided in the answer",
        parent=info_node,
        critical=True
    )

    # 8b) Resort name supported by sources
    name_supported_node = evaluator.add_leaf(
        id="Resort_Name_Supported",
        desc="The resort name is supported by the provided sources",
        parent=info_node,
        critical=True
    )
    name_supported_claim = (
        f"The provided sources explicitly identify or correspond to the resort named '{resort_name}'."
    )
    await evaluator.verify(
        claim=name_supported_claim,
        node=name_supported_node,
        sources=all_sources,
        additional_instruction=(
            "Check that the page(s) are clearly about the same resort name as in the answer. "
            "Allow minor naming variants (e.g., inclusion of city or brand)."
        ),
    )

    # 8c) Waterpark size supported by sources (exact size text)
    size_supported_node = evaluator.add_leaf(
        id="Waterpark_Size_Supported",
        desc="The indoor waterpark size (square feet) stated in the answer is supported by the provided sources",
        parent=info_node,
        critical=True
    )
    size_supported_claim = (
        f"The indoor waterpark size for '{resort_name}' is {size_text}."
    )
    await evaluator.verify(
        claim=size_supported_claim,
        node=size_supported_node,
        sources=all_sources,
        additional_instruction=(
            "Verify that the specific square footage mentioned in the answer is explicitly stated or clearly supported on the sources. "
            "Allow minor rounding differences (e.g., 125,000 vs 125000)."
        ),
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Wisconsin Dells indoor waterpark resort identification task.
    Returns a standard-format summary dict from the evaluator.
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

    # Extract structured resort info from the answer
    resort_info = await evaluator.extract(
        prompt=prompt_extract_resort_info(),
        template_class=ResortExtraction,
        extraction_name="resort_info",
    )

    # Build verification tree and run checks
    await build_resort_verification_tree(evaluator, root, resort_info)

    # Return structured result
    return evaluator.get_summary()