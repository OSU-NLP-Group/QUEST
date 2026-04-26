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
TASK_ID = "sd_cleveland_nf_waterfall_permit"
TASK_DESCRIPTION = """
Identify a popular hiking destination with a waterfall in San Diego County that requires advance permits and is located within Cleveland National Forest. For this destination, provide the following information: (1) The name of the hiking destination, (2) The cost per permit, (3) The maximum number of permits issued per day, and (4) A reference URL from Recreation.gov or the U.S. Forest Service that confirms the permit requirements.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DestinationPermitExtraction(BaseModel):
    """
    Extraction of the destination and its permit details from the answer text.
    All fields are strings as stated in the answer whenever possible.
    """
    destination_name: Optional[str] = None
    permit_cost: Optional[str] = None
    max_permits_per_day: Optional[str] = None
    group_size_policy: Optional[str] = None
    reference_url: Optional[str] = None  # Expected to be a Recreation.gov or U.S. Forest Service URL
    additional_urls: List[str] = Field(default_factory=list)  # Any other cited URLs that support claims


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_destination_and_permits() -> str:
    return """
    Extract details about the identified hiking destination and its permit information from the answer.

    You must extract the following fields exactly as they appear in the answer:
    - destination_name: The specific hiking destination's name (e.g., "Cedar Creek Falls"). If not provided, return null.
    - permit_cost: The stated cost per permit (e.g., "$10 per permit"). If not provided, return null.
    - max_permits_per_day: The stated maximum number of permits issued per day (e.g., "75 per day" or "75"). If not provided, return null.
    - group_size_policy: The stated group size covered by a single permit (e.g., "valid for up to 5 people"). If not provided, return null.
    - reference_url: A single official reference URL that confirms the permit requirement(s), preferably from Recreation.gov or the U.S. Forest Service. If multiple official URLs are provided, choose the one that most directly confirms the permit requirements. If no official URL is provided, return null.
    - additional_urls: A list of any other cited URLs in the answer that relate to this destination (e.g., pages that describe the location, the waterfall, popularity, or permit details). If none, return an empty list.

    Rules:
    - Only extract information explicitly present in the answer text.
    - For URLs, extract the actual URLs (including from markdown links).
    - Do not infer or invent details. If a field is not mentioned, return null (or empty list for additional_urls).
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not _non_empty(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _gather_sources(ex: DestinationPermitExtraction) -> List[str]:
    urls: List[str] = []
    if _non_empty(ex.reference_url):
        urls.append(ex.reference_url.strip())  # type: ignore
    urls.extend(ex.additional_urls or [])
    return _dedup_urls(urls)


def _is_official_url(url: Optional[str]) -> bool:
    if not _non_empty(url):
        return False
    u = url.lower().strip()  # type: ignore
    return ("recreation.gov" in u) or ("fs.usda.gov" in u)


# --------------------------------------------------------------------------- #
# Verification tree builders                                                  #
# --------------------------------------------------------------------------- #
async def build_destination_selection(
    evaluator: Evaluator,
    parent,
    ex: DestinationPermitExtraction
) -> None:
    """
    Build and verify the 'destination_selection' critical parallel node.
    This covers:
      - destination name provided
      - location within Cleveland NF (split with explicit San Diego County check)
      - has a waterfall
      - requires advance permits
      - permits obtainable on Recreation.gov
      - popularity supported by reputable source
    """
    ds_node = evaluator.add_parallel(
        id="destination_selection",
        desc="Destination is identified (named) and meets all selection constraints",
        parent=parent,
        critical=True
    )

    # Existence of destination name (critical existence check)
    evaluator.add_custom_node(
        result=_non_empty(ex.destination_name),
        id="destination_name_provided",
        desc="A specific hiking destination name is provided",
        parent=ds_node,
        critical=True
    )

    # Prepare sources for selection-level checks
    selection_sources = _gather_sources(ex)

    # Leaves for location checks (split into two concrete steps for clarity)
    within_cnf_node = evaluator.add_leaf(
        id="within_cleveland_nf",
        desc="The destination is located within Cleveland National Forest",
        parent=ds_node,
        critical=True
    )
    in_sd_county_node = evaluator.add_leaf(
        id="in_san_diego_county",
        desc="The destination is located in San Diego County",
        parent=ds_node,
        critical=True
    )

    # Waterfall existence
    waterfall_node = evaluator.add_leaf(
        id="has_waterfall",
        desc="The destination features a waterfall",
        parent=ds_node,
        critical=True
    )

    # Advance permit requirement
    advance_permit_node = evaluator.add_leaf(
        id="requires_advance_permit",
        desc="The destination requires advance permits",
        parent=ds_node,
        critical=True
    )

    # Permits available via Recreation.gov
    recgov_available_node = evaluator.add_leaf(
        id="permits_available_on_recreation_gov",
        desc="Permits for the destination are obtainable through Recreation.gov",
        parent=ds_node,
        critical=True
    )

    # Popularity support
    popular_node = evaluator.add_leaf(
        id="popular_hiking_location",
        desc="Popularity is supported by evidence from a reputable source (explicitly 'popular' or objective indicator)",
        parent=ds_node,
        critical=True
    )

    dest_name = ex.destination_name or ""

    # Batch verify the destination selection claims
    claims_and_sources = [
        (
            f"The hiking destination '{dest_name}' is located within Cleveland National Forest.",
            selection_sources,
            within_cnf_node,
            "Look for explicit mention that the site/hike is within the Cleveland National Forest. "
            "Accept if the official page is clearly within Cleveland NF and is about this destination."
        ),
        (
            f"The hiking destination '{dest_name}' is located in San Diego County, California.",
            selection_sources,
            in_sd_county_node,
            "Prefer explicit textual evidence like 'San Diego County.' "
            "If the page only mentions a town (e.g., Ramona) without stating 'San Diego County', that is insufficient—do not rely on your own knowledge."
        ),
        (
            f"The hiking destination '{dest_name}' includes or leads to a waterfall.",
            selection_sources,
            waterfall_node,
            "Look for words like 'waterfall', 'falls', or explicit description of a waterfall at or as the destination."
        ),
        (
            f"Advance permits are required to access or hike to '{dest_name}'.",
            selection_sources,
            advance_permit_node,
            "Look for 'permit required', 'advance permits', 'day-use permit required', or similar language indicating permits must be obtained in advance."
        ),
        (
            f"Permits for '{dest_name}' are obtainable via Recreation.gov.",
            selection_sources,
            recgov_available_node,
            "This can be confirmed either by a Recreation.gov permit page or by an official U.S. Forest Service page explicitly stating that permits are available on Recreation.gov."
        ),
        (
            f"'{dest_name}' is a popular hiking destination.",
            selection_sources,
            popular_node,
            "Accept clear statements like 'popular', 'very popular area', 'heavily visited', or objective indicators of popularity (e.g., a frequently featured listing). "
            "Reputable sources include Recreation.gov, U.S. Forest Service pages, and well-known hiking platforms like AllTrails."
        ),
    ]
    await evaluator.batch_verify(claims_and_sources)


async def build_required_details_and_citation(
    evaluator: Evaluator,
    parent,
    ex: DestinationPermitExtraction
) -> None:
    """
    Build and verify the 'required_permit_details_and_citation' critical parallel node.
    This covers:
      - permit cost (stated as $10 per permit) and group size limit (up to 5 people)
      - daily maximum permits (75/day)
      - a valid official reference URL (Recreation.gov or U.S. Forest Service) that confirms permit requirements
    """
    rp_node = evaluator.add_parallel(
        id="required_permit_details_and_citation",
        desc="All required permit details and a confirming reference URL are provided",
        parent=parent,
        critical=True
    )

    # --- Reference URL verification group (evaluate first) ---
    ref_group = evaluator.add_parallel(
        id="reference_url_group",
        desc="Reference URL validation and confirmation of permit requirement",
        parent=rp_node,
        critical=True
    )

    # Existence
    ref_url_provided = evaluator.add_custom_node(
        result=_non_empty(ex.reference_url),
        id="reference_url_provided",
        desc="An official reference URL is provided",
        parent=ref_group,
        critical=True
    )

    # Domain validity
    domain_valid = evaluator.add_custom_node(
        result=_is_official_url(ex.reference_url),
        id="reference_url_is_official_domain",
        desc="Reference URL domain is Recreation.gov or U.S. Forest Service (fs.usda.gov)",
        parent=ref_group,
        critical=True
    )

    # Content confirmation: advance permits required
    ref_confirms_permit = evaluator.add_leaf(
        id="reference_url_confirms_permit_requirement",
        desc="Reference URL confirms that permits are required",
        parent=ref_group,
        critical=True
    )
    # Verify using only the official reference URL when available
    await evaluator.verify(
        claim=f"Advance permits are required to visit or hike to '{ex.destination_name or ''}'.",
        node=ref_confirms_permit,
        sources=ex.reference_url if _non_empty(ex.reference_url) else None,
        additional_instruction="Use only this page to confirm that permits are required (e.g., 'permit required', 'advance permits')."
    )

    # --- Permit Cost group ---
    cost_group = evaluator.add_parallel(
        id="permit_cost_group",
        desc="Permit cost details",
        parent=rp_node,
        critical=True
    )

    # Presence of permit cost in the answer
    evaluator.add_custom_node(
        result=_non_empty(ex.permit_cost),
        id="permit_cost_provided",
        desc="Permit cost is provided in the answer",
        parent=cost_group,
        critical=True
    )

    # The answer itself states $10 per permit (simple check against the answer text)
    cost_answer_is_10 = evaluator.add_leaf(
        id="permit_cost_answer_is_10",
        desc="The answer states the permit cost is $10 per permit",
        parent=cost_group,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the permit costs $10 per permit.",
        node=cost_answer_is_10,
        sources=None,
        additional_instruction="Accept minor formatting variants like '$10', '10 USD', or '10 dollars'. The key is that the answer clearly indicates $10 per permit."
    )

    # Official source confirms $10 per permit
    cost_supported_by_source = evaluator.add_leaf(
        id="permit_cost_supported_by_official_source",
        desc="Official source confirms the permit cost is $10 per permit",
        parent=cost_group,
        critical=True
    )
    await evaluator.verify(
        claim="The cost per permit is $10.",
        node=cost_supported_by_source,
        sources=ex.reference_url if _non_empty(ex.reference_url) else _gather_sources(ex),
        additional_instruction="Check the official page for the permit fee. Ignore separate processing/booking fees; the core permit cost should be $10."
    )

    # Group size limit per permit: up to 5 people
    group_size_supported = evaluator.add_leaf(
        id="permit_group_size_limit_supported",
        desc="Official source confirms a single permit is valid for up to 5 people",
        parent=cost_group,
        critical=True
    )
    await evaluator.verify(
        claim="Each permit is valid for up to 5 people.",
        node=group_size_supported,
        sources=ex.reference_url if _non_empty(ex.reference_url) else _gather_sources(ex),
        additional_instruction="Look for language like 'valid for up to 5 people', 'group size: 1–5', or equivalent statements on the official page."
    )

    # --- Daily permit limit group ---
    daily_group = evaluator.add_parallel(
        id="daily_permit_limit_group",
        desc="Daily permit limit details",
        parent=rp_node,
        critical=True
    )

    # Presence of daily permit limit in the answer
    evaluator.add_custom_node(
        result=_non_empty(ex.max_permits_per_day),
        id="daily_permit_limit_provided",
        desc="Maximum number of permits per day is provided in the answer",
        parent=daily_group,
        critical=True
    )

    # The answer itself states 75 per day (simple check)
    daily_answer_is_75 = evaluator.add_leaf(
        id="daily_permit_limit_answer_is_75",
        desc="The answer states the daily permit limit is 75",
        parent=daily_group,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states that the maximum number of permits issued per day is 75.",
        node=daily_answer_is_75,
        sources=None,
        additional_instruction="Accept variants like '75 per day', '75 permits/day', or 'daily quota: 75'."
    )

    # Official source confirms 75 per day
    daily_supported_by_source = evaluator.add_leaf(
        id="daily_permit_limit_supported_by_official_source",
        desc="Official source confirms that 75 permits are issued per day",
        parent=daily_group,
        critical=True
    )
    await evaluator.verify(
        claim="A maximum of 75 permits are issued per day.",
        node=daily_supported_by_source,
        sources=ex.reference_url if _non_empty(ex.reference_url) else _gather_sources(ex),
        additional_instruction="Confirm that the official page states a daily limit of 75 permits (e.g., 'Daily quota: 75')."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the San Diego County waterfall destination in Cleveland NF permit task.
    Returns a structured evaluation summary dictionary.
    """
    # Initialize evaluator with a sequential root as per rubric
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured details from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_destination_and_permits(),
        template_class=DestinationPermitExtraction,
        extraction_name="destination_permit_extraction"
    )

    # Build and verify the destination selection block
    await build_destination_selection(evaluator, root, extraction)

    # Build and verify the required permit details and citation block
    await build_required_details_and_citation(evaluator, root, extraction)

    # Return the evaluation summary
    return evaluator.get_summary()