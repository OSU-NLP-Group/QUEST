import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_airport_overlook_facilities"
TASK_DESCRIPTION = (
    "A family is planning a summer road trip across the United States and wants to visit airport overlook facilities "
    "where their children can enjoy outdoor recreation while watching aircraft operations. They are specifically "
    "looking for locations that offer a complete free family experience without admission fees.\n\n"
    "Identify 4 different airport overlook facilities in the United States that meet ALL of the following requirements:\n"
    "1. The facility must offer free public access to the outdoor overlook area with no admission fee required\n"
    "2. The facility must include playground equipment or a designated children's play area\n"
    "3. The facility must provide clear views of aircraft operations (such as takeoffs, landings, or taxiing)\n"
    "4. The facility must offer free parking for visitors\n"
    "5. The facility must include on-site restroom facilities\n\n"
    "For each facility, provide:\n"
    "- The facility name\n"
    "- The associated airport name and city/state location\n"
    "- A brief description of its key features\n"
    "- A reference URL from an official airport website, government website, or established travel resource that confirms the facility details"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FacilityItem(BaseModel):
    facility_name: Optional[str] = None
    airport_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    location_text: Optional[str] = None
    description: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    facilities: List[FacilityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return (
        "Extract up to 6 airport overlook or aircraft viewing facilities mentioned in the answer. Return a JSON object "
        "with a 'facilities' array, where each element has:\n"
        "- facility_name: The specific facility/park/overlook name as written\n"
        "- airport_name: The associated airport name\n"
        "- city: The city name (if provided)\n"
        "- state: The U.S. state name or postal abbreviation (if provided)\n"
        "- location_text: The location string as written in the answer (e.g., 'Charlotte, NC' or 'Kent, Washington')\n"
        "- description: A brief description of key features, as provided in the answer\n"
        "- reference_urls: A list of URLs explicitly cited in the answer that confirm details about this facility. "
        "Include only valid URLs. Prefer official airport sites, government sites, or established travel resources.\n\n"
        "Rules:\n"
        "1) Do not invent any data. Only extract what is explicitly stated.\n"
        "2) If a field is missing, set it to null. If no URLs are cited, use an empty list.\n"
        "3) Extract URLs in their full form. If a URL is missing a protocol, prepend 'http://'.\n"
        "4) If the answer lists more than 4 facilities, still extract all, we will filter to the first 4 later."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _non_empty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _format_location_for_claim(item: FacilityItem) -> str:
    # Build a human-readable location string prioritizing city/state
    city = item.city.strip() if _non_empty(item.city) else None
    state = item.state.strip() if _non_empty(item.state) else None
    if city and state:
        return f"{city}, {state}, United States"
    if state:
        return f"{state}, United States"
    # Fall back to location_text if present
    if _non_empty(item.location_text):
        # Try to append ", United States" to the free-form text
        lt = item.location_text.strip()
        if "United States" in lt or "USA" in lt or "U.S." in lt:
            return lt
        return f"{lt}, United States"
    return "United States"


def _first_n_facilities(extraction: FacilitiesExtraction, n: int = 4) -> List[FacilityItem]:
    items = list(extraction.facilities[:n])
    while len(items) < n:
        items.append(FacilityItem())
    return items


def _sources(item: FacilityItem) -> List[str]:
    # Return only valid-looking URLs
    return [u for u in (item.reference_urls or []) if _non_empty(u)]


# --------------------------------------------------------------------------- #
# Verification routines                                                       #
# --------------------------------------------------------------------------- #
async def verify_facility(
    evaluator: Evaluator,
    parent_node,
    item: FacilityItem,
    idx: int,
) -> None:
    """
    Build and execute verification subtree for a single facility.
    Maps directly to rubric leaf nodes for Facility_{i}.
    """
    fac_num = idx + 1
    fac_node = evaluator.add_parallel(
        id=f"Facility_{fac_num}",
        desc=f"{['First','Second','Third','Fourth'][idx]} qualifying US airport overlook facility",
        parent=parent_node,
        critical=False
    )

    # Existence checks required by rubric
    name_exists = evaluator.add_custom_node(
        result=_non_empty(item.facility_name),
        id=f"F{fac_num}_Facility_Name_Provided",
        desc="Solution provides the facility name",
        parent=fac_node,
        critical=True
    )
    loc_exists = evaluator.add_custom_node(
        result=_non_empty(item.airport_name) and (_non_empty(item.city) or _non_empty(item.state)),
        id=f"F{fac_num}_Airport_Location_Provided",
        desc="Solution provides the associated airport name and city/state location",
        parent=fac_node,
        critical=True
    )
    desc_exists = evaluator.add_custom_node(
        result=_non_empty(item.description),
        id=f"F{fac_num}_Description_Provided",
        desc="Solution provides a brief description of key features",
        parent=fac_node,
        critical=True
    )

    # Reference URL leaf (critical) – verify credibility and relevance
    ref_leaf_sources = _sources(item)
    ref_leaf_desc = (
        "Valid reference URL from an official airport website, government website, or established travel resource "
        "confirming the facility details"
    )
    if len(ref_leaf_sources) == 0:
        ref_leaf = evaluator.add_leaf(
            id=f"F{fac_num}_Reference_URL",
            desc=ref_leaf_desc,
            parent=fac_node,
            critical=True,
            score=0.0,
            status="failed"
        )
    else:
        ref_leaf = evaluator.add_leaf(
            id=f"F{fac_num}_Reference_URL",
            desc=ref_leaf_desc,
            parent=fac_node,
            critical=True
        )
        ref_claim = (
            f"The provided URL(s) are from an official airport website, a government site, or an established travel "
            f"resource, and they confirm the existence and basic details of the facility "
            f"'{item.facility_name or ''}' at '{item.airport_name or ''}' in {_format_location_for_claim(item)}."
        )
        await evaluator.verify(
            claim=ref_claim,
            node=ref_leaf,
            sources=ref_leaf_sources,
            additional_instruction=(
                "Judge credibility by domain and on-page signals:\n"
                "- Official airport websites (airport-operated pages) typically include airport branding or official sections.\n"
                "- Government sites often use .gov TLDs or municipal/parks department pages.\n"
                "- Established travel resources are well-known, reputable travel information sites (e.g., "
                "state tourism boards, long-standing travel guides, or widely-recognized platforms). "
                "Blogs or random personal pages should not count.\n"
                "Also confirm the page is clearly about the specific facility, not a generic airport page."
            )
        )

    # Prepare verification leaves for all constraints; gate them on reference URL
    # 1) US location
    us_loc_leaf = evaluator.add_leaf(
        id=f"F{fac_num}_US_Location",
        desc="Facility is located in the United States",
        parent=fac_node,
        critical=True
    )
    us_claim = f"The facility '{item.facility_name or ''}' is located in {_format_location_for_claim(item)}."
    await evaluator.verify(
        claim=us_claim,
        node=us_loc_leaf,
        sources=ref_leaf_sources if len(ref_leaf_sources) > 0 else None,
        additional_instruction=(
            "Confirm the facility is in the United States. Accept U.S. state names or postal abbreviations as evidence. "
            "If the page clearly indicates a U.S. locality (city/state), consider the claim supported."
        ),
        extra_prerequisites=[ref_leaf, name_exists, loc_exists]
    )

    # 2) Airport association
    assoc_leaf = evaluator.add_leaf(
        id=f"F{fac_num}_Airport_Association",
        desc="Facility is airport-adjacent or airport-operated",
        parent=fac_node,
        critical=True
    )
    assoc_claim = (
        f"The facility '{item.facility_name or ''}' is adjacent to the airport grounds or operated by "
        f"the airport '{item.airport_name or ''}'."
    )
    await evaluator.verify(
        claim=assoc_claim,
        node=assoc_leaf,
        sources=ref_leaf_sources if len(ref_leaf_sources) > 0 else None,
        additional_instruction=(
            "Look for wording such as 'observation park/area', 'airport viewing area', 'airport-operated', or "
            "explicit mention that the facility is on/next to airport property."
        ),
        extra_prerequisites=[ref_leaf, name_exists, loc_exists]
    )

    # 3) Free public access (no admission fee)
    free_access_leaf = evaluator.add_leaf(
        id=f"F{fac_num}_Free_Public_Access",
        desc="Facility offers free public access to outdoor overlook area with no admission fee",
        parent=fac_node,
        critical=True
    )
    free_access_claim = (
        f"The outdoor overlook area at '{item.facility_name or ''}' offers free public access with no admission fee."
    )
    await evaluator.verify(
        claim=free_access_claim,
        node=free_access_leaf,
        sources=ref_leaf_sources if len(ref_leaf_sources) > 0 else None,
        additional_instruction=(
            "Prefer explicit language: 'free', 'no admission', 'no fee'. If the page strongly indicates a public park "
            "or airport-operated observation area with no mention of fees, you may consider it supported."
        ),
        extra_prerequisites=[ref_leaf, name_exists]
    )

    # 4) Playground present
    playground_leaf = evaluator.add_leaf(
        id=f"F{fac_num}_Playground_Present",
        desc="Facility includes playground equipment or designated children's play area",
        parent=fac_node,
        critical=True
    )
    playground_claim = (
        f"The facility '{item.facility_name or ''}' includes playground equipment or a designated children's play area."
    )
    await evaluator.verify(
        claim=playground_claim,
        node=playground_leaf,
        sources=ref_leaf_sources if len(ref_leaf_sources) > 0 else None,
        additional_instruction=(
            "Look for terms such as 'playground', 'play area', 'slides', 'swings', 'jungle gym', or similar."
        ),
        extra_prerequisites=[ref_leaf, name_exists]
    )

    # 5) Aircraft viewing capability
    viewing_leaf = evaluator.add_leaf(
        id=f"F{fac_num}_Aircraft_Viewing",
        desc="Facility provides views of aircraft operations (takeoffs, landings, or taxiing)",
        parent=fac_node,
        critical=True
    )
    viewing_claim = (
        f"The facility '{item.facility_name or ''}' provides clear views of aircraft operations at "
        f"'{item.airport_name or ''}' (e.g., takeoffs, landings, taxiing)."
    )
    await evaluator.verify(
        claim=viewing_claim,
        node=viewing_leaf,
        sources=ref_leaf_sources if len(ref_leaf_sources) > 0 else None,
        additional_instruction=(
            "Accept phrases such as 'watch planes', 'runway views', 'aircraft viewing area', 'plane spotting', etc."
        ),
        extra_prerequisites=[ref_leaf, name_exists]
    )

    # 6) Free parking
    parking_leaf = evaluator.add_leaf(
        id=f"F{fac_num}_Free_Parking",
        desc="Facility offers free parking for visitors",
        parent=fac_node,
        critical=True
    )
    parking_claim = f"Visitors can park for free at '{item.facility_name or ''}'."
    await evaluator.verify(
        claim=parking_claim,
        node=parking_leaf,
        sources=ref_leaf_sources if len(ref_leaf_sources) > 0 else None,
        additional_instruction=(
            "Look for explicit mention of 'free parking' or 'no parking fee'. If only 'parking available' is stated "
            "without fee information, do NOT consider it supported."
        ),
        extra_prerequisites=[ref_leaf, name_exists]
    )

    # 7) Restroom facilities
    restroom_leaf = evaluator.add_leaf(
        id=f"F{fac_num}_Restroom_Facilities",
        desc="Facility includes on-site restroom facilities",
        parent=fac_node,
        critical=True
    )
    restroom_claim = f"On-site restroom facilities are available at '{item.facility_name or ''}'."
    await evaluator.verify(
        claim=restroom_claim,
        node=restroom_leaf,
        sources=ref_leaf_sources if len(ref_leaf_sources) > 0 else None,
        additional_instruction=(
            "Look for words like 'restrooms', 'bathrooms', 'toilets'. If the page clearly indicates on-site restrooms, "
            "consider supported."
        ),
        extra_prerequisites=[ref_leaf, name_exists]
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the US airport overlook facilities task.
    """
    # Initialize evaluator and root
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

    # Create rubric root node to mirror provided rubric tree
    rubric_root = evaluator.add_parallel(
        id="US_Airport_Overlook_Facilities",
        desc="Identify 4 US airport overlook facilities that offer free public outdoor recreation areas with specific family-friendly amenities",
        parent=root,
        critical=False
    )

    # Extract structured facilities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # Limit to the first 4; pad if fewer
    facilities = _first_n_facilities(extracted, n=4)

    # Build verification subtrees for each facility
    for i, item in enumerate(facilities):
        await verify_facility(evaluator, rubric_root, item, i)

    # Return evaluation summary
    return evaluator.get_summary()