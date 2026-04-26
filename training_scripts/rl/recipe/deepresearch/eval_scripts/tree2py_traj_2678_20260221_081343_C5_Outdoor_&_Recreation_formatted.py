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
TASK_ID = "accessible_parks_west"
TASK_DESCRIPTION = (
    "I am planning a trip to national parks in the western United States and need to find parks that are highly accessible for wheelchair users. "
    "Please identify two national parks in the western United States that meet all of the following accessibility requirements:\n\n"
    "1. Each park must have at least one wheelchair-accessible paved trail that is at least 0.5 miles in length and provides access to a significant natural feature (such as a waterfall, geyser basin, canyon viewpoint, or similar landmark)\n\n"
    "2. Each park must have a visitor center with ADA-compliant accessible restrooms and accessible parking spaces\n\n"
    "3. Each park must have at least one campground with ADA-accessible campsites that can be reserved through Recreation.gov\n\n"
    "For each park, please provide:\n"
    "- The park name and confirm it is located in the western United States\n"
    "- The name and key details of the accessible trail (length, surface type, what natural feature it accesses)\n"
    "- Information about the visitor center's accessible facilities\n"
    "- Information about the accessible campground and its availability on Recreation.gov\n"
    "- Reference URLs supporting each piece of information\n\n"
    "Additionally, please provide information about the America the Beautiful Access Pass, specifically confirming eligibility requirements for US citizens or permanent residents with permanent disabilities, along with a reference URL."
)

# Western US states list used for judging "western"
WESTERN_US_STATES = [
    "Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho", "Montana", "Nevada",
    "New Mexico", "Oregon", "Utah", "Washington", "Wyoming",
    # Allow postal abbreviations
    "AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"
]


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkTrailInfo(BaseModel):
    name: Optional[str] = None
    length_miles: Optional[str] = None  # Keep as string to accommodate "0.8 mile", "1 km", etc.
    surface: Optional[str] = None       # e.g., "paved", "boardwalk"
    significant_feature: Optional[str] = None
    accessible_parking_info: Optional[str] = None  # textual note if available
    sources: List[str] = Field(default_factory=list)


class VisitorCenterInfo(BaseModel):
    name: Optional[str] = None
    ada_restrooms_info: Optional[str] = None
    accessible_parking_info: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CampgroundInfo(BaseModel):
    name: Optional[str] = None
    ada_campsites_info: Optional[str] = None
    recreation_gov_url: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ParkInfo(BaseModel):
    name: Optional[str] = None
    state_or_location: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)
    trail: ParkTrailInfo = Field(default_factory=ParkTrailInfo)
    visitor_center: VisitorCenterInfo = Field(default_factory=VisitorCenterInfo)
    campground: CampgroundInfo = Field(default_factory=CampgroundInfo)


class AccessPassInfo(BaseModel):
    eligibility_text: Optional[str] = None
    source_url: Optional[str] = None


class ParksAndPassExtraction(BaseModel):
    parks: List[ParkInfo] = Field(default_factory=list)
    access_pass: AccessPassInfo = Field(default_factory=AccessPassInfo)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks_and_pass() -> str:
    return """
    Extract exactly two national parks (if the answer mentions more than two, select the first two; if fewer than two, include the available ones and set missing fields to null). 
    For each park, extract the following structured fields:

    Park-level:
    - name: Full park name as stated in the answer
    - state_or_location: State(s) or region associated with the park (e.g., "Wyoming", "UT", "California")
    - location_sources: URLs used in the answer to support the park's location/western US claim (array of URLs)

    Accessible trail:
    - trail.name: Trail name
    - trail.length_miles: Length expressed in miles if available (keep the original string; it can be "0.8 miles", "1 mile", "1.2 mi", or a km value)
    - trail.surface: Surface type (e.g., paved, boardwalk)
    - trail.significant_feature: The significant natural feature accessed (e.g., waterfall name, geyser basin name, canyon viewpoint)
    - trail.accessible_parking_info: Any note about accessible parking at/near the trailhead (free text from the answer)
    - trail.sources: URLs used to support the trail details (array of URLs)

    Visitor center:
    - visitor_center.name: Visitor center name
    - visitor_center.ada_restrooms_info: Free text indicating ADA-compliant/accessible restrooms are present
    - visitor_center.accessible_parking_info: Free text indicating accessible parking spaces are present
    - visitor_center.sources: URLs used to support visitor center details (array of URLs)

    Campground:
    - campground.name: Campground name
    - campground.ada_campsites_info: Free text indicating ADA-accessible campsites are present
    - campground.recreation_gov_url: Recreation.gov URL for reservations (if provided; otherwise null)
    - campground.sources: URLs used to support campground details (array of URLs)

    Access Pass (America the Beautiful – Access Pass):
    - access_pass.eligibility_text: The eligibility statement extracted from the answer (e.g., "US citizens or permanent residents with permanent disabilities")
    - access_pass.source_url: URL used to support the eligibility statement (official or authoritative)

    IMPORTANT EXTRACTION RULES:
    - Extract only what is explicitly present in the answer; do not invent.
    - For any missing information, return null (or an empty array for URLs).
    - URLs may appear as plain text or Markdown links; extract the actual URL string.
    - Keep all strings exactly as they appear (do not normalize or paraphrase).
    - Always include arrays for the per-section sources (trail.sources, visitor_center.sources, campground.sources, and location_sources).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth"][n] if n < 4 else f"#{n+1}"


def _valid_urls(urls: List[str]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://"))]


def _any_valid_url(urls: List[str]) -> bool:
    return len(_valid_urls(urls)) > 0


def _park_all_sources(park: ParkInfo) -> List[str]:
    urls: List[str] = []
    urls.extend(_valid_urls(park.location_sources))
    urls.extend(_valid_urls(park.trail.sources))
    urls.extend(_valid_urls(park.visitor_center.sources))
    urls.extend(_valid_urls(park.campground.sources))
    if park.campground.recreation_gov_url:
        urls.append(park.campground.recreation_gov_url)
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_park(evaluator: Evaluator, parent_node, park: ParkInfo, park_index: int) -> None:
    park_label = _ordinal(park_index)
    park_node = evaluator.add_parallel(
        id=f"{park_label.lower()}_park",
        desc=f"{park_label} national park meeting all accessibility criteria",
        parent=parent_node,
        critical=True  # All child criteria are critical for a park to pass
    )

    # Park Location (critical leaf)
    location_leaf = evaluator.add_leaf(
        id=f"{park_label.lower()}_park_location",
        desc="Park is located in the western United States",
        parent=park_node,
        critical=True
    )
    location_claim = f"{park.name or 'The park'} is located in the western United States."
    # Use all available sources to support location/state determination
    loc_sources = _park_all_sources(park)
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=loc_sources if loc_sources else None,
        additional_instruction=(
            "Use only the provided URLs to determine the park's state(s). "
            "Judge 'Correct' if the park is in any of these Western US states: "
            f"{', '.join(WESTERN_US_STATES)}. "
            "If no URLs are provided or the sources do not identify the state, judge 'Incorrect'."
        ),
    )

    # Accessible Trail group (critical)
    trail_group = evaluator.add_parallel(
        id=f"{park_label.lower()}_accessible_trail",
        desc="Park has a wheelchair-accessible paved trail meeting specified requirements",
        parent=park_node,
        critical=True
    )
    trail_sources = _valid_urls(park.trail.sources)

    # Trail_Length
    trail_len_leaf = evaluator.add_leaf(
        id=f"{park_label.lower()}_trail_length",
        desc="Trail is at least 0.5 miles in length",
        parent=trail_group,
        critical=True
    )
    len_claim = f"The trail '{park.trail.name or 'the trail'}' is at least 0.5 miles long."
    await evaluator.verify(
        claim=len_claim,
        node=trail_len_leaf,
        sources=trail_sources if trail_sources else None,
        additional_instruction=(
            "Examine the sources carefully for trail length. Accept if the length is ≥ 0.5 miles. "
            "If the length is given in kilometers, convert (0.8 km ≈ 0.5 miles). "
            "If any source indicates the length < 0.5 miles or no length is provided, judge 'Incorrect'. "
            f"The answer's stated length is: {park.trail.length_miles or 'unknown'}."
        ),
    )

    # Trail_Accessibility
    trail_access_leaf = evaluator.add_leaf(
        id=f"{park_label.lower()}_trail_accessibility",
        desc="Trail is described as wheelchair-accessible (paved or boardwalk)",
        parent=trail_group,
        critical=True
    )
    access_claim = f"The trail '{park.trail.name or 'the trail'}' is wheelchair-accessible and is paved or boardwalk."
    await evaluator.verify(
        claim=access_claim,
        node=trail_access_leaf,
        sources=trail_sources if trail_sources else None,
        additional_instruction=(
            "Look for explicit indications such as 'wheelchair-accessible', 'ADA accessible', 'paved path', or 'boardwalk'. "
            "If accessibility is not stated, or surface is not paved/boardwalk, judge 'Incorrect'."
        ),
    )

    # Significant_Feature
    trail_feature_leaf = evaluator.add_leaf(
        id=f"{park_label.lower()}_significant_feature",
        desc="Trail provides access to a significant natural feature (waterfall, geyser basin, canyon viewpoint, or similar)",
        parent=trail_group,
        critical=True
    )
    feature_desc = park.trail.significant_feature or "a significant natural feature"
    feature_claim = f"The trail '{park.trail.name or 'the trail'}' provides access to {feature_desc}."
    await evaluator.verify(
        claim=feature_claim,
        node=trail_feature_leaf,
        sources=trail_sources if trail_sources else None,
        additional_instruction=(
            "Confirm that the trail leads to, passes through, or provides views of the specified significant feature. "
            "If the sources do not mention such a feature, judge 'Incorrect'."
        ),
    )

    # Trailhead_Parking
    trail_parking_leaf = evaluator.add_leaf(
        id=f"{park_label.lower()}_trailhead_parking",
        desc="Accessible parking is available at the trailhead",
        parent=trail_group,
        critical=True
    )
    trail_parking_claim = f"Accessible parking is available at or near the trailhead for the trail '{park.trail.name or 'the trail'}'."
    await evaluator.verify(
        claim=trail_parking_claim,
        node=trail_parking_leaf,
        sources=trail_sources if trail_sources else None,
        additional_instruction=(
            "Look for mentions of 'accessible parking' or designated accessible spaces at/near the trailhead or starting area. "
            "Generic parking without accessibility designation is insufficient."
        ),
    )

    # Trail_Reference (custom existence check)
    trail_ref_leaf = evaluator.add_custom_node(
        result=_any_valid_url(park.trail.sources),
        id=f"{park_label.lower()}_trail_reference",
        desc="Provides valid reference URL supporting trail information",
        parent=trail_group,
        critical=True
    )

    # Visitor Center group (critical)
    vc_group = evaluator.add_parallel(
        id=f"{park_label.lower()}_visitor_center",
        desc="Park has a visitor center with required accessible amenities",
        parent=park_node,
        critical=True
    )
    vc_sources = _valid_urls(park.visitor_center.sources)

    # ADA_Restrooms
    vc_rest_leaf = evaluator.add_leaf(
        id=f"{park_label.lower()}_vc_ada_restrooms",
        desc="Visitor center has ADA-compliant accessible restrooms",
        parent=vc_group,
        critical=True
    )
    vc_rest_claim = f"The visitor center '{park.visitor_center.name or 'the visitor center'}' has ADA-compliant accessible restrooms."
    await evaluator.verify(
        claim=vc_rest_claim,
        node=vc_rest_leaf,
        sources=vc_sources if vc_sources else None,
        additional_instruction=(
            "Confirm explicit mention of accessible or ADA-compliant restrooms at the visitor center. "
            "If no such mention exists or no URLs are provided, judge 'Incorrect'."
        ),
    )

    # Accessible_Parking
    vc_parking_leaf = evaluator.add_leaf(
        id=f"{park_label.lower()}_vc_accessible_parking",
        desc="Visitor center has accessible parking spaces",
        parent=vc_group,
        critical=True
    )
    vc_parking_claim = f"The visitor center '{park.visitor_center.name or 'the visitor center'}' has accessible parking spaces."
    await evaluator.verify(
        claim=vc_parking_claim,
        node=vc_parking_leaf,
        sources=vc_sources if vc_sources else None,
        additional_instruction=(
            "Look for explicit mentions of 'accessible parking' or designated accessible spaces at the visitor center. "
            "Generic 'parking available' without accessibility designation is insufficient."
        ),
    )

    # Visitor_Center_Reference (custom existence check)
    vc_ref_leaf = evaluator.add_custom_node(
        result=_any_valid_url(park.visitor_center.sources),
        id=f"{park_label.lower()}_visitor_center_reference",
        desc="Provides valid reference URL supporting visitor center information",
        parent=vc_group,
        critical=True
    )

    # Accessible Campground group (critical)
    cg_group = evaluator.add_parallel(
        id=f"{park_label.lower()}_accessible_campground",
        desc="Park has a campground with ADA-accessible sites available through Recreation.gov",
        parent=park_node,
        critical=True
    )
    cg_sources = _valid_urls(park.campground.sources)
    recgov_url = park.campground.recreation_gov_url if isinstance(park.campground.recreation_gov_url, str) else None

    # ADA_Campsites
    cg_ada_leaf = evaluator.add_leaf(
        id=f"{park_label.lower()}_cg_ada_campsites",
        desc="Campground has ADA-accessible campsites",
        parent=cg_group,
        critical=True
    )
    cg_ada_claim = f"The campground '{park.campground.name or 'the campground'}' has ADA-accessible campsites."
    cg_ada_sources = cg_sources + ([recgov_url] if recgov_url else [])
    await evaluator.verify(
        claim=cg_ada_claim,
        node=cg_ada_leaf,
        sources=cg_ada_sources if cg_ada_sources else None,
        additional_instruction=(
            "Check for mentions of 'accessible' or 'ADA' campsites in the campground or Recreation.gov listing. "
            "If accessibility is not indicated, judge 'Incorrect'."
        ),
    )

    # Recreation_Gov
    cg_recgov_leaf = evaluator.add_leaf(
        id=f"{park_label.lower()}_cg_recreation_gov",
        desc="Campground reservations are available through Recreation.gov",
        parent=cg_group,
        critical=True
    )
    cg_recgov_claim = f"Reservations for the campground '{park.campground.name or 'the campground'}' are available through Recreation.gov."
    await evaluator.verify(
        claim=cg_recgov_claim,
        node=cg_recgov_leaf,
        sources=recgov_url if recgov_url else None,
        additional_instruction=(
            "Verify that the provided URL is a valid Recreation.gov listing for the campground or area (domain recreation.gov) "
            "and indicates reservation capability. If no Recreation.gov URL is provided, judge 'Incorrect'."
        ),
    )

    # Campground_Reference (custom existence check; accept either rec.gov url or other campground sources)
    cg_ref_leaf = evaluator.add_custom_node(
        result=(_any_valid_url(park.campground.sources) or (isinstance(recgov_url, str) and recgov_url.lower().startswith(("http://", "https://")))),
        id=f"{park_label.lower()}_campground_reference",
        desc="Provides valid reference URL supporting campground information",
        parent=cg_group,
        critical=True
    )


async def verify_access_pass(evaluator: Evaluator, parent_node, access_pass: AccessPassInfo) -> None:
    pass_group = evaluator.add_parallel(
        id="access_pass_information",
        desc="Provide accurate information about the America the Beautiful Access Pass",
        parent=parent_node,
        critical=True
    )

    # Pass_Eligibility (leaf)
    eligibility_leaf = evaluator.add_leaf(
        id="pass_eligibility",
        desc="Confirms that US citizens or permanent residents with permanent disabilities qualify for the free Access Pass",
        parent=pass_group,
        critical=True
    )
    eligibility_claim = (
        "The America the Beautiful Access Pass is available to U.S. citizens or permanent residents who have permanent disabilities."
    )
    await evaluator.verify(
        claim=eligibility_claim,
        node=eligibility_leaf,
        sources=access_pass.source_url if access_pass.source_url else None,
        additional_instruction=(
            "Use the provided official/authoritative URL to confirm eligibility. "
            "If no URL is provided, judge 'Incorrect'."
        ),
    )

    # Access_Pass_Reference (custom existence check)
    pass_ref_leaf = evaluator.add_custom_node(
        result=(isinstance(access_pass.source_url, str) and access_pass.source_url.strip().lower().startswith(("http://", "https://"))),
        id="access_pass_reference",
        desc="Provides valid reference URL supporting Access Pass information",
        parent=pass_group,
        critical=True
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
    Evaluate an answer for accessible national parks in the western US, plus Access Pass info.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root parallel aggregation
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

    # Extract parks and pass info
    extraction = await evaluator.extract(
        prompt=prompt_extract_parks_and_pass(),
        template_class=ParksAndPassExtraction,
        extraction_name="parks_and_pass"
    )

    # Normalize to exactly two parks (pad with empty ParkInfo if needed)
    parks: List[ParkInfo] = (extraction.parks or [])
    if len(parks) < 2:
        parks = parks + [ParkInfo() for _ in range(2 - len(parks))]
    parks = parks[:2]

    # Record western states list used in judging
    evaluator.add_custom_info(
        info={"western_us_states": WESTERN_US_STATES},
        info_type="criteria",
        info_name="western_states_criteria"
    )

    # Build Park Selection node (non-critical to allow partial credit if one park fails)
    park_selection = evaluator.add_parallel(
        id="park_selection",
        desc="Identify two national parks in the western United States that each meet all accessibility requirements",
        parent=root,
        critical=False
    )

    # Verify each park
    for idx, park in enumerate(parks[:2]):
        await verify_park(evaluator, park_selection, park, idx)

    # Access Pass Information (critical)
    await verify_access_pass(evaluator, root, extraction.access_pass or AccessPassInfo())

    # Return structured summary
    return evaluator.get_summary()