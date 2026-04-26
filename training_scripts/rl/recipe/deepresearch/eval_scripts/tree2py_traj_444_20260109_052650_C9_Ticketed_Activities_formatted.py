import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "la_venues_selection"
TASK_DESCRIPTION = """
Identify four performing arts venues in Louisiana suitable for hosting a professional touring theatrical production series, where each venue must meet all of the following requirements:

Location Requirements:
- Each venue must be located in Louisiana
- The four venues must be in four different cities within Louisiana to maximize regional coverage

Capacity Requirements:
- Each venue must have a minimum seating capacity of 1,000 seats
- Provide the specific seating capacity number for each venue

Technical Infrastructure Requirements:
- Each venue must have stage specifications suitable for theatrical productions
- Each venue must have backstage facilities including dressing rooms and/or green rooms
- Confirm technical systems capabilities (sound and lighting systems) where information is available

Accessibility Requirements:
- Each venue must meet ADA compliance requirements
- Each venue must provide wheelchair accessible seating

Amenities Requirements:
- Each venue must have parking facilities available for patrons
- Identify concession or hospitality areas where available

For each venue, provide the venue name, city location, seating capacity, and reference URLs that verify the venue meets these specifications.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """One venue entry extracted from the agent's answer."""
    name: Optional[str] = None
    city: Optional[str] = None
    seating_capacity: Optional[str] = None  # Prefer strings to allow ranges or approximations
    stage_spec: Optional[str] = None
    backstage_facilities: Optional[str] = None
    tech_systems: Optional[str] = None
    ada_compliance: Optional[str] = None
    wheelchair_access: Optional[str] = None
    parking: Optional[str] = None
    concessions: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    """Top-level extraction of up to 4 venues."""
    venues: List[VenueItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract up to four performing arts venues listed in the answer (filter to the first four if more than four are present).
    For each venue, return a JSON object with the following fields (use null if a field is missing in the answer):
    - name: The venue name as stated.
    - city: The city in Louisiana where the venue is located (as stated).
    - seating_capacity: The seating capacity number mentioned in the answer. If given as a range, include the range string (e.g., "1200-1300").
    - stage_spec: Any mention of stage specifications suitable for theatrical productions (e.g., stage dimensions, proscenium stage).
    - backstage_facilities: Any mention of backstage facilities like dressing rooms and/or green rooms.
    - tech_systems: Any mention of sound and/or lighting systems capabilities (if available).
    - ada_compliance: Any statement indicating ADA compliance (if available).
    - wheelchair_access: Any statement indicating wheelchair accessible seating (if available).
    - parking: Any statement indicating parking availability (if available).
    - concessions: Any statement indicating concessions or hospitality areas (if available).
    - reference_urls: A list of URLs explicitly provided in the answer that substantiate or describe the venue and its specifications. Extract only valid URLs (including those in markdown). Do not invent URLs.

    Notes:
    - Only extract information explicitly present in the answer.
    - For URLs, use only those explicitly provided; include full URLs with protocol (prepend http:// if missing).
    - If a field is not mentioned for a venue, set it to null (or [] for reference_urls).
    - Return the JSON object { "venues": [ ... up to 4 venue objects ... ] }.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _has_digits(s: Optional[str]) -> bool:
    return any(ch.isdigit() for ch in (s or ""))


def _normalize_city(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _get_sources(venue: VenueItem) -> List[str]:
    return [u for u in (venue.reference_urls or []) if _nonempty(u)]


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_venue(
    evaluator: Evaluator,
    parent_node,
    venue: VenueItem,
    idx: int
) -> None:
    """
    Build verification sub-tree and run checks for one venue.
    """
    venue_num = idx + 1
    v_node = evaluator.add_parallel(
        id=f"Venue_{venue_num}",
        desc=f"{['First','Second','Third','Fourth'][idx]} performing arts venue entry.",
        parent=parent_node,
        critical=False
    )

    sources = _get_sources(venue)
    venue_name = venue.name or "the venue"
    city_name = venue.city or ""

    # 0. Reference URLs existence (critical precondition for evidence-based checks)
    evaluator.add_custom_node(
        result=len(sources) > 0,
        id=f"V{venue_num}_Reference_URLs",
        desc="Reference URL(s) are provided that verify the venue meets the stated specifications (sufficient to support the required claims).",
        parent=v_node,
        critical=True
    )

    # 1. Name provided (critical)
    evaluator.add_custom_node(
        result=_nonempty(venue.name),
        id=f"V{venue_num}_Name_Provided",
        desc="Venue name is provided.",
        parent=v_node,
        critical=True
    )

    # 2. Location: city provided + verify Louisiana location (critical)
    evaluator.add_custom_node(
        result=_nonempty(venue.city),
        id=f"V{venue_num}_City_Provided",
        desc="Venue city value is provided in the answer.",
        parent=v_node,
        critical=True
    )

    v_loc_leaf = evaluator.add_leaf(
        id=f"V{venue_num}_Location_Louisiana_City",
        desc="Venue city is provided and the venue is located in Louisiana.",
        parent=v_node,
        critical=True
    )
    loc_claim = f"The venue '{venue_name}' is located in {city_name}, Louisiana."
    await evaluator.verify(
        claim=loc_claim,
        node=v_loc_leaf,
        sources=sources,
        additional_instruction="Confirm from the provided URL(s) that the venue is in the specified Louisiana city. Accept minor naming variations (e.g., abbreviations). If the URL does not clearly place the venue in a Louisiana city, mark as not supported."
    )

    # 3. Capacity: number provided (critical) + minimum >= 1000 verified (critical)
    evaluator.add_custom_node(
        result=_nonempty(venue.seating_capacity) and _has_digits(venue.seating_capacity),
        id=f"V{venue_num}_Capacity_Number_Provided",
        desc="A specific seating capacity number is provided in the answer.",
        parent=v_node,
        critical=True
    )

    v_cap_leaf = evaluator.add_leaf(
        id=f"V{venue_num}_Capacity_Number_And_Minimum",
        desc="A specific seating capacity number is provided and it is at least 1,000 seats.",
        parent=v_node,
        critical=True
    )
    cap_claim = f"The venue '{venue_name}' has a seating capacity of at least 1,000 seats."
    await evaluator.verify(
        claim=cap_claim,
        node=v_cap_leaf,
        sources=sources,
        additional_instruction="Use the provided URL(s) to verify the seating capacity. The claim should pass if the page indicates capacity ≥ 1000. Accept reasonable variants like 'approx. 1000' or exact numbers ≥ 1000."
    )

    # 4. Stage suitability (critical)
    v_stage_leaf = evaluator.add_leaf(
        id=f"V{venue_num}_Stage_Suitable",
        desc="Stage specifications are suitable for theatrical productions.",
        parent=v_node,
        critical=True
    )
    stage_claim = f"The venue '{venue_name}' has stage specifications suitable for professional theatrical productions (e.g., proscenium stage or detailed stage dimensions)."
    await evaluator.verify(
        claim=stage_claim,
        node=v_stage_leaf,
        sources=sources,
        additional_instruction="Look for pages indicating stage specifications, stage type, dimensions, or suitability for touring theatrical shows. If the provided sources lack stage details, mark as not supported."
    )

    # 5. Backstage facilities (critical)
    v_backstage_leaf = evaluator.add_leaf(
        id=f"V{venue_num}_Backstage_Facilities",
        desc="Backstage facilities (dressing rooms and/or green rooms) are available.",
        parent=v_node,
        critical=True
    )
    backstage_claim = f"The venue '{venue_name}' offers backstage facilities such as dressing rooms and/or green rooms."
    await evaluator.verify(
        claim=backstage_claim,
        node=v_backstage_leaf,
        sources=sources,
        additional_instruction="Look for references to backstage amenities like dressing rooms, green rooms, company rooms, or similar. If not mentioned in sources, mark as not supported."
    )

    # 6. Technical systems (non-critical)
    v_tech_leaf = evaluator.add_leaf(
        id=f"V{venue_num}_Tech_Systems_If_Available",
        desc="Sound and lighting capabilities are confirmed where information is available.",
        parent=v_node,
        critical=False
    )
    tech_claim = f"The venue '{venue_name}' has sound and lighting systems suitable for professional theatrical productions."
    await evaluator.verify(
        claim=tech_claim,
        node=v_tech_leaf,
        sources=sources,
        additional_instruction="Check whether the provided pages mention sound/PA systems, mixing consoles, lighting rigs, dimmers, or similar technical specs. If no technical info is present, mark as not supported."
    )

    # 7. ADA compliance (critical)
    v_ada_leaf = evaluator.add_leaf(
        id=f"V{venue_num}_ADA_Compliance",
        desc="Venue meets ADA compliance requirements.",
        parent=v_node,
        critical=True
    )
    ada_claim = f"The venue '{venue_name}' meets ADA compliance requirements."
    await evaluator.verify(
        claim=ada_claim,
        node=v_ada_leaf,
        sources=sources,
        additional_instruction="Look for accessibility/ADA statements or policies. Accept synonyms like 'accessible', 'ADA compliant', 'accessibility features'."
    )

    # 8. Wheelchair accessible seating (critical)
    v_wheel_leaf = evaluator.add_leaf(
        id=f"V{venue_num}_Wheelchair_Seating",
        desc="Wheelchair accessible seating is available.",
        parent=v_node,
        critical=True
    )
    wheelchair_claim = f"The venue '{venue_name}' provides wheelchair accessible seating."
    await evaluator.verify(
        claim=wheelchair_claim,
        node=v_wheel_leaf,
        sources=sources,
        additional_instruction="Confirm that wheelchair seating is provided. Look for seating maps or policy pages explicitly mentioning wheelchair or accessible seating."
    )

    # 9. Parking available (critical)
    v_parking_leaf = evaluator.add_leaf(
        id=f"V{venue_num}_Parking_Available",
        desc="Parking facilities are available for patrons.",
        parent=v_node,
        critical=True
    )
    parking_claim = f"The venue '{venue_name}' has parking facilities available for patrons."
    await evaluator.verify(
        claim=parking_claim,
        node=v_parking_leaf,
        sources=sources,
        additional_instruction="Look for on-site parking, structured garage, or nearby parking arrangements officially referenced by the venue. If parking is not mentioned, mark as not supported."
    )

    # 10. Concessions/hospitality (non-critical)
    v_conc_leaf = evaluator.add_leaf(
        id=f"V{venue_num}_Concessions_If_Available",
        desc="Concession or hospitality areas are identified where available.",
        parent=v_node,
        critical=False
    )
    concessions_claim = f"The venue '{venue_name}' offers concessions or hospitality (e.g., bars, café, lobby concessions)."
    await evaluator.verify(
        claim=concessions_claim,
        node=v_conc_leaf,
        sources=sources,
        additional_instruction="Look for mentions of concessions, bars, café, or hospitality services. If none are mentioned on provided sources, mark as not supported."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Louisiana venues selection task and return a structured result.
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

    # Top-level task node (non-critical to allow partial credit across venues; critical checks are inside)
    task_node = evaluator.add_parallel(
        id="Venue_Selection_Task",
        desc="Identify four performing arts venues in Louisiana suitable for hosting a professional touring theatrical production series, meeting all stated requirements and providing verifying URLs.",
        parent=root,
        critical=False
    )

    # 1) Extract venue data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Limit/pad to exactly 4 venues
    venues = list(extracted.venues[:4])
    while len(venues) < 4:
        venues.append(VenueItem())

    # 2) Global uniqueness check for city values (critical)
    city_values = [_normalize_city(v.city) for v in venues]
    # Count distinct non-empty city names among the four
    distinct_nonempty = len({c for c in city_values if c})
    global_unique_result = (distinct_nonempty == 4)

    evaluator.add_custom_node(
        result=global_unique_result,
        id="Global_City_Uniqueness",
        desc="The four venues are in four different cities within Louisiana (all city values are distinct across Venue_1–Venue_4).",
        parent=task_node,
        critical=True
    )

    # 3) Build verification subtrees for each venue
    for i, v in enumerate(venues):
        await verify_single_venue(evaluator, task_node, v, i)

    # 4) Return unified summary
    return evaluator.get_summary()