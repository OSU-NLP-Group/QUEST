import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "oc_rv_state_park"
TASK_DESCRIPTION = (
    "I am planning an RV camping trip to Orange County, California and need to find a California State Park that "
    "meets all of the following specific requirements:\n\n"
    "1. The park must be located in Orange County along the Southern California coast\n"
    "2. The park must be an official California State Park facility (not a county park or private campground)\n"
    "3. The park must provide direct beach access for visitors\n"
    "4. The park must accept the California Explorer Annual Pass ($195) for day-use vehicle entry\n"
    "5. The park must have RV campsites available with electrical hookups\n"
    "6. The park must have RV campsites available with water hookups\n"
    "7. The park must accommodate RVs that are at least 35 feet in length\n"
    "8. The park must provide restroom facilities for campers\n"
    "9. The park must provide shower facilities for campers (coin-operated showers are acceptable)\n"
    "10. The park must allow dogs in the campground area\n"
    "11. The park must prohibit dogs on the beach\n"
    "12. The park must require dogs to be kept on leashes no longer than 6 feet\n"
    "13. The park must use the ReserveCalifornia.com online reservation system for booking campsites\n"
    "14. The park must allow campsite reservations to be made up to 6 months in advance\n\n"
    "Identify which California State Park in Orange County meets all of these requirements. Provide the park name and "
    "include reference URLs that support each requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ParkSelection(BaseModel):
    """Extracted park selection and requirement-specific supporting URLs from the answer."""
    park_name: Optional[str] = None

    orange_county_coastal_location_urls: List[str] = Field(default_factory=list)
    california_state_park_facility_urls: List[str] = Field(default_factory=list)
    beach_access_urls: List[str] = Field(default_factory=list)
    california_explorer_pass_acceptance_urls: List[str] = Field(default_factory=list)
    rv_electrical_hookups_urls: List[str] = Field(default_factory=list)
    rv_water_hookups_urls: List[str] = Field(default_factory=list)
    rv_length_accommodation_urls: List[str] = Field(default_factory=list)
    restroom_facilities_urls: List[str] = Field(default_factory=list)
    shower_facilities_urls: List[str] = Field(default_factory=list)
    dogs_allowed_campground_urls: List[str] = Field(default_factory=list)
    dogs_prohibited_beach_urls: List[str] = Field(default_factory=list)
    leash_requirement_urls: List[str] = Field(default_factory=list)
    reservecalifornia_system_urls: List[str] = Field(default_factory=list)
    six_month_advance_booking_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_selection() -> str:
    return (
        "Extract the single California State Park chosen in the answer and the specific reference URLs that support "
        "each listed requirement. Only return URLs explicitly present in the answer text (including markdown links). "
        "If the answer omits URLs for a requirement, return an empty list for that requirement.\n\n"
        "Fields to extract:\n"
        "1) park_name: The name of the selected California State Park in Orange County.\n\n"
        "For each of the following, return an array of URLs explicitly cited in the answer that support the requirement:\n"
        "- orange_county_coastal_location_urls: URLs supporting that the park is located in Orange County on the Southern California coast.\n"
        "- california_state_park_facility_urls: URLs supporting that the site is an official California State Park facility.\n"
        "- beach_access_urls: URLs supporting that the park provides direct beach access.\n"
        "- california_explorer_pass_acceptance_urls: URLs supporting that the California Explorer Annual Pass is accepted for day-use vehicle entry.\n"
        "- rv_electrical_hookups_urls: URLs supporting that RV campsites with electrical hookups are available.\n"
        "- rv_water_hookups_urls: URLs supporting that RV campsites with water hookups are available.\n"
        "- rv_length_accommodation_urls: URLs supporting that RVs at least 35 feet in length are accommodated (e.g., max length >= 35 ft).\n"
        "- restroom_facilities_urls: URLs supporting that restroom facilities are provided for campers.\n"
        "- shower_facilities_urls: URLs supporting that shower facilities are provided (coin-operated showers acceptable).\n"
        "- dogs_allowed_campground_urls: URLs supporting that dogs are allowed in the campground area.\n"
        "- dogs_prohibited_beach_urls: URLs supporting that dogs are prohibited on the beach.\n"
        "- leash_requirement_urls: URLs supporting that dogs must be on a leash no longer than 6 feet.\n"
        "- reservecalifornia_system_urls: URLs supporting that campsite reservations use the ReserveCalifornia.com system.\n"
        "- six_month_advance_booking_urls: URLs supporting that campsite reservations can be made up to 6 months in advance.\n\n"
        "Return JSON with exactly these fields. Use full URLs (including protocol). If a URL is missing protocol, prepend http://."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(urls: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize sources: None if empty; otherwise return list."""
    if not urls:
        return None
    # Filter blatantly invalid entries
    cleaned = [u for u in urls if isinstance(u, str) and len(u.strip()) > 3]
    return cleaned or None


def _park_or_placeholder(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else "the identified park"


def _build_claims(selection: ParkSelection) -> List[Tuple[str, Optional[List[str]], str, str]]:
    """
    Build (claim, sources, node_id, additional_instruction) for each rubric leaf.
    """
    park = _park_or_placeholder(selection.park_name)
    claims: List[Tuple[str, Optional[List[str]], str, str]] = []

    # 1. Orange County coastal location
    claims.append((
        f"The park {park} is located in Orange County, California along the Southern California coast.",
        _safe_sources(selection.orange_county_coastal_location_urls),
        "orange_county_coastal_location",
        "Confirm the park lies within Orange County and is coastal (on/adjacent to the Pacific Ocean). "
        "Accept pages that clearly show the park's location within Orange County or a coastal address."
    ))

    # 2. Official California State Park facility
    claims.append((
        f"The park {park} is an official California State Park facility operated by California State Parks.",
        _safe_sources(selection.california_state_park_facility_urls),
        "california_state_park_facility",
        "Look for indicators like 'California State Parks', 'parks.ca.gov', or official unit pages."
    ))

    # 3. Direct beach access
    claims.append((
        f"The park {park} provides direct beach access for visitors.",
        _safe_sources(selection.beach_access_urls),
        "beach_access",
        "Evidence may include statements like 'beach access', 'walk-in access to beach', or maps showing direct access."
    ))

    # 4. Explorer Pass acceptance
    claims.append((
        f"The park {park} accepts the California Explorer Annual Pass for day-use vehicle entry.",
        _safe_sources(selection.california_explorer_pass_acceptance_urls),
        "california_explorer_pass_acceptance",
        "Pass acceptance language should be explicit; statewide policy pages are acceptable if they clearly apply to this park unit."
    ))

    # 5. RV electrical hookups
    claims.append((
        f"The park {park} offers RV campsites with electrical hookups.",
        _safe_sources(selection.rv_electrical_hookups_urls),
        "rv_electrical_hookups",
        "Accept wording like 'electric', 'electrical', 'hookups', or specific amperage (e.g., 30A/50A)."
    ))

    # 6. RV water hookups
    claims.append((
        f"The park {park} offers RV campsites with water hookups.",
        _safe_sources(selection.rv_water_hookups_urls),
        "rv_water_hookups",
        "Accept wording like 'water hook-ups', 'water connections', or site amenities listings."
    ))

    # 7. RV length accommodation (>= 35 ft)
    claims.append((
        f"The park {park} can accommodate RVs that are at least 35 feet in length.",
        _safe_sources(selection.rv_length_accommodation_urls),
        "rv_length_accommodation",
        "Accept evidence of max RV length >= 35 ft or explicit statements indicating 35 ft or more allowed."
    ))

    # 8. Restroom facilities
    claims.append((
        f"The park {park} provides restroom facilities for campers.",
        _safe_sources(selection.restroom_facilities_urls),
        "restroom_facilities",
        "Look for amenities sections, campground descriptions, or maps indicating restroom availability."
    ))

    # 9. Shower facilities
    claims.append((
        f"The park {park} provides shower facilities for campers.",
        _safe_sources(selection.shower_facilities_urls),
        "shower_facilities",
        "Coin-operated showers are acceptable; accept any clear mention of showers in the campground."
    ))

    # 10. Dogs allowed in campground
    claims.append((
        f"Dogs are allowed in the campground area at {park}.",
        _safe_sources(selection.dogs_allowed_campground_urls),
        "dogs_allowed_campground",
        "Pet policy pages or unit regulations should explicitly allow dogs in the campground."
    ))

    # 11. Dogs prohibited on the beach
    claims.append((
        f"Dogs are prohibited on the beach at {park}.",
        _safe_sources(selection.dogs_prohibited_beach_urls),
        "dogs_prohibited_beach",
        "Pet policy or unit regulations should explicitly prohibit dogs on the beach."
    ))

    # 12. Leash requirement (<= 6 feet)
    claims.append((
        f"Dogs at {park} must be kept on a leash no longer than 6 feet.",
        _safe_sources(selection.leash_requirement_urls),
        "leash_requirement",
        "Accept standard state park leash policy language requiring leashes of 6 feet or less."
    ))

    # 13. ReserveCalifornia system
    claims.append((
        f"Campsite reservations for {park} are made via ReserveCalifornia.com.",
        _safe_sources(selection.reservecalifornia_system_urls),
        "reservecalifornia_system",
        "Evidence may include ReserveCalifornia unit pages for the park or official references indicating ReserveCalifornia is used."
    ))

    # 14. Six-month advance booking window
    claims.append((
        f"Campsite reservations for {park} can be made up to 6 months in advance.",
        _safe_sources(selection.six_month_advance_booking_urls),
        "six_month_advance_booking",
        "Accept official ReserveCalifornia policy pages or unit references that clearly state a 6-month advance booking window."
    ))

    return claims


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify(evaluator: Evaluator, root_node, selection: ParkSelection) -> None:
    """
    Build all leaf nodes as critical requirements and run verifications using cited URLs.
    """
    claims = _build_claims(selection)

    # Create leaf nodes for each requirement and prepare batch verification entries
    batch_items: List[Tuple[str, Optional[List[str]], Any, Optional[str]]] = []
    for claim_text, sources, node_id, add_ins in claims:
        leaf = evaluator.add_leaf(
            id=node_id,
            desc=claim_text,  # Leaf description is the claim itself for traceability
            parent=root_node,
            critical=True,
        )
        batch_items.append((claim_text, sources, leaf, add_ins))

    # Run all checks in parallel
    await evaluator.batch_verify(batch_items)


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
    Evaluate the answer for the Orange County California State Park RV camping requirements task.
    """
    # Initialize evaluator with parallel root aggregation
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="California State Park in Orange County meeting all RV camping and pet policy requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured selection & sources from the answer
    selection = await evaluator.extract(
        prompt=prompt_extract_park_selection(),
        template_class=ParkSelection,
        extraction_name="park_selection",
    )

    # Add custom info (optional) for transparency
    evaluator.add_custom_info(
        info={"park_name": selection.park_name or None},
        info_type="extraction_meta",
        info_name="selected_park"
    )

    # Build verification tree and run checks
    await build_and_verify(evaluator, root, selection)

    # Return summary
    return evaluator.get_summary()