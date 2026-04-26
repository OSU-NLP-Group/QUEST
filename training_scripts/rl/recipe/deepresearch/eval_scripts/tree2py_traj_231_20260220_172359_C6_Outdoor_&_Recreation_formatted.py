import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "accessible_maine_trip_2026"
TASK_DESCRIPTION = (
    "You are planning an accessible multi-family camping trip to Maine for summer 2026. "
    "You need to identify two specific campgrounds that meet the following requirements:\n\n"
    "Campground 1 must be a Maine state park campground that:\n"
    "- Has a three-star (Good access) overall accessibility rating from the Maine Bureau of Parks and Lands\n"
    "- Has between 200 and 300 total campsites\n"
    "- Offers at least 80 campsites with both electric and water hookups\n"
    "- Provides hot showers that are not coin-operated\n"
    "- Has flush toilets available\n"
    "- Provides beach wheelchairs for visitors\n"
    "- Accepts campground reservations starting on February 1, 2, or 3 for the 2026 camping season\n\n"
    "Campground 2 must be an Acadia National Park campground that:\n"
    "- Has at least 75 accessible campsites\n"
    "- Provides ADA-compliant picnic tables at all sites (not just accessible sites)\n"
    "- Offers electric and water hookups on at least one designated loop or section\n"
    "- Has flush toilets and potable running water\n"
    "- Features paved campground roads\n"
    "- Accepts reservations six months in advance starting December 1, 2024 or later\n\n"
    "Provide the name of each campground along with supporting reference URLs."
)

# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class Campground1(BaseModel):
    name: Optional[str] = None
    classification: Optional[str] = None  # Expected to indicate "Maine state park campground"
    identity_urls: List[str] = Field(default_factory=list)

    accessibility_rating: Optional[str] = None  # e.g., "3-star (Good access)"
    accessibility_urls: List[str] = Field(default_factory=list)
    beach_wheelchairs: Optional[str] = None  # any mention confirming availability

    total_campsites: Optional[str] = None  # number or phrase
    hookup_sites_both_electric_water: Optional[str] = None  # number or phrase
    capacity_urls: List[str] = Field(default_factory=list)

    flush_toilets: Optional[str] = None  # mention confirming availability
    hot_showers_non_coin: Optional[str] = None  # mention confirming "not coin-operated"
    facilities_urls: List[str] = Field(default_factory=list)

    reservation_open_date_2026: Optional[str] = None  # date text or phrase
    reservation_urls: List[str] = Field(default_factory=list)


class Campground2(BaseModel):
    name: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)  # confirms Acadia NP location

    accessible_campsites_count: Optional[str] = None  # number or phrase (>= 75)
    ada_tables_all_sites: Optional[str] = None  # explicit confirmation "at all sites"
    accessibility_urls: List[str] = Field(default_factory=list)

    hookup_availability: Optional[str] = None  # mentions electric+water hookups on a loop/section
    paved_roads: Optional[str] = None  # mentions paved roads
    infrastructure_urls: List[str] = Field(default_factory=list)

    flush_toilets: Optional[str] = None  # mentions flush toilets
    potable_running_water: Optional[str] = None  # mentions potable running water
    facilities_urls: List[str] = Field(default_factory=list)

    six_month_advance_policy: Optional[str] = None  # mentions six-month advance policy and start date
    reservation_urls: List[str] = Field(default_factory=list)


class CampgroundsExtraction(BaseModel):
    campground_1: Optional[Campground1] = None
    campground_2: Optional[Campground2] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract exactly two campgrounds described in the answer and return a JSON object with fields "campground_1" and "campground_2".
    For each campground, extract the following fields from the answer text exactly as written (do not infer):

    For campground_1 (Maine state park campground):
    - name: The campground's name
    - classification: Any phrase indicating it is a Maine state park campground (e.g., "Maine state park campground")
    - identity_urls: All URLs that confirm the campground identity and Maine state park status
    - accessibility_rating: The stated overall accessibility rating (e.g., "three-star (Good access)")
    - accessibility_urls: All URLs that directly confirm the accessibility rating and beach wheelchair availability
    - beach_wheelchairs: Any phrase indicating beach wheelchairs are provided
    - total_campsites: The total campsite count (string as written)
    - hookup_sites_both_electric_water: The count or phrase indicating the number of campsites with both electric and water hookups
    - capacity_urls: All URLs that confirm total campsite count and hookup availability
    - flush_toilets: Any phrase confirming flush toilets availability
    - hot_showers_non_coin: Any phrase confirming hot showers are provided and NOT coin-operated
    - facilities_urls: All URLs that confirm flush toilets and non-coin-operated hot showers
    - reservation_open_date_2026: The stated reservation opening date for 2026 (e.g., "February 1, 2026")
    - reservation_urls: All URLs that confirm the reservation opening date for the 2026 season

    For campground_2 (Acadia National Park campground):
    - name: The campground's name
    - location_urls: All URLs that confirm the campground is located in Acadia National Park
    - accessible_campsites_count: The count or phrase indicating at least 75 accessible campsites
    - ada_tables_all_sites: Any phrase confirming ADA-compliant picnic tables are at all sites (not just accessible sites)
    - accessibility_urls: All URLs that confirm accessible campsite count and ADA table coverage
    - hookup_availability: Any phrase indicating electric and water hookups are available on at least one loop/section
    - paved_roads: Any phrase confirming the campground roads are paved
    - infrastructure_urls: All URLs that confirm hookups availability and paved roads
    - flush_toilets: Any phrase confirming flush toilets are available
    - potable_running_water: Any phrase confirming potable running water is available
    - facilities_urls: All URLs that confirm flush toilets and potable running water
    - six_month_advance_policy: Any phrase confirming reservations are accepted six months in advance starting December 1, 2024 or later
    - reservation_urls: All URLs that confirm the six-month advance reservation policy and effective start date

    Rules:
    - Extract only URLs explicitly present in the answer. If a requested URL is missing, return an empty array for that field.
    - If a requested field is not mentioned, return null for that field.
    - Preserve the original text for counts/dates/phrases (do not normalize to numbers).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(items: Optional[List[str]]) -> List[str]:
    return [u for u in (items or []) if isinstance(u, str) and u.strip()]


def _combine_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in url_lists:
        for u in lst:
            if u and u not in seen:
                seen.add(u)
                combined.append(u)
    return combined


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_campground_1(evaluator: Evaluator, parent_node, cg: Campground1) -> None:
    cg = cg or Campground1()

    # Parent node for Campground 1 (critical: if it fails, the overall should fail)
    cg1_node = evaluator.add_parallel(
        id="campground_1",
        desc="Identify a Maine state park campground meeting all specified requirements",
        parent=parent_node,
        critical=True
    )

    # 1) Basic identification & classification
    basic_node = evaluator.add_parallel(
        id="campground_1_basic_identification",
        desc="Basic campground identification and classification requirements",
        parent=cg1_node,
        critical=True
    )

    # Existence of identity URL(s)
    evaluator.add_custom_node(
        result=len(_safe_list(cg.identity_urls)) > 0,
        id="campground_1_identification_url",
        desc="Provide reference URL confirming campground identity and state park status",
        parent=basic_node,
        critical=True
    )

    # Verify Maine state park status
    leaf_state_park = evaluator.add_leaf(
        id="campground_1_maine_state_park",
        desc="The campground is a Maine state park campground",
        parent=basic_node,
        critical=True
    )
    claim_state_park = f"The campground named '{(cg.name or '').strip()}' is a Maine state park campground."
    await evaluator.verify(
        claim=claim_state_park,
        node=leaf_state_park,
        sources=_safe_list(cg.identity_urls),
        additional_instruction="Confirm the page explicitly indicates the campground is part of the Maine State Park system (Maine Bureau of Parks and Lands). Accept official Maine.gov/BPL pages or authoritative references that explicitly state 'Maine State Park'."
    )

    # 2) Accessibility rating & beach wheelchairs
    access_node = evaluator.add_parallel(
        id="campground_1_accessibility_features",
        desc="Accessibility rating and wheelchair access features",
        parent=cg1_node,
        critical=True
    )

    # Existence of accessibility URL(s)
    evaluator.add_custom_node(
        result=len(_safe_list(cg.accessibility_urls)) > 0,
        id="campground_1_accessibility_url",
        desc="Provide reference URL confirming accessibility rating and beach wheelchair availability",
        parent=access_node,
        critical=True
    )

    # Verify three-star rating
    leaf_three_star = evaluator.add_leaf(
        id="campground_1_three_star_rating",
        desc="Has a three-star (Good access) overall accessibility rating from Maine Bureau of Parks and Lands",
        parent=access_node,
        critical=True
    )
    claim_three_star = "This campground has a three-star ('Good access') overall accessibility rating from the Maine Bureau of Parks and Lands."
    await evaluator.verify(
        claim=claim_three_star,
        node=leaf_three_star,
        sources=_safe_list(cg.accessibility_urls),
        additional_instruction="Look for explicit text indicating 'three-star' or 'Good access' accessibility rating under Maine BPL's rating system."
    )

    # Verify beach wheelchairs
    leaf_beach_wc = evaluator.add_leaf(
        id="campground_1_beach_wheelchairs",
        desc="Provides beach wheelchairs for visitors",
        parent=access_node,
        critical=True
    )
    claim_beach_wc = "The campground provides beach wheelchairs for visitors."
    await evaluator.verify(
        claim=claim_beach_wc,
        node=leaf_beach_wc,
        sources=_combine_urls(_safe_list(cg.accessibility_urls), _safe_list(cg.identity_urls)),
        additional_instruction="Confirm that beach wheelchairs are available; allow variants like 'beach accessibility wheelchair'."
    )

    # 3) Capacity & hookups
    capacity_node = evaluator.add_parallel(
        id="campground_1_capacity_infrastructure",
        desc="Campground capacity and hookup infrastructure",
        parent=cg1_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(cg.capacity_urls)) > 0,
        id="campground_1_capacity_url",
        desc="Provide reference URL confirming total campsite count and hookup availability",
        parent=capacity_node,
        critical=True
    )

    leaf_total = evaluator.add_leaf(
        id="campground_1_total_capacity",
        desc="Has between 200 and 300 total campsites",
        parent=capacity_node,
        critical=True
    )
    claim_total = "This campground has between 200 and 300 total campsites."
    await evaluator.verify(
        claim=claim_total,
        node=leaf_total,
        sources=_safe_list(cg.capacity_urls),
        additional_instruction="Confirm the page lists a total campsite count within [200, 300]. If it lists an exact number like 250, that satisfies the claim."
    )

    leaf_hookups = evaluator.add_leaf(
        id="campground_1_hookup_sites",
        desc="Offers at least 80 campsites with both electric and water hookups",
        parent=capacity_node,
        critical=True
    )
    claim_hookups = "This campground offers at least 80 campsites with both electric and water hookups."
    await evaluator.verify(
        claim=claim_hookups,
        node=leaf_hookups,
        sources=_safe_list(cg.capacity_urls),
        additional_instruction="Look for language indicating the count of sites that have both electric and water hookups is ≥ 80."
    )

    # 4) Restroom & showers
    facilities_node_1 = evaluator.add_parallel(
        id="campground_1_restroom_facilities",
        desc="Bathroom and shower facilities meeting requirements",
        parent=cg1_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(cg.facilities_urls)) > 0,
        id="campground_1_facilities_url",
        desc="Provide reference URL confirming flush toilets and non-coin-operated hot showers",
        parent=facilities_node_1,
        critical=True
    )

    leaf_flush = evaluator.add_leaf(
        id="campground_1_flush_toilets",
        desc="Has flush toilets available",
        parent=facilities_node_1,
        critical=True
    )
    claim_flush = "The campground has flush toilets available."
    await evaluator.verify(
        claim=claim_flush,
        node=leaf_flush,
        sources=_safe_list(cg.facilities_urls),
        additional_instruction="Confirm the presence of flush toilets (not vault/pit-only)."
    )

    leaf_showers = evaluator.add_leaf(
        id="campground_1_hot_showers",
        desc="Provides hot showers that are not coin-operated",
        parent=facilities_node_1,
        critical=True
    )
    claim_showers = "The campground provides hot showers that are not coin-operated."
    await evaluator.verify(
        claim=claim_showers,
        node=leaf_showers,
        sources=_safe_list(cg.facilities_urls),
        additional_instruction="Confirm hot showers are available and explicitly not coin-operated. If it states coin-operated, this should fail."
    )

    # 5) Reservations
    reserve_node_1 = evaluator.add_parallel(
        id="campground_1_reservations",
        desc="Reservation system and opening dates",
        parent=cg1_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(cg.reservation_urls)) > 0,
        id="campground_1_reservation_url",
        desc="Provide reference URL confirming reservation opening date for 2026",
        parent=reserve_node_1,
        critical=True
    )

    leaf_res_open = evaluator.add_leaf(
        id="campground_1_reservation_dates",
        desc="Accepts campground reservations starting on February 1, 2, or 3 for the 2026 camping season",
        parent=reserve_node_1,
        critical=True
    )
    claim_res_open = "For the 2026 camping season, reservations open on February 1, 2, or 3."
    await evaluator.verify(
        claim=claim_res_open,
        node=leaf_res_open,
        sources=_safe_list(cg.reservation_urls),
        additional_instruction="Confirm that the official reservation policy page indicates the opening date for 2026 is one of Feb 1, Feb 2, or Feb 3."
    )


async def verify_campground_2(evaluator: Evaluator, parent_node, cg: Campground2) -> None:
    cg = cg or Campground2()

    # Parent node for Campground 2 (critical)
    cg2_node = evaluator.add_parallel(
        id="campground_2",
        desc="Identify an Acadia National Park campground meeting all specified requirements",
        parent=parent_node,
        critical=True
    )

    # 1) Location & identification
    loc_node = evaluator.add_parallel(
        id="campground_2_location_identification",
        desc="Location and basic identification requirements",
        parent=cg2_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(cg.location_urls)) > 0,
        id="campground_2_location_url",
        desc="Provide reference URL confirming the campground is in Acadia National Park",
        parent=loc_node,
        critical=True
    )

    leaf_in_acadia = evaluator.add_leaf(
        id="campground_2_acadia_location",
        desc="The campground is located in Acadia National Park",
        parent=loc_node,
        critical=True
    )
    claim_in_acadia = f"The campground named '{(cg.name or '').strip()}' is located within Acadia National Park."
    await evaluator.verify(
        claim=claim_in_acadia,
        node=leaf_in_acadia,
        sources=_safe_list(cg.location_urls),
        additional_instruction="Confirm the page explicitly indicates the campground is within Acadia National Park. Official NPS or Recreation.gov pages preferred."
    )

    # 2) Accessibility features
    access_node_2 = evaluator.add_parallel(
        id="campground_2_accessibility_features",
        desc="Accessible sites and ADA-compliant features",
        parent=cg2_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(cg.accessibility_urls)) > 0,
        id="campground_2_accessibility_url",
        desc="Provide reference URL confirming accessible campsite count and ADA table coverage",
        parent=access_node_2,
        critical=True
    )

    leaf_access_count = evaluator.add_leaf(
        id="campground_2_accessible_count",
        desc="Has at least 75 accessible campsites",
        parent=access_node_2,
        critical=True
    )
    claim_access_count = "This campground has at least 75 accessible campsites."
    await evaluator.verify(
        claim=claim_access_count,
        node=leaf_access_count,
        sources=_safe_list(cg.accessibility_urls),
        additional_instruction="Confirm the accessible campsite count is ≥ 75; allow evidence stating a number over 75 or explicit 'at least 75'."
    )

    leaf_ada_tables = evaluator.add_leaf(
        id="campground_2_ada_tables_all_sites",
        desc="Provides ADA-compliant picnic tables at all sites (not just designated accessible sites)",
        parent=access_node_2,
        critical=True
    )
    claim_ada_tables = "ADA-compliant picnic tables are provided at all campsites, not only designated accessible sites."
    await evaluator.verify(
        claim=claim_ada_tables,
        node=leaf_ada_tables,
        sources=_safe_list(cg.accessibility_urls),
        additional_instruction="Look for explicit confirmation that ADA-compliant picnic tables are at all sites, not only accessible sites."
    )

    # 3) Infrastructure (hookups & roads)
    infra_node = evaluator.add_parallel(
        id="campground_2_infrastructure",
        desc="Hookup infrastructure and road conditions",
        parent=cg2_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(cg.infrastructure_urls)) > 0,
        id="campground_2_infrastructure_url",
        desc="Provide reference URL confirming hookup availability and paved roads",
        parent=infra_node,
        critical=True
    )

    leaf_hookups_2 = evaluator.add_leaf(
        id="campground_2_hookup_availability",
        desc="Offers electric and water hookups on at least one designated loop or section",
        parent=infra_node,
        critical=True
    )
    claim_hookups_2 = "Electric and water hookups are available on at least one designated loop or section of the campground."
    await evaluator.verify(
        claim=claim_hookups_2,
        node=leaf_hookups_2,
        sources=_safe_list(cg.infrastructure_urls),
        additional_instruction="Confirm the presence of both electric and water hookups on a loop/section; allow evidence that a specific loop offers them."
    )

    leaf_paved = evaluator.add_leaf(
        id="campground_2_paved_roads",
        desc="Features paved campground roads",
        parent=infra_node,
        critical=True
    )
    claim_paved = "The campground features paved roads."
    await evaluator.verify(
        claim=claim_paved,
        node=leaf_paved,
        sources=_safe_list(cg.infrastructure_urls),
        additional_instruction="Confirm that roads within the campground are paved; allow synonyms like 'asphalt' or 'blacktop'."
    )

    # 4) Facilities (toilets & water)
    facilities_node_2 = evaluator.add_parallel(
        id="campground_2_facilities",
        desc="Restroom and water facilities",
        parent=cg2_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(cg.facilities_urls)) > 0,
        id="campground_2_facilities_url",
        desc="Provide reference URL confirming flush toilets and potable running water",
        parent=facilities_node_2,
        critical=True
    )

    leaf_flush_2 = evaluator.add_leaf(
        id="campground_2_flush_toilets",
        desc="Has flush toilets available",
        parent=facilities_node_2,
        critical=True
    )
    claim_flush_2 = "The campground has flush toilets available."
    await evaluator.verify(
        claim=claim_flush_2,
        node=leaf_flush_2,
        sources=_safe_list(cg.facilities_urls),
        additional_instruction="Confirm flush toilets are available (not vault-only)."
    )

    leaf_potable = evaluator.add_leaf(
        id="campground_2_potable_water",
        desc="Has potable running water available",
        parent=facilities_node_2,
        critical=True
    )
    claim_potable = "The campground has potable running water available."
    await evaluator.verify(
        claim=claim_potable,
        node=leaf_potable,
        sources=_safe_list(cg.facilities_urls),
        additional_instruction="Confirm potable running water is available (drinkable water at taps)."
    )

    # 5) Reservations policy
    reserve_node_2 = evaluator.add_parallel(
        id="campground_2_reservations",
        desc="Reservation policy and advance booking requirements",
        parent=cg2_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=len(_safe_list(cg.reservation_urls)) > 0,
        id="campground_2_reservation_url",
        desc="Provide reference URL confirming six-month advance reservation policy effective December 1, 2024 or later",
        parent=reserve_node_2,
        critical=True
    )

    leaf_six_month = evaluator.add_leaf(
        id="campground_2_six_month_advance",
        desc="Accepts reservations six months in advance starting December 1, 2024 or later",
        parent=reserve_node_2,
        critical=True
    )
    claim_six_month = "Reservations are accepted six months in advance, effective December 1, 2024 or later."
    await evaluator.verify(
        claim=claim_six_month,
        node=leaf_six_month,
        sources=_safe_list(cg.reservation_urls),
        additional_instruction="Confirm an official policy indicates a six-month advance reservation window and that this policy starts no earlier than Dec 1, 2024."
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
    Evaluate the answer for the accessible Maine camping trip task.
    Builds a verification tree aligned with the rubric and returns a structured summary.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Two campgrounds are independent checks
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify two campgrounds in Maine that meet specific accessibility and amenity requirements for an accessible multi-family camping trip",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundsExtraction,
        extraction_name="campgrounds_extraction"
    )

    cg1 = extracted.campground_1 or Campground1()
    cg2 = extracted.campground_2 or Campground2()

    # Build verification subtrees
    await verify_campground_1(evaluator, root, cg1)
    await verify_campground_2(evaluator, root, cg2)

    # Return consolidated summary
    return evaluator.get_summary()