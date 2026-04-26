import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "montauk_rv_march_2026"
TASK_DESCRIPTION = (
    "Identify a state or county park campground in the Montauk, New York area that is suitable for a family RV camping "
    "trip during March 2026. The campground must meet ALL of the following requirements: "
    "(1) Have at least 100 campsites available, "
    "(2) Offer electrical hookups at campsites, "
    "(3) Provide on-site restroom facilities, "
    "(4) Provide on-site shower facilities, "
    "(5) Have hiking trails accessible within or from the park, "
    "(6) Provide access to ocean beaches, "
    "(7) Accept advance reservations (not first-come-first-served only), and "
    "(8) Be open and accepting reservations for March 2026. "
    "For the identified campground, provide: the official name of the park/campground, the total number of campsites, "
    "confirmation of electrical hookups availability, the reservation website URL or phone number, at least one hiking "
    "trail name or description available at or near the park, confirmation of ocean beach access, and supporting "
    "reference URLs from official sources for each claim."
)

TARGET_MONTH = "March 2026"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundExtraction(BaseModel):
    # Identity
    campground_name: Optional[str] = None
    location_description: Optional[str] = None  # e.g., "Montauk, NY", "Hither Hills State Park, Montauk"
    official_website_url: Optional[str] = None  # The official park/campground URL (prefer .gov or official org)

    # Capacity & hookups
    total_campsites: Optional[str] = None  # Keep as string to be flexible ("168", "over 190")
    electrical_hookups_available: Optional[str] = None  # e.g., "Yes, 30/50 amp at sites", "Some sites have electric"
    capacity_reference_urls: List[str] = Field(default_factory=list)  # pages confirming capacity/hookups

    # On-site facilities
    restrooms_available: Optional[str] = None  # e.g., "Yes", "Comfort stations"
    showers_available: Optional[str] = None    # e.g., "Yes"
    facilities_reference_urls: List[str] = Field(default_factory=list)

    # Trails
    trails_available: Optional[str] = None  # e.g., "Yes, hiking trails in the park"
    trail_name_or_description: Optional[str] = None  # e.g., "Hither Hills Nature Trail"
    trail_reference_urls: List[str] = Field(default_factory=list)

    # Beach
    beach_access_available: Optional[str] = None  # e.g., "Yes, ocean beach access"
    beach_reference_urls: List[str] = Field(default_factory=list)

    # Reservations & March 2026
    advance_reservations_accepted: Optional[str] = None  # e.g., "Yes, via ReserveAmerica / NY State Parks"
    reservation_url_or_phone: Optional[str] = None  # URL or phone provided in the answer
    open_in_march_2026: Optional[str] = None  # e.g., "Open year-round", "Open in March 2026"
    march_2026_reservations_accepted: Optional[str] = None  # e.g., "Yes"
    reservation_reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campground() -> str:
    return """
Extract exactly one campground (the main one the answer recommends) that is a state or county park in or around Montauk, New York, and return the following fields. If multiple are mentioned, pick the first one that is actually recommended as suitable.

Return a JSON object with these fields (use null for missing values; strings for all non-list fields):

Identity
- campground_name: the official name as written in the answer
- location_description: a short phrase for location (e.g., "Montauk, NY" or "Hither Hills State Park, Montauk")
- official_website_url: the official park or campground webpage URL (prefer .gov or official organization domain)

Capacity & hookups
- total_campsites: the total site count as stated (string, e.g., "168")
- electrical_hookups_available: a short string confirming availability (e.g., "Yes, electric hookups at sites")
- capacity_reference_urls: list of URL(s) in the answer that directly support capacity and/or electrical hookups info

On-site facilities
- restrooms_available: short confirmation string (e.g., "Yes" or "Comfort stations available")
- showers_available: short confirmation string (e.g., "Yes")
- facilities_reference_urls: list of URL(s) in the answer supporting restroom and shower availability

Trails
- trails_available: short confirmation string (e.g., "Yes, hiking trails in the park")
- trail_name_or_description: at least one trail name or a brief description mentioned in the answer
- trail_reference_urls: list of URL(s) in the answer supporting the trail(s) being accessible from the park

Beach
- beach_access_available: short confirmation string (e.g., "Yes, ocean beach access")
- beach_reference_urls: list of URL(s) in the answer supporting ocean beach access

Reservations and March 2026
- advance_reservations_accepted: short confirmation string (e.g., "Yes, advance reservations accepted")
- reservation_url_or_phone: the reservations website URL (preferred) or phone number as given in the answer
- open_in_march_2026: short statement confirming the park is open in March 2026 (e.g., "Open year-round" or "Open in March 2026")
- march_2026_reservations_accepted: short confirmation string (e.g., "Yes, can book March 2026")
- reservation_reference_urls: list of URL(s) in the answer supporting reservation policy and March 2026 availability

Important:
- Extract only what appears explicitly in the answer. Do not invent data.
- For URLs, include only valid, explicit URLs found in the answer text (or in markdown links).
- If a value isn't clearly in the answer, set it to null (or [] for URL lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_url(s: Optional[str]) -> bool:
    if not s or not isinstance(s, str):
        return False
    s2 = s.strip().lower()
    return s2.startswith("http://") or s2.startswith("https://") or s2.startswith("www.")


def combine_urls(*parts: Union[None, str, List[str]]) -> List[str]:
    """Combine multiple URL strings/lists into a unique list; filter to valid URLs only."""
    out: List[str] = []
    seen = set()
    for p in parts:
        if p is None:
            continue
        if isinstance(p, str):
            if is_url(p) and p not in seen:
                out.append(p)
                seen.add(p)
        elif isinstance(p, list):
            for u in p:
                if is_url(u) and u not in seen:
                    out.append(u)
                    seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_identity(evaluator: Evaluator, parent, info: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Campground_Identity",
        desc="Correctly identify the specific campground by name and location",
        parent=parent,
        critical=True,
    )

    identity_sources = combine_urls(info.official_website_url)

    # Park_Name
    leaf_name = evaluator.add_leaf(
        id="Park_Name",
        desc="Provide the correct official name of the campground",
        parent=node,
        critical=True,
    )
    name_claim = f"The official name of the campground is '{info.campground_name}'."
    await evaluator.verify(
        claim=name_claim,
        node=leaf_name,
        sources=identity_sources,
        additional_instruction="Verify that the campground name on the provided official page matches the stated name (minor formatting variations acceptable).",
    )

    # Location_Verification
    leaf_loc = evaluator.add_leaf(
        id="Location_Verification",
        desc="Confirm the campground is in the Montauk, New York area",
        parent=node,
        critical=True,
    )
    loc_text = info.location_description or ""
    loc_claim = (
        f"The campground '{info.campground_name}' is located in or very near Montauk, New York (the Montauk area on the eastern end of Long Island). "
        f"Location mentioned in the answer: '{loc_text}'."
    )
    await evaluator.verify(
        claim=loc_claim,
        node=leaf_loc,
        sources=identity_sources,
        additional_instruction="Accept if the official page clearly indicates Montauk, NY, or a park within the Montauk area (e.g., Hither Hills State Park in Montauk).",
    )

    # Official_Website
    leaf_official = evaluator.add_leaf(
        id="Official_Website",
        desc="Provide URL to the official park or campground website",
        parent=node,
        critical=True,
    )
    official_claim = (
        f"This webpage is the official website page for the campground '{info.campground_name}' (e.g., on a government or official park domain)."
    )
    await evaluator.verify(
        claim=official_claim,
        node=leaf_official,
        sources=info.official_website_url if is_url(info.official_website_url) else None,
        additional_instruction="Consider it official if it is clearly from a government or official park domain (e.g., parks.ny.gov, suffolkcountyny.gov) or explicitly states it is the official park page.",
    )


async def verify_capacity_and_hookups(evaluator: Evaluator, parent, info: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Campsite_Capacity_and_Hookups",
        desc="Verify campsite capacity and electrical hookup availability",
        parent=parent,
        critical=True,
    )

    capacity_sources = combine_urls(info.capacity_reference_urls, info.official_website_url, info.reservation_reference_urls)

    # Capacity_Reference_URL (existence and relevance gate)
    ref_node = evaluator.add_custom_node(
        result=len(capacity_sources) > 0,
        id="Capacity_Reference_URL",
        desc="Provide URL confirming campsite capacity and hookup information",
        parent=node,
        critical=True,
    )

    # Minimum_100_Sites
    leaf_min100 = evaluator.add_leaf(
        id="Minimum_100_Sites",
        desc="Campground has at least 100 campsites",
        parent=node,
        critical=True,
    )
    min_claim = "The campground has at least 100 campsites (i.e., 100 or more total sites)."
    await evaluator.verify(
        claim=min_claim,
        node=leaf_min100,
        sources=capacity_sources,
        additional_instruction="Verify the total site count is >= 100 on the provided official/reference pages. Accept synonyms like 'sites' or 'campsites'.",
        extra_prerequisites=[ref_node],
    )

    # Total_Site_Count
    leaf_total = evaluator.add_leaf(
        id="Total_Site_Count",
        desc="Provide the specific total number of campsites",
        parent=node,
        critical=True,
    )
    total_txt = info.total_campsites or ""
    total_claim = f"The total number of campsites at '{info.campground_name}' is {total_txt}."
    await evaluator.verify(
        claim=total_claim,
        node=leaf_total,
        sources=capacity_sources,
        additional_instruction="Verify the exact total site count as stated (allowing minor formatting, e.g., 'sites' suffix). Prefer the official park page if multiple sources disagree.",
        extra_prerequisites=[ref_node],
    )

    # Electrical_Hookups
    leaf_elec = evaluator.add_leaf(
        id="Electrical_Hookups",
        desc="Campsites have electrical hookups available",
        parent=node,
        critical=True,
    )
    elec_claim = (
        "Electrical hookups (e.g., 30/50 amp electric) are available at some or all campsites at this campground."
    )
    await evaluator.verify(
        claim=elec_claim,
        node=leaf_elec,
        sources=capacity_sources,
        additional_instruction="Accept if the page indicates electric sites, electrical hookups, or similar wording for RVs. It can be a subset; it does not need to be every site.",
        extra_prerequisites=[ref_node],
    )


async def verify_facilities(evaluator: Evaluator, parent, info: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="On_Site_Facilities",
        desc="Verify essential on-site facilities are available",
        parent=parent,
        critical=True,
    )

    facilities_sources = combine_urls(info.facilities_reference_urls, info.official_website_url, info.reservation_reference_urls, info.capacity_reference_urls)

    # Facilities_Reference_URL (existence gate)
    ref_node = evaluator.add_custom_node(
        result=len(facilities_sources) > 0,
        id="Facilities_Reference_URL",
        desc="Provide URL confirming restroom and shower facilities",
        parent=node,
        critical=True,
    )

    # Restrooms
    leaf_rest = evaluator.add_leaf(
        id="Restrooms",
        desc="On-site restroom facilities are available",
        parent=node,
        critical=True,
    )
    rest_claim = "On-site restroom facilities (e.g., comfort stations or bathrooms) are available at the campground."
    await evaluator.verify(
        claim=rest_claim,
        node=leaf_rest,
        sources=facilities_sources,
        additional_instruction="Look for 'restrooms', 'bathrooms', or 'comfort stations' on the official/reference pages.",
        extra_prerequisites=[ref_node],
    )

    # Showers
    leaf_show = evaluator.add_leaf(
        id="Showers",
        desc="On-site shower facilities are available",
        parent=node,
        critical=True,
    )
    show_claim = "On-site shower facilities are available at the campground."
    await evaluator.verify(
        claim=show_claim,
        node=leaf_show,
        sources=facilities_sources,
        additional_instruction="Look for 'showers' information on the official/reference pages.",
        extra_prerequisites=[ref_node],
    )


async def verify_trails(evaluator: Evaluator, parent, info: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Hiking_Trail_Access",
        desc="Verify hiking trail availability and provide trail information",
        parent=parent,
        critical=True,
    )

    trail_sources = combine_urls(info.trail_reference_urls, info.official_website_url)

    # Trail_Reference_URL (existence gate)
    ref_node = evaluator.add_custom_node(
        result=len(trail_sources) > 0,
        id="Trail_Reference_URL",
        desc="Provide URL documenting the hiking trail information",
        parent=node,
        critical=True,
    )

    # Trails_Available
    leaf_trails = evaluator.add_leaf(
        id="Trails_Available",
        desc="Hiking trails are available within or accessible from the park",
        parent=node,
        critical=True,
    )
    trails_claim = "Hiking trails are available within the park or are directly accessible from the campground."
    await evaluator.verify(
        claim=trails_claim,
        node=leaf_trails,
        sources=trail_sources,
        additional_instruction="Confirm that the park/campground provides access to hiking trails (within the park boundary or directly connected).",
        extra_prerequisites=[ref_node],
    )

    # Trail_Name_or_Description
    leaf_trail_name = evaluator.add_leaf(
        id="Trail_Name_or_Description",
        desc="Provide at least one hiking trail name or description",
        parent=node,
        critical=True,
    )
    trail_text = info.trail_name_or_description or ""
    trail_claim = f"At least one hiking trail at or near the park is: '{trail_text}'."
    await evaluator.verify(
        claim=trail_claim,
        node=leaf_trail_name,
        sources=trail_sources,
        additional_instruction="Verify that the named or described trail is indeed at or directly accessible from the park/campground.",
        extra_prerequisites=[ref_node],
    )


async def verify_beach(evaluator: Evaluator, parent, info: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Beach_Access",
        desc="Verify ocean beach access availability",
        parent=parent,
        critical=True,
    )

    beach_sources = combine_urls(info.beach_reference_urls, info.official_website_url)

    # Beach_Reference_URL (existence gate)
    ref_node = evaluator.add_custom_node(
        result=len(beach_sources) > 0,
        id="Beach_Reference_URL",
        desc="Provide URL confirming beach access",
        parent=node,
        critical=True,
    )

    # Beach_Available
    leaf_beach = evaluator.add_leaf(
        id="Beach_Available",
        desc="Ocean beach access is available from the campground",
        parent=node,
        critical=True,
    )
    beach_claim = "The park/campground provides access to ocean beaches (Atlantic Ocean shoreline)."
    await evaluator.verify(
        claim=beach_claim,
        node=leaf_beach,
        sources=beach_sources,
        additional_instruction="Confirm that the park fronts or provides access to the Atlantic Ocean beach, not just a lake or bay beach.",
        extra_prerequisites=[ref_node],
    )


async def verify_reservations(evaluator: Evaluator, parent, info: CampgroundExtraction) -> None:
    node = evaluator.add_parallel(
        id="Reservation_System_and_March_2026",
        desc="Verify reservation system and March 2026 availability",
        parent=parent,
        critical=True,
    )

    # Compile reservation-related sources (URLs only)
    reservation_sources = combine_urls(info.reservation_reference_urls, info.official_website_url)

    # Reservation_Contact_Info (existence: URL or phone in answer)
    contact_exists = bool(info.reservation_url_or_phone and str(info.reservation_url_or_phone).strip())
    contact_node = evaluator.add_custom_node(
        result=contact_exists,
        id="Reservation_Contact_Info",
        desc="Provide reservation website URL or phone number",
        parent=node,
        critical=True,
    )

    # Reservation_Reference_URL (existence gate: need at least one URL)
    ref_node = evaluator.add_custom_node(
        result=len(reservation_sources) > 0,
        id="Reservation_Reference_URL",
        desc="Provide URL confirming reservation policies and March 2026 availability",
        parent=node,
        critical=True,
    )

    # Advance_Reservations_Accepted
    leaf_adv = evaluator.add_leaf(
        id="Advance_Reservations_Accepted",
        desc="Campground accepts advance reservations (not first-come-first-served only)",
        parent=node,
        critical=True,
    )
    adv_claim = "The campground accepts advance reservations (i.e., it is not exclusively first-come, first-served)."
    await evaluator.verify(
        claim=adv_claim,
        node=leaf_adv,
        sources=reservation_sources,
        additional_instruction="Confirm via the official reservation platform/policy (e.g., parks.ny.gov, ReserveAmerica, county reservation page) that advance reservations are accepted.",
        extra_prerequisites=[ref_node],
    )

    # Open_in_March_2026
    leaf_open_march = evaluator.add_leaf(
        id="Open_in_March_2026",
        desc="Park is open and operational during March 2026",
        parent=node,
        critical=True,
    )
    open_claim = f"The park/campground is open and operating during {TARGET_MONTH}."
    await evaluator.verify(
        claim=open_claim,
        node=leaf_open_march,
        sources=reservation_sources,
        additional_instruction=f"Verify season dates or operating calendar includes {TARGET_MONTH}. Accept if the park is stated as 'year-round' or shows operating dates covering March 2026.",
        extra_prerequisites=[ref_node],
    )

    # March_2026_Reservations_Accepted
    leaf_book_march = evaluator.add_leaf(
        id="March_2026_Reservations_Accepted",
        desc="Reservations can be made for March 2026",
        parent=node,
        critical=True,
    )
    book_claim = f"Reservations can be made for {TARGET_MONTH} for this campground."
    await evaluator.verify(
        claim=book_claim,
        node=leaf_book_march,
        sources=reservation_sources,
        additional_instruction=f"Check the reservation site/policy for booking windows or calendars indicating booking is accepted for {TARGET_MONTH}. Accept if the system is year-round and allows booking for that month.",
        extra_prerequisites=[ref_node],
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
    # Initialize evaluator (root is parallel to independently assess each requirement set)
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

    # Extract structured info from the answer
    extracted: CampgroundExtraction = await evaluator.extract(
        prompt=prompt_extract_campground(),
        template_class=CampgroundExtraction,
        extraction_name="campground_extraction",
    )

    # Build the verification tree according to the rubric
    # Root-level "task" node (critical aggregator for all requirement clusters)
    task_node = evaluator.add_parallel(
        id="Montauk_Campground_Task",
        desc="Identify and verify a suitable campground in Montauk, NY meeting all requirements for March 2026",
        parent=root,
        critical=True,
    )

    # Identity
    await verify_identity(evaluator, task_node, extracted)

    # Capacity & Hookups
    await verify_capacity_and_hookups(evaluator, task_node, extracted)

    # Facilities
    await verify_facilities(evaluator, task_node, extracted)

    # Trails
    await verify_trails(evaluator, task_node, extracted)

    # Beach access
    await verify_beach(evaluator, task_node, extracted)

    # Reservations and March 2026
    await verify_reservations(evaluator, task_node, extracted)

    # Return aggregated summary
    return evaluator.get_summary()