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
TASK_ID = "la_state_park_campground_requirements"
TASK_DESCRIPTION = """
Identify a Louisiana state park campground that meets ALL of the following requirements:
(1) Offers premium campsites with full hookups (water, electric with both 30 and 50 amp service, and sewer);
(2) Has ADA-accessible campsites available;
(3) Has group camping facilities that can accommodate at least 65 people;
(4) Individual campsites must accommodate up to 6 people;
(5) Allows pets with leash restrictions of no more than 10 feet;
(6) Has a check-in time of 3 PM or earlier;
(7) Has a check-out time of 11 AM or later;
(8) Enforces quiet hours between 10-11 PM and 6-7 AM;
(9) Allows camping for at least 15 consecutive days;
(10) Uses the Louisiana State Parks online reservation system (gooutdoorslouisiana.com);
(11) Has picnic areas with tables and covered pavilions;
(12) Has playground facilities on-site;
(13) Provides shower and bathhouse facilities;
(14) Has an RV dump station on-site;
(15) Offers water access (lake or bayou) for recreational activities;
(16) Provides multiple recreational activities.

Provide the name of the state park, confirm it is in Louisiana, and provide reference URLs that verify each of the major requirement categories (site types, policies, amenities, recreation features, and reservation system).
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CampgroundExtraction(BaseModel):
    """
    Extract essential fields from the agent's answer. We focus on:
    - The identified Louisiana state park name
    - Reference URLs grouped by verification category
    """
    park_name: Optional[str] = None

    # Category reference URLs (only URLs explicitly present in the answer)
    site_type_urls: List[str] = Field(default_factory=list)      # RV hookups, ADA, group camping, per-site capacity
    policy_urls: List[str] = Field(default_factory=list)         # Pets/leash, check-in/out, quiet hours, maximum stay
    reservation_urls: List[str] = Field(default_factory=list)    # Reservation system references (gooutdoorslouisiana)
    amenities_urls: List[str] = Field(default_factory=list)      # Picnic/pavilion, playground, showers/bathhouse, dump station
    recreation_urls: List[str] = Field(default_factory=list)     # Water access & multiple activities


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campground() -> str:
    return """
    Extract the following from the answer:

    1) park_name:
       - The exact name of the Louisiana state park campground being proposed.

    2) site_type_urls:
       - All URLs cited that substantiate campsite types/specifications for this park, including:
         • Premium RV sites with full hookups (water, both 30 & 50 amp electric, and sewer)
         • ADA-accessible campsites availability
         • Group camping facilities and capacity
         • Individual campsite capacity (up to 6 people per site)

    3) policy_urls:
       - All URLs cited that substantiate campground policies, including:
         • Pets allowed AND leash restriction length (10 ft or less)
         • Check-in time (3 PM or earlier)
         • Check-out time (11 AM or later)
         • Quiet hours (start between 10–11 PM and end between 6–7 AM)
         • Maximum consecutive stay (at least 15 days)

    4) reservation_urls:
       - All URLs cited that substantiate the reservation system usage for Louisiana State Parks
         (e.g., gooutdoorslouisiana.com pages for this park or the LA State Parks reservation portal).

    5) amenities_urls:
       - All URLs cited that substantiate physical amenities, including:
         • Picnic areas with tables and covered pavilions
         • Playground facilities
         • Shower and bathhouse facilities
         • RV dump station

    6) recreation_urls:
       - All URLs cited that substantiate recreation features, including:
         • Water access (lake or bayou) suitable for recreation
         • Multiple recreational activities (e.g., boating, fishing, hiking, paddling, etc.)

    IMPORTANT:
    - Only include URLs explicitly present in the answer. Do NOT invent any.
    - Extract the full, valid URLs. If a URL is missing a protocol, prepend http://.
    - If a category has no URLs provided, return an empty array for that category.
    - Return a single JSON object with fields: park_name, site_type_urls, policy_urls, reservation_urls, amenities_urls, recreation_urls.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _unique_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _ensure_sources(primary: List[str], fallback: List[str]) -> List[str]:
    """Use primary list if non-empty; otherwise fallback."""
    return primary if (primary and len(primary) > 0) else fallback


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_answer_identification_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CampgroundExtraction,
) -> None:
    """
    Build checks for:
    - State_Park_Name_Provided
    - Louisiana_State_Park (confirm located in Louisiana and part of Louisiana State Parks system)
    """
    group = evaluator.add_parallel(
        id="Answer_Identification",
        desc="The response clearly identifies the campground/state park requested",
        parent=parent_node,
        critical=True
    )

    # State_Park_Name_Provided (existence)
    name_ok = extracted.park_name is not None and isinstance(extracted.park_name, str) and extracted.park_name.strip() != ""
    evaluator.add_custom_node(
        result=name_ok,
        id="State_Park_Name_Provided",
        desc="Provides the name of the state park (campground location) being proposed",
        parent=group,
        critical=True
    )

    # Louisiana_State_Park (verification)
    leaf = evaluator.add_leaf(
        id="Louisiana_State_Park",
        desc="Confirms the campground is located within a Louisiana state park (i.e., in Louisiana and part of the Louisiana State Parks system)",
        parent=group,
        critical=True
    )
    park_name = extracted.park_name or "the identified park"
    all_urls = _unique_urls(
        extracted.site_type_urls,
        extracted.policy_urls,
        extracted.reservation_urls,
        extracted.amenities_urls,
        extracted.recreation_urls,
    )
    claim = f"'{park_name}' is a Louisiana State Park located in Louisiana and part of the Louisiana State Parks system."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=all_urls,
        additional_instruction="Consider official Louisiana State Parks (lastateparks.com) pages or the official gooutdoorslouisiana.com reservation portal as strong evidence. Accept reasonable park name variants."
    )


async def build_campsite_types_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CampgroundExtraction,
) -> None:
    """
    Build checks for campsite types/specifications:
    - Premium RV sites with full hookups (water, 30/50A electric, sewer)
    - ADA-accessible sites available
    - Group camping facilities accommodate at least 65 people
    - Individual campsite capacity up to 6 people per site
    - Reference URL(s) provided for this category
    """
    group = evaluator.add_parallel(
        id="Campsite_Types_And_Specifications",
        desc="The campground offers the required campsite types and capacities",
        parent=parent_node,
        critical=True
    )

    # Reference URL existence for this category
    evaluator.add_custom_node(
        result=bool(extracted.site_type_urls),
        id="Reference_URL_Site_Types",
        desc="Provides reference URL(s) verifying the campsite types/specifications category",
        parent=group,
        critical=True
    )

    # Sources to use; if empty, fallback to all URLs
    all_urls = _unique_urls(
        extracted.site_type_urls,
        extracted.policy_urls,
        extracted.amenities_urls,
        extracted.reservation_urls,
        extracted.recreation_urls,
    )
    site_sources = _ensure_sources(extracted.site_type_urls, all_urls)
    park_name = extracted.park_name or "the identified park"

    # Premium RV sites with full hookups (water + 30/50 amp electric + sewer)
    leaf_full_hookups = evaluator.add_leaf(
        id="Premium_RV_Sites_Available",
        desc="Offers premium RV campsites with full hookups (water, electric 30/50 amp, and sewer)",
        parent=group,
        critical=True
    )
    claim_full = f"{park_name} offers premium RV campsites with full hookups: water, electric service including 30 and 50 amps, and sewer."
    await evaluator.verify(
        claim=claim_full,
        node=leaf_full_hookups,
        sources=site_sources,
        additional_instruction="Look for 'Premium Campsite' or similar wording listing water + electric (30/50 amp) + sewer explicitly. Accept equivalent wording or icons indicating 30/50A and sewer."
    )

    # ADA-accessible campsites available
    leaf_ada = evaluator.add_leaf(
        id="ADA_Accessible_Sites_Available",
        desc="Has ADA-accessible campsites available",
        parent=group,
        critical=True
    )
    claim_ada = f"{park_name} has ADA-accessible campsites available."
    await evaluator.verify(
        claim=claim_ada,
        node=leaf_ada,
        sources=site_sources,
        additional_instruction="Accept terms such as 'ADA-accessible', 'accessible campsite', or similar. It should be explicit that ADA/accessible sites exist."
    )

    # Group camping facilities accommodate at least 65 people
    leaf_group = evaluator.add_leaf(
        id="Group_Camp_Facilities",
        desc="Has group camping facilities that can accommodate at least 65 people",
        parent=group,
        critical=True
    )
    claim_group = f"{park_name} provides group camping facilities that can accommodate at least 65 people."
    await evaluator.verify(
        claim=claim_group,
        node=leaf_group,
        sources=site_sources,
        additional_instruction="Look for 'Group Camp' capacity numbers (e.g., group camp lodging with bunk beds). Capacity must be 65 or higher; if multiple group camps exist, any single facility meeting >=65 satisfies this."
    )

    # Individual campsite capacity up to 6 people
    leaf_capacity = evaluator.add_leaf(
        id="Individual_Site_Capacity",
        desc="Individual campsites accommodate up to 6 people per site",
        parent=group,
        critical=True
    )
    claim_capacity = f"Individual campsites at {park_name} accommodate up to 6 people per site."
    await evaluator.verify(
        claim=claim_capacity,
        node=leaf_capacity,
        sources=_ensure_sources(extracted.policy_urls, site_sources),
        additional_instruction="Check park or statewide campground occupancy rules indicating max 6 people per campsite. If the rule is statewide for Louisiana State Parks, it applies here."
    )


async def build_policy_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CampgroundExtraction,
) -> None:
    """
    Build checks for campground policies:
    - Pets allowed; leash length no more than 10 feet
    - Check-in time 3 PM or earlier
    - Check-out time 11 AM or later
    - Quiet hours between 10–11 PM and 6–7 AM
    - Maximum stay at least 15 consecutive days
    - Reference URL(s) provided for this category
    """
    group = evaluator.add_parallel(
        id="Campground_Policies",
        desc="The campground meets required policy constraints",
        parent=parent_node,
        critical=True
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(extracted.policy_urls),
        id="Reference_URL_Policies",
        desc="Provides reference URL(s) verifying the policies category",
        parent=group,
        critical=True
    )

    all_urls = _unique_urls(
        extracted.policy_urls,
        extracted.site_type_urls,
        extracted.reservation_urls,
        extracted.amenities_urls,
        extracted.recreation_urls,
    )
    policy_sources = _ensure_sources(extracted.policy_urls, all_urls)
    park_name = extracted.park_name or "the identified park"

    # Pet policy: pets allowed; leash length <= 10 ft
    leaf_pets = evaluator.add_leaf(
        id="Pet_Policy",
        desc="Allows pets with leash restrictions of 10 feet or less",
        parent=group,
        critical=True
    )
    claim_pets = f"Pets are allowed at {park_name}, and leashes must be no longer than 10 feet."
    await evaluator.verify(
        claim=claim_pets,
        node=leaf_pets,
        sources=policy_sources,
        additional_instruction="Accept stricter leash limits (e.g., 6-foot leash) as satisfying 'no more than 10 feet'. The rule must clearly allow pets and state a leash length requirement not exceeding 10 feet."
    )

    # Check-in time: 3 PM or earlier
    leaf_checkin = evaluator.add_leaf(
        id="Check_In_Time_Policy",
        desc="Check-in time is 3 PM or earlier",
        parent=group,
        critical=True
    )
    claim_checkin = f"The campsite check-in time at {park_name} is at 3 PM or earlier."
    await evaluator.verify(
        claim=claim_checkin,
        node=leaf_checkin,
        sources=_ensure_sources(_unique_urls(extracted.policy_urls, extracted.reservation_urls), policy_sources),
        additional_instruction="Times such as 2 PM or 3 PM satisfy this requirement. If multiple times are listed, campsite check-in time must be <= 3 PM."
    )

    # Check-out time: 11 AM or later
    leaf_checkout = evaluator.add_leaf(
        id="Check_Out_Time_Policy",
        desc="Check-out time is 11 AM or later",
        parent=group,
        critical=True
    )
    claim_checkout = f"The campsite check-out time at {park_name} is at 11 AM or later."
    await evaluator.verify(
        claim=claim_checkout,
        node=leaf_checkout,
        sources=_ensure_sources(_unique_urls(extracted.policy_urls, extracted.reservation_urls), policy_sources),
        additional_instruction="Times such as 11 AM or 12 PM (noon) satisfy this requirement. If multiple times are listed, campsite check-out time must be >= 11 AM."
    )

    # Quiet hours: start between 10–11 PM, end between 6–7 AM
    leaf_quiet = evaluator.add_leaf(
        id="Quiet_Hours_Enforcement",
        desc="Enforces quiet hours between 10-11 PM and 6-7 AM",
        parent=group,
        critical=True
    )
    claim_quiet = f"Quiet hours at {park_name} begin around 10–11 PM and end around 6–7 AM."
    await evaluator.verify(
        claim=claim_quiet,
        node=leaf_quiet,
        sources=policy_sources,
        additional_instruction="Accept quiet hours that start at 10 PM or 11 PM and end at 6 AM or 7 AM. Typical phrasing like '10 PM to 6 AM' or '10 PM to 7 AM' is acceptable."
    )

    # Maximum stay: at least 15 consecutive days
    leaf_stay = evaluator.add_leaf(
        id="Maximum_Stay_Policy",
        desc="Allows camping for at least 15 consecutive days",
        parent=group,
        critical=True
    )
    claim_stay = f"Campers may stay at {park_name} for at least 15 consecutive days."
    await evaluator.verify(
        claim=claim_stay,
        node=leaf_stay,
        sources=policy_sources,
        additional_instruction="Verify maximum stay length. The policy must allow at least 15 consecutive days (>= 15). If policy states 14 days, this requirement is NOT satisfied."
    )


async def build_reservation_system_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CampgroundExtraction,
) -> None:
    """
    Build checks for reservation system:
    - Uses the Louisiana State Parks online reservation system (gooutdoorslouisiana.com)
    - Reference URL(s) provided for this category
    """
    group = evaluator.add_parallel(
        id="Reservation_System",
        desc="The campground uses the required reservation system",
        parent=parent_node,
        critical=True
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(extracted.reservation_urls),
        id="Reference_URL_Reservation",
        desc="Provides reference URL(s) verifying the reservation system category",
        parent=group,
        critical=True
    )

    all_urls = _unique_urls(
        extracted.reservation_urls,
        extracted.policy_urls,
        extracted.site_type_urls,
        extracted.amenities_urls,
        extracted.recreation_urls,
    )
    reservation_sources = _ensure_sources(extracted.reservation_urls, all_urls)
    park_name = extracted.park_name or "the identified park"

    # Uses gooutdoorslouisiana.com
    leaf_res_sys = evaluator.add_leaf(
        id="Louisiana_System_Usage",
        desc="Uses the Louisiana State Parks online reservation system (gooutdoorslouisiana.com)",
        parent=group,
        critical=True
    )
    claim_res = f"Reservations for {park_name} are made through the Louisiana State Parks system at gooutdoorslouisiana.com."
    await evaluator.verify(
        claim=claim_res,
        node=leaf_res_sys,
        sources=reservation_sources,
        additional_instruction="Look for official reservation links or pages under gooutdoorslouisiana.com. If the reservation button/URL points to this domain, that confirms usage."
    )


async def build_amenities_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CampgroundExtraction,
) -> None:
    """
    Build checks for physical amenities:
    - Picnic areas with tables and covered pavilions
    - Playground facilities
    - Shower and bathhouse facilities
    - RV dump station
    - Reference URL(s) provided for this category
    """
    group = evaluator.add_parallel(
        id="Physical_Amenities",
        desc="The campground provides required physical amenities",
        parent=parent_node,
        critical=True
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(extracted.amenities_urls),
        id="Reference_URL_Amenities",
        desc="Provides reference URL(s) verifying the amenities category",
        parent=group,
        critical=True
    )

    all_urls = _unique_urls(
        extracted.amenities_urls,
        extracted.site_type_urls,
        extracted.policy_urls,
        extracted.reservation_urls,
        extracted.recreation_urls,
    )
    amenity_sources = _ensure_sources(extracted.amenities_urls, all_urls)
    park_name = extracted.park_name or "the identified park"

    # Picnic areas with tables and covered pavilions
    leaf_picnic = evaluator.add_leaf(
        id="Picnic_Areas_With_Pavilions",
        desc="Has picnic areas with tables and covered pavilions",
        parent=group,
        critical=True
    )
    claim_picnic = f"{park_name} has picnic areas with tables and covered pavilions."
    await evaluator.verify(
        claim=claim_picnic,
        node=leaf_picnic,
        sources=amenity_sources,
        additional_instruction="Look specifically for mentions of both picnic tables and covered pavilions/shelters."
    )

    # Playground facilities
    leaf_playground = evaluator.add_leaf(
        id="Playground_Facilities",
        desc="Has playground facilities on-site",
        parent=group,
        critical=True
    )
    claim_playground = f"{park_name} has playground facilities on-site."
    await evaluator.verify(
        claim=claim_playground,
        node=leaf_playground,
        sources=amenity_sources,
        additional_instruction="Look for 'playground' or 'play area' listed among amenities."
    )

    # Shower and bathhouse facilities
    leaf_bathhouse = evaluator.add_leaf(
        id="Shower_Bathhouse_Facilities",
        desc="Provides shower and bathhouse facilities",
        parent=group,
        critical=True
    )
    claim_bathhouse = f"{park_name} provides shower and bathhouse facilities for campers."
    await evaluator.verify(
        claim=claim_bathhouse,
        node=leaf_bathhouse,
        sources=amenity_sources,
        additional_instruction="Look for restrooms with showers, bathhouses, or similar phrasing under amenities."
    )

    # RV dump station on-site
    leaf_dump = evaluator.add_leaf(
        id="RV_Dump_Station",
        desc="Has an RV dump station on-site",
        parent=group,
        critical=True
    )
    claim_dump = f"{park_name} has an RV dump station on-site."
    await evaluator.verify(
        claim=claim_dump,
        node=leaf_dump,
        sources=amenity_sources,
        additional_instruction="Look for 'dump station' listed under amenities/services for campers."
    )


async def build_recreation_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CampgroundExtraction,
) -> None:
    """
    Build checks for recreation features:
    - Water access (lake or bayou)
    - Multiple recreational activities
    - Reference URL(s) provided for this category
    """
    group = evaluator.add_parallel(
        id="Recreational_Features",
        desc="The campground offers required recreation-related features",
        parent=parent_node,
        critical=True
    )

    # Reference URL existence
    evaluator.add_custom_node(
        result=bool(extracted.recreation_urls),
        id="Reference_URL_Recreation",
        desc="Provides reference URL(s) verifying the recreation features category",
        parent=group,
        critical=True
    )

    all_urls = _unique_urls(
        extracted.recreation_urls,
        extracted.amenities_urls,
        extracted.site_type_urls,
        extracted.policy_urls,
        extracted.reservation_urls,
    )
    recreation_sources = _ensure_sources(extracted.recreation_urls, all_urls)
    park_name = extracted.park_name or "the identified park"

    # Water access
    leaf_water = evaluator.add_leaf(
        id="Water_Access",
        desc="Offers water access (lake or bayou) for recreational activities",
        parent=group,
        critical=True
    )
    claim_water = f"{park_name} offers water access such as a lake, bayou, or similar body of water suitable for recreation."
    await evaluator.verify(
        claim=claim_water,
        node=leaf_water,
        sources=recreation_sources,
        additional_instruction="Look for explicit mention of a lake, bayou, pond, or river with recreation like boating, paddling, or fishing. Photos/maps with captions are acceptable evidence if clearly indicating water access."
    )

    # Multiple recreational activities
    leaf_multi = evaluator.add_leaf(
        id="Multiple_Activities",
        desc="Provides multiple recreational activities",
        parent=group,
        critical=True
    )
    claim_multi = f"{park_name} provides multiple recreational activities."
    await evaluator.verify(
        claim=claim_multi,
        node=leaf_multi,
        sources=recreation_sources,
        additional_instruction="Confirm there are at least two distinct activities (e.g., hiking + fishing, paddling + birding). It can be explicitly listed or apparent from amenities/feature sections."
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
    Evaluate an answer for the Louisiana state park campground requirements task.
    """
    # Initialize evaluator with a parallel root; create the main critical node under it
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

    # Add a summary of URL counts as custom info (for debugging/result context)
    evaluator.add_custom_info(
        {
            "park_name": extracted.park_name,
            "site_type_urls_count": len(extracted.site_type_urls),
            "policy_urls_count": len(extracted.policy_urls),
            "reservation_urls_count": len(extracted.reservation_urls),
            "amenities_urls_count": len(extracted.amenities_urls),
            "recreation_urls_count": len(extracted.recreation_urls),
        },
        info_type="extraction_stats",
        info_name="url_summary"
    )

    # Build the main critical task node
    main = evaluator.add_parallel(
        id="Louisiana_State_Park_Campground_Identification",
        desc="Identify a Louisiana state park campground that meets all specified requirements and provide required outputs",
        parent=root,
        critical=True
    )

    # Build subtrees
    await build_answer_identification_checks(evaluator, main, extracted)
    await build_campsite_types_checks(evaluator, main, extracted)
    await build_policy_checks(evaluator, main, extracted)
    await build_reservation_system_checks(evaluator, main, extracted)
    await build_amenities_checks(evaluator, main, extracted)
    await build_recreation_checks(evaluator, main, extracted)

    # Return standardized evaluation summary
    return evaluator.get_summary()