import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nysp_accessible_camping_2026"
TASK_DESCRIPTION = (
    "I am planning a family camping trip to New York State Parks during summer 2026 and need to find campgrounds that accommodate our accessibility needs.\n\n"
    "Please identify three different New York State Park campgrounds that meet ALL of the following requirements:\n\n"
    "1. The campground must be located within a New York State Park and accept reservations through the ReserveAmerica reservation system\n"
    "2. The campground must be open and available for camping during the summer season (June through August 2026)\n"
    "3. The campground must offer ADA-accessible campsites\n"
    "4. The campground must have wheelchair-accessible restroom facilities with hot water and showers\n"
    "5. The campsites must include electrical hookups with at least 30-amp service\n"
    "6. Each campsite must be equipped with a picnic table and either a fire ring or grill\n"
    "7. The park must have accessible hiking trails or nature trails available (trails that are wheelchair-accessible or rated as easy difficulty)\n\n"
    "For each of the three campgrounds, provide:\n"
    "- The name of the campground and the New York State Park where it is located\n"
    "- A direct URL link to the campground's official ReserveAmerica reservation page or the official New York State Parks website page for that specific campground\n"
    "- Verification that all seven requirements listed above are met, with supporting reference URLs for the accessibility features, amenities, and trail information"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampgroundItem(BaseModel):
    campground_name: Optional[str] = None
    park_name: Optional[str] = None
    # Direct official page URL (either ReserveAmerica reservation page or the official NY State Parks page)
    official_url: Optional[str] = None
    # Any additional URLs the answer cited that support amenities/accessibility/trails/reservations
    support_urls: List[str] = Field(default_factory=list)


class CampgroundList(BaseModel):
    campgrounds: List[CampgroundItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_campgrounds() -> str:
    return """
    Extract up to three specific New York State Park campgrounds mentioned in the answer that are intended to meet the listed requirements.
    For each campground, extract exactly the following fields:
    - campground_name: The campground's name as stated in the answer (e.g., "North-South Lake Campground").
    - park_name: The New York State Park name where the campground is located (e.g., "Letchworth State Park"). If not explicitly stated, return null.
    - official_url: A direct URL to either (a) the campground’s ReserveAmerica reservation page or (b) the official New York State Parks web page for that specific campground. If multiple are present, prefer ReserveAmerica first; otherwise use the official NY State Parks page. If not present, return null.
    - support_urls: A list of all other URLs in the answer that support the campground’s amenities, accessibility (ADA/wheelchair-accessible, showers with hot water), electrical service (30 amp or greater), and accessible/easy trails at the park. Include only actual URLs from the answer, not descriptions.

    Return a JSON object with a top-level array "campgrounds" of at most three items in the exact order they appear in the answer.
    Do not fabricate any information. If a field is missing, set it to null (for single-value fields) or an empty list (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(item: CampgroundItem) -> List[str]:
    """Combine the official URL and all support URLs into a deduplicated list."""
    urls: List[str] = []
    if item.official_url and isinstance(item.official_url, str) and item.official_url.strip():
        urls.append(item.official_url.strip())
    for u in item.support_urls or []:
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _safe_name(value: Optional[str], fallback: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


# --------------------------------------------------------------------------- #
# Verification builder for one campground                                     #
# --------------------------------------------------------------------------- #
async def verify_one_campground(
    evaluator: Evaluator,
    parent_node,
    item: CampgroundItem,
    idx: int
) -> None:
    """
    Build the verification subtree for a single campground according to the rubric.
    """
    cg_name = _safe_name(item.campground_name, f"Campground #{idx+1}")
    pk_name = _safe_name(item.park_name, "a New York State Park")
    all_sources = _combine_sources(item)

    # location_and_reservation (critical, parallel)
    loc_res_node = evaluator.add_parallel(
        id=f"cg{idx+1}_location_and_reservation",
        desc="Verify campground location and reservation system",
        parent=parent_node,
        critical=True
    )

    # official_page_url (critical leaf): presence of an official page URL
    official_url_exists = bool(item.official_url and item.official_url.strip())
    evaluator.add_custom_node(
        result=official_url_exists,
        id=f"cg{idx+1}_official_page_url",
        desc="Provide direct URL to the campground's official ReserveAmerica reservation page or official New York State Parks page",
        parent=loc_res_node,
        critical=True
    )

    # ny_state_park (critical leaf): located within a NY State Park
    nysp_node = evaluator.add_leaf(
        id=f"cg{idx+1}_ny_state_park",
        desc="Campground is located within a New York State Park",
        parent=loc_res_node,
        critical=True
    )
    if item.park_name and item.park_name.strip():
        claim_nysp = f"The campground '{cg_name}' is located within New York State Park '{pk_name}'."
    else:
        claim_nysp = f"The campground '{cg_name}' is located within a New York State Park."
    await evaluator.verify(
        claim=claim_nysp,
        node=nysp_node,
        sources=all_sources,
        additional_instruction="Confirm that the campground is part of the New York State Parks system. Accept clear indications such as the domain parks.ny.gov or the page explicitly saying it is a NY State Park."
    )

    # reserveamerica_booking (critical leaf): accepts reservations via ReserveAmerica
    ra_node = evaluator.add_leaf(
        id=f"cg{idx+1}_reserveamerica_booking",
        desc="Campground accepts reservations through ReserveAmerica system",
        parent=loc_res_node,
        critical=True
    )
    await evaluator.verify(
        claim="This campground accepts reservations through the ReserveAmerica reservation system.",
        node=ra_node,
        sources=all_sources,
        additional_instruction="Look for explicit ReserveAmerica booking links/buttons or language like 'ReserveAmerica'. Pages on reserveamerica.com (or newyorkstateparks.reserveamerica.com) are strong evidence."
    )

    # summer_availability (critical leaf): open during June–August 2026
    summer_node = evaluator.add_leaf(
        id=f"cg{idx+1}_summer_availability",
        desc="Campground operates during summer season (June-August 2026)",
        parent=loc_res_node,
        critical=True
    )
    await evaluator.verify(
        claim="The campground is open for camping during June, July, or August 2026.",
        node=summer_node,
        sources=all_sources,
        additional_instruction=(
            "Verify season dates or availability calendars. It is acceptable if the page states a recurring summer season (e.g., late May through September), "
            "as that implies summer months in 2026 are open unless otherwise noted."
        )
    )

    # accessibility_and_facilities (critical, parallel)
    acc_fac_node = evaluator.add_parallel(
        id=f"cg{idx+1}_accessibility_and_facilities",
        desc="Verify accessibility features and restroom facilities",
        parent=parent_node,
        critical=True
    )

    # ada_accessible_sites (critical leaf)
    ada_node = evaluator.add_leaf(
        id=f"cg{idx+1}_ada_accessible_sites",
        desc="Campground offers ADA-accessible campsites",
        parent=acc_fac_node,
        critical=True
    )
    await evaluator.verify(
        claim="The campground offers at least one ADA-accessible campsite.",
        node=ada_node,
        sources=all_sources,
        additional_instruction="Look for 'accessible campsite', 'ADA site', 'universal access', or wheelchair symbol indicating designated accessible campsites."
    )

    # accessible_restrooms (critical leaf)
    rest_node = evaluator.add_leaf(
        id=f"cg{idx+1}_accessible_restrooms",
        desc="Provides wheelchair-accessible restrooms with hot water and shower facilities",
        parent=acc_fac_node,
        critical=True
    )
    await evaluator.verify(
        claim="The campground provides wheelchair-accessible restrooms that include hot water and showers.",
        node=rest_node,
        sources=all_sources,
        additional_instruction="Explicit mentions of accessible restrooms and showers (with hot water) count. Accept phrasing like 'accessible comfort stations with hot showers.'"
    )

    # campsite_features_and_trails (critical, parallel)
    features_node = evaluator.add_parallel(
        id=f"cg{idx+1}_campsite_features_and_trails",
        desc="Verify campsite amenities and trail access",
        parent=parent_node,
        critical=True
    )

    # electrical_hookups (critical leaf)
    elec_node = evaluator.add_leaf(
        id=f"cg{idx+1}_electrical_hookups",
        desc="Campsites include electrical hookups (minimum 30-amp service)",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="Campsites at this campground include electrical hookups with at least 30-amp service.",
        node=elec_node,
        sources=all_sources,
        additional_instruction="Accept 30-amp or 50-amp service. If multiple site types exist, it suffices that electric sites with ≥30A are available."
    )

    # site_amenities (critical leaf)
    amenities_node = evaluator.add_leaf(
        id=f"cg{idx+1}_site_amenities",
        desc="Each campsite includes picnic table and fire ring or grill",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="Campsites include a picnic table and either a fire ring or a grill.",
        node=amenities_node,
        sources=all_sources,
        additional_instruction="Standard campsite amenity lists typically state 'picnic table' and 'fire ring' or 'grill'. Equivalents like 'fireplace' are acceptable."
    )

    # accessible_trails (critical leaf)
    trails_node = evaluator.add_leaf(
        id=f"cg{idx+1}_accessible_trails",
        desc="Park has accessible hiking trails or nature trails (wheelchair-accessible or easy difficulty)",
        parent=features_node,
        critical=True
    )
    await evaluator.verify(
        claim="The park has accessible hiking or nature trails that are wheelchair-accessible or rated as easy.",
        node=trails_node,
        sources=all_sources,
        additional_instruction="Look for accessible trail designations, paved/boardwalk paths, ADA-compliant trails, or clearly 'easy' trails suitable for wheelchairs or mobility devices."
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
    Evaluate an answer for accessible NY State Park campgrounds against the rubric.
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

    # 1) Extract campgrounds from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_campgrounds(),
        template_class=CampgroundList,
        extraction_name="campground_extraction"
    )

    # 2) Prepare exactly three items (pad with empty if fewer)
    items: List[CampgroundItem] = list(extracted.campgrounds[:3])
    while len(items) < 3:
        items.append(CampgroundItem())

    # 3) Build verification tree for each campground
    for i in range(3):
        cg_node = evaluator.add_parallel(
            id=f"campground_{i+1}",
            desc=f"{['First','Second','Third'][i]} qualifying campground with complete information",
            parent=root,
            critical=False
        )
        await verify_one_campground(evaluator, cg_node, items[i], i)

    # 4) Return standardized summary
    return evaluator.get_summary()