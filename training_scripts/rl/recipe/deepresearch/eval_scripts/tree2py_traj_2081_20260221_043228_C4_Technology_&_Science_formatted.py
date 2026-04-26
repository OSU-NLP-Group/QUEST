import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "colocation_tier3_connectivity"
TASK_DESCRIPTION = (
    "I'm researching colocation data center options for my company's infrastructure expansion with a focus on "
    "international connectivity. Find three Tier III certified colocation facilities in the United States that are "
    "located in cities with either submarine cable landing stations or major Internet Exchange Points (IXPs). For "
    "each facility, provide the Tier III certification status, uptime guarantee, redundancy specifications (N+1 "
    "redundancy and concurrent maintainability capabilities), and include reference URLs for verification. The three "
    "facilities should be in different cities to ensure geographic diversity."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityItem(BaseModel):
    """
    One colocation facility as described in the answer.
    """
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    tier_status: Optional[str] = None  # e.g., "Uptime Institute Tier III Certified", "Tier III compliant"
    uptime_guarantee: Optional[str] = None  # e.g., "99.99%", "99.982%", etc.
    redundancy: Optional[str] = None  # free text; may include "N+1"
    concurrent_maintainability: Optional[str] = None  # free text; may include "concurrently maintainable"
    reference_urls: List[str] = Field(default_factory=list)  # All URLs cited for this facility


class FacilitiesExtraction(BaseModel):
    """
    Extraction result: list of facilities found in the answer.
    """
    facilities: List[FacilityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facilities() -> str:
    return """
    Extract all colocation facilities mentioned in the answer that the answer proposes for the task.
    For each facility, return the following fields:

    - name: The facility or data center name (e.g., "Equinix DC11", "CoreSite LA2", "Digital Realty 32 Avenue of the Americas").
    - city: The city where the facility is located.
    - state: The U.S. state (abbreviation or full name) if provided.
    - tier_status: The Tier level description as stated in the answer (e.g., "Uptime Institute Tier III Certified", "Tier III", "Tier III equivalent").
    - uptime_guarantee: Any uptime/SLA guarantee text in the answer (e.g., "99.99%", "99.982%", "five nines").
    - redundancy: Any redundancy description in the answer (e.g., "N+1 power and cooling", "2N", "N+1 UPS").
    - concurrent_maintainability: Any mention that the facility supports "concurrent maintainability" or similar language.
    - reference_urls: All URLs mentioned in the answer that are relevant to this facility (including provider pages, Uptime Institute pages, city/IXP pages, submarine cable map pages, etc.). 
      Include any URL that the answer associates with this facility’s certification/specs or with the city’s IXP/submarine landing status.

    IMPORTANT:
    - Only extract information explicitly present in the answer.
    - Do not invent any URLs or facts.
    - The URLs may appear as plain links or markdown links; extract the actual URL strings.
    - If a field is not present, set it to null (or an empty array for reference_urls).
    - Include all facilities the answer lists; do not filter to three at extraction time (we will select the first three later).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_str(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _distinct_by_name(facilities: List[FacilityItem]) -> List[FacilityItem]:
    """
    Deduplicate facilities by normalized name (prefer those with non-empty name).
    """
    seen = set()
    out: List[FacilityItem] = []
    for f in facilities:
        key = _normalize_str(f.name)
        if not key:
            # Skip nameless entries when selecting distinct facilities by name
            continue
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def select_first_three_distinct(facilities: List[FacilityItem]) -> List[FacilityItem]:
    """
    From all extracted facilities, select the first three with distinct names.
    """
    distinct = _distinct_by_name(facilities)
    return distinct[:3]


def summarize_selected(selected: List[FacilityItem]) -> Dict:
    return {
        "selected_count": len(selected),
        "facilities": [
            {
                "name": f.name,
                "city": f.city,
                "state": f.state,
                "urls_count": len(f.reference_urls),
            }
            for f in selected
        ]
    }


def cities_geographically_diverse(selected: List[FacilityItem]) -> bool:
    """
    Check if the three selected facilities are in three different cities (case-insensitive).
    Requires all three to have a city value.
    """
    if len(selected) != 3:
        return False
    cities = [_normalize_str(f.city) for f in selected]
    if any(c == "" for c in cities):
        return False
    return len(set(cities)) == 3


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def add_reference_url_nodes(
    evaluator: Evaluator,
    parent_node,
    selected: List[FacilityItem],
) -> List:
    """
    Under a critical 'Reference_URLs' parent, add a critical existence node for each facility:
    'Facility #i has at least one reference URL'.
    Returns the list of created nodes (in index order) to be used as prerequisites for other checks.
    """
    ref_group = evaluator.add_parallel(
        id="Reference_URLs",
        desc="Reference URLs are provided for each facility's Tier certification or specifications documentation",
        parent=parent_node,
        critical=True
    )

    prereq_nodes = []
    for i, fac in enumerate(selected):
        has_refs = bool(fac.reference_urls and len([u for u in fac.reference_urls if isinstance(u, str) and u.strip()]) > 0)
        node = evaluator.add_custom_node(
            result=has_refs,
            id=f"facility_{i}_reference_urls_present",
            desc=f"Facility #{i + 1} has at least one reference URL",
            parent=ref_group,
            critical=True
        )
        prereq_nodes.append(node)
    return prereq_nodes


async def add_tier_certification_nodes(
    evaluator: Evaluator,
    parent_node,
    selected: List[FacilityItem],
    url_prereqs: List
) -> None:
    """
    Under 'Tier_III_Certification' (critical), add one critical leaf per facility verifying
    Tier III certification or explicit Tier III level per sources.
    """
    tier_group = evaluator.add_parallel(
        id="Tier_III_Certification",
        desc="All three facilities are certified Tier III by the Uptime Institute or explicitly meet Tier III specifications",
        parent=parent_node,
        critical=True
    )

    for i, fac in enumerate(selected):
        leaf = evaluator.add_leaf(
            id=f"facility_{i}_tier3_verified",
            desc=f"Facility #{i + 1} Tier III certification or Tier III specification is supported by sources",
            parent=tier_group,
            critical=True
        )
        fac_name = fac.name or f"Facility #{i + 1}"
        claim = (
            f"{fac_name} is a Tier III data center. This is satisfied if the sources show it is "
            f"Uptime Institute Tier III certified OR explicitly described as Tier III (meeting Tier III specifications)."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=fac.reference_urls,
            additional_instruction=(
                "Accept as supported if the page states 'Uptime Institute Tier III Certified', 'Tier III certified', "
                "'Tier 3' by Uptime Institute, or clearly describes Tier III equivalence/specifications for the facility. "
                "If the page only mentions unrelated Tiers (e.g., Tier II) or lacks Tier information, mark as not supported."
            ),
            extra_prerequisites=[url_prereqs[i]]
        )


async def add_uptime_nodes(
    evaluator: Evaluator,
    parent_node,
    selected: List[FacilityItem],
    url_prereqs: List
) -> None:
    """
    Under 'Uptime_Specifications' (critical), add one critical leaf per facility verifying
    uptime guarantee is at least 99.982% (Tier III minimum).
    """
    uptime_group = evaluator.add_parallel(
        id="Uptime_Specifications",
        desc="All three facilities provide uptime guarantee information (minimum 99.982% for Tier III)",
        parent=parent_node,
        critical=True
    )

    for i, fac in enumerate(selected):
        leaf = evaluator.add_leaf(
            id=f"facility_{i}_uptime_min_verified",
            desc=f"Facility #{i + 1} uptime guarantee is at least 99.982% and supported by sources",
            parent=uptime_group,
            critical=True
        )
        fac_name = fac.name or f"Facility #{i + 1}"
        claim = (
            f"The documentation for {fac_name} provides an uptime or SLA guarantee of at least 99.982% "
            f"(values such as 99.99% or 99.999% also satisfy this)."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=fac.reference_urls,
            additional_instruction=(
                "Look for explicit uptime/SLA statements. Any value >= 99.982% qualifies (e.g., 99.99%, 99.995%, 99.999%). "
                "If only generic marketing language is present without a numeric uptime figure, treat as unsupported."
            ),
            extra_prerequisites=[url_prereqs[i]]
        )


async def add_redundancy_nodes(
    evaluator: Evaluator,
    parent_node,
    selected: List[FacilityItem],
    url_prereqs: List
) -> None:
    """
    Under 'Redundancy_Specifications' (critical), add two critical leaves per facility:
    - N+1 redundancy supported by sources
    - Concurrent maintainability capability supported by sources
    """
    red_group = evaluator.add_parallel(
        id="Redundancy_Specifications",
        desc="All three facilities provide redundancy specifications (N+1 redundancy, concurrent maintainability)",
        parent=parent_node,
        critical=True
    )

    for i, fac in enumerate(selected):
        # N+1 redundancy
        nplus1_leaf = evaluator.add_leaf(
            id=f"facility_{i}_nplus1_verified",
            desc=f"Facility #{i + 1} has N+1 redundancy supported by sources",
            parent=red_group,
            critical=True
        )
        fac_name = fac.name or f"Facility #{i + 1}"
        nplus1_claim = (
            f"The documentation for {fac_name} states there is at least N+1 redundancy for critical systems "
            f"(power and/or cooling)."
        )
        await evaluator.verify(
            claim=nplus1_claim,
            node=nplus1_leaf,
            sources=fac.reference_urls,
            additional_instruction=(
                "Accept mentions like 'N+1' explicitly; '2N' or 'N+2' are stricter and also satisfy N+1 or better. "
                "If no redundancy detail is provided, do not mark as supported."
            ),
            extra_prerequisites=[url_prereqs[i]]
        )

        # Concurrent maintainability
        cm_leaf = evaluator.add_leaf(
            id=f"facility_{i}_concurrent_maintainability_verified",
            desc=f"Facility #{i + 1} is concurrently maintainable (no downtime during maintenance), supported by sources",
            parent=red_group,
            critical=True
        )
        cm_claim = (
            f"The documentation for {fac_name} indicates 'concurrent maintainability' or equivalent capability "
            f"(e.g., maintenance can occur without downtime)."
        )
        await evaluator.verify(
            claim=cm_claim,
            node=cm_leaf,
            sources=fac.reference_urls,
            additional_instruction=(
                "Look for the term 'concurrently maintainable', 'concurrent maintainability', or equivalent language "
                "indicating maintenance activities can be performed without downtime."
            ),
            extra_prerequisites=[url_prereqs[i]]
        )


async def add_city_location_nodes(
    evaluator: Evaluator,
    parent_node,
    selected: List[FacilityItem],
    url_prereqs: List
) -> None:
    """
    Under 'Qualifying_City_Locations' (critical), add one critical leaf per facility verifying that the
    city where the facility is located has either a submarine cable landing station or a major IXP.
    """
    loc_group = evaluator.add_parallel(
        id="Qualifying_City_Locations",
        desc="All three facilities are located in US cities that have either submarine cable landing stations or major Internet Exchange Points (IXPs)",
        parent=parent_node,
        critical=True
    )

    for i, fac in enumerate(selected):
        leaf = evaluator.add_leaf(
            id=f"facility_{i}_city_qualifies",
            desc=f"Facility #{i + 1} city qualifies (submarine landing OR major IXP) supported by sources",
            parent=loc_group,
            critical=True
        )
        city_part = fac.city or "the city where this facility is located"
        claim = (
            f"The city associated with this facility ({city_part}) hosts either a submarine cable landing station "
            f"or a major Internet Exchange Point (IXP)."
        )
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=fac.reference_urls,
            additional_instruction=(
                "Evidence may include: mention of a submarine cable landing station in the city, or presence of a "
                "major IXP (e.g., DE-CIX, Equinix IX, NYIIX, Any2, SIX, MICE, etc.). The sources may be facility "
                "provider pages, city IX/IXP pages, submarine cable maps, or reputable telecom/IXP documentation. "
                "If none of the provided URLs substantiate this, mark as not supported."
            ),
            extra_prerequisites=[url_prereqs[i]]
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
    Evaluate an answer for the Tier III colocation with international connectivity task.
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

    # 1) Extract facilities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_facilities(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # 2) Select the first three distinct facilities by name as per guidance
    selected: List[FacilityItem] = select_first_three_distinct(extracted.facilities)

    # Pad to length 3 with empty entries if fewer found (so we still build 3 slots)
    while len(selected) < 3:
        selected.append(FacilityItem())

    # Record selection summary for debugging/transparency
    evaluator.add_custom_info(
        info=summarize_selected(selected),
        info_type="selection_summary",
        info_name="selected_facilities"
    )

    # 3) Build the main task node (non-critical to allow a mix of critical and non-critical children)
    main_node = evaluator.add_parallel(
        id="Find_Three_Colocation_Facilities",
        desc="Task requires identifying exactly three colocation data center facilities in the United States that meet specific infrastructure and location requirements for international connectivity",
        parent=root,
        critical=False
    )

    # 3.a) Exactly three distinct facilities identified (by name) - critical
    unique_names = [ _normalize_str(f.name) for f in selected if _normalize_str(f.name) ]
    three_distinct = (len(selected) == 3) and (len(unique_names) == 3) and (len(set(unique_names)) == 3)
    evaluator.add_custom_node(
        result=three_distinct,
        id="Three_Facilities_Identified",
        desc="Exactly three distinct colocation facilities are identified",
        parent=main_node,
        critical=True
    )

    # 3.b) Reference URLs existence, critical per facility (used as preconditions for other leaves)
    url_prereqs = await add_reference_url_nodes(evaluator, main_node, selected)

    # 3.c) Tier III Certification verification, critical per facility
    await add_tier_certification_nodes(evaluator, main_node, selected, url_prereqs)

    # 3.d) Uptime Specification (>= 99.982%), critical per facility
    await add_uptime_nodes(evaluator, main_node, selected, url_prereqs)

    # 3.e) Redundancy Specs: N+1 and concurrent maintainability, critical per facility
    await add_redundancy_nodes(evaluator, main_node, selected, url_prereqs)

    # 3.f) Qualifying City Locations (submarine landing or major IXP), critical per facility
    await add_city_location_nodes(evaluator, main_node, selected, url_prereqs)

    # 3.g) Geographic diversity, non-critical
    evaluator.add_custom_node(
        result=cities_geographically_diverse(selected),
        id="Geographic_Diversity",
        desc="The three facilities are located in different cities to provide geographic redundancy",
        parent=main_node,
        critical=False
    )

    # 4) Return summary
    return evaluator.get_summary()