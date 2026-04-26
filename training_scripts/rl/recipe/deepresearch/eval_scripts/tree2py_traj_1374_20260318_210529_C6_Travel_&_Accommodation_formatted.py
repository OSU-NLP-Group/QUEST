import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ritz_carlton_yacht_2027_mlk"
TASK_DESCRIPTION = (
    "Identify a yacht from The Ritz-Carlton Yacht Collection that meets all of the following criteria for booking a luxury Caribbean cruise during the Martin Luther King Jr. Day holiday weekend in January 2027: "
    "(1) The yacht must have a passenger capacity between 200 and 500 guests at double occupancy; "
    "(2) The yacht must maintain a crew-to-passenger ratio of at least 1:2 (meaning at least one crew member for every two passengers) to ensure ultra-luxury service standards; "
    "(3) All accommodations on the yacht must be suites (no standard cabins), with each suite featuring a private terrace or balcony; "
    "(4) Each suite must be at least 350 square feet in size, including the terrace or balcony area; "
    "(5) The yacht must be classified as ultra-luxury tier; "
    "(6) The yacht must be operational and offering cruises in January 2027. "
    "Provide the name of the yacht and verify that it meets all the specified requirements with supporting references."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class YachtSelection(BaseModel):
    # Identification and fleet membership
    yacht_name: Optional[str] = None
    fleet_urls: List[str] = Field(default_factory=list)

    # Capacity and crew
    capacity_text: Optional[str] = None  # e.g., "298 guests (double occupancy)"
    capacity_urls: List[str] = Field(default_factory=list)

    crew_text: Optional[str] = None  # e.g., "246 crew" or "nearly 1:1 ratio"
    crew_urls: List[str] = Field(default_factory=list)

    # Suites and accommodations
    all_suites_text: Optional[str] = None  # e.g., "all-suite yacht"
    private_terrace_text: Optional[str] = None  # e.g., "all suites have private terraces"
    min_suite_size_text: Optional[str] = None  # e.g., "Terrace Suite 298 sq ft + 64 sq ft terrace"
    suites_urls: List[str] = Field(default_factory=list)

    # Classification and operations
    classification_text: Optional[str] = None  # e.g., "ultra-luxury"
    classification_urls: List[str] = Field(default_factory=list)

    jan2027_text: Optional[str] = None  # e.g., itinerary or schedule mention
    jan2027_urls: List[str] = Field(default_factory=list)

    # Itinerary region (optional, non-critical per rubric)
    caribbean_text: Optional[str] = None
    caribbean_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_yacht_selection() -> str:
    return """
    From the provided answer, extract the single yacht (if any) that the answer proposes from The Ritz-Carlton Yacht Collection and all supporting details and URLs that the answer cites.

    Required fields:
    - yacht_name: The name of the yacht (e.g., "Evrima", "Ilma", or "Luminara"). If no yacht name is given, set to null.

    URLs: Extract the actual URLs explicitly present in the answer text for each category below. Return an empty list if none are provided.
    - fleet_urls: URLs that show the yacht is part of The Ritz-Carlton Yacht Collection fleet.
    - capacity_urls: URLs that state guest capacity (preferably double occupancy) for the yacht.
    - crew_urls: URLs that state the crew count and/or an explicit crew-to-guest ratio.
    - suites_urls: URLs that describe suite configuration, private terraces/balconies, and sizes.
    - classification_urls: URLs that describe the yacht/brand as ultra-luxury or equivalent tier.
    - jan2027_urls: URLs that show the yacht is operating or offering cruises in January 2027 (e.g., schedules or itineraries).
    - caribbean_urls: URLs that show the yacht offers Caribbean itineraries.

    Text fields (free-form, as written in the answer, if present; otherwise null):
    - capacity_text
    - crew_text
    - all_suites_text
    - private_terrace_text
    - min_suite_size_text
    - classification_text
    - jan2027_text
    - caribbean_text

    Rules:
    - Extract only what is explicitly present in the answer; do not infer or add new URLs or facts.
    - If a URL appears in markdown format [text](url), return only the url.
    - If a field is not mentioned, return null (for single text fields) or [] (for URL lists).
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _name_or_placeholder(name: Optional[str]) -> str:
    return name if (name and name.strip()) else "the yacht"


def _merge_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            u2 = (u or "").strip()
            if u2 and u2 not in seen:
                merged.append(u2)
                seen.add(u2)
    return merged


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_basic_eligibility(evaluator: Evaluator, parent, sel: YachtSelection):
    node = evaluator.add_parallel(
        id="Basic_Eligibility",
        desc="Verification that the yacht belongs to The Ritz-Carlton Yacht Collection",
        parent=parent,
        critical=True,
    )

    # Fleet membership claim
    fleet_leaf = evaluator.add_leaf(
        id="Fleet_Membership",
        desc="The yacht must be part of The Ritz-Carlton Yacht Collection",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The yacht named '{_name_or_placeholder(sel.yacht_name)}' is part of The Ritz-Carlton Yacht Collection fleet.",
        node=fleet_leaf,
        sources=sel.fleet_urls,
        additional_instruction="Prefer official pages (ritzcarltonyachtcollection.com). If the page clearly states the yacht is part of the brand's fleet, consider it supported.",
    )

    # URL reference for eligibility
    url_elig_leaf = evaluator.add_leaf(
        id="URL_Reference_Eligibility",
        desc="URL reference verifying fleet membership",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"At least one of the provided sources is an official Ritz-Carlton Yacht Collection page about '{_name_or_placeholder(sel.yacht_name)}'.",
        node=url_elig_leaf,
        sources=sel.fleet_urls,
        additional_instruction="A page on ritzcarltonyachtcollection.com that mentions the yacht by name suffices.",
    )


async def build_capacity_standards(evaluator: Evaluator, parent, sel: YachtSelection):
    node = evaluator.add_parallel(
        id="Capacity_Standards",
        desc="Verification of passenger capacity and crew-to-guest service ratio requirements",
        parent=parent,
        critical=True,
    )

    # Guest Capacity (Sequential)
    cap_seq = evaluator.add_sequential(
        id="Guest_Capacity",
        desc="Passenger capacity requirements for intimate luxury experience",
        parent=node,
        critical=True,
    )

    # Double occupancy range 200-500 inclusive
    cap_leaf = evaluator.add_leaf(
        id="Double_Occupancy_Range",
        desc="Passenger capacity at double occupancy must be between 200 and 500 guests",
        parent=cap_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The passenger capacity at double occupancy for '{_name_or_placeholder(sel.yacht_name)}' is between 200 and 500 guests (inclusive). "
              f"If a page lists 'guests' without qualifier, treat that as marketed double-occupancy capacity.",
        node=cap_leaf,
        sources=sel.capacity_urls,
        additional_instruction="Use the official ship facts or reputable sources. If both 'maximum passengers' and 'guests' appear, prioritize the standard 'guests' marketed figure.",
    )

    # URL Reference for capacity
    cap_ref_leaf = evaluator.add_leaf(
        id="URL_Reference_Capacity",
        desc="URL reference for passenger capacity data",
        parent=cap_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"At least one provided page states the '{_name_or_placeholder(sel.yacht_name)}' guest capacity (double occupancy or standard marketed capacity).",
        node=cap_ref_leaf,
        sources=sel.capacity_urls,
        additional_instruction="Look for terms like 'guests', 'double occupancy', or explicit capacity numbers.",
    )

    # Service Ratio (Sequential)
    svc_seq = evaluator.add_sequential(
        id="Service_Ratio",
        desc="Crew-to-passenger ratio requirements for ultra-luxury service",
        parent=node,
        critical=True,
    )

    ratio_leaf = evaluator.add_leaf(
        id="Crew_To_Passenger_Ratio",
        desc="Crew-to-passenger ratio must be at least 1:2 (0.5 or better)",
        parent=svc_seq,
        critical=True,
    )
    ratio_sources = _merge_urls(sel.crew_urls, sel.capacity_urls)
    await evaluator.verify(
        claim=f"'{_name_or_placeholder(sel.yacht_name)}' maintains a crew-to-passenger ratio of at least 1:2 (0.5 or higher), "
              f"as supported by the published crew size and guest capacity.",
        node=ratio_leaf,
        sources=ratio_sources,
        additional_instruction="If both crew count and guest capacity are provided, the ratio can be derived (crew/guests >= 0.5). "
                               "Also accept phrases like 'nearly 1:1' or 'better than 1:2' as sufficient.",
    )

    crew_ref_leaf = evaluator.add_leaf(
        id="URL_Reference_Crew",
        desc="URL reference for crew count and ratio data",
        parent=svc_seq,
        critical=True,
    )
    await evaluator.verify(
        claim=f"At least one provided page gives either the crew count or an explicit crew-to-guest ratio for '{_name_or_placeholder(sel.yacht_name)}'.",
        node=crew_ref_leaf,
        sources=ratio_sources,
        additional_instruction="The source may list a crew number, crew-to-guest ratio, or both.",
    )


async def build_suite_specifications(evaluator: Evaluator, parent, sel: YachtSelection):
    node = evaluator.add_parallel(
        id="Suite_Specifications",
        desc="Verification of suite-only configuration and accommodation standards",
        parent=parent,
        critical=True,
    )

    # All-suite configuration
    all_suite_leaf = evaluator.add_leaf(
        id="All_Suite_Configuration",
        desc="All accommodations must be suites (no standard cabins)",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"All accommodations aboard '{_name_or_placeholder(sel.yacht_name)}' are suites (no standard or interior cabins).",
        node=all_suite_leaf,
        sources=sel.suites_urls,
        additional_instruction="Look for phrases like 'all-suite yacht' or descriptions indicating no standard cabins.",
    )

    # Private terrace/balcony for each suite
    terrace_leaf = evaluator.add_leaf(
        id="Private_Outdoor_Spaces",
        desc="Each suite must have a private terrace or balcony",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Every suite aboard '{_name_or_placeholder(sel.yacht_name)}' features a private terrace or balcony.",
        node=terrace_leaf,
        sources=sel.suites_urls,
        additional_instruction="The page should indicate that all suites include a private terrace or balcony.",
    )

    # Minimum suite size threshold (including terrace)
    size_leaf = evaluator.add_leaf(
        id="Minimum_Suite_Size",
        desc="Each suite must be at least 350 square feet including the terrace/balcony area",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The smallest suite aboard '{_name_or_placeholder(sel.yacht_name)}' is at least 350 square feet in total (interior plus private terrace/balcony).",
        node=size_leaf,
        sources=sel.suites_urls,
        additional_instruction="Check the smallest suite category; if the size is quoted as interior + terrace/balcony, sum them. "
                               "If a page states a total size including the terrace that meets or exceeds 350 sq ft, it suffices.",
    )

    suites_ref_leaf = evaluator.add_leaf(
        id="URL_Reference_Suites",
        desc="URL reference for suite specifications and configurations",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"At least one provided page lists suite specifications for '{_name_or_placeholder(sel.yacht_name)}' including private terraces/balconies and suite sizes.",
        node=suites_ref_leaf,
        sources=sel.suites_urls,
        additional_instruction="Accept a combination page that describes suite features and provides size details.",
    )


async def build_classification_and_availability(evaluator: Evaluator, parent, sel: YachtSelection):
    node = evaluator.add_parallel(
        id="Classification_And_Availability",
        desc="Verification of luxury classification, operational availability, and itinerary suitability",
        parent=parent,
        critical=True,
    )

    # Ultra-luxury classification
    class_leaf = evaluator.add_leaf(
        id="Ultra_Luxury_Tier",
        desc="The vessel must be classified as ultra-luxury or luxury yacht collection tier",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{_name_or_placeholder(sel.yacht_name)}' is classified/marketed as ultra-luxury as part of The Ritz-Carlton Yacht Collection.",
        node=class_leaf,
        sources=sel.classification_urls,
        additional_instruction="Look for phrases like 'ultra-luxury', 'luxury yacht collection', or brand positioning indicating ultra-luxury.",
    )

    # January 2027 operations/availability
    ops_leaf = evaluator.add_leaf(
        id="January_2027_Operations",
        desc="The yacht must be operational and offering cruises in January 2027",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{_name_or_placeholder(sel.yacht_name)}' has published itineraries or cruises operating in January 2027.",
        node=ops_leaf,
        sources=sel.jan2027_urls,
        additional_instruction="A schedule or itinerary page showing any January 2027 departure for this yacht suffices.",
    )

    class_ref_leaf = evaluator.add_leaf(
        id="URL_Reference_Classification",
        desc="URL reference for luxury classification and January 2027 availability",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"At least one of the provided pages either describes '{_name_or_placeholder(sel.yacht_name)}' (or the brand) as ultra-luxury or lists a January 2027 departure for it.",
        node=class_ref_leaf,
        sources=_merge_urls(sel.classification_urls, sel.jan2027_urls),
        additional_instruction="Either classification evidence or January 2027 schedule evidence satisfies this URL reference requirement.",
    )


async def build_caribbean_itineraries_optional(evaluator: Evaluator, parent, sel: YachtSelection):
    # Non-critical, optional confirmation of Caribbean region suitability
    carib_leaf = evaluator.add_leaf(
        id="Caribbean_Itineraries",
        desc="The yacht should offer Caribbean itineraries",
        parent=parent,
        critical=False,
    )
    await evaluator.verify(
        claim=f"'{_name_or_placeholder(sel.yacht_name)}' offers Caribbean itineraries.",
        node=carib_leaf,
        sources=sel.caribbean_urls,
        additional_instruction="Any itinerary page indicating Caribbean ports/regions suffices (e.g., Eastern/Western/Southern Caribbean).",
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
    # Initialize evaluator (root node id is 'root' internally)
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

    # Extract structured selection and URLs
    selection: YachtSelection = await evaluator.extract(
        prompt=prompt_extract_yacht_selection(),
        template_class=YachtSelection,
        extraction_name="yacht_selection",
    )

    # Create top-level main node to mirror rubric naming
    main = evaluator.add_parallel(
        id="Yacht_Identification",
        desc="Identification and verification of an ultra-luxury yacht from The Ritz-Carlton Yacht Collection suitable for booking during MLK Day 2027 weekend",
        parent=root,
        critical=False,  # Non-critical container; criticality enforced at child groups
    )

    # Build verification subtrees
    await build_basic_eligibility(evaluator, main, selection)
    await build_capacity_standards(evaluator, main, selection)
    await build_suite_specifications(evaluator, main, selection)
    await build_classification_and_availability(evaluator, main, selection)
    await build_caribbean_itineraries_optional(evaluator, main, selection)  # Non-critical

    # Optional: record task requirement info
    evaluator.add_custom_info(
        info={
            "required_capacity_range_double_occupancy": "200-500 guests (inclusive)",
            "min_crew_to_passenger_ratio": ">= 1:2 (0.5)",
            "suite_requirements": {
                "all_accommodations_are_suites": True,
                "each_suite_has_private_terrace_or_balcony": True,
                "each_suite_min_total_size_sqft": 350,
            },
            "classification_required": "ultra-luxury",
            "operations_month_required": "January 2027",
            "optional_region": "Caribbean itineraries (non-critical)",
        },
        info_type="task_requirements",
        info_name="task_requirements",
    )

    return evaluator.get_summary()