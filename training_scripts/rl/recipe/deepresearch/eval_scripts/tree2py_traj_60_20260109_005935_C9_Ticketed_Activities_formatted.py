import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tour_venue_planning_2026"
TASK_DESCRIPTION = (
    "A major music artist is planning a 2026 national concert tour across the United States and needs to identify suitable indoor arena venues in four different geographic regions: "
    "Northeast, Southeast, Midwest, and West Coast. For each region, identify one major indoor arena venue that meets all professional requirements for hosting large-scale ticketed "
    "concert events (capacity, ADA, safety, security, insurance, box office, egress, restrooms, sound/noise, parking, accessible routes). Provide the venue name, city location, "
    "specific seating capacity, and reference URLs that verify the venue meets these requirements."
)

REGION_LABELS = {
    "northeast": "Northeast",
    "southeast": "Southeast",
    "midwest": "Midwest",
    "west_coast": "West Coast",
}

# Region guidance text for simple verification (geography only)
REGION_GUIDANCE = (
    "Use common U.S. regional groupings to judge whether the provided location belongs to the stated region, without needing a source URL:\n"
    "- Northeast: ME, NH, VT, MA, RI, CT, NY, NJ, PA\n"
    "- Southeast: AL, AR, FL, GA, KY, LA, MD, MS, NC, SC, TN, VA, DC, WV\n"
    "- Midwest: IL, IN, IA, KS, MI, MN, MO, NE, ND, OH, SD, WI\n"
    "- West Coast: CA, OR, WA"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None  # Prefer "City, State" if provided by the answer; keep as free-form string
    capacity: Optional[str] = None  # Concert configuration capacity as a string (e.g., "20,000", "approx. 18,500")
    urls: List[str] = Field(default_factory=list)  # All reference URLs cited for this venue


class VenuesExtraction(BaseModel):
    northeast: Optional[VenueItem] = None
    southeast: Optional[VenueItem] = None
    midwest: Optional[VenueItem] = None
    west_coast: Optional[VenueItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return (
        "From the answer, extract one proposed indoor arena venue per region for the 2026 concert tour planning. "
        "Regions: Northeast, Southeast, Midwest, West Coast.\n"
        "For each region, extract:\n"
        "- name: The venue name (arena)\n"
        "- city: City and state (if given) where the venue is located (free-form text)\n"
        "- capacity: A specific concert seating capacity value as mentioned in the answer (string). If multiple capacities are given, prefer the concert configuration. "
        "If the answer provides only a vague descriptor without a number (e.g., 'large'), capacity should be null.\n"
        "- urls: All reference URLs explicitly cited in the answer for that venue. Include all relevant links (venue official pages, accessibility/policies/A‑Z guides, event info, etc.).\n\n"
        "If the answer lists multiple venues for a region, select the first one mentioned. If a region is missing in the answer, set the whole region object to null. "
        "Do not invent any URLs or details.\n"
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_specific_number(text: Optional[str]) -> bool:
    """Return True if the text contains at least one digit, indicating a specific numeric value was provided."""
    if not text:
        return False
    return bool(re.search(r"\d", text))


def dedup_urls(urls: List[str]) -> List[str]:
    """Deduplicate URLs while preserving order."""
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def safe_get_region_item(extraction: VenuesExtraction, region_key: str) -> VenueItem:
    """Return a VenueItem for a region; if None, return an empty placeholder."""
    item = getattr(extraction, region_key, None)
    if item is None:
        return VenueItem()
    item.urls = dedup_urls(item.urls)
    return item


# --------------------------------------------------------------------------- #
# Verification construction per region                                        #
# --------------------------------------------------------------------------- #
async def verify_region_venue(
    evaluator: Evaluator,
    parent_node,
    region_key: str,
    region_label: str,
    venue: VenueItem,
) -> None:
    """
    Build verification subtree for a single region (parallel aggregation).
    Structure:
      - Region node (parallel, non-critical)
        - Venue_Basic_Info (parallel, critical)
            * Venue_Name (custom, critical)
            * Venue_City_Location (custom, critical)
            * Seating_Capacity_Value (custom, critical)
            * Reference_URLs_Present (custom, critical)
            * Indoor_Arena_Confirmed (leaf, critical) -> verify by URLs
            * Region_Match (leaf, critical) -> simple verify (knowledge allowed)
        - Must_Requirements_Compliance (parallel, critical)
            * 13 critical leaves, each verified with URLs
        - Should_Requirements (parallel, non-critical)
            * 2 non-critical leaves, verified with URLs
    """

    # Region root
    region_node = evaluator.add_parallel(
        id=f"{region_key}_venue",
        desc=f"Venue proposed for the {region_label} region.",
        parent=parent_node,
        critical=False,  # Allow partial credit across regions
    )

    # -------------------- Venue_Basic_Info --------------------
    basic_node = evaluator.add_parallel(
        id=f"{region_key}_basic_info",
        desc="Provide the required basic venue information.",
        parent=region_node,
        critical=True,
    )

    # Presence checks (custom nodes -> immediate pass/fail)
    evaluator.add_custom_node(
        result=bool(venue.name and venue.name.strip()),
        id=f"{region_key}_venue_name",
        desc="Venue name is provided.",
        parent=basic_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(venue.city and venue.city.strip()),
        id=f"{region_key}_venue_city",
        desc="City location is provided.",
        parent=basic_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=has_specific_number(venue.capacity),
        id=f"{region_key}_capacity_value",
        desc="A specific concert seating capacity value is provided (not just 'large'/'approximate' without a number).",
        parent=basic_node,
        critical=True,
    )

    urls_present = bool(venue.urls and len(venue.urls) > 0)
    evaluator.add_custom_node(
        result=urls_present,
        id=f"{region_key}_ref_urls_present",
        desc="At least one reference URL is provided for the venue to support verification of requirements.",
        parent=basic_node,
        critical=True,
    )

    # Indoor arena confirmation (by URLs)
    indoor_leaf = evaluator.add_leaf(
        id=f"{region_key}_indoor_arena",
        desc="Venue is an indoor arena (enclosed indoor facility).",
        parent=basic_node,
        critical=True,
    )
    indoor_claim = (
        f"The venue '{venue.name}' is an indoor arena (an enclosed indoor facility), not an outdoor stadium."
    )
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_leaf,
        sources=venue.urls,
        additional_instruction="Look for explicit phrases like 'indoor arena', 'indoor venue', 'enclosed', or building description that clearly indicates an indoor arena.",
    )

    # Region match (simple verify; allow geography knowledge)
    region_leaf = evaluator.add_leaf(
        id=f"{region_key}_region_match",
        desc=f"Venue is located within the {region_label} region of the United States.",
        parent=basic_node,
        critical=True,
    )
    region_claim = (
        f"The venue location '{venue.city}' is considered part of the {region_label} region of the United States."
    )
    await evaluator.verify(
        claim=region_claim,
        node=region_leaf,
        sources=None,
        additional_instruction=REGION_GUIDANCE,
    )

    # -------------------- Must_Requirements_Compliance --------------------
    must_node = evaluator.add_parallel(
        id=f"{region_key}_must",
        desc="Venue meets all stated MUST requirements (with support from the provided references).",
        parent=region_node,
        critical=True,
    )

    # 1) Capacity >= 15,000
    cap_leaf = evaluator.add_leaf(
        id=f"{region_key}_capacity_min_15000",
        desc="Concert seating capacity is at least 15,000.",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The concert seating capacity of '{venue.name}' is at least 15,000.",
        node=cap_leaf,
        sources=venue.urls,
        additional_instruction="Use a credible capacity statement. If multiple capacities are given (e.g., basketball vs concert), focus on concert configuration where available.",
    )

    # 2) ADA wheelchair seating >= 1%
    ada_min_leaf = evaluator.add_leaf(
        id=f"{region_key}_ada_wheelchair_min_1pct",
        desc="Wheelchair accessible seating meets ADA requirement (minimum 1% of total capacity).",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Wheelchair accessible seating at '{venue.name}' is at least 1% of total capacity, consistent with ADA standards.",
        node=ada_min_leaf,
        sources=venue.urls,
        additional_instruction="Look for ADA seating policies, accessible seating counts, or compliance statements indicating at least 1% accessible seating.",
    )

    # 3) ADA wheelchair seating dispersed
    ada_disp_leaf = evaluator.add_leaf(
        id=f"{region_key}_ada_wheelchair_dispersed",
        desc="Wheelchair accessible seating is dispersed throughout the venue at multiple levels/sections.",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Wheelchair accessible seating at '{venue.name}' is dispersed across multiple sections and/or levels.",
        node=ada_disp_leaf,
        sources=venue.urls,
        additional_instruction="Seek venue accessibility/A‑Z guides stating distribution of accessible seating in different sections/levels.",
    )

    # 4) ADA accessible parking ratio
    ada_parking_leaf = evaluator.add_leaf(
        id=f"{region_key}_ada_accessible_parking_ratio",
        desc="ADA-compliant accessible parking meets minimum ratio (≥1 accessible space per 25 total parking spaces).",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{venue.name}' provides ADA-compliant accessible parking at a ratio of at least 1 accessible space per 25 total spaces.",
        node=ada_parking_leaf,
        sources=venue.urls,
        additional_instruction="Look for venue parking and accessibility pages describing ADA parking counts or compliance with ADA ratios.",
    )

    # 5) Fire alarm system
    fire_alarm_leaf = evaluator.add_leaf(
        id=f"{region_key}_fire_alarm_system",
        desc="Automatic fire alarm system is present as required for assembly occupancies of 300+ persons.",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{venue.name}' has an automatic fire alarm system appropriate for an assembly occupancy over 300 people.",
        node=fire_alarm_leaf,
        sources=venue.urls,
        additional_instruction="Accept venue/building policy documents, safety plans, code compliance summaries, or official sources that indicate alarm systems.",
    )

    # 6) Sprinkler system
    sprinkler_leaf = evaluator.add_leaf(
        id=f"{region_key}_sprinkler_system",
        desc="Automatic fire sprinkler system is present as required for relevant assembly occupancies of 300+ persons.",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{venue.name}' has an automatic fire sprinkler system appropriate for a large assembly occupancy.",
        node=sprinkler_leaf,
        sources=venue.urls,
        additional_instruction="Look for building features, life-safety descriptions, or code compliance statements indicating sprinklers.",
    )

    # 7) Security staffing capability (1 officer per 100–150 attendees)
    security_leaf = evaluator.add_leaf(
        id=f"{region_key}_security_staffing",
        desc="Venue is capable of providing adequate security staffing (approx. 1 officer per 100–150 attendees).",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{venue.name}' can provide adequate security staffing around 1 officer per 100–150 attendees for large music events.",
        node=security_leaf,
        sources=venue.urls,
        additional_instruction="Seek venue/event planning guides, security policies, or promoter handbooks specifying staffing guidelines/ratios.",
    )

    # 8) Liability insurance requirement
    insurance_leaf = evaluator.add_leaf(
        id=f"{region_key}_liability_insurance",
        desc="Venue requires event organizers to carry at least $1M per occurrence / $2M aggregate general liability insurance.",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Event organizers/promoters at '{venue.name}' must carry minimum $1,000,000 per occurrence and $2,000,000 aggregate general liability insurance.",
        node=insurance_leaf,
        sources=venue.urls,
        additional_instruction="Look for venue rental/policies or promoter guides specifying minimum insurance requirements.",
    )

    # 9) Box office operates >= 3 hours before event start
    box_office_leaf = evaluator.add_leaf(
        id=f"{region_key}_box_office_3h",
        desc="Venue operates a box office at least 3 hours before event start time on event days.",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The box office at '{venue.name}' operates at least 3 hours before event start time on event days.",
        node=box_office_leaf,
        sources=venue.urls,
        additional_instruction="Check box office hours, ticketing policies, or event-day procedures. If hours vary, the claim must be explicitly supported.",
    )

    # 10) Emergency evacuation capacity
    egress_leaf = evaluator.add_leaf(
        id=f"{region_key}_emergency_evacuation_capacity",
        desc="Venue has adequate emergency evacuation capacity meeting building code requirements.",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{venue.name}' provides emergency egress capacity that meets applicable building code requirements for its occupancy.",
        node=egress_leaf,
        sources=venue.urls,
        additional_instruction="Look for life safety, emergency procedures, building code compliance summaries, or official statements indicating adequate egress capacity.",
    )

    # 11) Restroom facilities code-compliant
    restroom_leaf = evaluator.add_leaf(
        id=f"{region_key}_restrooms_code",
        desc="Venue has adequate restroom facilities complying with building codes for its occupancy load.",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{venue.name}' has restroom facilities adequate for its occupancy and compliant with building code requirements.",
        node=restroom_leaf,
        sources=venue.urls,
        additional_instruction="Accept venue specifications, code compliance notes, or official facility descriptions indicating code-compliant restrooms.",
    )

    # 12) Accessible routes from parking to seating
    accessible_routes_leaf = evaluator.add_leaf(
        id=f"{region_key}_accessible_routes",
        desc="Venue provides accessible routes from parking areas to seating areas.",
        parent=must_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"'{venue.name}' provides ADA-compliant accessible routes from parking areas to seating locations.",
        node=accessible_routes_leaf,
        sources=venue.urls,
        additional_instruction="Use accessibility pages or policies describing elevators, ramps, routes from parking to seating.",
    )

    # -------------------- Should_Requirements (non-critical) --------------------
    should_node = evaluator.add_parallel(
        id=f"{region_key}_should",
        desc="Venue satisfies stated SHOULD (preference) requirements when supported by references.",
        parent=region_node,
        critical=False,
    )

    # Sound system and noise compliance (non-critical)
    sound_leaf = evaluator.add_leaf(
        id=f"{region_key}_sound_noise_compliance",
        desc="Venue should have a professional sound system that complies with local noise regulations.",
        parent=should_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"'{venue.name}' operates professional sound reinforcement and complies with applicable local noise regulations/ordinances.",
        node=sound_leaf,
        sources=venue.urls,
        additional_instruction="Look for venue technical specs, production guides, or local policy mentions about noise compliance.",
    )

    # Adequate general parking availability (non-critical)
    parking_leaf = evaluator.add_leaf(
        id=f"{region_key}_general_parking",
        desc="Venue should have adequate general parking availability for large events.",
        parent=should_node,
        critical=False,
    )
    await evaluator.verify(
        claim=f"'{venue.name}' provides adequate general parking capacity appropriate for large events.",
        node=parking_leaf,
        sources=venue.urls,
        additional_instruction="Check venue parking pages, maps, or event-day parking guidance indicating substantial parking availability.",
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
    Evaluate a single answer for the 2026 tour venue planning task.
    1) Extract venues (one per region) with basic info and URLs.
    2) Verify basic info, MUST requirements, and SHOULD preferences per region.
    3) Aggregate results into the verification tree summary.
    """

    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Regions are independent; allow partial credit
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Add custom info (optional context)
    evaluator.add_custom_info(
        info={
            "regions_expected": list(REGION_LABELS.values()),
            "evaluation_focus": [
                "basic info presence and indoor/region checks",
                "must requirements verified by URLs",
                "should preferences verified by URLs",
            ],
        },
        info_type="eval_meta",
        info_name="evaluation_configuration",
    )

    # Build verification subtrees per region
    for region_key, region_label in REGION_LABELS.items():
        venue_item = safe_get_region_item(extracted, region_key)
        await verify_region_venue(
            evaluator=evaluator,
            parent_node=root,
            region_key=region_key,
            region_label=region_label,
            venue=venue_item,
        )

    # Return final summary
    return evaluator.get_summary()