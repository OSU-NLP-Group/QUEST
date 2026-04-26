import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nashville_venues_2026"
TASK_DESCRIPTION = (
    "A film production company is planning a two-part premiere event series in Nashville, Tennessee for their new "
    "documentary in 2026. They need to identify two different indoor performing arts venues that meet the following "
    "specifications:\n\n"
    "Venue 1: An indoor theater or performing arts venue with a seating capacity between 1,000 and 1,500 seats.\n\n"
    "Venue 2: An indoor theater or performing arts venue with a seating capacity between 2,000 and 3,000 seats.\n\n"
    "Both venues must:\n"
    "- Be located in Nashville, Tennessee\n"
    "- Be performing arts, theater, or entertainment venues (not sports arenas or outdoor amphitheaters)\n"
    "- Offer venue rental services for private events\n"
    "- Have documented seating capacity information from official sources\n\n"
    "For each venue, provide: the venue name, exact seating capacity, venue type description, and a reference URL from "
    "an official or reliable source that confirms the capacity and rental availability."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtractionItem(BaseModel):
    name: Optional[str] = None
    location_city: Optional[str] = None  # e.g., "Nashville, TN" or "Nashville, Tennessee"
    capacity: Optional[str] = None       # As written in the answer (e.g., "1,298", "about 2,300")
    capacity_number: Optional[str] = None  # Digits-only form if given (e.g., "1298")
    capacity_urls: List[str] = Field(default_factory=list)  # URLs that confirm capacity
    type_description: Optional[str] = None  # e.g., "performing arts center", "indoor theater"
    type_urls: List[str] = Field(default_factory=list)      # URLs that confirm venue type/indoor
    rental_description: Optional[str] = None                # e.g., "offers private event rentals"
    rental_urls: List[str] = Field(default_factory=list)    # URLs that confirm rental availability
    location_urls: List[str] = Field(default_factory=list)  # URLs that confirm Nashville location


class VenuesExtraction(BaseModel):
    venue1: Optional[VenueExtractionItem] = None
    venue2: Optional[VenueExtractionItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract the information the answer provides for exactly two indoor performing arts venues in Nashville, Tennessee.
    The venues are intended for a two-part premiere event series.

    For each venue in the answer, extract the following fields:

    - name: The venue's name as written in the answer.
    - location_city: The city/state string (e.g., "Nashville, TN" or "Nashville, Tennessee") if mentioned.
    - capacity: The exact seating capacity text as written in the answer (if any). Keep it as free text.
    - capacity_number: Digits-only capacity if a single exact capacity is clearly stated (e.g., "1298"). If unclear or a range, return null.
    - capacity_urls: All URLs cited that can confirm the seating capacity (official site, operator, or other reliable sources).
    - type_description: The venue type as written (e.g., "performing arts center", "indoor concert hall", "theater").
    - type_urls: All URLs cited that can confirm the venue type and indoor nature.
    - rental_description: Any text indicating the venue offers rental or private event hosting (if present).
    - rental_urls: All URLs cited that can confirm rental/private event availability.
    - location_urls: All URLs cited that can confirm the Nashville, Tennessee location.

    IMPORTANT:
    - Extract only URLs explicitly present in the answer. Do not invent or infer URLs.
    - If a single cited URL supports multiple aspects (e.g., capacity and rental), include it in the corresponding URL lists (capacity_urls, rental_urls, type_urls, location_urls).
    - If an aspect is mentioned but no URL is provided in the answer for that aspect, leave that URL list empty.
    - If an aspect is not mentioned, set its field to null (for strings) or an empty list (for URLs).

    Return a JSON object with two top-level objects: venue1 and venue2. If the answer lists more than two venues, choose the first two presented.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize_urls(urls: Optional[List[str]]) -> List[str]:
    """Normalize and deduplicate provided URLs."""
    if not urls:
        return []
    seen = set()
    out: List[str] = []
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u:
            continue
        # Very lightweight validity check; keep as-is to allow site to load
        if not (u.startswith("http://") or u.startswith("https://")):
            # Prefix protocol if missing as per extraction special rule
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _union_sources(*lists: Optional[List[str]]) -> List[str]:
    """Union lists of URLs with deduplication."""
    merged: List[str] = []
    seen = set()
    for lst in lists:
        for u in _normalize_urls(lst or []):
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _parse_capacity_number(item: VenueExtractionItem) -> Optional[int]:
    """
    Try to parse an integer seat count from capacity_number first; if not present,
    attempt to parse from free-text 'capacity'.
    """
    # Prefer capacity_number if clean digits are provided
    if item.capacity_number:
        digits = re.sub(r"[^\d]", "", item.capacity_number)
        if digits.isdigit():
            try:
                return int(digits)
            except Exception:
                pass

    # Fallback: parse the first plausible integer from the capacity free text
    if item.capacity:
        # Find sequences like "1,234" or "1234"
        candidates = re.findall(r"\d{1,3}(?:,\d{3})+|\d+", item.capacity)
        if candidates:
            # Choose the largest number found to avoid catching suite numbers, etc.
            try:
                values = [int(c.replace(",", "")) for c in candidates]
                return max(values) if values else None
            except Exception:
                return None
    return None


def _safe_name(name: Optional[str], fallback: str) -> str:
    name = (name or "").strip()
    return name if name else fallback


async def _verify_with_sources_or_fail(
    evaluator: Evaluator,
    claim: str,
    node: VerificationNode,
    sources: List[str],
    additional_instruction: str,
) -> bool:
    """
    Helper to enforce source-grounding for leaf verifications:
    - If sources are present, call evaluator.verify normally.
    - If sources are absent, directly mark the node as failed.
    """
    if sources:
        return await evaluator.verify(
            claim=claim,
            node=node,
            sources=sources,
            additional_instruction=additional_instruction,
        )
    else:
        node.score = 0.0
        node.status = "failed"
        return False


# --------------------------------------------------------------------------- #
# Verification logic per venue                                                #
# --------------------------------------------------------------------------- #
async def verify_one_venue(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    venue: Optional[VenueExtractionItem],
    is_first: bool,
    cap_min: int,
    cap_max: int,
) -> VerificationNode:
    """
    Build and evaluate the verification sub-tree for one venue.
    """
    group_id = "Medium_Capacity_Venue" if is_first else "Large_Capacity_Venue"
    group_desc = (
        "Verify that Venue 1 is correctly identified with capacity between 1,000-1,500 seats and meets all requirements"
        if is_first else
        "Verify that Venue 2 is correctly identified with capacity between 2,000-3,000 seats and meets all requirements"
    )

    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=False  # allow partial credit across the two venues
    )

    # Defensive default object if missing
    v = venue or VenueExtractionItem()

    # Prepare sources unions for robust checks
    location_sources = _union_sources(v.location_urls, v.capacity_urls, v.type_urls, v.rental_urls)
    capacity_sources = _union_sources(v.capacity_urls, v.type_urls, v.location_urls)
    type_sources = _union_sources(v.type_urls, v.capacity_urls, v.location_urls)
    rental_sources = _union_sources(v.rental_urls, v.capacity_urls, v.type_urls, v.location_urls)

    venue_label = "Venue1" if is_first else "Venue2"
    name = _safe_name(v.name, f"Unnamed {venue_label}")

    # 1) Basic Information (critical)
    basic_node = evaluator.add_parallel(
        id=f"{venue_label}_Basic_Information",
        desc="Verify that basic venue identification information is provided",
        parent=group_node,
        critical=True
    )

    # 1.a) Name provided (existence check)
    evaluator.add_custom_node(
        result=bool(v.name and v.name.strip()),
        id=f"{venue_label}_Name_Provided",
        desc="The venue name is provided",
        parent=basic_node,
        critical=True
    )

    # 1.b) Location verified (Nashville, Tennessee)
    loc_leaf = evaluator.add_leaf(
        id=f"{venue_label}_Location",
        desc="The venue is located in Nashville, Tennessee",
        parent=basic_node,
        critical=True
    )
    await _verify_with_sources_or_fail(
        evaluator,
        claim=f"The venue named '{name}' is located in Nashville, Tennessee (Nashville, TN).",
        node=loc_leaf,
        sources=location_sources,
        additional_instruction="Confirm that the page indicates the venue is in Nashville, Tennessee (accept variations like 'Nashville, TN' or addresses in Nashville)."
    )

    # 1.c) Location reference presence
    evaluator.add_custom_node(
        result=bool(location_sources),
        id=f"{venue_label}_Location_Reference",
        desc="A reference URL is provided that confirms the Nashville location",
        parent=basic_node,
        critical=True
    )

    # 2) Capacity requirements (critical)
    capacity_node = evaluator.add_parallel(
        id=f"{venue_label}_Capacity_Requirements",
        desc="Verify that the venue's capacity meets the specified range",
        parent=group_node,
        critical=True
    )

    # 2.a) Capacity in range (existence + numeric check)
    numeric_cap = _parse_capacity_number(v)
    in_range = numeric_cap is not None and (cap_min <= numeric_cap <= cap_max)
    evaluator.add_custom_node(
        result=in_range,
        id=f"{venue_label}_Capacity_In_Range",
        desc=f"The stated capacity is between {cap_min} and {cap_max} seats (inclusive)",
        parent=capacity_node,
        critical=True
    )

    # 2.b) Capacity documented (must be supported by a URL)
    cap_doc_leaf = evaluator.add_leaf(
        id=f"{venue_label}_Capacity_Documented",
        desc="The capacity figure is documented with a reference URL",
        parent=capacity_node,
        critical=True
    )
    capacity_text = (v.capacity or (str(numeric_cap) if numeric_cap is not None else None))
    if capacity_text:
        await _verify_with_sources_or_fail(
            evaluator,
            claim=f"The seating capacity of '{name}' is {capacity_text} seats (or an equivalent figure).",
            node=cap_doc_leaf,
            sources=capacity_sources,
            additional_instruction="Verify that the cited page explicitly states an overall seating capacity near or equal to this figure. Accept minor wording like 'seats', 'capacity of', or 'approximately'."
        )
    else:
        # No capacity text to verify against sources => fail
        cap_doc_leaf.score = 0.0
        cap_doc_leaf.status = "failed"

    # 3) Facility type (critical)
    type_node = evaluator.add_parallel(
        id=f"{venue_label}_Facility_Type",
        desc="Verify that the venue is an appropriate performing arts facility",
        parent=group_node,
        critical=True
    )

    # 3.a) Indoor venue
    indoor_leaf = evaluator.add_leaf(
        id=f"{venue_label}_Indoor_Venue",
        desc="The venue is confirmed to be an indoor facility",
        parent=type_node,
        critical=True
    )
    await _verify_with_sources_or_fail(
        evaluator,
        claim=f"'{name}' is an indoor venue (i.e., a closed building, not an outdoor amphitheater).",
        node=indoor_leaf,
        sources=type_sources,
        additional_instruction="Confirm from the page that the venue is an indoor theater/concert hall/performing arts space (not an outdoor amphitheater). Descriptions of auditoriums, theaters, indoor seating, or interior halls are acceptable."
    )

    # 3.b) Performing arts / theater / entertainment (not sports arena or outdoor amphitheater)
    pa_leaf = evaluator.add_leaf(
        id=f"{venue_label}_Performing_Arts",
        desc="The venue is identified as a performing arts, theater, or entertainment venue (not a sports arena or outdoor amphitheater)",
        parent=type_node,
        critical=True
    )
    await _verify_with_sources_or_fail(
        evaluator,
        claim=f"'{name}' is a performing arts, theater, or entertainment venue (not primarily a sports arena and not an outdoor amphitheater).",
        node=pa_leaf,
        sources=type_sources,
        additional_instruction="Verify the page indicates it's a theater, concert hall, performance venue, or similar. If it is mainly a sports arena or an outdoor amphitheater, do not support the claim."
    )

    # 3.c) Type reference presence
    evaluator.add_custom_node(
        result=bool(type_sources),
        id=f"{venue_label}_Type_Reference",
        desc="A reference URL is provided that confirms the venue type",
        parent=type_node,
        critical=True
    )

    # 4) Rental availability (critical)
    rental_node = evaluator.add_parallel(
        id=f"{venue_label}_Rental_Availability",
        desc="Verify that the venue offers rental services for private events",
        parent=group_node,
        critical=True
    )

    # 4.a) Rental confirmed
    rental_leaf = evaluator.add_leaf(
        id=f"{venue_label}_Rental_Confirmed",
        desc="Information is provided indicating the venue offers rental services for private events",
        parent=rental_node,
        critical=True
    )
    await _verify_with_sources_or_fail(
        evaluator,
        claim=f"'{name}' offers rental services for private events (e.g., event rentals, venue bookings, host your event).",
        node=rental_leaf,
        sources=rental_sources,
        additional_instruction="Confirm the page indicates venue rentals, private event bookings, or a rentals/contact form for events."
    )

    # 4.b) Rental reference presence
    evaluator.add_custom_node(
        result=bool(rental_sources),
        id=f"{venue_label}_Rental_Reference",
        desc="A reference URL is provided that confirms rental availability",
        parent=rental_node,
        critical=True
    )

    return group_node


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
    Entry point for the Nashville venues evaluation.
    """
    # Initialize evaluator (root is non-critical by framework design)
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

    # Extract structured information for two venues
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Build verification subtrees for Venue 1 and Venue 2
    await verify_one_venue(
        evaluator=evaluator,
        parent_node=root,
        venue=extraction.venue1,
        is_first=True,
        cap_min=1000,
        cap_max=1500,
    )

    await verify_one_venue(
        evaluator=evaluator,
        parent_node=root,
        venue=extraction.venue2,
        is_first=False,
        cap_min=2000,
        cap_max=3000,
    )

    # Final check: Different venues
    v1_name = _safe_name(extraction.venue1.name if extraction.venue1 else None, "Venue 1")
    v2_name = _safe_name(extraction.venue2.name if extraction.venue2 else None, "Venue 2")

    diff_leaf = evaluator.add_leaf(
        id="Different_Venues",
        desc="The two venues identified are distinct facilities with different names (not the same venue in different configurations)",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim=f"The two venues '{v1_name}' and '{v2_name}' are different facilities (not the same venue in different configurations).",
        node=diff_leaf,
        additional_instruction="Judge based on the names as provided. Consider minor spelling/casing variations; if the names clearly refer to the same venue, mark as incorrect."
    )

    # Optional: Add custom info about numeric capacities parsed
    info_caps = {
        "venue1_parsed_capacity": _parse_capacity_number(extraction.venue1) if extraction.venue1 else None,
        "venue2_parsed_capacity": _parse_capacity_number(extraction.venue2) if extraction.venue2 else None,
    }
    evaluator.add_custom_info(info_caps, info_type="parsed_capacities")

    # Return evaluation summary
    return evaluator.get_summary()