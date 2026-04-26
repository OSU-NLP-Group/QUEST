import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "multi_venue_2025_2026_series"
TASK_DESCRIPTION = (
    "Identify four specific venues across different U.S. cities for a coordinated 2025-2026 entertainment event series. "
    "You must find:\n\n"
    "1. A Broadway theater located on West 44th Street in New York City with a seating capacity between 1,400 and 1,550 seats\n\n"
    "2. The stadium in California that will host Super Bowl LX in February 2026\n\n"
    "3. A stadium in Glendale, Arizona with a seating capacity exceeding 60,000 that features both a retractable roof and a retractable playing surface\n\n"
    "4. A concert arena in San Diego, California with a seating capacity between 10,000 and 15,000 seats\n\n"
    "For each venue, provide:\n"
    "- The complete venue name\n"
    "- The full street address (including street number, street name, city, state, and ZIP code where available)\n"
    "- The exact seating capacity\n"
    "- One current or upcoming event scheduled at the venue"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class Venue(BaseModel):
    """Generic venue structure extracted from the answer."""
    name: Optional[str] = None
    street_number: Optional[str] = None
    street_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip: Optional[str] = None
    capacity: Optional[str] = None
    event: Optional[str] = None
    event_date: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)


class MultiVenueExtraction(BaseModel):
    """All four venues required by the task."""
    venue1_broadway_w44_nyc: Optional[Venue] = None
    venue2_ca_superbowl_lx: Optional[Venue] = None
    venue3_glendale_retractable: Optional[Venue] = None
    venue4_sd_concert_arena: Optional[Venue] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract structured information for FOUR venues as described below. Only extract what is explicitly present in the answer; do NOT invent.

    For EACH venue, extract the following fields:
    - name: Complete venue name
    - street_number: Street number (numeric part, e.g., "225"). If not provided, return null.
    - street_name: Street name (e.g., "West 44th Street"). If not provided, return null.
    - city: City name (e.g., "New York"). If not provided, return null.
    - state: Two-letter state code (e.g., "NY", "CA", "AZ"). If not provided, return null.
    - zip: ZIP code if provided (e.g., "10036"), else null
    - capacity: Exact seating capacity as presented (keep text as-is, e.g., "1,468" or "12,414")
    - event: One current or upcoming event scheduled at the venue (title or name)
    - event_date: The event date if mentioned (any reasonable format), else null
    - sources: An array of all explicit URLs mentioned in the answer that support the venue’s information (venue pages, official event pages, Wikipedia, etc.). Extract actual URLs (markdown links should be converted to raw URLs). If none are provided, return an empty array.
    - features: A list of key phrases explicitly mentioned about the venue (e.g., "Broadway", "Manhattan Theater District", "retractable roof", "retractable playing surface", "Super Bowl LX", "Super Bowl 60", "February 8, 2026").

    Organize outputs under these keys:
    - venue1_broadway_w44_nyc
    - venue2_ca_superbowl_lx
    - venue3_glendale_retractable
    - venue4_sd_concert_arena

    If any venue is missing in the answer, set that venue to null.
    If a field is missing, set it to null (except for arrays which should be empty lists).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_capacity_to_int(cap_text: Optional[str]) -> Optional[int]:
    """
    Parse an integer seating capacity from a free-form capacity string.
    Examples handled: "1,468", "1468", "approx. 14,500", "10,000–15,000", "14k"
    Returns None if parsing fails.
    """
    if not cap_text:
        return None
    text = cap_text.strip().lower()

    # Handle 'k' suffix (e.g., "14k")
    m_k = re.search(r'(\d+(?:\.\d+)?)\s*k\b', text)
    if m_k:
        try:
            val = float(m_k.group(1))
            return int(val * 1000)
        except Exception:
            pass

    # Extract first integer-like token (allow thousands separators)
    m = re.search(r'(\d{1,3}(?:,\d{3})+|\d+)', text)
    if not m:
        return None
    num_str = m.group(1).replace(',', '')
    try:
        return int(num_str)
    except Exception:
        return None


def has_complete_address_components(v: Optional[Venue]) -> bool:
    """
    Check presence of address components: street_number, street_name, city, state.
    ZIP is optional; if provided, it must be non-empty.
    """
    if not v:
        return False
    required = [v.street_number, v.street_name, v.city, v.state]
    if any(x is None or str(x).strip() == "" for x in required):
        return False
    if v.zip is not None and str(v.zip).strip() == "":
        return False
    return True


def build_full_address(v: Optional[Venue]) -> str:
    """Construct a readable full address string for claims."""
    if not v:
        return ""
    parts = []
    if v.street_number and v.street_name:
        parts.append(f"{v.street_number} {v.street_name}")
    elif v.street_name:
        parts.append(v.street_name)
    if v.city:
        parts.append(v.city)
    if v.state:
        parts.append(v.state)
    if v.zip:
        parts.append(v.zip)
    return ", ".join(parts)


def sources_for(v: Optional[Venue]) -> List[str]:
    """Return sources list safely."""
    if not v or not v.sources:
        return []
    return v.sources


# --------------------------------------------------------------------------- #
# Venue-specific verification builders                                        #
# --------------------------------------------------------------------------- #
async def verify_venue1_broadway(
    evaluator: Evaluator,
    parent_node,
    v: Optional[Venue]
) -> None:
    """
    Venue 1: Broadway theater on West 44th Street in NYC with capacity 1,400–1,550 and required details.
    """
    venue_node = evaluator.add_parallel(
        id="venue1_broadway",
        desc="Broadway theater on West 44th Street in NYC with capacity 1,400–1,550, plus required details",
        parent=parent_node,
        critical=False
    )

    # Documentation / Sources
    doc_node = evaluator.add_parallel(
        id="v1_docs",
        desc="Supporting sources are provided for verification",
        parent=venue_node,
        critical=False
    )
    ref_node = evaluator.add_custom_node(
        result=bool(v and v.sources and len(v.sources) > 0),
        id="v1_reference",
        desc="Authoritative/verifiable source link(s) are provided supporting the venue information",
        parent=doc_node,
        critical=True
    )

    # Basic Information
    basic_node = evaluator.add_parallel(
        id="v1_basic",
        desc="Name and address information for the theater",
        parent=venue_node,
        critical=False
    )
    name_node = evaluator.add_custom_node(
        result=bool(v and v.name and v.name.strip()),
        id="v1_name",
        desc="Complete venue name is provided",
        parent=basic_node,
        critical=True
    )
    address_node = evaluator.add_parallel(
        id="v1_address",
        desc="Complete street address is provided with required components and correct street/city",
        parent=basic_node,
        critical=True
    )
    # Address components existence
    addr_components_node = evaluator.add_custom_node(
        result=has_complete_address_components(v),
        id="v1_address_components",
        desc="Address includes street number, street name, city, state, and ZIP code (where available)",
        parent=address_node,
        critical=True
    )
    # Address location check: West 44th Street in NYC
    addr_loc_leaf = evaluator.add_leaf(
        id="v1_address_on_w44_nyc",
        desc="Venue address is on West 44th Street in New York City",
        parent=address_node,
        critical=True
    )
    claim_addr_w44 = "The theater is located on West 44th Street in New York City (New York, NY)."
    await evaluator.verify(
        claim=claim_addr_w44,
        node=addr_loc_leaf,
        sources=sources_for(v),
        additional_instruction="Verify the street name contains 'West 44th Street' (or 'W 44th St') and the city is New York (NY). Allow reasonable abbreviations.",
        extra_prerequisites=[ref_node, addr_components_node, name_node]
    )

    # Constraints & Details
    constraints_node = evaluator.add_parallel(
        id="v1_constraints",
        desc="Theater-specific constraints and required per-venue details",
        parent=venue_node,
        critical=False
    )
    # Broadway definition (500+ seats & in Manhattan Theater District)
    broadway_leaf = evaluator.add_leaf(
        id="v1_broadway_definition",
        desc="Venue meets the standard definition of a Broadway theater (500+ seats and located in the Manhattan Theater District)",
        parent=constraints_node,
        critical=True
    )
    claim_broadway = ("This venue is a Broadway theater: it is located in the Manhattan Theater District and has 500 or more seats.")
    await evaluator.verify(
        claim=claim_broadway,
        node=broadway_leaf,
        sources=sources_for(v),
        additional_instruction="Use authoritative sources to confirm Broadway status. Accept standard definitions (Manhattan Theater District + >=500 seats).",
        extra_prerequisites=[ref_node, name_node]
    )

    # Capacity supported + range check
    capacity_checks_node = evaluator.add_parallel(
        id="v1_capacity_checks",
        desc="Capacity support and range verification",
        parent=constraints_node,
        critical=True
    )
    capacity_provided_node = evaluator.add_custom_node(
        result=bool(v and v.capacity and v.capacity.strip()),
        id="v1_capacity_provided",
        desc="Exact seating capacity value is provided in the answer",
        parent=capacity_checks_node,
        critical=True
    )
    capacity_supported_leaf = evaluator.add_leaf(
        id="v1_capacity_supported",
        desc="Exact seating capacity is supported by cited sources",
        parent=capacity_checks_node,
        critical=True
    )
    claim_capacity = f"The seating capacity of {v.name if v and v.name else 'the venue'} is {v.capacity}."
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_supported_leaf,
        sources=sources_for(v),
        additional_instruction="Confirm the stated seating capacity matches authoritative sources. Allow small rounding differences.",
        extra_prerequisites=[ref_node, name_node, capacity_provided_node]
    )
    # Numeric range check (1400–1550)
    cap_int = parse_capacity_to_int(v.capacity if v else None)
    capacity_in_range_node = evaluator.add_custom_node(
        result=bool(cap_int is not None and 1400 <= cap_int <= 1550),
        id="v1_capacity_in_range",
        desc="Capacity numeric value falls between 1,400 and 1,550 seats",
        parent=capacity_checks_node,
        critical=True
    )

    # Event leaf
    event_provided_node = evaluator.add_custom_node(
        result=bool(v and v.event and v.event.strip()),
        id="v1_event_provided",
        desc="An event title is provided",
        parent=constraints_node,
        critical=True
    )
    event_leaf = evaluator.add_leaf(
        id="v1_event",
        desc="At least one current or upcoming event scheduled at the venue is provided",
        parent=constraints_node,
        critical=True
    )
    claim_event = f"An event titled '{v.event}' is scheduled at {v.name}."
    await evaluator.verify(
        claim=claim_event,
        node=event_leaf,
        sources=sources_for(v),
        additional_instruction="Verify on official venue schedule or ticketing pages that the event is current or upcoming.",
        extra_prerequisites=[ref_node, name_node, event_provided_node]
    )


async def verify_venue2_ca_superbowl(
    evaluator: Evaluator,
    parent_node,
    v: Optional[Venue]
) -> None:
    """
    Venue 2: California stadium that will host Super Bowl LX in February 2026.
    """
    venue_node = evaluator.add_parallel(
        id="venue2_ca_sb",
        desc="California stadium that will host Super Bowl LX in February 2026, plus required details",
        parent=parent_node,
        critical=False
    )

    # Documentation / Sources
    doc_node = evaluator.add_parallel(
        id="v2_docs",
        desc="Supporting sources are provided for verification",
        parent=venue_node,
        critical=False
    )
    ref_node = evaluator.add_custom_node(
        result=bool(v and v.sources and len(v.sources) > 0),
        id="v2_reference",
        desc="Authoritative/verifiable source link(s) are provided supporting the venue information",
        parent=doc_node,
        critical=True
    )

    # Basic Information
    basic_node = evaluator.add_parallel(
        id="v2_basic",
        desc="Name and address information for the stadium",
        parent=venue_node,
        critical=False
    )
    name_node = evaluator.add_custom_node(
        result=bool(v and v.name and v.name.strip()),
        id="v2_name",
        desc="Complete venue name is provided",
        parent=basic_node,
        critical=True
    )
    address_node = evaluator.add_parallel(
        id="v2_address",
        desc="Complete street address is provided and is in California",
        parent=basic_node,
        critical=True
    )
    addr_components_node = evaluator.add_custom_node(
        result=has_complete_address_components(v),
        id="v2_address_components",
        desc="Address includes street number, street name, city, state, and ZIP code (where available)",
        parent=address_node,
        critical=True
    )
    addr_in_ca_leaf = evaluator.add_leaf(
        id="v2_address_in_california",
        desc="Venue address state is California (CA)",
        parent=address_node,
        critical=True
    )
    claim_addr_ca = "The stadium's address is in California (state code CA)."
    await evaluator.verify(
        claim=claim_addr_ca,
        node=addr_in_ca_leaf,
        sources=sources_for(v),
        additional_instruction="Confirm the state is California (CA).",
        extra_prerequisites=[ref_node, addr_components_node, name_node]
    )

    # Constraints & Details
    constraints_node = evaluator.add_parallel(
        id="v2_constraints",
        desc="Super Bowl hosting constraint and required per-venue details",
        parent=venue_node,
        critical=False
    )

    # Capacity supported
    capacity_checks_node = evaluator.add_parallel(
        id="v2_capacity_checks",
        desc="Capacity support verification",
        parent=constraints_node,
        critical=True
    )
    capacity_provided_node = evaluator.add_custom_node(
        result=bool(v and v.capacity and v.capacity.strip()),
        id="v2_capacity_provided",
        desc="Exact seating capacity value is provided in the answer",
        parent=capacity_checks_node,
        critical=True
    )
    capacity_supported_leaf = evaluator.add_leaf(
        id="v2_capacity_supported",
        desc="Exact seating capacity is provided and supported",
        parent=capacity_checks_node,
        critical=True
    )
    claim_capacity = f"The seating capacity of {v.name if v and v.name else 'the stadium'} is {v.capacity}."
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_supported_leaf,
        sources=sources_for(v),
        additional_instruction="Verify capacity from authoritative sources; allow minor rounding.",
        extra_prerequisites=[ref_node, name_node, capacity_provided_node]
    )

    # Super Bowl LX host confirmation
    sb_host_leaf = evaluator.add_leaf(
        id="v2_sb_lx_host",
        desc="Venue is confirmed as the host stadium for Super Bowl LX",
        parent=constraints_node,
        critical=True
    )
    claim_sb_host = f"{v.name if v and v.name else 'The stadium'} will host Super Bowl LX (Super Bowl 60)."
    await evaluator.verify(
        claim=claim_sb_host,
        node=sb_host_leaf,
        sources=sources_for(v),
        additional_instruction="Confirm official announcements for Super Bowl LX (60) host stadium.",
        extra_prerequisites=[ref_node, name_node]
    )

    # Super Bowl LX date confirmation
    sb_date_leaf = evaluator.add_leaf(
        id="v2_sb_lx_date",
        desc="Super Bowl LX date is given as February 8, 2026",
        parent=constraints_node,
        critical=True
    )
    claim_sb_date = "Super Bowl LX is scheduled for February 8, 2026."
    await evaluator.verify(
        claim=claim_sb_date,
        node=sb_date_leaf,
        sources=sources_for(v),
        additional_instruction="Confirm the stated date (Feb 8, 2026) from authoritative sources (NFL, host announcements).",
        extra_prerequisites=[ref_node]
    )

    # Event verification
    event_provided_node = evaluator.add_custom_node(
        result=bool(v and v.event and v.event.strip()),
        id="v2_event_provided",
        desc="An event title is provided",
        parent=constraints_node,
        critical=True
    )
    event_leaf = evaluator.add_leaf(
        id="v2_event",
        desc="At least one current or upcoming event scheduled at the venue is provided",
        parent=constraints_node,
        critical=True
    )
    claim_event = f"An event titled '{v.event}' is scheduled at {v.name}."
    await evaluator.verify(
        claim=claim_event,
        node=event_leaf,
        sources=sources_for(v),
        additional_instruction="Verify on official stadium schedule or ticketing pages.",
        extra_prerequisites=[ref_node, name_node, event_provided_node]
    )


async def verify_venue3_glendale(
    evaluator: Evaluator,
    parent_node,
    v: Optional[Venue]
) -> None:
    """
    Venue 3: Glendale, AZ stadium with capacity >60,000 and both retractable roof & retractable playing surface.
    """
    venue_node = evaluator.add_parallel(
        id="venue3_glendale",
        desc="Glendale, Arizona stadium with capacity >60,000 and both retractable roof and retractable playing surface, plus required details",
        parent=parent_node,
        critical=False
    )

    # Documentation / Sources
    doc_node = evaluator.add_parallel(
        id="v3_docs",
        desc="Supporting sources are provided for verification",
        parent=venue_node,
        critical=False
    )
    ref_node = evaluator.add_custom_node(
        result=bool(v and v.sources and len(v.sources) > 0),
        id="v3_reference",
        desc="Authoritative/verifiable source link(s) are provided supporting the venue information",
        parent=doc_node,
        critical=True
    )

    # Basic Information
    basic_node = evaluator.add_parallel(
        id="v3_basic",
        desc="Name and address information for the stadium",
        parent=venue_node,
        critical=False
    )
    name_node = evaluator.add_custom_node(
        result=bool(v and v.name and v.name.strip()),
        id="v3_name",
        desc="Complete venue name is provided",
        parent=basic_node,
        critical=True
    )
    address_node = evaluator.add_parallel(
        id="v3_address",
        desc="Complete street address is provided and is in Glendale, Arizona",
        parent=basic_node,
        critical=True
    )
    addr_components_node = evaluator.add_custom_node(
        result=has_complete_address_components(v),
        id="v3_address_components",
        desc="Address includes street number, street name, city, state, and ZIP code (where available)",
        parent=address_node,
        critical=True
    )
    addr_glendale_leaf = evaluator.add_leaf(
        id="v3_address_glendale_az",
        desc="Venue address is in Glendale, Arizona",
        parent=address_node,
        critical=True
    )
    claim_addr_glendale = "The stadium's address is in Glendale, Arizona (AZ)."
    await evaluator.verify(
        claim=claim_addr_glendale,
        node=addr_glendale_leaf,
        sources=sources_for(v),
        additional_instruction="Confirm city is Glendale and state is AZ.",
        extra_prerequisites=[ref_node, addr_components_node, name_node]
    )

    # Constraints & Details
    constraints_node = evaluator.add_parallel(
        id="v3_constraints",
        desc="Stadium-specific constraints and required per-venue details",
        parent=venue_node,
        critical=False
    )

    # Capacity supported + numeric threshold check
    capacity_checks_node = evaluator.add_parallel(
        id="v3_capacity_checks",
        desc="Capacity support and threshold verification",
        parent=constraints_node,
        critical=True
    )
    capacity_provided_node = evaluator.add_custom_node(
        result=bool(v and v.capacity and v.capacity.strip()),
        id="v3_capacity_provided",
        desc="Exact seating capacity value is provided in the answer",
        parent=capacity_checks_node,
        critical=True
    )
    capacity_supported_leaf = evaluator.add_leaf(
        id="v3_capacity_supported",
        desc="Exact seating capacity is supported by cited sources",
        parent=capacity_checks_node,
        critical=True
    )
    claim_capacity = f"The seating capacity of {v.name if v and v.name else 'the stadium'} is {v.capacity}."
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_supported_leaf,
        sources=sources_for(v),
        additional_instruction="Confirm capacity from authoritative sources; allow minor rounding.",
        extra_prerequisites=[ref_node, name_node, capacity_provided_node]
    )
    cap_int = parse_capacity_to_int(v.capacity if v else None)
    capacity_over_60000_node = evaluator.add_custom_node(
        result=bool(cap_int is not None and cap_int > 60000),
        id="v3_capacity_over_60000",
        desc="Capacity numeric value exceeds 60,000",
        parent=capacity_checks_node,
        critical=True
    )

    # Retractable roof
    roof_leaf = evaluator.add_leaf(
        id="v3_retractable_roof",
        desc="Venue has a retractable roof",
        parent=constraints_node,
        critical=True
    )
    claim_roof = f"{v.name if v and v.name else 'The stadium'} has a retractable roof."
    await evaluator.verify(
        claim=claim_roof,
        node=roof_leaf,
        sources=sources_for(v),
        additional_instruction="Verify structural feature (retractable roof) from authoritative venue/stadium sources.",
        extra_prerequisites=[ref_node, name_node]
    )

    # Retractable playing surface
    field_leaf = evaluator.add_leaf(
        id="v3_retractable_field",
        desc="Venue has a retractable playing surface",
        parent=constraints_node,
        critical=True
    )
    claim_field = f"{v.name if v and v.name else 'The stadium'} has a retractable playing surface (the field moves in/out)."
    await evaluator.verify(
        claim=claim_field,
        node=field_leaf,
        sources=sources_for(v),
        additional_instruction="Verify the field/playing surface can retract or roll in/out from authoritative sources.",
        extra_prerequisites=[ref_node, name_node]
    )

    # Event verification
    event_provided_node = evaluator.add_custom_node(
        result=bool(v and v.event and v.event.strip()),
        id="v3_event_provided",
        desc="An event title is provided",
        parent=constraints_node,
        critical=True
    )
    event_leaf = evaluator.add_leaf(
        id="v3_event",
        desc="At least one current or upcoming event scheduled at the venue is provided",
        parent=constraints_node,
        critical=True
    )
    claim_event = f"An event titled '{v.event}' is scheduled at {v.name}."
    await evaluator.verify(
        claim=claim_event,
        node=event_leaf,
        sources=sources_for(v),
        additional_instruction="Verify on official stadium schedule or ticketing pages.",
        extra_prerequisites=[ref_node, name_node, event_provided_node]
    )


async def verify_venue4_sd_arena(
    evaluator: Evaluator,
    parent_node,
    v: Optional[Venue]
) -> None:
    """
    Venue 4: San Diego concert arena with capacity 10,000–15,000 seats.
    """
    venue_node = evaluator.add_parallel(
        id="venue4_sd_arena",
        desc="San Diego, California concert arena with capacity 10,000–15,000, plus required details",
        parent=parent_node,
        critical=False
    )

    # Documentation / Sources
    doc_node = evaluator.add_parallel(
        id="v4_docs",
        desc="Supporting sources are provided for verification",
        parent=venue_node,
        critical=False
    )
    ref_node = evaluator.add_custom_node(
        result=bool(v and v.sources and len(v.sources) > 0),
        id="v4_reference",
        desc="Authoritative/verifiable source link(s) are provided supporting the venue information",
        parent=doc_node,
        critical=True
    )

    # Basic Information
    basic_node = evaluator.add_parallel(
        id="v4_basic",
        desc="Name and address information for the arena",
        parent=venue_node,
        critical=False
    )
    name_node = evaluator.add_custom_node(
        result=bool(v and v.name and v.name.strip()),
        id="v4_name",
        desc="Complete venue name is provided",
        parent=basic_node,
        critical=True
    )
    address_node = evaluator.add_parallel(
        id="v4_address",
        desc="Complete street address is provided and is in San Diego, California",
        parent=basic_node,
        critical=True
    )
    addr_components_node = evaluator.add_custom_node(
        result=has_complete_address_components(v),
        id="v4_address_components",
        desc="Address includes street number, street name, city, state, and ZIP code (where available)",
        parent=address_node,
        critical=True
    )
    addr_sd_leaf = evaluator.add_leaf(
        id="v4_address_san_diego_ca",
        desc="Venue address is in San Diego, California",
        parent=address_node,
        critical=True
    )
    claim_addr_sd = "The arena's address is in San Diego, California (CA)."
    await evaluator.verify(
        claim=claim_addr_sd,
        node=addr_sd_leaf,
        sources=sources_for(v),
        additional_instruction="Confirm city is San Diego and state is CA.",
        extra_prerequisites=[ref_node, addr_components_node, name_node]
    )

    # Constraints & Details
    constraints_node = evaluator.add_parallel(
        id="v4_constraints",
        desc="Arena-specific constraints and required per-venue details",
        parent=venue_node,
        critical=False
    )

    # Capacity supported + range check
    capacity_checks_node = evaluator.add_parallel(
        id="v4_capacity_checks",
        desc="Capacity support and range verification",
        parent=constraints_node,
        critical=True
    )
    capacity_provided_node = evaluator.add_custom_node(
        result=bool(v and v.capacity and v.capacity.strip()),
        id="v4_capacity_provided",
        desc="Exact seating capacity value is provided in the answer",
        parent=capacity_checks_node,
        critical=True
    )
    capacity_supported_leaf = evaluator.add_leaf(
        id="v4_capacity_supported",
        desc="Exact seating capacity is supported by cited sources",
        parent=capacity_checks_node,
        critical=True
    )
    claim_capacity = f"The seating capacity of {v.name if v and v.name else 'the arena'} is {v.capacity}."
    await evaluator.verify(
        claim=claim_capacity,
        node=capacity_supported_leaf,
        sources=sources_for(v),
        additional_instruction="Confirm capacity from authoritative sources; allow minor rounding.",
        extra_prerequisites=[ref_node, name_node, capacity_provided_node]
    )
    cap_int = parse_capacity_to_int(v.capacity if v else None)
    capacity_in_range_node = evaluator.add_custom_node(
        result=bool(cap_int is not None and 10000 <= cap_int <= 15000),
        id="v4_capacity_in_range",
        desc="Capacity numeric value falls between 10,000 and 15,000 seats",
        parent=capacity_checks_node,
        critical=True
    )

    # Event verification
    event_provided_node = evaluator.add_custom_node(
        result=bool(v and v.event and v.event.strip()),
        id="v4_event_provided",
        desc="An event title is provided",
        parent=constraints_node,
        critical=True
    )
    event_leaf = evaluator.add_leaf(
        id="v4_event",
        desc="At least one current or upcoming event scheduled at the venue is provided",
        parent=constraints_node,
        critical=True
    )
    claim_event = f"An event titled '{v.event}' is scheduled at {v.name}."
    await evaluator.verify(
        claim=claim_event,
        node=event_leaf,
        sources=sources_for(v),
        additional_instruction="Verify on official arena schedule or ticketing pages.",
        extra_prerequisites=[ref_node, name_node, event_provided_node]
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
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
) -> Dict[str, Any]:
    """
    Entry point to evaluate the agent's answer for the four-venue identification task.
    """
    # Initialize evaluator (root is non-critical by default to allow partial credit)
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

    # Extract all venues
    extraction = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=MultiVenueExtraction,
        extraction_name="venues_extraction"
    )

    # Build top-level aggregation node for clarity (non-critical, parallel)
    multi_node = evaluator.add_parallel(
        id="multi_venue_identification",
        desc="Identify four venues that satisfy the specified constraints and provide required details for each",
        parent=root,
        critical=False
    )

    # Venue 1
    await verify_venue1_broadway(
        evaluator=evaluator,
        parent_node=multi_node,
        v=extraction.venue1_broadway_w44_nyc
    )

    # Venue 2
    await verify_venue2_ca_superbowl(
        evaluator=evaluator,
        parent_node=multi_node,
        v=extraction.venue2_ca_superbowl_lx
    )

    # Venue 3
    await verify_venue3_glendale(
        evaluator=evaluator,
        parent_node=multi_node,
        v=extraction.venue3_glendale_retractable
    )

    # Venue 4
    await verify_venue4_sd_arena(
        evaluator=evaluator,
        parent_node=multi_node,
        v=extraction.venue4_sd_concert_arena
    )

    # Return evaluation summary
    return evaluator.get_summary()