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
TASK_ID = "denver_comedy_club_2025"
TASK_DESCRIPTION = (
    "Identify a comedy club in Denver, Colorado that meets all of the following requirements: "
    "(1) has a seating capacity between 300 and 400 seats, "
    "(2) charges ticket prices in the $100-$150 range for premium comedy performances, "
    "(3) hosted at least one nationally recognized comedian during the 2025 calendar year, and "
    "(4) meets ADA accessibility requirements with at least 5% accessible seating. "
    "Provide the venue's name, exact address, phone number, the name of a nationally recognized comedian who "
    "performed there in 2025, and the specific performance dates."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    # Core venue identity
    venue_name: Optional[str] = None
    venue_website: Optional[str] = None

    # Location and contact
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    phone: Optional[str] = None

    # Requirements
    seating_capacity: Optional[str] = None                  # free-form, e.g., "350", "300-400", "~350"
    premium_ticket_price: Optional[str] = None              # free-form, e.g., "$125", "$100–$150", "about $120+ fees"
    accessibility_seating_percent: Optional[str] = None     # e.g., "5%", "at least 5%"
    ada_statement: Optional[str] = None                     # free-form summary from answer
    venue_type_description: Optional[str] = None            # e.g., "comedy club", "comedy theater", etc.

    # 2025 performance
    comedian_name_2025: Optional[str] = None
    performance_dates_2025: List[str] = Field(default_factory=list)

    # Source URLs explicitly cited in the answer
    general_sources: List[str] = Field(default_factory=list)
    location_sources: List[str] = Field(default_factory=list)
    address_sources: List[str] = Field(default_factory=list)
    phone_sources: List[str] = Field(default_factory=list)
    capacity_sources: List[str] = Field(default_factory=list)
    pricing_sources: List[str] = Field(default_factory=list)
    ada_sources: List[str] = Field(default_factory=list)
    venue_type_sources: List[str] = Field(default_factory=list)
    comedian_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
Extract all the following information exactly as presented in the answer text. Do not invent any values.

Required fields:
1) venue_name: The specific venue name.
2) venue_website: The official venue website URL if explicitly cited.
3) address: The exact street address as presented.
4) city: The city as presented (if present).
5) state: The state or state abbreviation as presented (if present).
6) phone: The venue phone number for ticket inquiries as presented.
7) seating_capacity: Any seating capacity value or range mentioned (free text).
8) premium_ticket_price: Any price or price range mentioned for premium comedy performances (free text).
9) accessibility_seating_percent: The stated accessible seating percentage if mentioned (e.g., "5%") (free text).
10) ada_statement: Any explicit statement about ADA accessibility, accessible seating, wheelchair access, etc. (free text).
11) venue_type_description: A short phrase describing the venue type (e.g., "comedy club", "comedy theater", etc.) if present.
12) comedian_name_2025: The name of a nationally recognized comedian said to have performed there in 2025.
13) performance_dates_2025: A list of the specific performance date strings mentioned for the 2025 show(s) (e.g., ["June 12, 2025", "June 13, 2025"]).

Also extract all source URLs explicitly cited in the answer for each category (only include URLs explicitly present in the answer):
- general_sources: Any general URLs about the venue or shows cited in the answer.
- location_sources: URLs that support city/state/location.
- address_sources: URLs that show the street address.
- phone_sources: URLs that show the venue's phone number.
- capacity_sources: URLs that state or imply the seating capacity.
- pricing_sources: URLs that show premium ticket prices (ideally ticketing pages or official announcements).
- ada_sources: URLs that mention ADA accessibility or accessible seating details (ideally % or seating map/policy).
- venue_type_sources: URLs that indicate the venue type (comedy club/theater) or describe it as such.
- comedian_sources: URLs that show the named comedian performed at this venue in 2025 (e.g., event pages) and/or establish the comedian's national recognition (e.g., Wikipedia, major press).

Rules:
- If a field is not present in the answer, set it to null (or an empty list for arrays).
- Only include valid URLs that are explicitly present in the answer text (plain or markdown links).
- Do not transform values; keep them as they appear (for prices/capacity/dates).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*lists: Optional[List[str]], also_include: Optional[List[str]] = None) -> List[str]:
    seen: set = set()
    merged: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                merged.append(url)
    if also_include:
        for url in also_include:
            if url and url not in seen:
                seen.add(url)
                merged.append(url)
    return merged


def as_list_if_present(url: Optional[str]) -> List[str]:
    return [url] if url else []


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_venue_identification(evaluator: Evaluator, root, data: VenueExtraction) -> None:
    evaluator.add_custom_node(
        result=bool(data.venue_name and data.venue_name.strip()),
        id="venue_identification",
        desc="The specific venue name is provided",
        parent=root,
        critical=True
    )


async def build_location_verification(evaluator: Evaluator, root, data: VenueExtraction) -> None:
    node = evaluator.add_sequential(
        id="location_verification",
        desc="The venue is located in Denver, Colorado",
        parent=root,
        critical=True
    )

    loc_sources = combine_sources(
        data.location_sources,
        data.address_sources,
        data.general_sources,
        also_include=as_list_if_present(data.venue_website)
    )

    evaluator.add_custom_node(
        result=len(loc_sources) > 0,
        id="location_sources_provided",
        desc="At least one source URL is provided for location verification",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="location_in_denver",
        desc="Verify: Venue is located in Denver, Colorado",
        parent=node,
        critical=True
    )
    venue = data.venue_name or "the venue"
    claim = f"{venue} is located in Denver, Colorado (Denver, CO). Neighborhoods within Denver city limits still count as Denver."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=loc_sources,
        additional_instruction="Confirm the venue's city/state is Denver, Colorado. Equivalent abbreviations like 'Denver, CO' count."
    )


async def build_venue_type(evaluator: Evaluator, root, data: VenueExtraction) -> None:
    node = evaluator.add_sequential(
        id="venue_type",
        desc="The venue is a dedicated comedy club or comedy theater, not a multi-purpose arena",
        parent=root,
        critical=True
    )
    vt_sources = combine_sources(
        data.venue_type_sources,
        data.general_sources,
        also_include=as_list_if_present(data.venue_website)
    )
    evaluator.add_custom_node(
        result=len(vt_sources) > 0,
        id="venue_type_sources_provided",
        desc="At least one source URL indicates/depicts the venue type",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="venue_type_verify",
        desc="Verify: Venue is a dedicated comedy club or comedy theater (not a multi-purpose arena/stadium)",
        parent=node,
        critical=True
    )
    venue = data.venue_name or "the venue"
    claim = f"{venue} is a dedicated comedy club or comedy theater, and is not a large multi-purpose arena or stadium."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=vt_sources,
        additional_instruction="Use the venue's About/FAQ or official descriptions. If it seats only a few hundred and is branded as a comedy club/theater, it qualifies."
    )


async def build_capacity_requirement(evaluator: Evaluator, root, data: VenueExtraction) -> None:
    node = evaluator.add_sequential(
        id="capacity_requirement",
        desc="The venue's seating capacity is between 300 and 400 seats",
        parent=root,
        critical=True
    )
    cap_sources = combine_sources(
        data.capacity_sources,
        data.general_sources,
        also_include=as_list_if_present(data.venue_website)
    )
    evaluator.add_custom_node(
        result=len(cap_sources) > 0,
        id="capacity_sources_provided",
        desc="At least one capacity-related source URL is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="capacity_range_verify",
        desc="Verify: Seating capacity is between 300 and 400 (inclusive)",
        parent=node,
        critical=True
    )
    venue = data.venue_name or "the venue"
    claim = f"The seating capacity of {venue} is between 300 and 400 seats (inclusive)."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=cap_sources,
        additional_instruction="Accept approximate phrasing like 'about 350'. If multiple rooms, consider the primary performance room used for headline comedy shows."
    )


async def build_accessibility_compliance(evaluator: Evaluator, root, data: VenueExtraction) -> None:
    node = evaluator.add_sequential(
        id="accessibility_compliance",
        desc="The venue meets ADA accessibility requirements with at least 5% accessible seating",
        parent=root,
        critical=True
    )
    ada_sources = combine_sources(
        data.ada_sources,
        data.general_sources,
        also_include=as_list_if_present(data.venue_website)
    )
    evaluator.add_custom_node(
        result=len(ada_sources) > 0,
        id="ada_sources_provided",
        desc="At least one ADA/accessibility-related source URL is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="ada_verify",
        desc="Verify: At least 5% of seats are accessible; ADA requirement is met",
        parent=node,
        critical=True
    )
    venue = data.venue_name or "the venue"
    claim = f"{venue} meets ADA accessibility requirements and has at least 5% of its seats designated as accessible or wheelchair seating."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=ada_sources,
        additional_instruction="Look for ADA policy pages, seating maps, or official statements specifying accessible seating percentage. If percent is explicitly >=5%, pass."
    )


async def build_premium_pricing(evaluator: Evaluator, root, data: VenueExtraction) -> None:
    node = evaluator.add_sequential(
        id="premium_pricing",
        desc="Ticket prices for premium comedy performances are within the $100-$150 range",
        parent=root,
        critical=True
    )
    price_sources = combine_sources(
        data.pricing_sources,
        data.general_sources,
        also_include=as_list_if_present(data.venue_website)
    )
    evaluator.add_custom_node(
        result=len(price_sources) > 0,
        id="pricing_sources_provided",
        desc="At least one pricing/ticketing source URL is provided",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="pricing_verify",
        desc="Verify: Premium comedy tickets fall within $100–$150 (inclusive)",
        parent=node,
        critical=True
    )
    venue = data.venue_name or "the venue"
    claim = f"At {venue}, premium comedy performance tickets are priced within the $100 to $150 range (inclusive) for at least one show."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=price_sources,
        additional_instruction="Use official ticketing/venue pages. Look for VIP/front-row/premium options priced between $100 and $150 before fees; inclusive of fees is also acceptable."
    )


async def build_performance_history(evaluator: Evaluator, root, data: VenueExtraction) -> None:
    ph_node = evaluator.add_parallel(
        id="performance_history",
        desc="Information about a nationally recognized comedian who performed at the venue in 2025",
        parent=root,
        critical=True
    )

    # Require at least one relevant source for performance/notability
    perf_sources = combine_sources(
        data.comedian_sources,
        data.general_sources,
        also_include=as_list_if_present(data.venue_website)
    )
    evaluator.add_custom_node(
        result=len(perf_sources) > 0,
        id="performance_sources_provided",
        desc="At least one URL source is provided for the 2025 performance/notability",
        parent=ph_node,
        critical=True
    )

    # Comedian name + national recognition (interpretation consistent with requirement)
    comedian_name = data.comedian_name_2025 or ""
    comedian_name_node = evaluator.add_leaf(
        id="comedian_name",
        desc="The name of a nationally recognized comedian who performed at the venue during 2025 is provided",
        parent=ph_node,
        critical=True
    )
    comedian_claim = (
        f"{comedian_name} is a nationally recognized comedian in the United States (e.g., known through major tours, "
        f"TV specials, or mainstream press)."
    )
    await evaluator.verify(
        claim=comedian_claim,
        node=comedian_name_node,
        sources=perf_sources,
        additional_instruction="Use reputable sources (e.g., Wikipedia, major media, official bios) to judge national recognition."
    )

    # Performance dates verification for 2025 at the specified venue
    dates_str = "; ".join(data.performance_dates_2025) if data.performance_dates_2025 else ""
    perf_dates_node = evaluator.add_leaf(
        id="performance_dates",
        desc="The specific performance dates for the comedian's 2025 show(s) at the venue are provided",
        parent=ph_node,
        critical=True
    )
    venue = data.venue_name or "the venue"
    perf_claim = (
        f"{comedian_name} performed at {venue} in 2025 on the following date(s): {dates_str}."
    )
    await evaluator.verify(
        claim=perf_claim,
        node=perf_dates_node,
        sources=perf_sources,
        additional_instruction="Confirm the event listing or schedule shows this comedian at this venue in 2025 on the specified dates. Allow reasonable date formatting variants."
    )


async def build_address_verification(evaluator: Evaluator, root, data: VenueExtraction) -> None:
    node = evaluator.add_sequential(
        id="venue_address",
        desc="The venue's exact street address is provided",
        parent=root,
        critical=True
    )

    # Require the address value itself
    evaluator.add_custom_node(
        result=bool(data.address and data.address.strip()),
        id="address_value_provided",
        desc="An exact street address string is present in the answer",
        parent=node,
        critical=True
    )

    addr_sources = combine_sources(
        data.address_sources,
        data.general_sources,
        data.location_sources,
        also_include=as_list_if_present(data.venue_website)
    )
    evaluator.add_custom_node(
        result=len(addr_sources) > 0,
        id="address_sources_provided",
        desc="At least one URL source is provided for the street address",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="address_verify",
        desc="Verify: The provided street address matches the source(s)",
        parent=node,
        critical=True
    )
    claim = f"The venue's street address is exactly: {data.address or ''}"
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=addr_sources,
        additional_instruction="Compare the full street address, including number, street name, and unit/suite if any. Minor formatting/casing differences are acceptable."
    )


async def build_phone_verification(evaluator: Evaluator, root, data: VenueExtraction) -> None:
    node = evaluator.add_sequential(
        id="venue_phone",
        desc="The venue's phone number for ticket inquiries is provided",
        parent=root,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(data.phone and data.phone.strip()),
        id="phone_value_provided",
        desc="A phone number string is present in the answer",
        parent=node,
        critical=True
    )

    phone_sources = combine_sources(
        data.phone_sources,
        data.general_sources,
        also_include=as_list_if_present(data.venue_website)
    )
    evaluator.add_custom_node(
        result=len(phone_sources) > 0,
        id="phone_sources_provided",
        desc="At least one URL source is provided for the phone number",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="phone_verify",
        desc="Verify: The provided phone number matches the source(s)",
        parent=node,
        critical=True
    )
    claim = f"The venue's phone number is: {data.phone or ''}"
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=phone_sources,
        additional_instruction="Allow flexible formatting (dashes, spaces, parentheses, country code). Focus on matching the digits."
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Aggregate all required checks in parallel; all are critical
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

    # Extraction
    extracted: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Build verification tree per rubric
    await build_venue_identification(evaluator, root, extracted)
    await build_location_verification(evaluator, root, extracted)
    await build_venue_type(evaluator, root, extracted)
    await build_capacity_requirement(evaluator, root, extracted)
    await build_accessibility_compliance(evaluator, root, extracted)
    await build_premium_pricing(evaluator, root, extracted)
    await build_performance_history(evaluator, root, extracted)
    await build_address_verification(evaluator, root, extracted)
    await build_phone_verification(evaluator, root, extracted)

    # Return standard summary
    return evaluator.get_summary()