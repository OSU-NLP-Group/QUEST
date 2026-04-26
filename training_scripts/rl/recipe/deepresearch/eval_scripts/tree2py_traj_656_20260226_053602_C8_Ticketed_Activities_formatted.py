import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "multi_city_tour_venues"
TASK_DESCRIPTION = (
    "A concert tour promoter is planning a multi-city tour across four different U.S. regions and needs to identify one suitable venue in each region that meets specific capacity and amenity requirements for ticketed events. "
    "Identify four venues meeting the following specifications: "
    "(1) West Coast Large Arena Venue: An indoor arena located in California with a seating capacity between 15,000 and 25,000 that offers VIP packages including reserved seating and exclusive hospitality areas. "
    "(2) Southwest Mid-Size Venue: A concert-suitable venue (arena or amphitheater) located in Arizona with a seating capacity between 5,000 and 15,000 that offers VIP ticket packages with premium amenities. "
    "(3) Midwest Large Stadium Venue: A stadium located in Ohio with a seating capacity of 40,000 or more that is suitable for large-scale concerts and offers VIP or premium seating options. "
    "(4) Northeast Theater Venue: A Broadway theater located in New York with a minimum seating capacity of 500 seats (meeting Broadway classification standards, not Off-Broadway or Off-Off-Broadway) that offers premium or VIP seating options. "
    "For each venue, provide the venue name, specific location (city), exact seating capacity, venue type, and details about available VIP or premium ticket packages."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string; we will parse numerically later
    venue_type: Optional[str] = None
    vip_details: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class TourVenuesExtraction(BaseModel):
    west_coast: Optional[VenueInfo] = None      # California indoor arena (15k-25k) with VIP packages (reserved seating + hospitality)
    southwest: Optional[VenueInfo] = None       # Arizona arena/amphitheater (5k-15k) with VIP premium amenities
    midwest: Optional[VenueInfo] = None         # Ohio stadium (>=40k) suitable for large-scale concerts w/ VIP/premium seating
    northeast: Optional[VenueInfo] = None       # New York Broadway theater (>=500) classified Broadway (not Off) with VIP/premium seating


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract information for exactly four venues corresponding to the four regional specifications described in the answer. 
    Map the first clearly identified example for each region to the following fields:
    - west_coast: California indoor arena (15,000–25,000 capacity) with VIP packages that include reserved seating and exclusive hospitality.
    - southwest: Arizona arena or amphitheater (5,000–15,000 capacity) with VIP ticket packages with premium amenities.
    - midwest: Ohio stadium (40,000+ capacity) suitable for large-scale concerts and offers VIP/premium seating options.
    - northeast: New York Broadway theater (500+ capacity), explicitly classified as Broadway (not Off-/Off-Off-), offering premium or VIP seating.
    
    For each region, extract these fields:
    1) name: Venue name (as stated in the answer)
    2) city: City name
    3) state: State name or two-letter abbreviation if provided
    4) capacity: The exact seating capacity number as written in the answer (keep units/text if present; do not convert to number)
    5) venue_type: The venue type phrased in the answer (e.g., indoor arena, amphitheater, stadium, Broadway theater)
    6) vip_details: A brief description of VIP/premium ticket offerings as presented in the answer (e.g., reserved seating, suites, lounges, hospitality)
    7) urls: A list of all URLs cited in the answer that specifically support this venue (official venue pages, ticketing pages, or trusted sources). 
       Only include URLs explicitly present in the answer; if none are provided, return an empty list.
    
    If any individual field is missing for a region, set it to null (or empty list for urls).
    If the answer provides multiple candidates for a region, pick the first one mentioned for that region.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_capacity_int(capacity_str: Optional[str]) -> Optional[int]:
    """
    Parse a human-written capacity into an integer:
    - Handles commas, spaces, decimals, and 'k' suffix (e.g., 18k -> 18000, 18.5k -> 18500).
    - Returns None if no reasonable number found.
    """
    if not capacity_str:
        return None
    s = capacity_str.strip()
    # Match a number possibly with commas or decimals, optional 'k' suffix
    m = re.search(r'(\d{1,3}(?:[,\s]\d{3})+|\d+(?:\.\d+)?)(?:\s*(k|K))?', s)
    if not m:
        return None
    num_str = m.group(1)
    k_suffix = m.group(2)
    try:
        num = float(num_str.replace(",", "").replace(" ", ""))
    except ValueError:
        return None
    if k_suffix:
        num *= 1000
    return int(round(num))


def build_additional_instruction(urls: List[str], extra_instructions: str) -> str:
    """
    Build an additional instruction for the verifier, enforcing source-grounding.
    If no URLs are provided, instruct the judge to mark as not supported.
    """
    base = extra_instructions.strip()
    if urls and len(urls) > 0:
        tail = (
            "\nUse only the provided URL(s) as evidence. Prefer official venue pages, ticketing pages, or other reputable sources. "
            "Do not rely on your own external knowledge."
        )
    else:
        tail = (
            "\nImportant: The answer did not provide any supporting URL for this venue. "
            "Per evaluation policy, if no specific source URL is provided to verify a factual claim, you must mark the claim as NOT SUPPORTED/Incorrect."
        )
    return base + tail


async def add_leaf_and_verify(
    evaluator: Evaluator,
    node_id: str,
    desc: str,
    parent_node,
    claim: str,
    urls: Optional[List[str]],
    add_ins: str,
    critical: bool = True
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical
    )
    sources_arg = urls if (urls and len(urls) > 0) else None
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources_arg,
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Region verification functions                                               #
# --------------------------------------------------------------------------- #
async def verify_west_coast(evaluator: Evaluator, root_node, info: Optional[VenueInfo]) -> None:
    node = evaluator.add_parallel(
        id="West_Coast_Large_Arena_Venue",
        desc="Evaluate the West Coast venue meeting large arena specifications",
        parent=root_node,
        critical=False
    )

    # Existence checks (critical)
    name_ok = bool(info and info.name and info.name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="WC_Venue_Name_Provided",
        desc="Answer must provide the specific name of the venue",
        parent=node,
        critical=True
    )

    city_ok = bool(info and info.city and info.city.strip())
    evaluator.add_custom_node(
        result=city_ok,
        id="WC_City_Location_Provided",
        desc="Answer must provide the specific city location of the venue",
        parent=node,
        critical=True
    )

    cap_num = parse_capacity_int(info.capacity if info else None)
    evaluator.add_custom_node(
        result=(cap_num is not None),
        id="WC_Exact_Capacity_Provided",
        desc="Answer must provide the exact seating capacity number",
        parent=node,
        critical=True
    )

    urls = info.urls if info else []

    # State location (California)
    venue_name = info.name if info and info.name else "the venue"
    city_part = f"{info.city}, " if info and info.city else ""
    claim_state = f"{venue_name} is located in {city_part}California."
    add_ins_state = build_additional_instruction(
        urls,
        "Confirm that the venue is in the state of California (CA). City must be in California; abbreviations like 'CA' are acceptable."
    )
    await add_leaf_and_verify(
        evaluator, "WC_State_Location", "Venue must be located in California",
        node, claim_state, urls, add_ins_state, critical=True
    )

    # Capacity range 15,000–25,000
    if cap_num is not None:
        claim_cap = f"The seating capacity of {venue_name} is {cap_num}, and it falls between 15,000 and 25,000 seats (inclusive)."
    else:
        claim_cap = f"The seating capacity of {venue_name} is between 15,000 and 25,000 seats."
    add_ins_cap = build_additional_instruction(
        urls,
        "Verify the venue's stated seating capacity on the cited page. Accept the standard/event seating capacity for concerts. "
        "If multiple figures exist, use the seating capacity for events; it must be within 15,000–25,000 seats (inclusive)."
    )
    await add_leaf_and_verify(
        evaluator, "WC_Capacity_Range", "Venue must have a capacity between 15,000 and 25,000 seats",
        node, claim_cap, urls, add_ins_cap, critical=True
    )

    # Venue type: indoor arena
    claim_type = f"{venue_name} is an indoor arena."
    add_ins_type = build_additional_instruction(
        urls,
        "Verify that the venue is explicitly an indoor arena (e.g., 'indoor multi-purpose arena'). "
        "Do not accept outdoor amphitheaters or open-air venues."
    )
    await add_leaf_and_verify(
        evaluator, "WC_Venue_Type", "Venue must be an indoor arena",
        node, claim_type, urls, add_ins_type, critical=True
    )

    # VIP package: reserved seating + exclusive hospitality
    claim_vip = (
        f"{venue_name} offers VIP packages that include reserved seating and exclusive hospitality areas "
        "(e.g., lounges, clubs, suites, premium hospitality)."
    )
    add_ins_vip = build_additional_instruction(
        urls,
        "Confirm that VIP packages include BOTH reserved seating and some form of exclusive hospitality area "
        "(e.g., VIP lounges, clubs, suites, premium clubs). If only early entry or generic perks are listed without reserved seating/hospitality, do not accept."
    )
    await add_leaf_and_verify(
        evaluator, "WC_VIP_Package",
        "Venue must offer VIP packages with reserved seating and exclusive hospitality areas",
        node, claim_vip, urls, add_ins_vip, critical=True
    )


async def verify_southwest(evaluator: Evaluator, root_node, info: Optional[VenueInfo]) -> None:
    node = evaluator.add_parallel(
        id="Southwest_Mid_Size_Venue",
        desc="Evaluate the Southwest venue meeting mid-size specifications",
        parent=root_node,
        critical=False
    )

    # Existence checks (critical)
    name_ok = bool(info and info.name and info.name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="SW_Venue_Name_Provided",
        desc="Answer must provide the specific name of the venue",
        parent=node,
        critical=True
    )

    city_ok = bool(info and info.city and info.city.strip())
    evaluator.add_custom_node(
        result=city_ok,
        id="SW_City_Location_Provided",
        desc="Answer must provide the specific city location of the venue",
        parent=node,
        critical=True
    )

    cap_num = parse_capacity_int(info.capacity if info else None)
    evaluator.add_custom_node(
        result=(cap_num is not None),
        id="SW_Exact_Capacity_Provided",
        desc="Answer must provide the exact seating capacity number",
        parent=node,
        critical=True
    )

    urls = info.urls if info else []
    venue_name = info.name if info and info.name else "the venue"
    city_part = f"{info.city}, " if info and info.city else ""

    # State location (Arizona)
    claim_state = f"{venue_name} is located in {city_part}Arizona."
    add_ins_state = build_additional_instruction(
        urls,
        "Confirm that the venue is in the state of Arizona (AZ). City must be in Arizona; abbreviations like 'AZ' are acceptable."
    )
    await add_leaf_and_verify(
        evaluator, "SW_State_Location", "Venue must be located in Arizona",
        node, claim_state, urls, add_ins_state, critical=True
    )

    # Capacity range 5,000–15,000
    if cap_num is not None:
        claim_cap = f"The seating capacity of {venue_name} is {cap_num}, and it falls between 5,000 and 15,000 seats (inclusive)."
    else:
        claim_cap = f"The seating capacity of {venue_name} is between 5,000 and 15,000 seats."
    add_ins_cap = build_additional_instruction(
        urls,
        "Verify the venue's seating capacity on the cited page. Use the concert/event seating figure; it must be within 5,000–15,000 (inclusive)."
    )
    await add_leaf_and_verify(
        evaluator, "SW_Capacity_Range", "Venue must have a capacity between 5,000 and 15,000 seats",
        node, claim_cap, urls, add_ins_cap, critical=True
    )

    # Venue type: arena or amphitheater suitable for concerts
    claim_type = f"{venue_name} is an arena or amphitheater suitable for concert performances."
    add_ins_type = build_additional_instruction(
        urls,
        "Confirm that the venue is either an arena or an amphitheater and that it hosts (or is suitable for) concert performances."
    )
    await add_leaf_and_verify(
        evaluator, "SW_Venue_Type",
        "Venue must be suitable for concert performances (arena or amphitheater)",
        node, claim_type, urls, add_ins_type, critical=True
    )

    # VIP ticket packages with premium amenities
    claim_vip = (
        f"{venue_name} offers VIP ticket packages with premium amenities "
        "(e.g., lounge/club access, reserved premium seating, suites, priority services)."
    )
    add_ins_vip = build_additional_instruction(
        urls,
        "Verify that VIP ticket packages are offered and include premium amenities (e.g., lounges, clubs, suites, premium seating, hospitality)."
    )
    await add_leaf_and_verify(
        evaluator, "SW_VIP_Package",
        "Venue must offer VIP ticket packages with premium amenities",
        node, claim_vip, urls, add_ins_vip, critical=True
    )


async def verify_midwest(evaluator: Evaluator, root_node, info: Optional[VenueInfo]) -> None:
    node = evaluator.add_parallel(
        id="Midwest_Large_Stadium_Venue",
        desc="Evaluate the Midwest venue meeting large stadium specifications",
        parent=root_node,
        critical=False
    )

    # Existence checks (critical)
    name_ok = bool(info and info.name and info.name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="MW_Venue_Name_Provided",
        desc="Answer must provide the specific name of the venue",
        parent=node,
        critical=True
    )

    city_ok = bool(info and info.city and info.city.strip())
    evaluator.add_custom_node(
        result=city_ok,
        id="MW_City_Location_Provided",
        desc="Answer must provide the specific city location of the venue",
        parent=node,
        critical=True
    )

    cap_num = parse_capacity_int(info.capacity if info else None)
    evaluator.add_custom_node(
        result=(cap_num is not None),
        id="MW_Exact_Capacity_Provided",
        desc="Answer must provide the exact seating capacity number",
        parent=node,
        critical=True
    )

    urls = info.urls if info else []
    venue_name = info.name if info and info.name else "the venue"
    city_part = f"{info.city}, " if info and info.city else ""

    # State location (Ohio)
    claim_state = f"{venue_name} is located in {city_part}Ohio."
    add_ins_state = build_additional_instruction(
        urls,
        "Confirm that the venue is in the state of Ohio (OH). City must be in Ohio; abbreviations like 'OH' are acceptable."
    )
    await add_leaf_and_verify(
        evaluator, "MW_State_Location", "Venue must be located in Ohio",
        node, claim_state, urls, add_ins_state, critical=True
    )

    # Capacity >= 40,000
    if cap_num is not None:
        claim_cap = f"The seating capacity of {venue_name} is {cap_num}, which is 40,000 or more."
    else:
        claim_cap = f"The seating capacity of {venue_name} is at least 40,000."
    add_ins_cap = build_additional_instruction(
        urls,
        "Verify the venue's seating capacity on the cited page. Use the stated seating capacity; it must be 40,000 or greater."
    )
    await add_leaf_and_verify(
        evaluator, "MW_Capacity_Range", "Venue must have a capacity of 40,000 or more seats",
        node, claim_cap, urls, add_ins_cap, critical=True
    )

    # Venue type: stadium suitable for large-scale concerts
    claim_type = f"{venue_name} is a stadium and is suitable for large-scale concerts."
    add_ins_type = build_additional_instruction(
        urls,
        "Confirm that the venue is a stadium and that it hosts (or is suitable for) large-scale concerts (major touring acts)."
    )
    await add_leaf_and_verify(
        evaluator, "MW_Venue_Type",
        "Venue must be a stadium suitable for large-scale concerts",
        node, claim_type, urls, add_ins_type, critical=True
    )

    # VIP or premium seating options
    claim_vip = f"{venue_name} offers VIP or premium seating options (e.g., club seats, suites, premium clubs)."
    add_ins_vip = build_additional_instruction(
        urls,
        "Verify that VIP or premium seating options are offered (e.g., suites, club seats, premium clubs, hospitality)."
    )
    await add_leaf_and_verify(
        evaluator, "MW_VIP_Package",
        "Venue must offer VIP or premium seating options",
        node, claim_vip, urls, add_ins_vip, critical=True
    )


async def verify_northeast(evaluator: Evaluator, root_node, info: Optional[VenueInfo]) -> None:
    node = evaluator.add_parallel(
        id="Northeast_Theater_Venue",
        desc="Evaluate the Northeast theater venue meeting Broadway theater specifications",
        parent=root_node,
        critical=False
    )

    # Existence checks (critical)
    name_ok = bool(info and info.name and info.name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id="NE_Venue_Name_Provided",
        desc="Answer must provide the specific name of the venue",
        parent=node,
        critical=True
    )

    city_ok = bool(info and info.city and info.city.strip())
    evaluator.add_custom_node(
        result=city_ok,
        id="NE_City_Location_Provided",
        desc="Answer must provide the specific city location of the venue",
        parent=node,
        critical=True
    )

    cap_num = parse_capacity_int(info.capacity if info else None)
    evaluator.add_custom_node(
        result=(cap_num is not None),
        id="NE_Exact_Capacity_Provided",
        desc="Answer must provide the exact seating capacity number",
        parent=node,
        critical=True
    )

    urls = info.urls if info else []
    venue_name = info.name if info and info.name else "the venue"
    city_part = f"{info.city}, " if info and info.city else ""

    # State location (New York)
    claim_state = f"{venue_name} is located in {city_part}New York."
    add_ins_state = build_additional_instruction(
        urls,
        "Confirm that the venue is in the state of New York (NY). City must be in New York; abbreviations like 'NY' are acceptable."
    )
    await add_leaf_and_verify(
        evaluator, "NE_State_Location", "Venue must be located in New York",
        node, claim_state, urls, add_ins_state, critical=True
    )

    # Capacity >= 500 (Broadway)
    if cap_num is not None:
        claim_cap = f"The seating capacity of {venue_name} is {cap_num}, which is at least 500 seats."
    else:
        claim_cap = f"The seating capacity of {venue_name} is at least 500 seats."
    add_ins_cap = build_additional_instruction(
        urls,
        "Verify the theater's seating capacity on the cited page; it must be 500 or more to meet Broadway capacity standards."
    )
    await add_leaf_and_verify(
        evaluator, "NE_Capacity_Range",
        "Venue must be a Broadway theater with a minimum capacity of 500 seats",
        node, claim_cap, urls, add_ins_cap, critical=True
    )

    # Venue classification: Broadway theater (not Off-/Off-Off-)
    claim_class = (
        f"{venue_name} is classified as a Broadway theater (not Off-Broadway or Off-Off-Broadway)."
    )
    add_ins_class = build_additional_instruction(
        urls,
        "Confirm explicit classification as a Broadway theater (in NYC). "
        "If the page indicates Off-Broadway or Off-Off-Broadway, do not accept."
    )
    await add_leaf_and_verify(
        evaluator, "NE_Venue_Classification",
        "Venue must be classified as a Broadway theater (not Off-Broadway or Off-Off-Broadway)",
        node, claim_class, urls, add_ins_class, critical=True
    )

    # VIP / premium seating options
    claim_vip = f"{venue_name} offers premium or VIP seating options."
    add_ins_vip = build_additional_instruction(
        urls,
        "Verify that premium or VIP seating options exist (e.g., premium seats, boxes, lounges, club seating)."
    )
    await add_leaf_and_verify(
        evaluator, "NE_VIP_Seating",
        "Venue must offer premium or VIP seating options",
        node, claim_vip, urls, add_ins_vip, critical=True
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
    Evaluate an answer for the Multi-City Tour Venue Selection task.
    """
    # Initialize evaluator (root: parallel aggregation across regions)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=TourVenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Verify each regional venue according to rubric
    await verify_west_coast(evaluator, root, extracted.west_coast if extracted else None)
    await verify_southwest(evaluator, root, extracted.southwest if extracted else None)
    await verify_midwest(evaluator, root, extracted.midwest if extracted else None)
    await verify_northeast(evaluator, root, extracted.northeast if extracted else None)

    # Return structured evaluation summary
    return evaluator.get_summary()