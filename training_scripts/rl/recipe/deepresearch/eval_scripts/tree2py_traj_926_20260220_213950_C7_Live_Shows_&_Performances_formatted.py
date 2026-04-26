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
TASK_ID = "nyc_arena_accessibility"
TASK_DESCRIPTION = """
Identify a major concert arena in New York City with a seating capacity of at least 15,000 people for concert events. For your identified venue, provide comprehensive verification that it meets the following accessibility and facility standards: (1) Document the venue's specific concert seating capacity, (2) Confirm wheelchair accessible seating comprises at least 1% of total capacity as required by ADA, (3) Verify wheelchair space dimensions meet ADA standards (minimum 36 inches wide, 48 inches deep), (4) Confirm companion seats are available adjacent to wheelchair spaces, (5) Verify accessible seating is offered at multiple price levels, (6) Confirm elevator access is provided to wheelchair accessible seating areas for multi-level venues, (7) Document that accessible restrooms meet ADA dimensions (stalls at least 60 inches wide, 56 inches deep), (8) Verify accessible parking spaces are available, (9) Confirm appropriate accessible-to-regular restroom ratios, (10) Verify aisle transfer seats with removable armrests are available, (11) Confirm accessible entrance pathways with minimum 32-inch clearance exist, (12) Verify wheelchair accessible seating is distributed across different sections, and (13) Provide supporting reference URL(s) for the venue information. Provide specific details and documentation for each accessibility feature to demonstrate full compliance with ADA standards and venue accessibility best practices.
"""

# ADA thresholds and references (for context logging)
ADA_THRESHOLDS = {
    "capacity_min": 15000,
    "wheelchair_seating_min_percent": ">= 1%",
    "wheelchair_space_min_width": ">= 36 inches (or >= 33 inches if adjacent configuration provides required clearances)",
    "wheelchair_space_min_depth": ">= 48 inches",
    "accessible_restroom_stall_width": ">= 60 inches",
    "accessible_restroom_stall_depth": ">= 56 inches",
    "accessible_entrance_clear_width": ">= 32 inches",
    "accessible_parking_ratio": ">= 1 accessible space per 25 parking spaces",
    "restroom_ratio_recommended": "Accessible-to-regular stalls approx. 1:10 (recommendation)"
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    """Structured extraction of venue and accessibility details from the answer."""
    venue_name: Optional[str] = None
    venue_city: Optional[str] = None
    concert_capacity: Optional[str] = None
    wheelchair_accessible_percentage: Optional[str] = None
    wheelchair_space_dimensions: Optional[str] = None
    companion_seats: Optional[str] = None
    multi_price_accessible: Optional[str] = None
    elevator_access: Optional[str] = None
    accessible_restroom_dimensions: Optional[str] = None
    accessible_parking: Optional[str] = None
    restroom_ratio: Optional[str] = None
    aisle_transfer_seats: Optional[str] = None
    accessible_entrances: Optional[str] = None
    distributed_accessible_seating: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_info() -> str:
    return """
    Extract information about ONE major concert arena in New York City mentioned in the answer. 
    If multiple venues are mentioned, choose the first one that meets the capacity requirement.
    
    Return the following fields exactly as stated in the answer (use strings; do not invent):
    - venue_name: The venue's name.
    - venue_city: The city/borough (e.g., "New York", "Manhattan", "Brooklyn").
    - concert_capacity: The concert seating capacity number or phrase (e.g., "19,500", "about 18,000").
    - wheelchair_accessible_percentage: Any statement or percentage indicating the ratio of wheelchair accessible seating (e.g., "1%", "at least one percent").
    - wheelchair_space_dimensions: Any dimensions for wheelchair seating spaces (e.g., "36 inches wide, 48 inches deep").
    - companion_seats: Statement/claim about companion seating adjacent to wheelchair spaces.
    - multi_price_accessible: Statement/claim indicating accessible seating available at multiple price levels.
    - elevator_access: Statement/claim that elevators (or equivalent) provide access to accessible seating areas.
    - accessible_restroom_dimensions: Any dimensions for accessible restroom stalls (e.g., "60 inches wide, 56 inches deep").
    - accessible_parking: Statement/claim about accessible parking spaces and ratios/availability.
    - restroom_ratio: Any statement/claim about accessible-to-regular restroom ratios (e.g., "1:10").
    - aisle_transfer_seats: Statement/claim about aisle transfer seats with removable armrests.
    - accessible_entrances: Statement/claim that accessible entrance pathways exist; include any doorway width figures.
    - distributed_accessible_seating: Statement/claim that accessible seating is distributed across sections/price levels.
    - reference_urls: A list of all URLs explicitly cited in the answer that support the venue details. Include the full URL string for each.
    
    Rules:
    - Only extract what the answer explicitly provides. If a field is not mentioned, set it to null or an empty list for reference_urls.
    - For URLs: include all valid URLs (plain or in markdown). Do not infer or create URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else "the venue"


def _urls_from_extraction(ex: VenueInfo) -> List[str]:
    return [u for u in (ex.reference_urls or []) if isinstance(u, str) and u.strip()]


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
    Evaluate an answer for the NYC concert arena accessibility standards task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks for each criterion
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

    # Extract structured venue information from the answer
    venue_info = await evaluator.extract(
        prompt=prompt_extract_venue_info(),
        template_class=VenueInfo,
        extraction_name="venue_info",
    )

    # Add ADA thresholds as ground truth contextual information
    evaluator.add_ground_truth({
        "ada_thresholds": ADA_THRESHOLDS,
        "notes": "These thresholds provide context for judging claims; verification is done via cited sources."
    })

    # Record additional custom info for traceability
    evaluator.add_custom_info({
        "venue_name": venue_info.venue_name,
        "venue_city": venue_info.venue_city,
        "concert_capacity": venue_info.concert_capacity,
        "reference_url_count": len(venue_info.reference_urls or []),
    }, info_type="extracted_overview", info_name="extracted_overview")

    # Prepare sources and a critical existence node for URLs
    ref_urls = _urls_from_extraction(venue_info)
    urls_exist = len(ref_urls) > 0
    reference_url_node = evaluator.add_custom_node(
        result=urls_exist,
        id="reference_url",
        desc="Supporting reference URL(s) are provided that document the venue information",
        parent=root,
        critical=True
    )

    # Helper variables
    venue_name = _safe_name(venue_info.venue_name)

    # Build leaf nodes and verification tasks (each leaf is a single binary check)
    verify_tasks: List[asyncio.Task] = []

    # 1) Venue identification (critical)
    venue_ident_node = evaluator.add_leaf(
        id="venue_identification",
        desc="A major concert arena in New York City has been identified",
        parent=root,
        critical=True,
    )
    claim_venue_ident = (
        f"The venue named '{venue_name}' is a concert arena located in New York City (any borough of NYC) "
        f"and is suitable for major events (e.g., hosts large concerts)."
    )
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_venue_ident,
            node=venue_ident_node,
            sources=ref_urls,
            additional_instruction="Confirm the venue is in New York City and functions as a large-scale concert arena. "
                                   "Accept borough names (Manhattan, Brooklyn, Queens, Bronx, Staten Island). "
                                   "If the capacity listed on the sources is >= 15,000, treat 'major' as satisfied.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 2) Capacity documentation (critical)
    capacity_doc_node = evaluator.add_leaf(
        id="capacity_documentation",
        desc="The venue's concert seating capacity is documented and publicly available",
        parent=root,
        critical=True,
    )
    claim_capacity_doc = (
        f"The cited sources explicitly state the concert seating capacity for {venue_name}."
    )
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_capacity_doc,
            node=capacity_doc_node,
            sources=ref_urls,
            additional_instruction="Look for an explicit capacity number or phrase (e.g., seating capacity, concert capacity) "
                                   "on the venue or authoritative sources. Screenshots may show the number.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 3) Minimum capacity >= 15,000 (critical)
    capacity_min_node = evaluator.add_leaf(
        id="minimum_capacity",
        desc="The venue has a seating capacity of at least 15,000 for concert events",
        parent=root,
        critical=True,
    )
    claim_capacity_min = f"The concert seating capacity at {venue_name} is at least 15,000."
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_capacity_min,
            node=capacity_min_node,
            sources=ref_urls,
            additional_instruction="Use the capacity figure on the cited sources. If the number is 15,000 or higher, pass. "
                                   "Allow small phrasing variations or rounding.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 4) Wheelchair seating >= 1% of total capacity (critical)
    wc_percent_node = evaluator.add_leaf(
        id="wheelchair_seating_percentage",
        desc="Wheelchair accessible seating comprises at least 1% of total seating capacity",
        parent=root,
        critical=True,
    )
    claim_wc_percent = f"Wheelchair accessible seating at {venue_name} comprises at least 1% of the total seating capacity."
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_wc_percent,
            node=wc_percent_node,
            sources=ref_urls,
            additional_instruction="Look for ADA compliance statements, seating policies, or numbers indicating wheelchair space count. "
                                   "If explicit counts imply >=1% relative to capacity, consider satisfied.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 5) Wheelchair space dimensions (critical)
    wc_dim_node = evaluator.add_leaf(
        id="wheelchair_space_dimensions",
        desc="Wheelchair spaces meet ADA dimensional standards (minimum 36 inches wide or 33 inches if adjacent, 48 inches deep)",
        parent=root,
        critical=True,
    )
    claim_wc_dim = (
        f"Wheelchair seating spaces at {venue_name} meet ADA dimensions: at least 36 inches wide "
        f"(or at least 33 inches wide if an adjacent configuration allows required clearances) and at least 48 inches deep."
    )
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_wc_dim,
            node=wc_dim_node,
            sources=ref_urls,
            additional_instruction="Check any technical seating specifications, ADA policy pages, or seating charts that mention dimensions. "
                                   "Allow minor wording variations that clearly match the ADA minimums.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 6) Companion seats adjacent (critical)
    companion_node = evaluator.add_leaf(
        id="companion_seats",
        desc="Companion seats are available adjacent to or near wheelchair accessible spaces",
        parent=root,
        critical=True,
    )
    claim_companion = f"Companion seats adjacent to or near wheelchair spaces are available at {venue_name}."
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_companion,
            node=companion_node,
            sources=ref_urls,
            additional_instruction="Look for statements like 'companion seating', 'adjacent companion seats', or similar in policies.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 7) Accessible seating at multiple price levels (critical)
    multi_price_node = evaluator.add_leaf(
        id="multi_price_accessibility",
        desc="Accessible seating is available at multiple price levels",
        parent=root,
        critical=True,
    )
    claim_multi_price = f"Accessible seating at {venue_name} is available across multiple price levels."
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_multi_price,
            node=multi_price_node,
            sources=ref_urls,
            additional_instruction="Confirm that accessible seating is offered across different sections/price tiers, not limited to a single price.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 8) Elevator access (critical)
    elevator_node = evaluator.add_leaf(
        id="elevator_access",
        desc="Elevator access is provided to wheelchair accessible seating areas (for multi-level venues)",
        parent=root,
        critical=True,
    )
    claim_elevator = (
        f"Elevator access is provided to reach wheelchair accessible seating areas at {venue_name}. "
        f"If the venue is single-level with grade-level access, treat the requirement as satisfied."
    )
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_elevator,
            node=elevator_node,
            sources=ref_urls,
            additional_instruction="If the venue has multiple seating levels, verify elevators (or ramps/lifts) provide access to accessible seating. "
                                   "If the venue is single-level with accessible routes, consider satisfied.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 9) Accessible restrooms meet ADA dimensions (critical)
    restroom_dim_node = evaluator.add_leaf(
        id="accessible_restrooms",
        desc="Wheelchair accessible restrooms meet ADA dimensions (stalls at least 60 inches wide, 56 inches deep)",
        parent=root,
        critical=True,
    )
    claim_restroom_dim = (
        f"Accessible restroom stalls at {venue_name} meet ADA dimensions: at least 60 inches wide and at least 56 inches deep."
    )
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_restroom_dim,
            node=restroom_dim_node,
            sources=ref_urls,
            additional_instruction="Look for restroom specifications, ADA drawings, or policy statements. "
                                   "Allow equivalent ADA-compliant dimensions (e.g., deeper stalls for floor-mounted toilets).",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 10) Accessible parking spaces (critical)
    parking_node = evaluator.add_leaf(
        id="accessible_parking",
        desc="Accessible parking spaces are available at appropriate ratios (at least 1 per 25 spaces)",
        parent=root,
        critical=True,
    )
    claim_parking = (
        f"Accessible parking spaces are available for {venue_name} at appropriate ratios (at least 1 per 25 spaces). "
        f"If parking is off-site but provided via an official partner facility, verify availability there."
    )
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_parking,
            node=parking_node,
            sources=ref_urls,
            additional_instruction="Confirm accessible parking availability and ratio from venue or official partner garage information.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 11) Restroom ratio recommended 1:10 (non-critical)
    restroom_ratio_node = evaluator.add_leaf(
        id="restroom_ratio",
        desc="Appropriate accessible-to-regular restroom ratio is maintained (recommended 1:10)",
        parent=root,
        critical=False,
    )
    claim_restroom_ratio = (
        f"The accessible-to-regular restroom ratio at {venue_name} aligns with the recommended 1:10 (or comparable best-practice)."
    )
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_restroom_ratio,
            node=restroom_ratio_node,
            sources=ref_urls,
            additional_instruction="Check any available specifications or policy references regarding restroom ratios. "
                                   "Allow equivalent or better ratios if explicitly documented.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 12) Aisle transfer seats (critical)
    transfer_seats_node = evaluator.add_leaf(
        id="aisle_transfer_seats",
        desc="Aisle transfer seats with removable armrests are available",
        parent=root,
        critical=True,
    )
    claim_transfer_seats = f"Aisle transfer seats with removable (or movable) armrests are available at {venue_name}."
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_transfer_seats,
            node=transfer_seats_node,
            sources=ref_urls,
            additional_instruction="Look for 'transfer seats', 'aisle transfer seats', or 'removable armrests' in accessibility/policy pages.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 13) Accessible entrance pathways with 32-inch clearance (critical)
    entrances_node = evaluator.add_leaf(
        id="accessible_entrances",
        desc="Accessible entrance pathways with minimum 32-inch clearance exist",
        parent=root,
        critical=True,
    )
    claim_entrances = f"Accessible entrance pathways at {venue_name} provide a clear width of at least 32 inches."
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_entrances,
            node=entrances_node,
            sources=ref_urls,
            additional_instruction="Confirm entry/doorway widths or accessibility statements indicating ADA-compliant clear widths.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # 14) Distributed accessible seating across sections/price levels (critical)
    distributed_node = evaluator.add_leaf(
        id="distributed_seating",
        desc="Wheelchair accessible seating is distributed across different sections and price levels",
        parent=root,
        critical=True,
    )
    claim_distributed = (
        f"Wheelchair accessible seating at {venue_name} is distributed across different sections and price levels, "
        f"not restricted to a single area."
    )
    verify_tasks.append(asyncio.create_task(
        evaluator.verify(
            claim=claim_distributed,
            node=distributed_node,
            sources=ref_urls,
            additional_instruction="Look for statements about distribution across sections or tiers; seating maps can indicate distribution.",
            extra_prerequisites=[reference_url_node],
        )
    ))

    # Execute all verifications concurrently
    await asyncio.gather(*verify_tasks, return_exceptions=True)

    # Return structured result
    return evaluator.get_summary()