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
TASK_ID = "ca_indoor_venues_1000_3000"
TASK_DESCRIPTION = (
    "Identify three indoor concert venues in California—one in Los Angeles, one in San Francisco, and one in San Diego—"
    "each with a capacity between 1,000 and 3,000 people, suitable for hosting a touring music festival. Each venue must "
    "meet the following requirements: (1) Be an indoor venue, (2) Have a capacity between 1,000 and 3,000 people, (3) "
    "Comply with ADA wheelchair accessibility requirements for assembly occupancies, (4) Implement a clear bag security "
    "policy allowing bags up to 12\"×6\"×12\", (5) Support all-ages events with appropriate supervision requirements for "
    "minors under 18, (6) Accept standard event liability insurance with minimum $1 million per occurrence and $2 million "
    "aggregate coverage, (7) Offer group ticket sales with minimum purchase of 10 tickets, (8) Use ticketing platforms with "
    "service fees in the standard industry range of 2-8% plus fixed fees per ticket, (9) Comply with California Business and "
    "Professions Code §22507 requiring full refunds within 30 calendar days for canceled events, (10) Display maximum occupancy "
    "signs as required for assembly occupancies with 50+ capacity, (11) Provide companion seats adjacent to each wheelchair space, "
    "(12) Maintain a clearly stated prohibited items policy, and (13) Provide verifiable reference URLs for all information. "
    "For each venue, provide the venue name, full address, exact capacity, and supporting reference URL."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """Structured info for a single venue."""
    name: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    capacity: Optional[str] = None  # Keep as string to allow ranges or textual variants
    indoor: Optional[str] = None    # e.g., "indoor", "outdoor", or textual description
    sources: List[str] = Field(default_factory=list)  # All URLs cited in the answer for this venue


class VenuesExtraction(BaseModel):
    """Extraction for the three target cities."""
    los_angeles: Optional[VenueItem] = None
    san_francisco: Optional[VenueItem] = None
    san_diego: Optional[VenueItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    You must extract structured venue information for exactly three cities in California from the answer text: Los Angeles, San Francisco, and San Diego.
    For each city, identify the first suitable venue mentioned in the answer (if multiple are listed, choose the first that appears).
    For each of the three venues, extract:
      - name: The venue name as written in the answer.
      - address: The full street address, including city and state (CA).
      - city: The city name (e.g., "Los Angeles", "San Francisco", "San Diego").
      - capacity: The exact or stated capacity number (or range) as given in the answer (keep textual form if not a single number).
      - indoor: Whether the venue is indoor (if explicitly stated or implied). If unclear, extract any descriptive text indicating indoor/outdoor.
      - sources: All reference URLs mentioned in the answer for this venue. Include any official venue pages or credible third-party pages (ticketing, policy, legal references, etc.). Return full URLs, not markdown text.

    Return a JSON object of the form:
    {
      "los_angeles": { "name": ..., "address": ..., "city": ..., "capacity": ..., "indoor": ..., "sources": [...] },
      "san_francisco": { ... },
      "san_diego": { ... }
    }

    Rules:
    - If any field for a given city venue is missing in the answer, set it to null (for strings) or [] (for sources).
    - Do NOT invent URLs; only extract those explicitly present in the answer text.
    - If the answer mentions a venue but provides no URLs for it, set sources to an empty array.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_city_venue(
    evaluator: Evaluator,
    parent_node,
    city_key: str,
    city_node_desc: str,
    venue: Optional[VenueItem],
) -> None:
    """
    Build verification nodes and run checks for a single city's venue.

    Parameters
    ----------
    evaluator : Evaluator
        Mind2Web2 evaluator instance.
    parent_node : VerificationNode
        Root or parent node to attach the city sub-tree.
    city_key : str
        Short prefix key ('la', 'sf', 'sd') used for node IDs.
    city_node_desc : str
        Description for the city-level node.
    venue : VenueItem | None
        Extracted venue info for the city. If None, create nodes and most checks will fail or be skipped appropriately.
    """

    # Create the city node (non-critical, parallel aggregation)
    city_node = evaluator.add_parallel(
        id=f"{city_key}_venue",
        desc=city_node_desc,
        parent=parent_node,
        critical=False
    )

    # Prepare values
    name = venue.name if venue else None
    address = venue.address if venue else None
    capacity_str = venue.capacity if venue else None
    sources = venue.sources if venue else []
    # For claims that should not proceed when no sources are provided, create a critical existence gate
    ref_url_node = evaluator.add_custom_node(
        result=(len(sources) > 0),
        id=f"{city_key}_reference_url",
        desc="Verifiable reference URL provided from official venue source or credible third party",
        parent=city_node,
        critical=True
    )

    # Helper for prerequisites: if no sources, subsequent leaves should be skipped
    prereqs = [ref_url_node]

    # 1) City location
    location_leaf = evaluator.add_leaf(
        id=f"{city_key}_city_location",
        desc=f"Venue is located in {city_node_desc.split(' in ')[-1].split(' meeting')[0]}, California",
        parent=city_node,
        critical=True
    )
    claim_location = f"The referenced venue page indicates the venue is located in {city_node_desc.split(' in ')[-1].split(' meeting')[0]}, California."
    await evaluator.verify(
        claim=claim_location,
        node=location_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Confirm the venue city and state from the provided URLs. Allow minor naming variants like 'LA' for Los Angeles."
    )

    # 2) Indoor venue
    indoor_leaf = evaluator.add_leaf(
        id=f"{city_key}_indoor_venue",
        desc="Venue is an indoor concert venue, not an outdoor amphitheater",
        parent=city_node,
        critical=True
    )
    claim_indoor = "The venue is an indoor concert venue (e.g., theater, arena, club), not an outdoor amphitheater."
    await evaluator.verify(
        claim=claim_indoor,
        node=indoor_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Check descriptions/photos/policies indicating an enclosed indoor space. If the page says 'amphitheater' or outdoor lawn, treat as not indoor."
    )

    # 3) Capacity range
    cap_leaf = evaluator.add_leaf(
        id=f"{city_key}_capacity_range",
        desc="Venue capacity is between 1,000 and 3,000 people",
        parent=city_node,
        critical=True
    )
    capacity_note = capacity_str or "unknown capacity stated in the answer"
    claim_capacity = (
        f"The venue's capacity is reported as '{capacity_note}', and the capacity is within 1,000 to 3,000 people inclusive."
    )
    await evaluator.verify(
        claim=claim_capacity,
        node=cap_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Use any stated capacity from the venue or credible sources. Ranges like 'about 2,000' are acceptable. The acceptable inclusive range is 1,000–3,000."
    )

    # 4) ADA compliance (wheelchair accessibility for assembly occupancy)
    ada_leaf = evaluator.add_leaf(
        id=f"{city_key}_ada_compliance",
        desc="Venue complies with ADA wheelchair accessibility requirements appropriate for its capacity (minimum 10 wheelchair spaces for 1,000+ capacity venues, each 36\" wide or 66\" for double spaces)",
        parent=city_node,
        critical=True
    )
    claim_ada = (
        "The venue complies with ADA wheelchair accessibility requirements for assembly occupancies for venues over 1,000 capacity, "
        "including adequate wheelchair spaces and dimensions (≈36 inches for single, ≈66 inches for double spaces) and compliant accessible seating policies."
    )
    await evaluator.verify(
        claim=claim_ada,
        node=ada_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Look for accessibility/ADA statements, seating charts with wheelchair spaces, or compliance declarations. Consider the claim supported if the venue explicitly states ADA accessible seating and wheelchair spaces suitable for assembly occupancies."
    )

    # 5) Clear bag policy 12x6x12 + clutch 4.5x6.5
    bag_leaf = evaluator.add_leaf(
        id=f"{city_key}_clear_bag_policy",
        desc="Venue implements a clear bag policy allowing clear bags up to 12\"×6\"×12\" and small clutches up to 4.5\"×6.5\"",
        parent=city_node,
        critical=True
    )
    claim_bag = (
        "The venue implements a clear bag policy allowing clear bags up to 12 x 6 x 12 inches and small clutches up to approximately 4.5 x 6.5 inches."
    )
    await evaluator.verify(
        claim=claim_bag,
        node=bag_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Confirm the clear bag dimensions (12x6x12 inches) and clutch allowance (~4.5x6.5 inches) from the venue or event policy page."
    )

    # 6) Age policy (all-ages + supervision for under 18)
    age_leaf = evaluator.add_leaf(
        id=f"{city_key}_age_policy",
        desc="Venue supports all-ages events with supervision requirements for minors under 18 (must be accompanied by parent/guardian)",
        parent=city_node,
        critical=True
    )
    claim_age = (
        "The venue supports all-ages events, with minors under 18 required to be accompanied by a parent or guardian (or equivalent supervision policy)."
    )
    await evaluator.verify(
        claim=claim_age,
        node=age_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Check event pages or FAQ for all-ages policies and supervision requirements for minors (e.g., 'under 18 must be accompanied by adult')."
    )

    # 7) Insurance acceptance ($1M per occurrence, $2M aggregate)
    ins_leaf = evaluator.add_leaf(
        id=f"{city_key}_insurance_acceptance",
        desc="Venue accepts standard event liability insurance with minimum $1M per occurrence and $2M aggregate coverage",
        parent=city_node,
        critical=True
    )
    claim_ins = (
        "The venue accepts or requires standard event liability insurance with at least $1,000,000 per occurrence and $2,000,000 aggregate coverage."
    )
    await evaluator.verify(
        claim=claim_ins,
        node=ins_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Look for rental policies or event promoter guidelines specifying COI requirements of ≥$1M per occurrence and ≥$2M aggregate."
    )

    # 8) Group ticket sales (minimum purchase of 10)
    group_leaf = evaluator.add_leaf(
        id=f"{city_key}_group_tickets",
        desc="Venue offers group ticket sales with minimum purchase of 10 tickets",
        parent=city_node,
        critical=True
    )
    claim_group = "The venue offers group ticket sales that require a minimum purchase of approximately 10 tickets."
    await evaluator.verify(
        claim=claim_group,
        node=group_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Confirm a group sales offering and minimum purchase (e.g., 10 tickets) from the ticketing or venue policy pages."
    )

    # 9) Ticketing service fees (2–8% + fixed per-ticket fees)
    fees_leaf = evaluator.add_leaf(
        id=f"{city_key}_ticketing_fees",
        desc="Venue uses ticketing platforms with service fees in standard industry range (2-8% plus $0.50-$1.79 per ticket)",
        parent=city_node,
        critical=True
    )
    claim_fees = (
        "The venue uses ticketing platforms whose service fees typically fall within the standard range of 2–8% plus fixed per-ticket fees "
        "around $0.50–$1.79."
    )
    await evaluator.verify(
        claim=claim_fees,
        node=fees_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="You may verify via the venue's ticketing provider pages linked from the venue. Accept if the provider's published fees are within this range."
    )

    # 10) Refund policy BPC §22507 (full refunds within 30 days for canceled events)
    refund_leaf = evaluator.add_leaf(
        id=f"{city_key}_refund_policy",
        desc="Venue complies with California BPC §22507 requiring full refunds within 30 days for canceled events",
        parent=city_node,
        critical=True
    )
    claim_refund = (
        "The venue complies with California Business and Professions Code § 22507 by providing full refunds within 30 calendar days for canceled events."
    )
    await evaluator.verify(
        claim=claim_refund,
        node=refund_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Look for refund policy statements referencing California law or explicit timelines for canceled events (≤30 days). Credible third-party legal references cited by the answer can be used."
    )

    # 11) Occupancy signage
    occ_leaf = evaluator.add_leaf(
        id=f"{city_key}_occupancy_signage",
        desc="Venue displays maximum occupancy signs as required for assembly occupancies with 50+ capacity",
        parent=city_node,
        critical=True
    )
    claim_occ = (
        "The venue displays maximum occupancy signs as required for assembly occupancies (capacity ≥ 50), consistent with fire/building codes."
    )
    await evaluator.verify(
        claim=claim_occ,
        node=occ_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Check safety/code compliance pages or venue policies referencing occupancy signage. Accept credible compliance statements."
    )

    # 12) Companion seats adjacent to wheelchair spaces
    comp_leaf = evaluator.add_leaf(
        id=f"{city_key}_companion_seats",
        desc="Venue provides companion seats adjacent to each wheelchair space",
        parent=city_node,
        critical=True
    )
    claim_comp = "The venue provides companion seating immediately adjacent to each wheelchair space in accessible seating areas."
    await evaluator.verify(
        claim=claim_comp,
        node=comp_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Look for accessible seating policy or seat map notes indicating companion seats next to wheelchair spaces."
    )

    # 13) Prohibited items policy
    prohibited_leaf = evaluator.add_leaf(
        id=f"{city_key}_prohibited_items",
        desc="Venue maintains a clearly stated prohibited items policy",
        parent=city_node,
        critical=True
    )
    claim_prohibited = "The venue maintains a clearly stated prohibited items policy (e.g., list of items not allowed)."
    await evaluator.verify(
        claim=claim_prohibited,
        node=prohibited_leaf,
        sources=sources,
        extra_prerequisites=prereqs,
        additional_instruction="Check FAQ/security/policies pages for a list of prohibited items."
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
    Evaluate the answer for California indoor venues meeting specified requirements.
    """

    # Initialize evaluator (root is non-critical by framework design; use parallel aggregation)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Three suitable indoor concert venues identified—one each in Los Angeles, San Francisco, and San Diego—each meeting all specified requirements",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract venue info
    extracted = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction"
    )

    # Build verification subtrees per city as per rubric
    # Los Angeles
    await verify_city_venue(
        evaluator=evaluator,
        parent_node=root,
        city_key="la",
        city_node_desc="A suitable indoor concert venue in Los Angeles meeting all requirements",
        venue=extracted.los_angeles
    )

    # San Francisco
    await verify_city_venue(
        evaluator=evaluator,
        parent_node=root,
        city_key="sf",
        city_node_desc="A suitable indoor concert venue in San Francisco meeting all requirements",
        venue=extracted.san_francisco
    )

    # San Diego
    await verify_city_venue(
        evaluator=evaluator,
        parent_node=root,
        city_key="sd",
        city_node_desc="A suitable indoor concert venue in San Diego meeting all requirements",
        venue=extracted.san_diego
    )

    # Return summary: includes extraction and full verification tree
    return evaluator.get_summary()