import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "denver_convention_center_selection"
TASK_DESCRIPTION = """I am planning a large corporate conference in Denver, Colorado, and need to identify a suitable convention center venue in the downtown area. The venue must meet the following specific requirements:

1. Located in downtown Denver, Colorado
2. Provide at least 500,000 square feet of contiguous exhibit space
3. Have at least 60 meeting rooms available
4. Offer all meeting space on a single level for easy navigation
5. Include on-site covered parking facilities
6. Be fully ADA-compliant with accessible restrooms and elevators
7. Include complimentary wired microphones with meeting room rentals (minimum 2 per meeting room)
8. Provide wireless internet/WiFi capabilities in all meeting spaces
9. Have on-site or exclusive catering services
10. Feature at least one ballroom with a minimum of 30,000 square feet
11. Have a publicly available email contact address
12. Have a publicly listed phone number
13. Equip meeting rooms with computer-controlled audio and lighting systems
14. Have a verifiable street address in Denver

Please identify the convention center that meets all these requirements and provide its name, complete street address, contact phone number, and contact email address.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VenueInfo(BaseModel):
    name: Optional[str] = None
    street_address: Optional[str] = None
    phone_number: Optional[str] = None
    email_address: Optional[str] = None
    website_urls: List[str] = Field(default_factory=list)


class FeatureEvidence(BaseModel):
    location_denver: List[str] = Field(default_factory=list)
    exhibit_space_minimum: List[str] = Field(default_factory=list)
    meeting_room_count: List[str] = Field(default_factory=list)
    single_level_meeting: List[str] = Field(default_factory=list)
    onsite_parking: List[str] = Field(default_factory=list)
    ada_compliance: List[str] = Field(default_factory=list)
    included_av_equipment: List[str] = Field(default_factory=list)
    wifi_availability: List[str] = Field(default_factory=list)
    catering_services: List[str] = Field(default_factory=list)
    ballroom_space: List[str] = Field(default_factory=list)
    email_contact: List[str] = Field(default_factory=list)
    phone_contact: List[str] = Field(default_factory=list)
    computer_controlled_systems: List[str] = Field(default_factory=list)
    street_address_exists: List[str] = Field(default_factory=list)


class VenueExtraction(BaseModel):
    venue: Optional[VenueInfo] = None
    evidence: Optional[FeatureEvidence] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Identify the single convention center venue presented in the answer and extract the following fields exactly as provided:

    1) venue.name: The name of the convention center.
    2) venue.street_address: The complete street address.
    3) venue.phone_number: The publicly listed contact phone number.
    4) venue.email_address: The publicly available contact email address.
    5) venue.website_urls: All URLs in the answer that correspond to the venue's official site or relevant pages (e.g., floor plans, specs, contact page). Include only valid URLs explicitly present in the answer.

    Also, for each requirement below, extract the specific URLs cited in the answer that support that requirement (return an empty list if none are cited). Do not invent URLs.

    evidence.location_denver: URLs supporting that the venue is in downtown Denver, Colorado.
    evidence.exhibit_space_minimum: URLs supporting at least 500,000 sq ft of contiguous exhibit space.
    evidence.meeting_room_count: URLs supporting at least 60 meeting rooms.
    evidence.single_level_meeting: URLs supporting all meeting space is on a single level.
    evidence.onsite_parking: URLs supporting on-site covered parking facilities.
    evidence.ada_compliance: URLs supporting ADA compliance, accessible restrooms and elevators.
    evidence.included_av_equipment: URLs supporting complimentary wired microphones (minimum 2 per meeting room) with room rentals.
    evidence.wifi_availability: URLs supporting wireless internet/WiFi in all meeting spaces.
    evidence.catering_services: URLs supporting on-site or exclusive catering services.
    evidence.ballroom_space: URLs supporting at least one ballroom with minimum 30,000 sq ft.
    evidence.email_contact: URLs supporting the publicly available email contact address.
    evidence.phone_contact: URLs supporting the publicly listed phone number.
    evidence.computer_controlled_systems: URLs supporting computer-controlled audio and lighting systems in meeting rooms.
    evidence.street_address_exists: URLs showing/verifying the street address is in Denver.

    Rules:
    - Extract exactly what is present in the answer; if any field is not provided, return null (for single values) or an empty list (for URLs).
    - For URLs, accept plain URLs or markdown links; return the URL strings.
    - Do not infer or fabricate information.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _safe_text(value: Optional[str], fallback: str = "") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _union_sources(*lists: List[str]) -> Optional[List[str]]:
    merged: List[str] = []
    seen = set()
    for lst in lists:
        for url in lst or []:
            if isinstance(url, str):
                u = url.strip()
                if u and u not in seen:
                    seen.add(u)
                    merged.append(u)
    return merged if merged else None


def _name_for_claim(extracted: VenueExtraction) -> str:
    if extracted and extracted.venue and extracted.venue.name:
        return extracted.venue.name
    return "the venue"


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_venue_nodes(
    evaluator: Evaluator,
    parent: VerificationNode,
    extracted: VenueExtraction,
) -> None:
    """
    Construct critical leaf nodes for all requirements under the venue_identification
    parent node and perform batch verification with appropriate claims and sources.
    """
    venue_name = _name_for_claim(extracted)
    venue_urls = extracted.venue.website_urls if (extracted and extracted.venue) else []

    evidence = extracted.evidence or FeatureEvidence()

    # Create critical leaf nodes (one per requirement)
    nodes: Dict[str, VerificationNode] = {}

    def add_leaf_node(node_id: str, desc: str) -> VerificationNode:
        node = evaluator.add_leaf(
            id=node_id,
            desc=desc,
            parent=parent,
            critical=True,
        )
        nodes[node_id] = node
        return node

    # Map of requirement descriptions (exactly as in JSON)
    descriptions = {
        "location_denver": "The venue is located in Denver, Colorado, in the downtown area",
        "exhibit_space_minimum": "The venue has at least 500,000 square feet of contiguous exhibit space",
        "meeting_room_count": "The venue has at least 60 meeting rooms",
        "single_level_meeting": "The meeting space is provided on a single level",
        "onsite_parking": "The venue has on-site covered parking facilities",
        "ada_compliance": "The venue has ADA-compliant accessibility features including accessible restrooms and elevators",
        "included_av_equipment": "The venue includes complimentary wired microphones with room rentals (at least 2 per meeting room)",
        "wifi_availability": "The venue provides wireless internet/WiFi capabilities in meeting spaces",
        "catering_services": "The venue has on-site or exclusive catering services",
        "ballroom_space": "The venue has at least one ballroom with minimum 30,000 square feet",
        "email_contact": "The venue has a publicly available email contact address",
        "phone_contact": "The venue has a publicly listed phone number",
        "computer_controlled_systems": "Meeting rooms have computer-controlled audio and lighting systems",
        "street_address_exists": "The venue has a verifiable street address in Denver",
    }

    # Create all leaf nodes
    for node_id, desc in descriptions.items():
        add_leaf_node(node_id, desc)

    # Build claims and sources for batch verification
    claims_and_sources: List[tuple[str, Optional[List[str]] | Optional[str], VerificationNode, Optional[str]]] = []

    # 1. Location (Downtown Denver)
    claims_and_sources.append((
        f"The venue named '{venue_name}' is located in downtown Denver, Colorado.",
        _union_sources(evidence.location_denver, venue_urls),
        nodes["location_denver"],
        "Verify that the webpage(s) explicitly indicate the venue is in downtown Denver. If 'downtown' is implied by the address, consider reasonable interpretation."
    ))

    # 2. Exhibit space minimum
    claims_and_sources.append((
        f"The venue '{venue_name}' provides at least 500,000 square feet of contiguous exhibit space.",
        _union_sources(evidence.exhibit_space_minimum, venue_urls),
        nodes["exhibit_space_minimum"],
        "Confirm the total contiguous exhibit space is ≥ 500,000 sq ft. Prefer explicit 'contiguous' wording; minor numeric rounding is acceptable."
    ))

    # 3. Meeting room count
    claims_and_sources.append((
        f"The venue '{venue_name}' has at least 60 meeting rooms.",
        _union_sources(evidence.meeting_room_count, venue_urls),
        nodes["meeting_room_count"],
        "Check venue specifications or floor plan pages for the number of meeting rooms; accept '60 or more' and exact counts ≥ 60."
    ))

    # 4. Single-level meeting space
    claims_and_sources.append((
        f"All meeting space at '{venue_name}' is provided on a single level for easy navigation.",
        _union_sources(evidence.single_level_meeting, venue_urls),
        nodes["single_level_meeting"],
        "The evidence should indicate meeting spaces are on one level; wording like 'single level' or 'all on one floor' suffices."
    ))

    # 5. On-site covered parking
    claims_and_sources.append((
        f"'{venue_name}' has on-site covered parking facilities.",
        _union_sources(evidence.onsite_parking, venue_urls),
        nodes["onsite_parking"],
        "Confirm that parking is both on-site and covered. Pages may refer to a parking garage attached to the venue."
    ))

    # 6. ADA compliance
    claims_and_sources.append((
        f"'{venue_name}' is ADA-compliant, including accessible restrooms and elevators.",
        _union_sources(evidence.ada_compliance, venue_urls),
        nodes["ada_compliance"],
        "Look for explicit ADA/Accessibility statements on the venue website mentioning accessible restrooms and elevators."
    ))

    # 7. Complimentary wired microphones (≥ 2 per meeting room)
    claims_and_sources.append((
        f"Meeting room rentals at '{venue_name}' include complimentary wired microphones, with at least two per meeting room.",
        _union_sources(evidence.included_av_equipment, venue_urls),
        nodes["included_av_equipment"],
        "Evidence must state wired microphones are complimentary/included with room rentals, and that at least two are provided per room."
    ))

    # 8. WiFi availability in meeting spaces
    claims_and_sources.append((
        f"'{venue_name}' provides wireless internet/WiFi capabilities in all meeting spaces.",
        _union_sources(evidence.wifi_availability, venue_urls),
        nodes["wifi_availability"],
        "Verify WiFi availability in all meeting spaces (not just lobbies or common areas)."
    ))

    # 9. On-site or exclusive catering
    claims_and_sources.append((
        f"'{venue_name}' has on-site or exclusive catering services.",
        _union_sources(evidence.catering_services, venue_urls),
        nodes["catering_services"],
        "Check for venue-operated catering or exclusive partnerships listed on the venue site."
    ))

    # 10. Ballroom space ≥ 30,000 sq ft
    claims_and_sources.append((
        f"'{venue_name}' features at least one ballroom with a minimum of 30,000 square feet.",
        _union_sources(evidence.ballroom_space, venue_urls),
        nodes["ballroom_space"],
        "Find ballroom specifications indicating size ≥ 30,000 sq ft; accept 'approx.' or minor rounding."
    ))

    # 11. Publicly available email contact
    email_val = _safe_text(extracted.venue.email_address if extracted and extracted.venue else None)
    email_claim = (
        f"The venue '{venue_name}' has a publicly available email contact address."
        if not email_val else
        f"The publicly available contact email address for '{venue_name}' is '{email_val}'."
    )
    claims_and_sources.append((
        email_claim,
        _union_sources(evidence.email_contact, venue_urls),
        nodes["email_contact"],
        "Prefer contact pages or official listings. If a specific email is provided in the claim, verify it appears on the page."
    ))

    # 12. Publicly listed phone number
    phone_val = _safe_text(extracted.venue.phone_number if extracted and extracted.venue else None)
    phone_claim = (
        f"The venue '{venue_name}' has a publicly listed phone number."
        if not phone_val else
        f"The publicly listed phone number for '{venue_name}' is '{phone_val}'."
    )
    claims_and_sources.append((
        phone_claim,
        _union_sources(evidence.phone_contact, venue_urls),
        nodes["phone_contact"],
        "Use official site listings; if a specific phone number is claimed, verify it appears exactly or with minor formatting variations."
    ))

    # 13. Computer-controlled audio and lighting systems
    claims_and_sources.append((
        f"Meeting rooms at '{venue_name}' have computer-controlled audio and lighting systems.",
        _union_sources(evidence.computer_controlled_systems, venue_urls),
        nodes["computer_controlled_systems"],
        "Look for AV specs stating computer-controlled audio and lighting (e.g., integrated digital control systems)."
    ))

    # 14. Verifiable street address in Denver
    address_val = _safe_text(extracted.venue.street_address if extracted and extracted.venue else None)
    address_claim = (
        f"The venue '{venue_name}' has a verifiable street address in Denver."
        if not address_val else
        f"The venue '{venue_name}' has a verifiable Denver street address: '{address_val}'."
    )
    claims_and_sources.append((
        address_claim,
        _union_sources(evidence.street_address_exists, venue_urls),
        nodes["street_address_exists"],
        "Confirm the street address appears on official pages and that it is in Denver, CO."
    ))

    # Execute all verifications in parallel to avoid precondition short-circuiting across siblings
    await evaluator.batch_verify(claims_and_sources)

    # After verifications, add critical existence checks for provided info fields
    venue = extracted.venue or VenueInfo()
    evaluator.add_custom_node(
        result=bool(_safe_text(venue.name)),
        id="provided_venue_name",
        desc="The answer provides the name of the convention center",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_text(venue.street_address)),
        id="provided_street_address",
        desc="The answer provides the complete street address",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_text(venue.phone_number)),
        id="provided_phone_number",
        desc="The answer provides the contact phone number",
        parent=parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(_safe_text(venue.email_address)),
        id="provided_email_address",
        desc="The answer provides the contact email address",
        parent=parent,
        critical=True
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
    Evaluate an answer for the Denver convention center venue selection task.
    Builds a critical parallel verification tree under 'venue_identification'.
    """
    # Initialize evaluator with parallel root
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

    # Extract venue information and evidence URLs from the answer
    extracted: VenueExtraction = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # Create the top-level critical venue node
    venue_node = evaluator.add_parallel(
        id="venue_identification",
        desc="Identify a convention center venue that satisfies all specified requirements and provide the requested information",
        parent=root,
        critical=True  # All children must be critical per framework constraints
    )

    # Build and verify all requirement nodes under the venue node
    await build_and_verify_venue_nodes(evaluator, venue_node, extracted)

    # Return standardized summary
    return evaluator.get_summary()