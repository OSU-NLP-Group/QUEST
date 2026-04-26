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
TASK_ID = "austin_coworking_benchmark"
TASK_DESCRIPTION = """I am planning to open a satellite office for my tech company in Austin, Texas. My team consists of 18 employees who will work from this location regularly, and I need to identify a suitable coworking space in downtown Austin to use as a benchmark for comparison.

Find one coworking space currently operating in downtown Austin that can accommodate at least 18 people with dedicated workspace options (dedicated desks or private offices, not just hot-desking). The space must have meeting room facilities available for team use.

For the coworking space you identify, provide the following information:
1. Official name and complete street address
2. Capacity information (stated number of desks/people, or total square footage)
3. Confirmation that dedicated desks or private office options are available
4. Description of meeting room or conference room facilities
5. Parking arrangement details (on-site, nearby options, or specific information)
6. Confirmation of high-speed internet/WiFi availability
7. Information about membership term flexibility (month-to-month, flexible contracts, etc.)
8. Pricing information for dedicated desks or private offices (if publicly available)
9. Direct URL to the space's official website or verified listing page for verification
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CoworkingSpaceInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    capacity_info: Optional[str] = None
    workspace_options: Optional[str] = None  # mentions of "dedicated desks", "private offices", "team suites", etc.
    meeting_rooms: Optional[str] = None      # details about meeting/conference rooms
    parking: Optional[str] = None
    internet: Optional[str] = None           # mentions of wifi/high-speed internet
    membership_terms: Optional[str] = None   # month-to-month, flexible, no long-term contract
    pricing: Optional[str] = None            # price for dedicated desks or private offices
    website_url: Optional[str] = None
    listing_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_space_info() -> str:
    return """
    Extract details for a single coworking space mentioned in the answer. If multiple spaces are mentioned, extract only the first complete one with a URL. Provide the following fields exactly as stated in the answer:
    - name: The official name of the coworking space.
    - address: The complete street address as presented (include suite/floor if provided).
    - capacity_info: Any text that indicates capacity to host at least 18 people (e.g., number of desks, size of private offices, team suites, or total square footage).
    - workspace_options: Any text that confirms dedicated desks or private office options (not just hot desk/day pass).
    - meeting_rooms: Any text describing availability of meeting or conference rooms.
    - parking: Any text explaining parking options (on-site, garage, nearby lots, validation, etc.).
    - internet: Any text confirming high-speed internet or WiFi availability.
    - membership_terms: Any text about flexible terms (month-to-month, no long-term contracts, etc.).
    - pricing: Any text giving prices for dedicated desks or private offices (e.g., $X/month per desk or per office). If pricing is not provided, return null.
    - website_url: A single direct URL to the space’s official website location page. If not present, set to null.
    - listing_urls: An array of any additional URLs to verified listing pages for this space (e.g., WeWork, Industrious, Regus, LiquidSpace, Coworker, Deskpass, Upsuite, Office Evolution, Peerspace, etc.). If none, return an empty array.

    SPECIAL RULES:
    - Extract only URLs explicitly present in the answer (plain or markdown links). Do not invent URLs.
    - If a field is missing in the answer, set it to null (or empty list for listing_urls).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _get_all_sources(space: CoworkingSpaceInfo) -> List[str]:
    urls: List[str] = []
    if space.website_url and space.website_url.strip():
        urls.append(space.website_url.strip())
    if space.listing_urls:
        urls.extend([u for u in space.listing_urls if isinstance(u, str) and u.strip()])
    return urls


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root, space: CoworkingSpaceInfo) -> None:
    """
    Build the verification tree according to the rubric and run verifications.
    We slightly relax the top-level critical flag to allow partial credit for non-critical pricing.
    """

    # Top aggregator for the solution (set to non-critical to allow mixed children criticalities)
    solution_node = evaluator.add_parallel(
        id="Coworking_Space_Solution",
        desc="A coworking space in downtown Austin, Texas has been identified that meets the requirements for 18 employees",
        parent=root,
        critical=False
    )

    sources = _get_all_sources(space)

    # -------------------- URL Reference (Critical) -------------------- #
    url_ref_node = evaluator.add_parallel(
        id="URL_Reference",
        desc="A direct URL link to the space's official website or verified listing page is provided for verification",
        parent=solution_node,
        critical=True
    )

    url_provided_node = evaluator.add_custom_node(
        result=(len(sources) > 0),
        id="url_provided",
        desc="At least one direct URL (official site or verified listing) is provided",
        parent=url_ref_node,
        critical=True
    )

    url_matches_leaf = evaluator.add_leaf(
        id="url_matches_space",
        desc="Provided URL corresponds to the coworking space (official site or verified listing) in Austin, TX",
        parent=url_ref_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The provided URL is the official website or a recognized listing page for the coworking space '{space.name or ''}' in Austin, Texas.",
        node=url_matches_leaf,
        sources=sources,
        additional_instruction=(
            "Accept pages that clearly represent the specific coworking space location. "
            "Recognized listing marketplaces include brands like WeWork, Industrious, Regus, LiquidSpace, Coworker, Deskpass, Upsuite, Office Evolution, etc. "
            "The page should clearly reference the Austin, TX location. Minor name variations are acceptable."
        ),
        extra_prerequisites=[url_provided_node]
    )

    # -------------------- Space Identification (Critical) ------------- #
    ident_node = evaluator.add_parallel(
        id="Space_Identification",
        desc="The official name and complete street address of a coworking space located in downtown Austin, Texas is provided",
        parent=solution_node,
        critical=True
    )

    name_addr_present = evaluator.add_custom_node(
        result=(bool(space.name) and bool(space.address)),
        id="name_address_provided",
        desc="Official name and complete street address are provided",
        parent=ident_node,
        critical=True
    )

    name_addr_verified = evaluator.add_leaf(
        id="name_address_verified",
        desc="The space name and full street address are supported by the provided URL(s)",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The coworking space is named '{space.name or ''}' and its street address is '{space.address or ''}'.",
        node=name_addr_verified,
        sources=sources,
        additional_instruction=(
            "Verify that the page shows this exact space name and the same full street address. "
            "Allow minor formatting variations (e.g., Suite vs Ste, punctuation, abbreviations)."
        ),
        extra_prerequisites=[url_provided_node, name_addr_present]
    )

    downtown_verified = evaluator.add_leaf(
        id="located_in_downtown",
        desc="The space is located in downtown Austin, Texas",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim="This coworking space is located in Downtown Austin, Texas.",
        node=downtown_verified,
        sources=sources,
        additional_instruction=(
            "Treat the location as 'Downtown Austin' if the page explicitly states Downtown or CBD, "
            "or shows a downtown district (2nd Street District, Congress Ave, Warehouse District, Market District, Seaholm, Rainey Street), "
            "or shows the 78701 ZIP code. Use only information on the webpage/screenshot."
        ),
        extra_prerequisites=[url_provided_node, name_addr_present]
    )

    # -------------------- Capacity Verification (Critical) ------------ #
    capacity_node = evaluator.add_parallel(
        id="Capacity_Verification",
        desc="Capacity information is provided confirming the space can accommodate at least 18 people",
        parent=solution_node,
        critical=True
    )

    capacity_present = evaluator.add_custom_node(
        result=bool(space.capacity_info and space.capacity_info.strip()),
        id="capacity_info_provided",
        desc="Capacity information is provided in the answer",
        parent=capacity_node,
        critical=True
    )

    capacity_meets = evaluator.add_leaf(
        id="capacity_meets_18",
        desc="The space can accommodate at least 18 people",
        parent=capacity_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "Based on the content of the provided page(s), this coworking space can accommodate a team of at least 18 people "
            "via dedicated desks and/or private offices (or clearly sufficient total square footage)."
        ),
        node=capacity_meets,
        sources=sources,
        additional_instruction=(
            f"Capacity reference from the answer: '{space.capacity_info or ''}'. "
            "Look for explicit counts like '18+ dedicated desks', 'team suites/private offices for 18+', or multiple private offices whose total seats reach 18. "
            "If only square footage is provided, judge whether it clearly supports 18 people. If ambiguous, mark not supported."
        ),
        extra_prerequisites=[url_provided_node, capacity_present]
    )

    # -------------------- Workspace Type Confirmation (Critical) ------ #
    workspace_node = evaluator.add_parallel(
        id="Workspace_Type_Confirmation",
        desc="The space offers dedicated desks or private office options (not just hot-desking)",
        parent=solution_node,
        critical=True
    )

    workspace_present = evaluator.add_custom_node(
        result=bool(space.workspace_options and space.workspace_options.strip()),
        id="workspace_options_provided",
        desc="Workspace options information is provided",
        parent=workspace_node,
        critical=True
    )

    workspace_confirmed = evaluator.add_leaf(
        id="workspace_options_confirmed",
        desc="Dedicated desks or private office options are confirmed",
        parent=workspace_node,
        critical=True
    )
    await evaluator.verify(
        claim="The space offers dedicated desks or private offices (not just hot desk/day pass).",
        node=workspace_confirmed,
        sources=sources,
        additional_instruction=(
            f"Evidence snippet from answer: '{space.workspace_options or ''}'. "
            "Confirm that at least one of 'dedicated desk', 'private office', 'team suite' is available."
        ),
        extra_prerequisites=[url_provided_node, workspace_present]
    )

    # -------------------- Meeting Room Availability (Critical) -------- #
    meeting_node = evaluator.add_parallel(
        id="Meeting_Room_Availability",
        desc="The space includes conference room or meeting room facilities available for use",
        parent=solution_node,
        critical=True
    )

    meeting_present = evaluator.add_custom_node(
        result=bool(space.meeting_rooms and space.meeting_rooms.strip()),
        id="meeting_info_provided",
        desc="Meeting/conference room information is provided",
        parent=meeting_node,
        critical=True
    )

    meeting_confirmed = evaluator.add_leaf(
        id="meeting_rooms_confirmed",
        desc="Meeting or conference rooms are available for team use",
        parent=meeting_node,
        critical=True
    )
    await evaluator.verify(
        claim="The coworking space provides meeting or conference rooms available for booking/use by members or teams.",
        node=meeting_confirmed,
        sources=sources,
        additional_instruction=(
            f"Evidence snippet from answer: '{space.meeting_rooms or ''}'. "
            "Look for terms like 'meeting rooms', 'conference rooms', 'boardroom', 'bookable rooms'."
        ),
        extra_prerequisites=[url_provided_node, meeting_present]
    )

    # -------------------- Parking Information (Critical) -------------- #
    parking_node = evaluator.add_parallel(
        id="Parking_Information",
        desc="Information about parking arrangements is provided",
        parent=solution_node,
        critical=True
    )

    parking_present = evaluator.add_custom_node(
        result=bool(space.parking and space.parking.strip()),
        id="parking_info_provided",
        desc="Parking information is provided",
        parent=parking_node,
        critical=True
    )

    parking_verified = evaluator.add_leaf(
        id="parking_verified",
        desc="Parking arrangements (on-site or nearby options) are supported by the page(s)",
        parent=parking_node,
        critical=True
    )
    await evaluator.verify(
        claim="There are described parking arrangements for this location (on-site, garage, nearby lots, or specific details).",
        node=parking_verified,
        sources=sources,
        additional_instruction=(
            f"Evidence snippet from answer: '{space.parking or ''}'. "
            "Check for a 'Parking' section or mentions of garage/lot/validated parking."
        ),
        extra_prerequisites=[url_provided_node, parking_present]
    )

    # -------------------- Internet Connectivity (Critical) ------------ #
    internet_node = evaluator.add_parallel(
        id="Internet_Connectivity",
        desc="High-speed internet or WiFi is available at the space",
        parent=solution_node,
        critical=True
    )

    internet_present = evaluator.add_custom_node(
        result=bool(space.internet and space.internet.strip()),
        id="internet_info_provided",
        desc="Internet/WiFi information is provided",
        parent=internet_node,
        critical=True
    )

    internet_verified = evaluator.add_leaf(
        id="internet_verified",
        desc="High-speed internet or WiFi availability is supported by the page(s)",
        parent=internet_node,
        critical=True
    )
    await evaluator.verify(
        claim="The coworking space provides high-speed internet or WiFi for members.",
        node=internet_verified,
        sources=sources,
        additional_instruction=(
            f"Evidence snippet from answer: '{space.internet or ''}'. "
            "Accept 'high-speed internet', 'WiFi', 'fiber', 'gigabit', 'secure Wi-Fi', etc."
        ),
        extra_prerequisites=[url_provided_node, internet_present]
    )

    # -------------------- Membership Flexibility (Critical) ----------- #
    membership_node = evaluator.add_parallel(
        id="Membership_Flexibility",
        desc="Information about membership term flexibility is provided",
        parent=solution_node,
        critical=True
    )

    membership_present = evaluator.add_custom_node(
        result=bool(space.membership_terms and space.membership_terms.strip()),
        id="membership_info_provided",
        desc="Membership term flexibility information is provided",
        parent=membership_node,
        critical=True
    )

    membership_verified = evaluator.add_leaf(
        id="membership_flex_verified",
        desc="Flexible membership terms (month-to-month or similar) are supported by the page(s)",
        parent=membership_node,
        critical=True
    )
    await evaluator.verify(
        claim="The coworking space offers flexible membership terms (e.g., month-to-month, no long-term contracts).",
        node=membership_verified,
        sources=sources,
        additional_instruction=(
            f"Evidence snippet from answer: '{space.membership_terms or ''}'. "
            "Look for 'month-to-month', 'flexible terms', 'no long-term commitment', or similar phrases."
        ),
        extra_prerequisites=[url_provided_node, membership_present]
    )

    # -------------------- Pricing Information (Non-Critical) ---------- #
    pricing_node = evaluator.add_parallel(
        id="Pricing_Information",
        desc="Pricing information for dedicated desks or private offices is provided or referenced",
        parent=solution_node,
        critical=False
    )

    pricing_present = evaluator.add_custom_node(
        result=bool(space.pricing and space.pricing.strip()),
        id="pricing_info_provided",
        desc="Pricing information is provided in the answer",
        parent=pricing_node,
        critical=True  # gate internal pricing verification
    )

    pricing_verified = evaluator.add_leaf(
        id="pricing_supported",
        desc="Pricing for dedicated desks or private offices is supported by the page(s)",
        parent=pricing_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The page shows pricing for dedicated desks or private offices consistent with: '{space.pricing or ''}'.",
        node=pricing_verified,
        sources=sources,
        additional_instruction=(
            "The pricing must be explicit (e.g., '$X/month per dedicated desk' or 'Private office starting at $Y/month'). "
            "Do not accept 'contact us' or pricing hidden behind forms as explicit pricing."
        ),
        extra_prerequisites=[url_provided_node, pricing_present]
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
    Evaluate an answer for the downtown Austin coworking benchmark task.
    """
    # Initialize evaluator with a parallel root as checks are independent
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

    # Extract coworking space details from the answer
    extract_space: CoworkingSpaceInfo = await evaluator.extract(
        prompt=prompt_extract_space_info(),
        template_class=CoworkingSpaceInfo,
        extraction_name="coworking_space_info"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extract_space)

    # Return structured evaluation summary
    return evaluator.get_summary()