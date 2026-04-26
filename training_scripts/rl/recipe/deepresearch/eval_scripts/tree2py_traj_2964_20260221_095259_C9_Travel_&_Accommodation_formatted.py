import asyncio
import logging
from typing import Optional, List, Dict, Any, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "destin_beachfront_resort_2026"
TASK_DESCRIPTION = (
    "A corporate event planner needs to book a beachfront resort hotel in Destin, Florida for a 3-day company retreat in May 2026. "
    "The group consists of 50 employees who will arrive via Destin-Fort Walton Beach Airport (VPS). The hotel must meet ALL of the following requirements: "
    "(1) Provide wheelchair-accessible guest rooms with roll-in shower facilities, (2) Accept service animals (one employee travels with a service dog), "
    "(3) Offer at least 5,000 square feet of meeting and conference space, (4) Have wheelchair-accessible fitness center facilities, "
    "(5) Provide beachside or oceanfront dining options on the property, (6) Be located directly on the beach in Destin (beachfront property), "
    "(7) Have a cancellation policy allowing cancellations at least 48 hours before check-in without full forfeiture of deposit. "
    "Identify a hotel that satisfies ALL seven requirements above. For your answer, provide: the hotel name and complete street address, "
    "the official hotel website URL, and for EACH of the seven requirements listed above, provide specific evidence (description of the relevant feature/policy) "
    "with at least one supporting reference URL demonstrating how the hotel meets that particular requirement."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementEvidence(BaseModel):
    evidence_text: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class AllRequirementsExtraction(BaseModel):
    req1_accessible_rooms_rollin_shower: Optional[RequirementEvidence] = None
    req2_accepts_service_animals: Optional[RequirementEvidence] = None
    req3_meeting_space_5000_sqft: Optional[RequirementEvidence] = None
    req4_accessible_fitness_center: Optional[RequirementEvidence] = None
    req5_beachside_or_oceanfront_dining_on_property: Optional[RequirementEvidence] = None
    req6_beachfront_direct_beach_access_in_destin: Optional[RequirementEvidence] = None
    req7_cancellation_48h_no_full_deposit_forfeiture: Optional[RequirementEvidence] = None
    req8_accessible_from_vps: Optional[RequirementEvidence] = None


class HotelBasicExtraction(BaseModel):
    hotel_name: Optional[str] = None
    full_address: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    official_website_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_basic_info() -> str:
    return """
    Extract the single primary hotel the answer identifies as meeting all constraints (if multiple are mentioned, choose the first that is in Destin, Florida).
    Return:
    - hotel_name: the specific hotel/resort name.
    - full_address: the complete street address as a single line, including street, city, state, ZIP (if provided).
    - street_address: the street line(s) only (e.g., "123 Beach Blvd"), excluding city/state/ZIP.
    - city: the city.
    - state: the state (use "FL" if abbreviated or "Florida" if spelled out, as extracted).
    - postal_code: the ZIP/postal code (5 digits if available).
    - official_website_url: the official hotel website URL (not a third-party listing).
    Only extract information explicitly present in the answer text. Do not invent or infer missing parts.
    """


def prompt_extract_requirements() -> str:
    return """
    For the same identified hotel, extract the evidence and at least one reference URL that the answer provides for each constraint. For each item, return:
    - evidence_text: a concise description/quote from the answer relevant to the requirement.
    - reference_urls: an array of one or more URLs that the answer cites for that requirement. Only include URLs explicitly present in the answer.
    If a requirement lacks evidence or URLs in the answer, set evidence_text to null and return an empty reference_urls array for that requirement.
    Return fields with the following exact keys:
    - req1_accessible_rooms_rollin_shower
    - req2_accepts_service_animals
    - req3_meeting_space_5000_sqft
    - req4_accessible_fitness_center
    - req5_beachside_or_oceanfront_dining_on_property
    - req6_beachfront_direct_beach_access_in_destin
    - req7_cancellation_48h_no_full_deposit_forfeiture
    - req8_accessible_from_vps
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _text_present(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip() != "")


def _unique_urls(url_lists: List[List[str]]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for urls in url_lists:
        for u in urls:
            if not _text_present(u):
                continue
            uu = u.strip()
            if uu not in seen:
                seen.add(uu)
                result.append(uu)
    return result


def _destin_fl_address_ok(city: Optional[str], state: Optional[str], postal_code: Optional[str], street: Optional[str]) -> bool:
    if not (_text_present(city) and _text_present(state) and _text_present(postal_code) and _text_present(street)):
        return False
    city_ok = city.strip().lower() == "destin"
    st = state.strip().lower()
    state_ok = (st == "fl") or (st == "florida")
    zip_ok = any(ch.isdigit() for ch in postal_code)
    return city_ok and state_ok and zip_ok


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_requirement(
    evaluator: Evaluator,
    parent_node,
    *,
    node_id_prefix: str,
    node_desc: str,
    hotel_name: str,
    evidence: Optional[RequirementEvidence],
    claim: str,
    add_ins: str
) -> None:
    """
    Build a sequential verification sub-tree for one requirement:
    1) Provided: evidence text and >=1 URL present
    2) Supported: verify claim against provided URLs
    """
    seq_node = evaluator.add_sequential(
        id=node_id_prefix,
        desc=node_desc,
        parent=parent_node,
        critical=True
    )

    # Existence: evidence + ≥1 URL
    provided_ok = evidence is not None and _text_present(evidence.evidence_text) and bool(evidence.reference_urls)
    evaluator.add_custom_node(
        result=provided_ok,
        id=f"{node_id_prefix}_provided",
        desc=f"{node_desc} - evidence and ≥1 reference URL provided",
        parent=seq_node,
        critical=True
    )

    # Supported by sources
    supported_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_supported",
        desc=f"{node_desc} - supported by the provided reference URL(s)",
        parent=seq_node,
        critical=True
    )
    urls = evidence.reference_urls if (evidence and evidence.reference_urls) else []
    await evaluator.verify(
        claim=claim.replace("{HOTEL}", hotel_name),
        node=supported_leaf,
        sources=urls,
        additional_instruction=add_ins
    )


async def _build_and_verify_requirements(
    evaluator: Evaluator,
    root_parent,
    hotel_name: str,
    reqs: AllRequirementsExtraction
) -> List[str]:
    """
    Build the 'All_Requirements_With_Evidence_And_References' subtree and verify each requirement.
    Returns the flattened list of all reference URLs for reliability checks.
    """
    all_reqs_parent = evaluator.add_parallel(
        id="All_Requirements_With_Evidence_And_References",
        desc="For each requirement, the answer provides evidence + ≥1 supporting reference URL and the sources support the claim.",
        parent=root_parent,
        critical=True
    )

    # Req 1: Accessible rooms with roll-in showers
    await _verify_requirement(
        evaluator, all_reqs_parent,
        node_id_prefix="Req1_Accessible_Rooms_RollIn_Shower",
        node_desc="Req1: Wheelchair-accessible guest rooms with roll-in showers are available",
        hotel_name=hotel_name,
        evidence=reqs.req1_accessible_rooms_rollin_shower,
        claim="The hotel {HOTEL} offers wheelchair-accessible guest rooms that include roll-in shower facilities.",
        add_ins=(
            "Verify that at least one cited page explicitly states that the hotel offers accessible rooms with roll-in showers. "
            "Accept synonymous phrasing such as 'roll-in shower', 'roll in shower', or 'wheel-in shower'."
        )
    )

    # Req 2: Accepts service animals
    await _verify_requirement(
        evaluator, all_reqs_parent,
        node_id_prefix="Req2_Accepts_Service_Animals",
        node_desc="Req2: The hotel accepts service animals",
        hotel_name=hotel_name,
        evidence=reqs.req2_accepts_service_animals,
        claim="The hotel {HOTEL} accepts service animals for guests with disabilities.",
        add_ins=(
            "Confirm that the sources explicitly allow service animals (e.g., 'service animals allowed' or 'service animals are welcome'). "
            "Do not confuse with general pet policy unless service animals are explicitly permitted regardless of pet policy."
        )
    )

    # Req 3: ≥ 5,000 sq ft meeting space
    await _verify_requirement(
        evaluator, all_reqs_parent,
        node_id_prefix="Req3_Meeting_Space_At_Least_5000_Sq_Ft",
        node_desc="Req3: The hotel offers at least 5,000 sq ft of meeting/conference space",
        hotel_name=hotel_name,
        evidence=reqs.req3_meeting_space_5000_sqft,
        claim="The hotel {HOTEL} offers at least 5,000 square feet of meeting and conference space in total.",
        add_ins=(
            "Check for total event/meeting space area across rooms (e.g., 'total event space', 'meeting space'), "
            "and verify it is ≥ 5,000 square feet. If only metric values are provided, convert approximately (e.g., 465 m² ≈ 5,000 sq ft). "
            "If multiple rooms' areas are listed separately, sum them if the page indicates total area ≥ 5,000 sq ft."
        )
    )

    # Req 4: Accessible fitness center
    await _verify_requirement(
        evaluator, all_reqs_parent,
        node_id_prefix="Req4_Accessible_Fitness_Center",
        node_desc="Req4: The hotel has wheelchair-accessible fitness center facilities",
        hotel_name=hotel_name,
        evidence=reqs.req4_accessible_fitness_center,
        claim="The hotel {HOTEL} provides a fitness center that is wheelchair-accessible (ADA-accessible).",
        add_ins=(
            "Look for explicit ADA/accessibility notes about the fitness center (e.g., elevator access, accessible entrance, "
            "accessible equipment/space). General 'fitness center' without accessibility mention is insufficient."
        )
    )

    # Req 5: Beachside or oceanfront dining on property
    await _verify_requirement(
        evaluator, all_reqs_parent,
        node_id_prefix="Req5_Beachside_Or_Oceanfront_Dining_On_Property",
        node_desc="Req5: The hotel provides beachside or oceanfront dining on the property",
        hotel_name=hotel_name,
        evidence=reqs.req5_beachside_or_oceanfront_dining_on_property,
        claim="The hotel {HOTEL} offers beachside or oceanfront dining options located on the property.",
        add_ins=(
            "Confirm that at least one on‑property restaurant/bar/venue provides beachside or oceanfront dining "
            "(e.g., 'beachfront restaurant', 'oceanfront dining', 'on the beach'). Off‑property options do not satisfy this."
        )
    )

    # Req 6: Beachfront in Destin with direct beach access
    await _verify_requirement(
        evaluator, all_reqs_parent,
        node_id_prefix="Req6_Beachfront_Direct_Beach_Access_In_Destin",
        node_desc="Req6: The hotel is beachfront in Destin with direct beach access",
        hotel_name=hotel_name,
        evidence=reqs.req6_beachfront_direct_beach_access_in_destin,
        claim="The hotel {HOTEL} is a beachfront property in Destin, Florida, with direct beach access.",
        add_ins=(
            "Verify the property is directly on the beach (not across the street) and is located in Destin, FL. "
            "Look for phrases like 'beachfront', 'private beach', 'direct beach access', 'on the beach', and references to Destin specifically."
        )
    )

    # Req 7: Cancellation policy ≥ 48h prior without full deposit forfeiture
    await _verify_requirement(
        evaluator, all_reqs_parent,
        node_id_prefix="Req7_Cancellation_Policy_48h_No_Full_Deposit_Forfeiture",
        node_desc="Req7: Cancellation allowed ≥ 48 hours before check-in without full deposit forfeiture",
        hotel_name=hotel_name,
        evidence=reqs.req7_cancellation_48h_no_full_deposit_forfeiture,
        claim=(
            "The cancellation policy for the hotel {HOTEL} allows cancellation at least 48 hours before check‑in without forfeiting the full deposit."
        ),
        add_ins=(
            "Accept policies stating free cancellation until at least 48 hours prior to arrival (or more lenient, e.g., 3+ days). "
            "Policies that keep the full deposit even if cancelled ≥ 48 hours before check-in do NOT satisfy this. "
            "If a partial fee applies but the full deposit is not forfeited when ≥ 48 hours in advance, it can be acceptable."
        )
    )

    # Req 8: Accessible from VPS (airport linkage, distance/time, transport feasibility)
    await _verify_requirement(
        evaluator, all_reqs_parent,
        node_id_prefix="Req8_Accessible_From_VPS",
        node_desc="Req8: The hotel is accessible from Destin-Fort Walton Beach Airport (VPS)",
        hotel_name=hotel_name,
        evidence=reqs.req8_accessible_from_vps,
        claim=(
            "The hotel {HOTEL} is accessible from Destin‑Fort Walton Beach Airport (VPS), as shown by provided source(s) "
            "via distance/travel time, directions, or mention of transport options linking VPS to the hotel."
        ),
        add_ins=(
            "Accept official hotel 'Getting here' pages referencing VPS, map/directions pages indicating a route from VPS to the hotel, "
            "or reputable travel sources describing transport (taxi/shuttle/rideshare) from VPS to the hotel."
        )
    )

    all_urls = _unique_urls([
        (reqs.req1_accessible_rooms_rollin_shower.reference_urls if reqs.req1_accessible_rooms_rollin_shower else []),
        (reqs.req2_accepts_service_animals.reference_urls if reqs.req2_accepts_service_animals else []),
        (reqs.req3_meeting_space_5000_sqft.reference_urls if reqs.req3_meeting_space_5000_sqft else []),
        (reqs.req4_accessible_fitness_center.reference_urls if reqs.req4_accessible_fitness_center else []),
        (reqs.req5_beachside_or_oceanfront_dining_on_property.reference_urls if reqs.req5_beachside_or_oceanfront_dining_on_property else []),
        (reqs.req6_beachfront_direct_beach_access_in_destin.reference_urls if reqs.req6_beachfront_direct_beach_access_in_destin else []),
        (reqs.req7_cancellation_48h_no_full_deposit_forfeiture.reference_urls if reqs.req7_cancellation_48h_no_full_deposit_forfeiture else []),
        (reqs.req8_accessible_from_vps.reference_urls if reqs.req8_accessible_from_vps else []),
    ])
    return all_urls


async def _verify_reference_url_reliability(
    evaluator: Evaluator,
    root_parent,
    hotel_name: Optional[str],
    all_reference_urls: List[str],
) -> None:
    """
    Build the 'Reference_URL_Source_Reliability' node and verify each URL is from a reliable source.
    """
    reliability_parent = evaluator.add_parallel(
        id="Reference_URL_Source_Reliability",
        desc="All provided reference URLs are from reliable sources (official hotel site or major reputable platforms).",
        parent=root_parent,
        critical=True
    )

    if not all_reference_urls:
        evaluator.add_custom_node(
            result=False,
            id="reliability_no_urls",
            desc="No reference URLs were provided to assess reliability",
            parent=reliability_parent,
            critical=True
        )
        return

    # Create a child leaf per URL to judge reliability individually
    for idx, url in enumerate(all_reference_urls):
        node = evaluator.add_leaf(
            id=f"reliability_url_{idx+1}",
            desc=f"Reference URL #{idx+1} is from a reliable source",
            parent=reliability_parent,
            critical=True
        )
        claim = (
            "This webpage is from an official or reputable source (e.g., the hotel's own official website, "
            "a major hotel brand domain, or a well-known travel/booking or accessibility platform) and is not an unverified random forum or personal blog."
        )
        add_ins = (
            "Evaluate reliability by examining the page itself (domain/brand indicators in text or screenshot). "
            "Examples of reputable sources include: official hotel or hotel brand domains (e.g., marriott.com, hilton.com, hyatt.com, ihg.com), "
            "major booking/travel sites (e.g., expedia.com, booking.com, tripadvisor.com, hotels.com, kayak.com, travelocity.com, orbitz.com), "
            "Google Maps/Travel, official tourism/municipal sites, or recognized accessibility resources. "
            "The hotel's own official website is always acceptable. If the page appears to be a random blog or unmoderated forum, mark as not reliable."
        )
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=url,
            additional_instruction=add_ins
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluation entry point for the Destin beachfront resort hotel task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates major sub-areas in parallel
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

    # Extract basic info and requirements in parallel
    basic_info_task = evaluator.extract(
        prompt=prompt_extract_basic_info(),
        template_class=HotelBasicExtraction,
        extraction_name="hotel_basic_info"
    )
    requirements_task = evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=AllRequirementsExtraction,
        extraction_name="requirements_evidence"
    )
    basic_info, reqs = await asyncio.gather(basic_info_task, requirements_task)

    # ------------------ Hotel Basic Info (Critical) --------------------- #
    basic_parent = evaluator.add_parallel(
        id="Hotel_Basic_Info",
        desc="Provide the required basic hotel identification information.",
        parent=root,
        critical=True
    )

    # Hotel name provided
    evaluator.add_custom_node(
        result=_text_present(basic_info.hotel_name),
        id="Hotel_Name",
        desc="A specific hotel name is provided.",
        parent=basic_parent,
        critical=True
    )

    # Complete Destin, FL address provided
    evaluator.add_custom_node(
        result=_destin_fl_address_ok(basic_info.city, basic_info.state, basic_info.postal_code, basic_info.street_address),
        id="Complete_Street_Address_In_Destin_FL",
        desc="A complete street address is provided and it is in Destin, Florida (includes street, city, state, ZIP).",
        parent=basic_parent,
        critical=True
    )

    # Official website URL provided
    evaluator.add_custom_node(
        result=_text_present(basic_info.official_website_url),
        id="Official_Hotel_Website_URL",
        desc="The official hotel website URL is provided.",
        parent=basic_parent,
        critical=True
    )

    # ------------------ Requirements with Evidence (Critical) ----------- #
    hotel_name_val = basic_info.hotel_name or ""
    all_reference_urls = await _build_and_verify_requirements(
        evaluator=evaluator,
        root_parent=root,
        hotel_name=hotel_name_val,
        reqs=reqs
    )

    # ------------------ Reference URL Reliability (Critical) ------------ #
    await _verify_reference_url_reliability(
        evaluator=evaluator,
        root_parent=root,
        hotel_name=hotel_name_val,
        all_reference_urls=all_reference_urls
    )

    # Optionally record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "total_reference_urls": len(all_reference_urls),
            "unique_reference_urls": all_reference_urls
        },
        info_type="stats",
        info_name="reference_url_statistics"
    )

    return evaluator.get_summary()