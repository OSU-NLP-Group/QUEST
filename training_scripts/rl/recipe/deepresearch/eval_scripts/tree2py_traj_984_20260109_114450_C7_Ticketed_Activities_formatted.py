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
TASK_ID = "ny_metro_venue_profile_one_arena"
TASK_DESCRIPTION = (
    "You are preparing a comprehensive venue profile for concert promoters evaluating major arenas in the New York "
    "metropolitan area. Identify ONE major concert arena in the New York metropolitan area with a seating capacity "
    "between 17,000 and 20,000 for concerts, and provide detailed information about the following 12 venue attributes: "
    "(1) The venue's official name, (2) The venue's specific location (borough or city, and state), (3) The exact seating "
    "capacity specifically for concert events, (4) The total number of distinct seating levels or sections within the venue, "
    "(5) Confirmation that wheelchair accessible seating is available, with details on the policy, (6) The companion seat "
    "policy for individuals using wheelchair accessible seating, (7) The general age restriction policy for attending events, "
    "(8) The ticketing policy for children under 2 years of age, (9) The specific ID requirements for picking up tickets at "
    "will call, (10) Information about parking availability near or at the venue, (11) Confirmation of whether VIP entrance "
    "access and amenities are available, (12) Contact information (phone number) for the venue's accessibility services "
    "department. For each attribute, provide specific and verifiable information with supporting reference URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AttributeWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WillCallRequirements(BaseModel):
    requires_photo_id_with_signature: Optional[bool] = None
    requires_credit_card_used: Optional[bool] = None
    details: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VenueProfile(BaseModel):
    # 1. Official name
    venue_name: Optional[AttributeWithSources] = None
    # 2. Location (borough/city and state)
    venue_location: Optional[AttributeWithSources] = None
    # 3. Concert seating capacity (exact)
    concert_capacity: Optional[AttributeWithSources] = None
    # 4. Total number of distinct seating levels/sections
    seating_levels: Optional[AttributeWithSources] = None
    # 5. Wheelchair accessible seating availability and policy/details
    wheelchair_accessible_seating: Optional[AttributeWithSources] = None
    # 6. Companion seat policy (accessible seating)
    companion_seat_policy: Optional[AttributeWithSources] = None
    # 7. General age restriction policy
    age_restrictions: Optional[AttributeWithSources] = None
    # 8. Children under 2 ticketing policy
    children_under_2_policy: Optional[AttributeWithSources] = None
    # 9. Will call ticket pickup ID requirements (with specific constraints)
    will_call_requirements: Optional[WillCallRequirements] = None
    # 10. Parking availability
    parking_availability: Optional[AttributeWithSources] = None
    # 11. VIP entrance access and amenities
    vip_amenities: Optional[AttributeWithSources] = None
    # 12. Accessibility services contact phone number
    accessibility_contact_phone: Optional[AttributeWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_profile() -> str:
    return """
Extract a single venue profile from the answer text. If multiple venues are mentioned, extract ONLY the first one that fits a major concert arena in the New York metropolitan area.

For EACH of the 12 attributes below, return:
- value: The exact text the answer provides for this attribute (string). If the attribute is a yes/no item, you can still record a short descriptive sentence (e.g., "Yes, wheelchair accessible seating is available …").
- sources: An array of URL strings cited in the answer that specifically support that attribute. Use ONLY URLs explicitly present in the answer (plain URLs or markdown links). Do NOT invent URLs. If none are provided in the answer, return an empty array.

Additionally, for will-call requirements, explicitly extract two booleans based on the answer’s text:
- requires_photo_id_with_signature: true/false/null
- requires_credit_card_used: true/false/null
- details: optional short text summary of the will-call policy (string or null)

Return a single JSON object with this exact structure:

{
  "venue_name": { "value": string|null, "sources": string[] },
  "venue_location": { "value": string|null, "sources": string[] },
  "concert_capacity": { "value": string|null, "sources": string[] },
  "seating_levels": { "value": string|null, "sources": string[] },
  "wheelchair_accessible_seating": { "value": string|null, "sources": string[] },
  "companion_seat_policy": { "value": string|null, "sources": string[] },
  "age_restrictions": { "value": string|null, "sources": string[] },
  "children_under_2_policy": { "value": string|null, "sources": string[] },
  "will_call_requirements": {
    "requires_photo_id_with_signature": boolean|null,
    "requires_credit_card_used": boolean|null,
    "details": string|null,
    "sources": string[]
  },
  "parking_availability": { "value": string|null, "sources": string[] },
  "vip_amenities": { "value": string|null, "sources": string[] },
  "accessibility_contact_phone": { "value": string|null, "sources": string[] }
}

Important rules:
- The venue_location.value should include the borough or city and the state (e.g., "Brooklyn, NY", "Newark, NJ").
- The concert_capacity.value should be specifically the concert seating capacity (not basketball/hockey capacity unless the answer explicitly equates them for concerts).
- For seating_levels.value, capture the stated count or a clear description (e.g., "100/200/300 levels plus floor").
- For boolean fields in will_call_requirements, infer true/false only if the answer explicitly states it. Otherwise return null.
- If the answer omits any attribute or lacks a supporting URL for that attribute, set its value to null or sources to [] accordingly (do not fabricate).
"""


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _nonempty_value_and_sources(attr: Optional[AttributeWithSources]) -> bool:
    return bool(attr and attr.value and attr.value.strip() and attr.sources and len(attr.sources) > 0)


def _nonempty_sources_only(attr: Optional[AttributeWithSources]) -> bool:
    return bool(attr and attr.sources and len(attr.sources) > 0)


def _parse_int_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    t = text.strip().lower()

    # Handle forms like "18k"
    m_k = re.search(r"(\d{1,3})(?:\s*)k\b", t)
    if m_k:
        try:
            return int(m_k.group(1)) * 1000
        except Exception:
            pass

    # Handle ranges like "17,500-18,000" or "17,500 – 19,000"
    m_range = re.search(r"(\d{1,3}(?:,\d{3})*|\d+)\s*[–-]\s*(\d{1,3}(?:,\d{3})*|\d+)", t)
    if m_range:
        try:
            n1 = int(m_range.group(1).replace(",", ""))
            # We could choose to take the concert-specific number if clarified; in range check,
            # using the first number is conservative.
            return n1
        except Exception:
            pass

    # General number, e.g., "18,006"
    m_num = re.search(r"(\d{1,3}(?:,\d{3})*|\d+)", t)
    if m_num:
        try:
            return int(m_num.group(1).replace(",", ""))
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_venue_name_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="venue_name",
        desc="Provide the venue's official name with supporting reference URL(s).",
        parent=parent,
        critical=True
    )

    provided = evaluator.add_custom_node(
        result=_nonempty_value_and_sources(profile.venue_name),
        id="venue_name_provided",
        desc="Official name is provided with at least one supporting URL.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="venue_name_supported",
        desc="The stated official venue name is supported by the cited source(s).",
        parent=node,
        critical=True
    )
    name_value = profile.venue_name.value if profile.venue_name and profile.venue_name.value else ""
    sources = profile.venue_name.sources if profile.venue_name else []
    claim = f"The venue's official name is '{name_value}'."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm that the cited page(s) show this exact official venue name. Minor capitalization or punctuation differences are acceptable."
    )


async def build_location_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="venue_location",
        desc="Provide the venue's specific location (borough/city and state) and confirm it is in the New York metropolitan area, with supporting reference URL(s).",
        parent=parent,
        critical=True
    )

    provided = evaluator.add_custom_node(
        result=_nonempty_value_and_sources(profile.venue_location),
        id="venue_location_provided",
        desc="Location (borough/city and state) is provided with at least one supporting URL.",
        parent=node,
        critical=True
    )

    leaf_loc = evaluator.add_leaf(
        id="venue_location_supported",
        desc="The stated borough/city and state for the venue are supported by the cited source(s).",
        parent=node,
        critical=True
    )
    loc_value = profile.venue_location.value if profile.venue_location and profile.venue_location.value else ""
    sources = profile.venue_location.sources if profile.venue_location else []
    claim_loc = f"The venue is located at '{loc_value}'."
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        sources=sources,
        additional_instruction="Verify that the page explicitly shows the borough or city and the state for the venue."
    )

    # Metro area inference check (kept as a supported leaf using the same sources; allow reasonable inference)
    leaf_metro = evaluator.add_leaf(
        id="venue_location_in_ny_metro",
        desc="The venue's stated location is within the New York metropolitan area.",
        parent=node,
        critical=True
    )
    claim_metro = f"The location '{loc_value}' is within the New York metropolitan area."
    await evaluator.verify(
        claim=claim_metro,
        node=leaf_metro,
        sources=sources,
        additional_instruction=(
            "Use the location shown on the page (e.g., a NYC borough like Brooklyn/Queens/Manhattan/Bronx/Staten Island, "
            "or nearby cities such as Newark, Jersey City, East Rutherford, Elmont, Uniondale, etc.) to conclude whether it is "
            "within the New York metropolitan area. It is acceptable to use common geographic knowledge to make this determination."
        )
    )


async def build_capacity_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="concert_capacity",
        desc="Provide the exact seating capacity specifically for concerts, and it must be between 17,000 and 20,000 inclusive, with supporting reference URL(s).",
        parent=parent,
        critical=True
    )

    provided = evaluator.add_custom_node(
        result=_nonempty_value_and_sources(profile.concert_capacity),
        id="concert_capacity_provided",
        desc="Concert seating capacity is provided with at least one supporting URL.",
        parent=node,
        critical=True
    )

    leaf_supported = evaluator.add_leaf(
        id="concert_capacity_supported",
        desc="The stated concert seating capacity value is supported by the cited source(s).",
        parent=node,
        critical=True
    )
    cap_value = profile.concert_capacity.value if profile.concert_capacity and profile.concert_capacity.value else ""
    cap_sources = profile.concert_capacity.sources if profile.concert_capacity else []
    claim_cap = f"The venue's concert seating capacity is {cap_value}."
    await evaluator.verify(
        claim=claim_cap,
        node=leaf_supported,
        sources=cap_sources,
        additional_instruction="Check that the page explicitly states the concert seating capacity number. If multiple capacities are listed (e.g., basketball vs concerts), ensure the value corresponds to concerts."
    )

    # Numeric range check (critical)
    cap_num = _parse_int_from_text(cap_value)
    in_range = bool(cap_num is not None and 17000 <= cap_num <= 20000)
    evaluator.add_custom_node(
        result=in_range,
        id="concert_capacity_in_range",
        desc=f"Concert capacity numeric value is within 17,000–20,000 inclusive (parsed: {cap_num}).",
        parent=node,
        critical=True
    )


async def build_seating_levels_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="seating_levels",
        desc="Provide the total number of distinct seating levels/sections (showing the venue has multiple distinct levels/sections), with supporting reference URL(s).",
        parent=parent,
        critical=True
    )

    provided = evaluator.add_custom_node(
        result=_nonempty_value_and_sources(profile.seating_levels),
        id="seating_levels_provided",
        desc="Total number of distinct seating levels/sections provided with at least one supporting URL.",
        parent=node,
        critical=True
    )

    leaf = evaluator.add_leaf(
        id="seating_levels_supported",
        desc="The stated total number of distinct seating levels/sections is supported by the cited source(s).",
        parent=node,
        critical=True
    )
    val = profile.seating_levels.value if profile.seating_levels and profile.seating_levels.value else ""
    sources = profile.seating_levels.sources if profile.seating_levels else []
    claim = f"The venue has {val} distinct seating levels or sections for events."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=(
            "Interpret 'levels/sections' to include tiers such as 100/200/300 levels and floor (if used for seating). "
            "If the source lists distinct tiers or balcony/mezzanine/etc., it supports the claim."
        )
    )


async def build_accessible_seating_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="wheelchair_accessible_seating",
        desc="Confirm wheelchair accessible seating is available and provide policy/details indicating ADA-compliant accessibility, with supporting reference URL(s).",
        parent=parent,
        critical=True
    )
    provided = evaluator.add_custom_node(
        result=_nonempty_value_and_sources(profile.wheelchair_accessible_seating),
        id="wheelchair_accessible_seating_provided",
        desc="Statement and details about wheelchair accessible seating provided with at least one supporting URL.",
        parent=node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="wheelchair_accessible_seating_supported",
        desc="Wheelchair accessible seating availability and policy/details are supported by the cited source(s).",
        parent=node,
        critical=True
    )
    val = profile.wheelchair_accessible_seating.value if profile.wheelchair_accessible_seating and profile.wheelchair_accessible_seating.value else ""
    sources = profile.wheelchair_accessible_seating.sources if profile.wheelchair_accessible_seating else []
    claim = f"Wheelchair accessible seating is available for the venue; details/policy: {val}"
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm ADA-compliant accessible seating exists at the venue and the policy/details mentioned are reflected on the cited page(s)."
    )


async def build_companion_policy_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="companion_seat_policy",
        desc="Provide the companion seat policy for wheelchair accessible seating, with supporting reference URL(s).",
        parent=parent,
        critical=True
    )
    provided = evaluator.add_custom_node(
        result=_nonempty_value_and_sources(profile.companion_seat_policy),
        id="companion_seat_policy_provided",
        desc="Companion seat policy is provided with at least one supporting URL.",
        parent=node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="companion_seat_policy_supported",
        desc="Companion seat policy (for accessible seating) is supported by the cited source(s).",
        parent=node,
        critical=True
    )
    val = profile.companion_seat_policy.value if profile.companion_seat_policy and profile.companion_seat_policy.value else ""
    sources = profile.companion_seat_policy.sources if profile.companion_seat_policy else []
    claim = f"Companion seat policy (accessible seating): {val}"
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify that the cited page(s) describe the companion seat policy for wheelchair accessible seating."
    )


async def build_age_restrictions_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="age_restrictions",
        desc="Provide the general age restriction policy for attending events, with supporting reference URL(s).",
        parent=parent,
        critical=True
    )
    provided = evaluator.add_custom_node(
        result=_nonempty_value_and_sources(profile.age_restrictions),
        id="age_restrictions_provided",
        desc="General age restriction policy is provided with at least one supporting URL.",
        parent=node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="age_restrictions_supported",
        desc="General age restriction policy is supported by the cited source(s).",
        parent=node,
        critical=True
    )
    val = profile.age_restrictions.value if profile.age_restrictions and profile.age_restrictions.value else ""
    sources = profile.age_restrictions.sources if profile.age_restrictions else []
    claim = f"General age restriction policy for attending events: {val}"
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="If policies vary by event, the page should say so. Otherwise, confirm the general policy text matches."
    )


async def build_children_under2_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="children_under_2_policy",
        desc="Provide the ticketing policy for children under 2 years of age, with supporting reference URL(s).",
        parent=parent,
        critical=True
    )
    provided = evaluator.add_custom_node(
        result=_nonempty_value_and_sources(profile.children_under_2_policy),
        id="children_under_2_policy_provided",
        desc="Policy for children under 2 is provided with at least one supporting URL.",
        parent=node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="children_under_2_policy_supported",
        desc="Children under 2 ticketing policy is supported by the cited source(s).",
        parent=node,
        critical=True
    )
    val = profile.children_under_2_policy.value if profile.children_under_2_policy and profile.children_under_2_policy.value else ""
    sources = profile.children_under_2_policy.sources if profile.children_under_2_policy else []
    claim = f"Children under 2 ticketing policy: {val}"
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for statements like lap-sitting allowances, needing a ticket, or event-based policies for children under 2."
    )


async def build_will_call_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="will_call_requirements",
        desc="Provide will call ticket pickup ID requirements that include all specified constraint elements, with supporting reference URL(s).",
        parent=parent,
        critical=True
    )

    # Make sure sources exist
    will_call = profile.will_call_requirements or WillCallRequirements()
    provided = evaluator.add_custom_node(
        result=bool(will_call and will_call.sources and len(will_call.sources) > 0),
        id="will_call_requirements_provided",
        desc="Will call requirements sources are provided (at least one URL).",
        parent=node,
        critical=True
    )

    # Photo ID with signature requirement
    leaf_id = evaluator.add_leaf(
        id="will_call_photo_id_signature",
        desc="Will call pickup requires valid photo ID with signature.",
        parent=node,
        critical=True
    )
    claim_id = (
        "Will call ticket pickup requires a valid photo ID with a signature (e.g., government-issued photo ID). "
        "Equivalent phrasings such as 'valid photo ID' or 'government-issued photo identification' may satisfy this requirement."
    )
    await evaluator.verify(
        claim=claim_id,
        node=leaf_id,
        sources=will_call.sources,
        additional_instruction="Pass if the policy explicitly states requiring a valid photo ID at will call. The word 'signature' may be implied by 'government-issued photo ID'."
    )

    # Credit card used for purchase requirement
    leaf_cc = evaluator.add_leaf(
        id="will_call_credit_card",
        desc="Will call pickup requires the credit card used for purchase.",
        parent=node,
        critical=True
    )
    claim_cc = "Will call ticket pickup requires presenting the original credit card used to make the purchase."
    await evaluator.verify(
        claim=claim_cc,
        node=leaf_cc,
        sources=will_call.sources,
        additional_instruction="Pass if the policy states the buyer must present the original purchasing credit card (or equivalent phrasing)."
    )


async def build_parking_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="parking_availability",
        desc="Provide information about parking availability near or at the venue, with supporting reference URL(s).",
        parent=parent,
        critical=True
    )
    provided = evaluator.add_custom_node(
        result=_nonempty_value_and_sources(profile.parking_availability),
        id="parking_availability_provided",
        desc="Parking information provided with at least one supporting URL.",
        parent=node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="parking_availability_supported",
        desc="Parking availability information is supported by the cited source(s).",
        parent=node,
        critical=True
    )
    val = profile.parking_availability.value if profile.parking_availability and profile.parking_availability.value else ""
    sources = profile.parking_availability.sources if profile.parking_availability else []
    claim = f"Parking availability and details: {val}"
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Verify whether on-site or nearby parking exists and any key details mentioned (e.g., garages, lots, no on-site parking)."
    )


async def build_vip_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="vip_amenities",
        desc="Confirm whether VIP entrance access and amenities are available, with supporting reference URL(s).",
        parent=parent,
        critical=True
    )
    provided = evaluator.add_custom_node(
        result=_nonempty_value_and_sources(profile.vip_amenities),
        id="vip_amenities_provided",
        desc="VIP entrances/access and/or amenities information provided with at least one supporting URL.",
        parent=node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="vip_amenities_supported",
        desc="VIP entrance access and amenities are supported by the cited source(s).",
        parent=node,
        critical=True
    )
    val = profile.vip_amenities.value if profile.vip_amenities and profile.vip_amenities.value else ""
    sources = profile.vip_amenities.sources if profile.vip_amenities else []
    claim = f"VIP entrance access and/or amenities availability: {val}"
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Look for mentions of VIP entrances, clubs/lounges, suites benefits, premium amenities, priority lines, etc."
    )


async def build_accessibility_contact_checks(evaluator: Evaluator, parent, profile: VenueProfile):
    node = evaluator.add_parallel(
        id="accessibility_contact",
        desc="Provide contact information for the accessibility services department including a phone number, with supporting reference URL(s).",
        parent=parent,
        critical=True
    )
    provided = evaluator.add_custom_node(
        result=_nonempty_value_and_sources(profile.accessibility_contact_phone),
        id="accessibility_contact_provided",
        desc="Accessibility services contact phone number is provided with at least one supporting URL.",
        parent=node,
        critical=True
    )
    leaf = evaluator.add_leaf(
        id="accessibility_contact_supported",
        desc="Accessibility services department phone number is supported by the cited source(s).",
        parent=node,
        critical=True
    )
    val = profile.accessibility_contact_phone.value if profile.accessibility_contact_phone and profile.accessibility_contact_phone.value else ""
    sources = profile.accessibility_contact_phone.sources if profile.accessibility_contact_phone else []
    claim = f"The accessibility services (or guest services for accessibility) phone number is {val}."
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction="Confirm the phone number corresponds to accessibility services, guest services for accessibility, or a dedicated ADA/accessibility contact listed by the venue."
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
    Evaluate an answer for the 'one NY metro arena profile with 12 attributes' task.
    """
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

    # Critical task node to enforce all required attributes
    task_node = evaluator.add_parallel(
        id="task_requirements",
        desc="Identify ONE NY-metro concert arena (concert capacity 17,000–20,000) and provide all 12 attributes with supporting URLs.",
        parent=root,
        critical=True
    )

    # Extract venue profile from the answer
    profile: VenueProfile = await evaluator.extract(
        prompt=prompt_extract_venue_profile(),
        template_class=VenueProfile,
        extraction_name="venue_profile"
    )

    # Build verification subtrees for each required attribute
    await build_venue_name_checks(evaluator, task_node, profile)
    await build_location_checks(evaluator, task_node, profile)
    await build_capacity_checks(evaluator, task_node, profile)
    await build_seating_levels_checks(evaluator, task_node, profile)
    await build_accessible_seating_checks(evaluator, task_node, profile)
    await build_companion_policy_checks(evaluator, task_node, profile)
    await build_age_restrictions_checks(evaluator, task_node, profile)
    await build_children_under2_checks(evaluator, task_node, profile)
    await build_will_call_checks(evaluator, task_node, profile)
    await build_parking_checks(evaluator, task_node, profile)
    await build_vip_checks(evaluator, task_node, profile)
    await build_accessibility_contact_checks(evaluator, task_node, profile)

    # Final structured summary
    return evaluator.get_summary()