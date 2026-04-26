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
TASK_ID = "airline_in_cabin_pet_policies_comparison"
TASK_DESCRIPTION = """
I am planning to relocate from New York to California and need to transport my small dog (a 12-pound Cavalier King Charles Spaniel) and my cat in the passenger cabin. I want to compare the in-cabin pet travel policies of three major U.S. airlines: JetBlue, American Airlines, and United Airlines. For each airline, provide comprehensive documentation of their current pet travel policies, including: (1) Carrier Specifications: Maximum dimensions allowed for in-cabin pet carriers; (2) Fee Structure: Cost per pet per direction; (3) Capacity Limits: Maximum pets allowed per flight and per traveler; (4) Pet Type Restrictions: Which species and sizes are accepted for in-cabin travel; (5) Cabin Class Restrictions: Which cabin classes allow pets; (6) Transport Options: Whether cargo transport is available as an alternative; (7) Temperature Safety Policies (for American Airlines): Safe and prohibited temperature ranges, and acclimation requirements; (8) Breed Restrictions (for American Airlines): Whether brachycephalic or aggressive breeds are restricted; (9) Special Policies: Any additional restrictions such as sedation policies, aircraft type restrictions, flight duration limits, or comfort stop requirements. For each policy element, provide the specific values or requirements and include a direct URL reference to the official airline policy page where this information can be verified.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class InfoItem(BaseModel):
    text: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AirlinePolicy(BaseModel):
    # Core required for all 3 airlines
    carrier_specs: Optional[InfoItem] = None
    fee: Optional[InfoItem] = None
    capacity_per_flight: Optional[InfoItem] = None
    capacity_per_traveler: Optional[InfoItem] = None
    pet_type_restrictions: Optional[InfoItem] = None
    cabin_class_restrictions: Optional[InfoItem] = None
    cargo_alternative: Optional[InfoItem] = None

    # American-specific items (optional for others)
    temperature_safety_ranges: Optional[InfoItem] = None
    acclimation_letter_policy: Optional[InfoItem] = None
    breed_restrictions: Optional[InfoItem] = None

    # Special policy categories (for all airlines)
    sedation_policy: Optional[InfoItem] = None
    aircraft_type_restrictions: Optional[InfoItem] = None
    flight_duration_limits: Optional[InfoItem] = None
    comfort_stop_requirements: Optional[InfoItem] = None
    booking_itinerary_restrictions: Optional[InfoItem] = None


class PoliciesExtraction(BaseModel):
    jetblue: Optional[AirlinePolicy] = None
    american: Optional[AirlinePolicy] = None
    united: Optional[AirlinePolicy] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_policies() -> str:
    return """
Extract, from the provided answer text, the in-cabin pet travel policy details for three airlines: JetBlue, American Airlines, and United Airlines. For each airline, extract each policy element as plain text and the official URL(s) that the answer cites for that element.

VERY IMPORTANT RULES:
- Extract only what is explicitly present in the answer.
- For each element, include the URL(s) exactly as shown in the answer. If the answer does not include a URL for that element, return an empty list for urls.
- Prefer official airline domains only:
  • JetBlue: jetblue.com
  • American Airlines: aa.com or aacargo.com (AA Cargo is acceptable for cargo-related items)
  • United Airlines: united.com
- If the answer states “not specified,” “none,” or similar for an element, set text accordingly and still extract the cited official URL(s) if any exist.

For each airline, extract these fields (all fields are strings for 'text' and arrays of strings for 'urls'):
- carrier_specs: Maximum in-cabin carrier dimensions (L/W/H).
- fee: Cost per pet per direction (or similar phrasing).
- capacity_per_flight: Maximum in-cabin pets allowed per flight (or “not specified”).
- capacity_per_traveler: Maximum in-cabin pets per traveler (include conditions if present; or “not specified”).
- pet_type_restrictions: Accepted species (e.g., dogs/cats) and any size/weight limits.
- cabin_class_restrictions: Which cabin classes allow pets (and which do not).
- cargo_alternative: Whether cargo/checked pet transport is available or not.

American Airlines ONLY — also extract:
- temperature_safety_ranges: Safe/prohibited ambient temperature ranges and constraints (often for cargo).
- acclimation_letter_policy: Whether/when acclimation letters are needed and related temperature rules.
- breed_restrictions: Whether brachycephalic or aggressive breeds are restricted.

Special policies (for all airlines):
- sedation_policy
- aircraft_type_restrictions
- flight_duration_limits
- comfort_stop_requirements
- booking_itinerary_restrictions

Return JSON in this structure:
{
  "jetblue": {
    "carrier_specs": {"text": ..., "urls": [...]},
    "fee": {"text": ..., "urls": [...]},
    "capacity_per_flight": {"text": ..., "urls": [...]},
    "capacity_per_traveler": {"text": ..., "urls": [...]},
    "pet_type_restrictions": {"text": ..., "urls": [...]},
    "cabin_class_restrictions": {"text": ..., "urls": [...]},
    "cargo_alternative": {"text": ..., "urls": [...]},
    "sedation_policy": {"text": ..., "urls": [...]},
    "aircraft_type_restrictions": {"text": ..., "urls": [...]},
    "flight_duration_limits": {"text": ..., "urls": [...]},
    "comfort_stop_requirements": {"text": ..., "urls": [...]},
    "booking_itinerary_restrictions": {"text": ..., "urls": [...]}
  },
  "american": {
    ... same as above ...,
    "temperature_safety_ranges": {"text": ..., "urls": [...]},
    "acclimation_letter_policy": {"text": ..., "urls": [...]},
    "breed_restrictions": {"text": ..., "urls": [...]}
  },
  "united": {
    ... same as jetblue fields ...
  }
}
If an airline is missing entirely in the answer, set that airline to null. If a particular field is missing, set it to null or set its text to null and urls to [].
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _official_domains_for_airline(airline: str) -> List[str]:
    airline_lower = airline.lower()
    if "jetblue" in airline_lower:
        return ["jetblue.com"]
    if "american" in airline_lower:
        return ["aa.com", "aacargo.com"]
    if "united" in airline_lower:
        return ["united.com"]
    return []


def _has_info_and_official_link(item: Optional[InfoItem], official_domains: List[str]) -> bool:
    if not item or not item.text or not item.text.strip():
        return False
    if not item.urls:
        return False
    # At least one official URL should be present
    for u in item.urls:
        lu = (u or "").lower()
        if any(dom in lu for dom in official_domains):
            return True
    return False


def _filter_official_urls(urls: List[str], official_domains: List[str]) -> List[str]:
    filtered = []
    for u in urls:
        lu = (u or "").lower()
        if any(dom in lu for dom in official_domains):
            filtered.append(u)
    return filtered


def _is_marked_not_specified(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    return any(kw in t for kw in ["not specified", "unspecified", "n/a", "none stated", "no stated", "no limit specified"])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _verify_info_item(
    evaluator: Evaluator,
    parent,
    *,
    node_id: str,
    desc: str,
    airline_name: str,
    item: Optional[InfoItem],
    additional_instruction: str,
) -> None:
    """
    Turn a single rubric leaf ("..._With_URL") into a strict sequential group with:
      - critical existence check (value present + at least one official URL)
      - critical URL-based verification of the claimed text
    """
    # Build group node (critical) to reflect rubric's single critical leaf logic but with gated checks
    group = evaluator.add_sequential(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=True
    )

    official_domains = _official_domains_for_airline(airline_name)
    has_official = _has_info_and_official_link(item, official_domains)

    evaluator.add_custom_node(
        result=has_official,
        id=f"{node_id}_provided",
        desc=f"{airline_name}: Value present and includes at least one official URL ({', '.join(official_domains)})",
        parent=group,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=f"{airline_name}: {desc} — supported by cited official URL(s)",
        parent=group,
        critical=True
    )

    text = (item.text or "").strip() if item else ""
    urls = item.urls if (item and item.urls) else []
    urls_official = _filter_official_urls(urls, official_domains)
    sources_for_verification = urls_official if urls_official else urls  # fallback to all if filter removed all

    claim = text if text else f"{airline_name} policy element as described in the answer."

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=sources_for_verification,
        additional_instruction=additional_instruction
    )


async def _verify_capacity(
    evaluator: Evaluator,
    parent,
    *,
    node_id: str,
    desc: str,
    airline_name: str,
    item: Optional[InfoItem],
    per_what: str  # "per flight" or "per traveler"
) -> None:
    """
    Capacity leaves often may be 'not specified'. Handle both specified vs. not specified claims.
    """
    group = evaluator.add_sequential(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=True
    )

    official_domains = _official_domains_for_airline(airline_name)
    has_official = _has_info_and_official_link(item, official_domains)

    evaluator.add_custom_node(
        result=has_official,
        id=f"{node_id}_provided",
        desc=f"{airline_name}: Capacity {per_what} value (or 'not specified') present with at least one official URL",
        parent=group,
        critical=True
    )

    verify_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported",
        desc=f"{airline_name}: {desc} — supported by cited official URL(s)",
        parent=group,
        critical=True
    )

    text = (item.text or "").strip() if item else ""
    urls = item.urls if (item and item.urls) else []
    urls_official = _filter_official_urls(urls, official_domains)
    sources_for_verification = urls_official if urls_official else urls

    if _is_marked_not_specified(text):
        claim = f"The official {airline_name} policy page(s) do not specify a maximum number of in-cabin pets {per_what}."
        add_ins = (
            "Determine whether the provided official page(s) lack any explicit statement of a numeric cap "
            f"for in-cabin pets {per_what}. If the page clearly lists a cap, then this 'not specified' claim is incorrect."
        )
    else:
        claim = f"The maximum number of in-cabin pets {per_what} for {airline_name} is: {text}."
        add_ins = (
            "Verify that the page(s) explicitly state this per-{scope} cap or equivalent wording. Allow minor formatting differences."
            .replace("{scope}", per_what.replace("per ", ""))
        )

    await evaluator.verify(
        claim=claim,
        node=verify_leaf,
        sources=sources_for_verification,
        additional_instruction=add_ins
    )


async def _verify_airline_policies(
    evaluator: Evaluator,
    parent,
    *,
    airline_name: str,
    airline_node_id: str,
    data: Optional[AirlinePolicy],
    include_american_only: bool = False
) -> None:
    """
    Build the airline subtree and attach all policy checks.
    """
    airline_node = evaluator.add_parallel(
        id=airline_node_id,
        desc=f"{airline_name} in-cabin pet policy elements documented with official URLs.",
        parent=parent,
        critical=True
    )

    # Safety: If airline data is entirely missing, create failing existence node to reflect critical nature
    if data is None:
        evaluator.add_custom_node(
            result=False,
            id=f"{airline_node_id}_exists",
            desc=f"{airline_name}: No policy information extracted from the answer.",
            parent=airline_node,
            critical=True
        )
        return

    # Common verification instructions (tuned for web-source verification)
    instr_dims = (
        "Verify the page explicitly states the maximum in-cabin pet carrier dimensions (L/W/H). "
        "Allow minor formatting variations (e.g., inches vs cm) and rounding. Accept equivalent phrasing."
    )
    instr_fee = (
        "Verify the page explicitly states the in-cabin pet fee (per pet, per direction/one-way). "
        "Allow minor formatting variations and currency symbols."
    )
    instr_pet_types = (
        "Verify which species (e.g., cats/dogs) are accepted for in-cabin travel and any size/weight limits "
        "as described in the claim. Accept equivalent language and formatting."
    )
    instr_cabin_class = (
        "Verify which cabin classes allow or prohibit pets for in-cabin travel. "
        "Allow equivalent phrasing (e.g., 'no pets in premium cabins with lie-flat seats')."
    )
    instr_cargo = (
        "Verify whether cargo/checked transport for pets is available as an alternative. "
        "If the airline does not offer cargo for pets, the page should indicate unavailability."
    )
    instr_generic_special = (
        "Verify the stated special-policy detail is explicitly indicated or clearly supported on the official page. "
        "Allow equivalent phrasing."
    )

    # JetBlue_/American_/United_ nodes mapping (keep IDs as in rubric)
    # 1) Carrier specs
    await _verify_info_item(
        evaluator, airline_node,
        node_id=f"{airline_name.split()[0]}_Carrier_Specs_With_URL" if " " in airline_name else f"{airline_name}_Carrier_Specs_With_URL",
        desc="Provides maximum in-cabin carrier dimensions (L/W/H) and an official URL verifying them.",
        airline_name=airline_name,
        item=data.carrier_specs,
        additional_instruction=instr_dims
    )

    # 2) Fee
    await _verify_info_item(
        evaluator, airline_node,
        node_id=f"{airline_name.split()[0]}_Fee_With_URL" if " " in airline_name else f"{airline_name}_Fee_With_URL",
        desc="Provides cost per pet per direction and an official URL verifying it.",
        airline_name=airline_name,
        item=data.fee,
        additional_instruction=instr_fee
    )

    # 3) Capacity per flight
    await _verify_capacity(
        evaluator, airline_node,
        node_id=f"{airline_name.split()[0]}_Capacity_Per_Flight_With_URL" if " " in airline_name else f"{airline_name}_Capacity_Per_Flight_With_URL",
        desc="Provides maximum pets allowed per flight and an official URL verifying it (or explicitly states not specified, with an official URL).",
        airline_name=airline_name,
        item=data.capacity_per_flight,
        per_what="per flight"
    )

    # 4) Capacity per traveler
    await _verify_capacity(
        evaluator, airline_node,
        node_id=f"{airline_name.split()[0]}_Capacity_Per_Traveler_With_URL" if " " in airline_name else f"{airline_name}_Capacity_Per_Traveler_With_URL",
        desc="Provides maximum pets allowed per traveler and an official URL verifying it (or explicitly states not specified, with an official URL).",
        airline_name=airline_name,
        item=data.capacity_per_traveler,
        per_what="per traveler"
    )

    # 5) Pet type restrictions
    await _verify_info_item(
        evaluator, airline_node,
        node_id=f"{airline_name.split()[0]}_Pet_Type_Restrictions_With_URL" if " " in airline_name else f"{airline_name}_Pet_Type_Restrictions_With_URL",
        desc="Provides accepted in-cabin species and any size/weight limitations with an official URL verifying them.",
        airline_name=airline_name,
        item=data.pet_type_restrictions,
        additional_instruction=instr_pet_types
    )

    # 6) Cabin class restrictions
    await _verify_info_item(
        evaluator, airline_node,
        node_id=f"{airline_name.split()[0]}_Cabin_Class_Restrictions_With_URL" if " " in airline_name else f"{airline_name}_Cabin_Class_Restrictions_With_URL",
        desc="Provides which cabin classes allow pets (and which do not) with an official URL verifying them.",
        airline_name=airline_name,
        item=data.cabin_class_restrictions,
        additional_instruction=instr_cabin_class
    )

    # 7) Cargo alternative
    await _verify_info_item(
        evaluator, airline_node,
        node_id=f"{airline_name.split()[0]}_Cargo_Alternative_With_URL" if " " in airline_name else f"{airline_name}_Cargo_Alternative_With_URL",
        desc="States whether cargo/checked transport is available as an alternative and provides an official URL verifying it.",
        airline_name=airline_name,
        item=data.cargo_alternative,
        additional_instruction=instr_cargo
    )

    # American-only items
    if include_american_only:
        await _verify_info_item(
            evaluator, airline_node,
            node_id="American_Temperature_Safety_Ranges_With_URL",
            desc="Provides AA temperature safety policy (safe operating range and/or prohibited thresholds) with an official URL verifying it.",
            airline_name=airline_name,
            item=data.temperature_safety_ranges,
            additional_instruction="Verify temperature thresholds/ranges exactly or equivalently stated. These are often cargo-related; AA Cargo pages are acceptable."
        )
        await _verify_info_item(
            evaluator, airline_node,
            node_id="American_Acclimation_Letter_Policy_With_URL",
            desc="Provides AA acclimation letter requirements (including any temperature conditions) with an official URL verifying it.",
            airline_name=airline_name,
            item=data.acclimation_letter_policy,
            additional_instruction="Verify the acclimation letter policy as described is explicitly present. AA Cargo pages are acceptable."
        )
        await _verify_info_item(
            evaluator, airline_node,
            node_id="American_Breed_Restrictions_With_URL",
            desc="Documents whether brachycephalic and/or aggressive breeds are restricted with an official URL.",
            airline_name=airline_name,
            item=data.breed_restrictions,
            additional_instruction="Verify explicit breed restrictions (e.g., snub-nosed/brachycephalic). AA Cargo pages acceptable for cargo restrictions."
        )

    # Special policies group
    special_group = evaluator.add_parallel(
        id=f"{airline_name.split()[0]}_Special_Policies_With_URL" if " " in airline_name else f"{airline_name}_Special_Policies_With_URL",
        desc="Documents additional restrictions in the specified special-policy categories, each with an official URL.",
        parent=airline_node,
        critical=True
    )

    # Sedation
    await _verify_info_item(
        evaluator, special_group,
        node_id=f"{airline_name.split()[0]}_Sedation_Policy_With_URL" if " " in airline_name else f"{airline_name}_Sedation_Policy_With_URL",
        desc="States the airline policy on pet sedation for travel and provides an official URL verifying it (or states not specified, with official URL).",
        airline_name=airline_name,
        item=data.sedation_policy,
        additional_instruction=instr_generic_special
    )
    # Aircraft type restrictions
    await _verify_info_item(
        evaluator, special_group,
        node_id=f"{airline_name.split()[0]}_Aircraft_Type_Restrictions_With_URL" if " " in airline_name else f"{airline_name}_Aircraft_Type_Restrictions_With_URL",
        desc="States any aircraft-type restrictions affecting pet transport and provides an official URL verifying it (or not specified, with official URL).",
        airline_name=airline_name,
        item=data.aircraft_type_restrictions,
        additional_instruction=instr_generic_special
    )
    # Flight duration limits
    await _verify_info_item(
        evaluator, special_group,
        node_id=f"{airline_name.split()[0]}_Flight_Duration_Limits_With_URL" if " " in airline_name else f"{airline_name}_Flight_Duration_Limits_With_URL",
        desc="States any flight-duration/itinerary-length limits for pets and provides an official URL verifying it (or not specified, with official URL).",
        airline_name=airline_name,
        item=data.flight_duration_limits,
        additional_instruction=instr_generic_special
    )
    # Comfort stop requirements
    await _verify_info_item(
        evaluator, special_group,
        node_id=f"{airline_name.split()[0]}_Comfort_Stop_Requirements_With_URL" if " " in airline_name else f"{airline_name}_Comfort_Stop_Requirements_With_URL",
        desc="States any comfort-stop requirements (if any) and provides an official URL verifying it (or not specified/none, with official URL).",
        airline_name=airline_name,
        item=data.comfort_stop_requirements,
        additional_instruction=instr_generic_special
    )
    # Booking / itinerary restrictions
    await _verify_info_item(
        evaluator, special_group,
        node_id=f"{airline_name.split()[0]}_Booking_Itinerary_Restrictions_With_URL" if " " in airline_name else f"{airline_name}_Booking_Itinerary_Restrictions_With_URL",
        desc="States any booking/itinerary restrictions for pets (e.g., codeshare/interline) with an official URL (or not specified/none, with official URL).",
        airline_name=airline_name,
        item=data.booking_itinerary_restrictions,
        additional_instruction=instr_generic_special
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
    Evaluate the airline in-cabin pet policy comparison answer with URL-grounded verification.
    """
    # Initialize
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

    # Root rubric group (critical)
    top = evaluator.add_parallel(
        id="Complete_Airline_Pet_Policy_Comparison",
        desc="Compare in-cabin pet travel policies for JetBlue, American Airlines, and United Airlines; for each required policy element, include specific values/requirements and a direct official URL where it can be verified.",
        parent=root,
        critical=True
    )

    # Extract structured info
    extracted: PoliciesExtraction = await evaluator.extract(
        prompt=prompt_extract_policies(),
        template_class=PoliciesExtraction,
        extraction_name="policies_extraction"
    )

    # Build subtrees for each airline
    await _verify_airline_policies(
        evaluator, top,
        airline_name="JetBlue",
        airline_node_id="JetBlue_Policy_Verification",
        data=extracted.jetblue,
        include_american_only=False
    )

    await _verify_airline_policies(
        evaluator, top,
        airline_name="American Airlines",
        airline_node_id="American_Airlines_Policy_Verification",
        data=extracted.american,
        include_american_only=True
    )

    await _verify_airline_policies(
        evaluator, top,
        airline_name="United",
        airline_node_id="United_Airlines_Policy_Verification",
        data=extracted.united,
        include_american_only=False
    )

    return evaluator.get_summary()