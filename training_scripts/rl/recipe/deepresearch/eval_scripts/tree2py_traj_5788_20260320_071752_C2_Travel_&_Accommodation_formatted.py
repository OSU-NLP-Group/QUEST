import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "travel_planning_march_2026"
TASK_DESCRIPTION = """
A US citizen based in Nashville is planning a March 2026 trip to visit Universal Epic Universe in Orlando, with a possible short extension to Oman afterward. They need the following travel information:

1. What is the name of the budget airline that operates nonstop flights from Nashville International Airport (BNA) to Orlando Sanford International Airport (SFB)?

2. What is the complete street address of Universal Epic Universe, and is the theme park currently open as of March 2026?

3. For US passport holders planning to stay in Oman for less than 14 days: Is a visa required to be obtained in advance? If visa-free entry is available for short stays, what specific requirements must be met? What is the minimum passport validity period required for entry to Oman?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DirectFlightAirline(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EpicUniverseInfo(BaseModel):
    full_address: Optional[str] = None
    street_number: Optional[str] = None
    street_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    address_sources: List[str] = Field(default_factory=list)

    status_as_of_march_2026: Optional[str] = None  # use values like "open", "not_open", "closed", "opening <date>", etc.
    status_sources: List[str] = Field(default_factory=list)


class OmanEntryInfo(BaseModel):
    visa_in_advance_required: Optional[str] = None  # expected "yes" or "no" (for stays under 14 days)
    visa_free_max_duration_days: Optional[str] = None  # expected "14" or "14 days"
    visa_free_requirements: List[str] = Field(default_factory=list)  # should include hotel booking, health insurance, return/onward ticket
    oman_visa_sources: List[str] = Field(default_factory=list)

    passport_min_validity_months: Optional[str] = None  # expected "6 months" (or numeric "6")
    blank_pages_required: Optional[str] = None  # expected "at least one blank page", "1 blank page", etc.
    oman_passport_sources: List[str] = Field(default_factory=list)


class TravelInfoExtraction(BaseModel):
    airline: Optional[DirectFlightAirline] = None
    epic_universe: Optional[EpicUniverseInfo] = None
    oman: Optional[OmanEntryInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_info() -> str:
    return """
    Extract the requested travel-planning information exactly as stated in the answer. DO NOT invent any details.
    Return a single JSON object conforming to the following schema:

    {
      "airline": {
        "name": string | null,
        "sources": string[]    // URLs explicitly present in the answer that support the BNA↔SFB nonstop route on the named airline
      },
      "epic_universe": {
        "full_address": string | null,     // single-line complete street address as written in the answer
        "street_number": string | null,
        "street_name": string | null,
        "city": string | null,
        "state": string | null,
        "zip_code": string | null,
        "address_sources": string[],       // URLs cited in the answer that support the address
        "status_as_of_march_2026": string | null,   // use concise value from the answer like: "open", "not open", "closed", "opening 2026", etc.
        "status_sources": string[]         // URLs cited in the answer that support the status timing
      },
      "oman": {
        "visa_in_advance_required": string | null,     // for US passport holders staying <14 days; answer value like "yes" or "no"
        "visa_free_max_duration_days": string | null,  // e.g., "14" or "14 days"
        "visa_free_requirements": string[],            // list items exactly as phrased in the answer; include: confirmed hotel booking, health insurance, return/onward ticket if present
        "oman_visa_sources": string[],                 // URLs cited in the answer for the visa/visa-free policy
        "passport_min_validity_months": string | null, // e.g., "6 months" or "6"
        "blank_pages_required": string | null,         // e.g., "at least one blank page", "one blank page", "1 blank page"
        "oman_passport_sources": string[]              // URLs cited in the answer for passport validity/blank page rules
      }
    }

    Rules:
    - Extract only what is explicitly present in the answer.
    - For any missing field, return null (or [] for arrays).
    - For URL fields, extract only valid URLs actually present in the answer (plain URLs or markdown links).
    - Do not deduplicate or modify the extracted text; preserve it from the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _norm_lower(s: Optional[str]) -> str:
    return str(s or "").strip().lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    tl = text.lower()
    return any(k in tl for k in keywords)


def _requirements_covered(reqs: List[str]) -> Dict[str, bool]:
    """
    Check if the three specific visa-free requirements are all mentioned
    using fuzzy keyword matching:
      - confirmed hotel booking
      - health insurance
      - return or onward ticket
    """
    text_items = [r.lower() for r in reqs]
    def any_match(pred):
        return any(pred(t) for t in text_items)

    has_hotel = any_match(lambda t: ("hotel" in t and ("booking" in t or "reservation" in t or "confirmed" in t)))
    has_ins = any_match(lambda t: "insur" in t)  # health insurance, travel insurance
    has_return = any_match(lambda t: ("return" in t or "onward" in t) and ("ticket" in t or "flight" in t))

    return {"hotel": has_hotel, "insurance": has_ins, "return_ticket": has_return}


def _is_status_open(status_str: Optional[str]) -> Optional[bool]:
    """
    Normalize status to True(open)/False(not open)/None(unknown) based on extracted text.
    """
    s = _norm_lower(status_str)
    if not s:
        return None
    # Explicit negatives
    if any(neg in s for neg in ["not open", "closed", "temporarily closed"]):
        return False
    # Phrases indicating open/operational
    if "open" in s or "opened" in s or "in operation" in s:
        # Guard against "opening" (future)
        if "opening" in s and "opened" not in s:
            # ambiguous: "opening" suggests future
            return None
        return True
    # If it says "opening <date>" and that date is beyond March 2026, it's not open as of March 2026.
    if "opening" in s and ("2026" in s or "2027" in s):
        # Without exact month parsing, consider unknown rather than assertively false.
        return None
    return None


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_direct_flight_airline(
    evaluator: Evaluator,
    parent_node,
    airline: Optional[DirectFlightAirline],
) -> None:
    """
    Travel_Planning_Information -> Direct_Flight_Airline
    - Critical existence/sources check
    - Critical route support by cited sources
    """
    node = evaluator.add_parallel(
        id="Direct_Flight_Airline",
        desc="Identify the budget airline operating direct flights from Nashville (BNA) to Orlando Sanford (SFB)",
        parent=parent_node,
        critical=False
    )

    name_ok = _has_text(airline.name) if airline else False
    src_ok = bool(airline and airline.sources)
    evaluator.add_custom_node(
        result=name_ok and src_ok,
        id="direct_flight_airline_provided",
        desc="Airline name provided with at least one supporting source URL",
        parent=node,
        critical=True
    )

    route_leaf = evaluator.add_leaf(
        id="direct_flight_airline_route_supported",
        desc="The named airline operates nonstop flights from BNA to SFB (Orlando Sanford)",
        parent=node,
        critical=True
    )
    airline_name = airline.name if airline else ""
    claim = f"{airline_name} operates nonstop (direct) flights between Nashville International Airport (BNA) and Orlando Sanford International Airport (SFB)."
    await evaluator.verify(
        claim=claim,
        node=route_leaf,
        sources=(airline.sources if airline else []),
        additional_instruction=(
            "Accept evidence from the airline's official route map or schedule, airport route listings, "
            "or reputable third-party flight listings. The route must specifically reference Orlando Sanford (SFB), "
            "not Orlando International (MCO). Treat 'nonstop' and 'direct' as equivalent. If sources are unrelated or about MCO, mark as not supported."
        ),
    )


async def verify_epic_universe_details(
    evaluator: Evaluator,
    parent_node,
    epic: Optional[EpicUniverseInfo],
) -> None:
    """
    Travel_Planning_Information -> Epic_Universe_Details
      -> Epic_Universe_Address (critical)
         - completeness provided (critical)
         - supported by sources (critical)
      -> Epic_Universe_Status (critical)
         - status provided (critical)
         - status supported by sources (critical)
    """
    epic_node = evaluator.add_parallel(
        id="Epic_Universe_Details",
        desc="Provide accurate location and status information for Universal Epic Universe",
        parent=parent_node,
        critical=False
    )

    # Address group (critical)
    addr_group = evaluator.add_parallel(
        id="Epic_Universe_Address",
        desc="Provide the complete street address of Universal Epic Universe (must include street number, street name, city, state, and ZIP code)",
        parent=epic_node,
        critical=True
    )
    # Completeness + sources presence
    addr_complete = (
        bool(epic)
        and _has_text(epic.full_address)
        and _has_text(epic.street_number)
        and _has_text(epic.street_name)
        and _has_text(epic.city)
        and _has_text(epic.state)
        and _has_text(epic.zip_code)
        and bool(epic.address_sources)
    )
    evaluator.add_custom_node(
        result=addr_complete,
        id="epic_universe_address_complete",
        desc="Complete address fields present and at least one address source URL provided",
        parent=addr_group,
        critical=True
    )

    addr_supported_leaf = evaluator.add_leaf(
        id="epic_universe_address_supported",
        desc="The provided full street address for Universal Epic Universe is supported by the cited sources",
        parent=addr_group,
        critical=True
    )
    full_addr = epic.full_address if epic else ""
    await evaluator.verify(
        claim=f"The complete street address of Universal Epic Universe is '{full_addr}'.",
        node=addr_supported_leaf,
        sources=(epic.address_sources if epic else []),
        additional_instruction=(
            "Verify the specific full address (street number, street name, city, state, ZIP) on authoritative sources "
            "such as Universal's official site, resort/park page, or local government/venue pages. "
            "Allow minor formatting differences and common abbreviations (e.g., 'FL' vs 'Florida'). "
            "Ensure the address corresponds to Universal Epic Universe specifically, not other Universal parks."
        ),
    )

    # Status group (critical)
    status_group = evaluator.add_parallel(
        id="Epic_Universe_Status",
        desc="Confirm whether Universal Epic Universe is currently open as of March 2026",
        parent=epic_node,
        critical=True
    )
    status_ok = bool(epic and _has_text(epic.status_as_of_march_2026) and epic.status_sources)
    evaluator.add_custom_node(
        result=status_ok,
        id="epic_universe_status_provided",
        desc="Status as of March 2026 is stated and at least one status source URL is provided",
        parent=status_group,
        critical=True
    )

    status_supported_leaf = evaluator.add_leaf(
        id="epic_universe_status_supported",
        desc="The stated 'open as of March 2026' (or 'not open') status is supported by cited sources",
        parent=status_group,
        critical=True
    )
    norm_open = _is_status_open(epic.status_as_of_march_2026) if epic else None
    if norm_open is True:
        status_claim = "As of March 2026, Universal Epic Universe is open to the public."
    elif norm_open is False:
        status_claim = "As of March 2026, Universal Epic Universe is not open to the public."
    else:
        # Fall back to using the raw extracted phrasing to avoid over-normalizing
        raw_status = epic.status_as_of_march_2026 or ""
        status_claim = f"As of March 2026, Universal Epic Universe status is: {raw_status}."

    await evaluator.verify(
        claim=status_claim,
        node=status_supported_leaf,
        sources=(epic.status_sources if epic else []),
        additional_instruction=(
            "Confirm whether the park is open by March 2026. If a source states a confirmed opening date in 2025 (or earlier), "
            "it supports that the park is open by March 2026. If a source states opening is after March 2026, it supports 'not open'. "
            "Prefer official Universal pages or major reputable publications."
        ),
    )


async def verify_oman_entry_requirements(
    evaluator: Evaluator,
    parent_node,
    oman: Optional[OmanEntryInfo],
) -> None:
    """
    Travel_Planning_Information -> Oman_Entry_Requirements
      -> Visa_Free_Entry_Details (critical)
         - details provided (critical)
         - supported by sources (critical)
      -> Passport_Validity_Requirement (critical)
         - details provided (critical)
         - supported by sources (critical)
    """
    oman_node = evaluator.add_parallel(
        id="Oman_Entry_Requirements",
        desc="Provide complete entry requirements for US citizens traveling to Oman",
        parent=parent_node,
        critical=False
    )

    # Visa-free group (critical)
    visa_group = evaluator.add_parallel(
        id="Visa_Free_Entry_Details",
        desc="State whether a visa is required to be obtained in advance for stays under 14 days, the maximum duration of visa-free stay (14 days), and all three required conditions for visa-free entry: confirmed hotel booking, health insurance, and return ticket",
        parent=oman_node,
        critical=True
    )

    # Provided/completeness check
    has_visa_flag = bool(oman and _has_text(oman.visa_in_advance_required))
    has_days = bool(oman and _has_text(oman.visa_free_max_duration_days) and "14" in _norm_lower(oman.visa_free_max_duration_days))
    req_coverage = _requirements_covered(oman.visa_free_requirements if oman else [])
    has_all_three = req_coverage["hotel"] and req_coverage["insurance"] and req_coverage["return_ticket"]
    visa_src_ok = bool(oman and oman.oman_visa_sources)

    evaluator.add_custom_node(
        result=(has_visa_flag and has_days and has_all_three and visa_src_ok),
        id="oman_visa_details_provided",
        desc="Visa-free details present (visa-in-advance flag, '14 days' max, and the three requirements) with at least one source URL",
        parent=visa_group,
        critical=True
    )

    # Supported-by-sources check
    visa_details_leaf = evaluator.add_leaf(
        id="oman_visa_details_supported",
        desc="Short-stay visa-free policy and its three conditions for US citizens are supported by cited sources",
        parent=visa_group,
        critical=True
    )
    vflag = (oman.visa_in_advance_required or "").strip().lower() if oman else ""
    visa_needs_in_advance = "yes" if "yes" in vflag else ("no" if "no" in vflag else vflag)
    days_str = (oman.visa_free_max_duration_days or "").strip() if oman else ""
    reqs_list = oman.visa_free_requirements if oman else []
    claim_visa = (
        f"For U.S. passport holders staying in Oman for less than 14 days: "
        f"visa required in advance = '{visa_needs_in_advance}'. "
        f"The maximum duration of visa-free stay is '{days_str}'. "
        f"The following conditions are required for visa-free entry: {reqs_list}."
    )
    await evaluator.verify(
        claim=claim_visa,
        node=visa_details_leaf,
        sources=(oman.oman_visa_sources if oman else []),
        additional_instruction=(
            "Confirm that the policy specifically applies to U.S. citizens for short stays under 14 days. "
            "Verify the maximum visa-free duration (14 days) AND the presence of all three requirements: "
            "a confirmed hotel booking, valid health/travel insurance, and a return or onward ticket. "
            "Prefer official Omani government (e.g., Royal Oman Police/eVisa) or U.S. government/embassy resources."
        ),
    )

    # Passport validity group (critical)
    pass_group = evaluator.add_parallel(
        id="Passport_Validity_Requirement",
        desc="State the minimum passport validity required for entry to Oman (6 months from date of entry) and the requirement for at least one blank passport page for entry stamps",
        parent=oman_node,
        critical=True
    )

    # Provided/completeness check
    validity_ok = bool(oman and _has_text(oman.passport_min_validity_months) and (
        "6" in _norm_lower(oman.passport_min_validity_months) or "six" in _norm_lower(oman.passport_min_validity_months)
    ))
    blank_ok = bool(oman and _has_text(oman.blank_pages_required) and (
        "blank" in _norm_lower(oman.blank_pages_required) and ("one" in _norm_lower(oman.blank_pages_required) or "1" in _norm_lower(oman.blank_pages_required))
    ))
    pass_src_ok = bool(oman and oman.oman_passport_sources)

    evaluator.add_custom_node(
        result=(validity_ok and blank_ok and pass_src_ok),
        id="oman_passport_validity_provided",
        desc="Passport validity (6 months) and at least one blank page are stated with at least one source URL",
        parent=pass_group,
        critical=True
    )

    # Supported-by-sources check
    pass_supported_leaf = evaluator.add_leaf(
        id="oman_passport_validity_supported",
        desc="Passport validity (6 months) and blank-page requirement supported by cited sources",
        parent=pass_group,
        critical=True
    )
    validity_val = oman.passport_min_validity_months if oman else ""
    blank_val = oman.blank_pages_required if oman else ""
    claim_pass = (
        f"Oman requires that a U.S. passport be valid for at least '{validity_val}' from the date of entry, "
        f"and that there is '{blank_val}' available for entry/exit stamps."
    )
    await evaluator.verify(
        claim=claim_pass,
        node=pass_supported_leaf,
        sources=(oman.oman_passport_sources if oman else []),
        additional_instruction=(
            "Confirm both requirements: 1) a minimum passport validity of at least 6 months from the date of entry, "
            "and 2) at least one blank passport page available for entry/exit stamps. Prefer official Omani or U.S. government sources."
        ),
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the multi-destination travel-planning task.
    """
    # 1) Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Three major subtasks are independent
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

    # 2) Extract structured data from the answer
    extracted: TravelInfoExtraction = await evaluator.extract(
        prompt=prompt_extract_travel_info(),
        template_class=TravelInfoExtraction,
        extraction_name="travel_info_extraction",
    )

    # 3) Build the top-level rubric node (parallel aggregator)
    # Note: We set this as non-critical to allow partial credit per subtask.
    tpi_node = evaluator.add_parallel(
        id="Travel_Planning_Information",
        desc="Complete set of travel information for the multi-destination trip planning",
        parent=root,
        critical=False
    )

    # 4) Verification subtrees
    await verify_direct_flight_airline(evaluator, tpi_node, extracted.airline)
    await verify_epic_universe_details(evaluator, tpi_node, extracted.epic_universe)
    await verify_oman_entry_requirements(evaluator, tpi_node, extracted.oman)

    # 5) Optional: record a compact snapshot of extracted info
    evaluator.add_custom_info(
        info={
            "airline": (extracted.airline.dict() if extracted.airline else None),
            "epic_universe": (extracted.epic_universe.dict() if extracted.epic_universe else None),
            "oman": (extracted.oman.dict() if extracted.oman else None),
        },
        info_type="extraction_snapshot",
        info_name="extraction_snapshot"
    )

    # 6) Return structured evaluation summary
    return evaluator.get_summary()