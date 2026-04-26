import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "disney_destiny_maiden_voyage_planning"
TASK_DESCRIPTION = (
    "I'm planning to book the Disney Destiny maiden voyage cruise departing in November 2025. "
    "I need comprehensive information to prepare for this trip. Please provide details on: ship/departure, "
    "passport requirements, documentation requirements, embarkation logistics, ports of call (including Castaway Cay "
    "location), and optional port procedures and shore excursion policy, each with supporting URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ShipDepartureExtraction(BaseModel):
    maiden_voyage_date: Optional[FieldWithSources] = None
    departure_city: Optional[FieldWithSources] = None
    departure_port: Optional[FieldWithSources] = None
    cruise_terminal: Optional[FieldWithSources] = None


class PassportRequirementsExtraction(BaseModel):
    passport_validity_rule: Optional[FieldWithSources] = None
    earliest_passport_expiry_date: Optional[FieldWithSources] = None


class DocumentationRequirementsExtraction(BaseModel):
    original_documents_only_rule: Optional[FieldWithSources] = None
    government_photo_id_rule: Optional[FieldWithSources] = None
    name_matching_rule: Optional[FieldWithSources] = None


class EmbarkationLogisticsExtraction(BaseModel):
    latest_arrival_time_rule: Optional[FieldWithSources] = None
    parking_rate_regular: Optional[FieldWithSources] = None
    parking_rate_oversized: Optional[FieldWithSources] = None


class PortsOfCallExtraction(BaseModel):
    ports_of_call_list: List[str] = Field(default_factory=list)
    ports_of_call_sources: List[str] = Field(default_factory=list)
    castaway_cay_location: Optional[FieldWithSources] = None


class PortProceduresOptionalExtraction(BaseModel):
    typical_all_aboard_time: Optional[FieldWithSources] = None
    recommended_return_buffer: Optional[FieldWithSources] = None


class ShoreExcursionPolicyOptionalExtraction(BaseModel):
    cancellation_window_rule: Optional[FieldWithSources] = None
    exceptions_rule: Optional[FieldWithSources] = None


class DisneyDestinyExtraction(BaseModel):
    ship_departure: Optional[ShipDepartureExtraction] = None
    passport: Optional[PassportRequirementsExtraction] = None
    documentation: Optional[DocumentationRequirementsExtraction] = None
    embarkation: Optional[EmbarkationLogisticsExtraction] = None
    ports: Optional[PortsOfCallExtraction] = None
    port_procedures: Optional[PortProceduresOptionalExtraction] = None
    shore_excursion_policy: Optional[ShoreExcursionPolicyOptionalExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the requested planning information for the Disney Destiny maiden voyage from the provided answer text. For each requested item, extract the 'value' exactly as stated and all supporting reference URLs. Follow these rules carefully:

General rules:
- Extract only what appears in the answer text; do not invent information.
- For each item, extract a list of direct URLs provided in the answer as citations. If an item has no URLs, set sources to an empty list.
- Return null for missing values.

Extract the following structured fields:

1) ship_departure:
   - maiden_voyage_date: value (e.g., "November 20, 2025"); sources (list of URLs)
   - departure_city: value (e.g., "Fort Lauderdale, Florida"); sources (list of URLs)
   - departure_port: value (e.g., "Port Everglades"); sources (list of URLs)
   - cruise_terminal: value (e.g., "Cruise Terminal 4"); sources (list of URLs)

2) passport:
   - passport_validity_rule: value (e.g., "Passport must be valid for at least 6 months after the cruise ends"); sources (list of URLs)
   - earliest_passport_expiry_date: value (e.g., "May 24, 2026"); sources (list of URLs, if any; if they reused the validity rule URL, include it)

3) documentation:
   - original_documents_only_rule: value; sources (list of URLs)
   - government_photo_id_rule: value; sources (list of URLs)
   - name_matching_rule: value; sources (list of URLs)

4) embarkation:
   - latest_arrival_time_rule: value (e.g., "Arrive no later than 60 minutes prior to sail time"); sources (list of URLs)
   - parking_rate_regular: value (e.g., "$20 per day"); sources (list of URLs)
   - parking_rate_oversized: value (e.g., "$25 per day"); sources (list of URLs)

5) ports:
   - ports_of_call_list: array of port names (e.g., ["Lookout Cay", "Castaway Cay"]); sources (list of URLs supporting itinerary)
   - castaway_cay_location: value (e.g., "26.0833°N, 77.5334°W" OR "approximately 160 miles east of Miami"); sources (list of URLs)

6) port_procedures (optional):
   - typical_all_aboard_time: value (e.g., "30 minutes before scheduled departure"); sources (list of URLs)
   - recommended_return_buffer: value (e.g., "1–2 hours before sail time"); sources (list of URLs)

7) shore_excursion_policy (optional):
   - cancellation_window_rule: value (e.g., "Up to 48 hours prior to port arrival without penalty"); sources (list of URLs)
   - exceptions_rule: value (e.g., "Different policies for tours involving flights, trains, special events, or overnight stays"); sources (list of URLs)

Return a single JSON object matching the DisneyDestinyExtraction schema.
"""


# --------------------------------------------------------------------------- #
# Helper date utilities                                                       #
# --------------------------------------------------------------------------- #
def _try_parse_date(date_str: str) -> Optional[datetime]:
    """Try to parse a date string in several common formats."""
    if not date_str:
        return None
    date_str = date_str.strip()
    fmts = [
        "%B %d, %Y",   # November 20, 2025
        "%b %d, %Y",   # Nov 20, 2025
        "%Y-%m-%d",    # 2025-11-20
        "%m/%d/%Y",    # 11/20/2025
        "%d %B %Y",    # 20 November 2025
        "%d %b %Y",    # 20 Nov 2025
    ]
    for f in fmts:
        try:
            return datetime.strptime(date_str, f)
        except Exception:
            continue
    return None


def _days_in_month(year: int, month: int) -> int:
    import calendar
    return calendar.monthrange(year, month)[1]


def _add_months(dt: datetime, months: int) -> datetime:
    """Add months to a datetime, clipping the day to the month's length."""
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    d = min(dt.day, _days_in_month(y, m))
    return dt.replace(year=y, month=m, day=d)


def compute_earliest_expiry(start_date_str: Optional[str], nights: int = 4) -> Optional[str]:
    """
    Compute earliest acceptable passport expiry date given start date and nights:
    earliest_expiry = (start_date + nights days) + 6 months
    Returns formatted as 'Month DD, YYYY' or None if cannot compute.
    """
    if not start_date_str:
        return None
    start_dt = _try_parse_date(start_date_str)
    if not start_dt:
        return None
    end_dt = start_dt + timedelta(days=nights)
    expiry_dt = _add_months(end_dt, 6)
    return expiry_dt.strftime("%B %d, %Y")


def safe_sources(srcs: Optional[List[str]]) -> List[str]:
    """Ensure sources is a list of strings."""
    if not srcs:
        return []
    return [s for s in srcs if isinstance(s, str) and s.strip()]


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def add_claim_with_source_verification(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    sources: List[str],
    critical: bool = True,
    additional_instruction: str = "None",
) -> None:
    """
    Add a critical existence check for citations and then verify the claim against the URLs.
    """
    has_citation = evaluator.add_custom_node(
        result=(len(sources) > 0),
        id=f"{node_id}_has_citation",
        desc=f"{desc} — has at least one reference URL provided in the answer",
        parent=parent_node,
        critical=critical
    )

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Section verifications                                                       #
# --------------------------------------------------------------------------- #
async def verify_ship_and_departure_information(evaluator: Evaluator, root_node, data: DisneyDestinyExtraction):
    """
    Ship and Departure Information (critical, parallel):
    - Maiden voyage date: Nov 20, 2025, with citation
    - Departure city: Fort Lauderdale, Florida, with citation
    - Departure port: Port Everglades, with citation
    - Cruise terminal: Cruise Terminal #4, with citation
    """
    node = evaluator.add_parallel(
        id="ship_and_departure_information",
        desc="Provide maiden voyage and embarkation location details that match the constraints, each with a supporting reference URL.",
        parent=root_node,
        critical=True
    )
    sd = data.ship_departure or ShipDepartureExtraction()

    # Maiden voyage date
    mv_sources = safe_sources(sd.maiden_voyage_date.sources if sd.maiden_voyage_date else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "maiden_voyage_date_correct_with_citation",
        "States the maiden voyage departure date as November 20, 2025, and includes a supporting reference URL.",
        "The Disney Destiny maiden voyage (inaugural sailing) departs on November 20, 2025.",
        mv_sources,
        critical=True,
        additional_instruction="Treat 'maiden voyage' and 'inaugural sailing' as equivalent."
    )

    # Departure city
    city_sources = safe_sources(sd.departure_city.sources if sd.departure_city else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "departure_city_correct_with_citation",
        "Identifies the departure city as Fort Lauderdale, Florida, and includes a supporting reference URL.",
        "The Disney Destiny departs from Fort Lauderdale, Florida.",
        city_sources,
        critical=True,
        additional_instruction="If the page references Port Everglades, recognize it as the port located in Fort Lauderdale."
    )

    # Departure port
    port_sources = safe_sources(sd.departure_port.sources if sd.departure_port else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "departure_port_correct_with_citation",
        "Identifies the departure port as Port Everglades, and includes a supporting reference URL.",
        "The departure port for the Disney Destiny maiden voyage is Port Everglades.",
        port_sources,
        critical=True,
        additional_instruction="Confirm the port name explicitly as 'Port Everglades'."
    )

    # Cruise terminal
    terminal_sources = safe_sources(sd.cruise_terminal.sources if sd.cruise_terminal else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "cruise_terminal_correct_with_citation",
        "Specifies the embarkation terminal as Cruise Terminal #4, and includes a supporting reference URL.",
        "The embarkation terminal for the Disney Destiny at Port Everglades is Cruise Terminal 4 (Terminal #4).",
        terminal_sources,
        critical=True,
        additional_instruction="Allow variants like 'Cruise Terminal 4', 'Terminal 4', or 'CT4' to be considered equivalent."
    )


async def verify_passport_requirements(evaluator: Evaluator, root_node, data: DisneyDestinyExtraction):
    """
    Passport requirements (critical, sequential):
    - Validity rule: 6 months after cruise ends, with citation
    - Earliest passport expiry date: May 24, 2026 (4-night cruise ending Nov 24, 2025; +6 months), logically correct
    """
    node = evaluator.add_sequential(
        id="passport_requirements",
        desc="Provide the passport validity rule (per constraints) and compute the earliest acceptable passport expiration date for a 4-night sailing departing Nov 20, 2025, with appropriate citation for the rule.",
        parent=root_node,
        critical=True
    )
    ps = data.passport or PassportRequirementsExtraction()

    # Validity rule
    validity_sources = safe_sources(ps.passport_validity_rule.sources if ps.passport_validity_rule else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "passport_validity_rule_correct_with_citation",
        "States the passport validity requirement as: passport valid for at least 6 months after the cruise ends, and includes a supporting reference URL.",
        "Disney Cruise Line requires that passports remain valid for at least 6 months after the cruise ends.",
        validity_sources,
        critical=True,
        additional_instruction="Focus on Disney Cruise Line official documentation or policy pages. Allow equivalent phrasing."
    )

    # Earliest passport expiry date (simple logical verification)
    # Compute expected date from extracted maiden date if available; default to expected in rubric
    sd = data.ship_departure or ShipDepartureExtraction()
    departure_val = sd.maiden_voyage_date.value if sd.maiden_voyage_date else "November 20, 2025"
    computed_earliest = compute_earliest_expiry(departure_val, nights=4) or "May 24, 2026"
    extracted_earliest = ps.earliest_passport_expiry_date.value if ps.earliest_passport_expiry_date else None

    # Existence check for earliest date value (ensure the answer actually stated it)
    evaluator.add_custom_node(
        result=(extracted_earliest is not None and str(extracted_earliest).strip() != ""),
        id="earliest_passport_expiry_date_value_present",
        desc="Earliest acceptable passport expiration date is stated in the answer",
        parent=node,
        critical=True
    )

    # Verify correctness via simple logic (no URL needed because it's derived from validity rule and dates)
    earliest_leaf = evaluator.add_leaf(
        id="earliest_passport_expiry_date_correct",
        desc="Correctly computes and states the earliest acceptable passport expiration date as May 24, 2026 (cruise ends Nov 24, 2025 for a 4-night cruise; +6 months), and references the validity rule cited previously or includes a URL supporting the rule.",
        parent=node,
        critical=True
    )
    claim = (
        f"Given a 4-night cruise departing on {departure_val} and a passport validity requirement of at least "
        f"6 months after the cruise ends, the earliest acceptable passport expiration date is {computed_earliest}."
    )
    await evaluator.verify(
        claim=claim,
        node=earliest_leaf,
        sources=None,  # logical check
        additional_instruction=(
            "Allow date format variations (e.g., '2026-05-24' vs 'May 24, 2026'). "
            "Compute end date as departure + 4 nights and then add 6 months."
        )
    )


async def verify_documentation_requirements(evaluator: Evaluator, root_node, data: DisneyDestinyExtraction):
    """
    Documentation requirements (critical, parallel):
    - Original documents only, with citation
    - Government-issued photo ID required, with citation
    - Name matching across documents required, with citation
    """
    node = evaluator.add_parallel(
        id="documentation_requirements",
        desc="Provide boarding documentation requirements that match the constraints, each with a supporting reference URL.",
        parent=root_node,
        critical=True
    )
    dc = data.documentation or DocumentationRequirementsExtraction()

    # Original documents only
    orig_sources = safe_sources(dc.original_documents_only_rule.sources if dc.original_documents_only_rule else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "original_documents_only_correct_with_citation",
        "States that all travel documents must be original and photocopies are not accepted, and includes a supporting reference URL.",
        "All travel documents must be original; photocopies are not accepted for boarding.",
        orig_sources,
        critical=True,
        additional_instruction="Focus on Disney Cruise Line official documentation and boarding requirements."
    )

    # Government-issued photo ID required
    id_sources = safe_sources(dc.government_photo_id_rule.sources if dc.government_photo_id_rule else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "government_photo_id_required_correct_with_citation",
        "States that government-issued photo identification is required for boarding, and includes a supporting reference URL.",
        "Government-issued photo identification is required for boarding.",
        id_sources,
        critical=True,
        additional_instruction="Accept equivalent phrasing (e.g., 'government ID with photo') and verify on authoritative sources."
    )

    # Name matching required
    name_sources = safe_sources(dc.name_matching_rule.sources if dc.name_matching_rule else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "name_matching_required_correct_with_citation",
        "States that names must match across all travel documents, and includes a supporting reference URL.",
        "The guest's name must match across all travel documents (e.g., passport and reservation).",
        name_sources,
        critical=True,
        additional_instruction="Allow reasonable variants; the policy should require consistent legal names across documents."
    )


async def verify_embarkation_logistics(evaluator: Evaluator, root_node, data: DisneyDestinyExtraction):
    """
    Embarkation logistics (critical, parallel):
    - Latest arrival time: no later than 60 minutes prior to sail time, with citation
    - Parking regular rate: $20/day, with citation
    - Parking oversized rate: $25/day, with citation
    """
    node = evaluator.add_parallel(
        id="embarkation_logistics",
        desc="Provide terminal arrival timing guidance and official parking rates that match the constraints, each with a supporting reference URL.",
        parent=root_node,
        critical=True
    )
    eb = data.embarkation or EmbarkationLogisticsExtraction()

    # Latest arrival time
    arr_sources = safe_sources(eb.latest_arrival_time_rule.sources if eb.latest_arrival_time_rule else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "latest_arrival_time_correct_with_citation",
        "States guests should arrive no later than 60 minutes prior to the published sail time, and includes a supporting reference URL.",
        "Guests should arrive no later than 60 minutes prior to the published sail time.",
        arr_sources,
        critical=True,
        additional_instruction="Verify timing guidance from official Disney Cruise Line communications or terminal instructions."
    )

    # Parking rate regular
    pr_sources = safe_sources(eb.parking_rate_regular.sources if eb.parking_rate_regular else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "parking_rate_regular_correct_with_citation",
        "States official Port Everglades parking is $20 per day for regular-sized vehicles, and includes a supporting reference URL.",
        "Official Port Everglades parking is $20 per day for regular-sized vehicles.",
        pr_sources,
        critical=True,
        additional_instruction="Confirm rates via official Port Everglades sources or authoritative port materials."
    )

    # Parking rate oversized
    po_sources = safe_sources(eb.parking_rate_oversized.sources if eb.parking_rate_oversized else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "parking_rate_oversized_correct_with_citation",
        "States official Port Everglades parking is $25 per day for oversized vehicles, and includes a supporting reference URL.",
        "Official Port Everglades parking is $25 per day for oversized vehicles.",
        po_sources,
        critical=True,
        additional_instruction="Confirm rates via official Port Everglades sources or authoritative port materials."
    )


async def verify_ports_of_call(evaluator: Evaluator, root_node, data: DisneyDestinyExtraction):
    """
    Ports of call (critical, parallel):
    - Ports of call include Lookout Cay and Castaway Cay, with citation
    - Castaway Cay location matches accepted forms, with citation
    """
    node = evaluator.add_parallel(
        id="ports_of_call",
        desc="Provide the ports of call and Castaway Cay location matching the constraints, each with a supporting reference URL.",
        parent=root_node,
        critical=True
    )
    pt = data.ports or PortsOfCallExtraction()

    # Ports of call list and supporting URLs
    ports_sources = safe_sources(pt.ports_of_call_sources)
    # Existence check for ports-of-call sources
    evaluator.add_custom_node(
        result=(len(ports_sources) > 0),
        id="ports_of_call_sources_present",
        desc="Ports of call item has at least one supporting reference URL",
        parent=node,
        critical=True
    )

    # Verify the itinerary includes Lookout Cay and Castaway Cay
    ports_leaf = evaluator.add_leaf(
        id="ports_of_call_correct_with_citation",
        desc="Lists the ports of call as including both Lookout Cay and Castaway Cay (at minimum), and includes a supporting reference URL.",
        parent=node,
        critical=True
    )
    claim_ports = (
        "The Disney Destiny maiden voyage itinerary includes both Lookout Cay (at Lighthouse Point) and Castaway Cay."
    )
    await evaluator.verify(
        claim=claim_ports,
        node=ports_leaf,
        sources=ports_sources,
        additional_instruction="Accept 'Lookout Cay' as 'Lookout Cay at Lighthouse Point'. Verify both ports are part of the itinerary."
    )

    # Castaway Cay location with citation
    ccl = pt.castaway_cay_location or FieldWithSources()
    castaway_sources = safe_sources(ccl.sources)
    await add_claim_with_source_verification(
        evaluator,
        node,
        "castaway_cay_location_correct_with_citation",
        "Provides Castaway Cay location consistent with the constraints—either approximately 26.0833°N, 77.5334°W OR approximately 160 miles east of Miami—and includes a supporting reference URL.",
        (
            "Castaway Cay is located approximately at 26.0833°N, 77.5334°W or described as roughly 160 miles east of Miami."
        ),
        castaway_sources,
        critical=True,
        additional_instruction=(
            "Support either coordinate-based location near 26.08°N, 77.53°W, or the descriptive location 'approximately 160 miles east of Miami'. "
            "Minor numeric rounding and phrasing variants are acceptable."
        )
    )


async def verify_port_procedures_optional(evaluator: Evaluator, root_node, data: DisneyDestinyExtraction):
    """
    Optional port procedures (non-critical, parallel):
    - Typical 'all aboard' time: 30 minutes before scheduled departure, with citation
    - Recommended return buffer: 1–2 hours before sail time, with citation
    """
    node = evaluator.add_parallel(
        id="port_procedures_optional",
        desc="Optional but helpful: port-of-call procedure timing guidance matching the constraints, each with a supporting reference URL.",
        parent=root_node,
        critical=False
    )
    pp = data.port_procedures or PortProceduresOptionalExtraction()

    # Typical 'all aboard' time
    ab_sources = safe_sources(pp.typical_all_aboard_time.sources if pp.typical_all_aboard_time else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "typical_all_aboard_time_correct_with_citation",
        "States that at ports of call the 'all aboard' time is typically 30 minutes before scheduled departure, and includes a supporting reference URL.",
        "At ports of call, the 'all aboard' time is typically 30 minutes before the scheduled departure.",
        ab_sources,
        critical=False,
        additional_instruction="Accept reasonable variations or phrasing indicating a 30-minute buffer."
    )

    # Recommended return buffer
    rb_sources = safe_sources(pp.recommended_return_buffer.sources if pp.recommended_return_buffer else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "recommended_return_buffer_correct_with_citation",
        "States the recommended return-to-ship buffer is 1–2 hours before sail time for safety, and includes a supporting reference URL.",
        "It is recommended to plan a 1–2 hour return-to-ship buffer before the sail time for safety.",
        rb_sources,
        critical=False,
        additional_instruction="Accept phrasing indicating a recommended buffer of about 1 to 2 hours."
    )


async def verify_shore_excursion_policy_optional(evaluator: Evaluator, root_node, data: DisneyDestinyExtraction):
    """
    Optional shore excursion policy (non-critical, parallel):
    - Cancellation window: up to 48 hours prior to port arrival without penalty, with citation
    - Exceptions: different policies for tours involving flights, trains, special events, or overnight stays, with citation
    """
    node = evaluator.add_parallel(
        id="shore_excursion_policy_optional",
        desc="Optional but helpful: shore excursion cancellation policy details matching the constraints, each with a supporting reference URL.",
        parent=root_node,
        critical=False
    )
    sp = data.shore_excursion_policy or ShoreExcursionPolicyOptionalExtraction()

    # Cancellation window
    cw_sources = safe_sources(sp.cancellation_window_rule.sources if sp.cancellation_window_rule else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "cancellation_window_correct_with_citation",
        "States shore excursions can be modified/cancelled up to 48 hours prior to port arrival without penalty, and includes a supporting reference URL.",
        "Shore excursions may be modified or cancelled up to 48 hours prior to port arrival without penalty.",
        cw_sources,
        critical=False,
        additional_instruction="Verify via Disney Cruise Line excursion policy pages; accept equivalent phrasing."
    )

    # Exceptions policy
    ex_sources = safe_sources(sp.exceptions_rule.sources if sp.exceptions_rule else [])
    await add_claim_with_source_verification(
        evaluator,
        node,
        "exceptions_correct_with_citation",
        "States that exceptions/different policies may apply for tours involving flights, trains, special events, or overnight stays, and includes a supporting reference URL.",
        "Exceptions or different policies may apply to tours involving flights, trains, special events, or overnight stays.",
        ex_sources,
        critical=False,
        additional_instruction="Confirm the listed exceptions or equivalent categories from the policy source."
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
    Evaluate an answer for the Disney Destiny maiden voyage planning task.
    """
    evaluator = Evaluator()
    # Root: to allow optional sections (non-critical children), set root non-critical to comply with framework constraints
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

    # Extract structured info
    extracted: DisneyDestinyExtraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=DisneyDestinyExtraction,
        extraction_name="disney_destiny_planning"
    )

    # Add custom info: computed dates, if possible
    sd = extracted.ship_departure or ShipDepartureExtraction()
    start_date_val = sd.maiden_voyage_date.value if sd.maiden_voyage_date else None
    computed_end_date = None
    computed_earliest = None
    if start_date_val:
        start_dt = _try_parse_date(start_date_val)
        if start_dt:
            end_dt = start_dt + timedelta(days=4)  # 4-night cruise ends after 4 days
            computed_end_date = end_dt.strftime("%B %d, %Y")
            computed_earliest = _add_months(end_dt, 6).strftime("%B %d, %Y")

    evaluator.add_custom_info(
        info={
            "start_date_extracted": start_date_val,
            "computed_cruise_end_date": computed_end_date or "Nov 24, 2025 (expected for 4-night from Nov 20, 2025)",
            "computed_earliest_passport_expiry": computed_earliest or "May 24, 2026 (expected)"
        },
        info_type="computed_dates",
        info_name="date_computation_details"
    )

    # Build verification tree by sections
    await verify_ship_and_departure_information(evaluator, root, extracted)
    await verify_passport_requirements(evaluator, root, extracted)
    await verify_documentation_requirements(evaluator, root, extracted)
    await verify_embarkation_logistics(evaluator, root, extracted)
    await verify_ports_of_call(evaluator, root, extracted)
    await verify_port_procedures_optional(evaluator, root, extracted)
    await verify_shore_excursion_policy_optional(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()