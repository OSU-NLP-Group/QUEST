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
TASK_ID = "nevi_ev_pa_i76"
TASK_DESCRIPTION = (
    "Identify one electric vehicle charging station located on Pennsylvania Interstate 76 (PA Turnpike) that is funded through the federal National Electric Vehicle Infrastructure (NEVI) Formula Program and meets all current federal compliance requirements. For the charging station you identify, provide the following information:\n\n"
    "1. The specific service plaza name or physical address where the station is located\n"
    "2. Confirmation that the station has at least 4 DC fast charging ports\n"
    "3. Confirmation that each port delivers a minimum of 150 kW continuous power output\n"
    "4. Confirmation that the station supports CCS (Combined Charging System) Type 1 connectors\n"
    "5. Confirmation that contactless payment methods (credit/debit card or tap-to-pay) are available\n"
    "6. Confirmation that the station is publicly accessible to all EV drivers\n"
    "7. Evidence or reference confirming the station's NEVI program funding status\n\n"
    "Provide official sources or reference URLs to support your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StationExtraction(BaseModel):
    """Structured extraction for a single NEVI-compliant station on PA I-76."""
    station_name_or_plaza: Optional[str] = None
    physical_address: Optional[str] = None
    corridor_designation: Optional[str] = None  # e.g., "I-76", "Pennsylvania Turnpike"
    dc_fast_port_count: Optional[str] = None    # Prefer strings like "4", "≥4", "4+"
    per_port_power_kw: Optional[str] = None     # e.g., "150 kW", "≥150 kW"
    connector_types: List[str] = Field(default_factory=list)  # e.g., ["CCS", "CCS1", "SAE CCS"]
    payment_methods: List[str] = Field(default_factory=list)  # e.g., ["credit card", "tap-to-pay"]
    public_accessibility: Optional[str] = None  # e.g., "public", "open to all drivers"
    nevi_funding_status: Optional[str] = None  # e.g., "NEVI-funded", "funded by NEVI"
    source_urls: List[str] = Field(default_factory=list)      # official references


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_station() -> str:
    return (
        "Extract exactly one electric vehicle charging station described in the answer that is on Pennsylvania Interstate 76 (PA Turnpike) and claimed as NEVI-funded and compliant. "
        "If the answer mentions multiple stations, select the first one that fits I-76 PA Turnpike and NEVI funding. "
        "Return the following fields:\n"
        "1. station_name_or_plaza: Service plaza name or station name as written in the answer\n"
        "2. physical_address: The street address if provided; otherwise null\n"
        "3. corridor_designation: The highway designation mentioned (e.g., 'I-76', 'PA Turnpike')\n"
        "4. dc_fast_port_count: The claimed count of DC fast charging ports (e.g., '4', '≥4')\n"
        "5. per_port_power_kw: The claimed per-port continuous power (e.g., '150 kW', '≥150 kW')\n"
        "6. connector_types: List of connector types claimed (e.g., 'CCS', 'CCS1', 'SAE CCS')\n"
        "7. payment_methods: List of payment methods claimed (e.g., 'credit card', 'tap-to-pay', 'contactless')\n"
        "8. public_accessibility: Whether the answer claims it is publicly accessible (e.g., 'public', 'open to all EV drivers'); otherwise null\n"
        "9. nevi_funding_status: The claimed NEVI funding status (e.g., 'NEVI-funded'); otherwise null\n"
        "10. source_urls: All reference URLs provided in the answer as evidence for this station. Include URLs in plain or markdown link formats. If none provided, return an empty list.\n"
        "Do not invent information; only extract what is explicitly in the answer."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def location_label(station: StationExtraction) -> str:
    """Choose the most specific location label from station name/plaza or address."""
    if station.station_name_or_plaza and station.station_name_or_plaza.strip():
        return station.station_name_or_plaza.strip()
    if station.physical_address and station.physical_address.strip():
        return station.physical_address.strip()
    return "the identified station location"


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_nevi_station(
    evaluator: Evaluator,
    parent_node,
    station: StationExtraction,
) -> None:
    """
    Build the verification tree for the NEVI-compliant station and run verifications
    according to the rubric's critical parallel checks.
    """
    # Create the critical parent node as specified by the rubric
    nevi_node = evaluator.add_parallel(
        id="NEVI_Compliant_Station_Identification",
        desc="Complete identification of a NEVI-compliant EV charging station on Pennsylvania I-76 that meets all federal program requirements with supporting documentation",
        parent=parent_node,
        critical=True
    )

    # Convenience
    sources = station.source_urls
    loc_str = location_label(station)

    # 1. Geographic location on PA I-76 (PA Turnpike)
    geo_node = evaluator.add_leaf(
        id="Geographic_Location_I76",
        desc="The charging station is located along Pennsylvania Interstate 76 (PA Turnpike)",
        parent=nevi_node,
        critical=True
    )
    geo_claim = (
        f"The charging station at {loc_str} is located along Pennsylvania Interstate 76 (PA Turnpike). "
        f"If the sources specify a service plaza on the PA Turnpike I-76, that satisfies the claim."
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_node,
        sources=sources,
        additional_instruction=(
            "Verify the station lies on the PA Turnpike segment of I-76 in Pennsylvania. "
            "Accept explicit mentions of 'I-76', 'PA Turnpike', or the service plaza located on I-76."
        ),
    )

    # 2. Minimum four DC fast charging ports
    ports_node = evaluator.add_leaf(
        id="Minimum_Four_DC_Ports",
        desc="The charging station has at least 4 DC fast charging ports",
        parent=nevi_node,
        critical=True
    )
    ports_claim = "The charging station has at least 4 DC fast charging ports (dispensers/stalls)."
    await evaluator.verify(
        claim=ports_claim,
        node=ports_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit counts of DC fast chargers such as '4 ports', '4 stalls', or '4 dispensers'. "
            "Interpret 'stalls' or 'dispensers' as ports. Do not count Level 2 ports."
        ),
    )

    # 3. Power output per port of at least 150 kW continuous
    power_node = evaluator.add_leaf(
        id="Power_Output_150kW",
        desc="Each charging port delivers a continuous power output of at least 150 kW",
        parent=nevi_node,
        critical=True
    )
    power_claim = "Each charging port provides at least 150 kW continuous power output (per port)."
    await evaluator.verify(
        claim=power_claim,
        node=power_node,
        sources=sources,
        additional_instruction=(
            "Confirm per-port power is ≥150 kW. Accept '150 kW' or 'up to 150 kW per port' if clearly meeting NEVI requirement; "
            "reject if only site total power is mentioned or if per-plug power is below 150 kW."
        ),
    )

    # 4. CCS Type 1 connector support
    ccs_node = evaluator.add_leaf(
        id="CCS_Connector_Support",
        desc="The charging station supports CCS (Combined Charging System) Type 1 connectors",
        parent=nevi_node,
        critical=True
    )
    ccs_claim = "The station supports CCS Type 1 (SAE Combined Charging System, CCS1) connectors."
    await evaluator.verify(
        claim=ccs_claim,
        node=ccs_node,
        sources=sources,
        additional_instruction=(
            "Look for 'CCS', 'CCS1', 'SAE CCS', or 'Combined Charging System' indications for DC fast charging. "
            "Do not rely on Level 2 J1772 as CCS."
        ),
    )

    # 5. Contactless payment availability
    payment_node = evaluator.add_leaf(
        id="Contactless_Payment_Method",
        desc="The charging station offers contactless payment options such as credit/debit card readers or tap-to-pay systems",
        parent=nevi_node,
        critical=True
    )
    payment_claim = "Contactless payment (credit/debit card reader or tap-to-pay) is available at the station."
    await evaluator.verify(
        claim=payment_claim,
        node=payment_node,
        sources=sources,
        additional_instruction=(
            "Confirm availability of card readers or contactless/tap-to-pay. "
            "Do not accept 'membership-only' or 'app-only' without contactless options."
        ),
    )

    # 6. Public accessibility
    public_node = evaluator.add_leaf(
        id="Public_Accessibility",
        desc="The charging station is publicly accessible to all electric vehicle drivers without brand or membership restrictions",
        parent=nevi_node,
        critical=True
    )
    public_claim = "The station is publicly accessible to all EV drivers, without brand exclusivity or membership restrictions."
    await evaluator.verify(
        claim=public_claim,
        node=public_node,
        sources=sources,
        additional_instruction=(
            "Confirm the station is open to the public. If sources mention brand-neutral or 'open to all EV drivers', that satisfies the claim."
        ),
    )

    # 7. NEVI program funding status
    nevi_funding_node = evaluator.add_leaf(
        id="NEVI_Program_Funding",
        desc="The charging station is officially funded through the National Electric Vehicle Infrastructure (NEVI) Formula Program",
        parent=nevi_node,
        critical=True
    )
    nevi_funding_claim = (
        "This charging station is funded through the federal National Electric Vehicle Infrastructure (NEVI) Formula Program."
    )
    await evaluator.verify(
        claim=nevi_funding_claim,
        node=nevi_funding_node,
        sources=sources,
        additional_instruction=(
            "Look for explicit 'NEVI' funding statements, listings on Pennsylvania NEVI award pages, "
            "PTC/PennDOT announcements, or FHWA NEVI documentation that names this location."
        ),
    )

    # 8. Physical location details (service plaza name or address)
    physical_loc_node = evaluator.add_leaf(
        id="Physical_Location_Details",
        desc="The specific service plaza name or physical address where the charging station is located is provided",
        parent=nevi_node,
        critical=True
    )
    physical_loc_claim = (
        f"The sources explicitly provide the station's specific location as '{loc_str}' (service plaza name or physical address)."
    )
    await evaluator.verify(
        claim=physical_loc_claim,
        node=physical_loc_node,
        sources=sources,
        additional_instruction=(
            "Verify that the sources present the same location string (service plaza or address) as provided in the answer, "
            "or a clear equivalent (minor formatting differences acceptable)."
        ),
    )

    # 9. Reference URLs provided (existence check)
    refs_provided_node = evaluator.add_custom_node(
        result=(bool(sources) and len(sources) > 0),
        id="Reference_URL_Provided",
        desc="Official sources or reference URLs are provided to support the answer",
        parent=nevi_node,
        critical=True
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
    Evaluate an answer for the NEVI-compliant station identification task on PA I-76.
    """
    # Initialize evaluator with a parallel root
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

    # Extract station information from the answer
    station_info = await evaluator.extract(
        prompt=prompt_extract_station(),
        template_class=StationExtraction,
        extraction_name="station_extraction"
    )

    # Optionally record NEVI requirements as GT context (not used for scoring, for reporting only)
    evaluator.add_ground_truth({
        "requirements": [
            "Located on PA I-76 (PA Turnpike)",
            "At least 4 DC fast charging ports",
            "≥150 kW continuous per port",
            "Supports CCS Type 1 connectors",
            "Contactless payment (credit/debit card or tap-to-pay)",
            "Publicly accessible to all EV drivers",
            "Explicit NEVI funding status",
            "Provide service plaza name or physical address",
            "Provide official reference URLs"
        ]
    }, gt_type="nevi_requirements")

    # Build verification tree and run checks
    await build_and_verify_nevi_station(evaluator, root, station_info)

    # Return structured summary
    return evaluator.get_summary()