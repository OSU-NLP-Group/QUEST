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
TASK_ID = "us_semiconductor_locations_2024"
TASK_DESCRIPTION = """
Identify four distinct locations in the United States semiconductor industry, each meeting specific criteria as of December 2024:

1. First Facility: A semiconductor manufacturing facility in Arizona that produces chips using an advanced process node technology (smaller than 20nm). Provide the city, the company operating the facility, and the specific process node technology.

2. Second Facility: A semiconductor manufacturing facility in Arizona with a specified monthly production capacity for 12-inch wafers. Provide the city, the company operating the facility, and the monthly wafer production capacity or target capacity.

3. Third Facility: A semiconductor manufacturing facility in Texas with a publicly announced major investment (over $10 billion). Provide the city, the company operating the facility, and the total investment amount in US dollars.

4. Fourth Location: The corporate headquarters of a major semiconductor design company located in California. Provide the complete street address, the company name, and the city and state.

For each location, include a reference URL that supports your answer.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class Facility1(BaseModel):
    """Arizona fab producing chips on an advanced node (<20nm)."""
    city: Optional[str] = None
    state: Optional[str] = None
    company: Optional[str] = None
    process_node: Optional[str] = None           # e.g., "5nm", "N4", "7nm FinFET"
    process_node_nm_value: Optional[str] = None  # e.g., "5 nm", "18nm"; keep as string
    status_text: Optional[str] = None            # e.g., "operational in 2024", "announced in 2020"
    sources: List[str] = Field(default_factory=list)


class Facility2(BaseModel):
    """Arizona fab with specified monthly capacity for 12-inch (300mm) wafers."""
    city: Optional[str] = None
    state: Optional[str] = None
    company: Optional[str] = None
    wafer_size: Optional[str] = None             # e.g., "12-inch", "300mm"
    capacity_monthly: Optional[str] = None       # e.g., "20,000 wafers per month", "35k WSPM"
    status_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Facility3(BaseModel):
    """Texas fab with major investment > $10B."""
    city: Optional[str] = None
    state: Optional[str] = None
    company: Optional[str] = None
    investment_amount: Optional[str] = None      # e.g., "$25 billion", "USD 17B"
    status_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Location4(BaseModel):
    """Corporate HQ of a semiconductor design company in California."""
    company: Optional[str] = None
    address: Optional[str] = None                # complete street address
    city: Optional[str] = None
    state: Optional[str] = None
    status_text: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SemiconductorLocationsExtraction(BaseModel):
    """Full extraction container."""
    facility1: Optional[Facility1] = None
    facility2: Optional[Facility2] = None
    facility3: Optional[Facility3] = None
    location4: Optional[Location4] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_locations() -> str:
    return """
    Extract four distinct locations from the answer, organized into the following objects. Only extract information explicitly present in the answer text. If a field is missing, return null. For URLs, extract actual URLs (including full protocol) mentioned in the answer.

    Return a JSON object with fields: facility1, facility2, facility3, location4.

    1) facility1 (Arizona fab producing chips on <20nm process):
       - city: City name of the facility (e.g., Phoenix)
       - state: State name or abbreviation (e.g., Arizona or AZ)
       - company: Company operating the facility (e.g., TSMC, Intel)
       - process_node: The specific process node used (e.g., "5nm", "N4", "7nm")
       - process_node_nm_value: If available, extract the explicit nanometer value string (e.g., "5 nm", "7nm", "14 nm"). If not present, return null.
       - status_text: Any statement in the answer indicating announcement/operational status (e.g., "operational in 2024", "announced in 2020"). If missing, return null.
       - sources: Array of at least one reference URL supporting the claims for facility1

    2) facility2 (Arizona fab with monthly 12-inch wafer capacity):
       - city
       - state
       - company
       - wafer_size: The wafer size related to capacity (should be "12-inch" or "300mm" if provided)
       - capacity_monthly: Capacity figure or target for wafers per month (e.g., "20,000 wafers per month", "35k WSPM"). Include units/timeframe in the string.
       - status_text
       - sources: Array of at least one reference URL supporting the capacity/location/operator claims

    3) facility3 (Texas fab with investment > $10B):
       - city
       - state
       - company
       - investment_amount: The publicly announced total investment amount in USD string (e.g., "$25 billion", "USD 17B")
       - status_text
       - sources: Array of at least one reference URL supporting the location/operator/investment/announcement claims

    4) location4 (Corporate HQ of a semiconductor design company in California):
       - company: The semiconductor design company name
       - address: Complete street address for the headquarters (include street number/name; ZIP if provided)
       - city
       - state
       - status_text
       - sources: Array of at least one reference URL supporting the HQ address/company/location claims

    Special notes:
    - If the answer lists multiple options for any of the above, choose the first one that meets the criteria.
    - Extract URLs exactly as shown (markdown or plain text). Ignore invalid/malformed URLs.
    - Do not invent information; return null for any missing fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(x: Optional[str]) -> str:
    return x.strip() if isinstance(x, str) else ""

def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0

def _state_matches(state: Optional[str], targets: List[str]) -> bool:
    s = _safe_str(state).lower()
    return any(s == t.lower() for t in targets)

def _az_variants() -> List[str]:
    return ["Arizona", "AZ"]

def _tx_variants() -> List[str]:
    return ["Texas", "TX"]

def _ca_variants() -> List[str]:
    return ["California", "CA"]


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_facility_1(evaluator: Evaluator, parent_node, f1: Optional[Facility1]) -> None:
    node = evaluator.add_parallel(
        id="facility_1",
        desc="1) Arizona semiconductor manufacturing facility producing chips on a process node <20nm (as of Dec 2024).",
        parent=parent_node,
        critical=False
    )

    city = _safe_str(f1.city if f1 else None)
    state = _safe_str(f1.state if f1 else None)
    company = _safe_str(f1.company if f1 else None)
    proc = _safe_str(f1.process_node if f1 else None)
    proc_nm = _safe_str(f1.process_node_nm_value if f1 else None)
    sources = f1.sources if f1 else []

    # Reference URL existence (critical)
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="facility_1_reference_url",
        desc="Include at least one reference URL supporting the location, operator, process node, and status claims.",
        parent=node,
        critical=True
    )

    # Location city/state existence + verification
    evaluator.add_custom_node(
        result=(city != "" and _state_matches(state, _az_variants())),
        id="facility_1_location_city_state_exists",
        desc="Facility 1 location fields provided (city present and state is Arizona/AZ).",
        parent=node,
        critical=True
    )

    loc_leaf = evaluator.add_leaf(
        id="facility_1_location_city_state",
        desc="Provide the facility city and confirm the state is Arizona.",
        parent=node,
        critical=True
    )
    loc_claim = f"The semiconductor manufacturing facility is located in {city}, Arizona."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=sources,
        additional_instruction="Confirm the facility location is in Arizona (AZ) and the city matches the claim. Allow minor formatting variations."
    )

    # Company operator existence + verification
    evaluator.add_custom_node(
        result=(company != ""),
        id="facility_1_company_operator_exists",
        desc="Facility 1 company/operator field provided.",
        parent=node,
        critical=True
    )

    op_leaf = evaluator.add_leaf(
        id="facility_1_company_operator",
        desc="Identify the company operating the facility.",
        parent=node,
        critical=True
    )
    op_claim = f"The semiconductor manufacturing facility in {city}, Arizona is operated by {company}."
    await evaluator.verify(
        claim=op_claim,
        node=op_leaf,
        sources=sources,
        additional_instruction="Verify the operator/owner of the Arizona facility; synonyms like 'operator', 'owner', 'runs' are acceptable."
    )

    # Process node existence + support verification
    evaluator.add_custom_node(
        result=(proc != "" or proc_nm != ""),
        id="facility_1_process_node_exists",
        desc="Facility 1 process node field provided.",
        parent=node,
        critical=True
    )

    pn_leaf = evaluator.add_leaf(
        id="facility_1_process_node_support",
        desc="State a specific process node technology used for manufactured chips.",
        parent=node,
        critical=True
    )
    pn_text = proc if proc else proc_nm
    pn_claim = f"This facility produces chips using the {pn_text} process node."
    await evaluator.verify(
        claim=pn_claim,
        node=pn_leaf,
        sources=sources,
        additional_instruction="Confirm the page explicitly mentions the advanced node (e.g., 5nm/N5, 7nm, etc.) for manufacturing at the Arizona facility."
    )

    # Process node < 20nm simple check
    lt_leaf = evaluator.add_leaf(
        id="facility_1_process_node_lt_20nm",
        desc="Confirm the stated process node is smaller than 20nm.",
        parent=node,
        critical=True
    )
    lt_claim = f"The described process node '{pn_text}' is smaller than 20 nanometers."
    await evaluator.verify(
        claim=lt_claim,
        node=lt_leaf,
        additional_instruction="Reason whether the named node (e.g., 5nm, 7nm, N5, N4, etc.) is < 20nm. Accept standard industry naming."
    )

    # Status by Dec 2024 verification
    status_leaf = evaluator.add_leaf(
        id="facility_1_status_by_dec_2024",
        desc="Provide evidence the facility was announced or operational as of December 2024.",
        parent=node,
        critical=True
    )
    status_claim = "By December 2024, this Arizona facility had been announced or was operational."
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=sources,
        additional_instruction="Check publication dates, press releases, or status indicators on the cited page(s) to confirm announcement/operation by Dec 2024."
    )


async def verify_facility_2(evaluator: Evaluator, parent_node, f2: Optional[Facility2]) -> None:
    node = evaluator.add_parallel(
        id="facility_2",
        desc="2) Arizona semiconductor manufacturing facility with a specified monthly capacity for 12-inch (300mm) wafers (as of Dec 2024).",
        parent=parent_node,
        critical=False
    )

    city = _safe_str(f2.city if f2 else None)
    state = _safe_str(f2.state if f2 else None)
    company = _safe_str(f2.company if f2 else None)
    wafer_size = _safe_str(f2.wafer_size if f2 else None)
    capacity = _safe_str(f2.capacity_monthly if f2 else None)
    sources = f2.sources if f2 else []

    # Reference URL existence (critical)
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="facility_2_reference_url",
        desc="Include at least one reference URL supporting the location, operator, capacity figure, and status claims.",
        parent=node,
        critical=True
    )

    # Location existence + verify
    evaluator.add_custom_node(
        result=(city != "" and _state_matches(state, _az_variants())),
        id="facility_2_location_city_state_exists",
        desc="Facility 2 location fields provided (city present and state is Arizona/AZ).",
        parent=node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="facility_2_location_city_state",
        desc="Provide the facility city and confirm the state is Arizona.",
        parent=node,
        critical=True
    )
    loc_claim = f"The semiconductor manufacturing facility is located in {city}, Arizona."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=sources,
        additional_instruction="Confirm the facility location is in Arizona (AZ) and the city matches the claim."
    )

    # Company existence + verify
    evaluator.add_custom_node(
        result=(company != ""),
        id="facility_2_company_operator_exists",
        desc="Facility 2 company/operator field provided.",
        parent=node,
        critical=True
    )
    op_leaf = evaluator.add_leaf(
        id="facility_2_company_operator",
        desc="Identify the company operating the facility.",
        parent=node,
        critical=True
    )
    op_claim = f"The semiconductor manufacturing facility in {city}, Arizona is operated by {company}."
    await evaluator.verify(
        claim=op_claim,
        node=op_leaf,
        sources=sources,
        additional_instruction="Verify the operator/owner of the Arizona facility; synonyms like 'operator', 'owner', 'runs' are acceptable."
    )

    # Capacity existence + support verification (must be monthly; 12-inch/300mm context)
    evaluator.add_custom_node(
        result=(capacity != ""),
        id="facility_2_capacity_exists",
        desc="Facility 2 monthly capacity field provided.",
        parent=node,
        critical=True
    )
    cap_leaf = evaluator.add_leaf(
        id="facility_2_monthly_capacity_12inch",
        desc="Provide a specified monthly production capacity for 12-inch (300mm) wafers, including units/timeframe.",
        parent=node,
        critical=True
    )
    wafer_phrase = wafer_size if wafer_size else "12-inch (300mm)"
    cap_claim = f"This facility has a monthly production capacity (or target) of {capacity} for {wafer_phrase} wafers."
    await evaluator.verify(
        claim=cap_claim,
        node=cap_leaf,
        sources=sources,
        additional_instruction="Confirm the capacity figure relates to 12-inch (300mm) wafers and uses monthly units (e.g., wafers per month, WSPM)."
    )

    # Explicit 12-inch/300mm confirmation
    wafer_leaf = evaluator.add_leaf(
        id="facility_2_wafer_size_is_12inch",
        desc="Confirm the capacity refers to 12-inch (300mm) wafers.",
        parent=node,
        critical=True
    )
    wafer_claim = f"The stated capacity is specifically for 12-inch (300mm) wafers."
    await evaluator.verify(
        claim=wafer_claim,
        node=wafer_leaf,
        sources=sources,
        additional_instruction="Verify the page explicitly mentions 12-inch or 300mm context for the capacity."
    )

    # Status by Dec 2024 verification
    status_leaf = evaluator.add_leaf(
        id="facility_2_status_by_dec_2024",
        desc="Provide evidence the facility was announced or operational as of December 2024.",
        parent=node,
        critical=True
    )
    status_claim = "By December 2024, this Arizona facility had been announced or was operational."
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=sources,
        additional_instruction="Check publication dates, press releases, or status indicators on the cited page(s) to confirm announcement/operation by Dec 2024."
    )


async def verify_facility_3(evaluator: Evaluator, parent_node, f3: Optional[Facility3]) -> None:
    node = evaluator.add_parallel(
        id="facility_3",
        desc="3) Texas semiconductor manufacturing facility with a publicly announced major investment > $10B (as of Dec 2024).",
        parent=parent_node,
        critical=False
    )

    city = _safe_str(f3.city if f3 else None)
    state = _safe_str(f3.state if f3 else None)
    company = _safe_str(f3.company if f3 else None)
    invest = _safe_str(f3.investment_amount if f3 else None)
    sources = f3.sources if f3 else []

    # Reference URL existence (critical)
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="facility_3_reference_url",
        desc="Include at least one reference URL supporting the location, operator, investment amount, and status/announcement timing.",
        parent=node,
        critical=True
    )

    # Location existence + verify (Texas)
    evaluator.add_custom_node(
        result=(city != "" and _state_matches(state, _tx_variants())),
        id="facility_3_location_city_state_exists",
        desc="Facility 3 location fields provided (city present and state is Texas/TX).",
        parent=node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="facility_3_location_city_state",
        desc="Provide the facility city and confirm the state is Texas.",
        parent=node,
        critical=True
    )
    loc_claim = f"The semiconductor manufacturing facility is located in {city}, Texas."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=sources,
        additional_instruction="Confirm the facility location is in Texas (TX) and the city matches the claim."
    )

    # Company existence + verify
    evaluator.add_custom_node(
        result=(company != ""),
        id="facility_3_company_operator_exists",
        desc="Facility 3 company/operator field provided.",
        parent=node,
        critical=True
    )
    op_leaf = evaluator.add_leaf(
        id="facility_3_company_operator",
        desc="Identify the company operating the facility.",
        parent=node,
        critical=True
    )
    op_claim = f"The semiconductor manufacturing facility in {city}, Texas is operated by {company}."
    await evaluator.verify(
        claim=op_claim,
        node=op_leaf,
        sources=sources,
        additional_instruction="Verify operator/owner of the Texas facility; synonyms like 'operator', 'owner', 'runs' are acceptable."
    )

    # Investment existence + support verification
    evaluator.add_custom_node(
        result=(invest != ""),
        id="facility_3_investment_amount_exists",
        desc="Facility 3 investment amount field provided.",
        parent=node,
        critical=True
    )
    inv_leaf = evaluator.add_leaf(
        id="facility_3_investment_amount_support",
        desc="Provide the publicly announced total investment amount in USD.",
        parent=node,
        critical=True
    )
    inv_claim = f"The publicly announced total investment for this Texas facility is {invest}."
    await evaluator.verify(
        claim=inv_claim,
        node=inv_leaf,
        sources=sources,
        additional_instruction="Confirm the page mentions the USD investment magnitude (e.g., $25B, USD 17 billion)."
    )

    # > $10B simple check
    gt_leaf = evaluator.add_leaf(
        id="facility_3_investment_amount_gt_10b",
        desc="Confirm the investment exceeds $10 billion.",
        parent=node,
        critical=True
    )
    gt_claim = f"The investment amount described as '{invest}' exceeds $10 billion."
    await evaluator.verify(
        claim=gt_claim,
        node=gt_leaf,
        additional_instruction="Reason whether the described amount is greater than $10B (e.g., '$25B', 'USD 17 billion' should pass)."
    )

    # Status by Dec 2024 verification (announcement/operational)
    status_leaf = evaluator.add_leaf(
        id="facility_3_status_by_dec_2024",
        desc="Provide evidence the investment/facility was publicly announced by (or the facility was announced/operational as of) December 2024.",
        parent=node,
        critical=True
    )
    status_claim = "By December 2024, this Texas facility and its investment had been publicly announced (or the facility was announced/operational)."
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=sources,
        additional_instruction="Check announcement/press dates, or status indicators showing existence/announcement by Dec 2024."
    )


async def verify_location_4(evaluator: Evaluator, parent_node, l4: Optional[Location4]) -> None:
    node = evaluator.add_parallel(
        id="location_4",
        desc="4) Corporate headquarters of a semiconductor design company in California, with complete street address (as of Dec 2024).",
        parent=parent_node,
        critical=False
    )

    company = _safe_str(l4.company if l4 else None)
    address = _safe_str(l4.address if l4 else None)
    city = _safe_str(l4.city if l4 else None)
    state = _safe_str(l4.state if l4 else None)
    sources = l4.sources if l4 else []

    # Reference URL existence (critical)
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="location_4_reference_url",
        desc="Include at least one reference URL supporting the company HQ address and location details.",
        parent=node,
        critical=True
    )

    # HQ city/state existence + verify (California)
    evaluator.add_custom_node(
        result=(city != "" and _state_matches(state, _ca_variants())),
        id="location_4_hq_in_california_exists",
        desc="HQ location fields provided (city present and state is California/CA).",
        parent=node,
        critical=True
    )
    loc_leaf = evaluator.add_leaf(
        id="location_4_hq_in_california_city_state",
        desc="Provide the HQ city and confirm the state is California.",
        parent=node,
        critical=True
    )
    loc_claim = f"The corporate headquarters is located in {city}, California."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_leaf,
        sources=sources,
        additional_instruction="Confirm the HQ city is in California (CA)."
    )

    # Company name existence + verify
    evaluator.add_custom_node(
        result=(company != ""),
        id="location_4_company_name_exists",
        desc="HQ company name field provided.",
        parent=node,
        critical=True
    )
    comp_leaf = evaluator.add_leaf(
        id="location_4_company_name",
        desc="Provide the semiconductor design company name headquartered at the address.",
        parent=node,
        critical=True
    )
    comp_claim = f"The corporate headquarters at the cited address belongs to {company}."
    await evaluator.verify(
        claim=comp_claim,
        node=comp_leaf,
        sources=sources,
        additional_instruction="Confirm the company name associated with the HQ address."
    )

    # Complete street address existence + verify
    evaluator.add_custom_node(
        result=(address != ""),
        id="location_4_street_address_exists",
        desc="HQ complete street address field provided.",
        parent=node,
        critical=True
    )
    addr_leaf = evaluator.add_leaf(
        id="location_4_complete_street_address",
        desc="Provide the complete street address for the headquarters.",
        parent=node,
        critical=True
    )
    addr_claim = f"The complete street address of the corporate headquarters is '{address}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_leaf,
        sources=sources,
        additional_instruction="Verify the full street address as shown on the cited page (street number/name; ZIP if provided)."
    )

    # Status by Dec 2024 verification (HQ)
    status_leaf = evaluator.add_leaf(
        id="location_4_status_by_dec_2024",
        desc="Provide evidence the address corresponds to the company headquarters as of December 2024.",
        parent=node,
        critical=True
    )
    status_claim = "By December 2024, the cited address was the corporate headquarters of the stated semiconductor design company."
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=sources,
        additional_instruction="Confirm the HQ designation and that it is valid as of Dec 2024; official site pages or credible directory listings are acceptable."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the US semiconductor locations (Dec 2024) task.
    """
    # Initialize evaluator with root parallel aggregation
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

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_locations(),
        template_class=SemiconductorLocationsExtraction,
        extraction_name="semiconductor_locations_extraction",
    )

    # Build verification subtrees for each required item
    await verify_facility_1(evaluator, root, extraction.facility1)
    await verify_facility_2(evaluator, root, extraction.facility2)
    await verify_facility_3(evaluator, root, extraction.facility3)
    await verify_location_4(evaluator, root, extraction.location4)

    # Return the evaluation summary
    return evaluator.get_summary()