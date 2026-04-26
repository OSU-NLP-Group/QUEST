import asyncio
import logging
import calendar
from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "grenada_passport_card_cruise_eligibility"
TASK_DESCRIPTION = (
    "A U.S. citizen is planning to take a closed-loop cruise departing from Miami, Florida, on September 1, 2026, "
    "and returning to Miami on September 8, 2026. The cruise itinerary includes a port call in Grenada, with arrival "
    "scheduled for September 4, 2026. The traveler holds a U.S. passport card that expires on April 30, 2027. "
    "Determine whether this U.S. passport card is sufficient to meet all entry requirements for this specific cruise to Grenada. "
    "Your answer must include: (1) A clear determination (yes or no) of whether the passport card meets all requirements, "
    "(2) Verification of each relevant requirement (document type, travel method, validity period, and destination-specific rules), "
    "and (3) Official source URLs that support each key requirement."
)

# Scenario constants (from the task description)
DEPARTURE_PORT = "Miami, Florida, USA"
RETURN_PORT = "Miami, Florida, USA"
DEPARTURE_DATE = date(2026, 9, 1)
RETURN_DATE = date(2026, 9, 8)
DESTINATION_COUNTRY = "Grenada"
GRENADA_ARRIVAL_DATE = date(2026, 9, 4)
PASSPORT_CARD_EXPIRY = date(2027, 4, 30)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RequirementsExtraction(BaseModel):
    """
    Extract the final determination and official source URLs provided in the answer for each requirement.
    """
    determination: Optional[str] = None  # "yes" or "no"
    reasoning: Optional[str] = None

    # Official URLs cited in the answer
    closed_loop_policy_urls: List[str] = Field(default_factory=list)
    passport_card_sea_caribbean_urls: List[str] = Field(default_factory=list)
    grenada_acceptance_urls: List[str] = Field(default_factory=list)
    grenada_validity_requirement_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    Extract the following fields from the answer:

    1) determination: A clear "yes" or "no" indicating whether the U.S. passport card is sufficient for this cruise to Grenada.
    2) reasoning: A concise summary explaining the determination.

    3) closed_loop_policy_urls: All official URLs provided that explain closed‑loop cruise documentation rules for U.S. citizens
       (e.g., CBP, DHS, cruise line policy pages). Only include actual URLs mentioned in the answer.

    4) passport_card_sea_caribbean_urls: All official URLs that explain the scope and permitted use of the U.S. passport card
       (land/sea only) and its acceptability for cruising to Caribbean ports. Only include URLs from official sources
       (e.g., travel.state.gov, cbp.gov) if present in the answer.

    5) grenada_acceptance_urls: Official Grenada entry requirement URLs or the U.S. State Department country page for Grenada,
       specifically cited in the answer to support whether Grenada accepts U.S. passport cards for cruise ship arrivals.

    6) grenada_validity_requirement_urls: Official URLs cited in the answer that state Grenada's passport validity requirement
       (e.g., "valid for 6 months beyond entry").

    Rules:
    - Return null for any missing scalar field and an empty list for any missing URL list.
    - Do not invent URLs. Extract only URLs explicitly present in the answer (plain URLs or within markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def add_months(base: date, months: int) -> date:
    """Add months to a date, clamping the day within the target month."""
    month = base.month - 1 + months
    year = base.year + month // 12
    month = month % 12 + 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(base.day, last_day))


def is_official_url(url: str) -> bool:
    """Simple heuristic to identify official or authoritative sources."""
    if not isinstance(url, str):
        return False
    url_lower = url.lower()
    official_markers = [
        "travel.state.gov",  # U.S. State Department
        "state.gov",
        "cbp.gov",           # U.S. Customs and Border Protection
        ".gov.gd",           # Grenada government domains
        "gov.gd",
        "usembassy.gov",     # U.S. embassies network
        "bb.usembassy.gov",  # U.S. Embassy Barbados (covers Grenada region)
        "dhs.gov",
    ]
    return any(m in url_lower for m in official_markers)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_document_and_method_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: RequirementsExtraction,
) -> None:
    """
    Build and verify the Document_and_Travel_Method_Requirements branch.
    """
    doc_node = evaluator.add_sequential(
        id="Document_and_Travel_Method_Requirements",
        desc="Verify that the passport card is an acceptable document for this specific travel scenario",
        parent=parent_node,
        critical=True,
    )

    # Closed loop cruise status (sequential chain root)
    closed_loop_node = evaluator.add_sequential(
        id="Closed_Loop_Cruise_Status",
        desc="Confirm the cruise begins and ends at the same U.S. port (Miami), qualifying as a closed-loop cruise",
        parent=doc_node,
        critical=True,
    )

    # Leaf: confirm closed-loop based on scenario
    closed_loop_leaf = evaluator.add_leaf(
        id="Closed_Loop_Confirmed",
        desc="Cruise departs from and returns to Miami, Florida, qualifying as a closed-loop cruise",
        parent=closed_loop_node,
        critical=True,
    )
    closed_loop_claim = (
        f"The cruise departs on {DEPARTURE_DATE.isoformat()} from {DEPARTURE_PORT} and returns on "
        f"{RETURN_DATE.isoformat()} to {RETURN_PORT}, so it begins and ends at the same U.S. port "
        f"and qualifies as a closed-loop cruise."
    )
    await evaluator.verify(
        claim=closed_loop_claim,
        node=closed_loop_leaf,
        additional_instruction="Use the provided scenario details to confirm closed-loop status; no external URL is required."
    )

    # Next: general passport card sea travel validity for Caribbean closed-loop cruises
    sea_valid_node = evaluator.add_sequential(
        id="Passport_Card_Sea_Travel_Validity",
        desc="Verify that U.S. passport cards are valid for sea travel to Caribbean destinations on closed-loop cruises",
        parent=closed_loop_node,
        critical=True,
    )

    sea_valid_leaf = evaluator.add_leaf(
        id="Passport_Card_Sea_Travel_Validity_Check",
        desc="U.S. passport card is acceptable for sea travel (not air), including closed-loop Caribbean cruises",
        parent=sea_valid_node,
        critical=True,
    )
    sea_valid_sources: List[str] = list(set(
        (extracted.passport_card_sea_caribbean_urls or []) + (extracted.closed_loop_policy_urls or [])
    ))
    sea_valid_claim = (
        "The U.S. passport card is valid for land and sea travel, and is acceptable for closed-loop cruises "
        "that visit Caribbean ports (but not for international air travel)."
    )
    await evaluator.verify(
        claim=sea_valid_claim,
        node=sea_valid_leaf,
        sources=sea_valid_sources,
        additional_instruction=(
            "Verify using official sources (e.g., travel.state.gov, cbp.gov, dhs.gov, or official cruise line policies). "
            "The statement must be supported explicitly by the provided URLs."
        ),
    )

    # Next: Grenada-specific acceptance of U.S. passport card for sea entry
    grenada_accept_node = evaluator.add_parallel(
        id="Grenada_Passport_Card_Acceptance",
        desc="Verify that Grenada specifically accepts U.S. passport cards for entry by sea (cruise ship arrival)",
        parent=sea_valid_node,
        critical=True,
    )

    grenada_accept_leaf = evaluator.add_leaf(
        id="Grenada_Passport_Card_Acceptance_Check",
        desc="Grenada accepts U.S. passport cards for sea entry (cruise passengers)",
        parent=grenada_accept_node,
        critical=True,
    )
    grenada_accept_sources = extracted.grenada_acceptance_urls or []
    grenada_accept_claim = (
        "Grenada accepts U.S. passport cards for entry for cruise ship passengers arriving by sea."
    )
    await evaluator.verify(
        claim=grenada_accept_claim,
        node=grenada_accept_leaf,
        sources=grenada_accept_sources,
        additional_instruction=(
            "This must be confirmed by an official or authoritative source (Grenada government domain .gov.gd, "
            "U.S. State Department country page for Grenada, or an official port authority). "
            "If the URLs do not explicitly support the claim, mark as not supported."
        ),
    )

    # Explicit existence check for an official acceptance source URL
    has_official_acceptance = any(is_official_url(u) for u in grenada_accept_sources)
    evaluator.add_custom_node(
        result=has_official_acceptance,
        id="Document_Acceptance_Source_URL",
        desc="Provide official URL confirming Grenada accepts passport cards for sea entry",
        parent=grenada_accept_node,
        critical=True,
    )


async def build_passport_validity_requirements(
    evaluator: Evaluator,
    parent_node,
    extracted: RequirementsExtraction,
) -> None:
    """
    Build and verify the Passport_Validity_Requirements branch.
    """
    validity_node = evaluator.add_parallel(
        id="Passport_Validity_Requirements",
        desc="Verify that the passport card's expiration date satisfies all validity requirements",
        parent=parent_node,
        critical=True,
    )

    # 1) Valid through entire cruise duration
    valid_through_cruise = PASSPORT_CARD_EXPIRY >= RETURN_DATE
    evaluator.add_custom_node(
        result=valid_through_cruise,
        id="Valid_Throughout_Cruise_Duration",
        desc=(
            f"Verify the passport card remains valid for the entire cruise period ({DEPARTURE_DATE.isoformat()}–"
            f"{RETURN_DATE.isoformat()}). Passport card expiration: {PASSPORT_CARD_EXPIRY.isoformat()}."
        ),
        parent=validity_node,
        critical=True,
    )

    # 2) Six-month validity from Grenada arrival
    six_month_node = evaluator.add_parallel(
        id="Six_Month_Validity_From_Arrival",
        desc=(
            f"Check 6-month validity from Grenada arrival ({GRENADA_ARRIVAL_DATE.isoformat()}); "
            "required validity until arrival+6 months."
        ),
        parent=validity_node,
        critical=True,
    )

    required_valid_until = add_months(GRENADA_ARRIVAL_DATE, 6)  # expected March 4, 2027
    six_month_computation_ok = PASSPORT_CARD_EXPIRY >= required_valid_until
    evaluator.add_custom_node(
        result=six_month_computation_ok,
        id="Six_Month_Validity_Computation",
        desc=(
            f"Required validity until: {required_valid_until.isoformat()} (arrival + 6 months). "
            f"Passport card expiration: {PASSPORT_CARD_EXPIRY.isoformat()}. "
            "Verify expiration is on/after the required date."
        ),
        parent=six_month_node,
        critical=True,
    )

    # Source existence for validity requirement
    validity_sources = extracted.grenada_validity_requirement_urls or []
    has_official_validity = any(is_official_url(u) for u in validity_sources)
    evaluator.add_custom_node(
        result=has_official_validity,
        id="Validity_Requirement_Source_URL",
        desc="Provide official URL confirming the 6‑month passport validity requirement for Grenada",
        parent=six_month_node,
        critical=True,
    )

    # Policy verification leaf: Grenada requires 6-month validity beyond entry
    validity_policy_leaf = evaluator.add_leaf(
        id="Six_Month_Validity_Policy_Verified",
        desc="Official sources confirm Grenada requires passports valid at least 6 months beyond the date of entry",
        parent=six_month_node,
        critical=True,
    )
    validity_policy_claim = (
        "Grenada requires passports to be valid for at least 6 months beyond the date of entry."
    )
    await evaluator.verify(
        claim=validity_policy_claim,
        node=validity_policy_leaf,
        sources=validity_sources,
        additional_instruction=(
            "Confirm the policy explicitly from official sources (Grenada government or U.S. State Department). "
            "General travel blogs or unofficial sources are insufficient."
        ),
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
    Evaluate the answer for the Grenada passport card cruise eligibility task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel aggregator
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

    # Add a top-level assessment node (to mirror JSON naming)
    assessment_node = evaluator.add_parallel(
        id="Passport_Card_Eligibility_Assessment",
        desc="Determine whether a U.S. passport card is sufficient for a closed-loop cruise from Miami to Grenada with specific dates",
        parent=root,
        critical=False,
    )

    # Extract determination and sources from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="requirements_extraction",
    )

    # Record ground truth scenario info
    evaluator.add_ground_truth({
        "departure_port": DEPARTURE_PORT,
        "return_port": RETURN_PORT,
        "departure_date": DEPARTURE_DATE.isoformat(),
        "return_date": RETURN_DATE.isoformat(),
        "destination_country": DESTINATION_COUNTRY,
        "grenada_arrival_date": GRENADA_ARRIVAL_DATE.isoformat(),
        "passport_card_expiry": PASSPORT_CARD_EXPIRY.isoformat(),
        "arrival_plus_6_months_required_until": add_months(GRENADA_ARRIVAL_DATE, 6).isoformat(),
    }, gt_type="scenario")

    # Optionally record the extracted determination text
    evaluator.add_custom_info(
        info={
            "determination": extracted.determination,
            "reasoning": extracted.reasoning,
            "closed_loop_policy_urls": extracted.closed_loop_policy_urls,
            "passport_card_sea_caribbean_urls": extracted.passport_card_sea_caribbean_urls,
            "grenada_acceptance_urls": extracted.grenada_acceptance_urls,
            "grenada_validity_requirement_urls": extracted.grenada_validity_requirement_urls,
        },
        info_type="extraction_debug",
        info_name="extracted_answer_fields",
    )

    # Build verification branches
    await build_document_and_method_requirements(evaluator, assessment_node, extracted)
    await build_passport_validity_requirements(evaluator, assessment_node, extracted)

    # Return summary
    return evaluator.get_summary()