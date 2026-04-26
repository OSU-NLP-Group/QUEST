import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "glacier_ulcc_iata_fees"
TASK_DESCRIPTION = (
    "You are planning a budget trip to visit Glacier National Park in Montana. Identify the three "
    "ultra-low-cost carriers (ULCCs) that serve Glacier Park International Airport in Kalispell, Montana, "
    "and provide the airport's three-letter IATA code. Additionally, provide the entrance fee for a private "
    "vehicle for a 7-day pass at Glacier National Park, and state the cost of the America the Beautiful Annual "
    "Pass that covers all national parks for one year. Include reference URLs to official sources that verify: "
    "(1) the airlines serving Glacier Park International Airport, and (2) the entrance fees for Glacier National Park."
)

EXPECTED_ULCCS = ["Allegiant Air", "Frontier Airlines", "Avelo Airlines"]
EXPECTED_IATA_CODE = "FCA"
EXPECTED_GLACIER_VEHICLE_FEE = "$35.00"  # Allow $35 formatting variants in verification
EXPECTED_ATB_ANNUAL_PASS = "$80.00"      # Allow $80 formatting variants in verification


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TripInfoExtraction(BaseModel):
    """
    Structured extraction of the answer content.
    """
    ulccs: List[str] = Field(default_factory=list, description="ULCC airline names stated as serving FCA")
    iata_code: Optional[str] = Field(default=None, description="Three-letter IATA code stated for Glacier Park International Airport")
    vehicle_fee: Optional[str] = Field(default=None, description="Stated 7-day private vehicle entrance fee for Glacier NP")
    annual_pass_cost: Optional[str] = Field(default=None, description="Stated America the Beautiful Annual Pass price")
    airport_urls: List[str] = Field(default_factory=list, description="Reference URLs for airport/airlines verification (official airport or airline sites)")
    park_fee_urls: List[str] = Field(default_factory=list, description="Reference URLs for Glacier NP fees and/or Annual Pass (prefer official NPS pages)")


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_info() -> str:
    return """
    Extract the following fields strictly from the answer text. Do not invent or infer anything not present.

    Fields to extract:
    - ulccs: array of airline names that the answer claims are ultra-low-cost carriers (ULCCs) serving Glacier Park International Airport in Kalispell, Montana (e.g., "Allegiant Air", "Frontier Airlines", "Avelo Airlines"). Include exactly the names stated in the answer (tolerate minor variations like "Allegiant").
    - iata_code: the three-letter IATA airport code provided for Glacier Park International Airport (Kalispell). If multiple codes appear, pick the one clearly associated with the airport code claim.
    - vehicle_fee: the 7-day private vehicle entrance fee stated for Glacier National Park (as written, e.g., "$35", "$35.00", "35 dollars").
    - annual_pass_cost: the America the Beautiful Annual Pass price stated (as written, e.g., "$80", "$80.00", "80 dollars").
    - airport_urls: array of URLs that the answer cites to verify the airlines serving the airport or official airport information (airport website or official airline sites). Extract only actual URLs present.
    - park_fee_urls: array of URLs that the answer cites to verify Glacier National Park entrance fees and/or the America the Beautiful Annual Pass price. Prefer official National Park Service (nps.gov) links if present. Extract only actual URLs present.

    Special URL extraction rules:
    - Extract only URLs explicitly present (including markdown links).
    - Return full URLs with protocol when possible.
    - If a field is not present in the answer, return null (for string) or [] (for arrays).
    """


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
    Evaluate an answer for the Glacier ULCC/IATA/fees task and return a structured result.
    """
    # Initialize evaluator with a parallel root (we'll enforce criticality via children)
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

    # Extract structured info from the answer
    extracted: TripInfoExtraction = await evaluator.extract(
        prompt=prompt_extract_trip_info(),
        template_class=TripInfoExtraction,
        extraction_name="trip_info"
    )

    # Add ground truth info for transparency in summary
    evaluator.add_ground_truth({
        "expected_ulccs": EXPECTED_ULCCS,
        "expected_iata_code": EXPECTED_IATA_CODE,
        "expected_glacier_vehicle_fee": EXPECTED_GLACIER_VEHICLE_FEE,
        "expected_america_the_beautiful_annual_pass": EXPECTED_ATB_ANNUAL_PASS,
        "notes": "Fees and airlines verified against official sources. IATA code expected for Glacier Park International Airport (Kalispell) is FCA."
    })

    # ------------------------ Build Verification Tree ------------------------ #

    # 1) ULCCs identified (critical leaf)
    ulccs_leaf = evaluator.add_leaf(
        id="ulccs_identified",
        desc="The response identifies the three ULCCs serving Glacier Park International Airport: Allegiant Air, Frontier Airlines, and Avelo Airlines.",
        parent=root,
        critical=True
    )
    ulcc_claim = (
        "The answer explicitly names Allegiant Air, Frontier Airlines, and Avelo Airlines as "
        "ultra-low-cost carriers (ULCCs) that serve Glacier Park International Airport in Kalispell, Montana."
    )
    await evaluator.verify(
        claim=ulcc_claim,
        node=ulccs_leaf,
        additional_instruction=(
            "Judge only based on the answer text. Allow minor name variants (e.g., 'Allegiant' vs 'Allegiant Air'). "
            "It is acceptable if the answer also lists other (non-ULCC) airlines; this check only requires that these "
            "three ULCCs are clearly identified as serving the airport."
        )
    )

    # 2) Airport IATA code group (critical, sequential: must provide and be correct)
    iata_group = evaluator.add_sequential(
        id="airport_iata_code",
        desc="The response provides the three-letter IATA code for Glacier Park International Airport, and the code is correct for that airport.",
        parent=root,
        critical=True
    )
    # 2.1 Existence (custom critical)
    iata_exists = evaluator.add_custom_node(
        result=(extracted.iata_code is not None and str(extracted.iata_code).strip() != ""),
        id="iata_code_provided",
        desc="IATA code is provided in the response",
        parent=iata_group,
        critical=True
    )
    # 2.2 Correctness stated in answer (critical leaf)
    iata_answer_leaf = evaluator.add_leaf(
        id="iata_code_correct_in_answer",
        desc=f"The answer states the IATA code 'FCA' for Glacier Park International Airport.",
        parent=iata_group,
        critical=True
    )
    await evaluator.verify(
        claim="The answer provides the IATA code 'FCA' for Glacier Park International Airport (Kalispell, Montana).",
        node=iata_answer_leaf,
        additional_instruction="Allow 'FCA' to appear in parentheses or next to the airport name; minor formatting is fine."
    )
    # 2.3 Correctness supported by airport sources (critical leaf)
    iata_source_leaf = evaluator.add_leaf(
        id="iata_code_supported_by_sources",
        desc="The IATA code for Glacier Park International Airport is FCA (verified by the provided official airport/airline source URLs).",
        parent=iata_group,
        critical=True
    )
    await evaluator.verify(
        claim="The IATA code for Glacier Park International Airport (Kalispell, Montana) is FCA.",
        node=iata_source_leaf,
        sources=extracted.airport_urls,  # If empty, will fall back to simple verify
        additional_instruction=(
            "Verify this claim directly from the provided URL(s). Prefer an official airport page. "
            "If an official airport page is not available, an official airline page referring to 'FCA' as the "
            "airport code for Kalispell/Glacier Park International is acceptable."
        )
    )

    # 3) Glacier vehicle fee (critical leaf) — presence and correct value in answer
    vehicle_fee_leaf = evaluator.add_leaf(
        id="glacier_vehicle_fee",
        desc="The response provides the entrance fee for a private vehicle for a 7-day pass at Glacier National Park as $35.00.",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the 7-day private vehicle entrance fee for Glacier National Park as $35.00 (or '$35').",
        node=vehicle_fee_leaf,
        additional_instruction=(
            "Treat '$35' and '$35.00' as equivalent. If the answer says '35 dollars' or similar phrasing, that also counts."
        )
    )

    # 4) America the Beautiful annual pass cost (critical leaf) — presence and correct value in answer
    atb_leaf = evaluator.add_leaf(
        id="america_beautiful_cost",
        desc="The response states the cost of the America the Beautiful Annual Pass as $80.00.",
        parent=root,
        critical=True
    )
    await evaluator.verify(
        claim="The answer states the America the Beautiful Annual Pass price as $80.00 (or '$80').",
        node=atb_leaf,
        additional_instruction=(
            "Treat '$80' and '$80.00' as equivalent. '80 dollars' phrasing is acceptable as well."
        )
    )

    # 5) Airport reference URL(s) (critical, sequential)
    airport_ref_group = evaluator.add_sequential(
        id="airport_reference_url",
        desc="The response includes at least one reference URL to an official airport website or official airline information page that verifies the airlines serving Glacier Park International Airport.",
        parent=root,
        critical=True
    )
    # 5.1 Existence of at least one airport/airline URL
    airport_urls_present = evaluator.add_custom_node(
        result=(len(extracted.airport_urls) > 0),
        id="airport_urls_present",
        desc="At least one airport/airline reference URL is provided",
        parent=airport_ref_group,
        critical=True
    )
    # 5.2 Official page that verifies airlines serving (airport page listing carriers OR airline page showing service)
    airport_ref_leaf = evaluator.add_leaf(
        id="airport_ref_official_support",
        desc="At least one provided URL is an official airport or airline page that verifies airlines serving the airport.",
        parent=airport_ref_group,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This page is either (a) an official Glacier Park International Airport website page that lists or confirms "
            "the airlines serving the airport, or (b) an official airline page (Allegiant Air, Frontier Airlines, or "
            "Avelo Airlines) confirming that the airline serves Glacier Park International Airport (FCA/Kalispell)."
        ),
        node=airport_ref_leaf,
        sources=extracted.airport_urls,
        additional_instruction=(
            "Consider a page 'official' if it is on the airport's own domain (e.g., an airport-run site) or on the airline's "
            "own domain (e.g., allegiantair.com, aveloair.com, flyfrontier.com). Third-party aggregators or travel blogs "
            "are not official. The page must clearly list airlines serving the airport or confirm that the airline flies "
            "to Glacier Park International Airport (FCA/Kalispell, Montana)."
        )
    )

    # 6) Park fee reference URL(s) (critical, sequential)
    park_fee_ref_group = evaluator.add_sequential(
        id="park_fee_reference_url",
        desc="The response includes at least one reference URL to an official National Park Service source verifying Glacier National Park entrance fees (and/or the America the Beautiful Annual Pass cost).",
        parent=root,
        critical=True
    )
    # 6.1 Existence of at least one park fee URL
    park_fee_urls_present = evaluator.add_custom_node(
        result=(len(extracted.park_fee_urls) > 0),
        id="park_fee_urls_present",
        desc="At least one park fee reference URL is provided",
        parent=park_fee_ref_group,
        critical=True
    )
    # 6.2 Official NPS page supports either the vehicle fee or the pass price
    park_fee_ref_leaf = evaluator.add_leaf(
        id="park_fee_official_nps_support",
        desc="At least one provided URL is an official NPS page that states either the Glacier NP 7-day private vehicle fee ($35) or the ATB Annual Pass price ($80).",
        parent=park_fee_ref_group,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This page is an official National Park Service (NPS) webpage (nps.gov) that explicitly states either: "
            "(1) Glacier National Park's 7-day private vehicle entrance fee is $35, or "
            "(2) the America the Beautiful Annual Pass price is $80."
        ),
        node=park_fee_ref_leaf,
        sources=extracted.park_fee_urls,
        additional_instruction=(
            "Verify that the domain is nps.gov and the page text explicitly mentions either the $35 7-day vehicle fee for "
            "Glacier National Park or the $80 America the Beautiful (Interagency) Annual Pass. USGS or other non-NPS domains "
            "are not considered NPS for this check."
        )
    )

    # ------------------------ Return Summary --------------------------------- #
    return evaluator.get_summary()