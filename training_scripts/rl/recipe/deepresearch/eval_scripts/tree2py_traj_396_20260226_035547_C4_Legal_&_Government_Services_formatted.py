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
TASK_ID = "ca_recording_fees_3page"
TASK_DESCRIPTION = (
    "I need to record a 3-page real property deed in California, and I want to find a county recorder's office "
    "where the recording fees are structured as follows: $15.00 for the first page and $3.00 for each additional page. "
    "Please identify one California county that has this exact fee structure, and provide the following information: "
    "the county name, the physical address of the recorder's office, the total recording fee for my 3-page document, "
    "a direct URL to the official county fee schedule page, and information about the fee charged for non-conforming "
    "page sizes (pages that are not standard 8.5 x 11 inches)."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CountyRecorderInfo(BaseModel):
    county_name: Optional[str] = None
    state: Optional[str] = None
    recorder_office_address: Optional[str] = None
    fee_schedule_url: Optional[str] = None
    address_source_url: Optional[str] = None
    non_conforming_page_fee_info: Optional[str] = None
    non_conforming_page_fee_url: Optional[str] = None
    first_page_fee: Optional[str] = None
    additional_page_fee: Optional[str] = None
    total_fee_for_3_pages: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_county_recorder_info() -> str:
    return """
    Extract exactly one California county that the answer claims matches the fee structure "$15.00 for the first page and $3.00 for each additional page" for recording real property documents.

    Return a JSON object with these fields (use null if the answer does not provide a field):
    - county_name: The county name (e.g., "Yolo County").
    - state: The state associated with the county if mentioned (expect "California" or "CA"; else null).
    - recorder_office_address: The complete physical street address of the county recorder's office as provided in the answer.
    - fee_schedule_url: A direct URL to the official county page (or PDF) that lists the recording fee schedule.
    - address_source_url: A direct URL to the official county page that lists the recorder's physical address; if not provided in the answer, set to null.
    - non_conforming_page_fee_info: The described fee wording for non-standard page sizes (pages not 8.5 x 11), exactly as stated in the answer.
    - non_conforming_page_fee_url: A direct URL to the official page that mentions the non-conforming page fee; if the answer didn’t provide a specific URL, set to null.
    - first_page_fee: The first page recording fee amount mentioned in the answer (keep exactly as written, e.g., "$15.00", "15", etc.).
    - additional_page_fee: The per additional page recording fee amount mentioned in the answer (keep exactly as written).
    - total_fee_for_3_pages: The total recording fee for a 3-page document as stated in the answer (keep exactly as written).

    IMPORTANT:
    - Extract only what the answer explicitly states. Do not invent values or URLs.
    - For URLs, extract the literal URL(s) the answer provides (including markdown links).
    - If the answer mentions multiple counties or pages, select the county the answer ties to the "$15 first page and $3 each additional page" structure. If multiple match, select the first one mentioned.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _parse_money_to_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Find the first money-like number, e.g., "$15.00", "15", "15.0"
    match = re.search(r"(\d+(?:\.\d{1,2})?)", text.replace(",", ""))
    try:
        return float(match.group(1)) if match else None
    except Exception:
        return None


def _normalize_url_list(*urls: Optional[str]) -> List[str]:
    out = []
    for u in urls:
        if u and isinstance(u, str) and len(u.strip()) > 0:
            out.append(u.strip())
    return out


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def run_verifications(evaluator: Evaluator, root_node, info: CountyRecorderInfo) -> None:
    # 1) Official website reference must be provided (critical)
    official_url_leaf = evaluator.add_leaf(
        id="Official_Website_Reference",
        desc="A direct URL to the official county recorder's office page showing the fee schedule must be provided",
        parent=root_node,
        critical=True,
    )
    official_url_claim = (
        f"The answer provides a direct official URL to the county recorder's fee schedule page: {info.fee_schedule_url}."
    )
    await evaluator.verify(
        claim=official_url_claim,
        node=official_url_leaf,
        additional_instruction=(
            "Judge only whether the answer text includes a direct URL to the official fee schedule page. "
            "You don't need to validate its content here—that is checked in other nodes. "
            "The URL should look like a county government page (often *.ca.gov or *.county.* domains) or a county-hosted PDF."
        ),
    )

    # 2) California location (critical)
    location_leaf = evaluator.add_leaf(
        id="California_Location",
        desc="The identified county must be located in California",
        parent=root_node,
        critical=True,
    )
    loc_sources = info.fee_schedule_url or None
    loc_claim = f"The county '{info.county_name}' is located in California."
    await evaluator.verify(
        claim=loc_claim,
        node=location_leaf,
        sources=loc_sources,
        additional_instruction=(
            "Look for page context indicating the county is in California (e.g., 'County of <Name>, California', "
            "California seal, or *.ca.gov domain). Minor ambiguity is acceptable if the page is clearly a California county site."
        ),
    )

    # 3) First page fee must be $15.00 (critical)
    first_fee_leaf = evaluator.add_leaf(
        id="First_Page_Recording_Fee",
        desc="The county's first page recording fee for real property documents must be $15.00",
        parent=root_node,
        critical=True,
    )
    first_fee_claim = (
        "The fee schedule explicitly lists the recording fee for the first page of a standard-sized real property document as $15.00."
    )
    await evaluator.verify(
        claim=first_fee_claim,
        node=first_fee_leaf,
        sources=info.fee_schedule_url,
        additional_instruction=(
            "Verify the base recorder's 'recording' fee for the first page is $15.00. "
            "Ignore unrelated surcharges (e.g., SB2 $75 Building Homes & Jobs Act, fraud prevention surcharges, indexing, "
            "documentary transfer tax, etc.). Focus on the recording fee row/section."
        ),
    )

    # 4) Additional page fee must be $3.00 per page (critical)
    addl_fee_leaf = evaluator.add_leaf(
        id="Additional_Page_Fee",
        desc="The county's additional page recording fee must be $3.00 per page",
        parent=root_node,
        critical=True,
    )
    addl_fee_claim = (
        "The fee schedule explicitly lists the recording fee for each additional page (beyond the first) as $3.00 per page."
    )
    await evaluator.verify(
        claim=addl_fee_claim,
        node=addl_fee_leaf,
        sources=info.fee_schedule_url,
        additional_instruction=(
            "Verify the per-page fee for additional pages in a recorded document is $3.00 each. "
            "Ignore unrelated or optional charges; focus on the standard recording fee line."
        ),
    )

    # 5) Total fee calculation must be $21.00 ($15 + $3 + $3) (critical) – custom arithmetic check
    first_amt = _parse_money_to_float(info.first_page_fee)
    addl_amt = _parse_money_to_float(info.additional_page_fee)
    total_amt = _parse_money_to_float(info.total_fee_for_3_pages)
    expected_total = (first_amt + 2 * addl_amt) if (first_amt is not None and addl_amt is not None) else None
    total_ok = (
        expected_total is not None
        and round(expected_total, 2) == 21.00
        and total_amt is not None
        and round(total_amt, 2) == 21.00
    )
    evaluator.add_custom_node(
        result=total_ok,
        id="Total_Fee_Calculation",
        desc="The total recording fee for a 3-page standard-sized real property document must be correctly calculated as $21.00 ($15.00 + $3.00 + $3.00)",
        parent=root_node,
        critical=True,
    )

    # 6) Physical office address must be provided and accurate (critical)
    address_leaf = evaluator.add_leaf(
        id="Physical_Office_Address",
        desc="A complete physical address for the county recorder's office must be provided",
        parent=root_node,
        critical=True,
    )
    address_claim = f'The physical address of the county recorder\'s office is "{info.recorder_office_address}".'
    address_sources = _normalize_url_list(info.address_source_url or None, info.fee_schedule_url or None)
    await evaluator.verify(
        claim=address_claim,
        node=address_leaf,
        sources=address_sources if address_sources else None,
        additional_instruction=(
            "Verify the stated street address against the official county page. "
            "Minor formatting variations are acceptable (e.g., punctuation, abbreviations like 'St.' vs 'Street'), "
            "but the street number and street name must match."
        ),
    )

    # 7) Non-conforming page size fee information (non-critical)
    nonconf_leaf = evaluator.add_leaf(
        id="Non_Conforming_Page_Fee",
        desc="The fee structure for non-standard page sizes (not 8.5 x 11 inches) must be documented",
        parent=root_node,
        critical=False,
    )
    nonconf_claim = (
        f"The fee schedule includes a fee for non-conforming page sizes (not 8.5 x 11 inches), described as: "
        f"{info.non_conforming_page_fee_info}."
    )
    nonconf_sources = _normalize_url_list(info.non_conforming_page_fee_url or None, info.fee_schedule_url or None)
    await evaluator.verify(
        claim=nonconf_claim,
        node=nonconf_leaf,
        sources=nonconf_sources if nonconf_sources else None,
        additional_instruction=(
            "Look for phrases like 'non-conforming', 'non-standard page size', 'pages exceeding 8.5 x 11', 'oversized pages'. "
            "Confirm the fee referenced in the answer is present on the official schedule. Minor wording differences are acceptable."
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
    Evaluate an answer for the California county recorder fee structure task.
    """
    # Initialize evaluator – root node is non-critical by default; use PARALLEL aggregation
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
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_county_recorder_info(),
        template_class=CountyRecorderInfo,
        extraction_name="county_recorder_info",
    )

    # Add expected structure info for transparency
    evaluator.add_ground_truth(
        {
            "expected_fee_structure": {"first_page": "$15.00", "additional_page": "$3.00"},
            "expected_total_for_3_pages": "$21.00",
            "requirements": [
                "California county",
                "Physical recorder office address",
                "Official fee schedule URL",
                "Non-conforming page size fee info (non-critical)",
            ],
        },
        gt_type="expected_requirements",
    )

    # Run verifications to build the tree and assign leaf results
    await run_verifications(evaluator, root, extracted_info)

    # Return evaluation summary
    return evaluator.get_summary()