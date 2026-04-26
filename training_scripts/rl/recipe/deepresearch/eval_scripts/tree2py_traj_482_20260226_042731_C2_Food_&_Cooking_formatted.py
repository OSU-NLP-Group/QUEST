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
TASK_ID = "starbucks_thanksgiving_2025_hours"
TASK_DESCRIPTION = (
    "Provides the requested information about Starbucks operating hours on Thanksgiving Day 2025: "
    "(1) Whether Starbucks stores are generally open or closed on Thanksgiving Day 2025 (November 27, 2025); "
    "(2) Typical opening and closing times if stores are generally open; "
    "(3) Whether hours vary by location; "
    "(4) Reference to official Starbucks company information or statements about Thanksgiving hours; "
    "(5) How customers can verify hours for a specific Starbucks location."
)

EXPECTED_OPEN_RANGE_HINT = "approximately 6:00 AM to 8:00 AM"
EXPECTED_CLOSE_RANGE_HINT = "approximately 12:00 PM to 4:00 PM"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ThanksgivingHoursInfo(BaseModel):
    """Extraction of structured info as stated in the answer."""
    subject: Optional[str] = None  # e.g., "Starbucks", "Starbucks stores", etc.
    thanksgiving_date_str: Optional[str] = None  # e.g., "November 27, 2025"
    operating_status: Optional[str] = None  # expected values: "open", "closed", "mixed/varies", "unknown"
    typical_open_range: Optional[str] = None  # e.g., "6:00 AM–8:00 AM", "around 6 to 8 am"
    typical_close_range: Optional[str] = None  # e.g., "12:00 PM–4:00 PM", "noon to 4 pm"
    hours_vary_by_location_mentioned: Optional[bool] = None  # True if explicitly acknowledged in the answer
    official_sources: List[str] = Field(default_factory=list)  # URLs to official Starbucks pages cited in the answer
    store_locator_url: Optional[str] = None  # Starbucks store locator URL if provided
    verify_specific_location_method: Optional[str] = None  # e.g., "store locator", "Starbucks app", etc.


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_thanksgiving_hours() -> str:
    return """
    Extract structured information about Starbucks operating hours on Thanksgiving Day 2025 from the provided answer.

    Return a JSON object with the following fields:
    1. subject: The subject as framed in the answer (e.g., "Starbucks" or "Starbucks stores").
    2. thanksgiving_date_str: The explicit date string mentioned in the answer for Thanksgiving Day 2025. Use the exact text from the answer (e.g., "November 27, 2025" or "Nov 27, 2025"). If not mentioned, set to null.
    3. operating_status: Whether the answer states Starbucks stores are generally "open", "closed", "mixed/varies", or "unknown" on Thanksgiving Day 2025. Use these exact labels. If unclear, set "unknown".
    4. typical_open_range: If the answer states typical opening times for Thanksgiving Day, extract the string exactly as written (e.g., "6:00 AM–8:00 AM"). Otherwise, set to null.
    5. typical_close_range: If the answer states typical closing times for Thanksgiving Day, extract the string exactly as written (e.g., "12:00 PM–4:00 PM"). Otherwise, set to null.
    6. hours_vary_by_location_mentioned: true if the answer explicitly acknowledges that hours vary by location; false otherwise.
    7. official_sources: Extract ALL URLs in the answer that appear to be official Starbucks company pages (domains like starbucks.com, stories.starbucks.com, news.starbucks.com, customerservice.starbucks.com). Return as an array. If none, return an empty array.
    8. store_locator_url: If a Starbucks Store Locator URL is provided (e.g., https://www.starbucks.com/store-locator), extract it exactly; otherwise set to null.
    9. verify_specific_location_method: If the answer explains how to verify hours for a specific location, extract a short label capturing the method (e.g., "store locator", "Starbucks app", "call the store"). If not explained, set to null.

    Important:
    - Do not invent any URLs; extract only those explicitly present in the answer.
    - Preserve the exact phrasing for time ranges and dates.
    - If a required field is not present in the answer, set it to null or "unknown" as instructed.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _filter_official_starbucks_urls(urls: List[str]) -> List[str]:
    """Return only official Starbucks domains."""
    official = []
    for u in urls or []:
        if not u:
            continue
        lu = u.strip().lower()
        if "starbucks.com" in lu:
            official.append(u)
    # Deduplicate while preserving order
    seen = set()
    ordered_unique = []
    for u in official:
        if u not in seen:
            seen.add(u)
            ordered_unique.append(u)
    return ordered_unique


def _normalize_operating_status(text: Optional[str]) -> str:
    """Normalize free text to one of {'open','closed','mixed/varies','unknown'}."""
    if not text:
        return "unknown"
    t = text.strip().lower()
    if "closed" in t and "open" not in t:
        return "closed"
    if "open" in t and "closed" not in t:
        return "open"
    if "vary" in t or "varies" in t or "depends" in t or "mixed" in t:
        return "mixed/varies"
    return "unknown"


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def add_subject_and_date_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: ThanksgivingHoursInfo,
) -> None:
    """Build and run the Subject & Date verification subtree."""
    subject_date_node = evaluator.add_parallel(
        id="Subject_And_Date",
        desc="Correctly frames the subject and the holiday date.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: Identify Subject as Starbucks stores
    subject_leaf = evaluator.add_leaf(
        id="Identify_Subject_Starbucks",
        desc="Identifies the subject as Starbucks coffee shops/stores.",
        parent=subject_date_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer clearly identifies Starbucks coffee shops/stores as the subject.",
        node=subject_leaf,
        additional_instruction="Check that the answer explicitly frames the subject as Starbucks stores/cafés (not unrelated entities). Minor wording variations are acceptable."
    )

    # Leaf: Specify Thanksgiving 2025 Date correctly
    date_leaf = evaluator.add_leaf(
        id="Specify_Thanksgiving_2025_Date",
        desc="Specifies Thanksgiving Day 2025 as November 27, 2025.",
        parent=subject_date_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly specifies Thanksgiving Day 2025 as November 27, 2025 (Nov 27, 2025).",
        node=date_leaf,
        additional_instruction="Accept reasonable formatting variants like 'Nov 27, 2025' or 'November 27, 2025'."
    )


async def add_operating_status_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: ThanksgivingHoursInfo,
    official_urls: List[str],
    official_ref_gate_node,
) -> None:
    """Operating status verification with source grounding."""
    normalized_status = _normalize_operating_status(extracted.operating_status)

    # Leaf: Operating Status (critical)
    status_leaf = evaluator.add_leaf(
        id="Operating_Status",
        desc="States whether Starbucks stores are generally open or closed on Thanksgiving Day 2025.",
        parent=parent_node,
        critical=True,
    )

    # If no official reference, this leaf will be skipped via prerequisite; but Official_Reference is critical and failure will fail root.
    claim = ""
    if normalized_status == "open":
        claim = "Starbucks stores are generally open on Thanksgiving Day 2025 (November 27, 2025)."
    elif normalized_status == "closed":
        claim = "Starbucks stores are generally closed on Thanksgiving Day 2025 (November 27, 2025)."
    elif normalized_status == "mixed/varies":
        claim = "Whether Starbucks stores are open on Thanksgiving Day 2025 varies by location, but many are open with reduced hours."
    else:
        # If the answer did not state a status, mark as failed directly.
        status_leaf.score = 0.0
        status_leaf.status = "failed"
        return

    # Verify using official Starbucks URLs; enforce prerequisite on Official_Reference gate
    await evaluator.verify(
        claim=claim,
        node=status_leaf,
        sources=official_urls if official_urls else None,
        additional_instruction="Judge support strictly based on official Starbucks sources provided; accept statements indicating many stores remain open with limited hours, or clear 'closed' statements if present.",
        extra_prerequisites=[official_ref_gate_node] if official_ref_gate_node else None
    )


async def add_typical_hours_check(
    evaluator: Evaluator,
    parent_node,
    extracted: ThanksgivingHoursInfo,
) -> None:
    """
    Typical hours check (non-critical).
    Only relevant if the provided answer claims stores are open.
    This check verifies that the answer provides typical opening range ~6–8 AM and closing range ~12–4 PM.
    """
    normalized_status = _normalize_operating_status(extracted.operating_status)

    typ_leaf = evaluator.add_leaf(
        id="Typical_Hours_If_Open",
        desc=(
            "If the answer states stores are generally open, it provides typical opening time range "
            f"({EXPECTED_OPEN_RANGE_HINT}) and typical closing time range ({EXPECTED_CLOSE_RANGE_HINT})."
        ),
        parent=parent_node,
        critical=False,  # Conditional requirement: do not fail entire evaluation when stores are stated closed.
    )

    if normalized_status != "open":
        # Not applicable when stores are closed or status is unknown/mixed – skip this check
        typ_leaf.score = 0.0
        typ_leaf.status = "skipped"
        return

    # Verify presence of expected ranges in the answer (simple verification, checks answer content)
    claim = (
        f"The answer provides typical opening times around {EXPECTED_OPEN_RANGE_HINT} and "
        f"typical closing times around {EXPECTED_CLOSE_RANGE_HINT} for Thanksgiving Day 2025."
    )
    await evaluator.verify(
        claim=claim,
        node=typ_leaf,
        additional_instruction=(
            "Accept reasonable phrasing equivalents like '6-8 am' or 'noon to 4 pm', "
            "and words such as 'around', 'approximately', or 'typical'."
        ),
    )


async def add_hours_variability_check(
    evaluator: Evaluator,
    parent_node,
    extracted: ThanksgivingHoursInfo,
) -> None:
    """Check that the answer acknowledges that hours vary by location."""
    vary_leaf = evaluator.add_leaf(
        id="Hours_Vary_By_Location",
        desc="Acknowledges that Starbucks hours vary by location on Thanksgiving.",
        parent=parent_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The answer explicitly acknowledges that Starbucks store hours vary by location on Thanksgiving Day 2025.",
        node=vary_leaf,
        additional_instruction="Look for phrases like 'hours vary by location', 'depends on store', 'check location-specific hours', etc."
    )


async def add_official_reference_check(
    evaluator: Evaluator,
    parent_node,
    official_urls: List[str],
) -> Any:
    """
    Ensure at least one official Starbucks URL is provided in the answer.
    Implemented as a critical custom node (existence check), which also acts as a gate for source-grounded verifications.
    """
    has_official_ref = len(official_urls) > 0
    official_ref_node = evaluator.add_custom_node(
        result=has_official_ref,
        id="Official_Reference",
        desc="Provides a reference to official Starbucks company information or statements about Thanksgiving hours.",
        parent=parent_node,
        critical=True
    )
    return official_ref_node


async def add_verify_specific_location_check(
    evaluator: Evaluator,
    parent_node,
    extracted: ThanksgivingHoursInfo,
) -> None:
    """Explain how to verify hours for a specific Starbucks location (store locator)."""
    verify_leaf = evaluator.add_leaf(
        id="Verify_Specific_Location",
        desc="Explains how customers can verify hours for a specific Starbucks location by mentioning the Starbucks store locator tool.",
        parent=parent_node,
        critical=True,
    )

    # Prefer verifying with the actual store locator URL if provided
    claim = (
        "Customers can verify hours for a specific Starbucks location by using the Starbucks Store Locator tool on starbucks.com."
    )
    if extracted.store_locator_url:
        await evaluator.verify(
            claim=claim,
            node=verify_leaf,
            sources=extracted.store_locator_url,
            additional_instruction=(
                "Confirm that the provided URL is Starbucks' official Store Locator. The page should allow searching/selecting a store to view its hours."
            ),
        )
    else:
        # Fall back to answer-content verification (simple)
        await evaluator.verify(
            claim=claim,
            node=verify_leaf,
            additional_instruction=(
                "Check that the answer explicitly mentions the Starbucks Store Locator (or equivalent official store-finder on starbucks.com) as the method."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for Starbucks Thanksgiving 2025 hours.
    """
    # Initialize evaluator (root is non-critical to allow partial credit for conditional items)
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
    extracted_info: ThanksgivingHoursInfo = await evaluator.extract(
        prompt=prompt_extract_thanksgiving_hours(),
        template_class=ThanksgivingHoursInfo,
        extraction_name="thanksgiving_hours_extraction",
    )

    # Compute official Starbucks URLs based on extraction
    official_urls = _filter_official_starbucks_urls(extracted_info.official_sources)
    # If store locator is present but not included in official_sources, append for downstream use
    if extracted_info.store_locator_url:
        sl_lower = extracted_info.store_locator_url.strip().lower()
        if "starbucks.com" in sl_lower and extracted_info.store_locator_url not in official_urls:
            official_urls.append(extracted_info.store_locator_url)

    # Build verification tree following rubric
    # 1) Subject & Date
    await add_subject_and_date_checks(evaluator, root, extracted_info)

    # 2) Official Reference (critical gate)
    official_ref_gate = await add_official_reference_check(evaluator, root, official_urls)

    # 3) Operating Status (critical, source-grounded, gated by official reference)
    await add_operating_status_checks(evaluator, root, extracted_info, official_urls, official_ref_gate)

    # 4) Typical Hours (non-critical; conditional on 'open')
    await add_typical_hours_check(evaluator, root, extracted_info)

    # 5) Hours Vary By Location (critical; answer acknowledgment)
    await add_hours_variability_check(evaluator, root, extracted_info)

    # 6) Verify Specific Location method (critical; prefer store locator URL)
    await add_verify_specific_location_check(evaluator, root, extracted_info)

    # Return evaluation summary
    return evaluator.get_summary()