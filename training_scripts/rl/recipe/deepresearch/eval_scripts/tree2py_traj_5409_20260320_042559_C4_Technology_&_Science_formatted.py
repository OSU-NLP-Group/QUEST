import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_mobile_outage_2024_highest_911_blocks"
TASK_DESCRIPTION = (
    "In 2024, several major U.S. mobile telecommunications carriers experienced significant network outages that impacted emergency services. "
    "Among all documented major U.S. mobile carrier outages that occurred in calendar year 2024 and resulted in blocked emergency 911 calls, "
    "identify the specific outage that had the highest confirmed number of blocked 911 call attempts. For this outage, provide: "
    "1. The carrier name, 2. The specific date (month, day, year), 3. The technical root cause as documented in official sources, "
    "4. The duration of the outage. Include reference URLs from credible sources (such as FCC reports, official carrier statements, or reputable news reporting) "
    "to support your answer."
)


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class OutageSelection(BaseModel):
    """
    Extracted details for the single outage the answer claims had the highest
    number of blocked 911 call attempts in 2024.
    """
    carrier_name: Optional[str] = None
    outage_date: Optional[str] = None  # Free-form (e.g., 'February 22, 2024')
    root_cause: Optional[str] = None   # Free-form technical cause text
    outage_duration: Optional[str] = None  # Free-form (e.g., '5 hours', 'from 7am to noon')
    blocked_911_calls: Optional[str] = None  # Keep as string to allow 'xxx+' or 'approximately N'
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt builder                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_selection() -> str:
    return """
    From the provided answer, extract the SINGLE outage that the answer claims had the highest confirmed number of blocked 911 call attempts among all major U.S. mobile carrier outages in calendar year 2024.

    Extract these fields exactly as written in the answer:
    1) carrier_name: The mobile carrier's name (e.g., AT&T, Verizon, T-Mobile).
    2) outage_date: The specific date of the outage as stated in the answer (month, day, year). Use the same format the answer uses (e.g., 'February 22, 2024').
    3) root_cause: The technical root cause as documented (e.g., software update error, configuration change, backbone failure). Keep the key technical wording.
    4) outage_duration: How long the outage lasted (e.g., '5 hours', 'approximately 7 hours').
    5) blocked_911_calls: The number of blocked 911 call attempts during this outage as stated (allow any formatting, e.g., '12,300', 'about 10k', '12,300+').
    6) reference_urls: A list of ALL explicit URLs included in the answer that pertain to or support this outage (FCC reports, official carrier statements, reputable news). Extract only actual URL strings (plain URLs or those inside markdown links). Do not invent URLs.

    Rules:
    - If a field is not present in the answer, set it to null (or an empty list for reference_urls).
    - Do not normalize numbers or dates; keep them as the answer presents them.
    - Extract only explicitly mentioned URLs for reference_urls. If none, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nz(s: Optional[str]) -> str:
    """Return empty string if None, else the stripped string."""
    return (s or "").strip()


def _trim_urls(urls: List[str], max_n: int = 12) -> List[str]:
    """Limit number of URLs to avoid excessive verification overhead."""
    return [u for u in urls if isinstance(u, str) and u.strip()][:max_n]


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_outage_details(
    evaluator: Evaluator,
    parent_node,
    extracted: OutageSelection,
) -> None:
    """
    Build the verification subtree under the critical parent node and
    run the checks following the rubric.
    """
    # Use up to 12 URLs to control cost while keeping sufficient coverage
    sources = _trim_urls(extracted.reference_urls, max_n=12)

    # 1) Carrier Name
    carrier_node = evaluator.add_leaf(
        id="Carrier_Name",
        desc="The answer identifies the correct mobile telecommunications carrier that experienced the outage",
        parent=parent_node,
        critical=True
    )
    carrier_claim = (
        f"The selected outage occurred in 2024 and was experienced by the mobile carrier '{_nz(extracted.carrier_name)}'. "
        f"If verifying via URLs, the sources should clearly tie this same outage (same timeframe/date) to this carrier and indicate it resulted in blocked 911 calls."
    )
    await evaluator.verify(
        claim=carrier_claim,
        node=carrier_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Mark as supported only if at least one provided URL explicitly documents that this 2024 outage (same timeframe/date) "
            "was experienced by the named carrier and that it involved 911 call impact. "
            "If no valid URLs are provided or they don't substantiate this, mark as not supported."
        ),
    )

    # 2) Outage Date
    date_node = evaluator.add_leaf(
        id="Outage_Date",
        desc="The answer provides the correct specific date (month, day, year) when the outage occurred",
        parent=parent_node,
        critical=True
    )
    date_claim = (
        f"The specific date of the selected outage was '{_nz(extracted.outage_date)}' (in 2024). "
        f"Accept common formatting variations for the same calendar date (e.g., 'Feb 22, 2024' vs. 'February 22, 2024')."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Mark as supported only if at least one provided URL explicitly states the same calendar date for the outage. "
            "Minor formatting differences are acceptable. If URLs are missing or do not confirm the date, mark as not supported."
        ),
    )

    # 3) Root Cause
    cause_node = evaluator.add_leaf(
        id="Root_Cause",
        desc="The answer states the correct documented technical root cause of the outage",
        parent=parent_node,
        critical=True
    )
    cause_claim = (
        f"The documented technical root cause of the selected outage was: '{_nz(extracted.root_cause)}'. "
        f"This phrasing should match what official or credible sources report."
    )
    await evaluator.verify(
        claim=cause_claim,
        node=cause_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Mark as supported only if at least one provided URL explicitly describes the same technical root cause "
            "(allowing minor paraphrasing). Prefer FCC or official carrier statements when available. "
            "If URLs are missing or do not confirm the cause, mark as not supported."
        ),
    )

    # 4) Outage Duration
    duration_node = evaluator.add_leaf(
        id="Outage_Duration",
        desc="The answer provides accurate information about how long the outage lasted",
        parent=parent_node,
        critical=True
    )
    duration_claim = (
        f"The selected outage lasted '{_nz(extracted.outage_duration)}'. "
        f"Reasonable rounding or approximate expressions are acceptable if they represent the same duration window."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Mark as supported only if at least one provided URL states a duration that reasonably matches the claimed duration "
            "(allow small rounding/wording differences). If URLs are missing or do not confirm duration, mark as not supported."
        ),
    )

    # 5) Blocked 911 Calls Count
    blocked_node = evaluator.add_leaf(
        id="Blocked_911_Calls_Count",
        desc="The answer provides the correct number of blocked emergency 911 call attempts during the outage",
        parent=parent_node,
        critical=True
    )
    blocked_claim = (
        f"The selected outage resulted in '{_nz(extracted.blocked_911_calls)}' blocked 911 call attempts. "
        f"Minor formatting differences (e.g., commas, rounding like '12,300' vs '12.3k') are acceptable if they represent the same magnitude/value."
    )
    await evaluator.verify(
        claim=blocked_claim,
        node=blocked_node,
        sources=sources if sources else None,
        additional_instruction=(
            "Mark as supported only if at least one provided URL reports the same blocked 911 call count (allowing minor numeric formatting or rounding). "
            "If URLs are missing or do not confirm the count, mark as not supported."
        ),
    )

    # 6) Selection Criterion: Highest among 2024 outages with 911 blocks
    selection_node = evaluator.add_leaf(
        id="Selection_Criterion_Verification",
        desc="The identified outage is verifiably the one with the highest number of blocked 911 calls among all major U.S. mobile carrier outages in calendar year 2024",
        parent=parent_node,
        critical=True
    )
    selection_claim = (
        f"Among all documented major U.S. mobile carrier outages in calendar year 2024 that resulted in blocked 911 calls, "
        f"the outage involving '{_nz(extracted.carrier_name)}' on '{_nz(extracted.outage_date)}' had the highest confirmed number of blocked 911 call attempts."
    )
    await evaluator.verify(
        claim=selection_claim,
        node=selection_node,
        sources=sources if sources else None,
        additional_instruction=(
            "To mark as supported, the provided sources should either (a) explicitly state that this 2024 outage had the most/highest number of "
            "blocked 911 calls among comparable 2024 outages, or (b) collectively provide specific blocked-911-call counts for multiple 2024 outages "
            "showing this one is the largest. If sources are missing or insufficient to reasonably establish 'highest', mark as not supported."
        ),
    )

    # 7) Reference URLs presence and credibility
    refs_node = evaluator.add_leaf(
        id="Reference_URLs",
        desc="The answer includes credible and verifiable reference URLs (such as FCC reports, official carrier statements, or reputable news sources) that support the provided information",
        parent=parent_node,
        critical=True
    )
    refs_claim = (
        "At least one of the provided reference URLs is a credible, verifiable source (e.g., FCC report, an official carrier statement, or reputable news reporting) "
        "that documents this outage and supports its details (carrier, date, root cause, duration, and blocked 911 call attempts)."
    )
    await evaluator.verify(
        claim=refs_claim,
        node=refs_node,
        sources=sources if sources else None,
        additional_instruction=(
            "If no URLs are provided, or the URLs are not credible/relevant to the outage details, mark as not supported. "
            "Prefer FCC and official statements when available; high-quality news outlets are acceptable if they substantiate the details."
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
    Entry point used by the evaluation harness. Builds the verification tree
    and returns a standard summary dict.
    """
    # 1) Initialize evaluator/root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Only one major group below, but keep general
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

    # 2) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_outage_selection(),
        template_class=OutageSelection,
        extraction_name="selected_outage_2024_highest_911_blocks",
    )

    # 3) Build the critical node for all checks
    main_node = evaluator.add_parallel(
        id="Outage_Identification_and_Details",
        desc="Correctly identify the 2024 U.S. mobile carrier outage with the highest number of blocked 911 calls and provide all required details",
        parent=root,
        critical=True,  # All children must be critical as per rubric
    )

    # 4) Run verifications
    await verify_outage_details(evaluator, main_node, extracted)

    # 5) Optionally record custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "extracted_summary": extracted.dict(),
            "num_reference_urls": len(extracted.reference_urls or []),
        },
        info_type="extraction_summary",
        info_name="extracted_outage_selection"
    )

    # 6) Return the standardized summary
    return evaluator.get_summary()