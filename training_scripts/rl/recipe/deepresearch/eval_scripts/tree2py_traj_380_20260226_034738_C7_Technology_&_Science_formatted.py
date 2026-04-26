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
TASK_ID = "us_carrier_outages_2024_2026"
TASK_DESCRIPTION = (
    "Identify two major telecommunications carrier network outages that occurred in the United States between January 1, 2024 and January 31, 2026 (inclusive), "
    "where each outage met ALL of the following criteria: (1) the outage lasted at least 10 hours continuously, "
    "(2) the outage either affected at least 1 million customers OR blocked at least 90 million voice calls, and "
    "(3) the outage was investigated or documented by the FCC or resulted in official company statements. "
    "For each qualifying outage, provide: the carrier name, the specific date (month, day, year) when the outage began, "
    "evidence that the duration met the 10-hour threshold, evidence that the impact threshold was met, the documented technical cause of the outage, "
    "whether the FCC issued an investigation or report, and what compensation (if any) the carrier offered to affected customers."
)

DATE_RANGE_START = "January 1, 2024"
DATE_RANGE_END = "January 31, 2026"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OutageItem(BaseModel):
    """Structured information for a single outage."""
    carrier_name: Optional[str] = None
    outage_start_date: Optional[str] = None

    duration_hours_claim: Optional[str] = None
    duration_evidence_text: Optional[str] = None
    duration_sources: List[str] = Field(default_factory=list)

    impact_metric: Optional[str] = None   # e.g., "customers" or "voice calls"
    impact_value: Optional[str] = None    # e.g., "1.2 million", "92 million"
    impact_evidence_text: Optional[str] = None
    impact_sources: List[str] = Field(default_factory=list)

    technical_cause: Optional[str] = None
    cause_sources: List[str] = Field(default_factory=list)

    fcc_investigation_or_company_statement: Optional[str] = None  # free text indicating FCC involvement or official statements
    fcc_sources: List[str] = Field(default_factory=list)

    compensation: Optional[str] = None
    compensation_sources: List[str] = Field(default_factory=list)

    general_sources: List[str] = Field(default_factory=list)  # any additional URLs provided for this outage


class OutagesExtraction(BaseModel):
    """Top-level extraction containing up to two outages."""
    outages: List[OutageItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outages() -> str:
    return (
        "Extract up to two distinct major U.S. telecommunications carrier network outages described in the answer text. "
        "Only include outages that occurred between January 1, 2024 and January 31, 2026 (inclusive) IF the answer claims they qualify. "
        "For each outage, return the following fields:\n"
        "1. carrier_name: The telecommunications carrier involved (e.g., AT&T, Verizon, T-Mobile).\n"
        "2. outage_start_date: The specific calendar date (Month Day, Year) when the outage began (as stated in the answer).\n"
        "3. duration_hours_claim: The duration value or phrase used in the answer (e.g., '12 hours', 'over 10 hours').\n"
        "4. duration_evidence_text: A brief text snippet the answer uses to support duration (e.g., quoting news or official statements).\n"
        "5. duration_sources: All URLs cited in the answer that support the duration.\n"
        "6. impact_metric: Which impact metric is asserted in the answer for qualifying (use 'customers' or 'voice calls').\n"
        "7. impact_value: The number/value used (e.g., '1 million', '1.5 million', '90 million').\n"
        "8. impact_evidence_text: A brief text snippet the answer uses to support impact threshold.\n"
        "9. impact_sources: All URLs cited in the answer that support the impact threshold.\n"
        "10. technical_cause: The documented technical cause (e.g., 'software update error', 'fiber cut', 'routing configuration').\n"
        "11. cause_sources: All URLs cited that support the technical cause.\n"
        "12. fcc_investigation_or_company_statement: A short note indicating whether there was either FCC involvement (investigation/report/inquiry) OR official company statements.\n"
        "13. fcc_sources: All URLs cited that support FCC involvement or documentation.\n"
        "14. compensation: A short description of compensation (if any), such as 'bill credit', '$5 credit', 'data allowance', or 'none'.\n"
        "15. compensation_sources: All URLs cited that support compensation details.\n"
        "16. general_sources: Any other URLs the answer cites for this outage.\n\n"
        "IMPORTANT URL RULES:\n"
        "- Extract only actual URLs explicitly present in the answer text; include both plain and markdown-formatted links.\n"
        "- Include full URLs with protocol. If a URL is missing protocol, prepend 'http://'.\n"
        "- If any field is missing in the answer, set it to null; if URLs are missing for a sources field, return an empty array.\n\n"
        "Return JSON with an 'outages' array of objects following the schema described."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_sources(*lists: List[str]) -> List[str]:
    """Combine, deduplicate, and keep non-empty sources."""
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst or []:
            if not u:
                continue
            key = u.strip()
            if key and key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _all_sources_for_outage(item: OutageItem) -> List[str]:
    """Union of all source lists for an outage."""
    return _unique_sources(
        item.duration_sources,
        item.impact_sources,
        item.cause_sources,
        item.fcc_sources,
        item.compensation_sources,
        item.general_sources,
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_outage(
    evaluator: Evaluator,
    parent_node,
    outage: OutageItem,
    index: int,
) -> None:
    """
    Build verification nodes and run checks for a single outage.

    The rubric tree defines seven critical checks under a parallel node for each outage:
    - Carrier Name
    - Date
    - Duration Evidence (>=10 hours, continuous)
    - Impact Evidence (>=1M customers OR >=90M voice calls)
    - Technical Cause
    - FCC Investigation (adapted to allow FCC investigation OR official company statements, matching task criterion)
    - Compensation
    """
    # Determine node IDs and labels
    if index == 0:
        outage_node_id = "First_Qualifying_Outage"
        prefix = "First_Outage"
        outage_desc = "First outage meeting all specified criteria"
    else:
        outage_node_id = "Second_Qualifying_Outage"
        prefix = "Second_Outage"
        outage_desc = "Second outage meeting all specified criteria"

    # Parent outage node (parallel, non-critical to allow partial scoring across outages)
    outage_node = evaluator.add_parallel(
        id=outage_node_id,
        desc=outage_desc,
        parent=parent_node,
        critical=False,
    )

    # Consolidated sources for general checks
    all_sources = _all_sources_for_outage(outage)

    # 1) Carrier Name (critical)
    carrier_node = evaluator.add_leaf(
        id=f"{prefix}_Carrier_Name",
        desc="Provide the name of the telecommunications carrier that experienced the outage",
        parent=outage_node,
        critical=True,
    )
    carrier_name = outage.carrier_name or ""
    carrier_claim = (
        f"The telecommunications carrier that experienced this outage was {carrier_name}."
        if carrier_name.strip()
        else "The outage is associated with a specific named U.S. telecommunications carrier."
    )
    await evaluator.verify(
        claim=carrier_claim,
        node=carrier_node,
        sources=all_sources,
        additional_instruction=(
            "Confirm that the provided sources clearly identify the carrier involved in the outage. "
            "The source must explicitly mention the carrier in the context of the outage."
        ),
    )

    # 2) Outage Date (critical)
    date_node = evaluator.add_leaf(
        id=f"{prefix}_Date",
        desc="Provide the specific date (month, day, year) when the outage occurred",
        parent=outage_node,
        critical=True,
    )
    date_str = outage.outage_start_date or ""
    date_claim = (
        f"The outage began on {date_str} (month, day, year)."
        if date_str.strip()
        else "The outage has a specific start date (month, day, year) in the provided timeframe."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=_unique_sources(outage.duration_sources, all_sources),
        additional_instruction=(
            f"Verify the start date on the source(s). The date must fall between {DATE_RANGE_START} and {DATE_RANGE_END} inclusive. "
            "Allow minor timezone-related reporting variations (e.g., late evening vs. early morning next day), but ensure the start date is within the timeframe."
        ),
    )

    # 3) Duration Evidence: at least 10 hours continuously (critical)
    duration_node = evaluator.add_leaf(
        id=f"{prefix}_Duration_Evidence",
        desc="Provide evidence that the outage lasted at least 10 hours continuously",
        parent=outage_node,
        critical=True,
    )
    duration_claim = "The outage lasted at least 10 hours continuously."
    await evaluator.verify(
        claim=duration_claim,
        node=duration_node,
        sources=outage.duration_sources if outage.duration_sources else all_sources,
        additional_instruction=(
            "Confirm that the sources explicitly indicate a continuous outage of 10 or more hours. "
            "Phrases like 'about 12 hours', 'over 10 hours', 'lasted all day' are acceptable. "
            "If continuous duration is not clearly supported, mark as not supported."
        ),
    )

    # 4) Impact Evidence: >= 1M customers OR >= 90M voice calls (critical)
    impact_node = evaluator.add_leaf(
        id=f"{prefix}_Impact_Evidence",
        desc="Provide evidence that the outage either affected at least 1 million customers OR blocked at least 90 million voice calls",
        parent=outage_node,
        critical=True,
    )
    metric = (outage.impact_metric or "").lower()
    if "call" in metric:
        impact_claim = "The outage blocked at least 90 million voice calls."
    elif "customer" in metric:
        impact_claim = "The outage affected at least 1 million customers."
    else:
        impact_claim = (
            "This outage met the impact threshold by either affecting at least 1 million customers or blocking at least 90 million voice calls."
        )
    await evaluator.verify(
        claim=impact_claim,
        node=impact_node,
        sources=outage.impact_sources if outage.impact_sources else all_sources,
        additional_instruction=(
            "Check the source(s) for explicit numeric impact: either ≥1,000,000 customers affected OR ≥90,000,000 voice calls blocked. "
            "Roundings and phrasing like 'over 1 million' or 'more than 90 million' are acceptable."
        ),
    )

    # 5) Technical Cause (critical)
    cause_node = evaluator.add_leaf(
        id=f"{prefix}_Technical_Cause",
        desc="Provide the documented technical cause of the outage",
        parent=outage_node,
        critical=True,
    )
    cause_text = outage.technical_cause or ""
    cause_claim = (
        f"The documented technical cause of the outage was: {cause_text}."
        if cause_text.strip()
        else "There is a documented technical cause for this outage (as stated by official or credible sources)."
    )
    await evaluator.verify(
        claim=cause_claim,
        node=cause_node,
        sources=outage.cause_sources if outage.cause_sources else all_sources,
        additional_instruction=(
            "Confirm the technical/root cause as documented by official company statements, FCC materials, or credible reporting that cites official statements."
        ),
    )

    # 6) FCC Investigation OR Official Company Statements (critical, adapted to match task criterion)
    fcc_node = evaluator.add_leaf(
        id=f"{prefix}_FCC_Investigation",
        desc="Indicate whether the FCC issued an investigation, report, or sought information about the outage",
        parent=outage_node,
        critical=True,
    )
    fcc_or_statement_claim = (
        "The outage was either investigated/documented by the FCC (investigation, report, or inquiry) OR there were official company statements about the outage."
    )
    await evaluator.verify(
        claim=fcc_or_statement_claim,
        node=fcc_node,
        sources=_unique_sources(outage.fcc_sources, outage.compensation_sources, outage.general_sources),
        additional_instruction=(
            "Mark as supported if ANY of the following are explicitly evidenced: (a) FCC investigation/report/inquiry/documentation, "
            "or (b) official company statements (e.g., press releases, newsroom posts, investor statements, or official social posts) about the outage."
        ),
    )

    # 7) Compensation (critical)
    compensation_node = evaluator.add_leaf(
        id=f"{prefix}_Compensation",
        desc="Describe what compensation (if any) the carrier offered to affected customers",
        parent=outage_node,
        critical=True,
    )
    comp_text = outage.compensation or ""
    comp_claim = (
        f"The carrier offered the following compensation to affected customers: {comp_text}."
        if comp_text.strip()
        else "Compensation details (if any) for affected customers are available for this outage."
    )
    await evaluator.verify(
        claim=comp_claim,
        node=compensation_node,
        sources=outage.compensation_sources if outage.compensation_sources else all_sources,
        additional_instruction=(
            "Verify compensation details (e.g., bill credits, fee waivers, extra data). "
            "Accept official statements or credible reporting that cites official company confirmations. "
            "If sources do not provide compensation details and the answer claims specifics, mark as not supported."
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
    Evaluate an answer for the 'Major_US_Carrier_Outages_2024_2026' task.

    Returns a standardized summary dictionary including the verification tree and aggregated score.
    """
    # Initialize evaluator with root node (parallel aggregation)
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

    # Root description aligns with rubric
    root.desc = (
        "Identify two major United States telecommunications carrier network outages between January 1, 2024 and January 31, 2026 (inclusive) "
        "where each outage lasted at least 10 hours continuously and either affected at least 1 million customers or blocked at least 90 million voice calls."
    )

    # Extract structured outages from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_outages(),
        template_class=OutagesExtraction,
        extraction_name="outages_extraction",
    )

    # Keep only the first two outages; pad with empty if fewer
    outages = list(extracted.outages[:2])
    while len(outages) < 2:
        outages.append(OutageItem())

    # Build and verify both outages
    await verify_outage(evaluator, root, outages[0], index=0)
    await verify_outage(evaluator, root, outages[1], index=1)

    # Return summary
    return evaluator.get_summary()