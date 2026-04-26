import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_citizen_warsaw_property_permit_2026"
TASK_DESCRIPTION = (
    "A US citizen plans to purchase a residential apartment in Warsaw, Poland in 2026 for personal living purposes. "
    "What is the required permit process for this purchase? Specifically, identify: "
    "(1) whether a permit is required and from which government authority, "
    "(2) the stamp duty amount that must be paid when applying for this permit, and "
    "(3) the typical processing time for the permit application."
)

# Optional ground truth reference (used only for debugging/summary display)
GROUND_TRUTH = {
    "permit_and_authority": "Permit required; competent authority is the Polish Ministry of Interior and Administration (MSWiA).",
    "stamp_duty": "PLN 1,570 (opłata skarbowa) for the real estate acquisition permit application.",
    "processing_timeline": "Typically around 2–4 months; can take up to 6 months."
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PermitBlock(BaseModel):
    requires_permit: Optional[str] = None  # e.g., "required", "not required"
    authority: Optional[str] = None        # e.g., "MSWiA", "Ministry of Interior and Administration"
    sources: List[str] = Field(default_factory=list)


class FeeBlock(BaseModel):
    stamp_duty_amount: Optional[str] = None  # e.g., "PLN 1,570"
    sources: List[str] = Field(default_factory=list)


class TimelineBlock(BaseModel):
    processing_timeline: Optional[str] = None  # e.g., "2-4 months", "up to 6 months"
    sources: List[str] = Field(default_factory=list)


class RequirementsExtraction(BaseModel):
    permit: Optional[PermitBlock] = None
    fee: Optional[FeeBlock] = None
    timeline: Optional[TimelineBlock] = None
    general_sources: List[str] = Field(default_factory=list)  # any extra URLs cited without explicit mapping


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_requirements() -> str:
    return """
    From the provided answer, extract the following fields as a JSON object.

    1) permit:
       - requires_permit: Does the answer state that a permit is required? Use one of: "required", "not required".
                          If unclear or not stated, return null.
       - authority: The government authority named as responsible for granting the permit
                    (e.g., "MSWiA", "Ministry of Interior and Administration", "Minister of the Interior and Administration").
                    If not present, return null.
       - sources: URLs the answer cites that specifically support the permit requirement/authority claim.

    2) fee:
       - stamp_duty_amount: The stamp duty (state fee / opłata skarbowa) amount stated for the permit application (e.g., "PLN 1,570").
                            Keep the original formatting from the answer (including currency and separators).
                            If not present, return null.
       - sources: URLs the answer cites that specifically support the stamp duty amount.

    3) timeline:
       - processing_timeline: The typical processing time stated for the permit application
                              (e.g., "2-4 months", "up to 6 months", "around 3 months").
                              If not present, return null.
       - sources: URLs the answer cites that specifically support the processing timeline.

    4) general_sources:
       - Any additional URLs explicitly mentioned in the answer that are not clearly tied to a specific field above.

    SPECIAL RULES FOR URL SOURCES EXTRACTION:
    - Only include URLs explicitly present in the answer text (including markdown links). Do not invent URLs.
    - Return valid, full URLs. If a URL lacks a protocol, prepend "http://".
    - Deduplicate URLs within each list.

    Return a single JSON object with keys: permit, fee, timeline, general_sources.
    For any missing subfield, use null (for strings) or [] (for arrays).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        if not lst:
            continue
        for url in lst:
            if not url:
                continue
            if url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_permit_requirement_checks(
    evaluator: Evaluator,
    parent,
    extracted: RequirementsExtraction,
):
    """
    Build verification nodes for:
    - Whether a permit is required and which authority (MSWiA).
    - Source grounding for that claim.
    """
    # Aggregate node for permit requirement and authority
    permit_node = evaluator.add_parallel(
        id="permit_requirement",
        desc="Permit requirement and competent authority identification",
        parent=parent,
        critical=True  # Treat this section as critical (must be correct)
    )

    # Extract data
    permit = extracted.permit or PermitBlock()
    all_permit_sources = combine_sources(permit.sources, extracted.general_sources)

    # 1) Leaf: The answer explicitly states permit is required from MSWiA (simple check on the answer text)
    leaf_stmt = evaluator.add_leaf(
        id="permit_stmt_present",
        desc="Answer states that a US (non‑EU) citizen must obtain a permit from MSWiA (Polish Ministry of Interior and Administration) to purchase a residential apartment in Warsaw",
        parent=permit_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "Within the provided answer text, it is explicitly stated that a US citizen (a non‑EU citizen) "
            "must obtain a permit from the Polish Ministry of Interior and Administration (MSWiA) "
            "to purchase a residential apartment in Warsaw, Poland."
        ),
        node=leaf_stmt,
        additional_instruction=(
            "Judge only based on the answer text. Consider as equivalent phrasings such as "
            "'Minister of the Interior and Administration', 'Ministry of the Interior and Administration', "
            "'MSWiA', or the Polish term 'Minister właściwy do spraw wewnętrznych'. "
            "Minor wording differences are acceptable."
        )
    )

    # 2) Leaf: Sources for permit requirement are provided
    leaf_src_exist = evaluator.add_custom_node(
        result=(len(all_permit_sources) > 0),
        id="permit_sources_provided",
        desc="Sources (URLs) provided for permit requirement and authority claim",
        parent=permit_node,
        critical=True
    )

    # 3) Leaf: Claim supported by cited sources (web-grounded)
    leaf_supported = evaluator.add_leaf(
        id="permit_supported_by_sources",
        desc="The claim that a US (non‑EU) citizen must obtain a permit from MSWiA to purchase a residential apartment in Warsaw is supported by cited sources",
        parent=permit_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "A US citizen (a non‑EU citizen) must obtain a permit from the Polish Ministry of Interior and Administration (MSWiA) "
            "to purchase a residential apartment in Warsaw, Poland."
        ),
        node=leaf_supported,
        sources=all_permit_sources,
        additional_instruction=(
            "Accept reasonable synonyms for the authority (e.g., 'Minister of the Interior and Administration', 'Ministry of the Interior and Administration', 'MSWiA'). "
            "The claim should be clearly supported by the provided webpages. "
            "If the webpages explicitly say that no permit is required for purchasing a self‑contained residential apartment, then the claim is NOT supported."
        )
    )


async def build_application_details_checks(
    evaluator: Evaluator,
    parent,
    extracted: RequirementsExtraction,
):
    """
    Build checks for:
    - Stamp duty amount (PLN 1,570) and its sources (critical).
    - Processing timeline (2–4 months typical or up to 6 months) and its sources (non‑critical).
    """
    app_details = evaluator.add_parallel(
        id="application_details",
        desc="Application details (stamp duty and processing timeline)",
        parent=parent,
        critical=False  # Contains both critical and non‑critical subsections
    )

    # -------------------- Stamp Duty (Critical Subsection) --------------------
    stamp_section = evaluator.add_parallel(
        id="stamp_duty_section",
        desc="Stamp duty amount for the permit application",
        parent=app_details,
        critical=True  # This subsection is critical
    )

    fee = extracted.fee or FeeBlock()
    all_fee_sources = combine_sources(fee.sources, extracted.general_sources)

    # A1) Leaf: The answer states PLN 1,570 as the stamp duty
    stamp_stmt = evaluator.add_leaf(
        id="stamp_duty_stated",
        desc="Answer states the stamp duty (opłata skarbowa) for the permit application is PLN 1,570",
        parent=stamp_section,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "Within the provided answer text, it is explicitly stated that the stamp duty (state fee / opłata skarbowa) "
            "for the real estate acquisition permit application is PLN 1,570."
        ),
        node=stamp_stmt,
        additional_instruction=(
            "Judge only based on the answer text. Consider formatting variations such as 'PLN 1,570', '1 570 PLN', or 'PLN 1570' as equivalent."
        )
    )

    # A2) Leaf: Sources for the stamp duty are provided
    stamp_src_exist = evaluator.add_custom_node(
        result=(len(all_fee_sources) > 0),
        id="stamp_duty_sources_provided",
        desc="Sources (URLs) provided for stamp duty amount",
        parent=stamp_section,
        critical=True
    )

    # A3) Leaf: The PLN 1,570 amount is supported by cited sources
    stamp_supported = evaluator.add_leaf(
        id="stamp_duty_supported_by_sources",
        desc="The stated stamp duty PLN 1,570 is supported by cited sources",
        parent=stamp_section,
        critical=True
    )
    await evaluator.verify(
        claim="The stamp duty (state fee / opłata skarbowa) payable upon filing an application for a permit to acquire real estate is PLN 1,570.",
        node=stamp_supported,
        sources=all_fee_sources,
        additional_instruction=(
            "Verify on the cited webpages that the fee for the permit to acquire real estate (not notary or tax on civil law transactions) "
            "is 1,570 PLN. Accept minor formatting variations in the fee figure."
        )
    )

    # -------------------- Processing Timeline (Non‑Critical Subsection) --------------------
    timeline_section = evaluator.add_parallel(
        id="processing_timeline_section",
        desc="Typical processing timeline for the permit application",
        parent=app_details,
        critical=False
    )

    timeline = extracted.timeline or TimelineBlock()
    all_timeline_sources = combine_sources(timeline.sources, extracted.general_sources)

    # B1) Leaf: The answer provides the typical processing time (2–4 months or up to 6 months)
    timeline_stmt = evaluator.add_leaf(
        id="processing_timeline_stated",
        desc="Answer provides the processing timeline (2–4 months average, or mentions up to 6 months)",
        parent=timeline_section,
        critical=False
    )
    await evaluator.verify(
        claim=(
            "Within the provided answer text, it is stated that the permit processing typically takes about 2–4 months, "
            "or that it can take up to 6 months."
        ),
        node=timeline_stmt,
        additional_instruction=(
            "Judge only based on the answer text. Accept equivalent phrasings conveying: around 2–4 months; or up to 6 months."
        )
    )

    # B2) Leaf: Sources for processing timeline are provided (non‑critical)
    timeline_src_exist = evaluator.add_custom_node(
        result=(len(all_timeline_sources) > 0),
        id="processing_timeline_sources_provided",
        desc="Sources (URLs) provided for processing timeline",
        parent=timeline_section,
        critical=False
    )

    # B3) Leaf: The 2–4 months (up to 6 months) timeline is supported by cited sources (non‑critical)
    timeline_supported = evaluator.add_leaf(
        id="processing_timeline_supported_by_sources",
        desc="The typical processing time (2–4 months or up to 6 months) is supported by cited sources",
        parent=timeline_section,
        critical=False
    )
    await evaluator.verify(
        claim="The typical processing time for the permit to acquire real estate is around 2–4 months and may take up to 6 months.",
        node=timeline_supported,
        sources=all_timeline_sources,
        additional_instruction=(
            "Verify on the cited webpages that the usual processing time is approximately 2–4 months; acceptance of 'up to 6 months' as an upper bound is allowed."
        )
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
    Build and execute the evaluation according to the rubric for the US citizen purchasing a residential apartment in Warsaw (2026).
    """
    # Initialize evaluator and root
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Record reference info for debugging/inspection (not used for scoring)
    evaluator.add_ground_truth(GROUND_TRUTH, gt_type="reference_expectations")

    # 1) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RequirementsExtraction,
        extraction_name="extracted_requirements",
    )

    # 2) Build verification tree per rubric
    # Root is sequential; first, check permit requirement + authority; then application details
    await build_permit_requirement_checks(evaluator, root, extracted)
    await build_application_details_checks(evaluator, root, extracted)

    # 3) Return summary
    return evaluator.get_summary()