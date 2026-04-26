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
TASK_ID = "orforglipron_fda_review_2026"
TASK_DESCRIPTION = """
Eli Lilly's orforglipron is an oral GLP-1 receptor agonist drug candidate for obesity treatment that is currently undergoing FDA review in 2026. Based on publicly available information as of February 2026, provide a comprehensive analysis that includes:

1. The current expected FDA decision date for orforglipron's approval
2. The specific type of FDA review process designation that orforglipron is utilizing
3. A comparison of orforglipron's review timeline to standard FDA review processes, including:
   - The standard FDA review duration
   - The priority review duration (if applicable)
   - How orforglipron's expedited process compares to these standard timelines
4. Information about orforglipron's Phase 3 clinical trials, including confirmation of Phase 3 completion and context about typical Phase 3 enrollment sizes
5. The drug delivery method (oral vs. injectable)

Additionally, explain any timeline changes or delays that have occurred in the approval process, including the original expected timeframe versus the current timeline.

Your analysis must be supported by verifiable sources with specific dates, durations, and accurate characterizations of the FDA review processes.
"""
AS_OF_CONTEXT = "as of February 2026"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DecisionDateInfo(BaseModel):
    expected_decision_date: Optional[str] = None  # e.g., "June 15, 2026"
    expected_timeframe: Optional[str] = None      # e.g., "Q2 2026", "mid-2026"
    sources: List[str] = Field(default_factory=list)


class FDAReviewDesignationInfo(BaseModel):
    designation_name: Optional[str] = None  # e.g., "Priority Review", "Accelerated Approval", etc.
    designation_implications: Optional[str] = None  # explanation of what it means for timeline
    sources: List[str] = Field(default_factory=list)


class ReviewTimelineComparisonInfo(BaseModel):
    standard_duration: Optional[str] = None  # e.g., "10 months", "up to 10 months"
    standard_duration_sources: List[str] = Field(default_factory=list)

    priority_duration: Optional[str] = None  # e.g., "6 months"
    priority_duration_sources: List[str] = Field(default_factory=list)

    orforglipron_timeline_characterization: Optional[str] = None  # e.g., "priority review targeting X date"
    orforglipron_timeline_sources: List[str] = Field(default_factory=list)

    quantitative_comparison: Optional[str] = None  # explicit comparison statement, e.g., "X months vs Y months"
    # Optional dedicated sources if provided; if empty, we will use a combination of above:
    quantitative_comparison_sources: List[str] = Field(default_factory=list)


class Phase3ClinicalTrialsInfo(BaseModel):
    phase3_status: Optional[str] = None  # e.g., "completed", "ongoing", "completed for obesity indication"
    phase3_status_sources: List[str] = Field(default_factory=list)

    typical_enrollment_context: Optional[str] = None  # e.g., "typically several hundred to several thousand participants"
    typical_enrollment_sources: List[str] = Field(default_factory=list)


class DrugDeliveryMethodInfo(BaseModel):
    delivery_method: Optional[str] = None  # e.g., "oral", "injectable"
    sources: List[str] = Field(default_factory=list)


class TimelineChangesInfo(BaseModel):
    original_expected_timeframe_or_date: Optional[str] = None  # e.g., "originally expected in late 2025"
    original_sources: List[str] = Field(default_factory=list)

    change_explanation: Optional[str] = None  # explanation of change/delay/acceleration
    change_sources: List[str] = Field(default_factory=list)


class OrforglipronAnalysisExtraction(BaseModel):
    decision_date: Optional[DecisionDateInfo] = None
    review_designation: Optional[FDAReviewDesignationInfo] = None
    timeline_comparison: Optional[ReviewTimelineComparisonInfo] = None
    phase3: Optional[Phase3ClinicalTrialsInfo] = None
    delivery_method: Optional[DrugDeliveryMethodInfo] = None
    timeline_changes: Optional[TimelineChangesInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_orforglipron_analysis() -> str:
    return f"""
    Extract structured information from the answer regarding Eli Lilly's orforglipron, strictly {AS_OF_CONTEXT}. 
    You must parse explicit statements and extract all cited URLs exactly as they appear (plain or markdown). 
    Do not invent any values. If something is missing, set it to null (for strings) or [] (for lists).

    Required JSON fields:

    1) decision_date:
       - expected_decision_date: A specific date (e.g., "June 15, 2026") if provided.
       - expected_timeframe: An unambiguous timeframe if a specific date is not provided (e.g., "Q2 2026", "mid-2026").
       - sources: All URLs that support the current expected FDA decision/action date or timeframe for orforglipron.

    2) review_designation:
       - designation_name: The specific FDA review designation/program for orforglipron (e.g., "Priority Review", "Accelerated Approval", "Rolling Review").
       - designation_implications: A brief explanation of what the designation implies operationally for review speed.
       - sources: All URLs that confirm the stated designation/program for orforglipron.

    3) timeline_comparison:
       - standard_duration: The typical FDA standard review duration (e.g., "10 months").
       - standard_duration_sources: URLs supporting the standard duration (e.g., FDA guidance).
       - priority_duration: The typical FDA priority review duration (e.g., "6 months") if applicable for comparison.
       - priority_duration_sources: URLs supporting the priority review duration.
       - orforglipron_timeline_characterization: The expected/target timeline for orforglipron under its designation/program (include dates/durations if stated).
       - orforglipron_timeline_sources: URLs supporting orforglipron's specific expected/target timeline characterization.
       - quantitative_comparison: A sentence explicitly comparing orforglipron’s expected timeline against the standard and priority review durations (e.g., "Orforglipron’s priority review targets a 6-month timeline versus ~10 months standard.")
       - quantitative_comparison_sources: If the answer cites specific URLs for this explicit comparison, list them; otherwise leave empty. (We will also use the above sources for verification.)

    4) phase3:
       - phase3_status: The Phase 3 status for obesity (e.g., "completed", "ongoing") stated clearly.
       - phase3_status_sources: URLs supporting the Phase 3 status claim.
       - typical_enrollment_context: Contextual information about typical Phase 3 enrollment sizes (e.g., ranges or explanatory context).
       - typical_enrollment_sources: URLs supporting the typical enrollment context.

    5) delivery_method:
       - delivery_method: "oral" or "injectable" or a phrase clearly indicating one of these.
       - sources: URLs supporting the delivery method claim for orforglipron.

    6) timeline_changes:
       - original_expected_timeframe_or_date: The original publicly communicated expected FDA decision timing/date earlier in the process.
       - original_sources: URLs supporting the original expected timeframe/date.
       - change_explanation: A clear explanation describing the change from the original expectation to the current expectation (e.g., a delay due to additional data requests).
       - change_sources: URLs supporting that the timeline changed (original vs updated expectation).

    Return the JSON object matching the OrforglipronAnalysisExtraction schema exactly.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_delivery_method(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if "oral" in v:
        return "oral"
    if "inject" in v:
        return "injectable"
    return None


def _has_nonempty_str(value: Optional[str]) -> bool:
    return bool(value and value.strip())


def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls and len([u for u in urls if _has_nonempty_str(u)]) > 0)


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_current_fda_decision_date(evaluator: Evaluator, parent_node, ext: OrforglipronAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="Current_FDA_Decision_Date",
        desc="States the current expected FDA decision/action date (or timeframe) for orforglipron as of Feb 2026.",
        parent=parent_node,
        critical=True
    )

    info = ext.decision_date or DecisionDateInfo()
    value = info.expected_decision_date if _has_nonempty_str(info.expected_decision_date) else info.expected_timeframe

    # Presence check
    evaluator.add_custom_node(
        result=_has_nonempty_str(value),
        id="Provides_Current_Decision_Date_or_Timeframe",
        desc="Gives a specific current expected FDA decision/action date OR an unambiguous official timeframe (as of Feb 2026).",
        parent=node,
        critical=True
    )

    # Source-supported claim
    if _has_sources(info.sources) and _has_nonempty_str(value):
        leaf = evaluator.add_leaf(
            id="Cites_Source_for_Current_Decision_Date",
            desc="Source supports the stated current expected decision date/timeframe.",
            parent=node,
            critical=True
        )
        claim = f"As of Feb 2026, the current expected FDA decision/action date or timeframe for orforglipron is '{value}'."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=info.sources,
            additional_instruction="Verify the page explicitly states or clearly supports the current expected FDA decision/action date or timeframe for orforglipron."
        )
    else:
        evaluator.add_leaf(
            id="Cites_Source_for_Current_Decision_Date",
            desc="Source supports the stated current expected decision date/timeframe.",
            parent=node,
            critical=True,
            score=0.0,
            status="failed"
        )


async def verify_fda_review_designation(evaluator: Evaluator, parent_node, ext: OrforglipronAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="FDA_Review_Designation",
        desc="Identifies the specific FDA review designation/program and implications.",
        parent=parent_node,
        critical=True
    )

    info = ext.review_designation or FDAReviewDesignationInfo()

    # Names designation
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.designation_name),
        id="Names_Review_Designation",
        desc="Names the specific FDA review designation/program for the application.",
        parent=node,
        critical=True
    )

    # Explains implications
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.designation_implications),
        id="Explains_Designation_Implications",
        desc="Explains operational implications for review speed vs. standard.",
        parent=node,
        critical=True
    )

    # Source-supported designation
    if _has_sources(info.sources) and _has_nonempty_str(info.designation_name):
        leaf = evaluator.add_leaf(
            id="Cites_Source_for_Designation",
            desc="Source confirms the stated designation/program for orforglipron.",
            parent=node,
            critical=True
        )
        claim = f"Orforglipron's FDA review designation/program is '{info.designation_name}'."
        await evaluator.verify(
            claim=claim,
            node=leaf,
            sources=info.sources,
            additional_instruction="Verify the source explicitly confirms orforglipron's FDA review designation/program."
        )
    else:
        evaluator.add_leaf(
            id="Cites_Source_for_Designation",
            desc="Source confirms the stated designation/program for orforglipron.",
            parent=node,
            critical=True,
            score=0.0,
            status="failed"
        )


async def verify_review_timeline_comparison(evaluator: Evaluator, parent_node, ext: OrforglipronAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="Review_Timeline_Comparison",
        desc="Compares orforglipron’s expected review speed to standard and priority timelines with durations and quantitative context.",
        parent=parent_node,
        critical=True
    )

    info = ext.timeline_comparison or ReviewTimelineComparisonInfo()

    # Standard duration presence
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.standard_duration),
        id="States_Standard_FDA_Review_Duration",
        desc="States typical standard FDA review duration (general).",
        parent=node,
        critical=True
    )

    # Source for standard
    if _has_sources(info.standard_duration_sources) and _has_nonempty_str(info.standard_duration):
        leaf_std = evaluator.add_leaf(
            id="Cites_Source_for_Standard_Duration",
            desc="Source supports the standard FDA review duration.",
            parent=node,
            critical=True
        )
        claim_std = f"The typical FDA standard review duration is '{info.standard_duration}'."
        await evaluator.verify(
            claim=claim_std,
            node=leaf_std,
            sources=info.standard_duration_sources,
            additional_instruction="Confirm the general standard review duration (e.g., PDUFA standard ~10 months) per FDA guidance or authoritative sources."
        )
    else:
        evaluator.add_leaf(
            id="Cites_Source_for_Standard_Duration",
            desc="Source supports the standard FDA review duration.",
            parent=node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # Priority duration presence
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.priority_duration),
        id="States_Priority_Review_Duration",
        desc="States typical FDA priority review duration (general).",
        parent=node,
        critical=True
    )

    # Source for priority
    if _has_sources(info.priority_duration_sources) and _has_nonempty_str(info.priority_duration):
        leaf_pri = evaluator.add_leaf(
            id="Cites_Source_for_Priority_Duration",
            desc="Source supports the priority review duration.",
            parent=node,
            critical=True
        )
        claim_pri = f"The typical FDA priority review duration is '{info.priority_duration}'."
        await evaluator.verify(
            claim=claim_pri,
            node=leaf_pri,
            sources=info.priority_duration_sources,
            additional_instruction="Confirm the general priority review duration (e.g., ~6 months) per FDA guidance or authoritative sources."
        )
    else:
        evaluator.add_leaf(
            id="Cites_Source_for_Priority_Duration",
            desc="Source supports the priority review duration.",
            parent=node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # Orforglipron timeline characterization presence
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.orforglipron_timeline_characterization),
        id="Characterizes_Orforglipron_Review_Speed",
        desc="States orforglipron’s expected/target review speed/timeline with dates/durations tied to its designation/program.",
        parent=node,
        critical=True
    )

    # Source for orforglipron timeline characterization
    if _has_sources(info.orforglipron_timeline_sources) and _has_nonempty_str(info.orforglipron_timeline_characterization):
        leaf_orf = evaluator.add_leaf(
            id="Cites_Source_for_Orforglipron_Timeline_Characterization",
            desc="Source supports the characterization/target timeline for orforglipron under its designation/program.",
            parent=node,
            critical=True
        )
        claim_orf = f"Orforglipron’s expected/target review speed/timeline is described as: '{info.orforglipron_timeline_characterization}'."
        await evaluator.verify(
            claim=claim_orf,
            node=leaf_orf,
            sources=info.orforglipron_timeline_sources,
            additional_instruction="Verify that the source explicitly supports orforglipron’s expected/target review timeline characterization under its designation/program."
        )
    else:
        evaluator.add_leaf(
            id="Cites_Source_for_Orforglipron_Timeline_Characterization",
            desc="Source supports the characterization/target timeline for orforglipron under its designation/program.",
            parent=node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # Quantitative comparison leaf (must be explicitly stated)
    # Use dedicated comparison sources if provided; otherwise combine related sources
    comp_sources = info.quantitative_comparison_sources
    if not _has_sources(comp_sources):
        comp_sources = []
        comp_sources.extend(info.orforglipron_timeline_sources or [])
        comp_sources.extend(info.standard_duration_sources or [])
        comp_sources.extend(info.priority_duration_sources or [])

    if _has_nonempty_str(info.quantitative_comparison) and _has_sources(comp_sources):
        leaf_cmp = evaluator.add_leaf(
            id="Quantitative_Comparison_to_Standard_and_Priority",
            desc="Explicit quantitative comparison between orforglipron timeline and standard/priority review durations.",
            parent=node,
            critical=True
        )
        claim_cmp = f"The following quantitative comparison is correct: '{info.quantitative_comparison}'."
        await evaluator.verify(
            claim=claim_cmp,
            node=leaf_cmp,
            sources=comp_sources,
            additional_instruction="Verify that the numbers/timelines stated in the comparison are supported by the cited sources for standard, priority, and orforglipron timelines."
        )
    else:
        evaluator.add_leaf(
            id="Quantitative_Comparison_to_Standard_and_Priority",
            desc="Explicit quantitative comparison between orforglipron timeline and standard/priority review durations.",
            parent=node,
            critical=True,
            score=0.0,
            status="failed"
        )


async def verify_phase3_clinical_trials(evaluator: Evaluator, parent_node, ext: OrforglipronAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="Phase_3_Clinical_Trials",
        desc="Addresses Phase 3 trial status and typical enrollment context.",
        parent=parent_node,
        critical=True
    )

    info = ext.phase3 or Phase3ClinicalTrialsInfo()

    # Phase 3 status presence
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.phase3_status),
        id="States_Phase_3_Status",
        desc="States Phase 3 status for obesity clearly (completed/ongoing/other).",
        parent=node,
        critical=True
    )

    # Source for Phase 3 status
    if _has_sources(info.phase3_status_sources) and _has_nonempty_str(info.phase3_status):
        leaf_p3 = evaluator.add_leaf(
            id="Cites_Source_for_Phase_3_Status",
            desc="Source supports the Phase 3 status claim.",
            parent=node,
            critical=True
        )
        claim_p3 = f"The Phase 3 status for orforglipron (obesity) is: '{info.phase3_status}'."
        await evaluator.verify(
            claim=claim_p3,
            node=leaf_p3,
            sources=info.phase3_status_sources,
            additional_instruction="Verify that the source explicitly confirms the Phase 3 status for orforglipron in obesity."
        )
    else:
        evaluator.add_leaf(
            id="Cites_Source_for_Phase_3_Status",
            desc="Source supports the Phase 3 status claim.",
            parent=node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # Typical enrollment context presence
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.typical_enrollment_context),
        id="Provides_Typical_Phase_3_Enrollment_Context",
        desc="Provides typical Phase 3 enrollment size context.",
        parent=node,
        critical=True
    )

    # Source for typical enrollment
    if _has_sources(info.typical_enrollment_sources) and _has_nonempty_str(info.typical_enrollment_context):
        leaf_enr = evaluator.add_leaf(
            id="Cites_Source_for_Typical_Enrollment_Context",
            desc="Source supports the typical Phase 3 enrollment context.",
            parent=node,
            critical=True
        )
        claim_enr = f"Typical Phase 3 enrollment context is described as: '{info.typical_enrollment_context}'."
        await evaluator.verify(
            claim=claim_enr,
            node=leaf_enr,
            sources=info.typical_enrollment_sources,
            additional_instruction="Verify that the source supports the stated typical Phase 3 enrollment context."
        )
    else:
        evaluator.add_leaf(
            id="Cites_Source_for_Typical_Enrollment_Context",
            desc="Source supports the typical Phase 3 enrollment context.",
            parent=node,
            critical=True,
            score=0.0,
            status="failed"
        )


async def verify_drug_delivery_method(evaluator: Evaluator, parent_node, ext: OrforglipronAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="Drug_Delivery_Method",
        desc="Identifies whether orforglipron is oral or injectable.",
        parent=parent_node,
        critical=True
    )

    info = ext.delivery_method or DrugDeliveryMethodInfo()
    normalized = _normalize_delivery_method(info.delivery_method)

    # Delivery method presence and validity ("oral" or "injectable")
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.delivery_method) and normalized in {"oral", "injectable"},
        id="States_Delivery_Method",
        desc="Clearly identifies delivery method (oral vs injectable) for orforglipron.",
        parent=node,
        critical=True
    )

    # Source for delivery method
    if _has_sources(info.sources) and normalized in {"oral", "injectable"}:
        leaf_dm = evaluator.add_leaf(
            id="Cites_Source_for_Delivery_Method",
            desc="Source supports the delivery method claim.",
            parent=node,
            critical=True
        )
        claim_dm = f"Orforglipron is {normalized}."
        await evaluator.verify(
            claim=claim_dm,
            node=leaf_dm,
            sources=info.sources,
            additional_instruction="Verify that the source indicates orforglipron is oral (tablet/capsule) or injectable."
        )
    else:
        evaluator.add_leaf(
            id="Cites_Source_for_Delivery_Method",
            desc="Source supports the delivery method claim.",
            parent=node,
            critical=True,
            score=0.0,
            status="failed"
        )


async def verify_timeline_changes_or_delays(evaluator: Evaluator, parent_node, ext: OrforglipronAnalysisExtraction) -> None:
    node = evaluator.add_parallel(
        id="Timeline_Changes_or_Delays",
        desc="Explains timeline changes/delays: original vs current expected decision timing.",
        parent=parent_node,
        critical=True
    )

    info = ext.timeline_changes or TimelineChangesInfo()
    current_info = ext.decision_date or DecisionDateInfo()
    current_val = current_info.expected_decision_date if _has_nonempty_str(current_info.expected_decision_date) else current_info.expected_timeframe

    # Original timeframe presence
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.original_expected_timeframe_or_date),
        id="States_Original_Expected_Timeframe",
        desc="States the original expected FDA decision timeframe/date.",
        parent=node,
        critical=True
    )

    # Source for original timeframe
    if _has_sources(info.original_sources) and _has_nonempty_str(info.original_expected_timeframe_or_date):
        leaf_orig = evaluator.add_leaf(
            id="Cites_Source_for_Original_Timeframe",
            desc="Source supports the original expected timeframe/date.",
            parent=node,
            critical=True
        )
        claim_orig = f"The originally communicated expected FDA decision timeframe/date for orforglipron was '{info.original_expected_timeframe_or_date}'."
        await evaluator.verify(
            claim=claim_orig,
            node=leaf_orig,
            sources=info.original_sources,
            additional_instruction="Verify that the source confirms the original expected decision timing/date."
        )
    else:
        evaluator.add_leaf(
            id="Cites_Source_for_Original_Timeframe",
            desc="Source supports the original expected timeframe/date.",
            parent=node,
            critical=True,
            score=0.0,
            status="failed"
        )

    # Explanation of change presence
    evaluator.add_custom_node(
        result=_has_nonempty_str(info.change_explanation),
        id="Explains_Change_from_Original_to_Current",
        desc="Explains the change (delay or acceleration) from original expectation to current expectation with temporal comparison.",
        parent=node,
        critical=True
    )

    # Source for timeline change (compare original vs current)
    change_sources_combined: List[str] = []
    change_sources_combined.extend(info.change_sources or [])
    change_sources_combined.extend(current_info.sources or [])
    change_sources_combined.extend(info.original_sources or [])

    if _has_sources(change_sources_combined) and _has_nonempty_str(info.original_expected_timeframe_or_date) and _has_nonempty_str(current_val):
        leaf_change = evaluator.add_leaf(
            id="Cites_Source_for_Timeline_Change",
            desc="Source supports that the timeline changed from original to current expectation.",
            parent=node,
            critical=True
        )
        claim_change = (
            f"The expected FDA decision timing for orforglipron changed from '{info.original_expected_timeframe_or_date}' "
            f"to '{current_val}', indicating a delay or acceleration. Explanation: '{info.change_explanation}'."
        )
        await evaluator.verify(
            claim=claim_change,
            node=leaf_change,
            sources=change_sources_combined,
            additional_instruction="Verify that the sources indicate a change in expected timing (original vs updated/current) and support the provided explanation."
        )
    else:
        evaluator.add_leaf(
            id="Cites_Source_for_Timeline_Change",
            desc="Source supports that the timeline changed from original to current expectation.",
            parent=node,
            critical=True,
            score=0.0,
            status="failed"
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Orforglipron FDA review comprehensive analysis task.
    """
    # Initialize evaluator (root is non-critical by framework design)
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
        default_model=model
    )

    # Extract structured analysis from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_orforglipron_analysis(),
        template_class=OrforglipronAnalysisExtraction,
        extraction_name="orforglipron_analysis_extraction"
    )

    # Add a critical wrapper node to reflect rubric root
    analysis_node = evaluator.add_parallel(
        id="Orforglipron_Comprehensive_Analysis",
        desc="Comprehensive, source-supported analysis covering decision date, review designation, timeline comparison, Phase 3 status/enrollment context, delivery method, and timeline changes.",
        parent=root,
        critical=True
    )

    # Build and verify all rubric subtrees
    await verify_current_fda_decision_date(evaluator, analysis_node, extraction)
    await verify_fda_review_designation(evaluator, analysis_node, extraction)
    await verify_review_timeline_comparison(evaluator, analysis_node, extraction)
    await verify_phase3_clinical_trials(evaluator, analysis_node, extraction)
    await verify_drug_delivery_method(evaluator, analysis_node, extraction)
    await verify_timeline_changes_or_delays(evaluator, analysis_node, extraction)

    # Optional: record evaluation context info
    evaluator.add_custom_info(
        info={"as_of": AS_OF_CONTEXT},
        info_type="context",
        info_name="evaluation_context"
    )

    # Return structured result
    return evaluator.get_summary()