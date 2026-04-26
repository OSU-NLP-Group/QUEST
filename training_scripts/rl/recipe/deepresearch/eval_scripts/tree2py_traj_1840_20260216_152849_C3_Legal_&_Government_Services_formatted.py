import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "federal_holidays_authority"
TASK_DESCRIPTION = (
    "Under the United States legal framework, which branch of government has the exclusive constitutional authority "
    "to establish permanent federal holidays, what specific statute codifies the list of these holidays, and how does "
    "this authority differ from the President's power to grant temporary workplace closures for federal employees? "
    "In your answer, you must: (1) identify the governmental branch with permanent holiday-creation authority, "
    "(2) cite the specific U.S. Code title and section number that establishes federal holidays, "
    "(3) explain the legal distinction between permanent federal holidays and temporary administrative leave granted by the President, "
    "(4) reference the executive order or statutory provisions that govern how presidential administrative leave grants are implemented "
    "for pay and leave purposes, and (5) describe the scope and temporal limitations of administrative leave granted through presidential executive order."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HolidaysAuthorityExtraction(BaseModel):
    # (1) Authority branch
    authority_branch: Optional[str] = None
    authority_sources: List[str] = Field(default_factory=list)

    # (2) Statute citation for federal holidays
    statute_citation: Optional[str] = None
    statute_sources: List[str] = Field(default_factory=list)

    # (3) Legal distinction explanation (statutory holidays vs presidential administrative leave)
    distinction_explanation: Optional[str] = None
    distinction_sources: List[str] = Field(default_factory=list)

    # (4) Implementation framework references (must include EO 11582 and Title 5 pay/leave authorities)
    eo_numbers: List[str] = Field(default_factory=list)  # e.g., ["11582", "XXXX"]
    implementation_statutes: List[str] = Field(default_factory=list)  # e.g., ["5 U.S.C. 6103", "5 U.S.C. 6101", "5 CFR 630"]
    implementation_sources: List[str] = Field(default_factory=list)

    # (5) Administrative leave: scope and temporal limits (split into 3 subpoints)
    exec_branch_only: Optional[str] = None
    exec_branch_sources: List[str] = Field(default_factory=list)

    temporary_revocable: Optional[str] = None
    temporary_revocable_sources: List[str] = Field(default_factory=list)

    nonbinding_statement: Optional[str] = None
    nonbinding_sources: List[str] = Field(default_factory=list)

    # Additional required constraints in rubric
    holiday_count_number: Optional[str] = None
    holiday_count_sources: List[str] = Field(default_factory=list)

    applicability_statement: Optional[str] = None
    applicability_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_holidays_authority() -> str:
    return """
    Extract from the answer the specific elements below and any URLs the answer cites to support them.
    IMPORTANT:
    - Extract exactly what the answer states (do not infer or add new information).
    - For each element, include all URLs explicitly provided in the answer that support that element. If the answer provides a single shared sources section, duplicate those URLs into each relevant sources list.
    - Accept URLs in plain form or markdown link form; extract actual URLs only.

    Fields to extract:
    1) authority_branch: The branch identified as having the exclusive constitutional authority to establish permanent federal holidays (e.g., "Congress", "Legislative Branch").
       authority_sources: URLs cited to support that claim.

    2) statute_citation: The specific U.S. Code citation for the list of federal holidays (e.g., "5 U.S.C. § 6103"; accept formatting variations like "5 USC 6103").
       statute_sources: URLs cited to support that citation and what it codifies.

    3) distinction_explanation: The explanation of the legal distinction that permanent federal holidays are statutory (created by Congress) while presidential administrative leave is not a statutory holiday and does not alter statutory deadlines/obligations.
       distinction_sources: URLs cited to support this legal distinction.

    4) Implementation framework references for presidential administrative leave:
       eo_numbers: A list of Executive Order numbers mentioned (strings only; e.g., "11582" for Executive Order 11582).
       implementation_statutes: A list of Title 5 U.S. Code or related pay/leave authorities mentioned (strings; do not require exact section numbers).
       implementation_sources: URLs cited that cover how presidential administrative leave is implemented for pay/leave purposes (e.g., EO texts, OPM policy pages, US Code pages, or CFR pages).

    5) Scope and temporal limitations of administrative leave granted via presidential executive order (split into three subpoints). For each, extract the specific statement the answer makes and the supporting URLs:
       exec_branch_only: Statement indicating it applies to executive-branch federal employees.
       exec_branch_sources: URLs supporting this scope.
       temporary_revocable: Statement indicating it is temporary (date/specific instance) and can be revoked/changed; not permanent or guaranteed to recur.
       temporary_revocable_sources: URLs supporting this temporal limitation.
       nonbinding_statement: Statement indicating it does not bind the private sector, state governments, or future administrations.
       nonbinding_sources: URLs supporting this statement.

    6) holiday_count_number: The exact count asserted for the number of permanent federal holidays "as of 2025" (as a string, e.g., "12").
       holiday_count_sources: URLs supporting that count (e.g., statutory text, official OPM schedule/guidance).

    7) applicability_statement: A statement indicating that congressionally-created federal holidays apply to federal employees and the District of Columbia and do not bind individual states (which set their own holidays).
       applicability_sources: URLs supporting this applicability limitation.

    Return a JSON object with the above fields exactly, using null for any missing text fields and empty lists for sources not provided in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _has_value(text: Optional[str]) -> bool:
    return bool(text and str(text).strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def _eo_11582_mentioned(eo_numbers: List[str]) -> bool:
    if not eo_numbers:
        return False
    s = " ".join(eo_numbers).lower()
    return "11582" in s


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_branch_authority(evaluator: Evaluator, parent, data: HolidaysAuthorityExtraction) -> None:
    node = evaluator.add_sequential(
        id="branch_authority",
        desc="Identifies Congress (legislative branch) as the exclusive constitutional authority to establish permanent federal holidays",
        parent=parent,
        critical=True,
    )

    # Existence: branch identified and supporting sources present
    evaluator.add_custom_node(
        result=_has_value(data.authority_branch) and _has_urls(data.authority_sources),
        id="branch_authority_exists",
        desc="Authority branch is identified and sources are provided",
        parent=node,
        critical=True,
    )

    # Match: Does the identified branch correspond to Congress (legislative branch)?
    match_leaf = evaluator.add_leaf(
        id="branch_authority_match",
        desc="The identified branch matches 'Congress' (the legislative branch)",
        parent=node,
        critical=True,
    )
    claim_match = f"The identified branch '{data.authority_branch}' refers to Congress (the legislative branch)."
    await evaluator.verify(
        claim=claim_match,
        node=match_leaf,
        additional_instruction="Treat 'Congress', 'U.S. Congress', 'the legislative branch', or close equivalents as a match. Be lenient to casing and minor wording differences.",
    )

    # Source support: Congress has exclusive authority to create permanent federal holidays by statute
    src_leaf = evaluator.add_leaf(
        id="branch_authority_source",
        desc="Sources support that Congress creates permanent federal holidays by statute",
        parent=node,
        critical=True,
    )
    claim_src = "Under U.S. law, permanent federal legal public holidays are created by Congress (the legislative branch) by statute."
    await evaluator.verify(
        claim=claim_src,
        node=src_leaf,
        sources=data.authority_sources,
        additional_instruction="The source should explicitly state or clearly imply that federal holidays are established by Congress via statute. Accept references to Title 5 U.S. Code or official OPM resources.",
    )


async def verify_statute_citation(evaluator: Evaluator, parent, data: HolidaysAuthorityExtraction) -> None:
    node = evaluator.add_sequential(
        id="holiday_statute_citation",
        desc="Cites the specific U.S. Code title and section that codifies federal holidays (5 U.S.C. § 6103)",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_value(data.statute_citation) and _has_urls(data.statute_sources),
        id="holiday_statute_citation_exists",
        desc="Statute citation is provided with sources",
        parent=node,
        critical=True,
    )

    match_leaf = evaluator.add_leaf(
        id="holiday_statute_citation_match",
        desc="The cited statute matches 5 U.S.C. § 6103 (allowing formatting variants)",
        parent=node,
        critical=True,
    )
    citation_str = data.statute_citation or ""
    claim_match = f"The cited statute '{citation_str}' refers to 5 U.S.C. § 6103 (the statute that codifies the list of federal holidays)."
    await evaluator.verify(
        claim=claim_match,
        node=match_leaf,
        additional_instruction="Allow formatting variants such as '5 USC 6103', '5 U.S.C. 6103', or '5 U.S.C. §6103'. Focus on whether the citation clearly denotes 5 U.S.C. § 6103.",
    )

    src_leaf = evaluator.add_leaf(
        id="holiday_statute_citation_source",
        desc="Sources support that 5 U.S.C. § 6103 codifies the list of federal holidays",
        parent=node,
        critical=True,
    )
    claim_src = "5 U.S.C. § 6103 establishes (codifies) the list of legal public holidays for federal employees."
    await evaluator.verify(
        claim=claim_src,
        node=src_leaf,
        sources=data.statute_sources,
        additional_instruction="Page should display or quote 5 U.S.C. § 6103 or an official explanation stating that § 6103 codifies the list of federal holidays.",
    )


async def verify_legal_distinction(evaluator: Evaluator, parent, data: HolidaysAuthorityExtraction) -> None:
    node = evaluator.add_sequential(
        id="legal_distinction_explanation",
        desc="Explains the legal distinction between statutory holidays and presidential administrative leave",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_value(data.distinction_explanation) and _has_urls(data.distinction_sources),
        id="legal_distinction_exists",
        desc="Distinction explanation is provided with supporting sources",
        parent=node,
        critical=True,
    )

    src_leaf = evaluator.add_leaf(
        id="legal_distinction_source",
        desc="Sources support the distinction: statutory holidays vs. non-statutory presidential administrative leave",
        parent=node,
        critical=True,
    )
    claim_src = (
        "Permanent federal holidays are statutory and created by Congress (e.g., under 5 U.S.C. § 6103), "
        "whereas a President's grant of administrative leave (e.g., closing executive departments/agencies for a specific date) "
        "is not a statutory holiday and does not create statutory obligations or alter statutory deadlines."
    )
    await evaluator.verify(
        claim=claim_src,
        node=src_leaf,
        sources=data.distinction_sources,
        additional_instruction="Accept authoritative OPM guidance, U.S. Code/CFR pages, or executive documents that clearly distinguish statutory holidays from presidentially granted administrative leave/office closures.",
    )


async def verify_implementation_framework(evaluator: Evaluator, parent, data: HolidaysAuthorityExtraction) -> None:
    node = evaluator.add_parallel(
        id="implementation_framework_reference",
        desc="References the executive order and/or statutory provisions for implementing presidential administrative leave",
        parent=parent,
        critical=True,
    )

    # EO 11582 sub-node
    eo_node = evaluator.add_sequential(
        id="eo_11582_reference",
        desc="Executive Order 11582 is referenced for holiday observance/implementation context",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_eo_11582_mentioned(data.eo_numbers) and _has_urls(data.implementation_sources),
        id="eo_11582_exists",
        desc="Executive Order 11582 is mentioned and sources for implementation are provided",
        parent=eo_node,
        critical=True,
    )

    eo_src_leaf = evaluator.add_leaf(
        id="eo_11582_source",
        desc="Sources include or support Executive Order 11582 (Observance of Holidays by Government Agencies)",
        parent=eo_node,
        critical=True,
    )
    claim_eo = (
        "The provided source is Executive Order 11582 (Observance of Holidays by Government Agencies) or an authoritative page "
        "that quotes/explains EO 11582 in the context of federal holiday observance and related pay/leave implementation."
    )
    await evaluator.verify(
        claim=claim_eo,
        node=eo_src_leaf,
        sources=data.implementation_sources,
        additional_instruction="Accept the official EO text or an authoritative OPM/US Code resource that directly references EO 11582.",
    )

    # Title 5 pay/leave authorities sub-node
    payleave_node = evaluator.add_sequential(
        id="pay_leave_authorities_reference",
        desc="Title 5 pay/leave authorities are referenced for implementing presidential administrative leave",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=(
            isinstance(data.implementation_statutes, list)
            and any("5" in s and ("U.S.C" in s or "USC" in s or "Title 5" in s) for s in data.implementation_statutes)
            and _has_urls(data.implementation_sources)
        ),
        id="pay_leave_authorities_exists",
        desc="Title 5 pay/leave authorities are referenced and sources are provided",
        parent=payleave_node,
        critical=True,
    )

    payleave_src_leaf = evaluator.add_leaf(
        id="pay_leave_authorities_source",
        desc="Sources support that Title 5 U.S. Code pay/leave provisions govern implementation for closures/holidays",
        parent=payleave_node,
        critical=True,
    )
    claim_pay = (
        "The provided source addresses Title 5 U.S. Code (or related OPM/CFR policies) governing federal employee pay and leave "
        "when holidays are observed or when the President closes executive agencies or grants excused absence/administrative leave."
    )
    await evaluator.verify(
        claim=claim_pay,
        node=payleave_src_leaf,
        sources=data.implementation_sources,
        additional_instruction="Accept U.S. Code (Title 5) sections, OPM policy pages, or relevant CFR parts (e.g., 5 CFR 630) that describe pay/leave implementation for holidays or excused absences.",
    )


async def verify_scope_temporal_limits(evaluator: Evaluator, parent, data: HolidaysAuthorityExtraction) -> None:
    node = evaluator.add_parallel(
        id="administrative_leave_scope_temporal_limits",
        desc="Describes the scope and temporal limitations of presidential administrative leave",
        parent=parent,
        critical=True,
    )

    # Scope: executive branch only
    scope_node = evaluator.add_sequential(
        id="scope_executive_branch_only",
        desc="States that presidential administrative leave applies only to executive-branch federal employees",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_value(data.exec_branch_only) and _has_urls(data.exec_branch_sources),
        id="scope_exec_only_exists",
        desc="Scope statement present and sources provided",
        parent=scope_node,
        critical=True,
    )
    scope_leaf = evaluator.add_leaf(
        id="scope_exec_only_source",
        desc="Sources support that administrative leave via presidential order applies to executive-branch federal employees",
        parent=scope_node,
        critical=True,
    )
    claim_scope = (
        "Presidential executive orders granting administrative leave or closing federal offices apply to Executive Branch "
        "federal employees (e.g., 'executive departments and agencies')."
    )
    await evaluator.verify(
        claim=claim_scope,
        node=scope_leaf,
        sources=data.exec_branch_sources,
        additional_instruction="Accept OPM or official documents specifying applicability to executive departments/agencies or executive branch employees.",
    )

    # Temporal limits: temporary and revocable
    temp_node = evaluator.add_sequential(
        id="temporal_limits_temporary_revocable",
        desc="States that administrative leave is temporary and revocable",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_value(data.temporary_revocable) and _has_urls(data.temporary_revocable_sources),
        id="temp_revocable_exists",
        desc="Temporal limitation statement present and sources provided",
        parent=temp_node,
        critical=True,
    )
    temp_leaf = evaluator.add_leaf(
        id="temp_revocable_source",
        desc="Sources support that administrative leave via presidential order is temporary (date-specific) and revocable/not permanent",
        parent=temp_node,
        critical=True,
    )
    claim_temp = (
        "Administrative leave granted via presidential executive order is temporary for a specific date or occasion and can be changed or revoked; "
        "it does not create a permanent holiday that recurs automatically."
    )
    await evaluator.verify(
        claim=claim_temp,
        node=temp_leaf,
        sources=data.temporary_revocable_sources,
        additional_instruction="Accept executive orders/memoranda or OPM guidance that describe closures or excused absence for a particular day or event.",
    )

    # Nonbinding on others and future
    nb_node = evaluator.add_sequential(
        id="nonbinding_on_others_and_future",
        desc="States that administrative leave via executive order does not bind the private sector, states, or future administrations",
        parent=node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=_has_value(data.nonbinding_statement) and _has_urls(data.nonbinding_sources),
        id="nonbinding_exists",
        desc="Nonbinding statement present and sources provided",
        parent=nb_node,
        critical=True,
    )
    nb_leaf = evaluator.add_leaf(
        id="nonbinding_source",
        desc="Sources support that such executive orders do not bind private sector, state governments, or future administrations",
        parent=nb_node,
        critical=True,
    )
    claim_nb = (
        "A presidential executive order granting administrative leave or closing executive departments does not bind private employers or state governments, "
        "and does not legally bind future administrations to grant the same leave."
    )
    await evaluator.verify(
        claim=claim_nb,
        node=nb_leaf,
        sources=data.nonbinding_sources,
        additional_instruction="Accept authoritative sources indicating that federal executive orders govern the federal executive branch and are not generally binding on non-federal actors or future administrations unless reissued.",
    )


async def verify_holiday_count_constraint(evaluator: Evaluator, parent, data: HolidaysAuthorityExtraction) -> None:
    node = evaluator.add_sequential(
        id="holiday_count_constraint",
        desc="States that, as of 2025, there are exactly 12 permanent federal holidays established under 5 U.S.C. § 6103",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_value(data.holiday_count_number) and ("12" in (data.holiday_count_number or "")) and _has_urls(data.holiday_count_sources),
        id="holiday_count_exists",
        desc="Holiday count '12' is stated with supporting sources",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="holiday_count_source",
        desc="Sources support that there are 12 permanent federal holidays as of 2025",
        parent=node,
        critical=True,
    )
    claim = (
        "As of 2025, there are 12 permanent federal holidays established under 5 U.S.C. § 6103."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.holiday_count_sources,
        additional_instruction=(
            "Accept sources that enumerate the 12 federal holidays or that state the number is 12. "
            "OPM's official holiday schedule or an authoritative legal source is acceptable."
        ),
    )


async def verify_applicability_constraint(evaluator: Evaluator, parent, data: HolidaysAuthorityExtraction) -> None:
    node = evaluator.add_sequential(
        id="federal_holiday_applicability_constraint",
        desc="Describes that federal holidays apply to federal employees and the District of Columbia; states have their own holidays",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_has_value(data.applicability_statement) and _has_urls(data.applicability_sources),
        id="applicability_exists",
        desc="Applicability statement provided with sources",
        parent=node,
        critical=True,
    )

    leaf = evaluator.add_leaf(
        id="applicability_source",
        desc="Sources support applicability to federal employees/DC and not binding on individual states",
        parent=node,
        critical=True,
    )
    claim = (
        "Federal holidays established by Congress (5 U.S.C. § 6103) apply to federal employees and the District of Columbia; "
        "they do not bind individual states, which set their own state holidays."
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=data.applicability_sources,
        additional_instruction="Accept U.S. Code or OPM sources stating applicability to federal employees and D.C., and clarifying that states determine their own holidays.",
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
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # We'll add a critical aggregator under root
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
        prompt=prompt_extract_holidays_authority(),
        template_class=HolidaysAuthorityExtraction,
        extraction_name="holidays_authority_extraction",
    )

    # Build a critical aggregator to mimic a critical root (framework root is non-critical)
    requirements = evaluator.add_parallel(
        id="all_requirements",
        desc="Answer satisfies all required elements and explicit constraints about permanent federal holidays vs. presidential administrative leave",
        parent=root,
        critical=True,
    )

    # Build verification subtrees (all critical)
    await verify_branch_authority(evaluator, requirements, extraction)
    await verify_statute_citation(evaluator, requirements, extraction)
    await verify_legal_distinction(evaluator, requirements, extraction)
    await verify_implementation_framework(evaluator, requirements, extraction)
    await verify_scope_temporal_limits(evaluator, requirements, extraction)
    await verify_holiday_count_constraint(evaluator, requirements, extraction)
    await verify_applicability_constraint(evaluator, requirements, extraction)

    # Return summary
    return evaluator.get_summary()