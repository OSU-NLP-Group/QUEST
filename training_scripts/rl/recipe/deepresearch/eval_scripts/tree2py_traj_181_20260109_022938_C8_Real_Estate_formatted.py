import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "ca_dre_first_renewal_ce"
TASK_DESCRIPTION = "California real estate salesperson completing first-time license renewal continuing education requirements"


# ----------------------------- Data Models --------------------------------- #
class CourseReq(BaseModel):
    hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CoreCourses(BaseModel):
    ethics: CourseReq = Field(default_factory=CourseReq)
    agency: CourseReq = Field(default_factory=CourseReq)
    trust_fund_handling: CourseReq = Field(default_factory=CourseReq)
    fair_housing: CourseReq = Field(default_factory=CourseReq)
    risk_management: CourseReq = Field(default_factory=CourseReq)


class ConsumerProtection(BaseModel):
    hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ConsumerService(BaseModel):
    min_hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Electives(BaseModel):
    hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AdminItem(BaseModel):
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class Administration(BaseModel):
    dre_course_approval: AdminItem = Field(default_factory=AdminItem)
    completion_timeframe: AdminItem = Field(default_factory=AdminItem)
    approved_providers: AdminItem = Field(default_factory=AdminItem)
    exam_passage: AdminItem = Field(default_factory=AdminItem)


class TotalHours(BaseModel):
    total_hours: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CERequirementsExtraction(BaseModel):
    core: CoreCourses = Field(default_factory=CoreCourses)
    consumer_protection: ConsumerProtection = Field(default_factory=ConsumerProtection)
    consumer_service: ConsumerService = Field(default_factory=ConsumerService)
    electives: Electives = Field(default_factory=Electives)
    administration: Administration = Field(default_factory=Administration)
    total: TotalHours = Field(default_factory=TotalHours)


# --------------------------- Extraction Prompt ------------------------------ #
def prompt_extract_ce_requirements() -> str:
    return """
    Extract the California real estate salesperson first-time license renewal continuing education requirements as presented in the answer. Focus on the first renewal for a salesperson who obtained their original license on March 15, 2021 and is renewing on March 15, 2025.

    Return a JSON object with the following structure and fields, extracting ONLY what is explicitly stated in the answer. If a field is missing, return null for that field or an empty list for sources.

    {
      "core": {
        "ethics": { "hours": string or null, "sources": [urls] },
        "agency": { "hours": string or null, "sources": [urls] },
        "trust_fund_handling": { "hours": string or null, "sources": [urls] },
        "fair_housing": { "hours": string or null, "sources": [urls] },
        "risk_management": { "hours": string or null, "sources": [urls] }
      },
      "consumer_protection": {
        "hours": string or null,
        "sources": [urls]
      },
      "consumer_service": {
        "min_hours": string or null,
        "sources": [urls]
      },
      "electives": {
        "hours": string or null,
        "sources": [urls]
      },
      "administration": {
        "dre_course_approval": { "statement": string or null, "sources": [urls] },
        "completion_timeframe": { "statement": string or null, "sources": [urls] },
        "approved_providers": { "statement": string or null, "sources": [urls] },
        "exam_passage": { "statement": string or null, "sources": [urls] }
      },
      "total": {
        "total_hours": string or null,
        "sources": [urls]
      }
    }

    Requirements for sources:
    - Extract only URLs explicitly present in the answer (plain URLs or in markdown links).
    - If the answer references a site without an actual URL, do not invent it; return an empty list.
    - Do not include duplicate URLs; keep only unique and valid URLs.

    Notes:
    - Keep all "hours" fields as strings exactly as written in the answer (e.g., "3", "3 hours", "three hours").
    - If the answer groups the 5 core courses as "each 3 hours", populate each course’s hours accordingly.
    - For consumer service, if the answer mentions a minimum requirement (e.g., "at least 3 hours"), record that in "min_hours".
    """


# --------------------------- Helper Mappings -------------------------------- #
COURSE_LABELS = {
    "ethics": "Ethics",
    "agency": "Agency",
    "trust_fund_handling": "Trust Fund Handling",
    "fair_housing": "Fair Housing",
    "risk_management": "Risk Management",
}

CORE_NODE_IDS = {
    "ethics": "ethics_course",
    "agency": "agency_course",
    "trust_fund_handling": "trust_fund_course",
    "fair_housing": "fair_housing_course",
    "risk_management": "risk_management_course",
}


def _normalize_hours_for_claim(hours: Optional[str]) -> str:
    if not hours or not hours.strip():
        return "UNKNOWN"
    return hours.strip()


# ------------------------- Verification Functions --------------------------- #
async def build_core_courses_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CERequirementsExtraction
) -> None:
    core_node = evaluator.add_parallel(
        id="required_core_courses",
        desc="Completion of all five required 3-hour core courses",
        parent=parent_node,
        critical=True
    )

    claims_batch = []
    for key in ["ethics", "agency", "trust_fund_handling", "fair_housing", "risk_management"]:
        item: CourseReq = getattr(extracted.core, key)
        hours_text = _normalize_hours_for_claim(item.hours)
        leaf = evaluator.add_leaf(
            id=CORE_NODE_IDS[key],
            desc=f"Completed exactly 3 clock hours in {COURSE_LABELS[key]} course",
            parent=core_node,
            critical=True
        )
        claim = (
            f"For the California real estate salesperson first-time license renewal, "
            f"the required {COURSE_LABELS[key]} continuing education course must be exactly {hours_text} clock hours."
        )
        claims_batch.append((claim, item.sources, leaf,
                             "Verify the hour requirement for the first-time salesperson renewal (not subsequent renewals). "
                             "Accept minor wording variants like '3-hour' vs '3 hours' if equivalent."))

    await evaluator.batch_verify(claims_batch)


async def build_consumer_requirements_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CERequirementsExtraction
) -> None:
    cons_node = evaluator.add_parallel(
        id="consumer_protection_requirement",
        desc="Completion of consumer protection hours and any specific minimum requirement",
        parent=parent_node,
        critical=True
    )

    # Consumer Protection total hours
    cp_hours_text = _normalize_hours_for_claim(extracted.consumer_protection.hours)
    cp_leaf = evaluator.add_leaf(
        id="consumer_protection_hours",
        desc="Completed exactly 18 clock hours in Consumer Protection courses",
        parent=cons_node,
        critical=True
    )
    cp_claim = (
        f"For the California real estate salesperson first-time license renewal, "
        f"the Consumer Protection coursework requirement is exactly {cp_hours_text} clock hours."
    )
    await evaluator.verify(
        claim=cp_claim,
        node=cp_leaf,
        sources=extracted.consumer_protection.sources,
        additional_instruction="Focus on the first-time renewal requirement for a salesperson. "
                               "Confirm the Consumer Protection hours as stated."
    )

    # Consumer Service minimum
    cs_min_text = _normalize_hours_for_claim(extracted.consumer_service.min_hours)
    cs_leaf = evaluator.add_leaf(
        id="consumer_service_minimum",
        desc="Completed minimum 3 clock hours specifically in Consumer Service course",
        parent=cons_node,
        critical=True
    )
    cs_claim = (
        f"For the first-time salesperson renewal in California, at least {cs_min_text} clock hours must be in Consumer Service coursework."
    )
    await evaluator.verify(
        claim=cs_claim,
        node=cs_leaf,
        sources=extracted.consumer_service.sources,
        additional_instruction="Verify whether a minimum Consumer Service hour requirement applies to the first-time renewal."
    )


async def build_elective_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CERequirementsExtraction
) -> None:
    elect_node = evaluator.add_parallel(
        id="elective_requirement",
        desc="Completion of elective course hours",
        parent=parent_node,
        critical=True
    )

    elective_text = _normalize_hours_for_claim(extracted.electives.hours)
    elect_leaf = evaluator.add_leaf(
        id="elective_hours",
        desc="Completed exactly 12 clock hours in elective courses",
        parent=elect_node,
        critical=True
    )
    elect_claim = (
        f"For the California real estate salesperson first-time license renewal, the elective coursework requirement is exactly {elective_text} clock hours."
    )
    await evaluator.verify(
        claim=elect_claim,
        node=elect_leaf,
        sources=extracted.electives.sources,
        additional_instruction="Confirm the elective hours required specifically for the first-time salesperson renewal."
    )


async def build_admin_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: CERequirementsExtraction
) -> None:
    admin_node = evaluator.add_parallel(
        id="administrative_compliance",
        desc="Met all administrative and procedural requirements for continuing education",
        parent=parent_node,
        critical=True
    )

    # DRE course approval
    dre_leaf = evaluator.add_leaf(
        id="dre_course_approval",
        desc="All courses are approved by California Department of Real Estate (DRE)",
        parent=admin_node,
        critical=True
    )
    dre_claim = "All continuing education courses must be approved by the California Department of Real Estate (DRE)."
    await evaluator.verify(
        claim=dre_claim,
        node=dre_leaf,
        sources=extracted.administration.dre_course_approval.sources,
        additional_instruction="Verify that DRE approval of CE courses is a requirement."
    )

    # Completion timeframe (within 4 years immediately preceding renewal)
    timeframe_leaf = evaluator.add_leaf(
        id="completion_timeframe",
        desc="All courses completed within the 4 years immediately preceding license renewal",
        parent=admin_node,
        critical=True
    )
    timeframe_claim = "All continuing education courses must be completed within the four years immediately preceding the license renewal date."
    await evaluator.verify(
        claim=timeframe_claim,
        node=timeframe_leaf,
        sources=extracted.administration.completion_timeframe.sources,
        additional_instruction="Confirm the completion window requirement for CE (4 years before renewal)."
    )

    # Approved providers
    providers_leaf = evaluator.add_leaf(
        id="approved_providers",
        desc="All courses taken from DRE-approved education providers",
        parent=admin_node,
        critical=True
    )
    providers_claim = "All continuing education must be taken from education providers approved by the California Department of Real Estate."
    await evaluator.verify(
        claim=providers_claim,
        node=providers_leaf,
        sources=extracted.administration.approved_providers.sources,
        additional_instruction="Verify that CE must be completed with DRE-approved education providers."
    )

    # Exam passage minimum 70%
    exam_leaf = evaluator.add_leaf(
        id="exam_passage",
        desc="Passed all course examinations with minimum score of 70%",
        parent=admin_node,
        critical=True
    )
    exam_claim = "Licensees must pass all continuing education course examinations with a minimum score of 70%."
    await evaluator.verify(
        claim=exam_claim,
        node=exam_leaf,
        sources=extracted.administration.exam_passage.sources,
        additional_instruction="Confirm that the minimum passing score for CE course exams is 70%."
    )


async def build_total_hours_check(
    evaluator: Evaluator,
    parent_node,
    extracted: CERequirementsExtraction
) -> None:
    total_node = evaluator.add_parallel(
        id="total_hour_verification",
        desc="Total continuing education hours meet the exact requirement",
        parent=parent_node,
        critical=True
    )

    total_text = _normalize_hours_for_claim(extracted.total.total_hours)
    total_leaf = evaluator.add_leaf(
        id="total_45_hours",
        desc="Completed exactly 45 total clock hours of continuing education",
        parent=total_node,
        critical=True
    )
    total_claim = (
        f"The total required continuing education hours for the California salesperson first-time license renewal is exactly {total_text} hours."
    )
    await evaluator.verify(
        claim=total_claim,
        node=total_leaf,
        sources=extracted.total.sources,
        additional_instruction="Confirm the total CE hours required for the first-time salesperson renewal (commonly 45 hours)."
    )


# ------------------------------ Main Entry ---------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_ce_requirements(),
        template_class=CERequirementsExtraction,
        extraction_name="ce_requirements_extraction"
    )

    # Add context info (not used for scoring)
    evaluator.add_custom_info(
        info={
            "license_issue_date": "2021-03-15",
            "license_expiration_date": "2025-03-15",
            "renewal_type": "first-time salesperson renewal in California"
        },
        info_type="context",
        info_name="renewal_context"
    )

    # Build verification tree according to rubric
    # Note: Root created by initialize is non-critical; we mark all top-level nodes as critical
    # to reflect rubric's strict gating.
    await build_core_courses_checks(evaluator, root, extracted)
    await build_consumer_requirements_checks(evaluator, root, extracted)
    await build_elective_checks(evaluator, root, extracted)
    await build_admin_checks(evaluator, root, extracted)
    await build_total_hours_check(evaluator, root, extracted)

    return evaluator.get_summary()