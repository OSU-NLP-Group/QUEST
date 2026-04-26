import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #

TASK_ID = "southeast_public_research_university_fall2026"
TASK_DESCRIPTION = (
    "Identify a public research university in the southeastern United States that meets ALL of the following requirements "
    "for Fall 2026 freshman applicants:\n\n"
    "Institution Classification:\n"
    "- Must be a public (state-funded) university\n"
    "- Must be classified as a research university\n"
    "- Must be located in one of these southeastern states: Georgia, Florida, Alabama, Tennessee, North Carolina, "
    "South Carolina, Virginia, Kentucky, Mississippi, Louisiana, Arkansas, or West Virginia\n\n"
    "Admission Requirements for Fall 2026:\n"
    "- Must require SAT or ACT test scores for freshman admissions\n"
    "- Must review core academic course GPA as part of admission criteria\n"
    "- Must consider the rigor of high school curriculum in admissions review\n\n"
    "Application Process:\n"
    "- Must offer an Early Action application option for Fall 2026\n"
    "- Early Action deadline must be in October or November 2025\n"
    "- Regular Decision deadline must be in January 2026 or later\n\n"
    "Academic Programs:\n"
    "- Must offer an undergraduate honors program or honors college\n"
    "- Must have a published policy for granting credit for AP exam scores\n"
    "- Must provide study abroad program opportunities for undergraduates\n\n"
    "Student Support:\n"
    "- Must have a published priority deadline for FAFSA submission that is earlier than June 30, 2026\n"
    "- Must maintain articulation agreements or transfer pathways with in-state community colleges\n"
    "- Must define full-time undergraduate enrollment as 12 or more credit hours per semester\n\n"
    "Provide the name of one university that satisfies all these requirements, along with supporting evidence for each criterion."
)

SOUTHEASTERN_STATES = [
    "Georgia", "Florida", "Alabama", "Tennessee", "North Carolina",
    "South Carolina", "Virginia", "Kentucky", "Mississippi",
    "Louisiana", "Arkansas", "West Virginia"
]

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #

class CriterionEvidence(BaseModel):
    urls: List[str] = Field(default_factory=list)
    note: Optional[str] = None  # Optional free-form note captured from the answer (if any)


class LocationEvidence(BaseModel):
    state: Optional[str] = None
    urls: List[str] = Field(default_factory=list)
    note: Optional[str] = None


class InstitutionEvidence(BaseModel):
    public_university: Optional[CriterionEvidence] = None
    research_classification: Optional[CriterionEvidence] = None
    location: Optional[LocationEvidence] = None


class AdmissionEvidence(BaseModel):
    test_required: Optional[CriterionEvidence] = None
    gpa_consideration: Optional[CriterionEvidence] = None
    rigor_evaluation: Optional[CriterionEvidence] = None


class ApplicationProcessEvidence(BaseModel):
    early_action_available: Optional[CriterionEvidence] = None
    early_deadline_fall: Optional[CriterionEvidence] = None
    regular_deadline_winter: Optional[CriterionEvidence] = None


class AcademicProgramsEvidence(BaseModel):
    honors_program: Optional[CriterionEvidence] = None
    ap_credit_accepted: Optional[CriterionEvidence] = None
    study_abroad_options: Optional[CriterionEvidence] = None


class StudentSupportEvidence(BaseModel):
    fafsa_priority_deadline: Optional[CriterionEvidence] = None
    transfer_pathways: Optional[CriterionEvidence] = None
    full_time_standard: Optional[CriterionEvidence] = None


class UniversityEvidence(BaseModel):
    university_name: Optional[str] = None
    institution: Optional[InstitutionEvidence] = None
    admissions: Optional[AdmissionEvidence] = None
    application: Optional[ApplicationProcessEvidence] = None
    programs: Optional[AcademicProgramsEvidence] = None
    support: Optional[StudentSupportEvidence] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #

def prompt_extract_university_evidence() -> str:
    return f"""
Extract the university name and, for each requirement below, the URLs explicitly cited in the answer as evidence. 
Only extract URLs that are explicitly present in the answer. If a criterion has multiple URLs, include up to 5. 
If a criterion is not supported by any URL in the answer, return an empty list for that criterion.

Return JSON in this schema:

- university_name: string or null

- institution:
  - public_university: {{ urls: string[], note?: string }}
  - research_classification: {{ urls: string[], note?: string }}
  - location: {{ state: string|null, urls: string[], note?: string }}

- admissions:
  - test_required: {{ urls: string[], note?: string }}
  - gpa_consideration: {{ urls: string[], note?: string }}
  - rigor_evaluation: {{ urls: string[], note?: string }}

- application:
  - early_action_available: {{ urls: string[], note?: string }}
  - early_deadline_fall: {{ urls: string[], note?: string }}
  - regular_deadline_winter: {{ urls: string[], note?: string }}

- programs:
  - honors_program: {{ urls: string[], note?: string }}
  - ap_credit_accepted: {{ urls: string[], note?: string }}
  - study_abroad_options: {{ urls: string[], note?: string }}

- support:
  - fafsa_priority_deadline: {{ urls: string[], note?: string }}
  - transfer_pathways: {{ urls: string[], note?: string }}
  - full_time_standard: {{ urls: string[], note?: string }}

Special notes:
- For institution.location.state, extract the state name if explicitly mentioned (e.g., "Georgia"). If not present, set to null.
- Southeastern states to consider valid: {', '.join(SOUTHEASTERN_STATES)}.
- Do not invent or infer any URLs. Only include URLs that appear in the answer text (including in markdown links).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #

def _safe_urls(crit: Optional[CriterionEvidence] | Optional[LocationEvidence]) -> List[str]:
    if not crit:
        return []
    # LocationEvidence has 'urls', CriterionEvidence has 'urls'
    urls = getattr(crit, "urls", None)
    return urls or []


def _has_any_url(urls: List[str]) -> bool:
    return bool(urls and len([u for u in urls if isinstance(u, str) and u.strip()]) > 0)


async def _add_criterion_verification(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    claim: str,
    urls: List[str],
    add_ins: str,
) -> None:
    """
    Add a source-presence gate (critical) and a verification leaf (critical).
    """
    # Gate: Sources present
    evaluator.add_custom_node(
        result=_has_any_url(urls),
        id=f"{node_id}_sources_present",
        desc=f"Sources are provided to support: {node_desc}",
        parent=parent_node,
        critical=True
    )

    # Verification leaf
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,  # verify by URLs
        additional_instruction=add_ins
    )


# --------------------------------------------------------------------------- #
# Verification builders per rubric section                                    #
# --------------------------------------------------------------------------- #

async def build_institution_nodes(
    evaluator: Evaluator,
    parent_node,
    uni_name: Optional[str],
    inst: Optional[InstitutionEvidence],
):
    node = evaluator.add_parallel(
        id="institution_type",
        desc="Institutional classification requirements",
        parent=parent_node,
        critical=True
    )

    # Public university
    public_urls = _safe_urls(inst.public_university if inst else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "public_university",
        "The institution is a public (state-funded) university",
        claim=f"{uni_name or 'The university'} is a public (state-funded) university.",
        urls=public_urls,
        add_ins="Look for explicit phrases like 'public university', 'state university', 'state-funded', or membership in a state university system."
    )

    # Research classification
    research_urls = _safe_urls(inst.research_classification if inst else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "research_classification",
        "The institution is classified as a research university",
        claim=f"{uni_name or 'The university'} is classified as a research university (Carnegie R1 or R2 counts).",
        urls=research_urls,
        add_ins="Accept evidence such as 'Carnegie R1' or 'Carnegie R2' designations, or explicit 'research university' wording on official or authoritative pages."
    )

    # Southeastern location
    loc_state = (inst.location.state if (inst and inst.location) else None) if inst else None
    location_urls = _safe_urls(inst.location if inst else None)
    states_joined = ", ".join(SOUTHEASTERN_STATES)
    state_for_claim = loc_state or "a southeastern U.S. state"
    await _add_criterion_verification(
        evaluator,
        node,
        "southeastern_location",
        "Located in a southeastern U.S. state (Georgia, Florida, Alabama, Tennessee, North Carolina, South Carolina, Virginia, Kentucky, Mississippi, Louisiana, Arkansas, or West Virginia)",
        claim=f"{uni_name or 'The university'} is located in {state_for_claim}, which is among the allowed southeastern states.",
        urls=location_urls,
        add_ins=f"Verify that the university is located in one of these states: {states_joined}. "
                f"If the page shows the campus state, check it is in that list."
    )


async def build_admissions_nodes(
    evaluator: Evaluator,
    parent_node,
    uni_name: Optional[str],
    adm: Optional[AdmissionEvidence],
):
    node = evaluator.add_parallel(
        id="admission_requirements",
        desc="Fall 2026 freshman admission policies",
        parent=parent_node,
        critical=True
    )

    # Test required (SAT/ACT) for Fall 2026
    test_urls = _safe_urls(adm.test_required if adm else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "test_required",
        "Requires SAT or ACT scores for Fall 2026 freshman admissions",
        claim=f"For Fall 2026 freshman admissions, {uni_name or 'the university'} requires SAT or ACT test scores (not test-optional).",
        urls=test_urls,
        add_ins="Confirm the policy specifically for the Fall 2026 cycle; phrases like 'tests required' or 'submission of SAT/ACT required' should appear. "
                "If the policy is test-optional for 2026, this claim is not supported."
    )

    # GPA consideration (core academic course GPA)
    gpa_urls = _safe_urls(adm.gpa_consideration if adm else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "gpa_consideration",
        "Reviews core academic course GPA as part of admission criteria",
        claim=f"The admissions review at {uni_name or 'the university'} considers GPA in core academic courses.",
        urls=gpa_urls,
        add_ins="Accept terms like 'core GPA', 'recalculated academic GPA (core subjects)', or explicit mention that GPA in core courses factors into decisions."
    )

    # Rigor evaluation (HS curriculum strength)
    rigor_urls = _safe_urls(adm.rigor_evaluation if adm else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "rigor_evaluation",
        "Considers rigor of high school curriculum in admissions review",
        claim=f"The admissions review at {uni_name or 'the university'} evaluates the rigor/strength of the high school curriculum.",
        urls=rigor_urls,
        add_ins="Look for mentions such as 'curriculum rigor', 'strength of schedule', 'AP/IB/AICE/Honors/dual enrollment considered', or similar wording."
    )


async def build_application_nodes(
    evaluator: Evaluator,
    parent_node,
    uni_name: Optional[str],
    app: Optional[ApplicationProcessEvidence],
):
    node = evaluator.add_parallel(
        id="application_process",
        desc="Application timeline and options",
        parent=parent_node,
        critical=True
    )

    # Early Action available for Fall 2026
    ea_urls = _safe_urls(app.early_action_available if app else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "early_action_available",
        "Offers Early Action application option for Fall 2026",
        claim=f"{uni_name or 'The university'} offers an Early Action application option for the Fall 2026 intake.",
        urls=ea_urls,
        add_ins="Confirm Early Action is offered for first-year Fall 2026 applicants (term synonyms: 'EA')."
    )

    # Early Action deadline in Oct/Nov 2025
    ead_urls = _safe_urls(app.early_deadline_fall if app else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "early_deadline_fall",
        "Early Action deadline is in October or November 2025",
        claim="For the Fall 2026 cycle, the Early Action deadline occurs in either October 2025 or November 2025.",
        urls=ead_urls,
        add_ins="Verify that the posted Early Action deadline for Fall 2026 is on any date in Oct 2025 or Nov 2025 (e.g., Oct 15, Nov 1). "
                "Ensure the deadline is for the Fall 2026 cycle."
    )

    # Regular Decision deadline in Jan 2026 or later
    rd_urls = _safe_urls(app.regular_deadline_winter if app else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "regular_deadline_winter",
        "Regular Decision deadline is in January 2026 or later",
        claim="For the Fall 2026 cycle, the Regular Decision deadline is in January 2026 or later (e.g., February or March 2026).",
        urls=rd_urls,
        add_ins="Verify the Regular Decision deadline on the admissions page for Fall 2026. Any date in Jan 2026 or later qualifies."
    )


async def build_programs_nodes(
    evaluator: Evaluator,
    parent_node,
    uni_name: Optional[str],
    progs: Optional[AcademicProgramsEvidence],
):
    node = evaluator.add_parallel(
        id="academic_programs",
        desc="Available academic programs and policies",
        parent=parent_node,
        critical=True
    )

    # Honors program/college
    honors_urls = _safe_urls(progs.honors_program if progs else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "honors_program",
        "Offers an undergraduate honors program or honors college",
        claim=f"{uni_name or 'The university'} offers an undergraduate honors program or an honors college.",
        urls=honors_urls,
        add_ins="Accept 'Honors College' or 'Honors Program' on official sites describing undergraduate honors."
    )

    # AP credit policy
    ap_urls = _safe_urls(progs.ap_credit_accepted if progs else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "ap_credit_accepted",
        "Has a published policy for granting credit for AP exam scores",
        claim=f"{uni_name or 'The university'} publishes an AP credit policy granting credit based on AP exam scores.",
        urls=ap_urls,
        add_ins="Look for official credit-by-exam, AP credit charts/tables/policies. The policy must be published on official or authoritative pages."
    )

    # Study abroad
    abroad_urls = _safe_urls(progs.study_abroad_options if progs else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "study_abroad_options",
        "Provides study abroad program opportunities for undergraduates",
        claim=f"{uni_name or 'The university'} provides study abroad opportunities for undergraduate students.",
        urls=abroad_urls,
        add_ins="Accept evidence from the university's study abroad/global education office showing undergrad participation opportunities."
    )


async def build_support_nodes(
    evaluator: Evaluator,
    parent_node,
    uni_name: Optional[str],
    support: Optional[StudentSupportEvidence],
):
    node = evaluator.add_parallel(
        id="student_support",
        desc="Financial aid and enrollment policies",
        parent=parent_node,
        critical=True
    )

    # FAFSA priority deadline earlier than June 30, 2026
    fafsa_urls = _safe_urls(support.fafsa_priority_deadline if support else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "fafsa_priority_deadline",
        "Has a published priority deadline for FAFSA submission that is earlier than June 30, 2026",
        claim="There is a published FAFSA priority deadline earlier than June 30, 2026 for Fall 2026 entrants (2026-27 aid year).",
        urls=fafsa_urls,
        add_ins="Look for 'priority FAFSA deadline' dates on the financial aid/admissions pages. Accept any date before 2026-06-30."
    )

    # Transfer pathways / articulation agreements with in-state community colleges
    transfer_urls = _safe_urls(support.transfer_pathways if support else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "transfer_pathways",
        "Maintains articulation agreements or transfer pathways with in-state community colleges",
        claim=f"{uni_name or 'The university'} maintains articulation agreements or transfer pathways with in-state community colleges.",
        urls=transfer_urls,
        add_ins="Accept explicit mentions of 'articulation agreement', 'transfer pathway/map/guide' specifically with in-state community colleges."
    )

    # Full-time definition: 12+ credit hours
    fulltime_urls = _safe_urls(support.full_time_standard if support else None)
    await _add_criterion_verification(
        evaluator,
        node,
        "full_time_standard",
        "Defines full-time undergraduate enrollment as 12 or more credit hours per semester",
        claim="Full-time undergraduate status is defined as at least 12 credit hours per semester.",
        urls=fulltime_urls,
        add_ins="Check catalog, bursar, registrar, or financial aid definitions. Exact phrasing like '12 or more credit hours' should appear."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #

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
    """
    Evaluate an answer for the southeastern public research university Fall 2026 requirements task.
    Builds a critical verification tree where all criteria must be satisfied (with URL support) for a full score.
    """
    # 1) Initialize evaluator and root
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

    # Add a top-level critical aggregator to enforce "ALL requirements" behavior
    overall = evaluator.add_parallel(
        id="overall_requirements",
        desc="Identify a public research university in the southeastern United States that meets all specified admission, program, and support requirements for Fall 2026 freshman applicants",
        parent=root,
        critical=True
    )

    # 2) Extract structured evidence from the answer
    extracted: UniversityEvidence = await evaluator.extract(
        prompt=prompt_extract_university_evidence(),
        template_class=UniversityEvidence,
        extraction_name="university_evidence"
    )

    # Record contextual info
    evaluator.add_custom_info(
        info={
            "southeastern_states_allowed": SOUTHEASTERN_STATES,
            "note": "University must be public, research, located in one of these states, and meet all Fall 2026 requirements."
        },
        info_type="task_context"
    )

    uni_name = extracted.university_name

    # 3) Build and verify each rubric category as critical parallel groups
    await build_institution_nodes(evaluator, overall, uni_name, extracted.institution)
    await build_admissions_nodes(evaluator, overall, uni_name, extracted.admissions)
    await build_application_nodes(evaluator, overall, uni_name, extracted.application)
    await build_programs_nodes(evaluator, overall, uni_name, extracted.programs)
    await build_support_nodes(evaluator, overall, uni_name, extracted.support)

    # 4) Return evaluator summary
    return evaluator.get_summary()