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
TASK_ID = "mooc_masters_4_programs"
TASK_DESCRIPTION = """
Identify four distinct online master's degree programs from major MOOC platforms (Coursera or edX) that meet the following specific criteria:

Program 1: An affordable master's degree program with a total cost under $25,000 USD from a U.S. university partner. The program must be from a regionally accredited institution. Provide the platform name, university name, program URL, cost verification URL, and accreditation confirmation URL.

Program 2: A master's degree program (not a certificate or MicroMasters) in Computer Science, Data Science, or a closely related technology field that does not require GRE or GMAT for admission. The program must be from an accredited institution. Provide the platform name, university name, specific field of study, program URL, admission requirements URL showing no GRE/GMAT requirement, accreditation URL, and degree type confirmation URL.

Program 3: A master's degree program that explicitly offers financial aid or scholarships to students. The program must be from an accredited institution to be eligible for federal financial aid. Provide the platform name, university name, program URL, financial aid/scholarship information URL, and accreditation URL.

Program 4: A master's degree program from a university that is explicitly listed as an official partner on either Coursera's or edX's partners page, and that can be completed in under 24 months (2 years) for full-time students. The program must be from an accredited institution. Provide the platform name, university name, platform partners page URL showing this university, program URL, program duration verification URL, and accreditation URL.

For each program, ensure all four programs are distinct (not the same program counted multiple times). Provide all requested URLs as verification.
"""

US_REGIONAL_ACCREDITORS = [
    "Higher Learning Commission",
    "HLC",
    "Middle States Commission on Higher Education",
    "MSCHE",
    "New England Commission of Higher Education",
    "NECHE",
    "Northwest Commission on Colleges and Universities",
    "NWCCU",
    "Southern Association of Colleges and Schools Commission on Colleges",
    "SACSCOC",
    "WASC Senior College and University Commission",
    "WSCUC",
]
US_RECOGNIZED_HINT = "U.S. Department of Education (USDE) or Council for Higher Education Accreditation (CHEA)"

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ProgramBase(BaseModel):
    platform: Optional[str] = None  # expected "Coursera" or "edX"
    university: Optional[str] = None
    program_name: Optional[str] = None
    program_url: Optional[str] = None


class Program1(ProgramBase):
    cost_total_text: Optional[str] = None
    cost_verification_url: Optional[str] = None
    accreditation_url: Optional[str] = None


class Program2(ProgramBase):
    field_of_study: Optional[str] = None
    admission_requirements_url: Optional[str] = None
    no_gre_gmat_statement: Optional[str] = None
    degree_type_url: Optional[str] = None
    accreditation_url: Optional[str] = None


class Program3(ProgramBase):
    financial_aid_url: Optional[str] = None
    accreditation_url: Optional[str] = None


class Program4(ProgramBase):
    partners_page_url: Optional[str] = None
    duration_text: Optional[str] = None
    duration_verification_url: Optional[str] = None
    accreditation_url: Optional[str] = None


class AllPrograms(BaseModel):
    program1: Optional[Program1] = None
    program2: Optional[Program2] = None
    program3: Optional[Program3] = None
    program4: Optional[Program4] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_programs() -> str:
    return """
    Extract four distinct online master's degree programs (Program 1–4) as they appear in the answer text. For each program, extract only data explicitly present in the answer. Do not invent URLs.

    Shared fields for every program:
    - platform: The platform name, ideally "Coursera" or "edX"
    - university: The university partner name
    - program_name: The degree program name/title
    - program_url: Official program page URL

    Program 1 extra fields:
    - cost_total_text: Any total cost text/value stated in the answer (e.g., "Total tuition $23,000")
    - cost_verification_url: URL that documents or confirms the total program cost
    - accreditation_url: URL confirming that the institution is regionally accredited

    Program 2 extra fields:
    - field_of_study: The field (e.g., "Computer Science", "Data Science", or closely related tech field)
    - admission_requirements_url: URL to admission requirements that shows no GRE/GMAT required
    - no_gre_gmat_statement: A phrase from the answer indicating GRE/GMAT is not required (if present)
    - degree_type_url: URL confirming the credential is a master's degree (not certificate/MicroMasters)
    - accreditation_url: URL confirming institutional accreditation

    Program 3 extra fields:
    - financial_aid_url: URL documenting scholarships/financial aid for this program
    - accreditation_url: URL confirming institutional accreditation

    Program 4 extra fields:
    - partners_page_url: Platform partners page URL that explicitly lists the university
    - duration_text: Any duration info quoted in the answer (e.g., "12–18 months")
    - duration_verification_url: URL that verifies time-to-completion (full-time)
    - accreditation_url: URL confirming institutional accreditation

    If any field is not mentioned in the answer, return null for that field.
    Extract only URLs that are explicitly present in the answer text. If a URL is missing a protocol, prepend http:// as needed.
    """


# --------------------------------------------------------------------------- #
# Verification helper instructions                                            #
# --------------------------------------------------------------------------- #
ONLINE_MASTERS_INSTRUCTIONS = (
    "Verify the page is for an online master's degree program (not a certificate, MicroMasters, or non-degree). "
    "Accept synonyms like 'online MS', 'online MSc', 'online MEng', 'online MBA', 'Master of Science', etc., as long as it clearly indicates a master's degree delivered online."
)

FIELD_TECH_INSTRUCTIONS = (
    "Verify that the degree is in Computer Science, Data Science, or a closely related technology field. "
    "Closely related fields include: AI, Machine Learning, Data Analytics, Business Analytics (if substantially technical), Software Engineering, Cybersecurity, Information Systems, Information Technology, Computer Engineering, etc. "
    "Reject programs that are clearly non-technical or unrelated."
)

NO_GRE_GMAT_INSTRUCTIONS = (
    "Verify that the admission page explicitly states that GRE and GMAT are not required. "
    "Treat 'GRE/GMAT optional', 'not required', or 'waived' as NOT required. If unclear or only states scores may be submitted, treat as not required."
)

ACCREDITED_INSTRUCTIONS = (
    f"Verify the institution is accredited by a recognized accreditor (e.g., {US_RECOGNIZED_HINT}). "
    "Institutional accreditation (regional or national recognized by USDE/CHEA) is acceptable unless the node requires 'regional'."
)

REGIONAL_ACCREDITED_INSTRUCTIONS = (
    "Verify the institution is REGIONALLY accredited by a recognized U.S. regional accrediting body. "
    "Common regional accreditors include: HLC, MSCHE, NECHE, NWCCU, SACSCOC, WSCUC (WASC). "
    "If the page shows one of these, consider it regionally accredited."
)

FINANCIAL_AID_INSTRUCTIONS = (
    "Verify that the program explicitly offers scholarships or financial aid (e.g., 'financial aid', 'scholarships', 'grants', 'fellowships', 'tuition assistance'). "
    "The page must clearly indicate availability; ambiguous statements are insufficient."
)

DURATION_UNDER_24_INSTRUCTIONS = (
    "Verify that the program can be completed in under 24 months (strictly less than 24 months) for full-time students. "
    "If the page says '24 months', '2 years', or 'up to 24 months', that does NOT satisfy 'under 24 months'. "
    "If the page gives a range (e.g., 12–18 months), that satisfies the requirement."
)

PARTNERS_PAGE_INSTRUCTIONS = (
    "Verify that the specified platform's official partners page explicitly lists this university by name."
)


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
def _normalize_url_for_distinctness(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = url.strip().lower()
    while u.endswith("/"):
        u = u[:-1]
    return u


def _platform_ok(platform: Optional[str]) -> bool:
    if not platform:
        return False
    p = platform.strip().lower()
    return p in {"coursera", "edx", "ed.x", "ed x"} or p.replace(".", "").replace(" ", "") == "edx"


def _non_empty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _all_present_and_distinct(urls: List[Optional[str]]) -> bool:
    normed = [_normalize_url_for_distinctness(u) for u in urls]
    if any(u is None for u in normed):
        return False
    return len(set(normed)) == len(normed)


# --------------------------------------------------------------------------- #
# Verification builders for each program                                      #
# --------------------------------------------------------------------------- #
async def build_program_1_nodes(evaluator: Evaluator, parent, p: Program1) -> None:
    node = evaluator.add_parallel(
        id="program_1",
        desc="Program 1: Online master's degree from Coursera/edX; total cost under $25,000; U.S. university partner; regionally accredited; provide required URLs",
        parent=parent,
        critical=False
    )

    # Presence / value checks (critical)
    evaluator.add_custom_node(
        result=_platform_ok(p.platform),
        id="program_1_platform_name",
        desc="Specify platform (Coursera or edX)",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.university),
        id="program_1_university_us",
        desc="Identify the U.S. university partner offering the program",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.program_url),
        id="program_1_program_url",
        desc="Provide the official program URL",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.cost_verification_url),
        id="program_1_cost_verification_url",
        desc="Provide a URL documenting/verifying the total program cost",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.accreditation_url),
        id="program_1_accreditation_url",
        desc="Provide a URL confirming regional accreditation",
        parent=node,
        critical=True
    )

    # Verifications that rely on URLs (critical leaves)
    is_online_node = evaluator.add_leaf(
        id="program_1_is_online_masters",
        desc="Confirm it is an online master's degree program",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is for an online master's degree program (not a certificate or MicroMasters).",
        node=is_online_node,
        sources=p.program_url,
        additional_instruction=ONLINE_MASTERS_INSTRUCTIONS
    )

    cost_under_node = evaluator.add_leaf(
        id="program_1_cost_under_25000",
        desc="Verify total program cost is under $25,000 USD",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The total program cost is less than $25,000 USD.",
        node=cost_under_node,
        sources=[p.cost_verification_url, p.program_url] if p.program_url else p.cost_verification_url,
        additional_instruction="Confirm the total tuition/fees for completing the entire degree program are under $25,000 USD. If ambiguous or per-credit only without a clear total, treat as not under."
    )

    regional_acc_node = evaluator.add_leaf(
        id="program_1_regional_accreditation",
        desc="Confirm the institution is regionally accredited",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The institution is regionally accredited by a recognized U.S. regional accreditor.",
        node=regional_acc_node,
        sources=p.accreditation_url,
        additional_instruction=REGIONAL_ACCREDITED_INSTRUCTIONS
    )


async def build_program_2_nodes(evaluator: Evaluator, parent, p: Program2) -> None:
    node = evaluator.add_parallel(
        id="program_2",
        desc="Program 2: Online master's degree in CS/DS/closely related tech; no GRE/GMAT required; not a certificate/MicroMasters; accredited; provide required URLs",
        parent=parent,
        critical=False
    )

    # Presence / value checks (critical)
    evaluator.add_custom_node(
        result=_platform_ok(p.platform),
        id="program_2_platform_name",
        desc="Specify platform (Coursera or edX)",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.university),
        id="program_2_university",
        desc="Identify the university partner offering the program",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.program_url),
        id="program_2_program_url",
        desc="Provide the official program URL",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.admission_requirements_url),
        id="program_2_admission_requirements_url",
        desc="Provide a URL showing admission requirements and indicating no GRE/GMAT requirement",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.degree_type_url),
        id="program_2_degree_type_url",
        desc="Provide a URL confirming the credential is a master's degree (degree type confirmation)",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.accreditation_url),
        id="program_2_accreditation_url",
        desc="Provide a URL confirming accreditation",
        parent=node,
        critical=True
    )

    # Verifications (critical leaves)
    field_node = evaluator.add_leaf(
        id="program_2_field_tech",
        desc="Confirm the field is Computer Science, Data Science, or a closely related technology field",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This degree is in Computer Science, Data Science, or a closely related technology field.",
        node=field_node,
        sources=p.program_url,
        additional_instruction=FIELD_TECH_INSTRUCTIONS
    )

    no_gre_node = evaluator.add_leaf(
        id="program_2_no_gre_gmat",
        desc="Verify GRE/GMAT is not required for admission",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This program does not require GRE or GMAT for admission.",
        node=no_gre_node,
        sources=p.admission_requirements_url,
        additional_instruction=NO_GRE_GMAT_INSTRUCTIONS
    )

    is_full_masters_node = evaluator.add_leaf(
        id="program_2_is_full_masters",
        desc="Confirm it is a full master's degree program (not a certificate or MicroMasters)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This credential is a master's degree (e.g., MS/MSc/MEng/MBA), not a certificate or MicroMasters.",
        node=is_full_masters_node,
        sources=p.degree_type_url,
        additional_instruction="Check that the page explicitly identifies the credential as a master's degree and not a certificate/MicroMasters."
    )

    acc_node = evaluator.add_leaf(
        id="program_2_accredited",
        desc="Confirm the institution is accredited",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The institution is accredited by a recognized accrediting body.",
        node=acc_node,
        sources=p.accreditation_url,
        additional_instruction=ACCREDITED_INSTRUCTIONS
    )


async def build_program_3_nodes(evaluator: Evaluator, parent, p: Program3) -> None:
    node = evaluator.add_parallel(
        id="program_3",
        desc="Program 3: Online master's degree that explicitly offers financial aid or scholarships; accredited (federal aid eligibility); provide required URLs",
        parent=parent,
        critical=False
    )

    # Presence / value checks (critical)
    evaluator.add_custom_node(
        result=_platform_ok(p.platform),
        id="program_3_platform_name",
        desc="Specify platform (Coursera or edX)",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.university),
        id="program_3_university",
        desc="Identify the university partner offering the program",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.program_url),
        id="program_3_program_url",
        desc="Provide the official program URL",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.financial_aid_url),
        id="program_3_financial_aid_url",
        desc="Provide a URL documenting financial aid/scholarship information",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.accreditation_url),
        id="program_3_accreditation_url",
        desc="Provide a URL confirming accreditation",
        parent=node,
        critical=True
    )

    # Verifications (critical leaves)
    is_online_node = evaluator.add_leaf(
        id="program_3_is_online_masters",
        desc="Confirm it is an online master's degree program",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is for an online master's degree program (not a certificate or MicroMasters).",
        node=is_online_node,
        sources=p.program_url,
        additional_instruction=ONLINE_MASTERS_INSTRUCTIONS
    )

    aid_node = evaluator.add_leaf(
        id="program_3_financial_aid_or_scholarships",
        desc="Confirm the program explicitly offers financial aid or scholarships",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This program explicitly offers scholarships or financial aid to students.",
        node=aid_node,
        sources=p.financial_aid_url,
        additional_instruction=FINANCIAL_AID_INSTRUCTIONS
    )

    acc_node = evaluator.add_leaf(
        id="program_3_accredited",
        desc="Confirm the institution is accredited (required for federal aid eligibility as stated in the prompt)",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The institution is accredited by a recognized accrediting body.",
        node=acc_node,
        sources=p.accreditation_url,
        additional_instruction=ACCREDITED_INSTRUCTIONS
    )


async def build_program_4_nodes(evaluator: Evaluator, parent, p: Program4) -> None:
    node = evaluator.add_parallel(
        id="program_4",
        desc="Program 4: Online master's degree from a university listed on Coursera/edX official partners page; completable in under 24 months full-time; accredited; provide required URLs",
        parent=parent,
        critical=False
    )

    # Presence / value checks (critical)
    evaluator.add_custom_node(
        result=_platform_ok(p.platform),
        id="program_4_platform_name",
        desc="Specify platform (Coursera or edX)",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.university),
        id="program_4_university",
        desc="Identify the university partner offering the program",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.partners_page_url),
        id="program_4_partners_page_url",
        desc="Provide the platform partners page URL showing the university",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.program_url),
        id="program_4_program_url",
        desc="Provide the official program URL",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.duration_verification_url),
        id="program_4_duration_verification_url",
        desc="Provide a URL verifying the program duration/time-to-complete",
        parent=node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_non_empty(p.accreditation_url),
        id="program_4_accreditation_url",
        desc="Provide a URL confirming accreditation",
        parent=node,
        critical=True
    )

    # Verifications (critical leaves)
    partner_node = evaluator.add_leaf(
        id="program_4_partners_page_lists_university",
        desc="Confirm the university is explicitly listed on the platform's official partners page",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The partners page lists the university '{p.university or ''}'.",
        node=partner_node,
        sources=p.partners_page_url,
        additional_instruction=PARTNERS_PAGE_INSTRUCTIONS
    )

    is_online_node = evaluator.add_leaf(
        id="program_4_is_online_masters",
        desc="Confirm it is an online master's degree program",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is for an online master's degree program (not a certificate or MicroMasters).",
        node=is_online_node,
        sources=p.program_url,
        additional_instruction=ONLINE_MASTERS_INSTRUCTIONS
    )

    duration_node = evaluator.add_leaf(
        id="program_4_under_24_months_full_time",
        desc="Verify the program can be completed in under 24 months (2 years) for full-time students",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This program can be completed in under 24 months for full-time students.",
        node=duration_node,
        sources=p.duration_verification_url,
        additional_instruction=DURATION_UNDER_24_INSTRUCTIONS
    )

    acc_node = evaluator.add_leaf(
        id="program_4_accredited",
        desc="Confirm the institution is accredited",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The institution is accredited by a recognized accrediting body.",
        node=acc_node,
        sources=p.accreditation_url,
        additional_instruction=ACCREDITED_INSTRUCTIONS
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
    Evaluate an answer for the four-program MOOC master's task using the Mind2Web2 framework.
    """
    # Initialize evaluator
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

    # Extract structured info
    extracted: AllPrograms = await evaluator.extract(
        prompt=prompt_extract_programs(),
        template_class=AllPrograms,
        extraction_name="programs_extraction"
    )

    # Add optional info for context
    evaluator.add_custom_info(
        info={
            "note": "This evaluation checks 4 distinct MOOC master's programs across specified criteria and verifies claims against provided URLs."
        },
        info_type="evaluation_note"
    )

    # Global compulsory checks container (critical)
    global_checks = evaluator.add_parallel(
        id="global_checks",
        desc="Global compulsory checks (affect overall validity)",
        parent=root,
        critical=True
    )

    # Distinctness check (critical leaf under global_checks)
    urls_for_distinctness = [
        extracted.program1.program_url if extracted.program1 else None,
        extracted.program2.program_url if extracted.program2 else None,
        extracted.program3.program_url if extracted.program3 else None,
        extracted.program4.program_url if extracted.program4 else None,
    ]
    distinct_ok = _all_present_and_distinct(urls_for_distinctness)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="distinctness_check",
        desc="Confirm the four identified programs are all distinct (no duplicates)",
        parent=global_checks,
        critical=True
    )

    # Build per-program verification subtrees
    await build_program_1_nodes(evaluator, root, extracted.program1 or Program1())
    await build_program_2_nodes(evaluator, root, extracted.program2 or Program2())
    await build_program_3_nodes(evaluator, root, extracted.program3 or Program3())
    await build_program_4_nodes(evaluator, root, extracted.program4 or Program4())

    # Return structured result
    return evaluator.get_summary()