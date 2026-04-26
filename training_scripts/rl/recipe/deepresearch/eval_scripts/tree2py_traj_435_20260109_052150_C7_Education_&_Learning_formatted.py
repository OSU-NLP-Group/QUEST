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
TASK_ID = "online_masters_program_eval"
TASK_DESCRIPTION = """
A working professional is seeking an online master's degree program that meets the following requirements: (1) The institution must hold regional accreditation, (2) Tuition must be $700 or less per credit hour, (3) The program must require 30-36 total credit hours, (4) Courses must be offered in an asynchronous format (no fixed class meeting times), (5) The online platform must meet WCAG 2.1 Level AA accessibility standards, (6) Career services must be available to graduate students, (7) Digital library access (databases and journals) must be provided, (8) Faculty must have a stated email response time policy of 24 hours or less, (9) The institution must accept federal financial aid (FAFSA), (10) The program must not require GRE or GMAT scores for admission, (11) The program must offer rolling admissions, (12) The institution must accept transfer credits (minimum 6 credits), (13) A capstone project option must be available (as an alternative to a thesis), and (14) Academic support services (tutoring or writing center) must be provided. Identify one specific online master's degree program that satisfies these criteria. Provide the institution name, the specific program name, and supporting reference URLs for each criterion.
"""

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CriterionBlock(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ProgramExtraction(BaseModel):
    institution_name: Optional[str] = None
    program_name: Optional[str] = None
    program_url: Optional[str] = None

    regional_accreditation: CriterionBlock = Field(default_factory=CriterionBlock)
    tuition_per_credit: CriterionBlock = Field(default_factory=CriterionBlock)
    total_credits: CriterionBlock = Field(default_factory=CriterionBlock)
    asynchronous_format: CriterionBlock = Field(default_factory=CriterionBlock)
    wcag_compliance: CriterionBlock = Field(default_factory=CriterionBlock)
    career_services: CriterionBlock = Field(default_factory=CriterionBlock)
    digital_library: CriterionBlock = Field(default_factory=CriterionBlock)
    faculty_response_time: CriterionBlock = Field(default_factory=CriterionBlock)
    federal_financial_aid: CriterionBlock = Field(default_factory=CriterionBlock)
    no_standardized_tests: CriterionBlock = Field(default_factory=CriterionBlock)
    rolling_admissions: CriterionBlock = Field(default_factory=CriterionBlock)
    transfer_credits: CriterionBlock = Field(default_factory=CriterionBlock)
    capstone_option: CriterionBlock = Field(default_factory=CriterionBlock)
    academic_support: CriterionBlock = Field(default_factory=CriterionBlock)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return """
    Extract exactly one institution and one online master's program identified in the answer, along with supporting reference URLs for each of the listed criteria. If multiple options are mentioned, select the single program the answer ultimately recommends; if still ambiguous, select the first clearly presented program and institution.

    Required fields to extract:
    - institution_name: The institution's official name (string).
    - program_name: The specific online master's program name (string).
    - program_url: The main official page URL for the program, if present (string or null).

    For each criterion below, extract:
    - value: The exact phrase, number, or policy statement as stated in the answer that supports the criterion (string or null).
    - urls: An array of all URLs the answer cites as evidence for this criterion. If the same URL supports multiple criteria, include it in multiple lists. Only include URLs explicitly present in the answer text (plain links or markdown links). Do not invent URLs.

    Criteria keys:
    - regional_accreditation
    - tuition_per_credit
    - total_credits
    - asynchronous_format
    - wcag_compliance
    - career_services
    - digital_library
    - faculty_response_time
    - federal_financial_aid
    - no_standardized_tests
    - rolling_admissions
    - transfer_credits
    - capstone_option
    - academic_support

    Special rules for URL extraction:
    - Extract only valid URLs explicitly present in the answer.
    - If a URL is in markdown format, extract the actual link target.
    - If a URL misses protocol, prepend http://.

    If any field is missing in the answer, set it to null (for strings) or [] (for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Criteria metadata                                                           #
# --------------------------------------------------------------------------- #
def criteria_meta():
    return [
        {
            "key": "regional_accreditation",
            "short": "regional_accreditation",
            "desc": "The institution holds regional accreditation",
            "claim": "The institution holds regional (institutional) accreditation recognized by USDE/CHEA (e.g., HLC, MSCHE, SACSCOC, NECHE, NWCCU, WSCUC).",
            "instruction": "Accept listings on the institution's accreditation page or the accreditor's directory showing the institution is accredited by a regional accreditor."
        },
        {
            "key": "tuition_per_credit",
            "short": "affordable_tuition",
            "desc": "Tuition cost is $700 or less per credit hour",
            "claim": "The program's tuition per credit hour is $700 or less (USD).",
            "instruction": "Look specifically for per-credit tuition. If multiple rates exist, the relevant online program's per-credit rate must be ≤ $700. Ignore fees."
        },
        {
            "key": "total_credits",
            "short": "standard_credit_hours",
            "desc": "The program requires 30–36 total credit hours",
            "claim": "The program requires between 30 and 36 total credit hours (inclusive).",
            "instruction": "Accept ranges or exact totals that fall within 30 to 36 credits inclusive."
        },
        {
            "key": "asynchronous_format",
            "short": "asynchronous_format",
            "desc": "Courses are offered in an asynchronous format (no fixed class meeting times)",
            "claim": "Courses are offered in an asynchronous format with no fixed class meeting times.",
            "instruction": "Look for phrases like 'asynchronous', 'no set meeting times', or 'on your own schedule'; minor synchronous optional sessions are acceptable."
        },
        {
            "key": "wcag_compliance",
            "short": "wcag_compliance",
            "desc": "The online platform meets WCAG 2.1 Level AA accessibility standards",
            "claim": "The institution's online learning platform or accessibility policy states compliance with WCAG 2.1 Level AA.",
            "instruction": "Accept institutional accessibility statements or LMS accessibility conformance statements explicitly mentioning WCAG 2.1 AA."
        },
        {
            "key": "career_services",
            "short": "career_services",
            "desc": "Career services are available to graduate students",
            "claim": "Career services are available to graduate students.",
            "instruction": "Evidence can include dedicated career center pages or language stating services for graduate students."
        },
        {
            "key": "digital_library",
            "short": "digital_library_access",
            "desc": "Digital library access (databases and journals) is provided",
            "claim": "Students have access to digital library resources, including databases and journals.",
            "instruction": "Look for remote/online access to library databases and e‑journals for enrolled students."
        },
        {
            "key": "faculty_response_time",
            "short": "faculty_response_time",
            "desc": "Faculty have a stated email response time policy of 24 hours or less",
            "claim": "Faculty have a stated policy to respond to student emails within 24 hours or less.",
            "instruction": "Accept 'within 24 hours', 'by the next business day', or 'within one business day' language on policy pages or syllabi standards."
        },
        {
            "key": "federal_financial_aid",
            "short": "federal_financial_aid",
            "desc": "The institution accepts federal financial aid (FAFSA)",
            "claim": "The institution participates in federal financial aid (FAFSA/Title IV).",
            "instruction": "Evidence can be the institution's financial aid page referencing FAFSA/Title IV or a FAFSA school code listing."
        },
        {
            "key": "no_standardized_tests",
            "short": "no_standardized_tests",
            "desc": "The program does not require GRE or GMAT scores for admission",
            "claim": "The program does not require GRE or GMAT for admission.",
            "instruction": "Accept 'not required' or 'waived/optional' for GRE/GMAT; failure if required without exception."
        },
        {
            "key": "rolling_admissions",
            "short": "rolling_admissions",
            "desc": "The program offers rolling admissions",
            "claim": "The program offers rolling admissions.",
            "instruction": "Accept 'rolling admissions' or 'applications accepted and reviewed year‑round' language."
        },
        {
            "key": "transfer_credits",
            "short": "transfer_credits_accepted",
            "desc": "The institution accepts transfer credits with a minimum of 6 credits allowed",
            "claim": "The institution accepts transfer credits with a minimum allowance of at least 6 credits.",
            "instruction": "Look for graduate transfer credit policies specifying a minimum accepted number; pass if ≥ 6 credits are allowed."
        },
        {
            "key": "capstone_option",
            "short": "capstone_option",
            "desc": "A capstone project option is available as an alternative to a thesis",
            "claim": "The program offers a capstone project option as an alternative to a thesis.",
            "instruction": "Evidence can include program structure pages or catalogs showing capstone or thesis options."
        },
        {
            "key": "academic_support",
            "short": "academic_support",
            "desc": "Academic support services (tutoring or writing center) are provided",
            "claim": "Academic support services such as tutoring and/or a writing center are provided to graduate students.",
            "instruction": "Accept centralized academic support pages indicating grad student eligibility."
        },
    ]


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_text(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


def _get_block(extracted: ProgramExtraction, key: str) -> CriterionBlock:
    block = getattr(extracted, key, None)
    if isinstance(block, CriterionBlock):
        return block
    return CriterionBlock()


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: ProgramExtraction) -> None:
    # Top-level critical sequential node for this task
    program_root = evaluator.add_sequential(
        id="online_masters_program",
        desc="Identify one specific online master's program that satisfies all stated criteria and provide required supporting information/URLs",
        parent=evaluator.root,
        critical=True
    )

    # 1) Program Identified (critical)
    pi_node = evaluator.add_parallel(
        id="program_identified",
        desc="Provides exactly one specific institution name and one specific online master's program name",
        parent=program_root,
        critical=True
    )

    # Existence of both names
    exists_result = _has_text(extracted.institution_name) and _has_text(extracted.program_name)
    evaluator.add_custom_node(
        result=exists_result,
        id="program_identified_exists",
        desc="Institution name and program name are provided (non-empty)",
        parent=pi_node,
        critical=True
    )

    # Exactly one (not a list of options) judged against the answer text
    singleton_leaf = evaluator.add_leaf(
        id="program_identified_singleton",
        desc="The answer provides exactly one institution and one program (not multiple options)",
        parent=pi_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer provides exactly one specific institution and one specific online master's program; it does not list multiple different programs or institutions as options.",
        node=singleton_leaf,
        additional_instruction="If multiple institutions or programs are enumerated as alternatives, this should be considered incorrect."
    )

    # 2) Evidence Provided (critical): URLs for each criterion
    ev_node = evaluator.add_parallel(
        id="evidence_provided",
        desc="Provides supporting reference URL(s) for each required criterion (and they correspond to the named institution/program)",
        parent=program_root,
        critical=True
    )

    for meta in criteria_meta():
        block = _get_block(extracted, meta["key"])
        urls_provided = bool(block.urls and len(block.urls) > 0)
        evaluator.add_custom_node(
            result=urls_provided,
            id=f"urls_provided_{meta['short']}",
            desc=f"Supporting URL(s) provided for criterion: {meta['desc']}",
            parent=ev_node,
            critical=True
        )

    # 3) Meets All Criteria (critical parallel)
    mac_node = evaluator.add_parallel(
        id="meets_all_criteria",
        desc="The named institution/program satisfies every stated constraint",
        parent=program_root,
        critical=True
    )

    # Create leaves for each criterion and verify them against cited URLs
    batch: List[tuple[str, List[str], Any, Optional[str]]] = []
    for meta in criteria_meta():
        block = _get_block(extracted, meta["key"])
        leaf = evaluator.add_leaf(
            id=f"criterion_{meta['short']}",
            desc=meta["desc"],
            parent=mac_node,
            critical=True
        )

        # Construct claim and additional instruction
        claim = meta["claim"]
        add_ins = meta["instruction"]

        # Use the URLs provided for this criterion; could be empty -> will fail verification
        sources = block.urls if isinstance(block.urls, list) else []

        # Optionally include names in instruction context to help judge alignment
        extra_context = ""
        if _has_text(extracted.institution_name) or _has_text(extracted.program_name):
            extra_context = f" Institution: {extracted.institution_name or ''}. Program: {extracted.program_name or ''}."
        add_instruction_full = f"{add_ins}{extra_context}"

        batch.append((claim, sources, leaf, add_instruction_full))

    # Run batch verifications in parallel (each leaf has auto preconditions from earlier steps)
    await evaluator.batch_verify(batch)


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
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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
    extracted: ProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_selection"
    )

    # Record selected names for convenience in summary
    evaluator.add_custom_info(
        info={
            "institution_name": extracted.institution_name,
            "program_name": extracted.program_name,
            "program_url": extracted.program_url
        },
        info_type="extracted_program",
        info_name="selected_program_overview"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted)

    # Return evaluation summary
    return evaluator.get_summary()