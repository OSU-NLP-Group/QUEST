import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode


TASK_ID = "online_masters_affordable_flexible"
TASK_DESCRIPTION = """I am looking for an affordable and flexible online master's degree program to advance my career. Find one fully online master's degree program offered by a U.S. university that meets all of the following requirements:

1. Regional Accreditation: The institution must be regionally accredited by an accrediting agency recognized by the U.S. Department of Education or CHEA (Council for Higher Education Accreditation).

2. Affordable Tuition: The total tuition cost for the entire master's program must be under $15,000.

3. No Standardized Test Requirement: The program must not require GRE or GMAT scores for admission. This must be a complete waiver for all applicants, not conditional based on GPA or work experience.

4. Asynchronous Format: The program must be delivered in a fully asynchronous online format with no required synchronous class meetings or set class times.

5. Reasonable Completion Timeline: The program must be completable in 24 months or less when attending full-time.

6. Flexible Start Dates: The program must offer at least 3 start dates per year, or operate on rolling admissions.

7. Minimal Admission Requirements: The admission requirements must include only a bachelor's degree from an accredited institution, with no specific undergraduate major required, no professional work experience required, and no minimum undergraduate GPA requirement above 2.5.

For the program you identify, please provide:
- The name of the university
- The specific master's program title
- A direct URL to the program's official webpage
- Verification of the institution's regional accreditation (accrediting body name and verification URL)
- The total program tuition cost
- Evidence that GRE/GMAT is not required for any applicant
- Evidence of the asynchronous delivery format
- The program completion timeline for full-time students
- Information about available start dates or rolling admissions policy
- A summary of the admission requirements

All information must be current, accurate, and verifiable through official university sources.
"""


# ---------------------------
# Extraction Models
# ---------------------------
class ProgramCoreExtraction(BaseModel):
    university_name: Optional[str] = None
    program_title: Optional[str] = None
    program_url: Optional[str] = None


class AccreditationExtraction(BaseModel):
    accreditor_name: Optional[str] = None
    accred_verification_urls: List[str] = Field(default_factory=list)


class TuitionExtraction(BaseModel):
    total_program_tuition: Optional[str] = None
    tuition_source_urls: List[str] = Field(default_factory=list)


class TestPolicyExtraction(BaseModel):
    gre_gmat_policy_text: Optional[str] = None
    gre_gmat_source_urls: List[str] = Field(default_factory=list)


class DeliveryExtraction(BaseModel):
    asynchronous_text: Optional[str] = None
    delivery_source_urls: List[str] = Field(default_factory=list)


class TimelineExtraction(BaseModel):
    completion_timeline_text: Optional[str] = None
    timeline_source_urls: List[str] = Field(default_factory=list)


class StartDatesExtraction(BaseModel):
    start_dates_text: Optional[str] = None
    start_dates_source_urls: List[str] = Field(default_factory=list)


class AdmissionsExtraction(BaseModel):
    admissions_summary_text: Optional[str] = None
    admissions_source_urls: List[str] = Field(default_factory=list)


# ---------------------------
# Extraction Prompts
# ---------------------------
def prompt_extract_core() -> str:
    return """
    Extract the single program identified in the answer. Return:
    - university_name: Name of the university (U.S. institution).
    - program_title: The exact master's degree program title (e.g., "Master of Science in X").
    - program_url: A direct URL to the official program webpage on the university website (.edu or official subdomain).
    If any field is missing, return null for it. Use only what is explicitly present in the answer.
    """


def prompt_extract_accreditation() -> str:
    return """
    Extract the institution's regional accreditation information from the answer. Return:
    - accreditor_name: The regional accreditor name (e.g., HLC, MSCHE, SACSCOC, NECHE, NWCCU, WSCUC).
    - accred_verification_urls: One or more URLs cited that verify accreditation. Prefer official accreditor listings, CHEA, US Dept of Education listings, or the university's accreditation page.
    If any item is missing in the answer, leave it null or empty.
    """


def prompt_extract_tuition() -> str:
    return """
    Extract the total tuition cost for the entire master's program as stated in the answer (string is fine). Also extract:
    - tuition_source_urls: All official university URLs cited in the answer that support the tuition figure or allow calculation (tuition page, fee schedule, credit-hour requirement, etc.).
    If not provided, leave fields empty or null.
    """


def prompt_extract_test_policy() -> str:
    return """
    Extract the standardized test policy (GRE/GMAT) from the answer. Return:
    - gre_gmat_policy_text: A statement as given in the answer indicating GRE/GMAT requirement or waiver.
    - gre_gmat_source_urls: Official admissions/program URLs cited that support the policy.
    """


def prompt_extract_delivery() -> str:
    return """
    Extract the delivery format from the answer. Return:
    - asynchronous_text: A statement indicating the program is fully asynchronous (e.g., "no required live sessions" or "no set class times").
    - delivery_source_urls: Official program/university URLs cited that support asynchronous delivery.
    """


def prompt_extract_timeline() -> str:
    return """
    Extract the full-time completion timeline from the answer. Return:
    - completion_timeline_text: The timeline as stated (e.g., "can be completed in 12-18 months").
    - timeline_source_urls: Official program/university URLs cited that support the timeline.
    """


def prompt_extract_start_dates() -> str:
    return """
    Extract start dates or admissions cadence from the answer. Return:
    - start_dates_text: A statement indicating at least three starts per year or rolling admissions.
    - start_dates_source_urls: Official program/university URLs cited that support start dates or rolling admissions.
    """


def prompt_extract_admissions() -> str:
    return """
    Extract the admissions requirements summary from the answer. Return:
    - admissions_summary_text: The summarized requirements as given.
    - admissions_source_urls: Official admissions/program URLs cited that support the requirements.
    """


# ---------------------------
# Helper
# ---------------------------
def combine_sources(*args: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for lst in args:
        if lst:
            out.extend([u for u in lst if isinstance(u, str) and u.strip() != ""])
    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for u in out:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


# ---------------------------
# Verification Subtrees
# ---------------------------
async def build_required_response_fields(
    evaluator: Evaluator,
    parent: VerificationNode,
    core: ProgramCoreExtraction,
) -> Dict[str, VerificationNode]:
    req_node = evaluator.add_parallel(
        id="required_response_fields",
        desc="Response includes all explicitly requested identifying information and links for the selected program",
        parent=parent,
        critical=True,
    )

    # University name provided
    uni_present = evaluator.add_custom_node(
        result=bool(core.university_name and core.university_name.strip()),
        id="provide_university_name",
        desc="Provides the name of the university",
        parent=req_node,
        critical=True,
    )

    # Program title provided
    prog_present = evaluator.add_custom_node(
        result=bool(core.program_title and core.program_title.strip()),
        id="provide_program_title",
        desc="Provides the specific master's program title",
        parent=req_node,
        critical=True,
    )

    # Program URL provided (direct, official)
    url_present = evaluator.add_custom_node(
        result=bool(core.program_url and core.program_url.strip() and ("http://" in core.program_url or "https://" in core.program_url)),
        id="provide_official_program_url",
        desc="Provides a direct URL to the program's official webpage",
        parent=req_node,
        critical=True,
    )

    return {
        "uni_present": uni_present,
        "prog_present": prog_present,
        "url_present": url_present,
        "req_node": req_node,
    }


async def build_program_eligibility_and_evidence(
    evaluator: Evaluator,
    parent: VerificationNode,
    core: ProgramCoreExtraction,
    accreditation: AccreditationExtraction,
    tuition: TuitionExtraction,
    tests: TestPolicyExtraction,
    delivery: DeliveryExtraction,
    timeline: TimelineExtraction,
    start_dates: StartDatesExtraction,
    admissions: AdmissionsExtraction,
    prereq_nodes: Dict[str, VerificationNode],
) -> None:
    elig_node = evaluator.add_parallel(
        id="program_eligibility_and_evidence",
        desc="Chosen program and institution satisfy all stated eligibility constraints, with verifiable supporting URLs",
        parent=parent,
        critical=True,
    )

    # Subnode: US university + master's + fully online (not hybrid)
    us_online_node = evaluator.add_parallel(
        id="us_university_and_fully_online",
        desc="Program is a master's degree offered by a U.S. university and is fully online (not hybrid), supported by an official program/university webpage URL",
        parent=elig_node,
        critical=True,
    )

    # Local source presence gate
    official_url_present_local = evaluator.add_custom_node(
        result=bool(core.program_url and core.program_url.strip()),
        id="official_program_url_present_local",
        desc="Official program webpage URL is present to verify program properties",
        parent=us_online_node,
        critical=True,
    )

    # Verify US university
    is_us_university_leaf = evaluator.add_leaf(
        id="is_us_university",
        desc="Institution is a U.S. university",
        parent=us_online_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The institution that offers the program is a U.S. university located in the United States.",
        node=is_us_university_leaf,
        sources=core.program_url,
        additional_instruction="Use the official program/university page to confirm U.S. context (e.g., .edu domain, U.S. address/state, or explicit mention). If the page does not indicate U.S. location, consider the claim unsupported.",
        extra_prerequisites=[prereq_nodes["url_present"], official_url_present_local],
    )

    # Verify master's-level program
    masters_leaf = evaluator.add_leaf(
        id="is_masters_program",
        desc="Program is a master's degree program",
        parent=us_online_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The selected program is a master's degree (graduate-level) program.",
        node=masters_leaf,
        sources=core.program_url,
        additional_instruction="Look for 'Master of ...' or abbreviations like MS/MSc/MA/MBA/etc. on the official program page.",
        extra_prerequisites=[prereq_nodes["url_present"], official_url_present_local],
    )

    # Verify fully online (not hybrid)
    fully_online_leaf = evaluator.add_leaf(
        id="fully_online_not_hybrid",
        desc="Program is fully online (not hybrid)",
        parent=us_online_node,
        critical=True,
    )
    await evaluator.verify(
        claim="This master's program is delivered fully online (no on-campus component required).",
        node=fully_online_leaf,
        sources=[core.program_url] if core.program_url else None,
        additional_instruction="Confirm '100% online' or equivalent phrasing on official sources; if only 'hybrid' or unspecified, consider unsupported.",
        extra_prerequisites=[prereq_nodes["url_present"], official_url_present_local],
    )

    # Accreditation check
    accred_sources_present = evaluator.add_custom_node(
        result=bool(accreditation.accred_verification_urls),
        id="accreditation_sources_present",
        desc="Accreditation verification URL(s) provided",
        parent=elig_node,
        critical=True,
    )
    accreditation_leaf = evaluator.add_leaf(
        id="regional_accreditation",
        desc="Institution is regionally accredited by a recognized agency; accreditor name and verification URL provided",
        parent=elig_node,
        critical=True,
    )
    acc_claim = f"The university is regionally accredited by {accreditation.accreditor_name or 'a recognized regional accreditor'} and the accreditor is recognized by the U.S. Department of Education or CHEA."
    await evaluator.verify(
        claim=acc_claim,
        node=accreditation_leaf,
        sources=accreditation.accred_verification_urls,
        additional_instruction="Verify using the provided accreditor/CHEA/ED listing or official university accreditation page. Recognized regional accreditors include HLC, MSCHE, SACSCOC, NECHE, NWCCU, WSCUC.",
        extra_prerequisites=[accred_sources_present],
    )

    # Tuition under $15,000
    tuition_sources_present = evaluator.add_custom_node(
        result=bool(tuition.tuition_source_urls),
        id="tuition_sources_present",
        desc="Tuition source URL(s) provided",
        parent=elig_node,
        critical=True,
    )
    tuition_leaf = evaluator.add_leaf(
        id="tuition_under_15000",
        desc="Total tuition is under $15,000 with official support",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The total tuition for the entire master's program is under $15,000.",
        node=tuition_leaf,
        sources=tuition.tuition_source_urls,
        additional_instruction="Use official tuition/fee pages and program credit-hour requirements if needed. If calculation is required, consider standard multiplications. If evidence is insufficient or ambiguous, consider the claim unsupported.",
        extra_prerequisites=[tuition_sources_present],
    )

    # No GRE/GMAT unconditional
    gre_sources_present = evaluator.add_custom_node(
        result=bool(tests.gre_gmat_source_urls),
        id="gre_gmat_sources_present",
        desc="Admissions test policy source URL(s) provided",
        parent=elig_node,
        critical=True,
    )
    gre_leaf = evaluator.add_leaf(
        id="no_gre_gmat_unconditional",
        desc="Admission does not require GRE/GMAT for any applicant (not conditional)",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program does not require GRE or GMAT scores for any applicants; there are no conditional requirements based on GPA or work experience.",
        node=gre_leaf,
        sources=tests.gre_gmat_source_urls,
        additional_instruction="If the page indicates waivers only under certain conditions (e.g., specific GPA, experience), consider the requirement not fully waived.",
        extra_prerequisites=[gre_sources_present],
    )

    # Fully asynchronous
    async_sources_present = evaluator.add_custom_node(
        result=bool(delivery.delivery_source_urls),
        id="asynchronous_sources_present",
        desc="Delivery format source URL(s) provided",
        parent=elig_node,
        critical=True,
    )
    async_leaf = evaluator.add_leaf(
        id="fully_asynchronous",
        desc="Program is fully asynchronous with no required synchronous meetings",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program is fully asynchronous with no required synchronous class meetings or set class times.",
        node=async_leaf,
        sources=delivery.delivery_source_urls,
        additional_instruction="Look for explicit mention of 'asynchronous' or 'no set meeting times/no live sessions required'.",
        extra_prerequisites=[async_sources_present],
    )

    # Completion within 24 months
    timeline_sources_present = evaluator.add_custom_node(
        result=bool(timeline.timeline_source_urls),
        id="timeline_sources_present",
        desc="Timeline source URL(s) provided",
        parent=elig_node,
        critical=True,
    )
    timeline_leaf = evaluator.add_leaf(
        id="completion_within_24_months",
        desc="Full-time completion timeline is 24 months or less",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Full-time students can complete this master's program in 24 months or less.",
        node=timeline_leaf,
        sources=timeline.timeline_source_urls,
        additional_instruction="Confirm the stated completion timeline; if only ranges above 24 months are given, consider unsupported.",
        extra_prerequisites=[timeline_sources_present],
    )

    # Start dates or rolling admissions
    starts_sources_present = evaluator.add_custom_node(
        result=bool(start_dates.start_dates_source_urls),
        id="start_dates_sources_present",
        desc="Start dates/rolling admissions source URL(s) provided",
        parent=elig_node,
        critical=True,
    )
    starts_leaf = evaluator.add_leaf(
        id="start_dates_or_rolling",
        desc="Program offers at least 3 start dates per year or rolling admissions",
        parent=elig_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program offers at least three start dates per year or operates on rolling admissions.",
        node=starts_leaf,
        sources=start_dates.start_dates_source_urls,
        additional_instruction="Verify the cadence of intakes; if only one or two starts per year are offered, or if unclear, consider unsupported.",
        extra_prerequisites=[starts_sources_present],
    )

    # Minimal admissions requirements (breakdown)
    min_adm_node = evaluator.add_parallel(
        id="minimal_admissions",
        desc="Admission requirements meet ALL specified minimal criteria",
        parent=elig_node,
        critical=True,
    )
    adm_sources_present = evaluator.add_custom_node(
        result=bool(admissions.admissions_source_urls),
        id="admissions_sources_present",
        desc="Admissions requirements source URL(s) provided",
        parent=min_adm_node,
        critical=True,
    )

    # Bachelor's from accredited institution only
    adm_bach_leaf = evaluator.add_leaf(
        id="adm_bachelors_from_accredited",
        desc="Requires only a bachelor's degree from an accredited institution",
        parent=min_adm_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Admission requires only a bachelor's degree from an accredited institution.",
        node=adm_bach_leaf,
        sources=admissions.admissions_source_urls,
        additional_instruction="If the page requires additional credentials beyond the bachelor's degree by default, consider unsupported.",
        extra_prerequisites=[adm_sources_present],
    )

    # No specific undergraduate major required
    adm_no_major_leaf = evaluator.add_leaf(
        id="adm_no_specific_major",
        desc="No specific undergraduate major required",
        parent=min_adm_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Admission does not require a specific undergraduate major.",
        node=adm_no_major_leaf,
        sources=admissions.admissions_source_urls,
        additional_instruction="If the program requires or restricts to specific majors, consider unsupported.",
        extra_prerequisites=[adm_sources_present],
    )

    # No professional work experience required
    adm_no_exp_leaf = evaluator.add_leaf(
        id="adm_no_work_experience_required",
        desc="No professional work experience required",
        parent=min_adm_node,
        critical=True,
    )
    await evaluator.verify(
        claim="Admission does not require professional work experience.",
        node=adm_no_exp_leaf,
        sources=admissions.admissions_source_urls,
        additional_instruction="If the page requires prior professional experience for all applicants, consider unsupported.",
        extra_prerequisites=[adm_sources_present],
    )

    # Minimum GPA not above 2.5
    adm_gpa_leaf = evaluator.add_leaf(
        id="adm_min_gpa_at_most_2_5",
        desc="Minimum undergraduate GPA requirement, if any, is not above 2.5",
        parent=min_adm_node,
        critical=True,
    )
    await evaluator.verify(
        claim="If a minimum undergraduate GPA is specified, it is 2.5 or lower; alternatively there is no minimum above 2.5.",
        node=adm_gpa_leaf,
        sources=admissions.admissions_source_urls,
        additional_instruction="Consider '2.0' or '2.5' minimums acceptable. If '3.0' or higher is required, consider unsupported.",
        extra_prerequisites=[adm_sources_present],
    )


# ---------------------------
# Main evaluation entry
# ---------------------------
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

    # Add a critical task node under root
    task_main = evaluator.add_sequential(
        id="task_main",
        desc="Identify ONE qualifying fully online master's program from a U.S. university and provide the required details with verifiable sources",
        parent=root,
        critical=True,
    )

    # 1) Extract core info first (needed for gating)
    core: ProgramCoreExtraction = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=ProgramCoreExtraction,
        extraction_name="program_core",
    )

    # 2) Extract all other details concurrently
    acc_task = evaluator.extract(
        prompt=prompt_extract_accreditation(),
        template_class=AccreditationExtraction,
        extraction_name="accreditation_info",
    )
    tuition_task = evaluator.extract(
        prompt=prompt_extract_tuition(),
        template_class=TuitionExtraction,
        extraction_name="tuition_info",
    )
    tests_task = evaluator.extract(
        prompt=prompt_extract_test_policy(),
        template_class=TestPolicyExtraction,
        extraction_name="test_policy",
    )
    delivery_task = evaluator.extract(
        prompt=prompt_extract_delivery(),
        template_class=DeliveryExtraction,
        extraction_name="delivery_info",
    )
    timeline_task = evaluator.extract(
        prompt=prompt_extract_timeline(),
        template_class=TimelineExtraction,
        extraction_name="timeline_info",
    )
    starts_task = evaluator.extract(
        prompt=prompt_extract_start_dates(),
        template_class=StartDatesExtraction,
        extraction_name="start_dates_info",
    )
    admissions_task = evaluator.extract(
        prompt=prompt_extract_admissions(),
        template_class=AdmissionsExtraction,
        extraction_name="admissions_info",
    )

    accreditation, tuition, tests, delivery, timeline, start_dates, admissions = await asyncio.gather(
        acc_task, tuition_task, tests_task, delivery_task, timeline_task, starts_task, admissions_task
    )

    # 3) Required response fields (critical, evaluated first to gate)
    prereq_nodes = await build_required_response_fields(evaluator, task_main, core)

    # 4) Program eligibility and evidence (critical, depends on fields above)
    await build_program_eligibility_and_evidence(
        evaluator=evaluator,
        parent=task_main,
        core=core,
        accreditation=accreditation,
        tuition=tuition,
        tests=tests,
        delivery=delivery,
        timeline=timeline,
        start_dates=start_dates,
        admissions=admissions,
        prereq_nodes=prereq_nodes,
    )

    return evaluator.get_summary()