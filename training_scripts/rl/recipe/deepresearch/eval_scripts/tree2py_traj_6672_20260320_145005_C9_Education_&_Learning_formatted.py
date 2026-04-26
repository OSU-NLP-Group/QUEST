import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task metadata
# -----------------------------------------------------------------------------
TASK_ID = "su_master_eligibility_autumn_2026"
TASK_DESCRIPTION = """
An international student from India (non-EU/EEA citizen) plans to pursue a master's degree in Computer Science at a Swedish university starting in autumn 2026. The student has the following credentials and circumstances:

- Bachelor of Technology in Computer Engineering from an Indian university, completed in May 2025 with a CGPA of 8.2/10
- TOEFL iBT test taken in October 2025 with scores: Overall 95 (Reading 25, Listening 24, Writing 23, Speaking 23)
- Indian passport valid until March 2030
- Employed full-time with savings to cover expenses
- Can access bank statements showing adequate funds
- Aware of non-EU/EEA tuition fee obligations

Verify the student's complete eligibility to apply for and be admitted to a suitable English-taught master's program in Computer Science at Stockholm University for the autumn 2026 semester. Your verification must address:

1. Academic qualification eligibility (degree, field relevance, grades, timing)
2. English language proficiency requirements (test type, scores, validity)
3. Application process compliance (submission period, fees, required documents, deadlines)
4. Financial requirements (tuition awareness, living expense proof capability)
5. Student visa/residence permit requirements (process understanding, passport validity, enrollment status)

For each requirement category, provide:
- Verification that the student meets the specific requirement
- Supporting evidence with reference URLs from official sources
- Identification of any gaps or additional steps needed

Your response must be grounded in official requirements from Stockholm University, Swedish university admission system (universityadmissions.se), and Swedish Migration Agency (migrationsverket.se) as of early 2026.
"""


# -----------------------------------------------------------------------------
# Extraction Models
# -----------------------------------------------------------------------------
class Sources(BaseModel):
    # Only extract URLs explicitly present in the answer, grouped by category
    program: List[str] = Field(default_factory=list, description="Stockholm University programme/entry requirement pages")
    english: List[str] = Field(default_factory=list, description="English proficiency requirement/accepted tests pages (SU or universityadmissions.se)")
    application: List[str] = Field(default_factory=list, description="Application process, key dates, application fee pages (universityadmissions.se)")
    finances: List[str] = Field(default_factory=list, description="Tuition fees pages (SU) and possibly related financial info")
    migration: List[str] = Field(default_factory=list, description="Swedish Migration Agency (migrationsverket.se) pages about residence permits, maintenance funds, passport validity")


class ExtractedEligibility(BaseModel):
    # Academic fit
    degree_completion_date: Optional[str] = None  # e.g., "May 2025"
    degree_completed_by_aug_2026: Optional[bool] = None
    degree_field: Optional[str] = None            # e.g., "Computer Engineering"
    cgpa: Optional[str] = None                    # keep string for flexibility, e.g., "8.2/10"
    prerequisite_statement: Optional[str] = None  # student's claim that prerequisites are met or identified

    # English proficiency
    english_test: Optional[str] = None            # e.g., "TOEFL iBT"
    toefl_overall: Optional[int] = None
    toefl_reading: Optional[int] = None
    toefl_listening: Optional[int] = None
    toefl_writing: Optional[int] = None
    toefl_speaking: Optional[int] = None
    english_test_date: Optional[str] = None       # e.g., "October 2025"

    # Application process acknowledgements
    app_window_ack: Optional[bool] = None         # acknowledges Oct 16–Jan 15 window and platform (universityadmissions.se)
    app_fee_ack: Optional[bool] = None            # acknowledges 900 SEK fee and deadline (early Feb)
    docs_transcripts_ack: Optional[bool] = None   # acknowledges official transcripts submission
    docs_diploma_ack: Optional[bool] = None       # acknowledges degree certificate/proof of graduation
    docs_tests_ack: Optional[bool] = None         # acknowledges official TOEFL/IELTS submission procedure

    # Financials
    tuition_awareness_ack: Optional[bool] = None  # acknowledges non-EU/EEA tuition fee obligations (and ideally amount/program page)
    funds_proof_ack: Optional[bool] = None        # acknowledges ability to show bank statements meeting monthly requirement

    # Visa/residence permit understanding
    admission_then_visa_ack: Optional[bool] = None  # understands admission first, then residence permit application (and fee instalment)
    full_time_enroll_ack: Optional[bool] = None     # acknowledges full-time enrollment requirement for permit
    passport_valid_until: Optional[str] = None      # e.g., "March 2030"

    # Grouped sources (URLs must be explicitly present in the answer)
    sources: Sources = Field(default_factory=Sources)


# -----------------------------------------------------------------------------
# Extraction Prompt
# -----------------------------------------------------------------------------
def prompt_extract_eligibility() -> str:
    return """
Extract the student's stated plan/compliance and the cited official URLs from the answer, strictly based on what the answer explicitly contains.

Return a JSON with:
- degree_completion_date (string or null)
- degree_completed_by_aug_2026 (boolean or null): whether the answer says the bachelor's is/will be completed before Aug 2026
- degree_field (string or null)
- cgpa (string or null, e.g., "8.2/10")
- prerequisite_statement (string or null): any explicit statement about required prerequisite coursework being met or identified

- english_test (string or null), toefl_overall (int or null), toefl_reading (int or null), toefl_listening (int or null), toefl_writing (int or null), toefl_speaking (int or null), english_test_date (string or null)

- app_window_ack (boolean or null): acknowledges international application period and platform (universityadmissions.se)
- app_fee_ack (boolean or null): acknowledges 900 SEK application fee and deadline (early Feb 2026)
- docs_transcripts_ack (boolean or null)
- docs_diploma_ack (boolean or null)
- docs_tests_ack (boolean or null)

- tuition_awareness_ack (boolean or null): acknowledges non‑EU/EEA tuition obligations (ideally with programme fee awareness)
- funds_proof_ack (boolean or null): acknowledges ability to show bank statements/funds for living costs

- admission_then_visa_ack (boolean or null): understands admission/acceptance letter must come before residence permit application and fee instalment requirement
- full_time_enroll_ack (boolean or null): acknowledges full‑time enrollment requirement for residence permit
- passport_valid_until (string or null)

- sources: an object grouping only URLs explicitly present in the answer:
  - program: Stockholm University programme/entry requirements pages (su.se)
  - english: English proficiency requirement/accepted tests (universityadmissions.se and/or su.se)
  - application: application process, application fee, key dates (universityadmissions.se)
  - finances: tuition fee pages (su.se) or related official financial pages
  - migration: Swedish Migration Agency (migrationsverket.se) pages for student permits, maintenance funds, passport validity

Rules:
- Only include URLs that appear in the answer (including markdown links) and are valid.
- Do not invent URLs. If a category has no URLs, return an empty array for that category.
- Set booleans to true only if the acknowledgment is explicit in the answer text; otherwise false if contradicted, or null if not mentioned.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _merge_sources(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst or []:
            if isinstance(u, str) and u.strip() and u not in seen:
                out.append(u)
                seen.add(u)
    return out


# -----------------------------------------------------------------------------
# Verification subtrees
# -----------------------------------------------------------------------------
async def verify_program_academic_fit(evaluator: Evaluator, parent, ext: ExtractedEligibility) -> None:
    """
    Program_Academic_Fit: parallel aggregator
    Notes:
    - We set this parent as non-critical to allow partial credit (the rubric includes a non-critical prerequisite subcheck).
    """
    node = evaluator.add_parallel(
        id="program_academic_fit",
        desc="Verify academic qualifications and program fit",
        parent=parent,
        critical=False  # adjusted to allow non-critical children per framework rule
    )

    # Degree_Completion (critical leaf)
    lc = evaluator.add_leaf(
        id="degree_completion",
        desc="Bachelor's degree completed or will be completed before program start (before August 2026)",
        parent=node,
        critical=True
    )
    degree_date = ext.degree_completion_date or "May 2025"
    completed_flag = ext.degree_completed_by_aug_2026
    claim_dc = (
        f"According to the answer, the bachelor's degree is completed by {degree_date}, "
        f"which is before the autumn 2026 start (before August 2026). Therefore, the timing requirement is satisfied."
    )
    await evaluator.verify(
        claim=claim_dc,
        node=lc,
        additional_instruction="This is a logical/timing check based on the answer's stated completion date; no external URL is required."
    )

    # Field_Alignment (critical leaf) – requires official support that 'CS or equivalent' (incl. related fields) is acceptable
    fa = evaluator.add_leaf(
        id="field_alignment",
        desc="Bachelor's degree field (Computer Engineering) provides appropriate foundation for intended master's program in Computer Science",
        parent=node,
        critical=True
    )
    program_sources = ext.sources.program  # SU programme/entry requirement page(s), ideally
    degree_field = ext.degree_field or "Computer Engineering"
    claim_fa = (
        f"The Stockholm University Computer Science master's entry requirements accept a bachelor's degree in Computer Science "
        f"or an equivalent/closely related field provided required prerequisites are met. A bachelor's in {degree_field} is "
        f"considered an appropriate foundation if the listed CS prerequisites are covered."
    )
    await evaluator.verify(
        claim=claim_fa,
        node=fa,
        sources=program_sources,
        additional_instruction="Check the programme/entry-requirement page(s) for wording like 'Computer Science or equivalent' or similar acceptance of related fields (e.g., Computer Engineering) provided prerequisites are fulfilled."
    )

    # Grade_Standards (non-critical) – Swedish programmes typically do not state a fixed GPA minimum; selection is competitive.
    gs = evaluator.add_leaf(
        id="grade_standards",
        desc="Academic performance (CGPA 8.2/10) meets minimum competitive threshold for admission",
        parent=node,
        critical=False  # adjusted: often no fixed GPA is published; selection is competitive
    )
    cgpa = ext.cgpa or "8.2/10"
    claim_gs = (
        f"Official programme information does not specify a fixed minimum GPA threshold; selection is competitive and often based on "
        f"grades and other merits. Therefore, having a completed bachelor's with a CGPA of {cgpa} does not violate any posted minimum GPA requirement."
    )
    await evaluator.verify(
        claim=claim_gs,
        node=gs,
        sources=program_sources,
        additional_instruction="Verify that the official page does not state a fixed minimum GPA (instead indicates competitive/merit-based selection)."
    )

    # Prerequisite_Courses (non-critical) – validate that the answer addresses programme prerequisites with sources
    pc = evaluator.add_leaf(
        id="prerequisite_courses",
        desc="Any specific prerequisite coursework completed if required by program",
        parent=node,
        critical=False
    )
    prereq_stmt = (ext.prerequisite_statement or "").strip()
    claim_pc = (
        "The answer identifies the specific prerequisite coursework required by the Stockholm University Computer Science master's "
        "programme (e.g., programming, algorithms/data structures, mathematics) and states that the student has them (or equivalencies) "
        "so the prerequisite requirement is addressed."
    )
    await evaluator.verify(
        claim=claim_pc,
        node=pc,
        sources=program_sources,
        additional_instruction="Confirm both: (1) the official prerequisites listed on the programme page; and (2) the answer explicitly addresses meeting them."
    )


async def verify_language_proficiency(evaluator: Evaluator, parent, ext: ExtractedEligibility) -> None:
    """
    Language_Proficiency_Met: parallel aggregator
    Note: Set parent non-critical to permit a non-critical 'section minima' node while keeping key checks critical.
    """
    node = evaluator.add_parallel(
        id="language_proficiency",
        desc="English language requirements fulfilled",
        parent=parent,
        critical=False  # adjusted to allow a mix of critical and non-critical children
    )

    english_sources = ext.sources.english  # ideally universityadmissions.se English 6 and/or SU page
    test_name = ext.english_test or "TOEFL iBT"
    overall = ext.toefl_overall if isinstance(ext.toefl_overall, int) else 95
    r = ext.toefl_reading if isinstance(ext.toefl_reading, int) else 25
    l = ext.toefl_listening if isinstance(ext.toefl_listening, int) else 24
    w = ext.toefl_writing if isinstance(ext.toefl_writing, int) else 23
    s = ext.toefl_speaking if isinstance(ext.toefl_speaking, int) else 23
    test_date = ext.english_test_date or "October 2025"

    # Test_Type_Accepted (critical)
    tta = evaluator.add_leaf(
        id="test_type_accepted",
        desc="English proficiency test type (TOEFL iBT, IELTS Academic, or equivalent) is accepted by Stockholm University",
        parent=node,
        critical=True
    )
    claim_tta = (
        f"{test_name} is an accepted English test to demonstrate English 6/B for master's level admissions in Sweden/Stockholm University."
    )
    await evaluator.verify(
        claim=claim_tta,
        node=tta,
        sources=english_sources,
        additional_instruction="Verify the official list of accepted tests for English 6/B, ensuring TOEFL iBT is included."
    )

    # Overall_Score_Threshold (critical)
    ost = evaluator.add_leaf(
        id="overall_score_threshold",
        desc="Overall test score meets or exceeds minimum requirement (TOEFL iBT 90 or IELTS 6.5 equivalent)",
        parent=node,
        critical=True
    )
    claim_ost = (
        f"The minimum requirement for English 6/B using TOEFL iBT is an overall score of 90 (and specified written minimums). "
        f"The student's {test_name} score is {overall}, which meets or exceeds the minimum overall threshold."
    )
    await evaluator.verify(
        claim=claim_ost,
        node=ost,
        sources=english_sources,
        additional_instruction="Confirm that TOEFL iBT 90 overall satisfies English 6/B; then check that the answer's stated overall score meets/exceeds 90."
    )

    # Test_Validity_Period (critical)
    tvp = evaluator.add_leaf(
        id="test_validity_period",
        desc="Test was taken within validity period (typically within 2 years of application - test date October 2025 for autumn 2026 application)",
        parent=node,
        critical=True
    )
    claim_tvp = (
        f"English test results (e.g., TOEFL/IELTS) are accepted if within the standard validity/verification period (typically 2 years). "
        f"A {test_name} taken in {test_date} is valid for an autumn 2026 application."
    )
    await evaluator.verify(
        claim=claim_tvp,
        node=tvp,
        sources=english_sources,
        additional_instruction="Check official validity/verification policy (commonly 2 years) and confirm that a test from Oct 2025 is acceptable for Autumn 2026."
    )

    # Section_Scores_Adequate (non-critical)
    ssa = evaluator.add_leaf(
        id="section_scores_adequate",
        desc="Individual section scores meet any specified minimum thresholds if required",
        parent=node,
        critical=False
    )
    claim_ssa = (
        "For TOEFL iBT to meet English 6/B, the commonly stated sub-score requirement is at least 20 in the written (writing) section. "
        f"The student's writing score is {w}, so section minima (if specified) are satisfied."
    )
    await evaluator.verify(
        claim=claim_ssa,
        node=ssa,
        sources=english_sources,
        additional_instruction="Look for section minima (especially writing >= 20 for TOEFL iBT) and verify the answer's section scores satisfy them."
    )


async def verify_application_process(evaluator: Evaluator, parent, ext: ExtractedEligibility) -> None:
    """
    Application_Process_Compliance: sequential aggregator (order matters)
    Keep this node critical per rubric; all children under it are critical in the rubric.
    """
    node = evaluator.add_sequential(
        id="application_process_compliance",
        desc="Application process requirements and deadlines met",
        parent=parent,
        critical=True
    )

    app_sources = ext.sources.application

    # Application_Submission (critical)
    asub = evaluator.add_leaf(
        id="application_submission_window",
        desc="Application submitted during official period (October 16, 2025 - January 15, 2026) through universityadmissions.se platform",
        parent=node,
        critical=True
    )
    claim_asub = (
        "For international master's Autumn 2026 (first round), the official application period is October 16, 2025 to January 15, 2026, "
        "and applications are submitted via universityadmissions.se. The answer acknowledges applying within this window and using the platform."
    )
    await evaluator.verify(
        claim=claim_asub,
        node=asub,
        sources=app_sources,
        additional_instruction="Verify official 'Key dates' for Autumn 2026 first admission round and the use of universityadmissions.se as the platform. Also check that the answer acknowledges this."
    )

    # Fee_Payment (critical)
    fee = evaluator.add_leaf(
        id="application_fee_payment",
        desc="Application fee (900 SEK) paid by deadline (early February 2026)",
        parent=node,
        critical=True
    )
    claim_fee = (
        "The application fee for non-EU/EEA applicants is 900 SEK, and the deadline to pay for the Autumn 2026 round is in early February 2026. "
        "The answer acknowledges paying the fee by the deadline."
    )
    await evaluator.verify(
        claim=claim_fee,
        node=fee,
        sources=app_sources,
        additional_instruction="Confirm the application fee amount (SEK 900) and the specific payment deadline date in early February 2026 from official 'Key dates' and application fee pages."
    )

    # Documents_Submitted (critical parallel subnode)
    docs = evaluator.add_parallel(
        id="documents_submitted",
        desc="All required supporting documents submitted by deadline",
        parent=node,
        critical=True
    )

    # Transcripts_Submitted (critical)
    d_tr = evaluator.add_leaf(
        id="transcripts_submitted",
        desc="Official transcripts from all institutions submitted",
        parent=docs,
        critical=True
    )
    claim_tr = (
        "Official sources require applicants to submit official transcripts from all post-secondary studies by the documents deadline. "
        "The answer acknowledges submitting official transcripts accordingly."
    )
    await evaluator.verify(
        claim=claim_tr,
        node=d_tr,
        sources=app_sources,
        additional_instruction="Check 'Submit your documents' and programme/university instructions confirming official transcripts are required and that the answer acknowledges this."
    )

    # Diploma_Certificate_Submitted (critical)
    d_dc = evaluator.add_leaf(
        id="diploma_certificate_submitted",
        desc="Degree certificate or proof of expected graduation submitted",
        parent=docs,
        critical=True
    )
    claim_dc = (
        "Official instructions require a degree certificate (or an official document showing expected graduation if not yet awarded) to be submitted by the deadline. "
        "The answer acknowledges submitting this."
    )
    await evaluator.verify(
        claim=claim_dc,
        node=d_dc,
        sources=app_sources,
        additional_instruction="Verify document rules regarding degree certificate or proof of expected graduation for applicants who have recently completed/will complete their bachelor's."
    )

    # Test_Scores_Submitted (critical)
    d_ts = evaluator.add_leaf(
        id="test_scores_submitted",
        desc="Official English test scores submitted",
        parent=docs,
        critical=True
    )
    claim_ts = (
        "For TOEFL/IELTS, applicants must follow the official submission procedure (e.g., TOEFL results sent directly by ETS). "
        "The answer acknowledges ensuring official English test scores are properly submitted."
    )
    await evaluator.verify(
        claim=claim_ts,
        node=d_ts,
        sources=_merge_sources(app_sources, ext.sources.english),
        additional_instruction="Check the official instructions on how to submit TOEFL/IELTS results (e.g., ETS direct delivery) and that the answer acknowledges following this."
    )


async def verify_financials(evaluator: Evaluator, parent, ext: ExtractedEligibility) -> None:
    """
    Financial_Requirements_Met: parallel aggregator, critical (all children critical in rubric).
    """
    node = evaluator.add_parallel(
        id="financial_requirements_met",
        desc="Financial obligations and proof requirements satisfied",
        parent=parent,
        critical=True
    )

    tuition_sources = ext.sources.finances  # SU tuition pages/program page(s)
    migration_sources = ext.sources.migration  # MIG maintenance funds requirement

    # Tuition_Fee_Awareness (critical)
    tfa = evaluator.add_leaf(
        id="tuition_fee_awareness",
        desc="Student demonstrates awareness of specific tuition fee amount for the program and confirms ability to pay",
        parent=node,
        critical=True
    )
    claim_tfa = (
        "Non‑EU/EEA students must pay tuition fees to Stockholm University for English‑taught master's programmes; "
        "the answer acknowledges the tuition obligation (and, ideally, the programme's fee) and confirms ability to pay."
    )
    await evaluator.verify(
        claim=claim_tfa,
        node=tfa,
        sources=tuition_sources,
        additional_instruction="Verify official SU tuition fee policy (and, if available, the programme-specific fee page). Confirm the answer states awareness and ability to pay."
    )

    # Living_Expense_Proof_Capability (critical)
    lep = evaluator.add_leaf(
        id="living_expense_proof_capability",
        desc="Student can provide proof of funds meeting Swedish Migration Agency requirement (at least 10,314 SEK per month) for study duration through bank statements or scholarship letters",
        parent=node,
        critical=True
    )
    claim_lep = (
        "The Swedish Migration Agency requires students to show maintenance funds of at least 10,314 SEK per month for the period of study. "
        "The answer states the student can provide bank statements with adequate funds, thus satisfying the maintenance funds requirement."
    )
    await evaluator.verify(
        claim=claim_lep,
        node=lep,
        sources=migration_sources,
        additional_instruction="Confirm the current monthly maintenance funds amount (10,314 SEK/month) on migrationsverket.se and that the answer acknowledges capability to provide proof."
    )


async def verify_visa_permit(evaluator: Evaluator, parent, ext: ExtractedEligibility) -> None:
    """
    Visa_Legal_Compliance: parallel aggregator, critical (all children critical in rubric).
    """
    node = evaluator.add_parallel(
        id="visa_legal_compliance",
        desc="Student residence permit and legal requirements addressable",
        parent=parent,
        critical=True
    )

    migration_sources = ext.sources.migration

    # Admission_Before_Visa_Understanding (critical)
    abv = evaluator.add_leaf(
        id="admission_before_visa",
        desc="Student demonstrates understanding that residence permit application requires admission letter first and describes correct admission-then-visa sequence",
        parent=node,
        critical=True
    )
    claim_abv = (
        "The residence permit for higher education requires that the applicant has been admitted to full‑time studies in Sweden "
        "and (for first-time permits) has paid the first instalment of tuition fees before a permit can be granted. "
        "The answer correctly describes the sequence: admission first, then residence permit application with proof of fee payment and funds."
    )
    await evaluator.verify(
        claim=claim_abv,
        node=abv,
        sources=migration_sources,
        additional_instruction="Verify migrationsverket.se guidance that you must be admitted and (before a permit is granted) have paid the first tuition instalment, and that the answer reflects this."
    )

    # Passport_Validity_Adequate (critical)
    pva = evaluator.add_leaf(
        id="passport_validity",
        desc="Passport validity (until March 2030) covers intended study duration plus required margin (2-year program completion by August 2028)",
        parent=node,
        critical=True
    )
    passport_until = ext.passport_valid_until or "March 2030"
    claim_pva = (
        f"The student's passport is valid until {passport_until}, which covers a two‑year master's programme running roughly 2026–2028 "
        "and satisfies the requirement that the passport must be valid for the intended period of stay."
    )
    await evaluator.verify(
        claim=claim_pva,
        node=pva,
        sources=migration_sources,
        additional_instruction="Verify migrationsverket.se states the passport must be valid for the intended period of stay; then check that validity through 2030 sufficiently covers studies to 2028."
    )

    # Full_Time_Enrollment_Understanding (critical)
    fte = evaluator.add_leaf(
        id="full_time_enrollment_understanding",
        desc="Student acknowledges that full-time enrollment is mandatory to maintain residence permit status",
        parent=node,
        critical=True
    )
    claim_fte = (
        "For a residence permit for higher education, the student must be admitted to and pursue full‑time studies. "
        "The answer acknowledges that full‑time enrollment is required to obtain/maintain the residence permit."
    )
    await evaluator.verify(
        claim=claim_fte,
        node=fte,
        sources=migration_sources,
        additional_instruction="Verify the migrationsverket.se requirement that studies must be full-time, and confirm the answer acknowledges this."
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Entry point for evaluating eligibility for Stockholm University Master's (Autumn 2026).
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level categories evaluated in parallel
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_eligibility(),
        template_class=ExtractedEligibility,
        extraction_name="eligibility_extraction",
    )

    # Add some contextual "ground truth" info that may help reviewers (not used for verification)
    evaluator.add_ground_truth({
        "target_university": "Stockholm University",
        "target_programme_example": "Master’s Programme in Computer Science (English-taught)",
        "intake": "Autumn 2026 (first admission round: international)",
        "known_requirements_high_level": [
            "Bachelor’s degree in CS or equivalent with required prerequisites",
            "English 6/B demonstrated via accepted tests (e.g., TOEFL iBT 90 with writing 20, IELTS 6.5, etc.)",
            "Apply via universityadmissions.se during key dates, pay SEK 900 application fee, submit required documents by deadline",
            "Non‑EU/EEA tuition fee obligation (SU tuition fees apply)",
            "Residence permit: admitted to full‑time studies, passport valid, maintenance funds (10,314 SEK/month), proof of first tuition instalment before permit"
        ]
    }, gt_type="context")

    # Build verification tree as per rubric (with minor criticality adjustments explained in code)
    # Overall node (we keep root non-critical to allow partial scoring across categories)
    overall = evaluator.add_parallel(
        id="overall_eligibility",
        desc="Complete eligibility verification for master's program admission in Sweden for autumn 2026",
        parent=root,
        critical=False  # adjusted to allow partial credit across categories
    )

    # Subtrees
    await verify_program_academic_fit(evaluator, overall, extracted)
    await verify_language_proficiency(evaluator, overall, extracted)
    await verify_application_process(evaluator, overall, extracted)
    await verify_financials(evaluator, overall, extracted)
    await verify_visa_permit(evaluator, overall, extracted)

    # Return the structured summary with the verification tree and scores
    return evaluator.get_summary()