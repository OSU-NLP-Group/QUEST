import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# ------------------------------------------------------------------------------
# Task constants
# ------------------------------------------------------------------------------
TASK_ID = "power4_head_coach_requirements"
TASK_DESCRIPTION = (
    "What are the comprehensive minimum qualification requirements that a candidate must meet to be eligible for a head football coaching position at a Power 4 conference university? "
    "Your answer should identify and document: (1) The minimum educational credential required, including degree level and any accreditation requirements; "
    "(2) Any preferred advanced educational qualifications typically expected for Power 4 positions; "
    "(3) The specific NCAA certification test that must be passed, including the minimum passing score; "
    "(4) The NCAA bylaw number that establishes head coach compliance responsibilities; "
    "(5) The typical minimum number of years of collegiate coaching experience required; "
    "(6) The types of leadership coaching positions (if any) that are strongly preferred; "
    "(7) The mandatory screening requirements that all candidates must pass; "
    "(8) The types of professional success metrics that are typically evaluated. "
    "For each requirement, provide the specific details (such as exact degree level, test passing percentage, bylaw number, years of experience) and include reference URLs from your research that document these standards."
)

# Expected key facts to be asserted within the answer
EXPECTED_CONSTANTS = {
    "minimum_degree": "bachelor’s degree from an accredited institution",
    "preferred_masters": "master’s degree is strongly preferred",
    "cert_test_name": "NCAA Coaches Certification (Recruiting) Test",
    "cert_passing_score": "80% (24 out of 30 questions)",
    "bylaw_number": "11.1.2.1",
    "bylaw_resp_keywords": [
        "promote an atmosphere of compliance",
        "monitoring the activities of assistant coaches or staff",
    ],
    "min_collegiate_experience_years": "5 years of collegiate coaching experience",
    "preferred_leadership_roles": "head coach or coordinator experience (collegiate or professional) strongly preferred",
    "benchmark_avg_years": "approximately 17 years of coaching before first FBS head coach role",
    "background_check_components": [
        "criminal background check",
        "employment verification",
    ],
    "success_metrics_anyof": [
        "winning records",
        "championships",
        "player development",
        "recruiting achievements",
    ],
}

# ------------------------------------------------------------------------------
# Extraction models
# ------------------------------------------------------------------------------

class EducationInfo(BaseModel):
    minimum_degree_text: Optional[str] = None
    minimum_degree_urls: List[str] = Field(default_factory=list)
    preferred_masters_text: Optional[str] = None
    preferred_masters_urls: List[str] = Field(default_factory=list)


class CertificationComplianceInfo(BaseModel):
    test_name_text: Optional[str] = None
    passing_score_text: Optional[str] = None
    certification_urls: List[str] = Field(default_factory=list)
    bylaw_number_text: Optional[str] = None
    bylaw_responsibilities_text: Optional[str] = None
    bylaw_urls: List[str] = Field(default_factory=list)


class ExperienceLeadershipInfo(BaseModel):
    min_experience_text: Optional[str] = None
    min_experience_urls: List[str] = Field(default_factory=list)
    preferred_leadership_roles_text: Optional[str] = None
    preferred_leadership_roles_urls: List[str] = Field(default_factory=list)
    benchmark_avg_years_text: Optional[str] = None
    benchmark_avg_years_urls: List[str] = Field(default_factory=list)


class ScreeningInfo(BaseModel):
    background_check_text: Optional[str] = None
    screening_urls: List[str] = Field(default_factory=list)


class SuccessMetricsInfo(BaseModel):
    metrics_text: Optional[str] = None
    metrics_urls: List[str] = Field(default_factory=list)


class CoachRequirementsExtraction(BaseModel):
    education: Optional[EducationInfo] = None
    certification: Optional[CertificationComplianceInfo] = None
    experience_leadership: Optional[ExperienceLeadershipInfo] = None
    screening: Optional[ScreeningInfo] = None
    success_metrics: Optional[SuccessMetricsInfo] = None


# ------------------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------------------

def prompt_extract_requirements() -> str:
    return """
Extract the specific statements and the corresponding reference URLs the answer provides for each of the following eight requirement areas for a Power 4 head football coach. 
Return the exact quoted or paraphrased text from the answer for each item (if present), along with the list of URLs explicitly cited in the answer as supporting evidence for that item.

You must populate the following JSON structure:
- education:
  - minimum_degree_text: The sentence/phrase that states the minimum credential (e.g., "Bachelor’s degree from an accredited institution").
  - minimum_degree_urls: All URLs that the answer attaches to document the bachelor’s + accreditation minimum.
  - preferred_masters_text: The sentence/phrase that states that a Master’s degree is strongly preferred.
  - preferred_masters_urls: All URLs that document the Master’s preference.
- certification:
  - test_name_text: The sentence naming the NCAA Coaches Certification (Recruiting) Test.
  - passing_score_text: The sentence stating the minimum passing score (e.g., "80% (24/30)").
  - certification_urls: All URLs that document the test and passing-score requirement.
  - bylaw_number_text: The sentence identifying NCAA Division I Bylaw 11.1.2.1.
  - bylaw_responsibilities_text: The sentence stating head coach responsibilities: promoting an atmosphere of rules compliance AND monitoring subordinate staff.
  - bylaw_urls: All URLs that document Bylaw 11.1.2.1 and those responsibilities.
- experience_leadership:
  - min_experience_text: The sentence stating a typical minimum (e.g., 5 years) of collegiate coaching experience.
  - min_experience_urls: All URLs that document this typical minimum.
  - preferred_leadership_roles_text: The sentence identifying that prior head coach or coordinator experience (collegiate or professional) is strongly preferred.
  - preferred_leadership_roles_urls: All URLs that document preferred leadership experience types.
  - benchmark_avg_years_text: The sentence stating ~17 years average coaching experience before first FBS HC role.
  - benchmark_avg_years_urls: All URLs that document the ~17-year benchmark.
- screening:
  - background_check_text: The sentence stating that candidates must pass a comprehensive background check including criminal background AND employment verification.
  - screening_urls: All URLs that document the screening/background check requirements.
- success_metrics:
  - metrics_text: The sentence listing at least one of: winning records, championships, player development, and/or recruiting achievements as success metrics.
  - metrics_urls: All URLs that document the kinds of success metrics evaluated.

Special rules:
- Extract only what is explicitly stated in the answer. If any text is missing, set it to null.
- For URL fields, extract only actual URLs present in the answer (including markdown links). If none are provided, return an empty list.
- Do not invent or infer any URLs or statements.
    """.strip()


# ------------------------------------------------------------------------------
# Helper verification utilities
# ------------------------------------------------------------------------------

async def add_presence_check(
    evaluator: Evaluator,
    parent,
    leaf_id: str,
    desc: str,
    claim: str,
    critical: bool = True,
    additional_instruction: Optional[str] = None,
) -> None:
    """
    Add a leaf node that verifies the answer explicitly contains a required statement.
    Uses simple verification (no URLs), focusing on answer content.
    """
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction=(
            additional_instruction
            or "Judge only whether the answer text explicitly contains this statement (allow minor wording variations and synonyms)."
        ),
    )


async def add_url_support_check(
    evaluator: Evaluator,
    parent,
    leaf_id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str]],
    critical: bool = True,
    additional_instruction: Optional[str] = None,
) -> None:
    """
    Add a leaf node that verifies at least one provided URL explicitly supports the claim.
    Fails immediately if no URLs are provided.
    """
    node = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent,
        critical=critical,
    )
    url_list = urls or []
    if len(url_list) == 0:
        # No URLs provided -> this check inherently fails (the rubric requires >=1 supporting URL)
        node.score = 0.0
        node.status = "failed"
        evaluator.add_custom_info(
            {"reason": "no_urls_provided", "leaf_id": leaf_id, "desc": desc},
            info_type="missing_urls",
            info_name=f"missing_urls_{leaf_id}",
        )
        return

    await evaluator.verify(
        claim=claim,
        node=node,
        sources=url_list,
        additional_instruction=(
            additional_instruction
            or "Accept support if the page clearly states or strongly implies the claim, allowing minor wording variations (e.g., 'accredited college or university' for 'accredited institution')."
        ),
    )


# ------------------------------------------------------------------------------
# Category verifiers (tree builders)
# ------------------------------------------------------------------------------

async def verify_education_requirements(
    evaluator: Evaluator,
    parent,
    edu: Optional[EducationInfo],
) -> None:
    edu_node = evaluator.add_parallel(
        id="Education_Requirements",
        desc="Minimum and preferred educational qualifications (and supporting URLs).",
        parent=parent,
        critical=True,
    )

    # 1) Minimum_Bachelors_Accredited (presence in answer)
    await add_presence_check(
        evaluator,
        edu_node,
        "Minimum_Bachelors_Accredited",
        "States the minimum credential is a bachelor’s degree from an accredited institution (per constraints).",
        claim=(
            "The answer explicitly states that the minimum educational credential is a bachelor's degree from an accredited institution."
        ),
        additional_instruction=(
            "Check the answer text only. Allow synonyms like 'accredited college or university' and minor variations of 'bachelor’s degree'."
        ),
    )

    # 2) Minimum_Bachelors_URL (web support)
    await add_url_support_check(
        evaluator,
        edu_node,
        "Minimum_Bachelors_URL",
        "Provides ≥1 reference URL documenting the bachelor’s + accreditation minimum.",
        claim=(
            "This page documents that head football coach positions at major Division I (Power 4/Power 5) universities list a bachelor's degree from an accredited institution as a minimum requirement."
        ),
        urls=(edu.minimum_degree_urls if edu else []),
        additional_instruction=(
            "Accept job postings or official HR pages from universities within Power 4/Power 5 or equivalent Division I programs if they clearly state 'Bachelor’s degree' and mention accreditation (e.g., 'accredited college/university')."
        ),
    )

    # 3) Preferred_Masters (presence in answer)
    await add_presence_check(
        evaluator,
        edu_node,
        "Preferred_Masters",
        "States that a master’s degree is strongly preferred (per constraints).",
        claim="The answer explicitly states that a master's degree is strongly preferred for Power 4 head coaching positions.",
    )

    # 4) Preferred_Masters_URL (web support)
    await add_url_support_check(
        evaluator,
        edu_node,
        "Preferred_Masters_URL",
        "Provides ≥1 reference URL documenting the master’s preference.",
        claim=(
            "This page documents that a master's degree is strongly preferred (or an advanced degree preferred) for head coach positions at major Division I programs."
        ),
        urls=(edu.preferred_masters_urls if edu else []),
        additional_instruction=(
            "Accept phrasing such as 'master's preferred', 'advanced degree preferred', or 'graduate degree preferred', especially when listed for head coach roles at major programs."
        ),
    )


async def verify_ncaa_cert_and_compliance(
    evaluator: Evaluator,
    parent,
    cert: Optional[CertificationComplianceInfo],
) -> None:
    ncaa_node = evaluator.add_parallel(
        id="NCAA_Certification_And_Compliance",
        desc="NCAA certification test requirement and head coach compliance bylaw requirement (and supporting URLs).",
        parent=parent,
        critical=True,
    )

    # 1) Certification test name (presence in answer)
    await add_presence_check(
        evaluator,
        ncaa_node,
        "Certification_Test_Name",
        "Identifies the NCAA Coaches Certification (Recruiting) Test (per constraints).",
        claim="The answer explicitly identifies the 'NCAA Coaches Certification (Recruiting) Test' by name.",
        additional_instruction="Allow minor name variants like 'NCAA Recruiting Certification Test' or 'NCAA Coaches Recruiting Certification exam'.",
    )

    # 2) Certification passing score (presence in answer)
    await add_presence_check(
        evaluator,
        ncaa_node,
        "Certification_Test_Passing_Score",
        "States the minimum passing score is 80% (24 out of 30 questions) (per constraints).",
        claim=(
            "The answer explicitly states that the minimum passing score for the NCAA Coaches Certification (Recruiting) Test is 80% (24 out of 30 questions)."
        ),
        additional_instruction="Allow minor formatting differences; both '80%' and '24/30' should be present or clearly implied.",
    )

    # 3) Certification URL (web support)
    await add_url_support_check(
        evaluator,
        ncaa_node,
        "Certification_Test_URL",
        "Provides ≥1 reference URL documenting the test and the passing score requirement.",
        claim=(
            "This page describes the NCAA Coaches Certification (Recruiting) Test and states that the minimum passing score is 80% (i.e., 24 out of 30 questions)."
        ),
        urls=(cert.certification_urls if cert else []),
        additional_instruction=(
            "Accept official NCAA/compliance office pages or university compliance resources that explicitly mention the recruiting certification test and 80% (24/30) passing requirement."
        ),
    )

    # 4) Bylaw number (presence in answer)
    await add_presence_check(
        evaluator,
        ncaa_node,
        "Bylaw_Number",
        "Identifies NCAA Division I Bylaw 11.1.2.1 (per constraints).",
        claim="The answer explicitly identifies NCAA Division I Bylaw 11.1.2.1 as the head coach responsibility bylaw.",
        additional_instruction="Accept minor formatting such as '11.1.2.1' with or without 'Bylaw' text.",
    )

    # 5) Bylaw responsibilities (presence in answer)
    await add_presence_check(
        evaluator,
        ncaa_node,
        "Bylaw_Responsibilities",
        "States the head coach responsibilities include promoting an atmosphere of NCAA rules compliance AND monitoring subordinate staff (per constraints).",
        claim=(
            "The answer explicitly states that under Bylaw 11.1.2.1 the head coach is responsible for promoting an atmosphere of compliance and for monitoring the activities of assistant coaches and staff."
        ),
        additional_instruction="Both elements—promote an atmosphere of compliance AND monitor staff—must be present or clearly implied.",
    )

    # 6) Bylaw URL (web support)
    await add_url_support_check(
        evaluator,
        ncaa_node,
        "Bylaw_URL",
        "Provides ≥1 reference URL documenting Bylaw 11.1.2.1 and its responsibilities.",
        claim=(
            "This page presents NCAA Division I Bylaw 11.1.2.1 (Head Coach Responsibility) and makes clear the obligations to promote an atmosphere of rules compliance and to monitor staff."
        ),
        urls=(cert.bylaw_urls if cert else []),
        additional_instruction="Prefer official NCAA rulebook excerpts or institutional compliance pages that quote or paraphrase Bylaw 11.1.2.1.",
    )


async def verify_experience_and_leadership(
    evaluator: Evaluator,
    parent,
    exp: Optional[ExperienceLeadershipInfo],
) -> None:
    exp_node = evaluator.add_parallel(
        id="Experience_And_Leadership",
        desc="Coaching experience expectations and preferred leadership roles (and supporting URLs).",
        parent=parent,
        critical=True,
    )

    # 1) Typical minimum collegiate experience (presence in answer)
    await add_presence_check(
        evaluator,
        exp_node,
        "Typical_Minimum_Collegiate_Experience",
        "States the typical minimum is 5 years of collegiate coaching experience (per constraints).",
        claim="The answer explicitly states that a typical minimum is 5 years of collegiate coaching experience.",
    )

    # 2) Typical minimum experience URL (web support)
    await add_url_support_check(
        evaluator,
        exp_node,
        "Typical_Minimum_Experience_URL",
        "Provides ≥1 reference URL documenting the 5-year typical minimum.",
        claim=(
            "This page documents that job postings or norms for head football coach positions at major Division I programs typically list around 5 years of collegiate coaching experience as a minimum."
        ),
        urls=(exp.min_experience_urls if exp else []),
        additional_instruction="Accept representative job postings or HR policy pages from major Division I/Power 4/Power 5 programs that specify a 5-year minimum.",
    )

    # 3) Preferred leadership roles (presence in answer)
    await add_presence_check(
        evaluator,
        exp_node,
        "Preferred_Leadership_Roles",
        "Identifies head coaching or coordinator experience (collegiate or professional) as strongly preferred (per constraints).",
        claim="The answer explicitly identifies prior head coach or coordinator experience (collegiate or professional) as strongly preferred.",
        additional_instruction="Accept role names such as 'Offensive Coordinator', 'Defensive Coordinator', or 'Head Coach' experience.",
    )

    # 4) Preferred leadership roles URL (web support)
    await add_url_support_check(
        evaluator,
        exp_node,
        "Preferred_Leadership_Roles_URL",
        "Provides ≥1 reference URL documenting the preferred leadership experience types.",
        claim=(
            "This page documents that head coach or coordinator experience is strongly preferred for head coach positions at major programs."
        ),
        urls=(exp.preferred_leadership_roles_urls if exp else []),
        additional_instruction="Accept job postings or program announcements that explicitly prefer prior head coach/coordinator experience.",
    )

    # 5) Benchmark average years (presence in answer)
    await add_presence_check(
        evaluator,
        exp_node,
        "Benchmark_Average_Years",
        "States the benchmark that an average successful FBS head coach has approximately 17 years of coaching experience prior to first FBS head coach role (per constraints).",
        claim=(
            "The answer explicitly states that an average successful FBS head coach has approximately 17 years of coaching experience prior to the first FBS head coach role."
        ),
        additional_instruction="Accept approximate phrasings like 'around 17 years', 'about 17 years', or a range very close to 17.",
    )

    # 6) Benchmark average years URL (web support)
    await add_url_support_check(
        evaluator,
        exp_node,
        "Benchmark_Average_Years_URL",
        "Provides ≥1 reference URL documenting the ~17-year benchmark.",
        claim=(
            "This page documents that the average or typical years of prior coaching experience before becoming a first-time FBS head coach is approximately 17 years."
        ),
        urls=(exp.benchmark_avg_years_urls if exp else []),
        additional_instruction="Accept industry analyses, reputable articles, or research with data near ~17 years.",
    )


async def verify_mandatory_screening(
    evaluator: Evaluator,
    parent,
    screening: Optional[ScreeningInfo],
) -> None:
    screening_node = evaluator.add_parallel(
        id="Mandatory_Screening",
        desc="Mandatory screening requirements (and supporting URL).",
        parent=parent,
        critical=True,
    )

    # 1) Background check components (presence in answer)
    await add_presence_check(
        evaluator,
        screening_node,
        "Background_Check_Components",
        "States candidates must pass a comprehensive background check including criminal background AND employment verification (per constraints).",
        claim=(
            "The answer explicitly states that candidates must pass a comprehensive background check that includes criminal background and employment verification."
        ),
        additional_instruction="Both components—criminal background and employment verification—must appear or be clearly implied.",
    )

    # 2) Screening URL (web support)
    await add_url_support_check(
        evaluator,
        screening_node,
        "Screening_URL",
        "Provides ≥1 reference URL documenting the screening/background check requirements.",
        claim=(
            "This page documents that coaching hires must undergo mandatory screening that includes at least criminal background checks and employment verification."
        ),
        urls=(screening.screening_urls if screening else []),
        additional_instruction="Accept institutional HR policy pages or job postings that explicitly list these screening components.",
    )


async def verify_success_metrics(
    evaluator: Evaluator,
    parent,
    metrics: Optional[SuccessMetricsInfo],
) -> None:
    metrics_node = evaluator.add_parallel(
        id="Success_Metrics",
        desc="Professional success metrics evaluated (and supporting URL).",
        parent=parent,
        critical=True,
    )

    # 1) Metrics types (presence in answer)
    await add_presence_check(
        evaluator,
        metrics_node,
        "Metrics_Types",
        "States that success can be evidenced by at least one of the following metric types: winning records, championships, player development, and/or recruiting achievements (per constraints).",
        claim=(
            "The answer lists at least one of these success metric types: winning records, championships, player development, or recruiting achievements."
        ),
        additional_instruction=(
            "It is sufficient for the answer to list at least one of the four specified metric types. Allow minor wording variants (e.g., 'wins', 'titles', 'player development', 'recruiting success')."
        ),
    )

    # 2) Metrics URL (web support)
    await add_url_support_check(
        evaluator,
        metrics_node,
        "Metrics_URL",
        "Provides ≥1 reference URL documenting the kinds of success metrics evaluated.",
        claim=(
            "This page documents the kinds of success metrics commonly considered when evaluating head coach candidates, such as winning record, championships, player development, and/or recruiting accomplishments."
        ),
        urls=(metrics.metrics_urls if metrics else []),
        additional_instruction="Accept coaching evaluation criteria pages, official bios that describe evaluation factors, or credible articles outlining coach assessment metrics.",
    )


# ------------------------------------------------------------------------------
# Main evaluation entry point
# ------------------------------------------------------------------------------

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
    Evaluate an answer for Power 4 head football coach qualification requirements.
    Builds a critical parallel rubric tree that checks presence in the answer and
    URL-grounded support for each required standard.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The rubric root is parallel
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

    # Extract structured info from answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=CoachRequirementsExtraction,
        extraction_name="coach_requirements_extraction",
    )

    # Record expected constants as ground-truth context (not used for pass/fail directly)
    evaluator.add_ground_truth(
        {
            "expected": EXPECTED_CONSTANTS,
            "notes": "These reflect the constraints against which the answer is judged for presence and URL-supported evidence.",
        },
        gt_type="expected_constants",
    )

    # Create a critical top-level rubric node (parallel)
    top = evaluator.add_parallel(
        id="Power_4_Head_Coach_Qualification_Requirements",
        desc="Evaluate whether the answer covers the required qualification requirements and provides supporting reference URLs for each required standard.",
        parent=root,
        critical=True,
    )

    # Build category subtrees
    await verify_education_requirements(evaluator, top, extraction.education if extraction else None)
    await verify_ncaa_cert_and_compliance(evaluator, top, extraction.certification if extraction else None)
    await verify_experience_and_leadership(evaluator, top, extraction.experience_leadership if extraction else None)
    await verify_mandatory_screening(evaluator, top, extraction.screening if extraction else None)
    await verify_success_metrics(evaluator, top, extraction.success_metrics if extraction else None)

    # Return evaluation summary
    return evaluator.get_summary()