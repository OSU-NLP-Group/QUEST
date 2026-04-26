import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "career_services_excellence_three_universities"
TASK_DESCRIPTION = (
    "Identify three U.S. accredited universities that demonstrate excellence in career services by meeting all of the "
    "following criteria:\n\n"
    "1. Active participation in NACE's First Destination Survey initiative with publicly available survey results\n"
    "2. Achievement of a minimum 65% knowledge rate on their most recent first destination survey (collected within 6 months of graduation)\n"
    "3. Public reporting of career outcome rates showing the percentage of graduates who are employed, continuing education, in service programs, or in military service\n"
    "4. Maintenance of active alumni career networking services, platforms, or programs with documented features and accessibility\n"
    "5. Offering of professional certification or credential programs that are job-relevant and industry-recognized\n"
    "6. Public documentation of career services engagement metrics, such as student advising appointment participation rates or similar measurable indicators\n"
    "7. Published evidence demonstrating career services impact on student outcomes, such as improvements in persistence rates, retention rates, or employment outcomes\n\n"
    "For each identified university, provide:\n"
    "- The institution's complete name and location (city, state)\n"
    "- The specific knowledge rate percentage achieved on their most recent first destination survey\n"
    "- The reported career outcome rate as a percentage\n"
    "- A description of their alumni career networking platform, services, or programs\n"
    "- Specific examples of professional certification or credential programs offered\n"
    "- The career services engagement or impact metrics documented by the institution\n"
    "- Reference URLs that verify each of the above criteria"
)


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class UniversityInstitutional(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    accreditation_urls: List[str] = Field(default_factory=list)
    nace_participation_urls: List[str] = Field(default_factory=list)
    fds_urls: List[str] = Field(default_factory=list)

    knowledge_rate_percent: Optional[str] = None
    knowledge_rate_urls: List[str] = Field(default_factory=list)

    data_collection_timeline_desc: Optional[str] = None
    timeline_urls: List[str] = Field(default_factory=list)

    institutional_reference_urls: List[str] = Field(default_factory=list)


class UniversityOutcomes(BaseModel):
    career_outcome_rate_percent: Optional[str] = None
    outcome_urls: List[str] = Field(default_factory=list)


class UniversityNetworking(BaseModel):
    alumni_networking_description: Optional[str] = None
    alumni_networking_urls: List[str] = Field(default_factory=list)


class UniversityCredentials(BaseModel):
    credential_examples: List[str] = Field(default_factory=list)
    credential_urls: List[str] = Field(default_factory=list)


class UniversityEngagement(BaseModel):
    engagement_metrics_desc: Optional[str] = None
    engagement_urls: List[str] = Field(default_factory=list)


class UniversityImpact(BaseModel):
    impact_evidence_desc: Optional[str] = None
    impact_urls: List[str] = Field(default_factory=list)


class UniversityItem(BaseModel):
    institutional: Optional[UniversityInstitutional] = None
    outcomes: Optional[UniversityOutcomes] = None
    networking: Optional[UniversityNetworking] = None
    credentials: Optional[UniversityCredentials] = None
    engagement: Optional[UniversityEngagement] = None
    impact: Optional[UniversityImpact] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return (
        "Extract up to three (3) universities from the answer that are claimed to meet the specified career services "
        "excellence criteria. If more than 3 are provided, keep only the first three in order of appearance. If fewer "
        "are provided, include what is available and set missing fields to null or empty lists as appropriate.\n\n"
        "Return a JSON object with a 'universities' array. Each element should have this nested structure:\n"
        "{\n"
        "  'institutional': {\n"
        "    'name': string | null,\n"
        "    'city': string | null,\n"
        "    'state': string | null,\n"
        "    'accreditation_urls': string[] (urls explicitly mentioned for accreditation verification),\n"
        "    'nace_participation_urls': string[] (urls indicating participation in NACE First Destination Survey),\n"
        "    'fds_urls': string[] (urls for first destination survey results or outcomes pages),\n"
        "    'knowledge_rate_percent': string | null (e.g., '72%' or '72.5%'),\n"
        "    'knowledge_rate_urls': string[] (urls where the knowledge rate is reported),\n"
        "    'data_collection_timeline_desc': string | null (how the FDS data collection window is described),\n"
        "    'timeline_urls': string[] (urls supporting the 6-month FDS data collection window),\n"
        "    'institutional_reference_urls': string[] (any general reference urls used to support institutional checks)\n"
        "  },\n"
        "  'outcomes': {\n"
        "    'career_outcome_rate_percent': string | null,\n"
        "    'outcome_urls': string[] (urls supporting the career outcome rate)\n"
        "  },\n"
        "  'networking': {\n"
        "    'alumni_networking_description': string | null,\n"
        "    'alumni_networking_urls': string[] (urls describing the alumni networking platform/services)\n"
        "  },\n"
        "  'credentials': {\n"
        "    'credential_examples': string[] (examples of certifications/credentials),\n"
        "    'credential_urls': string[] (urls supporting those job-relevant certifications/credentials)\n"
        "  },\n"
        "  'engagement': {\n"
        "    'engagement_metrics_desc': string | null,\n"
        "    'engagement_urls': string[] (urls supporting engagement/participation metrics)\n"
        "  },\n"
        "  'impact': {\n"
        "    'impact_evidence_desc': string | null,\n"
        "    'impact_urls': string[] (urls supporting evidence of impact on persistence/retention/employment)\n"
        "  }\n"
        "}\n\n"
        "Important requirements:\n"
        "- Extract only URLs explicitly present in the answer text. Do not invent URLs.\n"
        "- Keep percentages as strings exactly as written (e.g., '68%' or '68.0%').\n"
        "- If a required piece is missing from the answer, return null for strings and an empty array for urls.\n"
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""

def _join_nonempty(values: List[str], sep: str = ", ") -> str:
    return sep.join([v for v in values if v and isinstance(v, str)])

def _dedup_urls(url_lists: List[List[str]]) -> List[str]:
    seen = set()
    result: List[str] = []
    for lst in url_lists:
        for u in lst or []:
            if not isinstance(u, str):
                continue
            u2 = u.strip()
            if not u2:
                continue
            if u2 not in seen:
                seen.add(u2)
                result.append(u2)
    return result


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def _build_institution_verification(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int,
) -> None:
    # Create critical parallel node for institutional verification
    inst_node = evaluator.add_parallel(
        id=f"univ_{idx}_institution_verification",
        desc="Verification of basic institutional requirements and NACE compliance",
        parent=parent_node,
        critical=True
    )

    institutional = uni.institutional or UniversityInstitutional()

    # U.S. Accreditation
    n_usaccr = evaluator.add_leaf(
        id=f"univ_{idx}_us_accreditation",
        desc="University is a U.S. accredited institution",
        parent=inst_node,
        critical=True
    )
    claim_accr = (
        f"The institution '{_safe(institutional.name)}' is a U.S.-accredited institution. "
        f"Accept if the page shows accreditation by a recognized U.S. accrediting agency "
        f"(regional or national) or US Department of Education/CHEA recognition."
    )
    await evaluator.verify(
        claim=claim_accr,
        node=n_usaccr,
        sources=_dedup_urls([institutional.accreditation_urls]),
        additional_instruction="Verify accreditation from an official accreditor, USDE, CHEA, or the institution's accreditation page."
    )

    # NACE Participation
    n_nace = evaluator.add_leaf(
        id=f"univ_{idx}_nace_participation",
        desc="University participates in NACE's First Destination Survey initiative",
        parent=inst_node,
        critical=True
    )
    nace_sources = _dedup_urls([institutional.nace_participation_urls, institutional.fds_urls])
    claim_nace = (
        f"The institution '{_safe(institutional.name)}' participates in NACE's First Destination Survey (FDS) initiative."
    )
    await evaluator.verify(
        claim=claim_nace,
        node=n_nace,
        sources=nace_sources,
        additional_instruction="Look for mentions of 'NACE', 'First Destination Survey (FDS)', or alignment with NACE FDS standards."
    )

    # FDS Published
    n_fds_pub = evaluator.add_leaf(
        id=f"univ_{idx}_fds_published",
        desc="First Destination Survey results are publicly available",
        parent=inst_node,
        critical=True
    )
    claim_fds_pub = (
        f"The institution '{_safe(institutional.name)}' publicly provides First Destination Survey results or outcomes reports."
    )
    await evaluator.verify(
        claim=claim_fds_pub,
        node=n_fds_pub,
        sources=_dedup_urls([institutional.fds_urls]),
        additional_instruction="Accept dedicated outcomes/FDS report pages, dashboards, or PDFs clearly linked from the institution."
    )

    # Knowledge rate threshold (>= 65%)
    n_kr = evaluator.add_leaf(
        id=f"univ_{idx}_knowledge_rate_threshold",
        desc="Knowledge rate meets or exceeds 65% minimum threshold",
        parent=inst_node,
        critical=True
    )
    kr_text = _safe(institutional.knowledge_rate_percent)
    claim_kr = (
        f"The most recent FDS knowledge rate reported is '{kr_text}', and it meets or exceeds 65%."
    )
    await evaluator.verify(
        claim=claim_kr,
        node=n_kr,
        sources=_dedup_urls([institutional.knowledge_rate_urls, institutional.fds_urls]),
        additional_instruction="Confirm the knowledge rate value on the page and judge whether it is >= 65%. Minor rounding differences are acceptable."
    )

    # Data collection timeline: within 6 months of graduation
    n_timeline = evaluator.add_leaf(
        id=f"univ_{idx}_data_collection_timeline",
        desc="Data collected within 6 months of graduation (July 1 - December 31 window)",
        parent=inst_node,
        critical=True
    )
    claim_timeline = (
        f"The FDS data for '{_safe(institutional.name)}' are collected within 6 months of graduation (NACE standard window). "
        f"Description: {_safe(institutional.data_collection_timeline_desc)}"
    )
    await evaluator.verify(
        claim=claim_timeline,
        node=n_timeline,
        sources=_dedup_urls([institutional.timeline_urls, institutional.fds_urls]),
        additional_instruction="Accept if the page states NACE's 6‑month post‑graduation data collection window or equivalent language."
    )

    # Verification reference URL (existence)
    # Treat as existence check that at least one institutional reference URL is provided
    inst_refs_union = _dedup_urls([
        institutional.institutional_reference_urls,
        institutional.accreditation_urls,
        institutional.nace_participation_urls,
        institutional.fds_urls,
        institutional.knowledge_rate_urls,
        institutional.timeline_urls
    ])
    n_ref = evaluator.add_custom_node(
        result=len(inst_refs_union) > 0,
        id=f"univ_{idx}_institution_verification_reference_url",
        desc="Reference URL provided verifying institutional requirements",
        parent=inst_node,
        critical=True
    )


async def _build_career_services_excellence(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int,
) -> None:
    # Create critical parallel node for comprehensive career services quality
    cs_node = evaluator.add_parallel(
        id=f"univ_{idx}_career_services_excellence",
        desc="Comprehensive career services quality and impact metrics",
        parent=parent_node,
        critical=True
    )

    outcomes = uni.outcomes or UniversityOutcomes()
    networking = uni.networking or UniversityNetworking()
    credentials = uni.credentials or UniversityCredentials()
    engagement = uni.engagement or UniversityEngagement()
    impact = uni.impact or UniversityImpact()
    institutional = uni.institutional or UniversityInstitutional()

    # Career_Outcome_Rate (sequential)
    out_node = evaluator.add_sequential(
        id=f"univ_{idx}_career_outcome_rate",
        desc="Career outcome rate reporting",
        parent=cs_node,
        critical=True
    )

    # Rate_Reported (leaf)
    n_rate = evaluator.add_leaf(
        id=f"univ_{idx}_rate_reported",
        desc="Career outcome rate (employed + continuing education + service + military) is reported as a percentage",
        parent=out_node,
        critical=True
    )
    claim_rate = (
        f"The page reports a career outcome rate as a percentage (employed + continuing education + service + military). "
        f"Reported value: '{_safe(outcomes.career_outcome_rate_percent)}'."
    )
    await evaluator.verify(
        claim=claim_rate,
        node=n_rate,
        sources=_dedup_urls([outcomes.outcome_urls, institutional.fds_urls]),
        additional_instruction="Confirm that a composite 'career outcomes rate' percentage is explicitly shown or computable as defined."
    )

    # Outcome_Reference_URL (existence)
    n_rate_ref = evaluator.add_custom_node(
        result=len(_dedup_urls([outcomes.outcome_urls])) > 0,
        id=f"univ_{idx}_outcome_reference_url",
        desc="Reference URL provided for career outcome rate",
        parent=out_node,
        critical=True
    )

    # Alumni_Networking (sequential)
    alum_node = evaluator.add_sequential(
        id=f"univ_{idx}_alumni_networking",
        desc="Alumni career networking infrastructure",
        parent=cs_node,
        critical=True
    )

    # Platform_Description (leaf)
    n_platform = evaluator.add_leaf(
        id=f"univ_{idx}_platform_description",
        desc="Active alumni networking platform, services, or programs are described with documented features",
        parent=alum_node,
        critical=True
    )
    claim_platform = (
        f"The institution provides an active alumni career networking platform/services with documented features and accessibility. "
        f"Description: {_safe(networking.alumni_networking_description)}"
    )
    await evaluator.verify(
        claim=claim_platform,
        node=n_platform,
        sources=_dedup_urls([networking.alumni_networking_urls]),
        additional_instruction="Accept platforms like alumni mentoring portals, Handshake alumni features, alumni networks, or similar with described features and access details."
    )

    # Networking_Reference_URL (existence)
    n_net_ref = evaluator.add_custom_node(
        result=len(_dedup_urls([networking.alumni_networking_urls])) > 0,
        id=f"univ_{idx}_networking_reference_url",
        desc="Reference URL provided for alumni networking services",
        parent=alum_node,
        critical=True
    )

    # Professional_Credentials (sequential)
    cred_node = evaluator.add_sequential(
        id=f"univ_{idx}_professional_credentials",
        desc="Professional certification and credential programs",
        parent=cs_node,
        critical=True
    )

    # Job_Relevant_Programs (leaf)
    n_creds = evaluator.add_leaf(
        id=f"univ_{idx}_job_relevant_programs",
        desc="Job-relevant professional certification or credential programs are offered with specific examples provided",
        parent=cred_node,
        critical=True
    )
    examples_str = _join_nonempty(credentials.credential_examples)
    claim_creds = (
        f"The institution offers job-relevant, industry-recognized certifications/credentials. Examples: {examples_str if examples_str else 'None provided'}."
    )
    await evaluator.verify(
        claim=claim_creds,
        node=n_creds,
        sources=_dedup_urls([credentials.credential_urls]),
        additional_instruction="Look for non-credit certificates, micro-credentials, badges, bootcamps, or credit-bearing certificates aligned to industry credentials."
    )

    # Credentials_Reference_URL (existence)
    n_cred_ref = evaluator.add_custom_node(
        result=len(_dedup_urls([credentials.credential_urls])) > 0,
        id=f"univ_{idx}_credentials_reference_url",
        desc="Reference URL provided for professional certification programs",
        parent=cred_node,
        critical=True
    )

    # Engagement_Metrics (sequential)
    eng_node = evaluator.add_sequential(
        id=f"univ_{idx}_engagement_metrics",
        desc="Career services engagement and participation tracking",
        parent=cs_node,
        critical=True
    )

    # Metrics_Documented (leaf)
    n_metrics = evaluator.add_leaf(
        id=f"univ_{idx}_metrics_documented",
        desc="Career services engagement metrics are publicly documented (e.g., advising participation rates, student usage statistics)",
        parent=eng_node,
        critical=True
    )
    claim_metrics = (
        f"The institution publicly documents career services engagement metrics (e.g., advising appointment participation, usage statistics). "
        f"Description: {_safe(engagement.engagement_metrics_desc)}"
    )
    await evaluator.verify(
        claim=claim_metrics,
        node=n_metrics,
        sources=_dedup_urls([engagement.engagement_urls]),
        additional_instruction="Accept pages showing quantitative engagement metrics, advising appointment counts/rates, event attendance, or similar measurable indicators."
    )

    # Engagement_Reference_URL (existence)
    n_eng_ref = evaluator.add_custom_node(
        result=len(_dedup_urls([engagement.engagement_urls])) > 0,
        id=f"univ_{idx}_engagement_reference_url",
        desc="Reference URL provided for engagement metrics",
        parent=eng_node,
        critical=True
    )

    # Impact_Evidence (sequential)
    imp_node = evaluator.add_sequential(
        id=f"univ_{idx}_impact_evidence",
        desc="Demonstrated career services impact on student outcomes",
        parent=cs_node,
        critical=True
    )

    # Impact_Documentation (leaf)
    n_impact = evaluator.add_leaf(
        id=f"univ_{idx}_impact_documentation",
        desc="Published evidence of career services impact on persistence, retention, or employment outcomes",
        parent=imp_node,
        critical=True
    )
    claim_impact = (
        f"The institution publishes evidence that career services impact student outcomes such as persistence, retention, or employment. "
        f"Evidence summary: {_safe(impact.impact_evidence_desc)}"
    )
    await evaluator.verify(
        claim=claim_impact,
        node=n_impact,
        sources=_dedup_urls([impact.impact_urls]),
        additional_instruction="Accept institutional reports, assessments, or studies attributing improved outcomes (retention, employment, persistence) to career services."
    )

    # Impact_Reference_URL (existence)
    n_imp_ref = evaluator.add_custom_node(
        result=len(_dedup_urls([impact.impact_urls])) > 0,
        id=f"univ_{idx}_impact_reference_url",
        desc="Reference URL provided for impact evidence",
        parent=imp_node,
        critical=True
    )


async def _build_university_tree(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int,
) -> None:
    # Sequential node per university (non-critical at university level to allow partial credit across universities)
    uni_node = evaluator.add_sequential(
        id=f"University_{idx+1}",
        desc=f"{['First','Second','Third'][idx]} university meeting all specified career services criteria",
        parent=parent_node,
        critical=False
    )

    # 1) Institutional verification (critical)
    await _build_institution_verification(evaluator, uni_node, uni, idx+1)

    # 2) Career services excellence (critical)
    await _build_career_services_excellence(evaluator, uni_node, uni, idx+1)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    # Initialize evaluator with PARALLEL root, as universities are independent
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

    # Extract structured information
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Prepare exactly 3 university items (pad with empty if fewer)
    universities: List[UniversityItem] = list(extraction.universities or [])
    while len(universities) < 3:
        universities.append(UniversityItem())
    universities = universities[:3]

    # Build verification subtrees for three universities in parallel branches
    tasks = []
    for i in range(3):
        tasks.append(_build_university_tree(evaluator, root, universities[i], i))

    # Execute building/verifications sequentially here to preserve deterministic order
    # (Alternatively, could use asyncio.gather if desired; verify() calls are already async)
    for t in tasks:
        await t

    # Return final summary
    return evaluator.get_summary()