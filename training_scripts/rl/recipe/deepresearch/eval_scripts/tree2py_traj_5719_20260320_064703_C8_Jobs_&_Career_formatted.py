import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "pa_career_centers_comprehensive_12_features"
TASK_DESCRIPTION = """
Identify 3 universities in Pennsylvania that operate comprehensive career centers meeting professional standards. For each university, verify and provide evidence that its career center offers ALL of the following 12 services/features: (1) One-on-one career advising appointments, (2) Resume and cover letter review services, (3) Mock interview preparation, (4) Career fair programming (in-person and/or virtual), (5) Access to a job and internship posting platform, (6) Career development workshops, (7) Career services for recent alumni (graduates within 1-2 years), (8) A physical career center location with posted operating hours, (9) Professional attire resources (such as a dress closet), (10) Conducts First Destination Survey following NACE standards, (11) Publicly available graduate outcomes data, and (12) Employer partnership programs for recruiting and engagement. For each university and each criterion, provide the specific URL reference that verifies the service/feature exists.
"""


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class UniversityFeatures(BaseModel):
    # Identification
    university_name: Optional[str] = None
    # Optional general URLs explicitly mentioned in the answer (e.g., university or career center homepage)
    university_urls: List[str] = Field(default_factory=list)
    career_center_homepage_urls: List[str] = Field(default_factory=list)

    # 12 feature URL buckets (only URLs explicitly present in the answer)
    career_advising_urls: List[str] = Field(default_factory=list)
    resume_services_urls: List[str] = Field(default_factory=list)
    mock_interview_urls: List[str] = Field(default_factory=list)
    career_fairs_urls: List[str] = Field(default_factory=list)
    job_platform_urls: List[str] = Field(default_factory=list)
    workshops_urls: List[str] = Field(default_factory=list)
    alumni_services_urls: List[str] = Field(default_factory=list)
    location_hours_urls: List[str] = Field(default_factory=list)
    dress_resources_urls: List[str] = Field(default_factory=list)
    fds_urls: List[str] = Field(default_factory=list)
    outcomes_data_urls: List[str] = Field(default_factory=list)
    employer_partnerships_urls: List[str] = Field(default_factory=list)


class PennsylvaniaUniversitiesExtraction(BaseModel):
    universities: List[UniversityFeatures] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities_and_feature_urls() -> str:
    return """
    Extract information about Pennsylvania universities and the specific URLs provided in the answer that support each required career-center feature.

    Output JSON schema:
    {
      "universities": [
        {
          "university_name": string | null,
          "university_urls": string[]  // General or homepage URLs explicitly mentioned in the answer
          "career_center_homepage_urls": string[]  // Career center homepage URLs explicitly mentioned

          "career_advising_urls": string[],        // URLs confirming one-on-one career advising or individual coaching appointments
          "resume_services_urls": string[],        // URLs confirming resume and/or cover letter review services
          "mock_interview_urls": string[],         // URLs confirming mock interview preparation (including tools like Big Interview, InterviewStream)
          "career_fairs_urls": string[],           // URLs confirming career fair programming (in-person and/or virtual; also accept job fair/expo terminology)
          "job_platform_urls": string[],           // URLs confirming access to a job/internship posting platform (e.g., Handshake, Symplicity, 12twenty)
          "workshops_urls": string[],              // URLs confirming career development workshops (also accept events, skill sessions, clinics)
          "alumni_services_urls": string[],        // URLs confirming career services offered to recent alumni (within 1–2 years of graduation)
          "location_hours_urls": string[],         // URLs showing a physical office location/address and posted operating hours for the career center
          "dress_resources_urls": string[],        // URLs confirming professional attire resources (career closet/suit closet/clothing closet/lending)
          "fds_urls": string[],                    // URLs confirming First Destination Survey (FDS) conducted following NACE standards/methodology
          "outcomes_data_urls": string[],          // URLs to publicly posted graduate outcomes data/reports/dashboards
          "employer_partnerships_urls": string[]   // URLs confirming employer partnership/recruiting/engagement programs
        },
        ...
      ]
    }

    Strict extraction rules:
    - Only extract URLs that are explicitly present in the answer (including markdown links). Do not infer or invent URLs.
    - Keep the full URL (with http/https).
    - If a category is mentioned without a URL, leave that category as an empty array.
    - If a university name is missing, set "university_name" to null.
    - Return all universities mentioned; do not deduplicate or merge.

    Goal reminder:
    - We will later verify up to 3 universities. Ensure the extracted URLs are specific to the stated features so they can be used as evidence.
    """


# --------------------------------------------------------------------------- #
# Feature specifications and helper utilities                                 #
# --------------------------------------------------------------------------- #
class FeatureSpec(BaseModel):
    key: str
    title: str
    url_attr: str
    claim_template: str
    additional_instruction: str


def get_feature_specs() -> List[FeatureSpec]:
    return [
        FeatureSpec(
            key="career_advising",
            title="one-on-one career advising appointments",
            url_attr="career_advising_urls",
            claim_template="The career center at {name} provides one-on-one career advising or coaching appointments for students.",
            additional_instruction="Look for 'appointments', '1:1 advising', 'individual advising', 'career coaching', or similar phrasing."
        ),
        FeatureSpec(
            key="resume_services",
            title="resume and cover letter review services",
            url_attr="resume_services_urls",
            claim_template="The career center at {name} offers resume and/or cover letter review services.",
            additional_instruction="Accept services such as 'resume reviews', 'CV critiques', 'cover letter feedback', 'drop-ins' for document review."
        ),
        FeatureSpec(
            key="mock_interview",
            title="mock interview preparation",
            url_attr="mock_interview_urls",
            claim_template="The career center at {name} provides mock interview preparation or practice interviewing.",
            additional_instruction="Evidence may include 'mock interviews', 'practice interviews', or tools like Big Interview/InterviewStream."
        ),
        FeatureSpec(
            key="career_fairs",
            title="career fair programming (in-person and/or virtual)",
            url_attr="career_fairs_urls",
            claim_template="The university hosts career fair programming (in-person and/or virtual).",
            additional_instruction="Accept 'career fair', 'job fair', 'career expo', 'internship fair'; page should clearly indicate such events exist."
        ),
        FeatureSpec(
            key="job_platform",
            title="access to a job and internship posting platform",
            url_attr="job_platform_urls",
            claim_template="Students have access to a job and internship posting platform through the university (e.g., Handshake, Symplicity, 12twenty or similar).",
            additional_instruction="Verify there is an official platform for job/internship postings; recognize names such as Handshake, Symplicity, 12twenty, CareerShift, etc."
        ),
        FeatureSpec(
            key="workshops",
            title="career development workshops",
            url_attr="workshops_urls",
            claim_template="The career center offers career development workshops.",
            additional_instruction="Accept 'workshops', 'seminars', 'events', 'skills sessions', 'clinics' hosted by the career center."
        ),
        FeatureSpec(
            key="alumni_services",
            title="career services for recent alumni (1–2 years after graduation)",
            url_attr="alumni_services_urls",
            claim_template="The career center provides career services to recent alumni, defined as within about 1–2 years of graduation.",
            additional_instruction="Evidence should explicitly reference alumni access or 'recent graduates' eligibility; acceptable if they specify up to 1 or 2 years."
        ),
        FeatureSpec(
            key="location_hours",
            title="a physical career center location with posted operating hours",
            url_attr="location_hours_urls",
            claim_template="The career center has a physical office location and publicly posted operating hours.",
            additional_instruction="The provided URL(s) should collectively indicate an address/location AND posted office hours (can be on one or multiple pages)."
        ),
        FeatureSpec(
            key="dress_resources",
            title="professional attire resources (e.g., a dress/closet/suit closet)",
            url_attr="dress_resources_urls",
            claim_template="The career center provides professional attire resources such as a career closet or suit closet.",
            additional_instruction="Accept 'career closet', 'suit closet', 'clothes closet', 'professional attire lending program', etc."
        ),
        FeatureSpec(
            key="fds",
            title="First Destination Survey (FDS) conducted following NACE standards",
            url_attr="fds_urls",
            claim_template="The university conducts the First Destination Survey following NACE standards or methodology.",
            additional_instruction="Look for explicit references to 'NACE' and 'First Destination Survey' or 'in accordance with NACE standards/methodology'."
        ),
        FeatureSpec(
            key="outcomes_data",
            title="publicly available graduate outcomes data",
            url_attr="outcomes_data_urls",
            claim_template="The university publicly publishes graduate outcomes data.",
            additional_instruction="Accept outcomes dashboards/reports that present aggregated post-graduation outcomes; should be publicly accessible pages."
        ),
        FeatureSpec(
            key="employer_partnerships",
            title="employer partnership programs for recruiting and engagement",
            url_attr="employer_partnerships_urls",
            claim_template="The university maintains employer partnership programs for recruiting and engagement.",
            additional_instruction="Evidence may include 'employer relations', 'recruit at {name}', partnership tiers, corporate relations, on-campus recruiting programs."
        ),
    ]


def safe_university_name(u: UniversityFeatures, fallback: str) -> str:
    nm = (u.university_name or "").strip()
    return nm if nm else fallback


def collect_all_urls(u: UniversityFeatures) -> List[str]:
    # Aggregate every URL field we defined
    url_fields = [
        "university_urls", "career_center_homepage_urls",
        "career_advising_urls", "resume_services_urls", "mock_interview_urls",
        "career_fairs_urls", "job_platform_urls", "workshops_urls",
        "alumni_services_urls", "location_hours_urls", "dress_resources_urls",
        "fds_urls", "outcomes_data_urls", "employer_partnerships_urls",
    ]
    seen = set()
    out: List[str] = []
    for f in url_fields:
        for url in getattr(u, f, []) or []:
            if isinstance(url, str):
                if url not in seen:
                    seen.add(url)
                    out.append(url)
    return out


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    root_node,
    u: UniversityFeatures,
    uni_index: int,
) -> None:
    """
    Build verification sub-tree for a single university and run verifications.
    """
    uni_display = safe_university_name(u, f"University #{uni_index + 1}")
    uni_node = evaluator.add_parallel(
        id=f"university_{uni_index + 1}",
        desc=f"{uni_display} – Comprehensive career center verification across 12 required features",
        parent=root_node,
        critical=False,  # Allow partial credit across different universities at root level
    )

    # -------------------- Pennsylvania location check (critical) --------------------
    # We add a critical existence check and a critical verification leaf.
    all_urls = collect_all_urls(u)
    pa_urls_exist = evaluator.add_custom_node(
        result=len(all_urls) > 0,
        id=f"u{uni_index + 1}_pa_location_urls_present",
        desc=f"{uni_display}: URL(s) provided to verify Pennsylvania location",
        parent=uni_node,
        critical=True
    )

    pa_location_leaf = evaluator.add_leaf(
        id=f"u{uni_index + 1}_pa_location_verified",
        desc=f"{uni_display}: Verified as a Pennsylvania university",
        parent=uni_node,
        critical=True
    )

    pa_claim = f"{uni_display} is a university located in Pennsylvania, United States."
    # We'll verify using any of the available URLs aggregated
    # Note: If no URLs present, leaf will be skipped due to failed critical sibling precondition above.
    await evaluator.verify(
        claim=pa_claim,
        node=pa_location_leaf,
        sources=all_urls,
        additional_instruction="Confirm that the page(s) indicate the institution is in the state of Pennsylvania (PA). City names within PA (e.g., Philadelphia, Pittsburgh, State College, etc.) are acceptable evidence."
    )

    # -------------------- 12 feature checks (each critical) -------------------------
    # For each feature, create a critical parent node with two children:
    #   - existence check (critical custom node)
    #   - verification by URL(s) (critical verification leaf)
    features = get_feature_specs()
    verify_tasks: List[Tuple[str, List[str], Any, Optional[str]]] = []

    for feat in features:
        feat_node = evaluator.add_parallel(
            id=f"u{uni_index + 1}_{feat.key}",
            desc=f"{uni_display}: {feat.title}",
            parent=uni_node,
            critical=True  # All 12 features are mandatory
        )

        # Existence of at least one URL for this feature (critical gating)
        urls: List[str] = getattr(u, feat.url_attr, []) or []
        urls_exist = evaluator.add_custom_node(
            result=len(urls) > 0,
            id=f"u{uni_index + 1}_{feat.key}_urls_present",
            desc=f"{uni_display}: URL(s) provided for '{feat.title}'",
            parent=feat_node,
            critical=True
        )

        # URL-supported verification leaf (critical)
        supported_leaf = evaluator.add_leaf(
            id=f"u{uni_index + 1}_{feat.key}_supported",
            desc=f"{uni_display}: '{feat.title}' is supported by cited URL(s)",
            parent=feat_node,
            critical=True
        )

        # Build claim text, substituting name when available
        name_for_claim = safe_university_name(u, "the university referenced by the provided URL(s)")
        claim_text = feat.claim_template.format(name=name_for_claim)

        # Queue for batch verification (preconditions will be auto-checked by Evaluator.verify)
        verify_tasks.append((
            claim_text,
            urls,
            supported_leaf,
            feat.additional_instruction
        ))

    # Run verifications for feature leaves in parallel for this university
    if verify_tasks:
        await evaluator.batch_verify(verify_tasks)


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
    Evaluate an answer for the Pennsylvania comprehensive career centers task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Universities evaluated independently
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities_and_feature_urls(),
        template_class=PennsylvaniaUniversitiesExtraction,
        extraction_name="extracted_universities_and_feature_urls"
    )

    # Record expected features for transparency
    evaluator.add_custom_info(
        info={
            "required_features": [f.title for f in get_feature_specs()],
            "notes": "All 12 features must be verified with URL evidence for each university. Universities must be in Pennsylvania."
        },
        info_type="rubric_requirements",
        info_name="expected_criteria"
    )

    # Ensure we evaluate exactly 3 universities: take the first 3 if more, pad with empty if fewer
    universities = (extracted.universities or [])[:3]
    while len(universities) < 3:
        universities.append(UniversityFeatures())

    # Build subtrees and verify each university
    for idx, uni in enumerate(universities):
        await verify_university(evaluator, root, uni, idx)

    # Return evaluation summary
    return evaluator.get_summary()