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
TASK_ID = "us_school_admin_hiring"
TASK_DESCRIPTION = """
Identify four large public school districts in the United States that are currently hiring for educational administrator positions (principal, assistant superintendent, or curriculum director level), where each district meets ALL of the following criteria:

1. District Enrollment: The district has a total student enrollment of 10,000 or more students.

2. State Location: The district is located in a state that participates in the NASDTEC Interstate Agreement for educator certification reciprocity.

3. Current Administrative Opening: The district has a currently posted job opening for an educational administrator position at the principal, assistant superintendent, curriculum director, or equivalent leadership level.

4. Position Qualifications: The posted position requires:
   - A master's degree in educational administration, educational leadership, or a related field
   - Minimum teaching or administrative experience (specific years may vary)
   - State administrative certification or eligibility to obtain such certification

5. District Diversity: The district serves a diverse student population, with minority student enrollment of 30% or higher.

6. Professional Development: The district or its state requires continuing professional development hours for administrator license renewal.

For each of the four districts, provide:
- District name and state
- Current student enrollment figure
- Title of the specific administrative position currently open
- Evidence that all six criteria are met
- Reference URLs that verify the enrollment, NASDTEC participation, job posting, position requirements, diversity statistics, and professional development requirements
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistrictItem(BaseModel):
    district_name: Optional[str] = None
    state: Optional[str] = None
    enrollment: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)
    nasdtec_urls: List[str] = Field(default_factory=list)
    job_title: Optional[str] = None
    job_urls: List[str] = Field(default_factory=list)
    diversity_percent: Optional[str] = None
    diversity_urls: List[str] = Field(default_factory=list)
    pd_urls: List[str] = Field(default_factory=list)


class DistrictsExtraction(BaseModel):
    districts: List[DistrictItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract up to the first four (4) distinct public school districts mentioned in the answer that the responder claims meet the criteria. For each district, extract ONLY what is explicitly present in the answer.

    For each district, return an object with:
    - district_name: The district name
    - state: The U.S. state (two-letter or full name as stated)
    - enrollment: The student enrollment figure or phrasing as written
    - enrollment_urls: Array of URLs that the answer cites to verify enrollment (district reports, NCES, etc.)
    - nasdtec_urls: Array of URLs that verify that the state participates in the NASDTEC Interstate Agreement
    - job_title: The exact title of the currently open administrative position
    - job_urls: Array of URLs that point to the current job posting (or district ATS) used to verify the opening and its requirements
    - diversity_percent: The minority student enrollment percentage text if provided
    - diversity_urls: Array of URLs that verify minority enrollment statistics for the district
    - pd_urls: Array of URLs that verify continuing professional development (CPE/PD/CEU/clock hours) requirements for administrator license renewal at the state or district level

    RULES:
    - Do NOT invent any URLs. Extract only URLs that appear in the answer (plain text or markdown links).
    - If a URL category is not provided in the answer, return an empty array for that field.
    - If a textual field is not provided, return null for that field.
    - Keep at most the first 4 districts in the order they appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_name(d: DistrictItem, idx: int) -> str:
    return d.district_name or f"District #{idx}"


def _safe_state(d: DistrictItem) -> str:
    return d.state or "the state"


def _plural_or_none(urls: List[str]) -> Optional[List[str]]:
    return urls if urls else None


# --------------------------------------------------------------------------- #
# Verification builder for one district                                       #
# --------------------------------------------------------------------------- #
async def verify_one_district(
    evaluator: Evaluator,
    root_parent,
    d: DistrictItem,
    ordinal_index: int,  # 1-based index to match rubric IDs
) -> None:
    district_label = _safe_name(d, ordinal_index)
    state_label = _safe_state(d)

    # Top-level node for this district (non-critical to allow partial credit across districts)
    district_node = evaluator.add_parallel(
        id=f"district_{ordinal_index}",
        desc=[
            "First", "Second", "Third", "Fourth"
        ][ordinal_index - 1] + " qualifying district with open administrative position",
        parent=root_parent,
        critical=False,
    )

    # ---------------- Enrollment block (critical) ----------------
    enroll_node = evaluator.add_parallel(
        id=f"district_{ordinal_index}_enrollment",
        desc="District enrollment meets large district threshold",
        parent=district_node,
        critical=True,
    )
    # Existence of reference URL(s)
    evaluator.add_custom_node(
        result=bool(d.enrollment_urls),
        id=f"district_{ordinal_index}_enrollment_reference",
        desc="URL provided verifies the enrollment figure",
        parent=enroll_node,
        critical=True,
    )
    # Verification: >= 10,000
    n_enroll_leaf = evaluator.add_leaf(
        id=f"district_{ordinal_index}_enrollment_verification",
        desc="District has enrollment of 10,000 or more students",
        parent=enroll_node,
        critical=True,
    )
    enroll_claim = (
        f"According to the provided sources, the public school district '{district_label}' in {state_label} "
        f"has a total student enrollment of 10,000 or more students."
    )

    # ---------------- Location / NASDTEC block (critical) ----------------
    loc_node = evaluator.add_parallel(
        id=f"district_{ordinal_index}_location",
        desc="District location meets state participation requirements",
        parent=district_node,
        critical=True,
    )
    # Existence of NASDTEC reference URL(s)
    evaluator.add_custom_node(
        result=bool(d.nasdtec_urls),
        id=f"district_{ordinal_index}_location_reference",
        desc="URL provided verifies state NASDTEC participation",
        parent=loc_node,
        critical=True,
    )
    # Verification: state participates in NASDTEC
    n_nasdtec_leaf = evaluator.add_leaf(
        id=f"district_{ordinal_index}_state_nasdtec",
        desc="District is located in a state participating in NASDTEC Interstate Agreement",
        parent=loc_node,
        critical=True,
    )
    nasdtec_claim = (
        f"The U.S. state of {state_label} participates in the NASDTEC Interstate Agreement for educator "
        f"certification reciprocity."
    )

    # ---------------- Position block (critical) ----------------
    pos_node = evaluator.add_parallel(
        id=f"district_{ordinal_index}_position",
        desc="District has current opening for qualifying administrative position",
        parent=district_node,
        critical=True,
    )

    # Position type sub-block (critical)
    pos_type_node = evaluator.add_parallel(
        id=f"district_{ordinal_index}_position_type",
        desc="Position is at principal, assistant superintendent, curriculum director, or equivalent administrative level",
        parent=pos_node,
        critical=True,
    )
    # Verify: position is correct leadership level
    n_pos_level_leaf = evaluator.add_leaf(
        id=f"district_{ordinal_index}_position_level_check",
        desc="Position title indicates administrative leadership role",
        parent=pos_type_node,
        critical=True,
    )
    role_title = d.job_title or "the position"
    pos_level_claim = (
        f"The job title '{role_title}' is an educational administrator leadership role at the level of "
        f"principal, assistant/associate/deputy superintendent, curriculum director, or an equivalent director-level "
        f"leadership role (e.g., Director of Curriculum & Instruction / Teaching & Learning), not a teacher, "
        f"coordinator, coach, or specialist role."
    )
    # Verify: URL shows current job posting (open/active)
    n_pos_ref_leaf = evaluator.add_leaf(
        id=f"district_{ordinal_index}_position_reference",
        desc="URL provided shows current job posting",
        parent=pos_type_node,
        critical=True,
    )
    pos_ref_claim = (
        f"The provided job posting URL(s) show a current/active job posting (accepting applications or open) "
        f"for '{role_title}' in {district_label}."
    )

    # Position qualifications sub-block (critical)
    pos_qual_node = evaluator.add_parallel(
        id=f"district_{ordinal_index}_position_qualifications",
        desc="Position qualifications meet standard administrative requirements",
        parent=pos_node,
        critical=True,
    )
    # Existence of qualification reference URL(s) (typically same job URLs)
    evaluator.add_custom_node(
        result=bool(d.job_urls),
        id=f"district_{ordinal_index}_qualifications_reference",
        desc="URL provided verifies position qualification requirements",
        parent=pos_qual_node,
        critical=True,
    )
    # Master's degree requirement
    n_masters_leaf = evaluator.add_leaf(
        id=f"district_{ordinal_index}_masters_requirement",
        desc="Position requires master's degree in educational administration or related field",
        parent=pos_qual_node,
        critical=True,
    )
    masters_claim = (
        "The job posting explicitly requires a master's degree in educational administration, "
        "educational leadership, or a closely related field (e.g., M.Ed., M.A., M.S. in leadership/administration)."
    )
    # Experience requirement
    n_exp_leaf = evaluator.add_leaf(
        id=f"district_{ordinal_index}_experience_requirement",
        desc="Position requires minimum teaching or administrative experience",
        parent=pos_qual_node,
        critical=True,
    )
    exp_claim = (
        "The job posting requires prior teaching and/or administrative leadership experience, with a specified minimum "
        "number of years or a clear statement that prior experience is required."
    )
    # Certification requirement
    n_cert_leaf = evaluator.add_leaf(
        id=f"district_{ordinal_index}_certification_requirement",
        desc="Position requires or leads to state administrative certification",
        parent=pos_qual_node,
        critical=True,
    )
    cert_claim = (
        "The job posting requires either current state administrative certification (e.g., principal/administrator "
        "license) or eligibility to obtain such certification."
    )

    # ---------------- District characteristics block (critical) ----------------
    char_node = evaluator.add_parallel(
        id=f"district_{ordinal_index}_characteristics",
        desc="District demonstrates specific operational characteristics",
        parent=district_node,
        critical=True,
    )

    # Diversity sub-block (critical)
    div_node = evaluator.add_parallel(
        id=f"district_{ordinal_index}_diversity",
        desc="District serves diverse student population (minority enrollment ≥30%)",
        parent=char_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(d.diversity_urls),
        id=f"district_{ordinal_index}_diversity_reference",
        desc="URL provided verifies diversity statistics",
        parent=div_node,
        critical=True,
    )
    n_div_leaf = evaluator.add_leaf(
        id=f"district_{ordinal_index}_diversity_verification",
        desc="District minority enrollment percentage meets or exceeds 30%",
        parent=div_node,
        critical=True,
    )
    div_claim = (
        f"The minority student enrollment for '{district_label}' is at least 30 percent."
    )

    # Professional development / license renewal sub-block (critical)
    pd_node = evaluator.add_parallel(
        id=f"district_{ordinal_index}_professional_development",
        desc="District or state requires continuing professional development for administrators",
        parent=char_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=bool(d.pd_urls),
        id=f"district_{ordinal_index}_pd_reference",
        desc="URL provided verifies professional development requirements",
        parent=pd_node,
        critical=True,
    )
    n_pd_leaf = evaluator.add_leaf(
        id=f"district_{ordinal_index}_pd_requirement_verification",
        desc="State or district has continuing professional development requirement for administrator license renewal",
        parent=pd_node,
        critical=True,
    )
    pd_claim = (
        f"The state of {state_label} or the district requires continuing professional development/education "
        f"hours (e.g., PD/CPE/CEUs/clock hours/PDPs/PL units) for administrator license/endorsement renewal."
    )

    # ---------------- Batch verifications ----------------
    verifications: List[tuple[str, Optional[List[str]], Any, Optional[str]]] = [
        # Enrollment threshold
        (
            enroll_claim,
            _plural_or_none(d.enrollment_urls),
            n_enroll_leaf,
            "Accept reasonable phrasing like 'over', 'approximately', or rounded values. The page(s) must clearly "
            "indicate a total district enrollment of at least 10,000 students.",
        ),
        # NASDTEC
        (
            nasdtec_claim,
            _plural_or_none(d.nasdtec_urls),
            n_nasdtec_leaf,
            "Confirm the state participates in the NASDTEC Interstate Agreement (educator license reciprocity). "
            "Allow official state DOE pages or NASDTEC references. If the provided pages are irrelevant or don't "
            "confirm participation, mark as not supported.",
        ),
        # Position type (leadership level)
        (
            pos_level_claim,
            _plural_or_none(d.job_urls),
            n_pos_level_leaf,
            "Treat as qualifying only if the role is principal (any school level), assistant/associate/deputy "
            "superintendent, curriculum director (or equivalent director-level title like Director of Curriculum & "
            "Instruction, Director of Teaching & Learning, Director of Instruction). Do NOT accept teacher, "
            "coordinator, coach, or specialist-only roles. Assistant principal should NOT be counted here.",
        ),
        # Position reference (current posting)
        (
            pos_ref_claim,
            _plural_or_none(d.job_urls),
            n_pos_ref_leaf,
            "The page should present a current/active posting (e.g., Apply button, Open until filled, posting date "
            "recent, or not marked 'closed/expired'). If clearly closed or archived, mark as not supported.",
        ),
        # Master's requirement
        (
            masters_claim,
            _plural_or_none(d.job_urls),
            n_masters_leaf,
            "Look for explicit mention of a master's degree requirement in educational leadership/administration or a "
            "closely related field. Synonyms (M.Ed., MA/MS in leadership/administration) are acceptable.",
        ),
        # Experience requirement
        (
            exp_claim,
            _plural_or_none(d.job_urls),
            n_exp_leaf,
            "Verify that the posting requires prior teaching and/or administrative experience. A minimum year count is "
            "preferred but not strictly required if the language clearly requires prior experience.",
        ),
        # Certification requirement
        (
            cert_claim,
            _plural_or_none(d.job_urls),
            n_cert_leaf,
            "Verify a requirement for state administrative certification (e.g., principal/administrator license) or "
            "explicit eligibility to obtain it. Equivalent phrasing is acceptable.",
        ),
        # Diversity >= 30%
        (
            div_claim,
            _plural_or_none(d.diversity_urls),
            n_div_leaf,
            "Accept phrasing like 'students of color', 'non-white', or aggregated minority enrollment. The combined "
            "minority share must be at least 30%.",
        ),
        # PD/CE for admin renewal
        (
            pd_claim,
            _plural_or_none(d.pd_urls),
            n_pd_leaf,
            "Look for administrator license renewal requirements that reference PD/CPE/CEU/clock hours/PDPs/PL units. "
            "If the page states renewal PD requirements apply to all licensed educators including administrators, that "
            "is acceptable.",
        ),
    ]

    await evaluator.batch_verify(verifications)


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
    # Initialize evaluator (root is non-critical; JSON root marked critical would force all children critical)
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
        default_model=model,
    )

    # Extract structured district info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictsExtraction,
        extraction_name="districts_extraction",
    )

    # Keep only the first 4 districts; pad with empty if fewer
    districts: List[DistrictItem] = list(extracted.districts[:4])
    while len(districts) < 4:
        districts.append(DistrictItem())

    # Build the tree according to rubric for four districts
    tasks = []
    for i in range(4):
        # Create district container node (matches rubric; non-critical at district level)
        # The children (enrollment, location, position, characteristics) will be critical.
        # The verify_one_district function will attach all subnodes using the prescribed IDs.
        tasks.append(verify_one_district(evaluator, root, districts[i], i + 1))

    # Run verifications (can run districts in parallel)
    await asyncio.gather(*tasks)

    # Record a brief custom info summary
    evaluator.add_custom_info(
        {
            f"district_{i+1}": {
                "name": districts[i].district_name,
                "state": districts[i].state,
                "job_title": districts[i].job_title,
                "enrollment": districts[i].enrollment,
                "diversity_percent": districts[i].diversity_percent,
                "counts": {
                    "enrollment_urls": len(districts[i].enrollment_urls or []),
                    "nasdtec_urls": len(districts[i].nasdtec_urls or []),
                    "job_urls": len(districts[i].job_urls or []),
                    "diversity_urls": len(districts[i].diversity_urls or []),
                    "pd_urls": len(districts[i].pd_urls or []),
                },
            }
            for i in range(4)
        },
        info_type="extracted_overview",
        info_name="extracted_overview",
    )

    return evaluator.get_summary()