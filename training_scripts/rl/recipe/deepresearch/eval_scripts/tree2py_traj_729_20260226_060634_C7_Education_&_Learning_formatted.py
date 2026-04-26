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
TASK_ID = "ma_state_university_med_selection"
TASK_DESCRIPTION = """
A student who completed an associate degree at a Massachusetts community college is seeking to pursue a Master of Education degree at a Massachusetts state university (excluding the UMass system). The student requires an institution that offers comprehensive graduate education programs with online options, including specializations in Educational Leadership, Elementary Education, and Secondary Education. Additionally, the student prefers a university located in Central Massachusetts with substantial enrollment and established transfer pathways from community colleges.

Which Massachusetts state university best meets all of the following requirements:
1. Is part of the Massachusetts state university system (not the UMass system)
2. Is accredited by the New England Commission of Higher Education (NECHE)
3. Offers Master of Education (M.Ed.) degree programs
4. Provides online or hybrid Master of Education program options
5. Has articulation agreements with Massachusetts community colleges
6. Offers M.Ed. specialization in Educational Leadership or Administration
7. Offers M.Ed. specialization in Elementary Education
8. Offers M.Ed. specialization in Secondary Education
9. Has a dedicated graduate school or graduate studies division
10. Offers graduate certificate programs in education
11. Has total student enrollment exceeding 4,000 students
12. Is located in Central Massachusetts
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    """Structured extraction of the identified university and per-criterion sources from the answer."""
    university_name: Optional[str] = None

    # Per-criterion source URL lists (explicitly cited in the answer)
    state_university_system_urls: List[str] = Field(default_factory=list)
    neche_accreditation_urls: List[str] = Field(default_factory=list)
    med_program_urls: List[str] = Field(default_factory=list)
    online_med_urls: List[str] = Field(default_factory=list)
    articulation_urls: List[str] = Field(default_factory=list)
    leadership_urls: List[str] = Field(default_factory=list)
    elementary_urls: List[str] = Field(default_factory=list)
    secondary_urls: List[str] = Field(default_factory=list)
    graduate_division_urls: List[str] = Field(default_factory=list)
    graduate_cert_urls: List[str] = Field(default_factory=list)
    enrollment_urls: List[str] = Field(default_factory=list)
    location_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_selection() -> str:
    return """
    You must extract the single Massachusetts state university (excluding UMass system) that the answer identifies as the best fit for the described requirements, along with the specific source URLs the answer cites to support each criterion.

    Extract exactly the following fields:
    - university_name: The name of the recommended university (e.g., "Worcester State University"). If multiple are listed, choose the one the answer recommends as best; if ambiguous, pick the first explicitly recommended. If none is clear, return null.
    - state_university_system_urls: URLs cited that show the institution is part of the Massachusetts state university system (NOT UMass). Accept official state/university pages or lists.
    - neche_accreditation_urls: URLs cited that show the institution is accredited by NECHE.
    - med_program_urls: URLs cited that show the institution offers Master of Education (M.Ed.) degree programs.
    - online_med_urls: URLs cited that show there are online or hybrid M.Ed. options.
    - articulation_urls: URLs cited that show articulation agreements or established transfer pathways with Massachusetts community colleges (e.g., MassTransfer or university transfer pages).
    - leadership_urls: URLs cited that show an M.Ed. specialization/program in Educational Leadership or Administration.
    - elementary_urls: URLs cited that show an M.Ed. specialization/program in Elementary Education.
    - secondary_urls: URLs cited that show an M.Ed. specialization/program in Secondary Education.
    - graduate_division_urls: URLs cited that show a dedicated Graduate School or Graduate Studies division/office.
    - graduate_cert_urls: URLs cited that show graduate certificate programs in education or education-related fields.
    - enrollment_urls: URLs cited that show total student enrollment exceeds 4,000 (e.g., university facts page, IPEDS, Common Data Set).
    - location_urls: URLs cited that show the university is located in the Central Massachusetts region (e.g., campus/location page or reputable regional designation).

    IMPORTANT:
    - Extract only URLs explicitly present in the answer (including plain links or markdown links). Do not invent any URLs.
    - If a field lacks cited URLs, return an empty array for that field.
    - If a URL appears without protocol, prepend http://.
    - Return a single JSON object with all fields listed above.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def _add_criterion_check(
    evaluator: Evaluator,
    parent_node,
    university_name: Optional[str],
    criterion_id: str,
    criterion_desc: str,
    claim_text: str,
    urls: List[str],
    additional_instruction: str
) -> None:
    """
    Add a two-step sequential check for a single criterion:
    1) Sources existence gate (critical).
    2) Evidence-backed verification leaf (critical).
    """
    # Step 1: Sources existence gate (critical)
    evaluator.add_custom_node(
        result=_has_sources(urls),
        id=f"{criterion_id}_sources_provided",
        desc=f"Sources provided for: {criterion_desc}",
        parent=parent_node,
        critical=True
    )

    # Step 2: Evidence-backed verification leaf (critical)
    leaf = evaluator.add_leaf(
        id=criterion_id,
        desc=criterion_desc,
        parent=parent_node,
        critical=True
    )

    # Build the claim incorporating the university name when available
    uni = university_name or "the university"
    claim = claim_text.replace("{UNIVERSITY}", uni)

    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,  # May be single or multiple URLs; verification routes accordingly
        additional_instruction=additional_instruction
    )


async def verify_university_criteria(
    evaluator: Evaluator,
    root_node,
    extracted: UniversityExtraction
) -> None:
    """
    Build and execute the verification tree based on the rubric with strict (critical) checks.
    Uses a sequential aggregation to gate later checks if earlier mandatory requirements fail.
    """
    # University Identification main node (critical, sequential aggregation)
    main = evaluator.add_sequential(
        id="University_Identification",
        desc=("Identify the Massachusetts state university (excluding UMass system) that meets all specified criteria for "
              "graduate education programs suitable for community college transfer students"),
        parent=root_node,
        critical=True
    )

    # Gate: university name must be provided
    evaluator.add_custom_node(
        result=(extracted.university_name is not None and str(extracted.university_name).strip() != ""),
        id="university_name_provided",
        desc="Recommended university name is provided in the answer",
        parent=main,
        critical=True
    )

    # 1) State University System (not UMass)
    await _add_criterion_check(
        evaluator=evaluator,
        parent_node=main,
        university_name=extracted.university_name,
        criterion_id="State_University_System",
        criterion_desc=("The identified institution is part of the Massachusetts state university system "
                        "(one of the nine state universities, not part of the UMass system)"),
        claim_text="{UNIVERSITY} is one of the Massachusetts state universities (not part of the UMass system).",
        urls=extracted.state_university_system_urls,
        additional_instruction=("Use official state or university sources to confirm membership in the Massachusetts "
                                "state university system. Explicitly ensure it is not part of the UMass system.")
    )

    # 2) NECHE Accreditation
    await _add_criterion_check(
        evaluator, main, extracted.university_name,
        "NECHE_Accreditation",
        "The institution is accredited by the New England Commission of Higher Education (NECHE)",
        "{UNIVERSITY} is accredited by the New England Commission of Higher Education (NECHE).",
        extracted.neche_accreditation_urls,
        "Confirm NECHE accreditation via the NECHE directory or the university accreditation page."
    )

    # 3) Master of Education (M.Ed.) programs
    await _add_criterion_check(
        evaluator, main, extracted.university_name,
        "Master_Education_Degree",
        "The institution offers Master of Education (M.Ed.) degree programs",
        "{UNIVERSITY} offers Master of Education (M.Ed.) degree programs.",
        extracted.med_program_urls,
        "Verify the presence of M.Ed. programs on official program or catalog pages."
    )

    # 4) Online or hybrid M.Ed. options
    await _add_criterion_check(
        evaluator, main, extracted.university_name,
        "Online_MEd_Programs",
        "The institution provides online or hybrid options for Master of Education programs",
        "{UNIVERSITY} provides online or hybrid options for M.Ed. programs.",
        extracted.online_med_urls,
        "Confirm explicit online or hybrid delivery options for M.Ed. programs on official program pages."
    )

    # 5) Articulation agreements with MA community colleges
    await _add_criterion_check(
        evaluator, main, extracted.university_name,
        "Community_College_Articulation",
        "The institution has established articulation agreements with Massachusetts community colleges",
        "{UNIVERSITY} has established articulation agreements or transfer pathways with Massachusetts community colleges.",
        extracted.articulation_urls,
        "Use MassTransfer, transfer/agreements pages, or official sources that explicitly show articulation agreements."
    )

    # 6) Educational Leadership or Administration (M.Ed.)
    await _add_criterion_check(
        evaluator, main, extracted.university_name,
        "Educational_Leadership_Program",
        "The institution offers an M.Ed. specialization or program in Educational Leadership or Administration",
        "{UNIVERSITY} offers an M.Ed. specialization or program in Educational Leadership or Administration.",
        extracted.leadership_urls,
        "Verify on official program/curriculum pages that Educational Leadership or Administration is offered at the M.Ed. level."
    )

    # 7) Elementary Education (M.Ed.)
    await _add_criterion_check(
        evaluator, main, extracted.university_name,
        "Elementary_Education_Program",
        "The institution offers an M.Ed. specialization or program in Elementary Education",
        "{UNIVERSITY} offers an M.Ed. specialization or program in Elementary Education.",
        extracted.elementary_urls,
        "Confirm Elementary Education at the M.Ed. level via official program pages."
    )

    # 8) Secondary Education (M.Ed.)
    await _add_criterion_check(
        evaluator, main, extracted.university_name,
        "Secondary_Education_Program",
        "The institution offers an M.Ed. specialization or program in Secondary Education",
        "{UNIVERSITY} offers an M.Ed. specialization or program in Secondary Education.",
        extracted.secondary_urls,
        "Confirm Secondary Education at the M.Ed. level via official program or catalog pages."
    )

    # 9) Dedicated Graduate Studies division/office
    await _add_criterion_check(
        evaluator, main, extracted.university_name,
        "Graduate_Studies_Division",
        "The institution has a dedicated Graduate School, Graduate Studies office, or graduate division",
        "{UNIVERSITY} has a dedicated Graduate School, Graduate Studies office, or graduate division.",
        extracted.graduate_division_urls,
        "Verify the presence of an official graduate school/graduate studies administrative unit."
    )

    # 10) Graduate certificates in education
    await _add_criterion_check(
        evaluator, main, extracted.university_name,
        "Graduate_Certificates",
        "The institution offers graduate certificate programs in education-related fields",
        "{UNIVERSITY} offers graduate certificate programs in education or education-related fields.",
        extracted.graduate_cert_urls,
        "Confirm graduate certificates in education via official program or catalog pages."
    )

    # 11) Enrollment exceeding 4,000 students
    await _add_criterion_check(
        evaluator, main, extracted.university_name,
        "Enrollment_Size",
        "The institution has total student enrollment exceeding 4,000 students",
        "{UNIVERSITY} has total student enrollment exceeding 4,000 students.",
        extracted.enrollment_urls,
        "Use official facts pages, IPEDS, or credible sources. Allow reasonable rounding differences."
    )

    # 12) Central Massachusetts location
    await _add_criterion_check(
        evaluator, main, extracted.university_name,
        "Central_Massachusetts_Location",
        "The institution is located in the Central Massachusetts region",
        "{UNIVERSITY} is located in the Central Massachusetts region.",
        extracted.location_urls,
        "Confirm campus location and region; allow reasonable regional designation evidence (e.g., Worcester area)."
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
    Evaluate the agent's answer for the Massachusetts state university M.Ed. selection task.
    Returns the evaluator summary with the verification tree and final score.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root non-critical aggregator; detailed logic in child nodes
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

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_university_selection(),
        template_class=UniversityExtraction,
        extraction_name="university_selection_and_sources",
    )

    # Build verification tree and run checks
    await verify_university_criteria(evaluator, root, extracted)

    # Return standard summary
    return evaluator.get_summary()