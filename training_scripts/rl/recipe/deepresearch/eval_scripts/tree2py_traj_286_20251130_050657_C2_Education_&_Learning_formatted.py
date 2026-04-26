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
TASK_ID = "nj_big_ten_university_2025"
TASK_DESCRIPTION = """
Identify the university that meets all of the following criteria: (1) Is a current member of the Big Ten Conference as of 2025, (2) Is located in the state of New Jersey, (3) Offers undergraduate degree programs in engineering, (4) Offers more than 130 different bachelor's degree programs, (5) Has NCAA Division I athletic programs, and (6) Has undergraduate admissions that do not require standardized test scores (test-optional or test-free) for most applicants. Provide the full official name of the university and reference URLs that confirm each of these criteria.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityAnswer(BaseModel):
    """
    Structured information extracted from the answer.

    Notes for extractor:
    - primary_university_name: The single university the answer claims satisfies all criteria.
    - official_full_name: The full official university name as provided in the answer (e.g., "Rutgers, The State University of New Jersey").
    - other_candidate_universities: Any additional universities the answer proposes as candidates that meet the criteria.
    - The six URL lists should contain only URLs explicitly present in the answer for the corresponding criterion.
    """
    primary_university_name: Optional[str] = None
    official_full_name: Optional[str] = None
    other_candidate_universities: List[str] = Field(default_factory=list)

    urls_big_ten: List[str] = Field(default_factory=list)
    urls_location: List[str] = Field(default_factory=list)
    urls_engineering: List[str] = Field(default_factory=list)
    urls_degree_count: List[str] = Field(default_factory=list)
    urls_ncaa_div1: List[str] = Field(default_factory=list)
    urls_test_optional: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university() -> str:
    return """
    Extract the single university identified by the answer as meeting ALL the specified criteria. Focus on the university the answer presents as the final/primary candidate. Do not include incidental mentions unrelated to the final selection.

    Required fields:
    1) primary_university_name: The one university the answer claims satisfies all criteria. If multiple final candidates are proposed, put one as primary and list the others under other_candidate_universities.
    2) official_full_name: The full official name of the university as written in the answer (for example: “Rutgers, The State University of New Jersey”). If the answer does not explicitly provide a full official long-form name, set this to null.
    3) other_candidate_universities: Any additional universities the answer proposes as also meeting the criteria (if the answer proposes multiple candidates). Exclude background mentions that are clearly not presented as final candidates.

    Also extract the URLs provided in the answer that support each criterion (only include URLs explicitly shown in the answer text; accept plain URLs or markdown links):
    - urls_big_ten: URLs supporting Big Ten membership as of 2025.
    - urls_location: URLs supporting that the university is in New Jersey.
    - urls_engineering: URLs supporting availability of undergraduate engineering programs.
    - urls_degree_count: URLs supporting that the university offers more than 130 distinct bachelor's degree programs (accept synonyms like “majors,” “undergraduate programs of study,” etc., only if explicitly stated).
    - urls_ncaa_div1: URLs supporting NCAA Division I athletics at the university.
    - urls_test_optional: URLs supporting that undergraduate admissions do not require standardized tests for most applicants (test-optional or test-free policy); if policy scope is restricted, the URL must still indicate “most applicants.”

    Rules for URL extraction:
    - Only include URLs explicitly present in the answer; do not infer or invent.
    - Extract full URLs; if missing protocol, prepend http:// as needed.
    - If a single URL plausibly supports multiple criteria, include it in all relevant URL lists.

    If any required field cannot be found, return null or an empty list accordingly.
    """


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify_criteria(
    evaluator: Evaluator,
    parent_node,
    extracted: UniversityAnswer,
) -> None:
    """
    Build the 'MeetsAllCriteria' subtree and run verifications against provided URLs.
    All nodes here are critical as per rubric.
    """
    uni_name = extracted.official_full_name or extracted.primary_university_name or "the university"

    meets_all_node = evaluator.add_parallel(
        id="MeetsAllCriteria",
        desc="University satisfies all six required criteria",
        parent=parent_node,
        critical=True
    )

    # Create leaf nodes for each criterion
    node_big_ten = evaluator.add_leaf(
        id="BigTenMembership",
        desc="University is a current member of the Big Ten Conference as of 2025",
        parent=meets_all_node,
        critical=True
    )

    node_location = evaluator.add_leaf(
        id="NewJerseyLocation",
        desc="University is located in the state of New Jersey",
        parent=meets_all_node,
        critical=True
    )

    node_engineering = evaluator.add_leaf(
        id="EngineeringPrograms",
        desc="University offers undergraduate degree programs in engineering",
        parent=meets_all_node,
        critical=True
    )

    node_degree_count = evaluator.add_leaf(
        id="BachelorDegreeCountOver130",
        desc="University offers more than 130 different bachelor's degree programs",
        parent=meets_all_node,
        critical=True
    )

    node_ncaa = evaluator.add_leaf(
        id="NCAADivisionI",
        desc="University has NCAA Division I athletic programs",
        parent=meets_all_node,
        critical=True
    )

    node_test_optional = evaluator.add_leaf(
        id="TestOptionalOrTestFreeMostApplicants",
        desc="University undergraduate admissions do not require standardized test scores (test-optional or test-free) for most applicants",
        parent=meets_all_node,
        critical=True
    )

    # Prepare claims
    claim_big_ten = f"{uni_name} is a current member of the Big Ten Conference as of 2025."
    claim_location = f"{uni_name} is located in the U.S. state of New Jersey."
    claim_engineering = f"{uni_name} offers undergraduate degree programs in engineering."
    claim_degree_count = f"{uni_name} offers more than 130 distinct bachelor's degree programs."
    claim_ncaa = f"{uni_name} fields NCAA Division I athletic programs."
    claim_test_optional = f"{uni_name}'s undergraduate admissions do not require standardized test scores (test-optional or test-free) for most applicants."

    # Batch verify (parallel)
    claims_and_sources = [
        (
            claim_big_ten,
            extracted.urls_big_ten,
            node_big_ten,
            "Verify the page explicitly indicates membership in the Big Ten Conference as of 2025 (including recent conference realignment, if applicable)."
        ),
        (
            claim_location,
            extracted.urls_location,
            node_location,
            "Confirm the university is in New Jersey (state-level location). Accept official pages or authoritative listings."
        ),
        (
            claim_engineering,
            extracted.urls_engineering,
            node_engineering,
            "Confirm that the university offers undergraduate engineering degree programs (e.g., a School/College of Engineering with bachelor's degrees)."
        ),
        (
            claim_degree_count,
            extracted.urls_degree_count,
            node_degree_count,
            "Confirm that the number of distinct bachelor's degree programs exceeds 130. Accept synonyms like 'majors' or 'undergraduate programs' only if the count clearly exceeds 130."
        ),
        (
            claim_ncaa,
            extracted.urls_ncaa_div1,
            node_ncaa,
            "Confirm that the university competes in NCAA Division I athletics (institution-level)."
        ),
        (
            claim_test_optional,
            extracted.urls_test_optional,
            node_test_optional,
            "Confirm that standardized test scores (SAT/ACT) are not required for most undergraduate applicants (test-optional or test-free policy). If policy scope is limited, ensure it still covers the majority of applicants."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


def build_reference_urls_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: UniversityAnswer,
) -> None:
    """
    Build the 'ReferenceURLsProvided' subtree with custom existence checks for each criterion.
    All nodes here are critical as per rubric.
    """
    ref_node = evaluator.add_parallel(
        id="ReferenceURLsProvided",
        desc="Answer provides reference URL(s) that confirm each required criterion",
        parent=parent_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.urls_big_ten),
        id="URLConfirmsBigTenMembership",
        desc="Provides at least one URL that supports Big Ten membership as of 2025",
        parent=ref_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.urls_location),
        id="URLConfirmsNewJerseyLocation",
        desc="Provides at least one URL that supports New Jersey location",
        parent=ref_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.urls_engineering),
        id="URLConfirmsEngineeringPrograms",
        desc="Provides at least one URL that supports availability of undergraduate engineering programs",
        parent=ref_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.urls_degree_count),
        id="URLConfirmsBachelorDegreeCountOver130",
        desc="Provides at least one URL that supports offering >130 bachelor's degree programs",
        parent=ref_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.urls_ncaa_div1),
        id="URLConfirmsNCAADivisionI",
        desc="Provides at least one URL that supports NCAA Division I athletics",
        parent=ref_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(extracted.urls_test_optional),
        id="URLConfirmsTestOptionalPolicy",
        desc="Provides at least one URL that supports test-optional/test-free policy for most applicants",
        parent=ref_node,
        critical=True
    )


def build_single_uni_and_name_checks(
    evaluator: Evaluator,
    parent_node,
    extracted: UniversityAnswer,
) -> None:
    """
    Build the 'SingleUniversityProvided' and 'OfficialFullNameProvided' leaf checks.
    Both are critical.
    """
    # Single university provided (no multiple candidates)
    exactly_one = (
        extracted.primary_university_name is not None
        and extracted.primary_university_name.strip() != ""
        and len(extracted.other_candidate_universities) == 0
    )
    evaluator.add_custom_node(
        result=exactly_one,
        id="SingleUniversityProvided",
        desc="Answer identifies exactly one university (not multiple candidates)",
        parent=parent_node,
        critical=True
    )

    # Official full name provided in the answer text
    official_name_provided = (
        extracted.official_full_name is not None
        and extracted.official_full_name.strip() != ""
    )
    evaluator.add_custom_node(
        result=official_name_provided,
        id="OfficialFullNameProvided",
        desc="Answer provides the university’s full official name",
        parent=parent_node,
        critical=True
    )


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the New Jersey Big Ten University (2025) identification task.
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_university(),
        template_class=UniversityAnswer,
        extraction_name="university_selection"
    )

    # UniversityIdentification node (critical, parallel as per rubric)
    uni_id_node = evaluator.add_parallel(
        id="UniversityIdentification",
        desc="Identify a single university that satisfies all specified criteria and provide supporting references",
        parent=root,
        critical=True
    )

    # Build critical checks for single university and official name
    build_single_uni_and_name_checks(evaluator, uni_id_node, extracted)

    # Build reference URLs presence checks (critical)
    build_reference_urls_checks(evaluator, uni_id_node, extracted)

    # Build and run criteria verifications (critical; evidence-based)
    await build_and_verify_criteria(evaluator, uni_id_node, extracted)

    # Return summary
    return evaluator.get_summary()