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
TASK_ID = "univ_constraints_us_app_sat_admit_enroll"
TASK_DESCRIPTION = (
    "Identify a university in the United States that satisfies all of the following criteria for undergraduate admissions: "
    "(1) The university must accept both the Common Application and the Coalition Application; "
    "(2) The university's middle 50% SAT score range must have a lower bound of at least 1350; "
    "(3) The university's middle 50% SAT score range must have an upper bound of at most 1520; "
    "(4) The university must offer Early Action as an application option; "
    "(5) The university must offer Early Decision as an application option; "
    "(6) The university must offer Regular Decision as an application option; "
    "(7) The university's campus size must be between 500 and 2,000 acres (inclusive); "
    "(8) The university must have an undergraduate admission acceptance rate below 35%; "
    "(9) The university must be a private institution; "
    "(10) The university must have an undergraduate enrollment of at least 5,000 students. "
    "Provide the exact name of the university and include reference URLs that verify each of these criteria."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer for a single university and
    URLs supporting each constraint.
    """
    university_name: Optional[str] = None

    # URLs for verifying the official name (e.g., main homepage, Wikipedia, About page); optional
    name_urls: List[str] = Field(default_factory=list)

    # Location verification URLs (United States)
    location_urls: List[str] = Field(default_factory=list)

    # Application platform acceptance (Common & Coalition) URLs
    apps_urls: List[str] = Field(default_factory=list)

    # SAT middle-50% range URLs; optional textual range if provided in the answer
    sat_middle50_text: Optional[str] = None
    sat_range_urls: List[str] = Field(default_factory=list)

    # Application plan option URLs
    early_action_urls: List[str] = Field(default_factory=list)
    early_decision_urls: List[str] = Field(default_factory=list)
    regular_decision_urls: List[str] = Field(default_factory=list)

    # Campus size URLs (acreage)
    campus_size_urls: List[str] = Field(default_factory=list)

    # Acceptance rate URLs
    acceptance_rate_urls: List[str] = Field(default_factory=list)

    # Private institution URLs
    private_urls: List[str] = Field(default_factory=list)

    # Undergraduate enrollment URLs
    undergrad_enrollment_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_info() -> str:
    return """
    Extract details for a single university identified in the answer and all reference URLs associated
    with each constraint listed below. Only extract URLs explicitly present in the answer text (including
    plain URLs or Markdown links). If a URL is missing the protocol, prepend http://. If no URL is given
    for a constraint, return an empty array for that constraint.

    Extract the following fields:
    - university_name: The exact official university name as stated in the answer (do not invent).
    - name_urls: URLs that show the official name of the university (e.g., the university homepage,
                 About page, or its Wikipedia page), if present in the answer.
    - location_urls: URLs that indicate the university is located in the United States.
    - apps_urls: URLs that show which application platforms are accepted for undergraduate admissions
                 (specifically Common Application and/or Coalition Application). If multiple pages are cited,
                 include them all.
    - sat_middle50_text: If the answer states an exact middle-50% SAT score range (e.g., "1350–1520"), extract it as text. Otherwise null.
    - sat_range_urls: URLs that provide the middle-50% SAT score range from publicly available admissions info.
    - early_action_urls: URLs that show Early Action is offered.
    - early_decision_urls: URLs that show Early Decision is offered (ED I or ED II both count).
    - regular_decision_urls: URLs that show Regular Decision is offered.
    - campus_size_urls: URLs that provide the campus size in acres.
    - acceptance_rate_urls: URLs that provide the undergraduate acceptance rate.
    - private_urls: URLs that state the institution is private.
    - undergrad_enrollment_urls: URLs that provide undergraduate enrollment figures.

    Notes:
    - Include every URL the answer associates with each criterion. If one URL is referenced for multiple
      criteria, include it in each corresponding array.
    - Do not infer or fabricate URLs; extract exactly as provided in the answer.
    - If the answer mentions "Coalition on Scoir" or "Coalition for College", treat them as Coalition Application acceptance sources.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _union_urls(ex: UniversityExtraction) -> List[str]:
    """Union of all extracted URLs to aid name verification when a dedicated name URL isn't provided."""
    union = set()
    for field in [
        ex.name_urls,
        ex.location_urls,
        ex.apps_urls,
        ex.sat_range_urls,
        ex.early_action_urls,
        ex.early_decision_urls,
        ex.regular_decision_urls,
        ex.campus_size_urls,
        ex.acceptance_rate_urls,
        ex.private_urls,
        ex.undergrad_enrollment_urls,
    ]:
        for u in field:
            if isinstance(u, str) and u.strip():
                union.add(u.strip())
    return list(union)


async def _add_existence_node_for_urls(
    evaluator: Evaluator,
    parent,
    id_suffix: str,
    desc_prefix: str,
    urls: List[str],
    critical: bool = True,
) -> None:
    """
    Add a critical existence check ensuring at least one URL is provided for a given criterion.
    """
    evaluator.add_custom_node(
        result=bool(urls) and len(urls) > 0,
        id=f"{id_suffix}_urls_provided",
        desc=f"{desc_prefix} - URLs are provided",
        parent=parent,
        critical=critical
    )


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_university_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: UniversityExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run checks.
    """

    # University_Response node (critical, parallel)
    uni_node = evaluator.add_parallel(
        id="University_Response",
        desc="Response identifies a single university and provides URLs that verify every stated constraint.",
        parent=parent_node,
        critical=True
    )

    # 1) Exact University Name provided (with supporting URL verification)
    #    - First, ensure the answer provides a name string
    evaluator.add_custom_node(
        result=bool(extracted.university_name) and bool(extracted.university_name.strip()),
        id="University_Name_Provided",
        desc="Provides a university name in the answer",
        parent=uni_node,
        critical=True
    )

    #    - Then verify the official name against available pages (prefer name_urls; fallback to union of all URLs)
    exact_name_leaf = evaluator.add_leaf(
        id="Exact_University_Name_Provided",
        desc="Provides the exact official name of the university.",
        parent=uni_node,
        critical=True
    )
    name_sources = extracted.name_urls if (extracted.name_urls and len(extracted.name_urls) > 0) else _union_urls(extracted)
    name_claim = f"The official name of the university is exactly '{(extracted.university_name or '').strip()}'."
    await evaluator.verify(
        claim=name_claim,
        node=exact_name_leaf,
        sources=name_sources,
        additional_instruction=(
            "Verify that the page shows the institution's official name matching the provided one. "
            "Allow minor punctuation/casing variants, but the substantive name must match. "
            "If multiple campuses or abbreviations exist, focus on the full official name as typically shown on the institution's homepage or Wikipedia."
        ),
    )

    # 2) All constraints verified (critical, parallel)
    constraints_node = evaluator.add_parallel(
        id="All_Constraints_Verified_With_URLs",
        desc="Each constraint from the question/constraints is satisfied and is supported by at least one publicly accessible reference URL.",
        parent=uni_node,
        critical=True
    )

    claims_and_sources: List[tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # 2.a Located in United States
    await _add_existence_node_for_urls(
        evaluator, constraints_node, "Located_In_United_States_With_URL", "Located in the United States", extracted.location_urls
    )
    loc_leaf = evaluator.add_leaf(
        id="Located_In_United_States_With_URL",
        desc="Provides a URL showing the university is located in the United States.",
        parent=constraints_node,
        critical=True
    )
    loc_claim = "The university is located in the United States."
    claims_and_sources.append((
        loc_claim,
        extracted.location_urls,
        loc_leaf,
        "Confirm the university's primary campus location is in the United States. "
        "International campuses do not disqualify it as long as the institution is based in the U.S."
    ))

    # 2.b Accepts Common & Coalition (split into two concrete leaves, under a critical parallel sub-node)
    apps_parent = evaluator.add_parallel(
        id="Accepts_Common_And_Coalition_With_URL",
        desc="Provides URL evidence that BOTH the Common Application and the Coalition Application are accepted.",
        parent=constraints_node,
        critical=True
    )
    await _add_existence_node_for_urls(
        evaluator, apps_parent, "Apps_Platforms", "Application platforms acceptance", extracted.apps_urls
    )

    # Common Application acceptance
    common_leaf = evaluator.add_leaf(
        id="Accepts_Common_With_URL",
        desc="Provides a URL showing the university accepts the Common Application.",
        parent=apps_parent,
        critical=True
    )
    common_claim = "The university accepts the Common Application for undergraduate admissions."
    claims_and_sources.append((
        common_claim,
        extracted.apps_urls,
        common_leaf,
        "Look for explicit statements such as 'Apply using the Common Application' or inclusion among accepted platforms."
    ))

    # Coalition Application acceptance (including Coalition on Scoir)
    coalition_leaf = evaluator.add_leaf(
        id="Accepts_Coalition_With_URL",
        desc="Provides a URL showing the university accepts the Coalition Application.",
        parent=apps_parent,
        critical=True
    )
    coalition_claim = (
        "The university accepts the Coalition Application (including Coalition on Scoir) for undergraduate admissions."
    )
    claims_and_sources.append((
        coalition_claim,
        extracted.apps_urls,
        coalition_leaf,
        "Treat 'Coalition Application', 'Coalition for College', and 'Coalition on Scoir' as equivalents."
    ))

    # 2.c SAT middle-50% exact range publicly stated
    await _add_existence_node_for_urls(
        evaluator, constraints_node, "SAT_Middle50_Exact_Range_Public_With_URL", "SAT middle-50% range", extracted.sat_range_urls
    )
    sat_range_leaf = evaluator.add_leaf(
        id="SAT_Middle50_Exact_Range_Public_With_URL",
        desc="Provides a URL from publicly available admissions information that states the exact middle-50% SAT score range.",
        parent=constraints_node,
        critical=True
    )
    sat_range_claim = (
        "The provided admissions page explicitly states the middle 50% SAT score range for admitted undergraduates (two specific numbers forming a range)."
        if not extracted.sat_middle50_text
        else f"The provided admissions page explicitly states the middle 50% SAT score range as '{extracted.sat_middle50_text}'."
    )
    claims_and_sources.append((
        sat_range_claim,
        extracted.sat_range_urls,
        sat_range_leaf,
        "The page must explicitly include two numbers (e.g., '1350–1520') labeled as the middle 50% SAT range."
    ))

    # 2.d SAT lower bound ≥ 1350
    await _add_existence_node_for_urls(
        evaluator, constraints_node, "SAT_Lower_Bound_At_Least_1350_With_URL", "SAT lower bound (middle-50%)", extracted.sat_range_urls
    )
    sat_lower_leaf = evaluator.add_leaf(
        id="SAT_Lower_Bound_At_Least_1350_With_URL",
        desc="Provides a URL supporting that the middle-50% SAT range lower bound is ≥ 1350.",
        parent=constraints_node,
        critical=True
    )
    sat_lower_claim = "The middle 50% SAT range lower bound is at least 1350."
    claims_and_sources.append((
        sat_lower_claim,
        extracted.sat_range_urls,
        sat_lower_leaf,
        "Use the page's explicitly stated lower bound of the middle 50% SAT range; allow minor rounding."
    ))

    # 2.e SAT upper bound ≤ 1520
    await _add_existence_node_for_urls(
        evaluator, constraints_node, "SAT_Upper_Bound_At_Most_1520_With_URL", "SAT upper bound (middle-50%)", extracted.sat_range_urls
    )
    sat_upper_leaf = evaluator.add_leaf(
        id="SAT_Upper_Bound_At_Most_1520_With_URL",
        desc="Provides a URL supporting that the middle-50% SAT range upper bound is ≤ 1520.",
        parent=constraints_node,
        critical=True
    )
    sat_upper_claim = "The middle 50% SAT range upper bound is at most 1520."
    claims_and_sources.append((
        sat_upper_claim,
        extracted.sat_range_urls,
        sat_upper_leaf,
        "Use the page's explicitly stated upper bound of the middle 50% SAT range; allow minor rounding."
    ))

    # 2.f Offers Early Action
    await _add_existence_node_for_urls(
        evaluator, constraints_node, "Offers_Early_Action_With_URL", "Early Action offering", extracted.early_action_urls
    )
    ea_leaf = evaluator.add_leaf(
        id="Offers_Early_Action_With_URL",
        desc="Provides a URL showing Early Action is offered as an application option.",
        parent=constraints_node,
        critical=True
    )
    ea_claim = "The university offers Early Action as an application option."
    claims_and_sources.append((
        ea_claim,
        extracted.early_action_urls,
        ea_leaf,
        "Restrictive or Single-Choice Early Action qualifies as Early Action."
    ))

    # 2.g Offers Early Decision
    await _add_existence_node_for_urls(
        evaluator, constraints_node, "Offers_Early_Decision_With_URL", "Early Decision offering", extracted.early_decision_urls
    )
    ed_leaf = evaluator.add_leaf(
        id="Offers_Early_Decision_With_URL",
        desc="Provides a URL showing Early Decision is offered as an application option.",
        parent=constraints_node,
        critical=True
    )
    ed_claim = "The university offers Early Decision (ED I or ED II) as an application option."
    claims_and_sources.append((
        ed_claim,
        extracted.early_decision_urls,
        ed_leaf,
        "Either Early Decision I or Early Decision II qualifies; look for explicit ED wording."
    ))

    # 2.h Offers Regular Decision
    await _add_existence_node_for_urls(
        evaluator, constraints_node, "Offers_Regular_Decision_With_URL", "Regular Decision offering", extracted.regular_decision_urls
    )
    rd_leaf = evaluator.add_leaf(
        id="Offers_Regular_Decision_With_URL",
        desc="Provides a URL showing Regular Decision is offered as an application option.",
        parent=constraints_node,
        critical=True
    )
    rd_claim = "The university offers Regular Decision as an application option."
    claims_and_sources.append((
        rd_claim,
        extracted.regular_decision_urls,
        rd_leaf,
        "Look for 'Regular Decision' or equivalent standard application round."
    ))

    # 2.i Campus size between 500 and 2,000 acres inclusive
    await _add_existence_node_for_urls(
        evaluator, constraints_node, "Campus_Size_500_to_2000_Acres_With_URL", "Campus size (acreage)", extracted.campus_size_urls
    )
    campus_leaf = evaluator.add_leaf(
        id="Campus_Size_500_to_2000_Acres_With_URL",
        desc="Provides a URL showing campus size is between 500 and 2,000 acres inclusive.",
        parent=constraints_node,
        critical=True
    )
    campus_claim = "The campus size is between 500 and 2,000 acres inclusive."
    claims_and_sources.append((
        campus_claim,
        extracted.campus_size_urls,
        campus_leaf,
        "Use the primary/main campus acreage stated; synonyms like 'campus area' or 'acres' are acceptable."
    ))

    # 2.j Acceptance rate below 35%
    await _add_existence_node_for_urls(
        evaluator, constraints_node, "Acceptance_Rate_Below_35_With_URL", "Undergraduate acceptance rate", extracted.acceptance_rate_urls
    )
    admit_leaf = evaluator.add_leaf(
        id="Acceptance_Rate_Below_35_With_URL",
        desc="Provides a URL showing the undergraduate admission acceptance rate is below 35%.",
        parent=constraints_node,
        critical=True
    )
    admit_claim = "The undergraduate acceptance (admit) rate is below 35%."
    claims_and_sources.append((
        admit_claim,
        extracted.acceptance_rate_urls,
        admit_leaf,
        "If multiple cycles are shown, any clearly labeled recent undergraduate admit rate under 35% qualifies."
    ))

    # 2.k Private institution
    await _add_existence_node_for_urls(
        evaluator, constraints_node, "Private_Institution_With_URL", "Institution type", extracted.private_urls
    )
    private_leaf = evaluator.add_leaf(
        id="Private_Institution_With_URL",
        desc="Provides a URL showing the institution is private.",
        parent=constraints_node,
        critical=True
    )
    private_claim = "The institution is private."
    claims_and_sources.append((
        private_claim,
        extracted.private_urls,
        private_leaf,
        "Accept phrasing like 'private university' or 'private research university'; do not accept 'public'."
    ))

    # 2.l Undergraduate enrollment at least 5,000
    await _add_existence_node_for_urls(
        evaluator, constraints_node, "Undergrad_Enrollment_At_Least_5000_With_URL", "Undergraduate enrollment", extracted.undergrad_enrollment_urls
    )
    enroll_leaf = evaluator.add_leaf(
        id="Undergrad_Enrollment_At_Least_5000_With_URL",
        desc="Provides a URL showing undergraduate enrollment is at least 5,000 students.",
        parent=constraints_node,
        critical=True
    )
    enroll_claim = "Undergraduate enrollment is at least 5,000 students."
    claims_and_sources.append((
        enroll_claim,
        extracted.undergrad_enrollment_urls,
        enroll_leaf,
        "Prefer undergrad-specific figure; do not use total/all-campus enrollment unless the page clearly identifies undergraduate count."
    ))

    # Execute verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the university constraints task.
    """
    # Initialize evaluator (root is non-critical per framework design)
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_university_info(),
        template_class=UniversityExtraction,
        extraction_name="university_constraints_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_university_tree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()