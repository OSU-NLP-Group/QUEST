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
TASK_ID = "aacsb_online_mba_top20_constraints"
TASK_DESCRIPTION = (
    "Identify an AACSB-accredited online MBA program that meets all of the following requirements: "
    "(1) offered by a university in Florida, Texas, or Arizona; "
    "(2) ranked in the top 20 of US News & World Report's 2025 Best Online MBA Programs; "
    "(3) total tuition cost under $50,000 for out-of-state students; "
    "(4) completable in 24 months or less for full-time students; "
    "(5) offers primarily asynchronous course delivery; "
    "(6) provides a Finance concentration or specialization; "
    "(7) minimum GPA requirement not exceeding 3.0; "
    "(8) requires no more than 3 years of professional work experience; "
    "(9) 100% online with no mandatory on-campus residency; "
    "(10) allows GMAT/GRE waiver or is test-optional; and "
    "(11) offers at least two start dates per year. "
    "Provide the program name, university name, and supporting reference URLs for accreditation status, ranking, "
    "program structure, admission requirements, and cost information."
)

ALLOWED_STATES = {"florida", "texas", "arizona"}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ProgramExtraction(BaseModel):
    # Identification
    program_name: Optional[str] = None
    university_name: Optional[str] = None
    # Optional explicit state mention in the answer (string like "Florida", "TX", etc.)
    state: Optional[str] = None

    # URLs for verifications (must be taken from the answer only)
    accreditation_urls: List[str] = Field(default_factory=list)
    ranking_urls: List[str] = Field(default_factory=list)

    duration_urls: List[str] = Field(default_factory=list)
    asynchronous_urls: List[str] = Field(default_factory=list)
    finance_urls: List[str] = Field(default_factory=list)
    residency_urls: List[str] = Field(default_factory=list)
    start_dates_urls: List[str] = Field(default_factory=list)

    gpa_urls: List[str] = Field(default_factory=list)
    work_experience_urls: List[str] = Field(default_factory=list)
    test_policy_urls: List[str] = Field(default_factory=list)

    cost_urls: List[str] = Field(default_factory=list)

    # Optional dedicated location URL(s) if provided
    location_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_program() -> str:
    return """
Extract the following structured information about the single online MBA program identified in the answer. Only extract information explicitly present in the answer text. Do not invent or infer anything.

Required fields:
- program_name: The MBA program's name (string).
- university_name: The university offering the program (string).
- state: The U.S. state of the university as explicitly mentioned in the answer (e.g., "Florida", "Texas", "Arizona", or an abbreviation like "FL", "TX", "AZ"). If not explicitly mentioned, return null.

Provide URL arrays (only include actual URLs explicitly present in the answer text; if none, return an empty list):
- accreditation_urls: URLs supporting AACSB accreditation of the program or the business school.
- ranking_urls: URLs supporting U.S. News & World Report 2025 Best Online MBA Programs top-20 ranking for this program.
- duration_urls: URLs supporting that the program can be completed in 24 months or less for full-time students.
- asynchronous_urls: URLs supporting that the program offers primarily asynchronous course delivery.
- finance_urls: URLs supporting that the program offers a Finance concentration/specialization/track.
- residency_urls: URLs supporting that the program is 100% online with no mandatory on-campus residency.
- start_dates_urls: URLs supporting that the program offers at least two start dates per year.
- gpa_urls: URLs supporting that the minimum GPA requirement does not exceed 3.0 on a 4.0 scale.
- work_experience_urls: URLs supporting that the program requires no more than 3 years of professional work experience (including "no experience required").
- test_policy_urls: URLs supporting GMAT/GRE waiver or test-optional policy.
- cost_urls: URLs supporting that the total tuition for out-of-state students is under $50,000.
- location_urls: Any URLs in the answer that explicitly show the university location/state (optional; leave empty if none provided).

Rules:
- Extract only URLs explicitly present in the answer (including markdown-style links). Do not fabricate or search for new links.
- Return null for missing scalar fields and an empty array for missing URL lists.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_valid_url(urls: List[str]) -> bool:
    return any(isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")) for u in urls)


def dedupe_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in url_lists:
        for u in lst:
            if isinstance(u, str):
                uu = u.strip()
                if uu and uu not in seen:
                    seen.add(uu)
                    merged.append(uu)
    return merged


def normalize_state_name(state_val: Optional[str]) -> Optional[str]:
    if not state_val:
        return None
    s = state_val.strip().lower()
    mapping = {
        "fl": "florida", "fla": "florida", "florida": "florida",
        "tx": "texas", "tex": "texas", "texas": "texas",
        "az": "arizona", "ariz": "arizona", "arizona": "arizona",
    }
    return mapping.get(s, state_val.strip().lower())


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_program_identification(evaluator: Evaluator, parent, ex: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Program_Identification",
        desc="Answer clearly identifies the program and institution.",
        parent=parent,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(ex.program_name and ex.program_name.strip()),
        id="Program_Name_Provided",
        desc="Provides the MBA program name.",
        parent=node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(ex.university_name and ex.university_name.strip()),
        id="University_Name_Provided",
        desc="Provides the university name offering the program.",
        parent=node,
        critical=True,
    )


async def build_accreditation_and_quality(evaluator: Evaluator, parent, ex: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Accreditation_and_Quality",
        desc="Program meets accreditation and ranking standards and provides supporting references for them.",
        parent=parent,
        critical=True,
    )

    # AACSB URL presence
    evaluator.add_custom_node(
        result=has_valid_url(ex.accreditation_urls),
        id="AACSB_Accreditation_URL",
        desc="Provides at least one valid URL supporting AACSB accreditation status.",
        parent=node,
        critical=True,
    )

    # AACSB Accreditation verification
    aacsb_leaf = evaluator.add_leaf(
        id="AACSB_Accreditation",
        desc="Program (or business school offering it) is AACSB-accredited.",
        parent=node,
        critical=True,
    )
    aacsb_claim = (
        f"The business school offering the online MBA at {ex.university_name or 'the university'} "
        f"is accredited by AACSB International."
    )
    await evaluator.verify(
        claim=aacsb_claim,
        node=aacsb_leaf,
        sources=ex.accreditation_urls,
        additional_instruction="Confirm accreditation by AACSB (Association to Advance Collegiate Schools of Business). "
                               "It is acceptable if the AACSB accreditation applies at the business-school level."
    )

    # US News Ranking URL presence
    evaluator.add_custom_node(
        result=has_valid_url(ex.ranking_urls),
        id="US_News_Ranking_URL",
        desc="Provides at least one valid URL supporting the U.S. News 2025 top-20 ranking claim.",
        parent=node,
        critical=True,
    )

    # US News Ranking verification
    rank_leaf = evaluator.add_leaf(
        id="US_News_Ranking",
        desc="Program is ranked in the top 20 of U.S. News & World Report's 2025 Best Online MBA Programs.",
        parent=node,
        critical=True,
    )
    rank_claim = (
        f"The online MBA program at {ex.university_name or 'the university'} "
        f"is ranked within the top 20 in U.S. News & World Report's 2025 Best Online MBA Programs."
    )
    await evaluator.verify(
        claim=rank_claim,
        node=rank_leaf,
        sources=ex.ranking_urls,
        additional_instruction="Verify specifically for the 'Best Online MBA Programs' 2025 ranking by U.S. News & World Report, "
                               "and confirm that the rank is within 1–20 (inclusive) for 2025."
    )

    # Geographic Location verification (no dedicated URL required by rubric)
    geo_leaf = evaluator.add_leaf(
        id="Geographic_Location",
        desc="University is located in Florida, Texas, or Arizona.",
        parent=node,
        critical=True,
    )

    # Prefer explicit state if provided in the answer; otherwise formulate a general claim
    normalized_state = normalize_state_name(ex.state)
    if normalized_state in ALLOWED_STATES:
        geo_claim = (
            f"The university {ex.university_name or 'the university'} is located in {normalized_state.capitalize()}."
        )
    else:
        geo_claim = (
            f"The university {ex.university_name or 'the university'} is located in one of Florida, Texas, or Arizona."
        )

    # Use location_urls if provided; otherwise, attempt with any program-related URLs that may show address/state
    fallback_urls = dedupe_urls(
        ex.location_urls,
        ex.duration_urls, ex.asynchronous_urls, ex.finance_urls, ex.residency_urls, ex.start_dates_urls,
        ex.gpa_urls, ex.work_experience_urls, ex.test_policy_urls, ex.cost_urls, ex.accreditation_urls
    )
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=fallback_urls if fallback_urls else None,
        additional_instruction="Check the institution's state location. Accept official program or university pages "
                               "that indicate the campus location (city/state) to confirm it is in Florida, Texas, or Arizona."
    )


async def build_program_structure_and_format(evaluator: Evaluator, parent, ex: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Program_Structure_and_Format",
        desc="Program structure aligns with duration, modality, concentration, residency, and start-date requirements and provides supporting references for each.",
        parent=parent,
        critical=True,
    )

    # Program Duration URL presence
    evaluator.add_custom_node(
        result=has_valid_url(ex.duration_urls),
        id="Program_Duration_URL",
        desc="Provides at least one valid URL supporting the program duration/completion-time claim.",
        parent=node,
        critical=True,
    )
    # Program Duration claim
    dur_leaf = evaluator.add_leaf(
        id="Program_Duration",
        desc="Program can be completed in 24 months or less for full-time students.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The online MBA program can be completed in 24 months or less for full-time students.",
        node=dur_leaf,
        sources=ex.duration_urls,
        additional_instruction="Confirm completion time (e.g., 12, 18, 20, or 24 months). "
                               "If multiple paces are listed, the full-time track must allow completion within 24 months."
    )

    # Asynchronous Delivery URL presence
    evaluator.add_custom_node(
        result=has_valid_url(ex.asynchronous_urls),
        id="Asynchronous_Delivery_URL",
        desc="Provides at least one valid URL supporting the primarily asynchronous delivery claim.",
        parent=node,
        critical=True,
    )
    # Asynchronous Delivery claim
    async_leaf = evaluator.add_leaf(
        id="Asynchronous_Delivery",
        desc="Program offers primarily asynchronous course delivery.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program offers primarily asynchronous course delivery.",
        node=async_leaf,
        sources=ex.asynchronous_urls,
        additional_instruction="It's acceptable if there are occasional or optional live (synchronous) sessions, "
                               "as long as the primary delivery is asynchronous."
    )

    # Finance Concentration URL presence
    evaluator.add_custom_node(
        result=has_valid_url(ex.finance_urls),
        id="Finance_Concentration_URL",
        desc="Provides at least one valid URL supporting the Finance concentration/specialization claim.",
        parent=node,
        critical=True,
    )
    # Finance Concentration claim
    fin_leaf = evaluator.add_leaf(
        id="Finance_Concentration",
        desc="Program offers a Finance concentration or specialization.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The online MBA program offers a Finance concentration or specialization (track/focus).",
        node=fin_leaf,
        sources=ex.finance_urls,
        additional_instruction="Look for explicit mention of a Finance concentration/specialization/track. "
                               "Synonyms like 'Finance emphasis' or 'Finance focus' are acceptable."
    )

    # No Residency Requirement URL presence
    evaluator.add_custom_node(
        result=has_valid_url(ex.residency_urls),
        id="No_Residency_Requirement_URL",
        desc="Provides at least one valid URL supporting the no-residency/fully-online claim.",
        parent=node,
        critical=True,
    )
    # No Residency Requirement claim
    resid_leaf = evaluator.add_leaf(
        id="No_Residency_Requirement",
        desc="Program is 100% online with no mandatory on-campus residency.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program is 100% online with no mandatory on-campus residency.",
        node=resid_leaf,
        sources=ex.residency_urls,
        additional_instruction="Optional campus visits or optional immersions are acceptable; the key is that there is no mandatory on-campus residency."
    )

    # Multiple Start Dates URL presence
    evaluator.add_custom_node(
        result=has_valid_url(ex.start_dates_urls),
        id="Multiple_Start_Dates_URL",
        desc="Provides at least one valid URL supporting the multiple-start-dates-per-year claim.",
        parent=node,
        critical=True,
    )
    # Multiple Start Dates claim
    starts_leaf = evaluator.add_leaf(
        id="Multiple_Start_Dates",
        desc="Program offers at least two start dates per year.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program offers at least two start dates per year.",
        node=starts_leaf,
        sources=ex.start_dates_urls,
        additional_instruction="Examples include fall/spring, spring/summer, or multiple rolling start sessions per year."
    )


async def build_admission_requirements(evaluator: Evaluator, parent, ex: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Admission_Requirements",
        desc="Program admission requirements meet GPA, work experience, and test policy constraints and provide supporting references for each.",
        parent=parent,
        critical=True,
    )

    # GPA URL presence
    evaluator.add_custom_node(
        result=has_valid_url(ex.gpa_urls),
        id="GPA_Requirement_URL",
        desc="Provides at least one valid URL supporting the minimum GPA requirement claim.",
        parent=node,
        critical=True,
    )
    # GPA claim
    gpa_leaf = evaluator.add_leaf(
        id="GPA_Requirement",
        desc="Minimum GPA requirement does not exceed 3.0 on a 4.0 scale.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The minimum GPA requirement for admission does not exceed 3.0 on a 4.0 scale.",
        node=gpa_leaf,
        sources=ex.gpa_urls,
        additional_instruction="Accept thresholds such as 3.0, 2.75, or 'holistic review' with stated minimum not exceeding 3.0. "
                               "If multiple thresholds exist, the baseline minimum for regular admission must be ≤ 3.0."
    )

    # Work Experience URL presence
    evaluator.add_custom_node(
        result=has_valid_url(ex.work_experience_urls),
        id="Work_Experience_URL",
        desc="Provides at least one valid URL supporting the work-experience requirement claim.",
        parent=node,
        critical=True,
    )
    # Work Experience claim
    work_leaf = evaluator.add_leaf(
        id="Work_Experience",
        desc="Program requires no more than 3 years of professional work experience.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program requires no more than 3 years of professional work experience for admission.",
        node=work_leaf,
        sources=ex.work_experience_urls,
        additional_instruction="If no experience is required or only 'preferred', that satisfies the 'no more than 3 years' constraint."
    )

    # Test Optional URL presence
    evaluator.add_custom_node(
        result=has_valid_url(ex.test_policy_urls),
        id="Test_Optional_URL",
        desc="Provides at least one valid URL supporting the GMAT/GRE waiver or test-optional policy claim.",
        parent=node,
        critical=True,
    )
    # Test Optional claim
    test_leaf = evaluator.add_leaf(
        id="Test_Optional",
        desc="Program allows a GMAT/GRE waiver or is test-optional for qualified applicants.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="The program allows a GMAT/GRE waiver or is test-optional for qualified applicants.",
        node=test_leaf,
        sources=ex.test_policy_urls,
        additional_instruction="Any official policy text stating GMAT/GRE waiver availability or test-optional admission is acceptable."
    )


async def build_cost_and_accessibility(evaluator: Evaluator, parent, ex: ProgramExtraction):
    node = evaluator.add_parallel(
        id="Cost_and_Accessibility",
        desc="Program cost meets the affordability requirement and includes supporting references.",
        parent=parent,
        critical=True,
    )

    # Cost URL presence
    evaluator.add_custom_node(
        result=has_valid_url(ex.cost_urls),
        id="Total_Tuition_Cost_URL",
        desc="Provides at least one valid URL supporting the out-of-state total tuition cost claim.",
        parent=node,
        critical=True,
    )
    # Cost claim
    cost_leaf = evaluator.add_leaf(
        id="Total_Tuition_Cost",
        desc="Total tuition cost is under $50,000 for out-of-state students.",
        parent=node,
        critical=True,
    )
    await evaluator.verify(
        claim="For out-of-state students, the total tuition cost for the online MBA is under $50,000.",
        node=cost_leaf,
        sources=ex.cost_urls,
        additional_instruction="Focus on total program tuition (not per-credit). If per-credit pricing is shown, "
                               "the page must explicitly provide a total that is under $50,000 for out-of-state students."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    extraction: ProgramExtraction = await evaluator.extract(
        prompt=prompt_extract_program(),
        template_class=ProgramExtraction,
        extraction_name="program_extraction",
    )

    # Build the critical Program_Suitability node (as the rubric root under our framework root)
    suitability = evaluator.add_parallel(
        id="Program_Suitability",
        desc="The identified online MBA program meets all specified requirements and includes required identifying information and supporting references.",
        parent=root,
        critical=True,
    )

    # Subsections
    await build_program_identification(evaluator, suitability, extraction)
    await build_accreditation_and_quality(evaluator, suitability, extraction)
    await build_program_structure_and_format(evaluator, suitability, extraction)
    await build_admission_requirements(evaluator, suitability, extraction)
    await build_cost_and_accessibility(evaluator, suitability, extraction)

    # Return final structured summary
    return evaluator.get_summary()