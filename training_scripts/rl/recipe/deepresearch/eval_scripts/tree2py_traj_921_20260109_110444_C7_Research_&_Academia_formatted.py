import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "public_r1_universities"
TASK_DESCRIPTION = (
    "Identify 4 public research universities in the United States that meet ALL of the following criteria:\n\n"
    "1. Each university must be a public (state-funded) institution\n"
    "2. Each university must be classified as R1: Doctoral Universities - Very High Research Activity according to the Carnegie Classification of Institutions of Higher Education\n"
    "3. Each university must have total undergraduate enrollment of at least 35,000 students as of Fall 2024 or Fall 2025\n"
    "4. Each university must have a six-year graduation rate of at least 80%\n"
    "5. The 4 universities must be located in 4 different U.S. states\n\n"
    "For each university, provide the following information:\n"
    "- Official university name\n"
    "- State location\n"
    "- Current undergraduate enrollment figure (Fall 2024 or Fall 2025)\n"
    "- Six-year graduation rate (as a percentage)\n"
    "- A reference URL from an authoritative source (such as the university's official website, U.S. News & World Report, Carnegie Classifications, or NCES) that verifies the university's R1 classification, enrollment, and graduation rate"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    """Information for a single university extracted from the answer."""
    official_name: Optional[str] = None
    state: Optional[str] = None
    ug_enrollment_term: Optional[str] = None  # e.g., "Fall 2024" or "Fall 2025"
    ug_enrollment_value: Optional[str] = None  # e.g., "40,123", "about 38k", "35,000+"
    graduation_rate_percent: Optional[str] = None  # e.g., "82%", "≈ 84 percent"
    reference_urls: List[str] = Field(default_factory=list)  # authoritative references mentioned


class UniversitiesExtraction(BaseModel):
    """Model for the list of universities extracted from the answer."""
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return (
        "Extract information about up to four U.S. public research universities listed in the answer. "
        "If the answer contains more than four, include only the first four mentioned. If fewer than four are present, "
        "still return four entries, using null for missing fields for the nonexistent ones.\n\n"
        "For each university, extract the following fields exactly as they appear in the answer:\n"
        "1. official_name: The official university name.\n"
        "2. state: The U.S. state where the university is located (prefer full state name if provided; otherwise use the abbreviation if that's what's in the answer).\n"
        "3. ug_enrollment_term: The term associated with the undergraduate enrollment number (must be Fall 2024 or Fall 2025 if provided), e.g., 'Fall 2024' or 'Fall 2025'.\n"
        "4. ug_enrollment_value: The undergraduate enrollment figure as stated (string; keep any formatting such as commas or 'approx').\n"
        "5. graduation_rate_percent: The six-year graduation rate percentage (string; keep formatting such as '%' or words like 'percent').\n"
        "6. reference_urls: An array of all URLs cited in the answer that are intended to support claims about R1 classification, enrollment, and graduation rate. "
        "Include only valid URLs explicitly present in the answer. These may include .edu pages, acenet/carnegie classifications, nces.ed.gov, or usnews.com.\n\n"
        "Return a JSON object with a single key 'universities' that is an array of up to 4 objects with the above fields. "
        "If a field is not mentioned for a university, set it to null. If no references are given for a university, return an empty array for 'reference_urls'."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
STATE_ABBREV_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia",
}

STATE_NAME_TO_NAME = {v.lower(): v for v in STATE_ABBREV_TO_NAME.values()}


def canonicalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    if not s:
        return None
    s_upper = s.upper()
    if s_upper in STATE_ABBREV_TO_NAME:
        return STATE_ABBREV_TO_NAME[s_upper]
    s_lower = s.lower()
    return STATE_NAME_TO_NAME.get(s_lower, s.strip())


def normalize_university_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return "".join(ch for ch in name.lower().strip() if ch.isalnum() or ch.isspace())


def is_authoritative_url(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        if not netloc:
            return False
        # Common authoritative domains
        if netloc.endswith(".edu"):
            return True
        if "carnegieclassifications.acenet.edu" in netloc:
            return True
        if "nces.ed.gov" in netloc:
            return True
        if "usnews.com" in netloc:
            return True
        # Some university info portals may be authoritative if under .gov or key edu subdomains
        if netloc.endswith(".gov"):
            return True
        return False
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index_one_based: int,
) -> None:
    """
    Build the verification subtree for one university.
    """
    # University node
    uni_node = evaluator.add_parallel(
        id=f"University_{index_one_based}",
        desc=f"University {index_one_based} satisfies all per-university constraints and required fields are provided.",
        parent=parent_node,
        critical=False,
    )

    # Official name provided (critical)
    name_provided = bool(uni.official_name and uni.official_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id=f"U{index_one_based}_Official_Name_Provided",
        desc=f"Official university name is provided for University {index_one_based}.",
        parent=uni_node,
        critical=True,
    )

    # State provided (critical)
    state_provided = bool(uni.state and uni.state.strip())
    evaluator.add_custom_node(
        result=state_provided,
        id=f"U{index_one_based}_State_Provided",
        desc=f"U.S. state location is provided for University {index_one_based}.",
        parent=uni_node,
        critical=True,
    )

    # Reference URLs existence and authoritativeness (critical)
    refs_exist = bool(uni.reference_urls)
    refs_authoritative = any(is_authoritative_url(u) for u in uni.reference_urls) if refs_exist else False
    evaluator.add_custom_node(
        result=(refs_exist and refs_authoritative),
        id=f"U{index_one_based}_Reference_URLs",
        desc=(
            f"Provides authoritative reference URL(s) supporting University {index_one_based}'s "
            f"R1 classification, enrollment, and graduation rate claims."
        ),
        parent=uni_node,
        critical=True,
    )

    # Public institution verification (critical)
    public_leaf = evaluator.add_leaf(
        id=f"U{index_one_based}_Public_Institution",
        desc=f"University {index_one_based} is a public (state-funded) institution (not private).",
        parent=uni_node,
        critical=True,
    )
    public_claim = (
        f"{uni.official_name or 'The university'} is a public (state-funded) university in the United States."
    )
    await evaluator.verify(
        claim=public_claim,
        node=public_leaf,
        sources=uni.reference_urls,
        additional_instruction=(
            "Confirm the institution's control/sector is public (state-funded). Accept synonyms like 'public research university'. "
            "Only consider it supported if the provided webpage explicitly indicates public status."
        ),
    )

    # R1 classification verification (critical)
    r1_leaf = evaluator.add_leaf(
        id=f"U{index_one_based}_R1_Classified",
        desc=(
            f"University {index_one_based} is classified as R1 (Doctoral Universities – Very High Research Activity) "
            f"per Carnegie Classification."
        ),
        parent=uni_node,
        critical=True,
    )
    r1_claim = (
        f"{uni.official_name or 'The university'} is classified as R1: Doctoral Universities – Very High Research Activity."
    )
    await evaluator.verify(
        claim=r1_claim,
        node=r1_leaf,
        sources=uni.reference_urls,
        additional_instruction=(
            "Verify that the page supports the 'R1: Very High Research Activity' classification as defined by Carnegie Classifications. "
            "Explicit mention of 'R1' or equivalent phrasing should be present."
        ),
    )

    # Undergraduate enrollment constraint and value (critical)
    enroll_leaf = evaluator.add_leaf(
        id=f"U{index_one_based}_Ug_Enrollment_Constraint_And_Value",
        desc=(
            f"University {index_one_based} undergraduate enrollment is provided with term (Fall 2024 or Fall 2025) "
            f"and is >= 35,000."
        ),
        parent=uni_node,
        critical=True,
    )
    term_text = uni.ug_enrollment_term or "Fall 2024 or Fall 2025"
    enroll_value_text = uni.ug_enrollment_value or "(value not specified)"
    enroll_claim = (
        f"As of {term_text}, the undergraduate enrollment at {uni.official_name or 'the university'} "
        f"is reported as '{enroll_value_text}', and the figure is at least 35,000."
    )
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=uni.reference_urls,
        additional_instruction=(
            "Confirm two parts from the reference(s): (1) the term is Fall 2024 or Fall 2025 specifically; "
            "(2) the reported undergraduate enrollment is at least 35,000. "
            "Accept minor formatting differences and rounding. If only older terms (e.g., 2023 or prior) "
            "are available, or the figure is < 35,000, mark as not supported."
        ),
    )

    # Graduation rate constraint and value (critical)
    grad_leaf = evaluator.add_leaf(
        id=f"U{index_one_based}_Grad_Rate_Constraint_And_Value",
        desc=(
            f"University {index_one_based} six-year graduation rate is provided as a percentage and is >= 80%."
        ),
        parent=uni_node,
        critical=True,
    )
    grad_rate_text = uni.graduation_rate_percent or "(value not specified)"
    grad_claim = (
        f"The six-year graduation rate for {uni.official_name or 'the university'} is '{grad_rate_text}', "
        f"and the rate is at least 80%."
    )
    await evaluator.verify(
        claim=grad_claim,
        node=grad_leaf,
        sources=uni.reference_urls,
        additional_instruction=(
            "Confirm that the reference(s) provide a six-year graduation rate that is at least 80%. "
            "Allow minor rounding (e.g., 79.6% ≈ 80%). The page should clearly indicate the six-year rate."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'public_r1_universities' task using the Mind2Web2 framework.
    """
    # Initialize evaluator (root non-critical to allow partial credit; critical children will gate failures)
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

    # Extract universities from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction",
    )

    # Normalize and select first four universities; pad if fewer than 4
    universities: List[UniversityItem] = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Count_Distinct_Universities (critical)
    names_normalized = [
        normalize_university_name(u.official_name) for u in universities if u.official_name
    ]
    distinct_names_count = len(set(n for n in names_normalized if n))
    count_is_four_distinct = (len(universities) == 4) and (distinct_names_count == 4)
    evaluator.add_custom_node(
        result=count_is_four_distinct,
        id="Count_Distinct_Universities",
        desc="Response identifies 4 distinct universities (not fewer or more).",
        parent=root,
        critical=True,
    )

    # Build verification subtrees for each university
    for idx, uni in enumerate(universities, start=1):
        await verify_university(evaluator, root, uni, idx)

    # Geographic_Diversity (critical)
    states_canonical = [canonicalize_state(u.state) for u in universities]
    # Geographic diversity requires 4 non-null states and all distinct
    states_present = all(s is not None and str(s).strip() != "" for s in states_canonical)
    states_distinct = len(set(states_canonical)) == 4 if states_present else False
    evaluator.add_custom_node(
        result=(states_present and states_distinct),
        id="Geographic_Diversity",
        desc="The 4 universities are located in 4 different U.S. states.",
        parent=root,
        critical=True,
    )

    # Return unified evaluation summary
    return evaluator.get_summary()