import asyncio
import logging
from typing import Optional, List, Dict, Any, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "edu_leadership_jobs_2026"
TASK_DESCRIPTION = """
I am an experienced K-12 educator with a Master's degree in Educational Administration looking to advance my career into educational leadership. I want to research current job opportunities for the 2026-2027 school year. Please identify three current educational leadership job openings that meet ALL of the following criteria:

1. Position must be at the Principal level or higher (Superintendent, Principal, Assistant Principal, or equivalent district-level administrator)
2. Position must explicitly require a Master's degree or higher in Educational Leadership, Educational Administration, or a related field
3. Position must require a minimum of 3 years of K-12 teaching or educational experience
4. Position must require valid state administrator, principal, or superintendent certification/licensure
5. Position must include a verifiable salary range or compensation information
6. Position must have a clearly stated application deadline (either a specific date in 2026 or listed as 'Until Filled')
7. The three positions must be located in three different U.S. states
8. Each position must include the district name and state location
9. Each position must have a valid reference URL to an official job posting

For each of the three positions, provide:
- Position title
- School district name and state
- Salary range
- Application deadline
- Brief summary of key requirements (education, experience, certification)
- Reference URL to the official job posting
"""

TARGET_DEADLINE_YEAR = 2026


# --------------------------------------------------------------------------- #
# US States utilities                                                         #
# --------------------------------------------------------------------------- #
US_STATE_TO_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY", "DISTRICT OF COLUMBIA": "DC", "WASHINGTON, D.C.": "DC",
    "WASHINGTON DC": "DC"
}
US_ABBRS = set(US_STATE_TO_ABBR.values())


def canonicalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip().upper()
    # Normalize punctuation variants
    s = s.replace(".", "").replace(",", "")
    if s in US_ABBRS:
        return s
    return US_STATE_TO_ABBR.get(s, None)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    position_title: Optional[str] = None
    district_name: Optional[str] = None
    state: Optional[str] = None  # Prefer 2-letter postal code if available
    salary_range: Optional[str] = None
    application_deadline: Optional[str] = None
    requirements_summary: Optional[str] = None
    education_requirement: Optional[str] = None
    experience_requirement: Optional[str] = None
    certification_requirement: Optional[str] = None
    reference_url: Optional[str] = None


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract up to three distinct educational leadership job openings mentioned in the answer.
    For each position, return an object with the following fields:
    - position_title: The exact position title as presented (e.g., "Principal", "Assistant Principal", "Superintendent", "Director of Curriculum")
    - district_name: The employing school district or equivalent education agency name
    - state: The U.S. state (use 2-letter abbreviation if present; otherwise the full state name)
    - salary_range: The salary range or explicit compensation text, if provided (e.g., "$125,000–$145,000", "$38–$42/hour", or "See salary schedule at <URL>")
    - application_deadline: The application deadline text as written (e.g., "Until Filled" or a specific date like "July 15, 2026")
    - requirements_summary: A brief, 1–2 sentence summary of key requirements (education, experience, certification)
    - education_requirement: The exact text snippet from the answer describing the education requirement (if present)
    - experience_requirement: The exact text snippet describing the experience requirement (if present)
    - certification_requirement: The exact text snippet describing the certification/licensure requirement (if present)
    - reference_url: The URL to the official job posting (district or official applicant tracking system). If multiple URLs are given, choose the official posting/application link. Only include URLs explicitly present in the answer.
    
    Rules:
    - Only extract URLs that are explicitly shown in the answer. Do not invent or infer URLs.
    - If any field is not provided in the answer, set it to null.
    - If more than three positions are provided, include only the first three in the 'positions' array.
    - Ensure 'state' is a valid U.S. state (full name or 2-letter code) if provided.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def pad_or_trim_positions(extracted: PositionsExtraction, k: int = 3) -> List[PositionItem]:
    items = list(extracted.positions or [])[:k]
    while len(items) < k:
        items.append(PositionItem())
    return items


def build_urls_list(primary_url: Optional[str]) -> List[str]:
    return [primary_url] if primary_url else []


# --------------------------------------------------------------------------- #
# Verification per-position                                                   #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    index_one_based: int,
    prior_states: List[Optional[str]]
) -> None:
    # Position container node (parallel aggregation; allow partial credit within each position)
    pos_node = evaluator.add_parallel(
        id=f"position_{index_one_based}",
        desc=(
            "Evaluation of the first educational leadership position" if index_one_based == 1 else
            "Evaluation of the second educational leadership position" if index_one_based == 2 else
            "Evaluation of the third educational leadership position"
        ),
        parent=parent_node,
        critical=False
    )

    # Existence check (critical): require at minimum title and posting URL to proceed
    exists_result = bool(pos.position_title and pos.position_title.strip() and pos.reference_url and pos.reference_url.strip())
    evaluator.add_custom_node(
        result=exists_result,
        id=f"position_{index_one_based}_exists",
        desc=(
            "A first qualifying educational leadership position is identified" if index_one_based == 1 else
            "A second qualifying educational leadership position is identified" if index_one_based == 2 else
            "A third qualifying educational leadership position is identified"
        ),
        parent=pos_node,
        critical=True
    )

    # Special cross-position constraint: different state for positions 2 and 3
    curr_state_code = canonicalize_state(pos.state)
    if index_one_based == 2:
        s1 = canonicalize_state(prior_states[0]) if len(prior_states) >= 1 else None
        diff_ok = bool(curr_state_code and s1 and curr_state_code != s1)
        evaluator.add_custom_node(
            result=diff_ok,
            id="position_2_different_state",
            desc="The second position is located in a different U.S. state than the first position",
            parent=pos_node,
            critical=True
        )
    if index_one_based == 3:
        s1 = canonicalize_state(prior_states[0]) if len(prior_states) >= 1 else None
        s2 = canonicalize_state(prior_states[1]) if len(prior_states) >= 2 else None
        diff_ok = bool(curr_state_code and s1 and s2 and curr_state_code not in {s1, s2})
        evaluator.add_custom_node(
            result=diff_ok,
            id="position_3_different_state",
            desc="The third position is located in a different U.S. state than the first two positions",
            parent=pos_node,
            critical=True
        )

    # Build source list for verification (single posting URL)
    src = build_urls_list(pos.reference_url)

    # 1) Principal-level or higher (critical)
    n_title_level = evaluator.add_leaf(
        id=f"position_{index_one_based}_title_level",
        desc=(
            "The first position is at principal level or higher (Superintendent, Principal, Assistant Principal, or equivalent district administrator)" if index_one_based == 1 else
            "The second position is at principal level or higher (Superintendent, Principal, Assistant Principal, or equivalent district administrator)" if index_one_based == 2 else
            "The third position is at principal level or higher (Superintendent, Principal, Assistant Principal, or equivalent district administrator)"
        ),
        parent=pos_node,
        critical=True
    )
    claim_title_level = (
        "This job posting is for a principal-level or higher role (e.g., Principal, Assistant/Associate Principal, "
        "Superintendent, Assistant/Associate/Deputy Superintendent), or a clearly equivalent district-level leadership "
        "administrator (e.g., Director/Executive Director/Chief Officer with district-wide scope)."
    )
    await evaluator.verify(
        claim=claim_title_level,
        node=n_title_level,
        sources=src,
        additional_instruction=(
            "Judge based on the actual position scope and title shown on the posting page. "
            "Accept clear synonyms (e.g., Head of School for principal; Chief/Director roles if they are district-level administrators). "
            "Reject roles that are not principal-level or district leadership (e.g., teacher, coach, coordinator without district-wide administrative scope)."
        )
    )

    # 2) Master's degree requirement (critical)
    n_masters = evaluator.add_leaf(
        id=f"position_{index_one_based}_master_degree",
        desc=(
            "The first position explicitly requires a Master's degree or higher in Educational Leadership, Educational Administration, or related field" if index_one_based == 1 else
            "The second position explicitly requires a Master's degree or higher in Educational Leadership, Educational Administration, or related field" if index_one_based == 2 else
            "The third position explicitly requires a Master's degree or higher in Educational Leadership, Educational Administration, or related field"
        ),
        parent=pos_node,
        critical=True
    )
    claim_masters = (
        "This posting explicitly requires a master's degree or higher in Educational Leadership, Educational Administration, "
        "or a closely related field (e.g., M.Ed./EdM/MAEd in leadership/administration, Ed.S., Ed.D., Ph.D. in education leadership/administration)."
    )
    await evaluator.verify(
        claim=claim_masters,
        node=n_masters,
        sources=src,
        additional_instruction=(
            "Look for explicit degree requirements in the minimum qualifications. Accept synonyms like 'Master's in School Administration', "
            "'M.Ed. in Educational Leadership', 'Ed.S./Ed.D.', or 'advanced degree in educational administration or closely related field'."
        )
    )

    # 3) Experience requirement (critical)
    n_experience = evaluator.add_leaf(
        id=f"position_{index_one_based}_experience",
        desc=(
            "The first position requires minimum 3 years of K-12 teaching or educational experience" if index_one_based == 1 else
            "The second position requires minimum 3 years of K-12 teaching or educational experience" if index_one_based == 2 else
            "The third position requires minimum 3 years of K-12 teaching or educational experience"
        ),
        parent=pos_node,
        critical=True
    )
    claim_experience = (
        "This posting requires at least three years of K-12 teaching or relevant K-12 educational experience "
        "(e.g., '3+ years', 'minimum of three years', 'at least 3 years')."
    )
    await evaluator.verify(
        claim=claim_experience,
        node=n_experience,
        sources=src,
        additional_instruction=(
            "Accept phrasing like '3-5 years', 'three or more years', or 'minimum three years'. "
            "The experience should be K-12 and educational in nature (teaching, instructional leadership, building or district administration)."
        )
    )

    # 4) Certification/licensure requirement (critical)
    n_cert = evaluator.add_leaf(
        id=f"position_{index_one_based}_certification",
        desc=(
            "The first position requires valid state administrator/principal/superintendent certification or licensure" if index_one_based == 1 else
            "The second position requires valid state administrator/principal/superintendent certification or licensure" if index_one_based == 2 else
            "The third position requires valid state administrator/principal/superintendent certification or licensure"
        ),
        parent=pos_node,
        critical=True
    )
    claim_cert = (
        "This posting requires a valid state-level administrator license/certificate appropriate to the role "
        "(e.g., principal certificate/endorsement, superintendent license/endorsement, or equivalent state administrator credential)."
    )
    await evaluator.verify(
        claim=claim_cert,
        node=n_cert,
        sources=src,
        additional_instruction=(
            "Accept language like 'valid [STATE] Principal Certification', 'appropriate administrative certificate', "
            "'Superintendent endorsement', or 'administrator licensure required'."
        )
    )

    # 5) Salary/compensation information (critical)
    n_salary = evaluator.add_leaf(
        id=f"position_{index_one_based}_salary",
        desc=(
            "The first position includes a verifiable salary range or compensation statement" if index_one_based == 1 else
            "The second position includes a verifiable salary range or compensation statement" if index_one_based == 2 else
            "The third position includes a verifiable salary range or compensation statement"
        ),
        parent=pos_node,
        critical=True
    )
    claim_salary = (
        "The posting includes verifiable salary or compensation information, such as an explicit numeric salary/range "
        "or a direct link to an official salary schedule."
    )
    await evaluator.verify(
        claim=claim_salary,
        node=n_salary,
        sources=src,
        additional_instruction=(
            "Accept explicit numeric amounts (annual/hourly) or a clear link to an official salary schedule (district or state). "
            "Do not accept only vague phrases like 'competitive' or 'commensurate with experience' with no numbers and no official schedule link."
        )
    )

    # 6) Application deadline in 2026 or 'Until Filled' (critical)
    n_deadline = evaluator.add_leaf(
        id=f"position_{index_one_based}_deadline",
        desc=(
            "The first position has a clearly stated application deadline (specific date or 'Until Filled')" if index_one_based == 1 else
            "The second position has a clearly stated application deadline (specific date or 'Until Filled')" if index_one_based == 2 else
            "The third position has a clearly stated application deadline (specific date or 'Until Filled')"
        ),
        parent=pos_node,
        critical=True
    )
    claim_deadline = (
        f"The posting states an application deadline that is either a specific date in {TARGET_DEADLINE_YEAR} "
        f"or uses the phrase 'Until Filled' (or a clear variant like 'Open Until Filled')."
    )
    await evaluator.verify(
        claim=claim_deadline,
        node=n_deadline,
        sources=src,
        additional_instruction=(
            f"Accept clear 2026 dates in any common format (e.g., 'July 1, {TARGET_DEADLINE_YEAR}', '2026-07-01') "
            f"or standard 'Until Filled' language ('Open until filled', 'Accepting applications until filled')."
        )
    )

    # 7) Location: district name and state (critical)
    n_location = evaluator.add_leaf(
        id=f"position_{index_one_based}_location",
        desc=(
            "The first position specifies the district name and U.S. state location" if index_one_based == 1 else
            "The second position specifies the district name and U.S. state location" if index_one_based == 2 else
            "The third position specifies the district name and U.S. state location"
        ),
        parent=pos_node,
        critical=True
    )
    district_display = pos.district_name or "the employing district"
    state_display = pos.state or "the stated U.S. state"
    claim_location = (
        f"The posting clearly identifies the employing school district (e.g., '{district_display}') "
        f"and the U.S. state location (e.g., '{state_display}')."
    )
    await evaluator.verify(
        claim=claim_location,
        node=n_location,
        sources=src,
        additional_instruction=(
            "It is sufficient if both the district (employer) name and the U.S. state can be identified from the posting page. "
            "Allow common abbreviations and naming variants (e.g., 'School District of X', 'X Public Schools', state postal codes)."
        )
    )

    # 8) Source is an official job posting (critical)
    n_source = evaluator.add_leaf(
        id=f"position_{index_one_based}_source",
        desc=(
            "The first position includes a valid reference URL to an official job posting" if index_one_based == 1 else
            "The second position includes a valid reference URL to an official job posting" if index_one_based == 2 else
            "The third position includes a valid reference URL to an official job posting"
        ),
        parent=pos_node,
        critical=True
    )
    employer_name = pos.district_name or "the hiring employer"
    claim_source = (
        f"This URL is an official job posting for {employer_name} (hosted on the district/agency site or its official "
        f"applicant tracking system), not a generic job board article or news page."
    )
    await evaluator.verify(
        claim=claim_source,
        node=n_source,
        sources=src,
        additional_instruction=(
            "Treat platforms like Frontline/Applitrack, TalentEd, iCIMS, Workday, and similar ATS as official if the posting "
            "shows the explicit employer (district/agency). Generic job aggregators without an official posting context should not count."
        )
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
    # Initialize evaluator with a parallel root (three positions evaluated independently for partial credit)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Evaluate whether the answer correctly identifies educational leadership job openings that meet all specified criteria",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured positions from the answer
    extracted_positions = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    # Normalize and prepare exactly three positions
    positions = pad_or_trim_positions(extracted_positions, k=3)

    # Record a small custom info block to aid debugging
    states_before_norm = [p.state for p in positions]
    states_after_norm = [canonicalize_state(p.state) for p in positions]
    evaluator.add_custom_info(
        info={
            "extracted_states_raw": states_before_norm,
            "extracted_states_normalized": states_after_norm,
            "target_deadline_year": TARGET_DEADLINE_YEAR
        },
        info_type="extraction_notes",
        info_name="position_state_notes"
    )

    # Verify each of the three positions; maintain prior states for cross-state checks
    prior_states: List[Optional[str]] = []
    for idx in range(3):
        await verify_position(
            evaluator=evaluator,
            parent_node=root,
            pos=positions[idx],
            index_one_based=idx + 1,
            prior_states=prior_states
        )
        # Update prior states list after verifying current position
        prior_states.append(positions[idx].state)

    # Return the structured evaluation summary
    return evaluator.get_summary()