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
TASK_ID = "nc_principal_positions"
TASK_DESCRIPTION = (
    "Identify four distinct school principal positions in North Carolina for the 2025-2026 or 2026-2027 school year "
    "that meet ALL of the following requirements:\n"
    "1. The position must be for a school principal (elementary, middle, or high school level)\n"
    "2. The position must require a master's degree in educational leadership, educational administration, "
    "school administration, or a closely related field\n"
    "3. The position must require a minimum of 3 years of teaching experience\n"
    "4. The position must require or accept North Carolina principal licensure/certification\n"
    "5. The position must include verifiable salary information or salary range in the job posting\n\n"
    "For each position, provide:\n"
    "- The specific school name and school district\n"
    "- Confirmation that the position requires a master's degree in the specified fields\n"
    "- Confirmation that the position requires at least 3 years of teaching experience\n"
    "- Confirmation that North Carolina principal licensure is required or accepted\n"
    "- The salary information or salary range listed for the position\n"
    "- A reference URL to the job posting"
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PositionInfo(BaseModel):
    """Model representing one principal position as extracted from the agent's answer."""
    school_name: Optional[str] = None
    district_name: Optional[str] = None
    level: Optional[str] = None  # e.g., "elementary", "middle", "high" (or any textual variant from the answer)
    school_year_text: Optional[str] = None  # e.g., "2025-2026", "2026-2027", or textual variants
    masters_requirement_text: Optional[str] = None  # the phrasing cited in the answer
    teaching_experience_required_years: Optional[str] = None  # e.g., "3 years", "three (3) years", "3+ years"
    nc_licensure_text: Optional[str] = None  # the phrasing cited in the answer
    salary_info: Optional[str] = None  # e.g., "$80,000 - $95,000", "per NC state schedule range $X-Y"
    job_posting_url: Optional[str] = None  # direct URL to the job posting
    additional_urls: List[str] = Field(default_factory=list)  # any other URL(s) cited for this position


class PositionsExtraction(BaseModel):
    """Container for multiple positions extracted from the answer."""
    positions: List[PositionInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return (
        "Extract up to 6 principal positions mentioned in the answer (we will later consider the first four). "
        "For each position, extract the following fields exactly as stated in the answer:\n"
        "1. school_name: The specific school name (e.g., 'Lincoln High School')\n"
        "2. district_name: The school district (e.g., 'Wake County Public School System')\n"
        "3. level: The school level if specified (e.g., 'elementary', 'middle', 'high', 'K-8', etc.). "
        "   Do NOT invent; use what the answer states. If missing, set to null.\n"
        "4. school_year_text: The school year as stated (e.g., '2025-2026', '2026-2027', '2025-26'), if present; "
        "   otherwise null.\n"
        "5. masters_requirement_text: The text in the answer indicating a master's degree is required in relevant fields; "
        "   if missing, null.\n"
        "6. teaching_experience_required_years: The minimum teaching experience as stated (ideally '3 years' or equivalent); "
        "   if missing, null.\n"
        "7. nc_licensure_text: The text indicating NC principal licensure/certification (required or accepted); "
        "   if missing, null.\n"
        "8. salary_info: The salary or salary range as provided in the answer; if missing, null.\n"
        "9. job_posting_url: The primary URL of the job posting for this position; if missing, null.\n"
        "10. additional_urls: Any other URLs cited in the answer that refer to this position (array; can be empty).\n\n"
        "GENERAL RULES:\n"
        "- Do not invent or infer information; only extract what appears in the answer.\n"
        "- For URLs: extract only valid, explicit URLs present in the answer (plain or markdown). Ignore invalid URLs.\n"
        "- If any field is not stated, set it to null (or empty array for additional_urls).\n"
        "- Return a JSON object with a 'positions' array containing these position objects."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def ordinal_from_index(idx: int) -> str:
    ords = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]
    return ords[idx] if idx < len(ords) else f"Position #{idx + 1}"


def build_sources_list(pos: PositionInfo) -> List[str]:
    """Build a deduplicated sources list prioritizing the job posting URL."""
    urls: List[str] = []
    if pos.job_posting_url and pos.job_posting_url.strip():
        urls.append(pos.job_posting_url.strip())
    for u in pos.additional_urls:
        if u and u.strip():
            urls.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_single_position(
    evaluator: Evaluator,
    parent_node,
    position: PositionInfo,
    index: int,
) -> None:
    """
    Build and execute verification checks for one position.
    Each leaf is a single verification step with a binary outcome.
    """
    pos_label = ordinal_from_index(index)
    position_node = evaluator.add_parallel(
        id=f"Position_{index + 1}",
        desc=f"{pos_label} qualifying principal position in North Carolina",
        parent=parent_node,
        critical=False  # Allow partial scoring across positions; each leaf under is critical
    )

    # 0) Reference URL presence (critical existence as per rubric)
    #    This is an existence check – directly judged via custom node.
    ref_url_present = bool(position.job_posting_url) and bool(position.job_posting_url.strip())
    ref_url_node = evaluator.add_custom_node(
        result=ref_url_present,
        id=f"Position_{index + 1}_Reference_URL",
        desc=f"{pos_label}: Provides a reference URL to the job posting",
        parent=position_node,
        critical=True
    )

    # Common sources and prerequisite (we gate all verifications on URL presence)
    sources = build_sources_list(position)
    extra_prereqs = [ref_url_node]

    # 1) School level (principal at elementary/middle/high)
    level_node = evaluator.add_leaf(
        id=f"Position_{index + 1}_School_Level",
        desc=f"{pos_label}: Verifies the position is for a principal at the elementary, middle, or high school level",
        parent=position_node,
        critical=True
    )
    if position.level and position.level.strip():
        level_claim = (
            f"The job posting indicates this is a school principal role at a {position.level.strip()} school level "
            f"(not assistant principal or district-level)."
        )
    else:
        level_claim = (
            "The job posting indicates this is a school principal role at the elementary, middle, or high school level "
            "(not assistant principal or district-level)."
        )
    await evaluator.verify(
        claim=level_claim,
        node=level_node,
        sources=sources,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Confirm it is a school principal position (not assistant principal and not central-office director). "
            "Determine the school level from the posting: elementary ≈ K-5/primary, middle ≈ 6-8/junior high/K-8, "
            "high ≈ 9-12/senior high. Allow reasonable naming variants."
        ),
    )

    # 2) School year (2025-2026 or 2026-2027)
    school_year_node = evaluator.add_leaf(
        id=f"Position_{index + 1}_School_Year",
        desc=f"{pos_label}: Verifies the position is for the 2025-2026 or 2026-2027 school year",
        parent=position_node,
        critical=True
    )
    if position.school_year_text and position.school_year_text.strip():
        year_claim = (
            f"The job posting shows the position is for the {position.school_year_text.strip()} school year, "
            "which must be one of 2025-2026 or 2026-2027."
        )
    else:
        year_claim = (
            "The job posting shows the position is for the 2025-2026 or 2026-2027 school year."
        )
    await evaluator.verify(
        claim=year_claim,
        node=school_year_node,
        sources=sources,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Accept explicit forms like '2025-2026', '2025-26', '2026-2027', '2026-27'. "
            "Alternatively, accept clear references to a start date that falls within these school years "
            "(e.g., July/Aug 2025 implies 2025-2026; July/Aug 2026 implies 2026-2027). "
            "If such evidence is absent, mark as not supported."
        ),
    )

    # 3) School and district (and NC location)
    district_node = evaluator.add_leaf(
        id=f"Position_{index + 1}_School_District",
        desc=f"{pos_label}: Identifies the specific school name and district for the position",
        parent=position_node,
        critical=True
    )
    school_name = position.school_name.strip() if position.school_name else None
    district_name = position.district_name.strip() if position.district_name else None
    if school_name and district_name:
        district_claim = (
            f"The job posting identifies the school as '{school_name}' and the district as '{district_name}', "
            "and it is located in North Carolina."
        )
    else:
        district_claim = (
            "The job posting identifies the specific school name and school district in North Carolina."
        )
    await evaluator.verify(
        claim=district_claim,
        node=district_node,
        sources=sources,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Verify the page names the school and the district, and that this position is in North Carolina. "
            "Allow minor naming variations or abbreviations (e.g., 'County Schools', 'City Schools', 'Public Schools')."
        ),
    )

    # 4) Master's degree requirement (in specified fields)
    masters_node = evaluator.add_leaf(
        id=f"Position_{index + 1}_Masters_Degree",
        desc=f"{pos_label}: Verifies the position requires a master's degree in educational leadership/administration/school administration or related",
        parent=position_node,
        critical=True
    )
    if position.masters_requirement_text and position.masters_requirement_text.strip():
        masters_claim = (
            f"The job posting includes the requirement: {position.masters_requirement_text.strip()} "
            "— which indicates a master's degree in educational leadership, educational administration, "
            "school administration, or a closely related field is required."
        )
    else:
        masters_claim = (
            "The job posting requires a master's degree in educational leadership, educational administration, "
            "school administration, or a closely related field."
        )
    await evaluator.verify(
        claim=masters_claim,
        node=masters_node,
        sources=sources,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Look for language such as 'Master’s degree in School Administration', 'Educational Leadership', "
            "'Educational Administration', or equivalent (e.g., M.Ed., Ed.S.) indicating the master's degree is required."
        ),
    )

    # 5) Teaching experience minimum (>= 3 years)
    teach_exp_node = evaluator.add_leaf(
        id=f"Position_{index + 1}_Teaching_Experience",
        desc=f"{pos_label}: Verifies the position requires a minimum of 3 years of teaching experience",
        parent=position_node,
        critical=True
    )
    if position.teaching_experience_required_years and position.teaching_experience_required_years.strip():
        teach_claim = (
            f"The job posting requires at least {position.teaching_experience_required_years.strip()} of teaching experience, "
            "meeting or exceeding 3 years."
        )
    else:
        teach_claim = "The job posting requires a minimum of 3 years of teaching experience."
    await evaluator.verify(
        claim=teach_claim,
        node=teach_exp_node,
        sources=sources,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Confirm the posting clearly requires at least three years (e.g., '3 years', 'three (3) years'). "
            "If it only says 'preferred' without minimum required, do not mark as supported."
        ),
    )

    # 6) NC principal licensure requirement or acceptance
    licensure_node = evaluator.add_leaf(
        id=f"Position_{index + 1}_NC_Licensure",
        desc=f"{pos_label}: Verifies the position requires or accepts North Carolina principal licensure",
        parent=position_node,
        critical=True
    )
    if position.nc_licensure_text and position.nc_licensure_text.strip():
        licensure_claim = (
            f"The job posting states: {position.nc_licensure_text.strip()} — indicating NC principal licensure/certification "
            "is required or accepted."
        )
    else:
        licensure_claim = "The job posting requires or accepts North Carolina principal licensure/certification."
    await evaluator.verify(
        claim=licensure_claim,
        node=licensure_node,
        sources=sources,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Look for terms like 'NC School Administrator license', 'Principal license', 'eligible for NC principal licensure'. "
            "If other states are mentioned, verify that NC licensure is required/accepted."
        ),
    )

    # 7) Salary information present (verifiable)
    salary_node = evaluator.add_leaf(
        id=f"Position_{index + 1}_Salary_Information",
        desc=f"{pos_label}: Provides verifiable salary information or salary range for the position",
        parent=position_node,
        critical=True
    )
    if position.salary_info and position.salary_info.strip():
        salary_claim = f"The job posting includes salary information: {position.salary_info.strip()}."
    else:
        salary_claim = "The job posting includes verifiable salary information or a salary range."
    await evaluator.verify(
        claim=salary_claim,
        node=salary_node,
        sources=sources,
        extra_prerequisites=extra_prereqs,
        additional_instruction=(
            "Confirm that explicit salary figures or a clear salary range are shown on the job posting. "
            "References only to 'per state schedule' without any numbers generally should not be considered sufficient "
            "unless a specific range or figures are included."
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
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the North Carolina principal positions task.
    """
    # Initialize evaluator (root node is non-critical to allow partial credit across positions)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The top-level rubric uses parallel aggregation
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

    # Record the requirements for context (GT-like info)
    evaluator.add_ground_truth({
        "requirements": [
            "Principal position (elementary/middle/high)",
            "Master's degree in educational leadership/administration/school administration or closely related",
            "Minimum 3 years teaching experience",
            "NC principal licensure required or accepted",
            "Verifiable salary info in posting",
            "Provide job posting URL"
        ],
        "school_years": ["2025-2026", "2026-2027"]
    }, gt_type="task_requirements")

    # Extract positions from the answer
    extracted_positions: PositionsExtraction = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )

    # Keep only the first four positions, pad with empty if fewer provided
    positions: List[PositionInfo] = list(extracted_positions.positions[:4])
    while len(positions) < 4:
        positions.append(PositionInfo())

    # Build four position verification subtrees
    # Following the rubric: four position groups under the root (parallel).
    for idx in range(4):
        await verify_single_position(
            evaluator=evaluator,
            parent_node=root,
            position=positions[idx],
            index=idx
        )

    # Return summary with verification tree
    return evaluator.get_summary()