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
TASK_ID = "educational_admin_positions_2026_2027"
TASK_DESCRIPTION = """
You are providing career guidance to a teacher who is ready to transition into educational administration. This teacher holds a master's degree in educational leadership (earned in 2023), has 6 years of K-12 teaching experience in Virginia public schools, and currently holds an active teaching license in Virginia. They are open to relocating anywhere in the United States and are specifically interested in understanding the range of administrative opportunities available for the 2026-2027 school year.

Identify four distinct educational leadership positions currently posted for the 2026-2027 school year that this candidate would qualify for based on their credentials. The four positions must satisfy the following criteria:
- The positions must be located in at least three different U.S. states
- At least one position must be an assistant principal role
- At least one position must be at the principal level or higher (such as principal, director of curriculum and instruction, director of student services, or similar administrative position)
- Each position's minimum qualification requirements (education, experience, and certification) must be met by the candidate's credentials

For each of the four positions, provide:
1. The specific school district name
2. The exact position title
3. The state where the position is located
4. Confirmation that the position is for the 2026-2027 school year
5. Verification that the candidate meets all minimum qualification requirements
6. The salary range or starting salary (if available in the posting)
"""

# Candidate profile (used in verification prompts)
CANDIDATE_SUMMARY = (
    "Candidate holds a Master's degree in Educational Leadership (earned in 2023), "
    "has 6 years of K-12 teaching experience in Virginia public schools, "
    "and currently holds an active Virginia teaching license. "
    "They are open to relocating anywhere in the United States."
)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    district_name: Optional[str] = None
    position_title: Optional[str] = None
    state: Optional[str] = None  # Prefer the full state name or 2-letter code only
    school_year: Optional[str] = None  # The school year text mentioned in the answer, if any
    posting_urls: List[str] = Field(default_factory=list)  # One or more direct job posting URLs
    salary_info: Optional[str] = None  # Salary range or starting salary text, if provided in the answer


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract up to FOUR distinct educational leadership positions from the answer, in the order they appear.
    For each position, extract the following fields:
    - district_name: The specific school district or employer name as stated
    - position_title: The exact position title as stated (e.g., "Assistant Principal", "Principal", "Director of Curriculum and Instruction")
    - state: The U.S. state where the position is located (prefer full state name or 2-letter code only, no city)
    - school_year: The school year as referenced for the posting (e.g., "2026-2027", "SY 2026-27"), if present in the answer text
    - posting_urls: All explicit job posting URLs cited for this position (do not infer; extract actual URLs from the answer; include multiple if provided)
    - salary_info: Salary range or starting salary text if provided in the answer text for this position; otherwise null

    Return a JSON object with a top-level "positions" array of up to four PositionItem objects.

    IMPORTANT:
    - Only extract URLs directly present in the answer (plain links or markdown links).
    - Do not fabricate or search for URLs; return an empty list if none are provided in the answer.
    - Keep the state field as a clean state value (e.g., "Texas" or "TX"), not city+state.
    - If any field is missing for a position, set it to null (or [] for posting_urls).
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def first_n_positions(extracted: PositionsExtraction, n: int = 4) -> List[PositionItem]:
    items = (extracted.positions or [])[:n]
    while len(items) < n:
        items.append(PositionItem())
    return items


def titles_list(positions: List[PositionItem]) -> List[str]:
    return [p.position_title or "" for p in positions]


def states_list(positions: List[PositionItem]) -> List[str]:
    return [p.state or "" for p in positions]


def unique_nonempty(items: List[str]) -> List[str]:
    return sorted(list({x.strip() for x in items if x and x.strip()}))


def is_assistant_principal_title(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return False
    candidates = [
        "assistant principal",
        "asst principal",
        "associate principal",
        "vice principal",
        "ap -",  # heuristic, edge cases
        "ap (assistant principal)",
    ]
    if any(kw in t for kw in candidates):
        return True
    # common shorthand "AP" alone is ambiguous; avoid
    return False


def is_principal_or_higher_title(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return False
    # Must include principal without assistant qualifiers, or clearly higher roles
    if "principal" in t and not any(q in t for q in ["assistant principal", "asst principal", "vice principal", "associate principal"]):
        return True
    higher = [
        "director", "superintendent", "chief", "executive director",
        "head of school", "dean", "assistant superintendent", "associate superintendent"
    ]
    return any(h in t for h in higher)


def build_salary_claim(position: PositionItem) -> str:
    if position.salary_info and position.salary_info.strip():
        return f"The job posting includes salary information that matches or is consistent with: {position.salary_info}."
    # Generic presence check if no salary extracted from answer
    return "The job posting includes salary information (a salary range or a starting salary)."


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def verify_position(
    evaluator: Evaluator,
    parent_node,
    position: PositionItem,
    idx: int,
) -> None:
    """
    Build and verify the subtree for a single position (index idx in [0..3]).
    """

    # Container node for this position (non-critical; allows partial credit per-position)
    pos_node = evaluator.add_parallel(
        id=f"position_{idx+1}",
        desc=f"Position #{idx+1} verification",
        parent=parent_node,
        critical=False
    )

    # Quick gate: posting URL(s) present
    urls_present = evaluator.add_custom_node(
        result=bool(position.posting_urls),
        id=f"position_{idx+1}_urls_present",
        desc=f"Position #{idx+1}: Posting URL(s) are provided in the answer",
        parent=pos_node,
        critical=True
    )

    # Identification node (critical: district, title, state, and 2026-2027 school year must be correctly supported)
    ident_node = evaluator.add_parallel(
        id=f"position_{idx+1}_identification",
        desc="Correct district, exact title, state location, and 2026-2027 timeframe supported by the posting",
        parent=pos_node,
        critical=True
    )

    # Required fields existence (district, title, state + at least one posting URL)
    required_fields = evaluator.add_custom_node(
        result=bool(position.district_name and position.position_title and position.state and position.posting_urls),
        id=f"position_{idx+1}_required_fields",
        desc=f"Position #{idx+1}: Has district, title, state, and posting URL(s) in the answer",
        parent=ident_node,
        critical=True
    )

    # District verification
    district_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_district_supported",
        desc=f"Position #{idx+1}: The posting clearly indicates the district/employer is '{position.district_name}'",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The job posting's district/employer matches '{position.district_name}'. "
              f"Treat common suffixes like 'Public Schools', 'ISD', 'USD', or 'School District' as equivalent.",
        node=district_leaf,
        sources=position.posting_urls,
        additional_instruction="Check the posting header/footer, employer banner, or 'About' sections for district/employer identity. Allow minor naming variants."
    )

    # Title verification
    title_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_title_supported",
        desc=f"Position #{idx+1}: The posting's position title matches '{position.position_title}'",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The job posting's position title matches or is equivalent to '{position.position_title}'.",
        node=title_leaf,
        sources=position.posting_urls,
        additional_instruction="Allow minor formatting/order variants (e.g., 'Assistant Principal - High School' vs 'High School Assistant Principal')."
    )

    # State verification
    state_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_state_supported",
        desc=f"Position #{idx+1}: The posting indicates the state is '{position.state}'",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The job is located in the U.S. state of {position.state}.",
        node=state_leaf,
        sources=position.posting_urls,
        additional_instruction="City references or addresses within this state count as valid evidence. State abbreviations (e.g., 'TX' for Texas) are acceptable."
    )

    # School year (2026-2027) verification
    year_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_year_2026_2027",
        desc=f"Position #{idx+1}: The posting confirms the 2026-2027 school year",
        parent=ident_node,
        critical=True
    )
    await evaluator.verify(
        claim="The job posting indicates the role is for the 2026-2027 school year.",
        node=year_leaf,
        sources=position.posting_urls,
        additional_instruction="Accept variants like '2026–2027', '2026-27', 'SY 2026-27', or a contract period spanning July 2026 to June 2027."
    )

    # Qualification match (critical)
    qual_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_qualification_match",
        desc=f"Position #{idx+1}: Candidate meets the minimum education, experience, and certification/licensure requirements",
        parent=pos_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "Given the candidate's credentials (Master's in Educational Leadership earned in 2023; "
            "6 years of K-12 teaching experience; active Virginia teaching license; open to relocation), "
            "the candidate meets or can satisfy the posting's minimum qualifications for education, experience, and certification/licensure."
        ),
        node=qual_leaf,
        sources=position.posting_urls,
        additional_instruction=(
            f"{CANDIDATE_SUMMARY} "
            "Evaluate the 'Minimum Requirements' or equivalent section. If the posting states acceptance of 'equivalent' "
            "licenses, reciprocity, or 'eligible/ability to obtain appropriate administrator license by start date', "
            "consider that as meeting requirements. If an explicit, non-waivable in-state administrator certificate is "
            "required with no path to obtain before start, judge as NOT meeting."
        )
    )

    # Salary information (non-critical)
    salary_leaf = evaluator.add_leaf(
        id=f"position_{idx+1}_salary_info",
        desc=f"Position #{idx+1}: Salary information presence/consistency is supported by the posting (if available)",
        parent=pos_node,
        critical=False
    )
    await evaluator.verify(
        claim=build_salary_claim(position),
        node=salary_leaf,
        sources=position.posting_urls,
        additional_instruction="If salary is stated in the posting, confirm presence and general consistency with any provided numbers."
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
    """
    Entry point for evaluating an answer for the educational administration positions task.
    """
    # Initialize evaluator (root is non-critical parallel aggregator)
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

    # Extract up to 4 positions from the answer
    extracted_positions = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction",
    )
    positions = first_n_positions(extracted_positions, 4)

    # Record candidate profile info for transparency
    evaluator.add_custom_info(
        info={"candidate_profile": CANDIDATE_SUMMARY},
        info_type="candidate",
        info_name="candidate_profile"
    )

    # ----------------- Overall set-level requirements (critical) ----------------- #
    overall_set_node = evaluator.add_parallel(
        id="overall_position_set_requirements",
        desc="Verification that the set of four positions meets diversity requirements",
        parent=root,
        critical=True
    )

    # Geographic diversity: at least 3 different states across the 4 positions
    states = unique_nonempty(states_list(positions))
    geo_leaf = evaluator.add_custom_node(
        result=len(states) >= 3,
        id="geographic_diversity",
        desc=f"The four positions are located in at least three different U.S. states (found states: {states})",
        parent=overall_set_node,
        critical=True
    )

    # Role diversity: at least one AP AND at least one principal-level or higher
    role_node = evaluator.add_parallel(
        id="role_diversity",
        desc="The set includes at least one assistant principal and at least one principal-level or higher role",
        parent=overall_set_node,
        critical=True
    )

    titles = titles_list(positions)

    has_ap = any(is_assistant_principal_title(t) for t in titles)
    has_principal_higher = any(is_principal_or_higher_title(t) for t in titles)

    # Split into two atomic checks (both critical)
    has_ap_leaf = evaluator.add_custom_node(
        result=has_ap,
        id="has_assistant_principal_role",
        desc=f"At least one Assistant Principal role present among titles: {titles}",
        parent=role_node,
        critical=True
    )

    has_principal_or_higher_leaf = evaluator.add_custom_node(
        result=has_principal_higher,
        id="has_principal_or_higher_role",
        desc=f"At least one Principal-level or higher role present among titles: {titles}",
        parent=role_node,
        critical=True
    )

    # ----------------- Per-position verification ----------------- #
    for i, pos in enumerate(positions, start=1):
        await verify_position(evaluator, root, pos, i - 1)

    # Add a small summary of extracted items for debugging/traceability
    evaluator.add_custom_info(
        info={
            "extracted_positions_count": len([p for p in positions if any([p.district_name, p.position_title, p.state, p.posting_urls])]),
            "states_found": states,
            "titles_found": titles,
            "urls_counts": [len(p.posting_urls) for p in positions],
        },
        info_type="extraction_summary",
        info_name="positions_snapshot"
    )

    return evaluator.get_summary()