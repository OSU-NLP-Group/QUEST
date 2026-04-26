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
TASK_ID = "se_r1_sec_acc_universities"
TASK_DESCRIPTION = (
    "Identify four universities located in states within the southeastern United States that meet all of the following "
    "criteria:\n\n"
    "1. Regional Accreditation: The university must be accredited by the Southern Association of Colleges and Schools "
    "Commission on Colleges (SACSCOC) to award graduate degrees.\n"
    "2. Research Classification: The university must hold the R1 Carnegie Classification (Research 1: Very High Research "
    "Spending and Doctorate Production), which requires spending at least $50 million annually on research and development "
    "and awarding at least 70 research doctorates annually.\n"
    "3. Athletic Conference Membership: The university must be a member of either the Southeastern Conference (SEC) or "
    "the Atlantic Coast Conference (ACC) as of 2026.\n"
    "4. Football Stadium Capacity: The university's primary football stadium must have a seating capacity exceeding 80,000.\n"
    "5. Basketball Arena Capacity: The university's primary basketball arena must have a seating capacity exceeding 20,000.\n"
    "6. Graduate Education Programs: The university must offer master's degree programs in education or related fields "
    "that require at least 30 semester credit hours, consistent with SACSCOC requirements.\n\n"
    "For each of the four universities, provide: the full official university name, state, confirmation of SACSCOC accreditation, "
    "confirmation of R1 Carnegie Classification, the SEC/ACC conference, the football stadium (name + capacity), the basketball "
    "arena (name + capacity), at least one example of a graduate education master's program with ≥30 credit hours, and a reference "
    "URL for each major requirement verified."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    # Basic
    official_name: Optional[str] = None
    state: Optional[str] = None

    # SACSCOC
    sacscoc_urls: List[str] = Field(default_factory=list)

    # Carnegie R1
    carnegie_urls: List[str] = Field(default_factory=list)
    r1_support_urls: List[str] = Field(
        default_factory=list,
        description="Optional extra URLs (e.g., Carnegie methodology or NSF HERD stats pages) that support R1 thresholds."
    )

    # Conference
    conference: Optional[str] = None  # Expected 'SEC' or 'ACC'
    conference_urls: List[str] = Field(default_factory=list)

    # Football stadium
    stadium_name: Optional[str] = None
    stadium_capacity: Optional[str] = None  # Keep as string per framework robustness guidance
    stadium_urls: List[str] = Field(default_factory=list)

    # Basketball arena
    arena_name: Optional[str] = None
    arena_capacity: Optional[str] = None  # Keep as string
    arena_urls: List[str] = Field(default_factory=list)

    # Graduate education program
    program_name: Optional[str] = None
    program_credit_hours: Optional[str] = None  # e.g., "30 credit hours", "36 hours", etc.
    grad_program_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four universities mentioned in the answer that purport to satisfy all specified requirements.
    For each university, return a JSON object with the following fields (use null for any missing scalar field and [] for any missing URL list):

    - official_name: The full official name of the university
    - state: The U.S. state where the university's main campus is located

    - sacscoc_urls: An array of URL(s) that confirm current SACSCOC accreditation and that the institution can award master's and/or doctoral degrees
      (e.g., SACSCOC member directory entry or the university's official SACSCOC accreditation page)

    - carnegie_urls: An array of URL(s) that directly confirm the university holds the R1 Carnegie Classification
      (e.g., Carnegie Classification institution lookup or an official site that cites the classification)
    - r1_support_urls: An array of supplemental URL(s) that explicitly state the R1 thresholds (>= $50M annual R&D and >= 70 research doctorates)
      or otherwise confirm the institution meets them (e.g., Carnegie methodology page, NSF HERD statistics page for the institution)

    - conference: The athletic conference name, expected 'SEC' or 'ACC'
    - conference_urls: URL(s) confirming the 2026 membership in that conference (official conference page, university athletics page, or reliable directory)

    - stadium_name: The official name of the university's primary football stadium
    - stadium_capacity: The seating capacity as written in the answer (string form; do not parse to number)
    - stadium_urls: URL(s) that confirm the football stadium capacity

    - arena_name: The official name of the university's primary men's basketball arena
    - arena_capacity: The seating capacity as written in the answer (string form)
    - arena_urls: URL(s) that confirm the basketball arena capacity

    - program_name: The name of one example master's degree program in education or a closely related field (e.g., Curriculum & Instruction, Educational Leadership, Special Education, Higher Education)
    - program_credit_hours: The stated credit-hour requirement string for that program (e.g., "30 credit hours", "36 hours", etc.)
    - grad_program_urls: URL(s) for the specific graduate program page or the graduate catalog page that confirm the program exists and its credit-hour requirement

    IMPORTANT:
    - Only extract URLs explicitly present in the answer. Do not invent or infer URLs.
    - Include all distinct URLs mentioned in the answer that are relevant to each field.
    - If the answer lists more than four universities, extract only the first four.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _norm_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


def _merge_urls(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for lst in url_lists:
        for u in lst:
            if u and u not in seen:
                merged.append(u)
                seen.add(u)
    return merged


# --------------------------------------------------------------------------- #
# University verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_university(evaluator: Evaluator, parent_node, uni: UniversityItem, idx: int) -> None:
    """
    Build the verification subtree for one university and run all checks.
    IDs follow the rubric's naming (u{idx}_...).
    """
    u_node = evaluator.add_parallel(
        id=f"university_{idx}",
        desc=f"University #{idx} meeting all specified requirements",
        parent=parent_node,
        critical=False
    )

    # --------------------------- Basic information --------------------------- #
    basic_node = evaluator.add_parallel(
        id=f"u{idx}_basic_information",
        desc="Basic identifying information for the university is provided",
        parent=u_node,
        critical=True
    )
    name_ok = bool(uni.official_name and uni.official_name.strip())
    state_ok = bool(uni.state and uni.state.strip())

    name_node = evaluator.add_custom_node(
        result=name_ok,
        id=f"u{idx}_official_name",
        desc="The full official name of the university is provided",
        parent=basic_node,
        critical=True
    )
    state_node = evaluator.add_custom_node(
        result=state_ok,
        id=f"u{idx}_state_location",
        desc="The state where the university is located is provided",
        parent=basic_node,
        critical=True
    )

    # --------------------- SACSCOC accreditation (critical) ----------------- #
    sacscoc_node = evaluator.add_parallel(
        id=f"u{idx}_sacscoc_accreditation",
        desc="University is accredited by SACSCOC to award graduate degrees",
        parent=u_node,
        critical=True
    )

    sacscoc_ref_present = len(_norm_urls(uni.sacscoc_urls)) > 0
    sacscoc_ref_node = evaluator.add_custom_node(
        result=sacscoc_ref_present,
        id=f"u{idx}_sacscoc_reference",
        desc="URL reference confirming SACSCOC accreditation status",
        parent=sacscoc_node,
        critical=True
    )

    sacscoc_status_leaf = evaluator.add_leaf(
        id=f"u{idx}_sacscoc_status_verified",
        desc="Current SACSCOC accreditation status is verified as 'Accredited' and authorizes awarding master's and doctoral degrees",
        parent=sacscoc_node,
        critical=True
    )
    sacscoc_claim = (
        f"{uni.official_name} is accredited by the Southern Association of Colleges and Schools Commission on Colleges "
        f"(SACSCOC) and authorized to award graduate degrees (master's and/or doctoral degrees)."
    )
    await evaluator.verify(
        claim=sacscoc_claim,
        node=sacscoc_status_leaf,
        sources=_norm_urls(uni.sacscoc_urls),
        additional_instruction=(
            "Use the SACSCOC member directory or the university's official SACSCOC accreditation page. "
            "Wording may indicate 'Level VI (Doctorate)' or explicitly list 'master's' and 'doctoral' degrees. "
            "Confirm that the institution is currently 'Accredited' (not candidate/probation) and can award graduate degrees."
        ),
        extra_prerequisites=[sacscoc_ref_node, name_node, state_node]
    )

    # ------------------- Carnegie R1 classification (critical) -------------- #
    carnegie_node = evaluator.add_parallel(
        id=f"u{idx}_carnegie_r1_classification",
        desc="University holds R1 Carnegie Classification",
        parent=u_node,
        critical=True
    )

    # First, verify the R1 classification via a URL
    carnegie_leaf = evaluator.add_leaf(
        id=f"u{idx}_carnegie_reference",
        desc="URL reference confirming R1 Carnegie Classification",
        parent=carnegie_node,
        critical=True
    )
    carnegie_claim = (
        f"{uni.official_name} is classified as an R1 institution "
        f"(Doctoral Universities: Very High Research Activity) by the Carnegie Classification."
    )
    await evaluator.verify(
        claim=carnegie_claim,
        node=carnegie_leaf,
        sources=_norm_urls(uni.carnegie_urls),
        additional_instruction=(
            "Verify that the page explicitly indicates the institution is 'R1' or 'Doctoral Universities: Very High Research Activity'. "
            "Accept official Carnegie Classification institution lookups or authoritative institutional listings that cite the classification."
        ),
        extra_prerequisites=[name_node]
    )

    # Then verify the two threshold implications using Carnegie methodology or HERD stats if provided
    r1_support_sources = _merge_urls(_norm_urls(uni.r1_support_urls), _norm_urls(uni.carnegie_urls))

    r1_spend_leaf = evaluator.add_leaf(
        id=f"u{idx}_research_expenditure",
        desc="University spends at least $50 million annually on research and development",
        parent=carnegie_node,
        critical=True
    )
    r1_spend_claim = (
        f"By virtue of holding the Carnegie R1 classification, {uni.official_name} meets the R1 research-expenditure "
        f"threshold of at least $50 million in annual research and development spending."
    )
    await evaluator.verify(
        claim=r1_spend_claim,
        node=r1_spend_leaf,
        sources=r1_support_sources,
        additional_instruction=(
            "Supported if the provided source is the Carnegie methodology (or equivalent authoritative description) that explicitly states "
            "the >= $50M threshold for R1, or if the source provides official HERD data showing >= $50M for the institution. "
            "If none of the provided URLs establish this threshold or the institution meeting it, mark as not supported."
        ),
        extra_prerequisites=[carnegie_leaf]
    )

    r1_docs_leaf = evaluator.add_leaf(
        id=f"u{idx}_doctorate_production",
        desc="University awards at least 70 research doctorates annually",
        parent=carnegie_node,
        critical=True
    )
    r1_docs_claim = (
        f"By virtue of holding the Carnegie R1 classification, {uni.official_name} meets the doctorate-production threshold "
        f"of awarding at least 70 research doctorates annually."
    )
    await evaluator.verify(
        claim=r1_docs_claim,
        node=r1_docs_leaf,
        sources=r1_support_sources,
        additional_instruction=(
            "Supported if the provided source is the Carnegie methodology (or equivalent authoritative description) that explicitly states "
            "the >= 70 research doctorates threshold for R1, or if the source provides official statistics for the institution showing >= 70. "
            "If none of the provided URLs establish this threshold or the institution meeting it, mark as not supported."
        ),
        extra_prerequisites=[carnegie_leaf]
    )

    # -------------------- Conference membership (critical) ------------------ #
    conf_node = evaluator.add_parallel(
        id=f"u{idx}_conference_membership",
        desc="University is a member of either SEC or ACC conference",
        parent=u_node,
        critical=True
    )

    conf_ref_present = len(_norm_urls(uni.conference_urls)) > 0
    conf_ref_node = evaluator.add_custom_node(
        result=conf_ref_present,
        id=f"u{idx}_conference_reference",
        desc="URL reference confirming conference membership",
        parent=conf_node,
        critical=True
    )

    conf_leaf = evaluator.add_leaf(
        id=f"u{idx}_conference_identified",
        desc="Specific conference membership (SEC or ACC) is identified and verified for 2026",
        parent=conf_node,
        critical=True
    )
    conf_name = (uni.conference or "").strip()
    conf_claim = f"As of 2026, {uni.official_name} is a member of the {conf_name}."
    await evaluator.verify(
        claim=conf_claim,
        node=conf_leaf,
        sources=_norm_urls(uni.conference_urls),
        additional_instruction=(
            "Prefer official conference membership pages (SECsports.com or theACC.com) or the university athletics site. "
            "Reliable updated directories (e.g., NCAA or Wikipedia membership tables) are acceptable if clearly current. "
            "The membership should reflect 2026 alignment."
        ),
        extra_prerequisites=[conf_ref_node, name_node]
    )

    # ------------------ Football stadium capacity (critical) ---------------- #
    stadium_node = evaluator.add_parallel(
        id=f"u{idx}_football_stadium",
        desc="University's main football stadium meets capacity requirement",
        parent=u_node,
        critical=True
    )

    stadium_name_ok = bool(uni.stadium_name and uni.stadium_name.strip())
    stadium_name_node = evaluator.add_custom_node(
        result=stadium_name_ok,
        id=f"u{idx}_stadium_name",
        desc="Official name of the university's primary football stadium is provided",
        parent=stadium_node,
        critical=True
    )

    stadium_ref_present = len(_norm_urls(uni.stadium_urls)) > 0
    stadium_ref_node = evaluator.add_custom_node(
        result=stadium_ref_present,
        id=f"u{idx}_stadium_reference",
        desc="URL reference confirming stadium capacity",
        parent=stadium_node,
        critical=True
    )

    stadium_cap_leaf = evaluator.add_leaf(
        id=f"u{idx}_stadium_capacity",
        desc="Stadium seating capacity exceeding 80,000 is provided and verified from official or reliable source",
        parent=stadium_node,
        critical=True
    )
    stadium_claim = (
        f"The primary football stadium for {uni.official_name}, named {uni.stadium_name}, has a seating capacity exceeding 80,000."
    )
    await evaluator.verify(
        claim=stadium_claim,
        node=stadium_cap_leaf,
        sources=_norm_urls(uni.stadium_urls),
        additional_instruction=(
            "Use an official athletics site, the stadium's official site, or a reliable directory (e.g., updated Wikipedia). "
            "Look for 'capacity' or 'seating capacity'. Expansion/renovation notes are acceptable if they establish > 80,000. "
            "Do not rely on single-game attendance records; verify the stated seating capacity."
        ),
        extra_prerequisites=[stadium_name_node, stadium_ref_node, name_node]
    )

    # ---------------- Basketball arena capacity (critical) ------------------ #
    arena_node = evaluator.add_parallel(
        id=f"u{idx}_basketball_arena",
        desc="University's main basketball arena meets capacity requirement",
        parent=u_node,
        critical=True
    )

    arena_name_ok = bool(uni.arena_name and uni.arena_name.strip())
    arena_name_node = evaluator.add_custom_node(
        result=arena_name_ok,
        id=f"u{idx}_arena_name",
        desc="Official name of the university's primary basketball arena is provided",
        parent=arena_node,
        critical=True
    )

    arena_ref_present = len(_norm_urls(uni.arena_urls)) > 0
    arena_ref_node = evaluator.add_custom_node(
        result=arena_ref_present,
        id=f"u{idx}_arena_reference",
        desc="URL reference confirming arena capacity",
        parent=arena_node,
        critical=True
    )

    arena_cap_leaf = evaluator.add_leaf(
        id=f"u{idx}_arena_capacity",
        desc="Arena seating capacity exceeding 20,000 is provided and verified from official or reliable source",
        parent=arena_node,
        critical=True
    )
    arena_claim = (
        f"The primary men's basketball arena for {uni.official_name}, named {uni.arena_name}, has a seating capacity exceeding 20,000."
    )
    await evaluator.verify(
        claim=arena_claim,
        node=arena_cap_leaf,
        sources=_norm_urls(uni.arena_urls),
        additional_instruction=(
            "Use an official athletics site, the arena's official site, or a reliable directory (e.g., updated Wikipedia). "
            "Look for 'capacity' or 'seating capacity'. Verify that the stated capacity strictly exceeds 20,000."
        ),
        extra_prerequisites=[arena_name_node, arena_ref_node, name_node]
    )

    # --------------- Graduate education programs (critical) ---------------- #
    grad_node = evaluator.add_parallel(
        id=f"u{idx}_graduate_education_programs",
        desc="University offers graduate education programs meeting SACSCOC requirements",
        parent=u_node,
        critical=True
    )

    grad_ref_present = len(_norm_urls(uni.grad_program_urls)) > 0
    grad_ref_node = evaluator.add_custom_node(
        result=grad_ref_present,
        id=f"u{idx}_graduate_programs_reference",
        desc="URL reference confirming graduate education program offerings and requirements",
        parent=grad_node,
        critical=True
    )

    # Verify that the named program is indeed a master's program in education or closely related field
    program_example_leaf = evaluator.add_leaf(
        id=f"u{idx}_program_example",
        desc="At least one specific example of a master's degree program in education or related field is provided",
        parent=grad_node,
        critical=True
    )
    program_example_claim = (
        f"{uni.official_name} offers a master's degree program in education or a closely related field named '{uni.program_name}'."
    )
    await evaluator.verify(
        claim=program_example_claim,
        node=program_example_leaf,
        sources=_norm_urls(uni.grad_program_urls),
        additional_instruction=(
            "Accept program names such as M.Ed., M.A.T., M.S. in Education, Curriculum & Instruction, Educational Leadership, "
            "Special Education, Higher Education, etc. The page should clearly indicate a master's-level education-related program."
        ),
        extra_prerequisites=[grad_ref_node, name_node]
    )

    # Verify that the program requires >= 30 semester credit hours
    credit_leaf = evaluator.add_leaf(
        id=f"u{idx}_credit_hour_requirement",
        desc="The provided graduate education program requires at least 30 semester credit hours, with confirmation",
        parent=grad_node,
        critical=True
    )
    credit_claim = (
        f"The program '{uni.program_name}' requires at least 30 semester credit hours (i.e., 30 or more)."
    )
    await evaluator.verify(
        claim=credit_claim,
        node=credit_leaf,
        sources=_norm_urls(uni.grad_program_urls),
        additional_instruction=(
            "Look for 'credit hours', 'semester hours', or similar language indicating the requirement is 30 or more credits. "
            "Phrases like 'minimum of 30 credit hours', '30+ credit hours', or explicit numbers (e.g., 30, 33, 36) qualify."
        ),
        extra_prerequisites=[program_example_leaf, grad_ref_node]
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'Southeastern R1 SEC/ACC universities with large venues and grad ed programs' task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel aggregation for independent universities
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

    # NOTE: We intentionally set root (created by initialize) as non-critical to allow partial credit across universities.
    # The provided JSON marks root as critical, but the framework requires critical parents to have all-critical children.
    # Allowing non-critical root ensures robust partial scoring when fewer than four universities meet all criteria.

    # 1) Extract structured university data (up to 4)
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Normalize to exactly 4 items (pad with empty)
    universities: List[UniversityItem] = list(extracted.universities or [])
    while len(universities) < 4:
        universities.append(UniversityItem())

    # 2) Build verification subtrees for each university
    for i in range(4):
        await verify_university(evaluator, root, universities[i], i + 1)

    # 3) Return summary
    return evaluator.get_summary()