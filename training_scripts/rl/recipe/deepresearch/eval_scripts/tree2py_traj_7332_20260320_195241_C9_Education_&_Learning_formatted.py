import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_universities_tx_mi_hlc_abet_div1_grad_40k"
TASK_DESCRIPTION = """
Identify four public universities in the United States—two located in Texas and two located in Michigan—that meet all of the following criteria:

1. Each university must be institutionally accredited by the Higher Learning Commission (HLC).

2. Each university must have a total student enrollment (undergraduate and graduate combined) of at least 40,000 students as of Fall 2024 or Fall 2025.

3. Each university must offer at least one undergraduate engineering program that is accredited by the Accreditation Board for Engineering and Technology (ABET).

4. Each university must participate in NCAA Division I intercollegiate athletics.

5. Each university must offer both master's degree programs and doctoral degree programs.

For each of the four universities, provide the following information:
- The official name of the institution
- The city where the main campus is located
- The specific total enrollment number (for Fall 2024 or Fall 2025)
- A direct URL to the university's HLC accreditation status or institutional accreditation page
- The name of at least one ABET-accredited engineering program offered by the university
- A URL linking to ABET verification for that program or the university's ABET accreditation information page
- The NCAA Division I athletic conference the university belongs to
- A URL to the university's graduate programs page or graduate school website
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    official_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # e.g., "Texas", "TX", "Michigan", "MI"
    public_or_private: Optional[str] = None  # e.g., "public", "private"
    total_enrollment: Optional[str] = None  # keep as string (e.g., "51,123", "about 50,000")
    enrollment_term: Optional[str] = None  # e.g., "Fall 2024", "Fall 2025"
    hlc_url: Optional[str] = None
    abet_program_name: Optional[str] = None
    abet_url: Optional[str] = None
    athletic_conference: Optional[str] = None
    athletics_url: Optional[str] = None  # university athletics site or conference membership page
    graduate_url: Optional[str] = None  # graduate programs page or graduate school
    supporting_urls: List[str] = Field(default_factory=list)  # any additional URLs cited in the answer


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract all universities mentioned in the answer that are candidates for the task (focus on schools in Texas or Michigan).
    For each university, extract the following fields EXACTLY as they appear in the answer. Do not invent data.

    For each university, return an object with:
    - official_name: The official or formal institution name as written in the answer (string or null)
    - city: The main campus city (string or null)
    - state: The U.S. state for the main campus location (e.g., "Texas", "TX", "Michigan", "MI") (string or null)
    - public_or_private: Whether the institution is public or private according to the answer (string or null). Use lowercase "public" or "private" if clearly stated; otherwise return null.
    - total_enrollment: The specific total enrollment number cited for Fall 2024 or Fall 2025 (as a raw string, keep punctuation, e.g., "51,123") (string or null)
    - enrollment_term: The term associated with that figure (e.g., "Fall 2024" or "Fall 2025") (string or null)
    - hlc_url: A direct URL to the institution’s HLC accreditation status page or institutional accreditation page (URL or null)
    - abet_program_name: The name of at least one undergraduate engineering program claimed to be ABET-accredited (string or null)
    - abet_url: A URL that verifies ABET accreditation for that program or the institution’s ABET accreditation information page (URL or null)
    - athletic_conference: The NCAA Division I conference named in the answer (string or null)
    - athletics_url: A URL to the university’s athletics site or a conference membership page that supports NCAA Division I participation (URL or null)
    - graduate_url: A URL to the university’s graduate programs page or graduate school website (URL or null)
    - supporting_urls: An array of any additional URLs in the answer that substantiate claims such as public status, location, enrollment, NCAA membership, or conference membership (array, can be empty)

    Return a JSON object with a top-level key "universities" that is an array of these per-university objects.
    If the answer lists more than four universities, include all that are in Texas or Michigan so we can select from them later.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    t = s.strip().lower()
    if t in {"tx", "texas"}:
        return "TX"
    if t in {"mi", "michigan"}:
        return "MI"
    return None


def _pick_by_state(items: List[UniversityItem], target: str, k: int = 2) -> List[UniversityItem]:
    norm = [u for u in items if _normalize_state(u.state) == target]
    return norm[:k]


def _gather_sources(u: UniversityItem) -> List[str]:
    raw = []
    for x in [u.hlc_url, u.abet_url, u.graduate_url, u.athletics_url]:
        if x and isinstance(x, str) and x.strip():
            raw.append(x.strip())
    if u.supporting_urls:
        for x in u.supporting_urls:
            if x and isinstance(x, str) and x.strip():
                raw.append(x.strip())
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for s in raw:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _parse_enrollment_value(enrollment: Optional[str]) -> Optional[int]:
    """
    Try to parse a reasonable integer enrollment value from a raw string.
    Strategy:
      - Extract all integers; filter out obvious years (2024, 2025).
      - Prefer numbers with at least 5 digits; otherwise take the largest number.
    """
    if not enrollment or not isinstance(enrollment, str):
        return None
    nums = re.findall(r"\d{2,}", enrollment.replace(",", ""))
    if not nums:
        return None
    candidates = []
    for n in nums:
        try:
            iv = int(n)
            if iv in (2024, 2025):
                continue
            candidates.append(iv)
        except Exception:
            continue
    if not candidates:
        return None
    # Prefer 5+ digits if present, else max
    long_nums = [x for x in candidates if x >= 10000]
    return max(long_nums) if long_nums else max(candidates)


def _nonempty(s: Optional[str]) -> bool:
    return bool(s and isinstance(s, str) and s.strip())


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def _verify_university(
    evaluator: Evaluator,
    parent_node,
    item: UniversityItem,
    prefix: str,           # e.g., "tx1", "tx2", "mi1", "mi2"
    agg_id: str,           # e.g., "texas_university_1"
    agg_desc: str,         # e.g., "First Texas public university meeting all criteria"
    expected_state_abbrev: str,  # "TX" or "MI"
    expected_state_name: str,    # "Texas" or "Michigan"
) -> None:
    """
    Build and verify all checks for a single university according to the rubric.
    Aggregator nodes are kept non-critical to satisfy the framework's constraint,
    while leaves that embody actual requirements are marked critical as in the rubric.
    """
    sources_all = _gather_sources(item)
    inst_name = item.official_name or "the institution"

    # Per-university aggregator (non-critical to allow mixed child criticalities)
    uni_node = evaluator.add_parallel(
        id=agg_id,
        desc=agg_desc,
        parent=parent_node,
        critical=False
    )

    # 1) Public status (critical)
    pub_node = evaluator.add_leaf(
        id=f"{prefix}_public_status",
        desc="The institution is a public university",
        parent=uni_node,
        critical=True
    )
    claim_public = f"{inst_name} is a public university."
    await evaluator.verify(
        claim=claim_public,
        node=pub_node,
        sources=sources_all,
        additional_instruction="Accept phrases like 'public research university' or 'public institution' as public."
    )

    # 2) State location (critical)
    state_node = evaluator.add_leaf(
        id=f"{prefix}_state_location",
        desc=f"The university is located in {expected_state_name}",
        parent=uni_node,
        critical=True
    )
    claim_state = f"The main campus of {inst_name} is located in {expected_state_name}."
    await evaluator.verify(
        claim=claim_state,
        node=state_node,
        sources=sources_all,
        additional_instruction=f"Verify that the primary campus location is within the state of {expected_state_name}. "
                               f"Allow common descriptions from official or authoritative sources."
    )

    # 3) City provided (critical, existence)
    evaluator.add_custom_node(
        result=_nonempty(item.city),
        id=f"{prefix}_city",
        desc="The city where the main campus is located is provided",
        parent=uni_node,
        critical=True
    )

    # 4) HLC accreditation (parent sequential, children: URL provided + supported-by-URL)
    hlc_parent = evaluator.add_sequential(
        id=f"{prefix}_hlc_accreditation",
        desc="The university is institutionally accredited by the Higher Learning Commission",
        parent=uni_node,
        critical=False
    )
    # 4.1 URL provided (critical)
    hlc_url_exists = evaluator.add_custom_node(
        result=_nonempty(item.hlc_url),
        id=f"{prefix}_hlc_url",
        desc="A direct URL to the university's HLC accreditation status or institutional accreditation page is provided",
        parent=hlc_parent,
        critical=True
    )
    # 4.2 Accreditation supported (critical)
    hlc_supported = evaluator.add_leaf(
        id=f"{prefix}_hlc_supported",
        desc="HLC institutional accreditation is supported by the cited HLC/institutional accreditation page",
        parent=hlc_parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"{inst_name} is institutionally accredited by the Higher Learning Commission (HLC).",
        node=hlc_supported,
        sources=item.hlc_url if _nonempty(item.hlc_url) else sources_all,
        additional_instruction="Confirm the institution-level accreditation (not just program-level). "
                               "HLC site or an official institutional accreditation page that names HLC should suffice. "
                               "Accept 'HLC' as abbreviation for the Higher Learning Commission."
    )

    # 5) Enrollment (parent sequential, children: figure provided + threshold >= 40,000)
    enroll_parent = evaluator.add_sequential(
        id=f"{prefix}_enrollment",
        desc="Total enrollment meets the minimum threshold",
        parent=uni_node,
        critical=False
    )
    # 5.1 figure provided (critical, existence)
    evaluator.add_custom_node(
        result=_nonempty(item.total_enrollment) and _nonempty(item.enrollment_term),
        id=f"{prefix}_enrollment_figure",
        desc="The specific total enrollment number for Fall 2024 or Fall 2025 is provided",
        parent=enroll_parent,
        critical=True
    )
    # 5.2 threshold >= 40,000 (critical, simple logic)
    parsed_val = _parse_enrollment_value(item.total_enrollment)
    evaluator.add_custom_node(
        result=(parsed_val is not None and parsed_val >= 40000),
        id=f"{prefix}_enrollment_threshold",
        desc="The enrollment is at least 40,000 students",
        parent=enroll_parent,
        critical=True
    )

    # 6) ABET-accredited program (parent sequential, children: program name provided + ABET URL provided + supported)
    abet_parent = evaluator.add_sequential(
        id=f"{prefix}_abet_program",
        desc="The university offers at least one ABET-accredited undergraduate engineering program",
        parent=uni_node,
        critical=False
    )
    evaluator.add_custom_node(
        result=_nonempty(item.abet_program_name),
        id=f"{prefix}_program_name",
        desc="The name of at least one ABET-accredited engineering program is provided",
        parent=abet_parent,
        critical=True
    )
    evaluator.add_custom_node(
        result=_nonempty(item.abet_url),
        id=f"{prefix}_abet_url",
        desc="A URL linking to ABET verification or the university's ABET accreditation page is provided",
        parent=abet_parent,
        critical=True
    )
    abet_supported = evaluator.add_leaf(
        id=f"{prefix}_abet_supported",
        desc="ABET accreditation for the named program is supported by the cited page",
        parent=abet_parent,
        critical=True
    )
    program_for_claim = item.abet_program_name or "the named undergraduate engineering program"
    await evaluator.verify(
        claim=f"The undergraduate engineering program '{program_for_claim}' at {inst_name} is ABET-accredited.",
        node=abet_supported,
        sources=item.abet_url if _nonempty(item.abet_url) else sources_all,
        additional_instruction="Accept explicit ABET program listings from ABET's official directory or "
                               "the university's ABET accreditation page that clearly shows accreditation."
    )

    # 7) NCAA Division I participation (critical)
    ncaa_node = evaluator.add_leaf(
        id=f"{prefix}_ncaa_division_i",
        desc="The university participates in NCAA Division I intercollegiate athletics",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{inst_name} participates in NCAA Division I intercollegiate athletics.",
        node=ncaa_node,
        sources=item.athletics_url if _nonempty(item.athletics_url) else sources_all,
        additional_instruction="Look for 'NCAA Division I', 'NCAA DI', or clear conference membership that is Division I."
    )

    # 8) Conference provided (critical, existence as per rubric)
    evaluator.add_custom_node(
        result=_nonempty(item.athletic_conference),
        id=f"{prefix}_conference",
        desc="The NCAA Division I athletic conference the university belongs to is provided",
        parent=uni_node,
        critical=True
    )

    # 9) Master's programs offered (critical)
    masters_node = evaluator.add_leaf(
        id=f"{prefix}_masters_programs",
        desc="The university offers master's degree programs",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{inst_name} offers master's degree programs.",
        node=masters_node,
        sources=item.graduate_url if _nonempty(item.graduate_url) else sources_all,
        additional_instruction="Graduate/Graduate School pages commonly list master's offerings."
    )

    # 10) Doctoral programs offered (critical)
    doctoral_node = evaluator.add_leaf(
        id=f"{prefix}_doctoral_programs",
        desc="The university offers doctoral degree programs",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"{inst_name} offers doctoral (PhD or equivalent) degree programs.",
        node=doctoral_node,
        sources=item.graduate_url if _nonempty(item.graduate_url) else sources_all,
        additional_instruction="Graduate/Graduate School pages commonly list doctoral offerings."
    )

    # 11) Graduate URL provided (critical, existence)
    evaluator.add_custom_node(
        result=_nonempty(item.graduate_url),
        id=f"{prefix}_graduate_url",
        desc="A URL to the university's graduate programs page is provided",
        parent=uni_node,
        critical=True
    )

    # 12) Athletics URL provided (non-critical, existence)
    evaluator.add_custom_node(
        result=_nonempty(item.athletics_url),
        id=f"{prefix}_athletics_url",
        desc="A URL to the university's athletics page or conference page is provided",
        parent=uni_node,
        critical=False
    )

    # 13) Official name correctness (critical) - verify name with available authoritative sources
    name_node = evaluator.add_leaf(
        id=f"{prefix}_official_name",
        desc="The official name of the institution is correctly provided",
        parent=uni_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the institution is '{item.official_name}'." if _nonempty(item.official_name)
              else "The institution's official name is correctly stated in the provided answer.",
        node=name_node,
        sources=item.hlc_url if _nonempty(item.hlc_url) else sources_all,
        additional_instruction="Allow minor stylistic variants (e.g., with or without leading 'The') if clearly the same institution."
    )


async def _build_state_group(
    evaluator: Evaluator,
    parent_node,
    group_id: str,
    group_desc: str,
    items: List[UniversityItem],
    state_abbrev: str,
    state_name: str
) -> None:
    """
    Build a state group (Texas or Michigan) with two universities.
    Aggregator kept non-critical to satisfy framework constraints; leaves enforce criteria.
    """
    group_node = evaluator.add_parallel(
        id=group_id,
        desc=group_desc,
        parent=parent_node,
        critical=False
    )

    # Pick up to two items; pad with empty placeholders if needed
    picks = _pick_by_state(items, state_abbrev, k=2)
    while len(picks) < 2:
        picks.append(UniversityItem())

    # University #1
    await _verify_university(
        evaluator,
        group_node,
        picks[0],
        prefix=f"{'tx' if state_abbrev=='TX' else 'mi'}1",
        agg_id=f"{'texas' if state_abbrev=='TX' else 'michigan'}_university_1",
        agg_desc=f"First {state_name} public university meeting all criteria",
        expected_state_abbrev=state_abbrev,
        expected_state_name=state_name
    )
    # University #2
    await _verify_university(
        evaluator,
        group_node,
        picks[1],
        prefix=f"{'tx' if state_abbrev=='TX' else 'mi'}2",
        agg_id=f"{'texas' if state_abbrev=='TX' else 'michigan'}_university_2",
        agg_desc=f"Second {state_name} public university meeting all criteria",
        expected_state_abbrev=state_abbrev,
        expected_state_name=state_name
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
    Evaluate an answer for the task of identifying four public universities
    (two in Texas and two in Michigan) satisfying HLC accreditation, >=40k enrollment,
    at least one ABET-accredited undergraduate engineering program, NCAA Division I,
    and graduate (master's + doctoral) offerings.
    """

    # Initialize evaluator (root is non-critical by design in the framework)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # follow rubric's root sequential intent
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
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Add a top-level distribution node (kept non-critical to satisfy framework constraints)
    state_distribution = evaluator.add_parallel(
        id="state_distribution",
        desc="The four universities must include two from Texas and two from Michigan",
        parent=root,
        critical=False
    )

    # Build Texas group (two universities)
    await _build_state_group(
        evaluator,
        state_distribution,
        group_id="texas_universities",
        group_desc="Two universities must be located in the state of Texas",
        items=extracted.universities,
        state_abbrev="TX",
        state_name="Texas"
    )

    # Build Michigan group (two universities)
    await _build_state_group(
        evaluator,
        state_distribution,
        group_id="michigan_universities",
        group_desc="Two universities must be located in the state of Michigan",
        items=extracted.universities,
        state_abbrev="MI",
        state_name="Michigan"
    )

    # Optionally record counts to help debugging
    tx_count = len([u for u in extracted.universities if _normalize_state(u.state) == "TX"])
    mi_count = len([u for u in extracted.universities if _normalize_state(u.state) == "MI"])
    evaluator.add_custom_info(
        {
            "total_universities_extracted": len(extracted.universities),
            "texas_candidates_found": tx_count,
            "michigan_candidates_found": mi_count
        },
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )

    # Return structured evaluation result
    return evaluator.get_summary()