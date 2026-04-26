import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "public_univ_coe_caep_regional_top15_states"
TASK_DESCRIPTION = """
I am researching public universities with strong Colleges of Education for a comparative study on graduate programs in educational leadership. I need to identify exactly 3 public universities that meet all of the following criteria:

1. Each university must be a public (state-funded) institution
2. Each university must be located in a different state, and each of those states must be ranked in the top 15 for PreK-12 education according to US News & World Report's 2025 state education rankings
3. Each university must have an established College of Education or School of Education
4. Each university's educator preparation programs must be accredited by CAEP (Council for the Accreditation of Educator Preparation)
5. Each university must be regionally accredited by a CHEA-recognized regional accrediting organization
6. Each university's College of Education must offer a doctoral-level graduate program (EdD or PhD) in Educational Leadership, Educational Administration, or an equivalent field
7. The minimum GPA requirement for admission to the doctoral program must be 3.0 or lower (on a 4.0 scale)

For each of the 3 universities you identify, please provide the following information:
- University name and state location
- Confirmation that it is a public institution
- The state's US News 2025 PreK-12 education ranking
- The name of the regional accrediting body and confirmation of its CHEA recognition
- CAEP accreditation status with a reference URL
- The specific name and degree type (EdD or PhD) of the Educational Leadership graduate program
- The minimum GPA requirement for admission to the program
- In-state and out-of-state tuition rates for the 2024-2025 or 2025-2026 academic year
- US News Best Graduate Schools ranking for the College of Education (if the institution is ranked)
- Reference URLs for all information provided
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityBasic(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    is_public: Optional[str] = None  # e.g., "public", "public (state-funded)"
    coe_name: Optional[str] = None   # College/School of Education name (if provided)
    basic_urls: List[str] = Field(default_factory=list)  # URLs that substantiate identity/public status/COE existence


class AccreditationInfo(BaseModel):
    regional_accreditor: Optional[str] = None  # e.g., HLC, SACSCOC, MSCHE, NECHE, NWCCU, WSCUC
    regional_url: Optional[str] = None         # URL confirming regional accreditation
    chea_recognition_url: Optional[str] = None # URL showing accreditor is CHEA-recognized
    caep_status: Optional[str] = None          # e.g., "CAEP accredited" (do not invent)
    caep_url: Optional[str] = None             # URL confirming CAEP accreditation for educator prep


class ProgramInfo(BaseModel):
    program_name: Optional[str] = None         # e.g., "EdD in Educational Leadership"
    degree_type: Optional[str] = None          # "EdD" or "PhD" (or spelled-out equivalents)
    program_url: Optional[str] = None
    min_gpa: Optional[str] = None              # text as stated, e.g., "3.0", "2.75", "3.0 or higher"
    admissions_url: Optional[str] = None       # URL for admissions requirements (GPA, etc.)


class TuitionInfo(BaseModel):
    in_state: Optional[str] = None             # textual amount (can be per credit, per year; extract literally)
    out_of_state: Optional[str] = None
    tuition_url: Optional[str] = None
    academic_year: Optional[str] = None        # e.g., "2024-2025" or "2025-2026"


class RankingInfo(BaseModel):
    state_pk12_rank_2025: Optional[str] = None # textual numeric rank for US News 2025 PreK-12 state ranking
    state_rank_url: Optional[str] = None       # URL to US News (or authoritative source) for that ranking
    coe_usnews_rank: Optional[str] = None      # optional: US News Grad Education ranking
    coe_usnews_rank_url: Optional[str] = None  # URL for that college ranking, if provided


class UniversityItem(BaseModel):
    basic: Optional[UniversityBasic] = None
    accreditation: Optional[AccreditationInfo] = None
    program: Optional[ProgramInfo] = None
    tuition: Optional[TuitionInfo] = None
    ranking: Optional[RankingInfo] = None


class UniversityList(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract all candidate public universities (and their education-college information) mentioned in the answer.
    For each university mentioned (in the order they appear), extract the following as a JSON array under "universities".
    Do not invent information; copy text exactly from the answer. Include only URLs explicitly present in the answer.

    For each university, include these nested sections:

    basic:
      - name: University name (string)
      - state: State location (string, full state name or postal abbreviation as written)
      - is_public: Text confirming public status if present (e.g., "public", "public (state-funded)"); null if not present
      - coe_name: The name of the College or School of Education if stated; null if not present
      - basic_urls: List of URL(s) in the answer that substantiate institutional identity or public status; can include a main "About", facts page, or Wikipedia

    accreditation:
      - regional_accreditor: Name of the regional accrediting body as written (e.g., HLC, SACSCOC, MSCHE, NECHE, NWCCU, WSCUC), or the full name if stated; null if not present
      - regional_url: URL in the answer that confirms the university's regional accreditation; null if not present
      - chea_recognition_url: URL in the answer that shows the accreditor is recognized by CHEA (e.g., a page on chea.org listing the accreditor); null if not present
      - caep_status: Text in the answer indicating CAEP accreditation of educator preparation programs; null if not present
      - caep_url: URL in the answer that confirms CAEP accreditation (e.g., CAEP "Accredited Provider" page, university accreditation page referencing CAEP); null if not present

    program:
      - program_name: Specific program name for doctoral Educational Leadership/Administration (or equivalent); null if not present
      - degree_type: Degree type as written (e.g., "EdD", "PhD", or spelled out); null if not present
      - program_url: URL in the answer for the program page; null if not present
      - min_gpa: Minimum GPA requirement text (e.g., "3.0", "2.75", "3.0 or higher"); null if not present
      - admissions_url: URL in the answer for admissions/requirements (may be the program page itself); null if not present

    tuition:
      - in_state: In-state tuition amount text for 2024-2025 or 2025-2026 (copy text as-is; per-credit or annual acceptable); null if not present
      - out_of_state: Out-of-state tuition amount text; null if not present
      - tuition_url: URL in the answer for tuition or cost page; null if not present
      - academic_year: Academic year label in the answer (e.g., "2024-2025" or "2025-2026"); null if not present

    ranking:
      - state_pk12_rank_2025: The 2025 US News PreK-12 state ranking number for the university's state as written; null if not present
      - state_rank_url: URL in the answer supporting the 2025 US News PreK-12 state ranking; null if not present
      - coe_usnews_rank: If provided in the answer, the US News Best Graduate Schools ranking for the College of Education (copy text as-is); null if not present
      - coe_usnews_rank_url: URL for the College of Education ranking (if any); null if not present

    Notes:
    - Only include URLs that are explicitly present in the answer text (including markdown links). Do not infer any URLs.
    - Preserve the answer's wording for text values (e.g., "EdD", "Doctor of Education", "3.0 minimum GPA").
    - Return ALL universities the answer mentions in "universities". Do not filter. The evaluator will use the first three.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(*args: Optional[List[str]]) -> List[str]:
    out: List[str] = []
    for a in args:
        if not a:
            continue
        for u in a:
            if isinstance(u, str):
                u2 = u.strip()
                if u2 and u2 not in out:
                    out.append(u2)
    return out


def _safe_url_list(*urls: Optional[str]) -> List[str]:
    out: List[str] = []
    for u in urls:
        if isinstance(u, str):
            s = u.strip()
            if s and s not in out:
                out.append(s)
    return out


def _normalize_state_name(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _parse_rank_int(rank_text: Optional[str]) -> Optional[int]:
    if not rank_text:
        return None
    # Extract first integer occurrence
    num = ""
    for ch in rank_text:
        if ch.isdigit():
            num += ch
        elif num:
            break
    try:
        return int(num) if num else None
    except Exception:
        return None


def _nonempty(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""


# --------------------------------------------------------------------------- #
# Verification builders per university                                        #
# --------------------------------------------------------------------------- #
async def _verify_basic_identity(
    evaluator: Evaluator,
    parent,
    uni: UniversityItem,
    uni_idx: int
):
    b = uni.basic or UniversityBasic()
    r = uni.ranking or RankingInfo()

    node = evaluator.add_parallel(
        id=f"u{uni_idx}_basic_identity",
        desc="Basic institutional identification",
        parent=parent,
        critical=True
    )

    # U*_Basic_URL: presence
    evaluator.add_custom_node(
        result=bool(b.basic_urls and len(b.basic_urls) > 0),
        id=f"u{uni_idx}_basic_url",
        desc="Reference URL provided for basic institutional information",
        parent=node,
        critical=True
    )

    # U*_Name_State: presence
    evaluator.add_custom_node(
        result=_nonempty(b.name) and _nonempty(b.state),
        id=f"u{uni_idx}_name_state",
        desc="University name and state location are provided",
        parent=node,
        critical=True
    )

    # U*_Public_Status: verify public institution
    leaf_public = evaluator.add_leaf(
        id=f"u{uni_idx}_public_status",
        desc="Institution is confirmed as a public university",
        parent=node,
        critical=True
    )
    claim_public = f"{b.name or 'The university'} is a public (state-funded) university."
    await evaluator.verify(
        claim=claim_public,
        node=leaf_public,
        sources=_safe_list(b.basic_urls),
        additional_instruction="Check the referenced page(s) to confirm the institution is public/state-funded. Accept official university pages or credible sources that explicitly state 'public'."
    )

    # U*_COE_Exists: verify COE existence (and optionally its name)
    leaf_coe = evaluator.add_leaf(
        id=f"u{uni_idx}_coe_exists",
        desc="Institution has a College/School of Education",
        parent=node,
        critical=True
    )
    coe_part = f" named '{b.coe_name}'" if _nonempty(b.coe_name) else ""
    claim_coe = f"{b.name or 'The university'} has a College or School of Education{coe_part}."
    coe_sources = _safe_list(b.basic_urls)  # may include COE home page if the answer listed it here
    await evaluator.verify(
        claim=claim_coe,
        node=leaf_coe,
        sources=coe_sources,
        additional_instruction="Confirm the presence of a College/School of Education (or similarly named unit) on the referenced page(s). Minor naming variations are acceptable."
    )

    # U*_State_Ranking: verify in top 15 for 2025 US News PreK-12 ranking
    leaf_rank = evaluator.add_leaf(
        id=f"u{uni_idx}_state_ranking",
        desc="State is ranked in top 15 for PreK-12 education in US News 2025 rankings",
        parent=node,
        critical=True
    )
    state_name = b.state or "the state"
    rank_num = _parse_rank_int(r.state_pk12_rank_2025)
    if rank_num is not None:
        rank_phrase = f"with a rank of {rank_num}"
    else:
        rank_phrase = "with a top-15 placement"

    claim_rank = f"The state of {state_name} is in the top 15 for PreK-12 education in the US News & World Report 2025 state education rankings, {rank_phrase}."
    await evaluator.verify(
        claim=claim_rank,
        node=leaf_rank,
        sources=_safe_url_list(r.state_rank_url),
        additional_instruction="Verify specifically for 'US News & World Report 2025' PreK-12 state rankings. The page should indicate the state's 2025 PreK-12 rank. Minor wording differences are okay."
    )


async def _verify_regional_accreditation(
    evaluator: Evaluator,
    parent,
    uni: UniversityItem,
    uni_idx: int
):
    b = uni.basic or UniversityBasic()
    a = uni.accreditation or AccreditationInfo()

    node = evaluator.add_parallel(
        id=f"u{uni_idx}_regional_accreditation",
        desc="Regional accreditation verification",
        parent=parent,
        critical=True
    )

    # U*_Regional_URL: presence (gate others)
    evaluator.add_custom_node(
        result=_nonempty(a.regional_url),
        id=f"u{uni_idx}_regional_url",
        desc="Reference URL for regional accreditation status",
        parent=node,
        critical=True
    )

    # U*_Regional_Accreditor: verify accreditor name matches what's claimed
    leaf_reg_body = evaluator.add_leaf(
        id=f"u{uni_idx}_regional_accreditor",
        desc="Regional accrediting body is identified",
        parent=node,
        critical=True
    )
    claim_reg_body = f"{b.name or 'The university'} is regionally accredited by {a.regional_accreditor or 'the stated accreditor'}."
    await evaluator.verify(
        claim=claim_reg_body,
        node=leaf_reg_body,
        sources=_safe_url_list(a.regional_url),
        additional_instruction="Confirm that the referenced page names the same regional accrediting body as claimed (e.g., HLC, SACSCOC, MSCHE, NECHE, NWCCU, or WSCUC)."
    )

    # U*_CHEA_Recognition: verify accreditor is CHEA-recognized
    leaf_chea = evaluator.add_leaf(
        id=f"u{uni_idx}_chea_recognition",
        desc="Accreditor is CHEA-recognized",
        parent=node,
        critical=True
    )
    chea_sources = _safe_url_list(a.chea_recognition_url) or _safe_url_list(a.regional_url)
    claim_chea = f"The accrediting organization {a.regional_accreditor or 'the stated accreditor'} is recognized by CHEA (Council for Higher Education Accreditation)."
    await evaluator.verify(
        claim=claim_chea,
        node=leaf_chea,
        sources=chea_sources,
        additional_instruction="Prefer using chea.org pages that list the accreditor as CHEA-recognized. If the provided URL is not a CHEA page, accept only if it clearly and credibly states CHEA recognition."
    )


async def _verify_caep(
    evaluator: Evaluator,
    parent,
    uni: UniversityItem,
    uni_idx: int
):
    b = uni.basic or UniversityBasic()
    a = uni.accreditation or AccreditationInfo()

    node = evaluator.add_parallel(
        id=f"u{uni_idx}_caep_accreditation",
        desc="CAEP accreditation verification",
        parent=parent,
        critical=True
    )

    # U*_CAEP_URL: presence (gate)
    evaluator.add_custom_node(
        result=_nonempty(a.caep_url),
        id=f"u{uni_idx}_caep_url",
        desc="Reference URL for CAEP accreditation",
        parent=node,
        critical=True
    )

    # U*_CAEP_Status: verify CAEP accreditation
    leaf_caep = evaluator.add_leaf(
        id=f"u{uni_idx}_caep_status",
        desc="CAEP accreditation status is confirmed",
        parent=node,
        critical=True
    )
    claim_caep = f"The educator preparation programs at {b.name or 'the university'} are accredited by CAEP (Council for the Accreditation of Educator Preparation)."
    await evaluator.verify(
        claim=claim_caep,
        node=leaf_caep,
        sources=_safe_url_list(a.caep_url),
        additional_instruction="Verify that the page explicitly confirms CAEP accreditation for educator preparation programs. Accept reasonable naming variations of CAEP."
    )


async def _verify_program(
    evaluator: Evaluator,
    parent,
    uni: UniversityItem,
    uni_idx: int
):
    b = uni.basic or UniversityBasic()
    p = uni.program or ProgramInfo()

    node = evaluator.add_parallel(
        id=f"u{uni_idx}_graduate_program",
        desc="Graduate program in Educational Leadership",
        parent=parent,
        critical=True
    )

    # U*_Program_URL: presence (gate)
    evaluator.add_custom_node(
        result=_nonempty(p.program_url),
        id=f"u{uni_idx}_program_url",
        desc="Reference URL for graduate program information",
        parent=node,
        critical=True
    )

    # U*_Program_Exists
    leaf_exists = evaluator.add_leaf(
        id=f"u{uni_idx}_program_exists",
        desc="EdD or PhD program in Educational Leadership/Administration exists",
        parent=node,
        critical=True
    )
    exists_claim = (
        f"{b.name or 'The university'} offers a doctoral program (EdD or PhD) in Educational Leadership, "
        f"Educational Administration, or an equivalent field within its College/School of Education."
    )
    await evaluator.verify(
        claim=exists_claim,
        node=leaf_exists,
        sources=_safe_url_list(p.program_url),
        additional_instruction="Verify the page clearly indicates a doctoral-level (EdD/PhD) program in Educational Leadership, Educational Administration, or an equivalent (e.g., Leadership & Policy, K-12 Leadership). Accept reasonable naming variants."
    )

    # U*_Program_Name
    leaf_name = evaluator.add_leaf(
        id=f"u{uni_idx}_program_name",
        desc="Specific program name is provided",
        parent=node,
        critical=True
    )
    prog_name = p.program_name or "the program"
    claim_name = f"The specific doctoral program name is stated as '{prog_name}' or a very close equivalent."
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        sources=_safe_url_list(p.program_url),
        additional_instruction="Confirm the program name or a very close equivalent appears on the referenced page."
    )

    # U*_Degree_Type
    leaf_degree = evaluator.add_leaf(
        id=f"u{uni_idx}_degree_type",
        desc="Degree type (EdD or PhD) is specified",
        parent=node,
        critical=True
    )
    deg = (p.degree_type or "").strip()
    claim_degree = f"The doctoral program's degree type is '{deg}' (i.e., an EdD or a PhD, or equivalents spelled out)."
    await evaluator.verify(
        claim=claim_degree,
        node=leaf_degree,
        sources=_safe_url_list(p.program_url),
        additional_instruction="Confirm that the page specifies the degree type as EdD or PhD (spelled out acceptable)."
    )


async def _verify_admissions(
    evaluator: Evaluator,
    parent,
    uni: UniversityItem,
    uni_idx: int
):
    p = uni.program or ProgramInfo()

    node = evaluator.add_parallel(
        id=f"u{uni_idx}_admission_requirements",
        desc="Graduate program admission requirements",
        parent=parent,
        critical=True
    )

    # U*_Admission_URL: presence (gate)
    evaluator.add_custom_node(
        result=_nonempty(p.admissions_url) or _nonempty(p.program_url),
        id=f"u{uni_idx}_admission_url",
        desc="Reference URL for admission requirements",
        parent=node,
        critical=True
    )

    # U*_GPA_Requirement
    leaf_gpa = evaluator.add_leaf(
        id=f"u{uni_idx}_gpa_requirement",
        desc="Minimum GPA requirement is 3.0 or lower",
        parent=node,
        critical=True
    )
    gpa_text = (p.min_gpa or "").strip()
    claim_gpa = (
        f"The minimum GPA requirement for admission to the doctoral program is at most 3.0 on a 4.0 scale. "
        f"The page states the requirement as: '{gpa_text}'."
    )
    gpa_sources = _safe_url_list(p.admissions_url) or _safe_url_list(p.program_url)
    await evaluator.verify(
        claim=claim_gpa,
        node=leaf_gpa,
        sources=gpa_sources,
        additional_instruction="Confirm that the minimum GPA requirement is 3.0 or lower (e.g., 3.0, 2.75). Phrases like 'minimum 3.0' or '3.0 or higher' should be accepted as meeting the ≤3.0 threshold."
    )


async def _verify_tuition(
    evaluator: Evaluator,
    parent,
    uni: UniversityItem,
    uni_idx: int
):
    t = uni.tuition or TuitionInfo()

    node = evaluator.add_parallel(
        id=f"u{uni_idx}_tuition",
        desc="Tuition information for 2024-2025 or 2025-2026",
        parent=parent,
        critical=True
    )

    # U*_Tuition_URL: presence (gate)
    evaluator.add_custom_node(
        result=_nonempty(t.tuition_url),
        id=f"u{uni_idx}_tuition_url",
        desc="Reference URL for tuition information",
        parent=node,
        critical=True
    )

    # U*_Instate_Tuition
    leaf_instate = evaluator.add_leaf(
        id=f"u{uni_idx}_instate_tuition",
        desc="In-state tuition rate provided",
        parent=node,
        critical=True
    )
    claim_instate = (
        f"The in-state tuition rate for academic year {(t.academic_year or '2024-2025/2025-2026')} is stated (or clearly derivable) as '{(t.in_state or '').strip()}'."
    )
    await evaluator.verify(
        claim=claim_instate,
        node=leaf_instate,
        sources=_safe_url_list(t.tuition_url),
        additional_instruction="Verify a clearly labeled in-state tuition amount on the referenced tuition/cost page. Accept annual totals or per-credit rates if they pertain to the 2024-2025 or 2025-2026 academic year. Minor formatting differences are acceptable."
    )

    # U*_Outstate_Tuition
    leaf_outstate = evaluator.add_leaf(
        id=f"u{uni_idx}_outstate_tuition",
        desc="Out-of-state tuition rate provided",
        parent=node,
        critical=True
    )
    claim_outstate = (
        f"The out-of-state tuition rate for academic year {(t.academic_year or '2024-2025/2025-2026')} is stated (or clearly derivable) as '{(t.out_of_state or '').strip()}'."
    )
    await evaluator.verify(
        claim=claim_outstate,
        node=leaf_outstate,
        sources=_safe_url_list(t.tuition_url),
        additional_instruction="Verify a clearly labeled out-of-state tuition amount on the referenced tuition/cost page. Accept annual totals or per-credit rates if they pertain to the 2024-2025 or 2025-2026 academic year. Minor formatting differences are acceptable."
    )


async def _verify_optional_rankings(
    evaluator: Evaluator,
    parent,
    uni: UniversityItem,
    uni_idx: int
):
    r = uni.ranking or RankingInfo()

    # Only create the optional Rankings node if the answer provided a COE ranking or ranking URL
    if not (_nonempty(r.coe_usnews_rank) or _nonempty(r.coe_usnews_rank_url)):
        return

    node = evaluator.add_parallel(
        id=f"u{uni_idx}_rankings",
        desc="US News graduate education rankings (if applicable)",
        parent=parent,
        critical=False
    )

    if _nonempty(r.coe_usnews_rank):
        # U*_USNews_Ranking
        leaf_rank = evaluator.add_leaf(
            id=f"u{uni_idx}_usnews_ranking",
            desc="US News ranking provided if institution is ranked",
            parent=node,
            critical=False
        )
        claim_rank = f"The College/School of Education is ranked as '{r.coe_usnews_rank.strip()}' by US News Best Graduate Schools."
        await evaluator.verify(
            claim=claim_rank,
            node=leaf_rank,
            sources=_safe_url_list(r.coe_usnews_rank_url),
            additional_instruction="Verify that the referenced page (preferably US News) shows the same or equivalent ranking statement for the College/School of Education."
        )

    if _nonempty(r.coe_usnews_rank_url):
        # U*_Ranking_URL: presence confirmation as a leaf-custom
        evaluator.add_custom_node(
            result=True,
            id=f"u{uni_idx}_ranking_url",
            desc="Reference URL for ranking information",
            parent=node,
            critical=False
        )


async def _verify_university(
    evaluator: Evaluator,
    parent,
    uni: UniversityItem,
    uni_number: int
):
    uni_node = evaluator.add_parallel(
        id=f"university_{uni_number}",
        desc=f"{['First','Second','Third'][uni_number-1]} qualifying public university",
        parent=parent,
        critical=False
    )

    # Build the subgroups
    await _verify_basic_identity(evaluator, uni_node, uni, uni_number)
    await _verify_regional_accreditation(evaluator, uni_node, uni, uni_number)
    await _verify_caep(evaluator, uni_node, uni, uni_number)
    await _verify_program(evaluator, uni_node, uni, uni_number)
    await _verify_admissions(evaluator, uni_node, uni, uni_number)
    await _verify_tuition(evaluator, uni_node, uni, uni_number)
    await _verify_optional_rankings(evaluator, uni_node, uni, uni_number)


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
    Evaluate an answer for the '3 public universities with Education Colleges' task.
    """
    # Initialize evaluator (root is always non-critical in the framework to avoid child critical consistency issues)
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
    extracted: UniversityList = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversityList,
        extraction_name="universities_extraction"
    )

    # Select exactly the first 3 items mentioned; pad with empty if fewer
    items: List[UniversityItem] = list(extracted.universities[:3])
    while len(items) < 3:
        items.append(UniversityItem())

    # Build Task Completion node (make non-critical to allow partial credit per-university; add critical subchecks inside)
    task_node = evaluator.add_parallel(
        id="Task_Completion",
        desc="Identify 3 public universities with Colleges of Education that meet all specified criteria",
        parent=root,
        critical=False
    )

    # Verify each university block
    await _verify_university(evaluator, task_node, items[0], 1)
    await _verify_university(evaluator, task_node, items[1], 2)
    await _verify_university(evaluator, task_node, items[2], 3)

    # Geographic diversity: Three universities are in three different states
    states: List[str] = [
        _normalize_state_name((items[i].basic.state if items[i].basic else None)) for i in range(3)
    ]
    # Valid only if all three are non-empty and pairwise distinct
    geo_ok = all(s for s in states) and len(set(states)) == 3

    evaluator.add_custom_node(
        result=geo_ok,
        id="Geographic_Diversity",
        desc="Three universities are located in three different states",
        parent=task_node,
        critical=True
    )

    # Add a small custom info block (not used for scoring)
    evaluator.add_custom_info(
        info={
            "extracted_state_values": states,
            "note": "States are compared case-insensitively for uniqueness."
        },
        info_type="diagnostics",
        info_name="geodiversity_check"
    )

    # Return standardized summary
    return evaluator.get_summary()