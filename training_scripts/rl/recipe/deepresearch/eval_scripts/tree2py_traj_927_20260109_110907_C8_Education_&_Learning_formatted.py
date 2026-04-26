import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "nwccu4_masters_ed"
TASK_DESCRIPTION = (
    "Identify 4 public universities located in states accredited by the Northwest Commission on Colleges and Universities "
    "(NWCCU) that offer online master's degree programs in education. Each identified university must have both NWCCU "
    "regional accreditation for the institution and CAEP (Council for the Accreditation of Educator Preparation) "
    "accreditation for their education programs. For each of the 4 universities, provide: (1) The university name, "
    "(2) Confirmation that the university has NWCCU regional accreditation, (3) Confirmation that the university is "
    "located in one of the NWCCU states (Alaska, Idaho, Montana, Nevada, Oregon, Utah, or Washington), (4) Confirmation "
    "that the university's education programs have CAEP accreditation, and (5) A reference URL to the online master's "
    "education program page. Each university should offer at least one online master's degree program in education (such "
    "as M.Ed., M.A.T., or Ed.M. in areas like Educational Leadership, Curriculum and Instruction, Special Education, or "
    "similar fields)."
)

NWCCU_STATES = ["Alaska", "Idaho", "Montana", "Nevada", "Oregon", "Utah", "Washington"]

# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class UniversityItem(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    institution_url: Optional[str] = None  # e.g., main homepage or accreditation page, if provided
    program_url: Optional[str] = None      # online master's education program page URL
    nwccu_urls: List[str] = Field(default_factory=list)  # URLs explicitly related to NWCCU accreditation/membership
    caep_urls: List[str] = Field(default_factory=list)   # URLs explicitly related to CAEP accreditation for education
    additional_urls: List[str] = Field(default_factory=list)  # Any other URLs associated with the university


class UniversityListExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_universities() -> str:
    return (
        "Extract all universities mentioned in the answer that relate to the requested task. For each university, return a JSON object "
        "with the following fields:\n"
        "1) name: The university name as written in the answer.\n"
        "2) state: The U.S. state where the university is located, if explicitly mentioned.\n"
        "3) institution_url: A URL for the university’s main homepage, accreditation page, or an 'about' page if such a URL is present in the answer.\n"
        "4) program_url: A URL to an online master's education program page (e.g., M.Ed., MAT, EdM), if provided.\n"
        "5) nwccu_urls: An array of any URLs in the answer that explicitly relate to NWCCU institutional accreditation or NWCCU membership.\n"
        "6) caep_urls: An array of any URLs in the answer that explicitly relate to CAEP accreditation of the university’s education programs.\n"
        "7) additional_urls: Any other URLs mentioned that are relevant to verifying public status, location, or program details.\n\n"
        "Rules:\n"
        "- Only include URLs that are explicitly present in the answer (plain, markdown, or embedded). Do not invent any URLs.\n"
        "- If a field is not present in the answer, set it to null (for strings) or an empty list (for arrays).\n"
        "- Return all universities the answer mentions; the evaluator will select the first four."
    )


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return " ".join(name.strip().lower().split())


def is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    return (u.startswith("http://") or u.startswith("https://")) and len(u) > 10


def dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not is_valid_url(u):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth"][n - 1] if 1 <= n <= 4 else f"#{n}"


# -----------------------------------------------------------------------------
# Verification functions
# -----------------------------------------------------------------------------
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int,
) -> None:
    """
    Build verification subtree for one university (index 1..4).
    """
    ord_label = ordinal(idx)
    uni_node = evaluator.add_parallel(
        id=f"University_{idx}",
        desc=f"{ord_label} university identification and verification",
        parent=parent_node,
        critical=False,
    )

    # Critical existence checks
    name_ok = bool(uni.name and uni.name.strip())
    evaluator.add_custom_node(
        result=name_ok,
        id=f"U{idx}_Name_Provided",
        desc=f"Verify the {ord_label.lower()} university name is provided",
        parent=uni_node,
        critical=True,
    )

    ref_ok = is_valid_url(uni.program_url)
    evaluator.add_custom_node(
        result=ref_ok,
        id=f"U{idx}_Reference_URL",
        desc=f"Verify a reference URL to the online master's education program page is provided for the {ord_label.lower()} university",
        parent=uni_node,
        critical=True,
    )

    # Assemble sources
    all_urls: List[str] = []
    if uni.program_url:
        all_urls.append(uni.program_url)
    if uni.institution_url:
        all_urls.append(uni.institution_url)
    all_urls.extend(uni.nwccu_urls or [])
    all_urls.extend(uni.caep_urls or [])
    all_urls.extend(uni.additional_urls or [])
    all_urls = dedup_urls(all_urls)

    # U*_Public_Institution (critical)
    public_node = evaluator.add_leaf(
        id=f"U{idx}_Public_Institution",
        desc=f"Verify the {ord_label.lower()} university is a public institution",
        parent=uni_node,
        critical=True,
    )
    if not all_urls:
        public_node.score = 0.0
        public_node.status = "failed"
    else:
        claim_public = f"{uni.name or 'The university'} is a public university."
        await evaluator.verify(
            claim=claim_public,
            node=public_node,
            sources=all_urls,
            additional_instruction=(
                "Support should explicitly indicate the institution is public (e.g., 'public university', 'public research university', 'state university', "
                "or part of a public state system). If sources are ambiguous or imply private status, treat as not supported."
            ),
        )

    # U*_NWCCU_Accreditation (critical)
    nwccu_node = evaluator.add_leaf(
        id=f"U{idx}_NWCCU_Accreditation",
        desc=f"Verify the {ord_label.lower()} university has NWCCU regional (institutional) accreditation",
        parent=uni_node,
        critical=True,
    )
    nwccu_sources = dedup_urls((uni.nwccu_urls or []) + (all_urls or []))
    if not nwccu_sources:
        nwccu_node.score = 0.0
        nwccu_node.status = "failed"
    else:
        claim_nwccu = (
            f"{uni.name or 'The university'} is accredited by the Northwest Commission on Colleges and Universities (NWCCU)."
        )
        await evaluator.verify(
            claim=claim_nwccu,
            node=nwccu_node,
            sources=nwccu_sources,
            additional_instruction=(
                "Accept explicit statements on NWCCU's official directory/member list or the university's accreditation page that names NWCCU. "
                "Wording variations like 'Northwest Commission on Colleges and Universities' or 'NWCCU' are equivalent."
            ),
        )

    # U*_NWCCU_State (critical)
    state_node = evaluator.add_leaf(
        id=f"U{idx}_NWCCU_State",
        desc=f"Verify the {ord_label.lower()} university is located in one of the NWCCU states (Alaska, Idaho, Montana, Nevada, Oregon, Utah, Washington)",
        parent=uni_node,
        critical=True,
    )
    state_sources = all_urls
    if not state_sources:
        state_node.score = 0.0
        state_node.status = "failed"
    else:
        if uni.state and isinstance(uni.state, str) and uni.state.strip():
            claim_state = (
                f"{uni.name or 'The university'} is located in {uni.state.strip()}, which is one of the NWCCU states "
                f"({', '.join(NWCCU_STATES)})."
            )
        else:
            claim_state = (
                f"{uni.name or 'The university'} is located in one of the NWCCU states "
                f"({', '.join(NWCCU_STATES)})."
            )
        await evaluator.verify(
            claim=claim_state,
            node=state_node,
            sources=state_sources,
            additional_instruction=(
                "Confirm the campus location is in one of the NWCCU states listed. Accept official university pages or authoritative profiles "
                "that explicitly show the city/state. For multi-campus systems, it's sufficient if the listed campus is in an NWCCU state."
            ),
        )

    # U*_CAEP_Accreditation (critical)
    caep_node = evaluator.add_leaf(
        id=f"U{idx}_CAEP_Accreditation",
        desc=f"Verify the {ord_label.lower()} university's education programs have CAEP accreditation",
        parent=uni_node,
        critical=True,
    )
    caep_sources = dedup_urls((uni.caep_urls or []) + (all_urls or []))
    if not caep_sources:
        caep_node.score = 0.0
        caep_node.status = "failed"
    else:
        claim_caep = (
            f"{uni.name or 'The university'}'s education programs are accredited by CAEP (Council for the Accreditation of Educator Preparation)."
        )
        await evaluator.verify(
            claim=claim_caep,
            node=caep_node,
            sources=caep_sources,
            additional_instruction=(
                "Accept explicit proof on CAEP's directory or university/college of education pages stating CAEP accreditation. "
                "CAEP logos plus textual confirmation on an official page are acceptable."
            ),
        )

    # U*_Online_Masters_Ed_Program (critical)
    program_node = evaluator.add_leaf(
        id=f"U{idx}_Online_Masters_Ed_Program",
        desc=f"Verify the {ord_label.lower()} university offers at least one online master's degree program in education",
        parent=uni_node,
        critical=True,
    )
    prog_claim = (
        f"This page is for an online master's degree program in education offered by {uni.name or 'the university'}."
    )
    # Always call verify to allow auto-skip based on critical preconditions
    await evaluator.verify(
        claim=prog_claim,
        node=program_node,
        sources=uni.program_url if is_valid_url(uni.program_url) else None,
        additional_instruction=(
            "Accept M.Ed., MAT, Ed.M., or similar master's degrees in education (e.g., Educational Leadership, Curriculum & Instruction, "
            "Special Education). The page must clearly indicate it is online or fully online."
        ),
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
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
    Entry point for evaluating answers for the NWCCU online master's in education task.
    """
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

    # Record target NWCCU states for context
    evaluator.add_custom_info(
        info={"nwccu_states": NWCCU_STATES},
        info_type="config",
        info_name="nwccu_state_list",
    )

    # Extract universities mentioned in the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversityListExtraction,
        extraction_name="universities_extraction",
    )

    # Select exactly 4 (pad if fewer)
    selected: List[UniversityItem] = list(extracted.universities[:4])
    while len(selected) < 4:
        selected.append(UniversityItem())

    # Global critical check: exactly 4 distinct university names (non-empty)
    names = [normalize_name(u.name) for u in selected if normalize_name(u.name)]
    distinct_ok = (len(names) == 4) and (len(set(names)) == 4)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="Global_Exactly_4_Distinct_Universities",
        desc="Verify the response provides exactly 4 universities and they are distinct (no duplicates) with no extra universities beyond the 4",
        parent=root,
        critical=True,
    )

    # Per-university verification (non-critical to root to allow partial credit)
    for i, uni in enumerate(selected, start=1):
        await verify_university(evaluator, root, uni, i)

    return evaluator.get_summary()