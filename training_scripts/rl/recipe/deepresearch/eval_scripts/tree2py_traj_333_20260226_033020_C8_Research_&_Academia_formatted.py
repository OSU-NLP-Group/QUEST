import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "midwest_r1_csphd_humanities_2025"
TASK_DESCRIPTION = (
    "Identify four universities located in Ohio, Michigan, Indiana, or Illinois that received the R1 designation "
    "(Research 1: Very High Spending and Doctorate Production) in the 2025 Carnegie Classification and currently "
    "offer doctoral programs in Computer Science. For each of the four universities, provide: "
    "(1) confirmation of their 2025 Carnegie R1 status with a supporting reference URL, "
    "(2) the official name of their Computer Science doctoral program with a supporting reference URL, and "
    "(3) evidence that the university hosts an active humanities research center or institute with a supporting reference URL."
)
ALLOWED_STATES = ["Ohio", "Michigan", "Indiana", "Illinois"]


# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    university_name: Optional[str] = None
    state_or_location: Optional[str] = None
    r1_reference_urls: List[str] = Field(default_factory=list)
    cs_phd_program_name: Optional[str] = None
    cs_reference_urls: List[str] = Field(default_factory=list)
    humanities_center_name: Optional[str] = None
    humanities_reference_urls: List[str] = Field(default_factory=list)
    location_reference_urls: List[str] = Field(default_factory=list)


class UniversityExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four universities from the answer that the author claims satisfy all of the following:
    – Located in one of the following U.S. states: Ohio (OH), Michigan (MI), Indiana (IN), or Illinois (IL).
    – Received the R1 designation (Research 1: Very High Spending and Doctorate Production) in the 2025 Carnegie Classification.
    – Offer a doctoral program in Computer Science (or closely named equivalent, e.g., Computer Science and Engineering; Computer and Information Science).
    – Host an active humanities research center or institute.

    For each university, extract the following fields exactly as mentioned in the answer:
    1) university_name: The university's official name as written in the answer.
    2) state_or_location: The state or location for the university if explicitly provided (e.g., "Ohio", "Ann Arbor, MI"). If not stated, return null.
    3) r1_reference_urls: All URL(s) the answer cites to support the 2025 Carnegie R1 status. Do not invent any URLs.
    4) cs_phd_program_name: The official PhD program name in CS/CSE/etc. If mentioned, extract it verbatim; otherwise null.
    5) cs_reference_urls: All URL(s) the answer cites to support the CS doctoral program. Do not invent any URLs.
    6) humanities_center_name: The humanities research center/institute name if provided. If not provided, return null.
    7) humanities_reference_urls: All URL(s) the answer cites to support the humanities center/institute. Do not invent any URLs.
    8) location_reference_urls: Any URL(s) the answer explicitly cites to support the university’s location/state; if none are provided, return an empty list.

    Important rules:
    – Return at most four universities; if the answer lists more, only include the first four.
    – For URL fields, only include URLs explicitly present in the answer text (including markdown links). Do not infer or create URLs.
    – If a field is missing in the answer, set it to null (or an empty list for URL arrays).
    """


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
def _dedup_and_filter_urls(urls: List[str]) -> List[str]:
    if not urls:
        return []
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        # Basic sanity: must look like a URL
        if not (u.startswith("http://") or u.startswith("https://")):
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _union_sources(*url_lists: List[str]) -> List[str]:
    merged: List[str] = []
    for lst in url_lists:
        merged.extend(lst or [])
    return _dedup_and_filter_urls(merged)


def _nonempty_string(s: Optional[str]) -> bool:
    return bool(s and s.strip())


# --------------------------------------------------------------------------- #
# Verification for a single university                                        #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index: int,
) -> None:
    uid = f"university_{index+1}"
    uni_label = uni.university_name or f"University #{index+1}"

    # University-level node (non-critical; allows partial credit across universities)
    uni_node = evaluator.add_parallel(
        id=uid,
        desc=f"{['First','Second','Third','Fourth'][index]} identified university meets all specified criteria",
        parent=parent_node,
        critical=False
    )

    # 1) Geographic location check (Critical leaf)
    geo_leaf = evaluator.add_leaf(
        id=f"{uid}_geographic_location",
        desc="University is located in Ohio, Michigan, Indiana, or Illinois",
        parent=uni_node,
        critical=True
    )

    geo_claim = (
        f"The university '{uni_label}' is located in one of these states: Ohio, Michigan, Indiana, or Illinois."
        if not _nonempty_string(uni.state_or_location)
        else f"The university '{uni_label}' is located in {uni.state_or_location}, which is one of Ohio, Michigan, Indiana, or Illinois."
    )

    geo_sources = _union_sources(
        uni.location_reference_urls,
        uni.r1_reference_urls,
        uni.cs_reference_urls,
        uni.humanities_reference_urls
    )

    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=geo_sources,
        additional_instruction=(
            "Support the claim using any provided URLs. It is sufficient if any page clearly shows the university's city/state "
            "or an address within Ohio, Michigan, Indiana, or Illinois (e.g., 'Ann Arbor, MI'). Minor wording differences are acceptable."
        )
    )

    # 2) R1 Status in 2025 Carnegie Classification (Critical group)
    r1_node = evaluator.add_parallel(
        id=f"{uid}_r1_status",
        desc="University has 2025 Carnegie R1 designation with valid reference URL",
        parent=uni_node,
        critical=True
    )

    # 2.1 Existence of R1 Reference URL (Critical existence check)
    r1_url_exists = evaluator.add_custom_node(
        result=len(_dedup_and_filter_urls(uni.r1_reference_urls)) > 0,
        id=f"{uid}_r1_reference_url",
        desc="Valid reference URL provided for R1 status verification",
        parent=r1_node,
        critical=True
    )

    # 2.2 Evidence confirms R1 status in 2025 (Critical leaf)
    r1_verify_leaf = evaluator.add_leaf(
        id=f"{uid}_r1_verification",
        desc="Evidence confirms R1 status in 2025 Carnegie Classification",
        parent=r1_node,
        critical=True
    )

    r1_claim = (
        f"The provided webpage confirms that '{uni_label}' has the R1 designation in the 2025 Carnegie Classification "
        f"(Research 1: Very High Spending and Doctorate Production)."
    )

    await evaluator.verify(
        claim=r1_claim,
        node=r1_verify_leaf,
        sources=_dedup_and_filter_urls(uni.r1_reference_urls),
        additional_instruction=(
            "Accept if the page explicitly indicates the university is R1 under the 2025 Carnegie Classification. "
            "Synonyms or close variants for the R1 label (e.g., 'Very High Research Spending and Doctorate Production') are acceptable. "
            "University news pages that directly state the 2025 R1 status also count."
        )
    )

    # 3) Computer Science PhD Program (Critical group)
    cs_node = evaluator.add_parallel(
        id=f"{uid}_cs_phd_program",
        desc="University offers Computer Science doctoral program with valid reference URL",
        parent=uni_node,
        critical=True
    )

    # 3.1 Existence of CS PhD reference URL (Critical existence check)
    cs_url_exists = evaluator.add_custom_node(
        result=len(_dedup_and_filter_urls(uni.cs_reference_urls)) > 0,
        id=f"{uid}_cs_phd_reference_url",
        desc="Valid reference URL provided for PhD program verification",
        parent=cs_node,
        critical=True
    )

    # 3.2 Evidence confirms active CS/CSE PhD program (Critical leaf)
    cs_verify_leaf = evaluator.add_leaf(
        id=f"{uid}_cs_phd_verification",
        desc="Evidence confirms active CS/CSE PhD program",
        parent=cs_node,
        critical=True
    )

    cs_name = uni.cs_phd_program_name or "a doctoral program in Computer Science or a closely named equivalent"
    cs_claim = (
        f"The university '{uni_label}' offers {cs_name}. This is a doctoral (PhD) program in the Computer Science domain "
        f"(or an equivalent title such as Computer Science and Engineering, or Computer and Information Science)."
    )

    await evaluator.verify(
        claim=cs_claim,
        node=cs_verify_leaf,
        sources=_dedup_and_filter_urls(uni.cs_reference_urls),
        additional_instruction=(
            "Confirm that the referenced page indicates a PhD/Doctoral program in Computer Science (or very close equivalent, "
            "e.g., Computer Science and Engineering, Computer and Information Science). Minor naming differences are acceptable."
        )
    )

    # 4) Humanities Research Center/Institute (Critical group)
    hum_node = evaluator.add_parallel(
        id=f"{uid}_humanities_center",
        desc="University hosts active humanities research center or institute with valid reference URL",
        parent=uni_node,
        critical=True
    )

    # 4.1 Existence of humanities center reference URL (Critical existence check)
    hum_url_exists = evaluator.add_custom_node(
        result=len(_dedup_and_filter_urls(uni.humanities_reference_urls)) > 0,
        id=f"{uid}_humanities_reference_url",
        desc="Valid reference URL provided for humanities center verification",
        parent=hum_node,
        critical=True
    )

    # 4.2 Evidence confirms active humanities research center/institute (Critical leaf)
    hum_verify_leaf = evaluator.add_leaf(
        id=f"{uid}_humanities_verification",
        desc="Evidence confirms active humanities research center/institute",
        parent=hum_node,
        critical=True
    )

    center_name = uni.humanities_center_name or "a humanities research center or institute"
    hum_claim = (
        f"The university '{uni_label}' hosts an active humanities research center or institute, such as '{center_name}'. "
        f"Active means the center/institute appears currently functioning (e.g., has staff, pages, events, or news)."
    )

    await evaluator.verify(
        claim=hum_claim,
        node=hum_verify_leaf,
        sources=_dedup_and_filter_urls(uni.humanities_reference_urls),
        additional_instruction=(
            "Verify that the page clearly shows a humanities-focused research center or institute at the university, "
            "with signs of activity (e.g., staff listings, events, programs, or current information). "
            "Accept names like 'Humanities Center', 'Institute for the Humanities', 'Center for Humanistic Studies', etc."
        )
    )


# --------------------------------------------------------------------------- #
# Main Evaluation Entry                                                       #
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
    # Initialize evaluator (root should be non-critical to allow partial credit)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Identify four R1 universities in specified Midwest states with Computer Science PhD programs and humanities research centers",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Record ground-truth constraints for transparency
    evaluator.add_custom_info(
        info={
            "allowed_states": ALLOWED_STATES,
            "required_designation": "Carnegie Classification 2025 R1 (Research 1: Very High Spending and Doctorate Production)",
            "required_program": "Doctoral (PhD) program in Computer Science (or equivalent naming)",
            "required_humanities_center": "Active humanities research center or institute"
        },
        info_type="constraints",
        info_name="evaluation_constraints"
    )

    # Extract universities and associated evidence from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversityExtraction,
        extraction_name="extracted_universities"
    )

    # Keep only the first 4 universities; pad with placeholders if fewer
    universities: List[UniversityItem] = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityItem())

    evaluator.add_custom_info(
        info={"reported_universities": len(extracted.universities), "evaluated_universities": 4},
        info_type="extraction_stats",
        info_name="extraction_summary"
    )

    # Build the verification tree: four parallel university nodes
    tasks = []
    for i in range(4):
        tasks.append(verify_university(evaluator, root, universities[i], i))
    # Execute verifications (can be done sequentially or concurrently)
    for t in tasks:
        await t

    return evaluator.get_summary()