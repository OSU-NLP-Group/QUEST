import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "big_ten_public_pre1900_four_unis"
TASK_DESCRIPTION = (
    "Identify four public universities that are current members of the Big Ten Conference and were founded "
    "(established or chartered) before the year 1900. For each of the four universities, provide the following information: "
    "(1) The university's official name, (2) The year it was founded, (3) The current total student enrollment from the "
    "2024-2025 or 2025-2026 academic year, (4) Whether the university holds land-grant status under the Morrill Act (yes or no), "
    "and (5) The current official seating capacity of the university's football stadium. For each university, include a reference "
    "URL to an official university website or authoritative source that verifies the information provided."
)


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def parse_year(year_text: Optional[str]) -> Optional[int]:
    if not year_text:
        return None
    m = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", year_text)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    url = url.strip()
    if not url:
        return False
    return url.startswith("http://") or url.startswith("https://")


def normalize_yes_no(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = s.strip().lower()
    if t in {"yes", "y", "true"}:
        return "yes"
    if t in {"no", "n", "false"}:
        return "no"
    return s.strip()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    founding_year: Optional[str] = None  # Keep string to be flexible (e.g., "1867" or "chartered 1817; established 1837")
    enrollment: Optional[str] = None     # Keep as string for ranges/commas
    enrollment_year: Optional[str] = None  # e.g., "2024-2025", "AY 2024–25", "Fall 2024", "2025-2026"
    land_grant: Optional[str] = None     # "yes" or "no" (case-insensitive); keep string, normalize at use
    stadium_name: Optional[str] = None
    stadium_capacity: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    From the provided answer, extract up to the FIRST FOUR universities the answer proposes for this task.
    For each university, extract the following fields exactly as stated in the answer:
    - name: The university's official name (string).
    - founding_year: The founding/established/chartered year as written (string; do not force numeric).
    - enrollment: The current TOTAL student enrollment figure (string; keep formatting like commas; do not compute).
    - enrollment_year: The academic year for that enrollment (e.g., "2024-2025", "2025-2026", "Fall 2024", etc.).
    - land_grant: "yes" or "no" exactly as stated (string; do not infer).
    - stadium_name: The official football stadium name if provided (string, else null).
    - stadium_capacity: The current official seating capacity as stated (string; keep formatting).
    - reference_urls: A list of all URLs explicitly cited in the answer for this university that could verify the above facts.
      Only include URLs explicitly present in the answer; do not invent any URLs.

    IMPORTANT:
    - Only extract universities explicitly listed in the answer text. Do not add or infer extra items.
    - If a field is missing for a university, set it to null (or [] for the list).
    - Return an object with a single key "universities" which is an array of up to 4 UniversityItem objects.
    """


# --------------------------------------------------------------------------- #
# Verification logic per university                                           #
# --------------------------------------------------------------------------- #
async def verify_one_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    index_one_based: int,
) -> None:
    # Prepare parent node
    uni_node = evaluator.add_parallel(
        id=f"University_{index_one_based}",
        desc=f"{['First','Second','Third','Fourth'][index_one_based-1]} university meeting all criteria with complete information",
        parent=parent_node,
        critical=False
    )

    # Normalize/prepare data
    urls = [u for u in (uni.reference_urls or []) if is_valid_url(u)]
    land_flag = normalize_yes_no(uni.land_grant)
    founding_year_int = parse_year(uni.founding_year)
    name = uni.name or "the university"
    stadium_target_name = uni.stadium_name or f"{name}'s football stadium"

    # 1) Reference existence (Critical) – gate other URL-based checks
    evaluator.add_custom_node(
        result=(len(urls) > 0),
        id=f"U{index_one_based}_Reference",
        desc="Valid reference URL provided to verify the information",
        parent=uni_node,
        critical=True
    )

    # 2) Big Ten current membership (Critical)
    node_bigten = evaluator.add_leaf(
        id=f"U{index_one_based}_BigTen_Member",
        desc="University is a current member of the Big Ten Conference (2024-2025)",
        parent=uni_node,
        critical=True
    )
    bigten_claim = (
        f"{name} is a current member institution of the Big Ten Conference during the 2024–2025 academic year."
    )
    await evaluator.verify(
        claim=bigten_claim,
        node=node_bigten,
        sources=urls,
        additional_instruction=(
            "Verify that the webpages explicitly indicate the university is a Big Ten (B1G) member at present time "
            "for the 2024–2025 season/year (or up-to-date equivalent). Language like 'member of the Big Ten' is sufficient. "
            "If the sources are outdated or indicate 'will join' in a future year or 'former member', mark as not supported."
        )
    )

    # 3) Public status (Critical)
    node_public = evaluator.add_leaf(
        id=f"U{index_one_based}_Public_Status",
        desc="University is a public institution",
        parent=uni_node,
        critical=True
    )
    public_claim = f"{name} is a public university (i.e., a public/state institution)."
    await evaluator.verify(
        claim=public_claim,
        node=node_public,
        sources=urls,
        additional_instruction=(
            "Confirm the institution is described as 'public', 'public research university', 'state university', "
            "or equivalent on the cited pages."
        )
    )

    # 4) Founded before 1900 (Critical) – derived from the claimed founding_year
    pre1900_ok = founding_year_int is not None and founding_year_int < 1900
    evaluator.add_custom_node(
        result=pre1900_ok,
        id=f"U{index_one_based}_Pre1900_Founding",
        desc="University was founded before 1900",
        parent=uni_node,
        critical=True
    )

    # 5) Founding year accurate (Critical) – verify exact year statement against sources
    node_found_year = evaluator.add_leaf(
        id=f"U{index_one_based}_Founding_Year",
        desc="Founding year is accurately stated",
        parent=uni_node,
        critical=True
    )
    fy_text = uni.founding_year or ""
    founding_claim = (
        f"The founding year of {name} is {fy_text}, considering the officially recognized founding/established/chartered year."
    )
    await evaluator.verify(
        claim=founding_claim,
        node=node_found_year,
        sources=urls,
        additional_instruction=(
            "Accept synonyms such as 'founded', 'established', or 'chartered'. "
            "If multiple years are given (e.g., chartered vs. later reorganization), prefer the commonly recognized founding/established/chartered year "
            "as used by the institution itself or authoritative sources. Minor formatting differences are acceptable."
        )
    )

    # 6) Enrollment accurate and from 2024–2025 or 2025–2026 (Critical)
    node_enroll = evaluator.add_leaf(
        id=f"U{index_one_based}_Enrollment",
        desc="Current total enrollment is accurately provided from 2024-2025 or 2025-2026",
        parent=uni_node,
        critical=True
    )
    ey_text = uni.enrollment_year or ""
    enroll_val = uni.enrollment or ""
    enroll_claim = (
        f"The total student enrollment at {name} for the {ey_text} academic year is {enroll_val}."
    )
    await evaluator.verify(
        claim=enroll_claim,
        node=node_enroll,
        sources=urls,
        additional_instruction=(
            "Confirm two things: (1) the enrollment number matches what's on the source; "
            "(2) it corresponds to the 2024–2025 or 2025–2026 academic year (including equivalent phrasings like 'Fall 2024' for AY 2024–25). "
            "If the cited year is outside 2024–2025 or 2025–2026, mark as not supported."
        )
    )

    # 7) Land-grant status (Critical)
    node_land = evaluator.add_leaf(
        id=f"U{index_one_based}_LandGrant",
        desc="Land-grant status is correctly identified (yes or no)",
        parent=uni_node,
        critical=True
    )
    if land_flag == "yes":
        land_claim = f"{name} is a land-grant university under the Morrill Act."
        land_add_ins = (
            "Look for explicit mention of 'land-grant' status (Morrill Act of 1862/1890). "
            "Mentions like 'one of the nation's land-grant universities' should count."
        )
    elif land_flag == "no":
        land_claim = f"{name} is not designated as a land-grant university under the Morrill Act."
        land_add_ins = (
            "To support a 'no' claim, check official or authoritative sources listing land-grant institutions and verify this university does not appear, "
            "or that the university is explicitly described as not being a land-grant institution. "
            "If the sources are inconclusive or lack explicit evidence, mark as not supported."
        )
    else:
        # If not provided, construct a conservative claim that will likely fail against sources
        land_claim = f"The land-grant status for {name} is correctly identified in the answer."
        land_add_ins = (
            "Verify whether the answer's stated land-grant status is supported by the source(s). "
            "If the answer did not clearly specify 'yes' or 'no', or sources are missing/inconclusive, mark as not supported."
        )
    await evaluator.verify(
        claim=land_claim,
        node=node_land,
        sources=urls,
        additional_instruction=land_add_ins
    )

    # 8) Stadium seating capacity (Critical)
    node_stadium = evaluator.add_leaf(
        id=f"U{index_one_based}_Stadium",
        desc="Current official football stadium seating capacity is accurately stated",
        parent=uni_node,
        critical=True
    )
    cap_text = uni.stadium_capacity or ""
    stadium_claim = f"The official seating capacity of {stadium_target_name} is {cap_text}."
    await evaluator.verify(
        claim=stadium_claim,
        node=node_stadium,
        sources=urls,
        additional_instruction=(
            "Confirm the 'official seating capacity' as stated on authoritative or official pages (e.g., the university's athletics site). "
            "Distinguish from record attendance or temporary configurations. Minor rounding or formatting differences are acceptable."
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
    model: str = "o4-mini",
) -> Dict:
    # Initialize evaluator
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

    # Extract structured university info
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Keep first four universities; pad with empty objects if fewer
    items: List[UniversityItem] = list(extracted.universities[:4])
    while len(items) < 4:
        items.append(UniversityItem())

    # Build verification subtrees for each of the four universities
    tasks = []
    for idx in range(4):
        tasks.append(verify_one_university(evaluator, root, items[idx], idx + 1))

    # Run verifications (sequentially to respect internal preconditions in our helper)
    for t in tasks:
        await t

    return evaluator.get_summary()