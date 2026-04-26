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
TASK_ID = "ncaa_div1_universities"
TASK_DESCRIPTION = """
I am conducting a comprehensive research project on the historical development and current state of NCAA Division I athletics across the United States. To represent the diversity of American higher education and college sports, I need to identify three NCAA Division I universities that meet the following criteria:

1. Each university must be located in a different U.S. state
2. Each university must belong to a different NCAA Division I athletic conference
3. The three universities must collectively represent three different founding centuries, with one university founded in the 17th century, one in the 18th century, and one in the 20th century
4. All three universities must currently compete at the NCAA Division I level

For each of the three universities, provide the following information:

Basic Information:
- Official name of the university
- U.S. state where the university is located
- City where the university is located
- Complete street address of the main university campus

Founding and History:
- Year the university was founded

Athletic Conference:
- Name of the NCAA Division I athletic conference to which the university belongs
- Confirmation that it competes in NCAA Division I

Primary Athletic Facility:
- Name of the university's primary athletic facility (football stadium or basketball arena)
- Complete street address of this facility
- Official seating capacity of this facility

Current Enrollment:
- Current undergraduate enrollment figure (provide the most recent available data, preferably Fall 2024)

Digital Presence:
- URL of the university's official athletics website

All information must be verifiable through official university sources or reliable public records. Please ensure that each university is from a different state, a different conference, and a different founding century.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityInfo(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None  # Expected: "football stadium" or "basketball arena" if available
    street_address: Optional[str] = None
    seating_capacity: Optional[str] = None
    url: Optional[str] = None  # Facility page URL if provided


class UniversityInfo(BaseModel):
    official_name: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    main_campus_street_address: Optional[str] = None
    founding_year: Optional[str] = None
    conference_name: Optional[str] = None
    division_i_confirmation_text: Optional[str] = None  # Any wording like "NCAA Division I", "Division I"
    primary_facility: FacilityInfo = Field(default_factory=FacilityInfo)
    undergraduate_enrollment: Optional[str] = None
    official_athletics_website_url: Optional[str] = None
    sources: List[str] = Field(default_factory=list)  # Any additional sources mentioned for this university


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to the first three NCAA Division I universities described in the answer, in the same order they appear.
    For each university, extract the following fields exactly as presented in the answer:

    - official_name: Official name of the university
    - state: U.S. state where the university is located
    - city: City where the university is located
    - main_campus_street_address: Complete street address of the main campus
    - founding_year: The year the university was founded (prefer a 4-digit year; if a range or fuzzy text is given, extract the best 4-digit year if present; otherwise extract the text as-is)
    - conference_name: The NCAA Division I athletic conference the university belongs to
    - division_i_confirmation_text: The wording used in the answer to indicate the university competes at NCAA Division I (e.g., "NCAA Division I", "Division I", "D-I"); return null if not explicitly stated
    - primary_facility: An object with:
        • name: Name of the primary athletic facility (football stadium or basketball arena)
        • type: One of "football stadium" or "basketball arena" if explicitly stated or clearly implied; otherwise null
        • street_address: Complete street address of the facility
        • seating_capacity: Official seating capacity (keep as a string exactly as provided in the answer)
        • url: A URL specifically for this facility if provided in the answer; otherwise null
    - undergraduate_enrollment: Most recent undergraduate enrollment figure (string as presented)
    - official_athletics_website_url: URL of the university's official athletics website
    - sources: An array of any additional URLs cited in the answer for this university (e.g., official pages like .edu, NCAA, conference pages, facility pages). Do not include the 'official_athletics_website_url' again here.

    Rules:
    - Do not invent data; only extract what is explicitly in the answer.
    - For URLs, extract valid URLs exactly as written (support both plain URLs and markdown links).
    - If a field is missing, set it to null (or [] for arrays).
    - Return a JSON object with a 'universities' array containing up to three such university objects.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _norm(s: Optional[str]) -> str:
    return s.strip() if isinstance(s, str) else ""


def dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def collect_sources(u: UniversityInfo) -> List[str]:
    urls: List[str] = []
    if u.official_athletics_website_url:
        urls.append(u.official_athletics_website_url)
    if u.primary_facility and u.primary_facility.url:
        urls.append(u.primary_facility.url)
    urls.extend(u.sources or [])
    return dedup_preserve_order(urls)


def interpret_facility_type(u: UniversityInfo) -> Optional[str]:
    """
    Determine if the facility is a football stadium or a basketball arena using provided type,
    or heuristics on the facility name if type is missing.
    """
    t = _norm(getattr(u.primary_facility, "type", None)).lower()
    if t:
        if "football" in t:
            return "football stadium"
        if "basketball" in t:
            return "basketball arena"

    # Heuristic from facility name if explicit type is missing
    name = _norm(getattr(u.primary_facility, "name", None)).lower()
    if name:
        if "stadium" in name:
            return "football stadium"
        if "arena" in name:
            return "basketball arena"
    return None


def parse_year_to_century(year_str: Optional[str]) -> Optional[str]:
    """
    Parse a year string to a century label ('17th', '18th', '20th') if within those ranges.
    Uses the first 4-digit year found.
    """
    if not year_str:
        return None
    m = re.search(r"(1[5-9]\d{2}|20\d{2}|17\d{2}|16\d{2})", year_str)
    if not m:
        return None
    try:
        year = int(m.group(0))
    except Exception:
        return None

    if 1601 <= year <= 1700:
        return "17th"
    if 1701 <= year <= 1800:
        return "18th"
    if 1901 <= year <= 2000:
        return "20th"
    return None


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    u: UniversityInfo,
    index_1based: int,
) -> None:
    """
    Build verification subtree for a single university.
    """
    uni_node = evaluator.add_parallel(
        id=f"university_{index_1based}",
        desc=f"University #{index_1based}: required fields provided",
        parent=parent_node,
        critical=False
    )

    # Presence checks (critical)
    evaluator.add_custom_node(
        result=bool(_norm(u.official_name)),
        id=f"university_{index_1based}_official_name",
        desc="Provides the official name of the university",
        parent=uni_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_norm(u.state)),
        id=f"university_{index_1based}_state",
        desc="Provides the U.S. state where the university is located",
        parent=uni_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_norm(u.city)),
        id=f"university_{index_1based}_city",
        desc="Provides the city where the university is located",
        parent=uni_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_norm(u.main_campus_street_address)),
        id=f"university_{index_1based}_main_campus_street_address",
        desc="Provides the complete street address of the main university campus",
        parent=uni_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_norm(u.founding_year)),
        id=f"university_{index_1based}_founding_year",
        desc="Provides the year the university was founded",
        parent=uni_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_norm(u.conference_name)),
        id=f"university_{index_1based}_conference_name",
        desc="Provides the name of the NCAA Division I athletic conference the university belongs to",
        parent=uni_node,
        critical=True
    )

    # Division I confirmation - verify with available sources
    division_leaf = evaluator.add_leaf(
        id=f"university_{index_1based}_division_i_confirmation",
        desc="Confirms the university currently competes at NCAA Division I level",
        parent=uni_node,
        critical=True
    )
    uni_name = _norm(u.official_name) or f"University #{index_1based}"
    di_claim = f"{uni_name} currently competes at the NCAA Division I level."
    await evaluator.verify(
        claim=di_claim,
        node=division_leaf,
        sources=collect_sources(u),
        additional_instruction="Use the provided URLs (official athletics site, NCAA, or conference pages) to confirm that the university competes in NCAA Division I. Accept synonyms like 'Division I', 'D-I', or 'NCAA Division 1'."
    )

    evaluator.add_custom_node(
        result=bool(_norm(getattr(u.primary_facility, "name", None))),
        id=f"university_{index_1based}_primary_facility_name",
        desc="Provides the name of the primary athletic facility",
        parent=uni_node,
        critical=True
    )

    # Facility type check (football stadium or basketball arena)
    computed_type = interpret_facility_type(u)
    fac_type_ok = computed_type in ("football stadium", "basketball arena")
    evaluator.add_custom_node(
        result=fac_type_ok,
        id=f"university_{index_1based}_primary_facility_type",
        desc="Primary athletic facility is a football stadium or a basketball arena (as specified in the prompt)",
        parent=uni_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_norm(getattr(u.primary_facility, "street_address", None))),
        id=f"university_{index_1based}_primary_facility_street_address",
        desc="Provides the complete street address of the primary athletic facility",
        parent=uni_node,
        critical=True
    )

    # Seating capacity verification (field is required; we verify against provided sources)
    capacity_leaf = evaluator.add_leaf(
        id=f"university_{index_1based}_primary_facility_seating_capacity",
        desc="Provides the official seating capacity of the primary athletic facility",
        parent=uni_node,
        critical=True
    )
    fac_name = _norm(getattr(u.primary_facility, "name", None)) or "the primary athletic facility"
    capacity_val = _norm(getattr(u.primary_facility, "seating_capacity", None))
    cap_claim = f"The official seating capacity of {fac_name} is {capacity_val}."
    await evaluator.verify(
        claim=cap_claim,
        node=capacity_leaf,
        sources=collect_sources(u),
        additional_instruction="Verify the seating capacity from official or reliable sources (facility page, athletics site, or university site). Allow minor rounding or formatting differences (e.g., commas)."
    )

    evaluator.add_custom_node(
        result=bool(_norm(u.undergraduate_enrollment)),
        id=f"university_{index_1based}_undergraduate_enrollment",
        desc="Provides the most recent available undergraduate enrollment figure (Fall 2024 preferred when available)",
        parent=uni_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=bool(_norm(u.official_athletics_website_url)),
        id=f"university_{index_1based}_official_athletics_website_url",
        desc="Provides the URL of the university's official athletics website",
        parent=uni_node,
        critical=True
    )

    # Verifiable sources presence (at least one source must be present)
    evaluator.add_custom_node(
        result=len(collect_sources(u)) > 0,
        id=f"university_{index_1based}_verifiable_sources",
        desc="Information is verifiable through official university sources or reliable public records (e.g., includes citations/links to such sources)",
        parent=uni_node,
        critical=True
    )


def all_distinct_nonempty(values: List[Optional[str]]) -> bool:
    cleaned = [v.strip().lower() for v in values if isinstance(v, str) and v.strip()]
    return len(cleaned) == 3 and len(set(cleaned)) == 3


def check_century_distribution(unis: List[UniversityInfo]) -> bool:
    centuries = [parse_year_to_century(u.founding_year) for u in unis]
    if any(c is None for c in centuries):
        return False
    target = {"17th", "18th", "20th"}
    # Exactly one in each of these centuries
    return set(centuries) == target and all(centuries.count(c) == 1 for c in target)


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the NCAA Division I universities task and return a structured result dictionary.
    """
    # Initialize evaluator (root kept non-critical to allow non-critical children; critical gating happens in child nodes)
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

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Keep first 3 universities; pad with empty placeholders if fewer than 3
    universities: List[UniversityInfo] = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityInfo())

    # Record some custom info
    evaluator.add_custom_info(
        info={"num_universities_extracted": len(extracted.universities), "used_first": 3},
        info_type="extraction_stats",
        info_name="extraction_statistics"
    )
    evaluator.add_ground_truth({
        "required_cross_constraints": {
            "different_states": True,
            "different_conferences": True,
            "founding_centuries_exactly": ["17th", "18th", "20th"]
        }
    }, gt_type="expected_constraints")

    # Build per-university verification subtrees
    for idx, uni in enumerate(universities, start=1):
        await verify_university(evaluator, root, uni, idx)

    # Global cross-university constraints (critical gate)
    global_node = evaluator.add_parallel(
        id="global_constraints",
        desc="Cross-university constraints required by the prompt",
        parent=root,
        critical=True
    )

    # Different states
    evaluator.add_custom_node(
        result=all_distinct_nonempty([u.state for u in universities]),
        id="different_states",
        desc="Each university is located in a different U.S. state",
        parent=global_node,
        critical=True
    )

    # Different conferences
    evaluator.add_custom_node(
        result=all_distinct_nonempty([u.conference_name for u in universities]),
        id="different_conferences",
        desc="Each university belongs to a different NCAA Division I athletic conference",
        parent=global_node,
        critical=True
    )

    # Founding century distribution
    evaluator.add_custom_node(
        result=check_century_distribution(universities),
        id="founding_century_distribution",
        desc="The set includes exactly one university founded in the 17th century, one in the 18th century, and one in the 20th century",
        parent=global_node,
        critical=True
    )

    # Return the structured evaluation summary
    return evaluator.get_summary()