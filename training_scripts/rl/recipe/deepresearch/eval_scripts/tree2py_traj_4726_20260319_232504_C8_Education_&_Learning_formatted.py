import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


TASK_ID = "four_public_universities_2026"
TASK_DESCRIPTION = """Identify four public universities in the United States, each located in a different state, that meet ALL of the following criteria for Fall 2026 freshman admissions:

1. Must be a public 4-year university (not a private institution or 2-year college)
2. Must participate in NCAA Division I athletics
3. Must offer an Early Action or Early Decision application option for Fall 2026 admission with a published deadline
4. Must have published 2025-2026 or 2026-2027 in-state undergraduate tuition and fees
5. Must have publicly available total undergraduate enrollment data
6. Must have a publicly available four-year graduation rate
7. Must offer merit-based scholarships with evidence of eligibility criteria or program descriptions
8. The four universities must represent at least three different U.S. Census regions (Northeast, Midwest, South, West)
9. Must have an official university website with verifiable information

For each of the four universities, provide:
- University name
- State location
- U.S. Census region (Northeast, Midwest, South, or West)
- Early Action or Early Decision deadline for Fall 2026 admission
- Published 2025-2026 or 2026-2027 in-state undergraduate tuition and fees (annual amount)
- Total undergraduate enrollment (most recent available data)
- Four-year graduation rate (as a percentage)
- Confirmation of NCAA Division I athletics membership with conference affiliation
- Evidence of merit-based scholarship availability with at least one specific scholarship program name
- Official university website URL

All information must be verifiable through official university websites, official NCAA sources, or recognized higher education data sources.
"""


# ------------------------- Data Models (Extraction) ------------------------- #
class UniversityItem(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    region: Optional[str] = None  # Northeast, Midwest, South, West
    official_website: Optional[str] = None

    # Institution type support
    institution_sources: List[str] = Field(default_factory=list)

    # NCAA
    ncaa_division: Optional[str] = None  # expect "Division I"
    ncaa_conference: Optional[str] = None
    ncaa_sources: List[str] = Field(default_factory=list)

    # Early application
    early_type: Optional[str] = None  # "Early Action" or "Early Decision"
    early_deadline: Optional[str] = None  # for Fall 2026
    early_sources: List[str] = Field(default_factory=list)

    # Tuition (AY 2025-2026 or 2026-2027)
    tuition_year: Optional[str] = None
    in_state_tuition_and_fees: Optional[str] = None
    tuition_sources: List[str] = Field(default_factory=list)

    # Enrollment (total undergraduate)
    enrollment_total: Optional[str] = None
    enrollment_sources: List[str] = Field(default_factory=list)

    # Graduation rate (4-year)
    grad_rate_4yr: Optional[str] = None
    grad_rate_sources: List[str] = Field(default_factory=list)

    # Merit scholarships
    merit_scholarship_name: Optional[str] = None
    scholarship_sources: List[str] = Field(default_factory=list)


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityItem] = Field(default_factory=list)


# ---------------------------- Extraction Prompt ---------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract up to four (4) universities from the answer in the same order they are presented. If more than four are mentioned, include only the first four. If fewer than four are presented, include only those provided. Do not invent information.

    For each university, extract the following fields exactly as presented in the answer:
    - name: full university name
    - state: U.S. state (full name or two-letter code)
    - region: U.S. Census region among {Northeast, Midwest, South, West} if explicitly stated; otherwise set to null
    - official_website: the official university website/homepage URL
    - institution_sources: URL list that explicitly supports the institution being a public 4-year university (e.g., "About" page, facts page, state system site)
    - ncaa_division: expected "Division I" (use whatever is explicitly written)
    - ncaa_conference: conference name (e.g., Big Ten Conference, SEC, ACC), if stated
    - ncaa_sources: URL list confirming NCAA Division I membership and/or conference (e.g., NCAA.org school page or official athletics site)
    - early_type: "Early Action" or "Early Decision" (or both if stated)
    - early_deadline: the Early Action or Early Decision published deadline for Fall 2026 admissions (keep the exact text or a clearly formatted date string as presented)
    - early_sources: URL list to the official admissions page that states the Fall 2026 early deadline
    - tuition_year: the academic year string for the in-state undergraduate tuition and fees (must be "2025-2026" or "2026-2027" if provided in the answer; otherwise null)
    - in_state_tuition_and_fees: the annual in-state undergraduate tuition+fees amount as a string (e.g., "$13,450")
    - tuition_sources: URL list to the official tuition/fee page that shows the above academic year and amount
    - enrollment_total: the total undergraduate enrollment (most recent available; keep the string as presented, e.g., "31,200")
    - enrollment_sources: URL list that supports the enrollment figure (e.g., factbook, IPEDS/NCES, common data set, institutional research)
    - grad_rate_4yr: the four-year undergraduate graduation rate as a percentage string (e.g., "55%", "55.2%")
    - grad_rate_sources: URL list that supports the four-year graduation rate
    - merit_scholarship_name: at least one specific merit scholarship program name
    - scholarship_sources: URL list to the scholarship page describing the merit-based program (must show merit criteria or say 'merit')

    Rules:
    - Extract only URLs that are explicitly present in the answer (full URLs including http/https).
    - Do not fabricate or infer URLs.
    - If a requested field is missing, set it to null (or [] for URL lists).
    - Keep strings as they appear; do not normalize numbers or dates.
    """


# --------------------------- Helper: Regions/States ------------------------- #
ALLOWED_REGIONS = {"Northeast", "Midwest", "South", "West"}

# U.S. Census regions mapping for states and DC
STATE_TO_REGION_FULL = {
    # Northeast
    "Maine": "Northeast", "New Hampshire": "Northeast", "Vermont": "Northeast",
    "Massachusetts": "Northeast", "Rhode Island": "Northeast", "Connecticut": "Northeast",
    "New York": "Northeast", "New Jersey": "Northeast", "Pennsylvania": "Northeast",
    # Midwest
    "Ohio": "Midwest", "Michigan": "Midwest", "Indiana": "Midwest", "Illinois": "Midwest",
    "Wisconsin": "Midwest", "Minnesota": "Midwest", "Iowa": "Midwest", "Missouri": "Midwest",
    "North Dakota": "Midwest", "South Dakota": "Midwest", "Nebraska": "Midwest", "Kansas": "Midwest",
    # South
    "Delaware": "South", "Maryland": "South", "District of Columbia": "South", "Virginia": "South",
    "West Virginia": "South", "North Carolina": "South", "South Carolina": "South", "Georgia": "South",
    "Florida": "South", "Kentucky": "South", "Tennessee": "South", "Mississippi": "South",
    "Alabama": "South", "Oklahoma": "South", "Texas": "South", "Arkansas": "South", "Louisiana": "South",
    # West
    "Montana": "West", "Idaho": "West", "Wyoming": "West", "Colorado": "West", "New Mexico": "West",
    "Arizona": "West", "Utah": "West", "Nevada": "West", "Washington": "West", "Oregon": "West",
    "California": "West", "Alaska": "West", "Hawaii": "West",
}

STATE_ABBR_TO_FULL = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}


def normalize_state_name(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    if len(s) == 2:
        abbr = s.upper()
        return STATE_ABBR_TO_FULL.get(abbr)
    # Title-case but keep important words
    t = " ".join(w.capitalize() if w.isalpha() else w for w in s.split())
    # Handle common variant
    if t in STATE_TO_REGION_FULL:
        return t
    # Try uppercase to match DC variants
    if s.upper() in STATE_ABBR_TO_FULL:
        return STATE_ABBR_TO_FULL[s.upper()]
    # Try known DC textual variants
    if s.lower() in {"washington, dc", "washington dc", "dc"}:
        return "District of Columbia"
    return t if t in STATE_TO_REGION_FULL else None


def region_for_state(state: Optional[str]) -> Optional[str]:
    full = normalize_state_name(state)
    if not full:
        return None
    return STATE_TO_REGION_FULL.get(full)


def unique_regions_from_unis(unis: List[UniversityItem]) -> List[str]:
    regs = []
    seen = set()
    for u in unis:
        r = u.region or region_for_state(u.state)
        if r and r in ALLOWED_REGIONS and r not in seen:
            regs.append(r)
            seen.add(r)
    return regs


def sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    res: List[str] = []
    if not urls:
        return res
    for u in urls:
        if not u:
            continue
        s = u.strip()
        if s and s.lower().startswith(("http://", "https://")) and s not in res:
            res.append(s)
    return res


def combine_sources(*lists: Optional[List[str]], also: Optional[List[str]] = None) -> List[str]:
    combined: List[str] = []
    for lst in lists:
        combined.extend(sanitize_urls(lst))
    if also:
        combined.extend(sanitize_urls(also))
    # dedupe keep order
    seen = set()
    out = []
    for u in combined:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def safe_str(x: Optional[str]) -> str:
    return x if x is not None else ""


# ----------------------------- Verification Logic -------------------------- #
async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityItem,
    idx: int,
    all_unis: List[UniversityItem],
):
    uni_num = idx + 1
    uni_node = evaluator.add_parallel(
        id=f"University_{uni_num}",
        desc=(
            "First university meeting all requirements" if idx == 0 else
            ("Second university meeting all requirements from a different state" if idx == 1 else
             ("Third university meeting all requirements from a different state" if idx == 2 else
              "Fourth university meeting all requirements from a different state"))
        ),
        parent=parent_node,
        critical=False,
    )

    # 1) Institution Type (public 4-year) - verify with sources
    inst_leaf = evaluator.add_leaf(
        id=f"U{uni_num}_Institution_Type",
        desc="University is confirmed as a public 4-year institution",
        parent=uni_node,
        critical=True
    )
    inst_claim = (
        f"The institution named '{safe_str(uni.name)}' is a public, four-year university "
        f"(i.e., not a private institution and not a 2-year college)."
    )
    inst_sources = combine_sources(uni.institution_sources, also=[safe_str(uni.official_website)])
    await evaluator.verify(
        claim=inst_claim,
        node=inst_leaf,
        sources=inst_sources,
        additional_instruction="Look for explicit evidence that the institution is 'public' and offers bachelor’s (4-year) programs. Accept phrases like 'public research university'. Reject if the page indicates private status or 2-year community college."
    )

    # 2) State Location - custom check for provided and uniqueness vs prior universities where specified
    curr_state_norm = normalize_state_name(uni.state)
    prior_states = [normalize_state_name(u.state) for u in all_unis[:idx]]
    is_unique_state = (curr_state_norm is not None) and all(
        (ps is None) or (ps != curr_state_norm) for ps in prior_states
    )
    # Description per item
    if idx == 0:
        state_desc = "University's state location is provided"
    elif idx == 1:
        state_desc = "University's state location is provided and different from University 1"
    elif idx == 2:
        state_desc = "University's state location is provided and different from Universities 1 and 2"
    else:
        state_desc = "University's state location is provided and different from Universities 1, 2, and 3"

    evaluator.add_custom_node(
        result=is_unique_state,
        id=f"U{uni_num}_State_Location",
        desc=state_desc,
        parent=uni_node,
        critical=True
    )

    # 3) Census Region - custom check for provided and valid; if state present, ensure consistency when possible
    provided_region = (uni.region or "").strip()
    valid_region_name = provided_region in ALLOWED_REGIONS if provided_region else False
    state_region = region_for_state(uni.state)
    region_consistent = True
    if valid_region_name and state_region:
        region_consistent = (provided_region == state_region)
    # If region not provided but state implies region, still fail this node per rubric ("is identified")
    evaluator.add_custom_node(
        result=valid_region_name and region_consistent,
        id=f"U{uni_num}_Census_Region",
        desc="U.S. Census region (Northeast, Midwest, South, or West) is identified",
        parent=uni_node,
        critical=True
    )

    # 4) NCAA Division I with conference - verify
    ncaa_leaf = evaluator.add_leaf(
        id=f"U{uni_num}_NCAA_Division_I",
        desc="University participates in NCAA Division I athletics with conference affiliation confirmed",
        parent=uni_node,
        critical=True
    )
    ncaa_claim = (
        f"'{safe_str(uni.name)}' participates in NCAA Division I athletics and is affiliated with the "
        f"'{safe_str(uni.ncaa_conference)}' conference (or an equivalent Division I conference)."
        if uni.ncaa_conference else
        f"'{safe_str(uni.name)}' participates in NCAA Division I athletics."
    )
    ncaa_sources = combine_sources(uni.ncaa_sources, also=[safe_str(uni.official_website)])
    await evaluator.verify(
        claim=ncaa_claim,
        node=ncaa_leaf,
        sources=ncaa_sources,
        additional_instruction="Confirm NCAA Division I membership. Accept common variants such as 'NCAA DI' or 'D1'. If a conference is named, ensure the page shows that conference affiliation."
    )

    # 5) Early Action or Early Decision deadline (Fall 2026) - verify
    early_leaf = evaluator.add_leaf(
        id=f"U{uni_num}_Early_Application_Deadline",
        desc="Early Action or Early Decision deadline for Fall 2026 is provided",
        parent=uni_node,
        critical=True
    )
    early_claim = (
        f"'{safe_str(uni.name)}' offers {safe_str(uni.early_type)} for Fall 2026 freshman admission "
        f"with a published deadline of '{safe_str(uni.early_deadline)}'."
    )
    early_sources = combine_sources(uni.early_sources, also=[safe_str(uni.official_website)])
    await evaluator.verify(
        claim=early_claim,
        node=early_leaf,
        sources=early_sources,
        additional_instruction="Verify that an Early Action or Early Decision option exists for Fall 2026 and that a specific published deadline is shown on the cited page. If multiple deadlines are listed, ensure the one provided corresponds to Early Action/Decision for Fall 2026."
    )

    # 6) Tuition (2025-2026 or 2026-2027) in-state undergrad tuition+fees - verify
    tuition_leaf = evaluator.add_leaf(
        id=f"U{uni_num}_Tuition_Published",
        desc="Published 2025-2026 or 2026-2027 in-state undergraduate tuition and fees is provided",
        parent=uni_node,
        critical=True
    )
    tuition_claim = (
        f"For academic year '{safe_str(uni.tuition_year)}', the published in-state undergraduate tuition and fees at "
        f"'{safe_str(uni.name)}' is '{safe_str(uni.in_state_tuition_and_fees)}' (annual amount)."
    )
    tuition_sources = combine_sources(uni.tuition_sources, also=[safe_str(uni.official_website)])
    await evaluator.verify(
        claim=tuition_claim,
        node=tuition_leaf,
        sources=tuition_sources,
        additional_instruction="Only mark correct if the academic year on the cited page is 2025-2026 or 2026-2027 and it shows the in-state undergraduate tuition and fees (annual). Reject if a different year, non-undergraduate, out-of-state, or per-credit-only information is given."
    )

    # 7) Enrollment (total undergraduate) - verify
    enroll_leaf = evaluator.add_leaf(
        id=f"U{uni_num}_Enrollment_Data",
        desc="Total undergraduate enrollment data is provided",
        parent=uni_node,
        critical=True
    )
    enroll_claim = (
        f"The total undergraduate enrollment at '{safe_str(uni.name)}' is '{safe_str(uni.enrollment_total)}' (most recent available)."
    )
    enroll_sources = combine_sources(uni.enrollment_sources, also=[safe_str(uni.official_website)])
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_leaf,
        sources=enroll_sources,
        additional_instruction="Verify the most recent total undergraduate enrollment figure on the cited page (institutional research, factbook, CDS, or IPEDS/NCES). Reasonable rounding differences are acceptable."
    )

    # 8) Four-year graduation rate - verify
    grad_leaf = evaluator.add_leaf(
        id=f"U{uni_num}_Graduation_Rate",
        desc="Four-year graduation rate is provided as a percentage",
        parent=uni_node,
        critical=True
    )
    grad_claim = (
        f"The four-year undergraduate graduation rate at '{safe_str(uni.name)}' is '{safe_str(uni.grad_rate_4yr)}'."
    )
    grad_sources = combine_sources(uni.grad_rate_sources, also=[safe_str(uni.official_website)])
    await evaluator.verify(
        claim=grad_claim,
        node=grad_leaf,
        sources=grad_sources,
        additional_instruction="Confirm that the cited page explicitly refers to the 4-year undergraduate graduation rate (not 5/6-year). Allow small rounding differences."
    )

    # 9) Merit-based scholarships (with named program) - verify
    merit_leaf = evaluator.add_leaf(
        id=f"U{uni_num}_Merit_Scholarships",
        desc="Evidence of merit-based scholarship availability with at least one specific program name is provided",
        parent=uni_node,
        critical=True
    )
    merit_claim = (
        f"'{safe_str(uni.name)}' offers merit-based scholarships; for example, a program named "
        f"'{safe_str(uni.merit_scholarship_name)}' exists and is merit-based."
    )
    merit_sources = combine_sources(uni.scholarship_sources, also=[safe_str(uni.official_website)])
    await evaluator.verify(
        claim=merit_claim,
        node=merit_leaf,
        sources=merit_sources,
        additional_instruction="The cited scholarship page should clearly indicate merit criteria (e.g., GPA, achievements) or explicitly say 'merit'. The named program should be visible or very close in wording."
    )

    # 10) Official website URL is provided - custom existence check
    has_official = bool(uni.official_website) and uni.official_website.strip().lower().startswith(("http://", "https://"))
    evaluator.add_custom_node(
        result=has_official,
        id=f"U{uni_num}_Official_Website",
        desc="Official university website URL is provided",
        parent=uni_node,
        critical=True
    )


# ------------------------------ Main Entrypoint ----------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
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
        extraction_name="universities_extraction",
    )

    # Normalize to exactly 4 entries (pad with empty if needed)
    universities = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityItem())

    # Build subtrees for each university
    for i in range(4):
        await verify_university(evaluator, root, universities[i], i, universities)

    # Regional diversity requirement (at least three different Census regions across the four universities)
    unique_regs = unique_regions_from_unis(universities)
    evaluator.add_custom_node(
        result=(len(unique_regs) >= 3),
        id="Regional_Diversity_Requirement",
        desc="The four universities represent at least three different U.S. Census regions",
        parent=root,
        critical=True
    )

    # Add custom information for transparency
    evaluator.add_custom_info(
        info={
            "unique_regions_detected": unique_regs,
            "total_unique_regions": len(unique_regs)
        },
        info_type="diversity_stats",
        info_name="regional_diversity_summary"
    )

    return evaluator.get_summary()