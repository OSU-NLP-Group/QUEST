import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "central_office_leadership_tx_nc_ma"
TASK_DESCRIPTION = (
    "Identify four currently open or recently posted (within the past 3 months) central office leadership "
    "positions in public school districts located in Texas, North Carolina, and Massachusetts. The positions "
    "must meet all specified criteria and include all required information elements."
)

EVAL_DATE = datetime(2026, 3, 22)
RECENT_WINDOW_DAYS = 90
RECENT_WINDOW_START = (EVAL_DATE - timedelta(days=RECENT_WINDOW_DAYS)).date()

# --------------------------------------------------------------------------- #
# Data Models                                                                 #
# --------------------------------------------------------------------------- #
class PositionItem(BaseModel):
    title: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    url: Optional[str] = None
    posting_date: Optional[str] = None  # Free text as stated in the answer
    education_requirements: Optional[str] = None
    certification_requirements: Optional[str] = None
    min_years_experience: Optional[str] = None
    salary_info: Optional[str] = None


class PositionsExtraction(BaseModel):
    positions: List[PositionItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction Prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_positions() -> str:
    return """
    Extract up to FOUR central office leadership positions mentioned in the answer text.
    For each position, extract the following fields exactly as stated in the answer:
    - title: the exact position title
    - district: the public school district name
    - state: the U.S. state (use full name or state code if explicitly given)
    - url: the URL where the posting can be verified
    - posting_date: the posting or update date text if provided (e.g., "Posted February 12, 2026" or "Posted 14 days ago")
    - education_requirements: the educational requirements as stated (e.g., "Master's degree in education required")
    - certification_requirements: the certification/licensure requirements as stated (or null if not mentioned)
    - min_years_experience: the minimum years of experience required as stated (e.g., "5 years", "5-7 years", "at least five years")
    - salary_info: salary or salary range text if provided (or null if not mentioned)
    
    Important rules:
    - Extract only what is explicitly present in the answer. Do not invent or infer details.
    - If more than 4 positions are present, keep only the first 4 mentioned.
    - If any field is not present for a position, set it to null.
    - The positions should be leadership roles at the district central office level (e.g., Director, Executive Director, Chief Officer, Assistant/Associate/Deputy Superintendent).
    - The verification URLs should correspond to official district sites, state education job boards, or recognized education employment platforms if such URLs are provided in the answer.
    
    Return a JSON object with field:
    - positions: an array of at most 4 items, each with the fields specified above.
    """


# --------------------------------------------------------------------------- #
# Helper Functions                                                            #
# --------------------------------------------------------------------------- #
STATE_MAP = {
    "texas": "TX",
    "tx": "TX",
    "north carolina": "NC",
    "nc": "NC",
    "massachusetts": "MA",
    "ma": "MA",
}


def normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    key = s.strip().lower()
    return STATE_MAP.get(key, s.strip())


def state_full_name(code_or_name: Optional[str]) -> Optional[str]:
    if not code_or_name:
        return None
    code = normalize_state(code_or_name)
    if not code:
        return None
    if code.upper() == "TX":
        return "Texas"
    if code.upper() == "NC":
        return "North Carolina"
    if code.upper() == "MA":
        return "Massachusetts"
    # If already a full name or unknown, return as is
    return code_or_name


def requires_masters(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    keywords = [
        "master's", "masters", "m.ed", "m.ed.", "m.a.", "m.s.", "ms", "ma", "m.a", "m.s",
        "mba", "m.p.a", "mpa", "mpp", "m.p.p", "m.sc", "msc"
    ]
    return any(k in t for k in keywords)


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12
}


def parse_min_years(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    t = text.lower()
    import re
    # numeric forms
    m = re.search(r'(\d+)\s*(?:\+|\s*or\s*more)?\s*(?:years?|yrs?)', t)
    if m:
        try:
            return int(m.group(1))
        except:
            pass
    rng = re.search(r'(\d+)\s*(?:-|to)\s*(\d+)\s*(?:years?|yrs?)', t)
    if rng:
        try:
            return int(rng.group(1))
        except:
            pass
    # word forms
    for w, n in NUMBER_WORDS.items():
        if re.search(rf'\b{w}\b.*?(years?|yrs?)', t):
            return n
    return None


def has_state_specific_cert(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    # Look for explicit state references or agency references together with cert/license terms
    cert_words = ["certification", "certificate", "licensure", "license", "licensing", "licensed"]
    state_terms = [
        "texas", "tea", "sbec",  # TX
        "north carolina", "nc dpi", "ncdpi",  # NC
        "massachusetts", "dese", "ma dese"  # MA
    ]
    return any(sw in t for sw in cert_words) and any(st in t for st in state_terms)


def first_n_positions(items: List[PositionItem], n: int = 4) -> List[PositionItem]:
    return items[:n]


def pad_positions(items: List[PositionItem], target: int = 4) -> List[PositionItem]:
    out = list(items)
    while len(out) < target:
        out.append(PositionItem())
    return out[:target]


# --------------------------------------------------------------------------- #
# Verification for a Single Position                                          #
# --------------------------------------------------------------------------- #
async def verify_single_position(
    evaluator: Evaluator,
    parent_node,
    pos: PositionItem,
    index_1based: int,
) -> None:
    # Parent position node (parallel, non-critical to allow partial credit per position)
    pos_node = evaluator.add_parallel(
        id=f"position_{index_1based}",
        desc=[
            "First", "Second", "Third", "Fourth"
        ][index_1based - 1] + " qualifying central office leadership position",
        parent=parent_node,
        critical=False
    )

    # 1) Identification (sequential, critical)
    ident_node = evaluator.add_sequential(
        id=f"position_{index_1based}_identification",
        desc="Position identification and verification",
        parent=pos_node,
        critical=True
    )

    # 1.a) Exists
    if pos.url and pos.url.strip():
        exists_leaf = evaluator.add_leaf(
            id=f"position_{index_1based}_exists",
            desc="Position posting exists and is accessible via provided URL",
            parent=ident_node,
            critical=True
        )
        claim_exists = (
            f"The provided URL resolves to a valid job posting or job listing page for a leadership/administrative role. "
            f"URL: {pos.url}"
        )
        await evaluator.verify(
            claim=claim_exists,
            node=exists_leaf,
            sources=pos.url,
            additional_instruction="If the page loads and clearly represents a job posting (district site, state job board, or recognized platform like Frontline/Applitrack/SchoolSpring/GovernmentJobs), consider this satisfied."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"position_{index_1based}_exists",
            desc="Position posting exists and is accessible via provided URL",
            parent=ident_node,
            critical=True
        )

    # 1.b) Timeframe within past 3 months
    timeframe_leaf = evaluator.add_leaf(
        id=f"position_{index_1based}_timeframe",
        desc="Position was posted within the past 3 months",
        parent=ident_node,
        critical=True
    )
    claim_timeframe = (
        f"This job posting was first posted (or last updated) within the past {RECENT_WINDOW_DAYS} days before {EVAL_DATE.date()}. "
        f"Use explicit posting dates or 'posted X days ago' style indicators to decide."
    )
    await evaluator.verify(
        claim=claim_timeframe,
        node=timeframe_leaf,
        sources=pos.url if (pos.url and pos.url.strip()) else None,
        additional_instruction=(
            "If only a 'posted X days ago' indicator is available, consider X <= 90 as within 3 months. "
            "If a clear posting date is shown, check that it is on or after "
            f"{RECENT_WINDOW_START.isoformat()}. If neither appears, mark as not supported."
        )
    )

    # 1.c) Title is director-level or above
    title_leaf = evaluator.add_leaf(
        id=f"position_{index_1based}_title",
        desc="Position title is director-level or above",
        parent=ident_node,
        critical=True
    )
    display_title = pos.title or "the role"
    claim_title = (
        f"The job title '{display_title}' is director-level or above (acceptable examples: Director, Executive Director, "
        f"Managing/Senior Director, Chief Officer, Chief of Staff, Assistant/Associate/Deputy/Superintendent)."
    )
    await evaluator.verify(
        claim=claim_title,
        node=title_leaf,
        sources=pos.url if (pos.url and pos.url.strip()) else None,
        additional_instruction="Use reasonable judgment from the posting title and context. If clearly below director-level (e.g., Coordinator, Specialist) mark as not supported."
    )

    # 1.d) Central office (district-level) administrative role
    central_office_leaf = evaluator.add_leaf(
        id=f"position_{index_1based}_central_office",
        desc="Position is a central office (district-level) administrative role",
        parent=ident_node,
        critical=True
    )
    claim_central = (
        "The role is a central office/district-level administrative position (not school-based like teacher, school principal, AP, or dean)."
    )
    await evaluator.verify(
        claim=claim_central,
        node=central_office_leaf,
        sources=pos.url if (pos.url and pos.url.strip()) else None,
        additional_instruction="Look for department/office context, district-level responsibilities, and non-school-based scope."
    )

    # 2) Location (parallel, critical)
    loc_node = evaluator.add_parallel(
        id=f"position_{index_1based}_location",
        desc="Geographic and institutional requirements",
        parent=pos_node,
        critical=True
    )

    # 2.a) State is TX/NC/MA
    state_leaf = evaluator.add_leaf(
        id=f"position_{index_1based}_state",
        desc="Position is in Texas, North Carolina, or Massachusetts",
        parent=loc_node,
        critical=True
    )
    sfull = state_full_name(pos.state) or "the claimed state"
    claim_state = (
        f"The employer's location on this posting is in {sfull}, which is one of: Texas, North Carolina, Massachusetts."
    )
    await evaluator.verify(
        claim=claim_state,
        node=state_leaf,
        sources=pos.url if (pos.url and pos.url.strip()) else None,
        additional_instruction="Allow state to be indicated by district name (e.g., 'ISD' for Texas), address, or page branding."
    )

    # 2.b) Public school district
    public_leaf = evaluator.add_leaf(
        id=f"position_{index_1based}_public_district",
        desc="Position is in a public school district",
        parent=loc_node,
        critical=True
    )
    display_dist = pos.district or "the listed employer"
    claim_public = (
        f"The employer '{display_dist}' is a public school district (e.g., ISD, County Public Schools, Public Schools)."
    )
    await evaluator.verify(
        claim=claim_public,
        node=public_leaf,
        sources=pos.url if (pos.url and pos.url.strip()) else None,
        additional_instruction="If the page indicates 'Independent School District', 'Public Schools', 'County Schools' (district), treat as public."
    )

    # 3) Qualifications (parallel, critical as per rubric; all children must be critical to satisfy framework rule)
    qual_node = evaluator.add_parallel(
        id=f"position_{index_1based}_qualifications",
        desc="Educational and professional qualifications",
        parent=pos_node,
        critical=True
    )

    # 3.a) Education requirements clearly stated (critical)
    edu_leaf = evaluator.add_leaf(
        id=f"position_{index_1based}_education",
        desc="Educational requirements are clearly stated",
        parent=qual_node,
        critical=True
    )
    claim_edu = (
        "The posting explicitly states educational requirements (degree level and/or field) for the role."
    )
    await evaluator.verify(
        claim=claim_edu,
        node=edu_leaf,
        sources=pos.url if (pos.url and pos.url.strip()) else None,
        additional_instruction="Look for requirements like Bachelor's, Master's, Doctorate and possible fields (education, administration, related)."
    )

    # 3.b) Certification/licensure documented if applicable
    # To satisfy the 'critical parent' rule, make this critical but pass as N/A when not specified
    if pos.certification_requirements and pos.certification_requirements.strip():
        cert_leaf = evaluator.add_leaf(
            id=f"position_{index_1based}_certification",
            desc="Certification/licensure requirements are documented if applicable",
            parent=qual_node,
            critical=True
        )
        claim_cert = (
            "The posting specifies administrator certification or licensure requirements (e.g., state-issued administrator, superintendent, or principal license/certificate)."
        )
        await evaluator.verify(
            claim=claim_cert,
            node=cert_leaf,
            sources=pos.url if (pos.url and pos.url.strip()) else None,
            additional_instruction="Wording like 'appropriate state certification' also satisfies this."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"position_{index_1based}_certification",
            desc="Certification/licensure requirements are documented if applicable (N/A - not specified in posting)",
            parent=qual_node,
            critical=True
        )

    # 3.c) Experience specified (critical)
    exp_leaf = evaluator.add_leaf(
        id=f"position_{index_1based}_experience",
        desc="Minimum years of experience are specified",
        parent=qual_node,
        critical=True
    )
    claim_exp = (
        "The posting specifies minimum years of professional experience required (e.g., 'at least 5 years', '5-7 years')."
    )
    await evaluator.verify(
        claim=claim_exp,
        node=exp_leaf,
        sources=pos.url if (pos.url and pos.url.strip()) else None,
        additional_instruction="Accept ranges (e.g., 5-7 years) or phrasing like 'minimum of five years'."
    )

    # 4) State alignment (parallel, critical)
    align_node = evaluator.add_parallel(
        id=f"position_{index_1based}_state_alignment",
        desc="Position qualifications align with state administrator standards",
        parent=pos_node,
        critical=True
    )

    # 4.a) Educational requirements align with state standards (critical)
    edu_align_leaf = evaluator.add_leaf(
        id=f"position_{index_1based}_state_education_requirements",
        desc="Educational requirements align with state standards for administrators",
        parent=align_node,
        critical=True
    )
    st_full = state_full_name(pos.state) or "the state"
    claim_align_edu = (
        f"The posting's stated educational qualifications align with typical {st_full} administrator standards "
        f"(for example, requiring a Master's degree and/or appropriate state administrator certification)."
    )
    await evaluator.verify(
        claim=claim_align_edu,
        node=edu_align_leaf,
        sources=pos.url if (pos.url and pos.url.strip()) else None,
        additional_instruction=(
            "Treat alignment as satisfied if the posting requires a graduate degree commonly expected for administrators "
            "and/or explicitly references appropriate state administrator certification for the state."
        )
    )

    # 4.b) If certification specified, it aligns with state standards (make critical but allow N/A pass)
    if pos.certification_requirements and pos.certification_requirements.strip():
        cert_align_leaf = evaluator.add_leaf(
            id=f"position_{index_1based}_state_certification_requirements",
            desc="If certification requirements are specified, they align with state education agency standards",
            parent=align_node,
            critical=True
        )
        claim_align_cert = (
            f"The certification/licensure referenced by the posting is a valid {st_full} administrator credential as defined by the state education agency."
        )
        await evaluator.verify(
            claim=claim_align_cert,
            node=cert_align_leaf,
            sources=pos.url if (pos.url and pos.url.strip()) else None,
            additional_instruction="Language such as 'appropriate state administrator certification', 'MA DESE license', 'NC DPI license', or 'Texas administrator certificate' indicates alignment."
        )
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"position_{index_1based}_state_certification_requirements",
            desc="If certification requirements are specified, they align with state standards (N/A - not specified in posting)",
            parent=align_node,
            critical=True
        )

    # 5) Complete required information (critical)
    complete_info_ok = all([
        bool(pos.title and pos.title.strip()),
        bool(pos.district and pos.district.strip()),
        bool(pos.state and pos.state.strip()),
        bool(pos.url and pos.url.strip()),
        bool(pos.education_requirements and pos.education_requirements.strip()),
        bool(pos.min_years_experience and pos.min_years_experience.strip()),
    ])
    evaluator.add_custom_node(
        result=complete_info_ok,
        id=f"position_{index_1based}_complete_information",
        desc="All required information elements are provided (title, district, state, URL, education requirements, experience requirements)",
        parent=pos_node,
        critical=True
    )

    # 6) Salary documented if available (non-critical)
    if pos.url and pos.url.strip():
        salary_leaf = evaluator.add_leaf(
            id=f"position_{index_1based}_salary_documented",
            desc="Salary information is provided if available in posting",
            parent=pos_node,
            critical=False
        )
        claim_salary = "The posting page includes salary information or a salary range for the position."
        await evaluator.verify(
            claim=claim_salary,
            node=salary_leaf,
            sources=pos.url,
            additional_instruction="If salary is not present on the page, this should be marked as not supported."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"position_{index_1based}_salary_documented",
            desc="Salary information is provided if available in posting",
            parent=pos_node,
            critical=False
        )


# --------------------------------------------------------------------------- #
# Aggregate Constraints                                                       #
# --------------------------------------------------------------------------- #
def compute_aggregate_counters(positions: List[PositionItem]) -> Dict[str, Any]:
    norm_states = []
    masters_flags = []
    cert_flags_state_specific = []
    five_plus_flags = []

    for p in positions:
        norm_states.append(normalize_state(p.state) if p and p.state else None)
        masters_flags.append(requires_masters(p.education_requirements))
        cert_flags_state_specific.append(has_state_specific_cert(p.certification_requirements))
        years = parse_min_years(p.min_years_experience)
        five_plus_flags.append((years is not None and years >= 5))

    return {
        "states": norm_states,
        "masters_count": sum(1 for x in masters_flags if x),
        "has_state_specific_cert": any(cert_flags_state_specific),
        "has_five_plus": any(five_plus_flags),
    }


async def add_aggregate_nodes(evaluator: Evaluator, root, positions: List[PositionItem]) -> None:
    counters = compute_aggregate_counters(positions)
    states_present = set([s.upper() for s in counters["states"] if isinstance(s, str)])
    masters_count = counters["masters_count"]
    has_state_specific_cert = counters["has_state_specific_cert"]
    has_five_plus = counters["has_five_plus"]

    agg_node = evaluator.add_parallel(
        id="aggregate_constraints",
        desc="Cross-position aggregate requirements",
        parent=root,
        critical=True
    )

    # a) State distribution (critical)
    state_dist_ok = all(code in states_present for code in ["TX", "NC", "MA"])
    evaluator.add_custom_node(
        result=state_dist_ok,
        id="state_distribution",
        desc="At least one position from each of the three states (Texas, North Carolina, Massachusetts)",
        parent=agg_node,
        critical=True
    )

    # b) Master's degree requirement (critical) - at least 2 positions
    masters_ok = masters_count >= 2
    evaluator.add_custom_node(
        result=masters_ok,
        id="masters_degree_requirement",
        desc="At least two of the four positions require a Master's degree",
        parent=agg_node,
        critical=True
    )

    # c) Certification requirement (critical) - at least one position with state-specific admin cert
    cert_ok = has_state_specific_cert
    evaluator.add_custom_node(
        result=cert_ok,
        id="certification_requirement",
        desc="At least one position explicitly requires state-specific administrator certification or licensure",
        parent=agg_node,
        critical=True
    )

    # d) Experience requirement (critical) - at least one with 5+ years
    exp_ok = has_five_plus
    evaluator.add_custom_node(
        result=exp_ok,
        id="experience_requirement",
        desc="At least one position specifies minimum 5 or more years of professional experience",
        parent=agg_node,
        critical=True
    )

    # Record some helpful custom info for debugging
    evaluator.add_custom_info(
        info={
            "states_present": sorted(list(states_present)),
            "masters_count": masters_count,
            "has_state_specific_cert": has_state_specific_cert,
            "has_five_plus": has_five_plus,
        },
        info_type="aggregate_counters",
        info_name="aggregate_counters"
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

    # Extract up to 4 positions from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_positions(),
        template_class=PositionsExtraction,
        extraction_name="positions_extraction"
    )

    positions = first_n_positions(extracted.positions if extracted and extracted.positions else [], 4)
    positions = pad_positions(positions, 4)

    # Add some GT/context info
    evaluator.add_ground_truth({
        "required_states": ["Texas (TX)", "North Carolina (NC)", "Massachusetts (MA)"],
        "min_window_days": RECENT_WINDOW_DAYS,
        "eval_date": EVAL_DATE.isoformat()
    })

    # Build verification subtrees for each of the 4 positions
    for i in range(4):
        await verify_single_position(
            evaluator=evaluator,
            parent_node=root,
            pos=positions[i],
            index_1based=i + 1
        )

    # Add cross-position aggregate constraints
    await add_aggregate_nodes(evaluator, root, positions)

    # Return structured summary
    return evaluator.get_summary()