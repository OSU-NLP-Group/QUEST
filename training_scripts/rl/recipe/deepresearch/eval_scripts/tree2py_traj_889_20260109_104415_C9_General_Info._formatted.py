import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cs_phd_4states"
TASK_DESCRIPTION = """
Identify four public universities in the United States, one from each of the following four states: California, Texas, New York, and one additional state of your choice (different from the first three), that offer PhD programs in Computer Science. For each of the four universities, provide the following information:

1. Admission Requirements:
   - The minimum GPA requirement for PhD admission
   - Whether GRE scores are required for Fall 2025 or Fall 2026 admission
   - The application deadline for PhD admission
   - The minimum TOEFL or IELTS score requirement for international students

2. Program Characteristics:
   - The typical or expected duration to complete the PhD degree (in years)
   - At least three research areas or specializations offered by the program

3. Funding Information:
   - Confirmation that research assistantships (RA) or teaching assistantships (TA) are available to PhD students

For each piece of information provided, include a reference URL from the university's official website that supports your answer.
"""

# --------------------------------------------------------------------------- #
# US States helpers                                                           #
# --------------------------------------------------------------------------- #
_STATE_LIST: List[Tuple[str, str]] = [
    ("Alabama", "AL"), ("Alaska", "AK"), ("Arizona", "AZ"), ("Arkansas", "AR"),
    ("California", "CA"), ("Colorado", "CO"), ("Connecticut", "CT"), ("Delaware", "DE"),
    ("Florida", "FL"), ("Georgia", "GA"), ("Hawaii", "HI"), ("Idaho", "ID"),
    ("Illinois", "IL"), ("Indiana", "IN"), ("Iowa", "IA"), ("Kansas", "KS"),
    ("Kentucky", "KY"), ("Louisiana", "LA"), ("Maine", "ME"), ("Maryland", "MD"),
    ("Massachusetts", "MA"), ("Michigan", "MI"), ("Minnesota", "MN"), ("Mississippi", "MS"),
    ("Missouri", "MO"), ("Montana", "MT"), ("Nebraska", "NE"), ("Nevada", "NV"),
    ("New Hampshire", "NH"), ("New Jersey", "NJ"), ("New Mexico", "NM"), ("New York", "NY"),
    ("North Carolina", "NC"), ("North Dakota", "ND"), ("Ohio", "OH"), ("Oklahoma", "OK"),
    ("Oregon", "OR"), ("Pennsylvania", "PA"), ("Rhode Island", "RI"), ("South Carolina", "SC"),
    ("South Dakota", "SD"), ("Tennessee", "TN"), ("Texas", "TX"), ("Utah", "UT"),
    ("Vermont", "VT"), ("Virginia", "VA"), ("Washington", "WA"), ("West Virginia", "WV"),
    ("Wisconsin", "WI"), ("Wyoming", "WY")
]
STATE_ABBR_BY_NAME: Dict[str, str] = {name.upper(): abbr for name, abbr in _STATE_LIST}
STATE_NAME_BY_ABBR: Dict[str, str] = {abbr: name for name, abbr in _STATE_LIST}


def normalize_state_to_abbr(state_str: Optional[str]) -> Optional[str]:
    if not state_str:
        return None
    s = state_str.strip()
    if not s:
        return None
    if len(s) == 2 and s.upper() in STATE_NAME_BY_ABBR:
        return s.upper()
    key = s.replace(".", "").strip().upper()
    return STATE_ABBR_BY_NAME.get(key)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FieldWithUrls(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class AreasWithUrls(BaseModel):
    areas: List[str] = Field(default_factory=list)
    urls: List[str] = Field(default_factory=list)


class UniversityItem(BaseModel):
    # Identification
    state: Optional[str] = None  # two-letter code or full name (we will normalize)
    university_name: Optional[str] = None
    program_name: Optional[str] = None  # e.g., "PhD in Computer Science" or "Computer Science & Engineering"
    program_urls: List[str] = Field(default_factory=list)  # official URLs verifying PhD program exists
    public_location_urls: List[str] = Field(default_factory=list)  # official URLs verifying public status + state

    # Required attributes
    minimum_gpa: FieldWithUrls = Field(default_factory=FieldWithUrls)
    gre_term_label: Optional[str] = None  # "Fall 2025" or "Fall 2026"
    gre_required: Optional[str] = None  # "required" | "not required" | "optional"
    gre_urls: List[str] = Field(default_factory=list)

    application_deadline: FieldWithUrls = Field(default_factory=FieldWithUrls)
    english_minimum: FieldWithUrls = Field(default_factory=FieldWithUrls)  # TOEFL/IELTS minimums
    typical_duration: FieldWithUrls = Field(default_factory=FieldWithUrls)  # in years (string like "4-6")
    research_areas: AreasWithUrls = Field(default_factory=AreasWithUrls)  # at least three
    ra_ta_availability: FieldWithUrls = Field(default_factory=FieldWithUrls)  # yes/no description


class FourUniversitiesExtraction(BaseModel):
    california: Optional[UniversityItem] = None
    texas: Optional[UniversityItem] = None
    new_york: Optional[UniversityItem] = None
    fourth: Optional[UniversityItem] = None  # state should be different from CA/TX/NY


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
Extract structured information for exactly four U.S. public universities offering an active PhD program in Computer Science (or Computer Science & Engineering):
- One in California
- One in Texas
- One in New York
- One in a fourth state that is different from California, Texas, and New York

For each university, extract the following fields:

Identification
- state: The U.S. state of the university (two-letter abbreviation preferred, or full name)
- university_name: Full institutional name
- program_name: The program name (e.g., "PhD in Computer Science" or "Computer Science & Engineering")
- program_urls: An array of official website URLs that verify the existence of the PhD program (department or graduate division pages are acceptable)
- public_location_urls: An array of official website URLs that support that the institution is a public university and located in the specified state

Required attributes (each MUST include official website URL(s) from the same university; use arrays even if only one URL)
- minimum_gpa: { "value": "...", "urls": ["...", ...] } – The minimum GPA for PhD admission; if there is no formal minimum, set value to "no minimum"
- gre_term_label: "Fall 2025" or "Fall 2026" (choose one term explicitly mentioned in the answer)
- gre_required: "required" or "not required" (or "optional" if stated) for the chosen gre_term_label
- gre_urls: array of official URLs that support the GRE policy for the chosen term
- application_deadline: { "value": "...", "urls": ["...", ...] } – Deadline for PhD applications (use the primary/final or priority deadline as given)
- english_minimum: { "value": "...", "urls": ["...", ...] } – Minimum TOEFL or IELTS requirement for international students (can include ranges or "TOEFL 100 / IELTS 7.5", etc.)
- typical_duration: { "value": "...", "urls": ["...", ...] } – Typical expected duration in years (e.g., "4-6")
- research_areas: { "areas": ["...", "...", "...", ...], "urls": ["...", ...] } – List at least three research areas/specializations
- ra_ta_availability: { "value": "...", "urls": ["...", ...] } – A statement confirming RA and/or TA availability to PhD students

Strict requirements:
- All URLs must be official university websites (e.g., domains ending in .edu or clearly official subdomains of the university). Do not use aggregators, third-party blogs, rankings, or Wikipedia.
- Extract only what is explicitly present in the answer text and its cited URLs. Do not invent.
- Use arrays for all URL fields (even if only one URL).
- If an item is not mentioned in the answer, return null for the entire university object or leave values null/empty as appropriate.

Return a JSON object with the following top-level keys:
{
  "california": UniversityItem or null,
  "texas": UniversityItem or null,
  "new_york": UniversityItem or null,
  "fourth": UniversityItem or null
}
"""


# --------------------------------------------------------------------------- #
# Helper instructions for verification                                        #
# --------------------------------------------------------------------------- #
OFFICIAL_URL_INSTRUCTION = (
    "Only mark as supported if the provided URL(s) are from the university's official website "
    "(typically a .edu domain or an official subdomain owned by the university). "
    "Third-party sites (e.g., rankings, application platforms, or Wikipedia) do not count."
)

PROGRAM_EXISTS_INSTRUCTION = (
    "Treat 'Computer Science' and 'Computer Science & Engineering' (or similar official naming) as equivalent "
    "for program existence. The page should clearly indicate a PhD (doctoral) program in this field."
)

GPA_INSTRUCTION = (
    "Verify the minimum GPA requirement as stated. If the page says 'no minimum', accept 'no minimum' as valid."
)

GRE_INSTRUCTION = (
    "Verify the GRE policy specifically for the stated term (Fall 2025 or Fall 2026). "
    "Interpret 'not required' to include 'optional' or 'waived' if stated for that term."
)

DEADLINE_INSTRUCTION = (
    "Accept common variations like 'priority' vs 'final' deadlines if the provided value matches one explicitly on the page."
)

ENGLISH_INSTRUCTION = (
    "Verify the minimum TOEFL or IELTS requirement for international students. Accept ranges or combined formats "
    "such as 'TOEFL 100 / IELTS 7.5'."
)

DURATION_INSTRUCTION = (
    "Verify the typical or expected time to degree (e.g., '4-6 years'). Reasonable ranges or wording are acceptable."
)

AREAS_INSTRUCTION = (
    "Verify that at least the listed research areas are present on the official page. Allow minor naming variations."
)

FUNDING_INSTRUCTION = (
    "Verify that PhD students have access to RA and/or TA positions. Accept if the page clearly states RA/TA support."
)


def combine_instruction(*parts: str) -> str:
    return " ".join([p for p in parts if p and p.strip()])


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and s.strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls and isinstance(urls, list) and len(urls) > 0)


def _unique_nonempty_urls(*url_lists: List[str]) -> List[str]:
    gathered = []
    for lst in url_lists:
        if not lst:
            continue
        for u in lst:
            if isinstance(u, str) and _nonempty(u):
                gathered.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    result = []
    for u in gathered:
        if u not in seen:
            result.append(u)
            seen.add(u)
    return result


def check_state_coverage(extraction: FourUniversitiesExtraction) -> Tuple[bool, Dict[str, Any]]:
    states_raw = {
        "california": extraction.california.state if extraction.california else None,
        "texas": extraction.texas.state if extraction.texas else None,
        "new_york": extraction.new_york.state if extraction.new_york else None,
        "fourth": extraction.fourth.state if extraction.fourth else None,
    }
    normalized = {k: normalize_state_to_abbr(v) for k, v in states_raw.items()}
    details = {
        "raw_states": states_raw,
        "normalized_states": normalized
    }

    # Must include CA, TX, NY explicitly and a 4th distinct state not in {CA,TX,NY}
    ca_ok = normalized.get("california") == "CA"
    tx_ok = normalized.get("texas") == "TX"
    ny_ok = normalized.get("new_york") == "NY"

    fourth_abbr = normalized.get("fourth")
    fourth_ok = fourth_abbr is not None and fourth_abbr not in {"CA", "TX", "NY"}

    # All four must be distinct
    unique_count = len({abbr for abbr in normalized.values() if abbr is not None})
    distinct_ok = unique_count == 4

    result = ca_ok and tx_ok and ny_ok and fourth_ok and distinct_ok
    details.update({
        "ca_ok": ca_ok,
        "tx_ok": tx_ok,
        "ny_ok": ny_ok,
        "fourth_ok": fourth_ok,
        "distinct_ok": distinct_ok
    })
    return result, details


# --------------------------------------------------------------------------- #
# Node builders                                                               #
# --------------------------------------------------------------------------- #
async def build_identification_checks(
    evaluator: Evaluator,
    parent_node,
    prefix: str,
    uni: UniversityItem,
    state_label_for_claim: Optional[str]
) -> None:
    """
    Build identification subtree:
    - A critical custom node checking that at least one official URL is provided for identification.
    - Two critical leaves (parallel):
        * public university in state
        * PhD CS/CSE program exists
    """
    ident_node = evaluator.add_parallel(
        id=f"{prefix}_identification_with_url",
        desc=f"Identify a U.S. public university in {state_label_for_claim or 'the specified state'} with an active CS/CS&E PhD, supported by an official URL.",
        parent=parent_node,
        critical=True
    )

    # Simple existence of at least one URL for identification gating
    ident_urls = _unique_nonempty_urls(uni.public_location_urls, uni.program_urls)
    ident_sources_present = _has_urls(ident_urls) and _nonempty(uni.university_name) and _nonempty(uni.state)
    evaluator.add_custom_node(
        result=ident_sources_present,
        id=f"{prefix}_identification_sources_present",
        desc=f"{prefix.upper()}: At least one official-website URL and basic identification fields (name, state) are provided.",
        parent=ident_node,
        critical=True
    )

    # Leaf: Public and in state
    public_state_leaf = evaluator.add_leaf(
        id=f"{prefix}_public_in_state",
        desc="University is a public institution located in the specified state.",
        parent=ident_node,
        critical=True
    )
    state_abbr = normalize_state_to_abbr(uni.state)
    state_name = STATE_NAME_BY_ABBR.get(state_abbr, state_label_for_claim or (uni.state or "")).strip()
    claim_public_state = f"The university '{uni.university_name or ''}' is a public university located in {state_name}, United States."
    await evaluator.verify(
        claim=claim_public_state,
        node=public_state_leaf,
        sources=ident_urls,
        additional_instruction=combine_instruction(OFFICIAL_URL_INSTRUCTION)
    )

    # Leaf: Program exists
    program_exists_leaf = evaluator.add_leaf(
        id=f"{prefix}_phd_program_exists",
        desc="University offers an active PhD program in Computer Science or Computer Science & Engineering.",
        parent=ident_node,
        critical=True
    )
    claim_program_exists = (
        f"The university '{uni.university_name or ''}' offers an active doctoral (PhD) program in "
        f"Computer Science or Computer Science & Engineering."
    )
    await evaluator.verify(
        claim=claim_program_exists,
        node=program_exists_leaf,
        sources=_unique_nonempty_urls(uni.program_urls),
        additional_instruction=combine_instruction(OFFICIAL_URL_INSTRUCTION, PROGRAM_EXISTS_INSTRUCTION)
    )


async def build_required_attributes_checks(
    evaluator: Evaluator,
    parent_node,
    prefix: str,
    uni: UniversityItem
) -> None:
    """
    Build the parallel node of all required attributes, with source presence gating per attribute.
    Each attribute verification leaf is critical and uses official URLs to verify.
    """
    req_node = evaluator.add_parallel(
        id=f"{prefix}_required_attributes_with_urls",
        desc=f"Provide all required attributes for the {prefix.split('_')[0].upper()} program, each supported by an official-website URL.",
        parent=parent_node,
        critical=True
    )

    # 1) Minimum GPA
    gpa_urls = _unique_nonempty_urls(uni.minimum_gpa.urls)
    evaluator.add_custom_node(
        result=_has_urls(gpa_urls) and _nonempty(uni.minimum_gpa.value),
        id=f"{prefix}_minimum_gpa_source_present",
        desc="Source URL(s) provided and minimum GPA value is present.",
        parent=req_node,
        critical=True
    )
    gpa_leaf = evaluator.add_leaf(
        id=f"{prefix}_minimum_gpa_with_url",
        desc="Provide the minimum GPA requirement for PhD admission with an official-website URL citation.",
        parent=req_node,
        critical=True
    )
    gpa_claim = (
        f"The minimum GPA requirement for PhD admission in Computer Science at '{uni.university_name or ''}' "
        f"is '{uni.minimum_gpa.value or ''}' (on the page)."
    )
    await evaluator.verify(
        claim=gpa_claim,
        node=gpa_leaf,
        sources=gpa_urls,
        additional_instruction=combine_instruction(OFFICIAL_URL_INSTRUCTION, GPA_INSTRUCTION)
    )

    # 2) GRE policy (Fall 2025 or Fall 2026)
    gre_urls = _unique_nonempty_urls(uni.gre_urls)
    gre_value_present = _nonempty(uni.gre_term_label) and _nonempty(uni.gre_required)
    evaluator.add_custom_node(
        result=_has_urls(gre_urls) and gre_value_present,
        id=f"{prefix}_gre_policy_source_present",
        desc="Source URL(s) provided and GRE policy term/value are present.",
        parent=req_node,
        critical=True
    )
    gre_leaf = evaluator.add_leaf(
        id=f"{prefix}_gre_policy_f25_f26_with_url",
        desc="State whether GRE scores are required for Fall 2025 or Fall 2026 admission with an official-website URL citation.",
        parent=req_node,
        critical=True
    )
    gre_claim = (
        f"For {uni.gre_term_label or ''}, GRE scores are '{uni.gre_required or ''}' for PhD admission in Computer Science at "
        f"'{uni.university_name or ''}'."
    )
    await evaluator.verify(
        claim=gre_claim,
        node=gre_leaf,
        sources=gre_urls,
        additional_instruction=combine_instruction(OFFICIAL_URL_INSTRUCTION, GRE_INSTRUCTION)
    )

    # 3) Application deadline
    deadline_urls = _unique_nonempty_urls(uni.application_deadline.urls)
    evaluator.add_custom_node(
        result=_has_urls(deadline_urls) and _nonempty(uni.application_deadline.value),
        id=f"{prefix}_application_deadline_source_present",
        desc="Source URL(s) provided and application deadline value is present.",
        parent=req_node,
        critical=True
    )
    deadline_leaf = evaluator.add_leaf(
        id=f"{prefix}_application_deadline_with_url",
        desc="Provide the PhD application deadline with an official-website URL citation.",
        parent=req_node,
        critical=True
    )
    deadline_claim = (
        f"The application deadline for the CS PhD at '{uni.university_name or ''}' is '{uni.application_deadline.value or ''}'."
    )
    await evaluator.verify(
        claim=deadline_claim,
        node=deadline_leaf,
        sources=deadline_urls,
        additional_instruction=combine_instruction(OFFICIAL_URL_INSTRUCTION, DEADLINE_INSTRUCTION)
    )

    # 4) English minimum (TOEFL/IELTS)
    eng_urls = _unique_nonempty_urls(uni.english_minimum.urls)
    evaluator.add_custom_node(
        result=_has_urls(eng_urls) and _nonempty(uni.english_minimum.value),
        id=f"{prefix}_english_minimum_source_present",
        desc="Source URL(s) provided and minimum TOEFL/IELTS value is present.",
        parent=req_node,
        critical=True
    )
    english_leaf = evaluator.add_leaf(
        id=f"{prefix}_toefl_or_ielts_min_with_url",
        desc="Provide the minimum TOEFL or IELTS score requirement for international students with an official-website URL citation.",
        parent=req_node,
        critical=True
    )
    english_claim = (
        f"The minimum English proficiency requirement (TOEFL or IELTS) for the CS PhD at '{uni.university_name or ''}' "
        f"is '{uni.english_minimum.value or ''}'."
    )
    await evaluator.verify(
        claim=english_claim,
        node=english_leaf,
        sources=eng_urls,
        additional_instruction=combine_instruction(OFFICIAL_URL_INSTRUCTION, ENGLISH_INSTRUCTION)
    )

    # 5) Typical duration
    duration_urls = _unique_nonempty_urls(uni.typical_duration.urls)
    evaluator.add_custom_node(
        result=_has_urls(duration_urls) and _nonempty(uni.typical_duration.value),
        id=f"{prefix}_typical_duration_source_present",
        desc="Source URL(s) provided and typical duration value is present.",
        parent=req_node,
        critical=True
    )
    duration_leaf = evaluator.add_leaf(
        id=f"{prefix}_typical_duration_with_url",
        desc="Provide the typical/expected PhD duration (in years) with an official-website URL citation.",
        parent=req_node,
        critical=True
    )
    duration_claim = (
        f"The typical time to complete the CS PhD at '{uni.university_name or ''}' is '{uni.typical_duration.value or ''}' years."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=duration_urls,
        additional_instruction=combine_instruction(OFFICIAL_URL_INSTRUCTION, DURATION_INSTRUCTION)
    )

    # 6) Research areas (3+)
    areas_urls = _unique_nonempty_urls(uni.research_areas.urls)
    areas_list = uni.research_areas.areas or []
    evaluator.add_custom_node(
        result=_has_urls(areas_urls) and len(areas_list) >= 3,
        id=f"{prefix}_research_areas_source_present",
        desc="Source URL(s) provided and at least three research areas are present.",
        parent=req_node,
        critical=True
    )
    areas_leaf = evaluator.add_leaf(
        id=f"{prefix}_research_areas_3plus_with_url",
        desc="List at least three research areas/specializations with an official-website URL citation.",
        parent=req_node,
        critical=True
    )
    # Build claim with at least three areas
    listed = areas_list[:3]
    areas_str = ", ".join(listed)
    areas_claim = (
        f"The CS PhD program at '{uni.university_name or ''}' lists at least these research areas: {areas_str}."
    )
    await evaluator.verify(
        claim=areas_claim,
        node=areas_leaf,
        sources=areas_urls,
        additional_instruction=combine_instruction(OFFICIAL_URL_INSTRUCTION, AREAS_INSTRUCTION)
    )

    # 7) RA/TA availability
    funding_urls = _unique_nonempty_urls(uni.ra_ta_availability.urls)
    evaluator.add_custom_node(
        result=_has_urls(funding_urls) and _nonempty(uni.ra_ta_availability.value),
        id=f"{prefix}_ra_ta_source_present",
        desc="Source URL(s) provided and RA/TA availability value is present.",
        parent=req_node,
        critical=True
    )
    funding_leaf = evaluator.add_leaf(
        id=f"{prefix}_ra_ta_availability_with_url",
        desc="Confirm RA and/or TA availability for PhD students with an official-website URL citation.",
        parent=req_node,
        critical=True
    )
    funding_claim = (
        f"PhD students in the CS program at '{uni.university_name or ''}' have access to RA and/or TA positions."
    )
    await evaluator.verify(
        claim=funding_claim,
        node=funding_leaf,
        sources=funding_urls,
        additional_instruction=combine_instruction(OFFICIAL_URL_INSTRUCTION, FUNDING_INSTRUCTION)
    )


async def verify_university_item(
    evaluator: Evaluator,
    root,
    node_id: str,
    node_desc: str,
    prefix: str,
    uni: Optional[UniversityItem],
    expected_state_label: Optional[str] = None
) -> None:
    """
    Build the verification subtree for a given state/university item.
    """
    uni_node = evaluator.add_sequential(
        id=node_id,
        desc=node_desc,
        parent=root,
        critical=False
    )

    # If no university extracted, add minimal nodes that will naturally fail/skipped
    uni_obj = uni or UniversityItem()

    # Identification checks (parallel, critical)
    await build_identification_checks(
        evaluator=evaluator,
        parent_node=uni_node,
        prefix=prefix,
        uni=uni_obj,
        state_label_for_claim=expected_state_label
    )

    # Required attributes checks (parallel, critical)
    await build_required_attributes_checks(
        evaluator=evaluator,
        parent_node=uni_node,
        prefix=prefix,
        uni=uni_obj
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'four CS PhD universities across four distinct states' task.
    """
    # Initialize evaluator (root should be non-critical to allow partial credit)
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=FourUniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Add overall state coverage and uniqueness check (critical leaf at root level)
    ok_state_cov, cov_details = check_state_coverage(extraction)
    evaluator.add_custom_info(cov_details, info_type="coverage_details", info_name="state_coverage_normalization")
    evaluator.add_custom_node(
        result=ok_state_cov,
        id="state_coverage_and_uniqueness",
        desc="Selected universities collectively cover CA, TX, NY, and exactly one additional distinct state (not CA/TX/NY).",
        parent=root,
        critical=True
    )

    # Build verification subtrees for each required state/university
    await verify_university_item(
        evaluator=evaluator,
        root=root,
        node_id="california_university",
        node_desc="California university item",
        prefix="ca",
        uni=extraction.california,
        expected_state_label="California"
    )

    await verify_university_item(
        evaluator=evaluator,
        root=root,
        node_id="texas_university",
        node_desc="Texas university item",
        prefix="tx",
        uni=extraction.texas,
        expected_state_label="Texas"
    )

    await verify_university_item(
        evaluator=evaluator,
        root=root,
        node_id="newyork_university",
        node_desc="New York university item",
        prefix="ny",
        uni=extraction.new_york,
        expected_state_label="New York"
    )

    # Fourth (distinct) state university
    # Use the extracted state label directly in identification claim
    fourth_state_label = None
    if extraction.fourth and extraction.fourth.state:
        abbr = normalize_state_to_abbr(extraction.fourth.state)
        fourth_state_label = STATE_NAME_BY_ABBR.get(abbr, extraction.fourth.state)

    await verify_university_item(
        evaluator=evaluator,
        root=root,
        node_id="fourth_state_university",
        node_desc="Fourth (non-CA/TX/NY) state university item",
        prefix="fourth",
        uni=extraction.fourth,
        expected_state_label=fourth_state_label
    )

    # Return summary
    return evaluator.get_summary()