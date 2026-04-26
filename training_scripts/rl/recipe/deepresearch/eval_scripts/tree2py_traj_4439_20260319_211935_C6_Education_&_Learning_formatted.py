import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "ihsa_2a3a_three_schools_2025_26"
TASK_DESCRIPTION = """
I am researching Illinois high schools with strong athletic programs for a comparative study. Identify three IHSA member schools that meet all of the following criteria:

1. The school must be classified in Class 2A or Class 3A for boys basketball during the 2025-26 season (enrollment between 300.01 and 1600 students).
2. The school must participate in both boys basketball and football programs during the 2025-26 IHSA season.
3. For each school, provide:
   - The school's official name as listed in IHSA records
   - The school's enrollment number from the IHSA 2025-26 cycle
   - The school's boys basketball classification (2A or 3A)
   - The school's physical address
   - The school's main phone number
   - A link to either the school's official website or its page on the IHSA school directory
   - Evidence that the school participated in at least one IHSA state series (playoffs, regionals, sectionals, or state tournament) for either boys basketball or football in the 2024-25 or 2025-26 season

All information must be verifiable through official IHSA sources or the schools' official websites.
"""

ALLOWED_BB_CLASSES = {"2A", "3A"}
ENROLLMENT_MIN_EXCLUSIVE = 300.0
ENROLLMENT_MAX_INCLUSIVE = 1600.0
CURRENT_SEASON = "2025-26"
PREV_SEASON = "2024-25"


# -----------------------------------------------------------------------------
# Extraction data models
# -----------------------------------------------------------------------------
class SchoolSources(BaseModel):
    ihsa_profile_url: Optional[str] = None         # IHSA directory entry URL for this school
    enrollment_urls: List[str] = Field(default_factory=list)      # IHSA 2025-26 enrollment/classification pages
    classification_urls: List[str] = Field(default_factory=list)  # IHSA Boys Basketball classification 2025-26
    sports_urls: List[str] = Field(default_factory=list)          # IHSA pages listing sports for 2025-26
    contact_urls: List[str] = Field(default_factory=list)         # Official school or IHSA pages listing address/phone
    state_series_urls: List[str] = Field(default_factory=list)    # IHSA playoff/state series pages 2024-25 or 2025-26
    website_url: Optional[str] = None             # Official school main website (if provided)
    directory_url: Optional[str] = None           # IHSA directory URL (if provided separately)


class SchoolItem(BaseModel):
    name: Optional[str] = None                                      # Official name as in IHSA records
    enrollment: Optional[str] = None                                 # Enrollment number text for 2025-26 cycle
    boys_basketball_classification: Optional[str] = None             # "2A" or "3A"
    address: Optional[str] = None                                    # Physical address
    phone: Optional[str] = None                                      # Main phone
    website_or_directory_url: Optional[str] = None                   # One link: official website or IHSA directory page
    sources: SchoolSources = Field(default_factory=SchoolSources)    # Source URLs used for verification


class SchoolsExtraction(BaseModel):
    schools: List[SchoolItem] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_schools() -> str:
    return f"""
You will extract up to three IHSA member high schools from the answer that are intended to meet all of the following criteria:
– Boys Basketball classification is Class 2A or Class 3A for the {CURRENT_SEASON} season
– School participates in BOTH Boys Basketball and Football in the {CURRENT_SEASON} IHSA season
– Include official IHSA-listed name, {CURRENT_SEASON} enrollment number, Boys Basketball classification (2A or 3A), address, phone,
  one link to either the official school website or the IHSA directory page, and evidence of IHSA state series participation
  in {PREV_SEASON} or {CURRENT_SEASON} for either Boys Basketball or Football.

IMPORTANT:
– Extract ONLY what appears explicitly in the provided answer text.
– For URLs, extract only actual URLs explicitly present in the answer (including markdown links).
– Do NOT invent or infer missing data. If something is missing, return null for that field.
– Prefer IHSA (ihsa.org) URLs where available. If the answer cites a school's official website for address/phone, include those links.
– Return at most three (3) schools in the 'schools' array. If the answer lists more than three, include only the first three. If fewer are listed, include as many as are present.

For each school, return a JSON object with the following fields:
- name: Official school name as listed in IHSA records (string or null)
- enrollment: The school's {CURRENT_SEASON} enrollment number as text exactly as written in the answer (string or null)
- boys_basketball_classification: The Boys Basketball classification for {CURRENT_SEASON} (e.g., "2A" or "3A") (string or null)
- address: The physical address exactly as written (string or null)
- phone: The main phone number exactly as written (string or null)
- website_or_directory_url: A single URL that is either the official school website home page (or contact page)
  OR the IHSA school directory page (string or null)

- sources: Object with the following URL fields (use empty lists if none explicitly present in the answer):
  - ihsa_profile_url: IHSA directory profile URL (string or null)
  - enrollment_urls: list of URLs supporting the {CURRENT_SEASON} enrollment figure
  - classification_urls: list of URLs supporting the Boys Basketball {CURRENT_SEASON} class (2A or 3A)
  - sports_urls: list of URLs showing the school participates in Boys Basketball and Football in {CURRENT_SEASON}
  - contact_urls: list of URLs supporting address and phone (official school site or IHSA directory)
  - state_series_urls: list of URLs evidencing participation in IHSA state series (playoffs/regionals/sectionals/state) in {PREV_SEASON} or {CURRENT_SEASON}
  - website_url: official school website home/contact (string or null)
  - directory_url: IHSA directory page URL if provided (string or null)

Return JSON with this structure:
{{
  "schools": [
     {{"name": ..., "enrollment": ..., "boys_basketball_classification": ..., "address": ..., "phone": ...,
       "website_or_directory_url": ..., "sources": {{ "ihsa_profile_url": ..., "enrollment_urls": [...],
       "classification_urls": [...], "sports_urls": [...], "contact_urls": [...], "state_series_urls": [...],
       "website_url": ..., "directory_url": ... }} }},
     ...
  ]
}}
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    try:
        p = urlparse(url)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def _is_ihsa_url(url: Optional[str]) -> bool:
    if not _is_valid_url(url):
        return False
    netloc = urlparse(url).netloc.lower()
    return "ihsa.org" in netloc


def _dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not _is_valid_url(u):
            continue
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n+1}"


def _safe(s: Optional[str]) -> str:
    return s or ""


def _any_ihsa(urls: List[str]) -> bool:
    return any(_is_ihsa_url(u) for u in urls)


# -----------------------------------------------------------------------------
# Verification logic per school
# -----------------------------------------------------------------------------
async def verify_one_school(evaluator: Evaluator, parent_node, school: SchoolItem, index: int) -> None:
    school_title = f"{_ordinal(index)} school meeting all requirements"

    school_node = evaluator.add_parallel(
        id=f"school_{index+1}",
        desc=school_title,
        parent=parent_node,
        critical=False,  # each school contributes to partial credit
    )

    # Convenience source buckets
    ihsa_profile = school.sources.ihsa_profile_url or school.sources.directory_url
    name_ref_urls = _dedup_urls([ihsa_profile])
    enrollment_ref_urls = _dedup_urls((school.sources.enrollment_urls or []) + ([] if _any_ihsa(school.sources.enrollment_urls) else school.sources.classification_urls))
    class_ref_urls = _dedup_urls((school.sources.classification_urls or []) + ([] if _any_ihsa(school.sources.classification_urls) else school.sources.enrollment_urls))
    sports_ref_urls = _dedup_urls((school.sources.sports_urls or []) + ([ihsa_profile] if _is_valid_url(ihsa_profile) else []))
    contact_ref_urls = _dedup_urls((school.sources.contact_urls or []) + ([school.website_or_directory_url] if _is_valid_url(school.website_or_directory_url) else []) + ([ihsa_profile] if _is_valid_url(ihsa_profile) else []))
    playoff_ref_urls = _dedup_urls(school.sources.state_series_urls or [])

    # 1) Identification (name + membership)
    ident_node = evaluator.add_parallel(
        id=f"school_{index+1}_identification",
        desc="School identity and basic information",
        parent=school_node,
        critical=True,
    )

    # 1.a) Name reference URL existence (IHSA)
    evaluator.add_custom_node(
        result=len(name_ref_urls) > 0 and _any_ihsa(name_ref_urls),
        id=f"school_{index+1}_name_ref_url",
        desc="Provide reference URL from IHSA website confirming school name and membership",
        parent=ident_node,
        critical=True,
    )

    # 1.b) School name matches IHSA page
    name_leaf = evaluator.add_leaf(
        id=f"school_{index+1}_name",
        desc="Provide the official school name as listed in IHSA records",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The IHSA school directory page shows the official school name is '{_safe(school.name)}' (allowing minor punctuation or abbreviation differences that clearly refer to the same school).",
        node=name_leaf,
        sources=name_ref_urls,
        additional_instruction="Focus on the official school name label on the IHSA directory page. Allow minor variations like 'HS' vs 'High School'.",
    )

    # 1.c) IHSA membership active (assessed via directory page presence)
    membership_leaf = evaluator.add_leaf(
        id=f"school_{index+1}_ihsa_membership",
        desc="Verify school is an active IHSA member for 2025-26",
        parent=ident_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided IHSA directory page indicates the school is an IHSA member for the {CURRENT_SEASON} season (a current directory listing counts as active membership unless it explicitly says otherwise).",
        node=membership_leaf,
        sources=name_ref_urls,
        additional_instruction="Treat a valid IHSA directory listing as proof of active membership unless clearly marked closed/inactive.",
    )

    # 2) Enrollment + Classification
    enroll_node = evaluator.add_parallel(
        id=f"school_{index+1}_enrollment",
        desc="School enrollment and classification verification",
        parent=school_node,
        critical=True,
    )

    # 2.a) Enrollment reference URL existence
    evaluator.add_custom_node(
        result=len(enrollment_ref_urls) > 0 and _any_ihsa(enrollment_ref_urls),
        id=f"school_{index+1}_enrollment_ref_url",
        desc=f"Provide reference URL from IHSA enrollment/classification pages for {CURRENT_SEASON}",
        parent=enroll_node,
        critical=True,
    )

    # 2.b) Enrollment number itself (from IHSA 2025-26)
    enr_leaf = evaluator.add_leaf(
        id=f"school_{index+1}_enrollment_number",
        desc=f"Provide the school's enrollment number from IHSA {CURRENT_SEASON} cycle data",
        parent=enroll_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"For the IHSA {CURRENT_SEASON} cycle, this school's enrollment is '{_safe(school.enrollment)}'.",
        node=enr_leaf,
        sources=enrollment_ref_urls,
        additional_instruction=f"Use IHSA {CURRENT_SEASON} enrollment/classification materials as primary evidence. Match the numeric value even if commas or formatting differ.",
    )

    # 2.c) Enrollment in required range (300.01 to 1600)
    enr_range_leaf = evaluator.add_leaf(
        id=f"school_{index+1}_enrollment_range",
        desc="Verify enrollment is between 300.01 and 1600 students",
        parent=enroll_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"Based on the verified IHSA {CURRENT_SEASON} enrollment '{_safe(school.enrollment)}', the value is strictly greater than 300 and less than or equal to 1600.",
        node=enr_range_leaf,
        sources=enrollment_ref_urls,
        additional_instruction="Interpret the enrollment number numerically. If multiple numbers appear, use the one explicitly labeled 'Enrollment' for the school in the IHSA {CURRENT_SEASON} context.",
    )

    # 2.d) Boys Basketball classification is 2A or 3A for 2025-26
    class_leaf = evaluator.add_leaf(
        id=f"school_{index+1}_bb_class",
        desc="Confirm school is classified as Class 2A or 3A for boys basketball",
        parent=enroll_node,
        critical=True,
    )
    bb_class = _safe(school.boys_basketball_classification)
    await evaluator.verify(
        claim=f"For Boys Basketball in {CURRENT_SEASON}, the school is in Class '{bb_class}', and that class is either 2A or 3A.",
        node=class_leaf,
        sources=class_ref_urls if len(class_ref_urls) > 0 else enrollment_ref_urls,
        additional_instruction=f"Verify the specific Boys Basketball class for {CURRENT_SEASON} from IHSA pages or official materials. Accept minor formatting differences (e.g., 'Class 2A' vs '2A').",
    )

    # 3) Athletic programs (Boys Basketball + Football)
    sports_node = evaluator.add_parallel(
        id=f"school_{index+1}_athletics",
        desc="Verification of required athletic programs",
        parent=school_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(sports_ref_urls) > 0 and any(_is_ihsa_url(u) or _is_valid_url(u) for u in sports_ref_urls),
        id=f"school_{index+1}_sports_ref_url",
        desc="Provide reference URL showing sports participation (IHSA preferred, school site acceptable)",
        parent=sports_node,
        critical=True,
    )

    bball_leaf = evaluator.add_leaf(
        id=f"school_{index+1}_boys_basketball_participation",
        desc=f"Confirm school participates in boys basketball for {CURRENT_SEASON} season",
        parent=sports_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The school participates in Boys Basketball during the IHSA {CURRENT_SEASON} season.",
        node=bball_leaf,
        sources=sports_ref_urls,
        additional_instruction=f"Use IHSA directory or sport listing pages; if the year is not explicitly shown but the listing indicates Boys Basketball for the current cycle, accept.",
    )

    football_leaf = evaluator.add_leaf(
        id=f"school_{index+1}_football_participation",
        desc=f"Confirm school participates in football for {CURRENT_SEASON} season",
        parent=sports_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The school participates in Football during the IHSA {CURRENT_SEASON} season.",
        node=football_leaf,
        sources=sports_ref_urls,
        additional_instruction=f"Use IHSA directory or sport listing pages; if the year is not explicitly shown but the listing indicates Football for the current cycle, accept.",
    )

    # 4) Contact information (address + phone + website/directory link)
    contact_node = evaluator.add_parallel(
        id=f"school_{index+1}_contact",
        desc="School contact details",
        parent=school_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(contact_ref_urls) > 0,
        id=f"school_{index+1}_contact_ref_url",
        desc="Provide reference URL confirming contact information",
        parent=contact_node,
        critical=True,
    )

    addr_leaf = evaluator.add_leaf(
        id=f"school_{index+1}_address",
        desc="Provide complete physical address of the school",
        parent=contact_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The school's physical address is '{_safe(school.address)}'.",
        node=addr_leaf,
        sources=contact_ref_urls,
        additional_instruction="Allow minor formatting differences (e.g., abbreviations like Rd/Road, punctuation, line breaks). Focus on the same address content.",
    )

    phone_leaf = evaluator.add_leaf(
        id=f"school_{index+1}_phone",
        desc="Provide main phone number for the school",
        parent=contact_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The school's main phone number is '{_safe(school.phone)}'.",
        node=phone_leaf,
        sources=contact_ref_urls,
        additional_instruction="Allow minor formatting differences (e.g., '(xxx) xxx-xxxx' vs 'xxx-xxx-xxxx').",
    )

    # Presence and validity of a website or IHSA directory link
    site_ok = _is_valid_url(school.website_or_directory_url) and (
        _is_ihsa_url(school.website_or_directory_url) or True  # accept official school sites as well
    )
    evaluator.add_custom_node(
        result=site_ok,
        id=f"school_{index+1}_website_or_directory",
        desc="Provide link to school's official website or IHSA school directory page",
        parent=contact_node,
        critical=True,
    )

    # 5) State series participation evidence
    series_node = evaluator.add_parallel(
        id=f"school_{index+1}_state_series",
        desc="Verification of state series participation",
        parent=school_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=len(playoff_ref_urls) > 0 and _any_ihsa(playoff_ref_urls),
        id=f"school_{index+1}_playoff_ref_url",
        desc="Provide reference URL from IHSA showing state series participation",
        parent=series_node,
        critical=True,
    )

    playoff_leaf = evaluator.add_leaf(
        id=f"school_{index+1}_playoff_participation",
        desc=f"Confirm school participated in IHSA state series (playoffs or state tournament) for boys basketball or football in {PREV_SEASON} or {CURRENT_SEASON}",
        parent=series_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The school participated in at least one IHSA state series event in Boys Basketball or Football during {PREV_SEASON} or {CURRENT_SEASON} (playoffs, regionals, sectionals, super-sectionals, or state).",
        node=playoff_leaf,
        sources=playoff_ref_urls,
        additional_instruction="Accept IHSA brackets, results, or state series pages that explicitly show the school's participation in those seasons.",
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    """
    Evaluation entry point for the IHSA schools (2A/3A, 2025-26) task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # schools are evaluated independently for partial credit
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

    # Extract structured school info from the answer
    extracted: SchoolsExtraction = await evaluator.extract(
        prompt=prompt_extract_schools(),
        template_class=SchoolsExtraction,
        extraction_name="schools_extraction",
    )

    # Ensure exactly three schools in evaluation by padding/truncating
    schools: List[SchoolItem] = list(extracted.schools[:3])
    while len(schools) < 3:
        schools.append(SchoolItem())

    # Root node (non-critical aggregation across 3 schools)
    task_node = evaluator.add_parallel(
        id="task_completion",
        desc="Successfully identify three IHSA member schools that meet all specified criteria",
        parent=root,
        critical=False,
    )

    # Add minimal task context info
    evaluator.add_custom_info(
        info={
            "required_boys_basketball_classes": sorted(list(ALLOWED_BB_CLASSES)),
            "required_enrollment_range": f"({ENROLLMENT_MIN_EXCLUSIVE}, {ENROLLMENT_MAX_INCLUSIVE}]",
            "seasons": {"current": CURRENT_SEASON, "previous": PREV_SEASON},
        },
        info_type="constraints",
        info_name="evaluation_constraints",
    )

    # Verify each of the first three schools
    for idx, school in enumerate(schools):
        await verify_one_school(evaluator, task_node, school, idx)

    return evaluator.get_summary()