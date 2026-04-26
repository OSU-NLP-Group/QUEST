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
TASK_ID = "university_va_nc_tn"
TASK_DESCRIPTION = (
    "Identify one university located in Virginia, North Carolina, or Tennessee that meets all of the following criteria: "
    "(1) The university must have a total undergraduate enrollment between 7,000 and 25,000 students (as of fall 2024 or fall 2025); "
    "(2) The main campus must be at least 90 acres in size; "
    "(3) The campus setting must be classified as either urban or suburban (not rural); "
    "(4) The university must be accredited by the Southern Association of Colleges and Schools Commission on Colleges (SACSCOC). "
    "For the university you identify, provide the following information: the official name of the university, the complete physical address of the main campus, "
    "the total undergraduate enrollment (fall 2024 or fall 2025), the campus size in acres, the campus setting classification (urban or suburban), "
    "a direct link to the university's official website, and a direct link to a page that confirms the university's SACSCOC accreditation."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityExtraction(BaseModel):
    university_name: Optional[str] = None
    main_campus_address: Optional[str] = None
    # Optional explicit state if the answer provides it separately; can be null
    stated_state: Optional[str] = None
    undergrad_enrollment: Optional[str] = None
    enrollment_term: Optional[str] = None  # e.g., "Fall 2024" or "Fall 2025"
    campus_size_acres: Optional[str] = None
    campus_setting: Optional[str] = None  # e.g., "urban" or "suburban"
    official_website_url: Optional[str] = None
    sacscoc_accreditation_url: Optional[str] = None
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university() -> str:
    return """
Extract exactly one university (the first one if multiple are mentioned) from the answer and return the following fields:

- university_name: The official name of the university exactly as stated in the answer.
- main_campus_address: The complete physical address of the main campus as stated in the answer (include street, city, state abbreviation or full state name, and ZIP if provided).
- stated_state: If the answer explicitly names the state (e.g., "Virginia", "North Carolina", "Tennessee", or "VA", "NC", "TN"), extract it. Otherwise, return null.
- undergrad_enrollment: The total undergraduate enrollment as stated in the answer (keep it as a string, including commas, approximations, or ranges if provided).
- enrollment_term: The referenced academic term for the enrollment figure (e.g., "Fall 2024" or "Fall 2025"). If not clearly stated, return null.
- campus_size_acres: The campus size of the main campus as stated in the answer (keep it as a string; include units if present, e.g., "120 acres").
- campus_setting: The campus setting classification as stated in the answer (e.g., "urban", "suburban", or "rural"). Return exactly what the answer says.
- official_website_url: A direct link to the university's official website (typically a .edu domain). If missing, return null.
- sacscoc_accreditation_url: A direct link to a page that confirms the university's SACSCOC accreditation (this can be on sacscoc.org or an official university accreditation page). If missing, return null.
- all_urls: Extract all URLs that appear anywhere in the answer (including the official website URL and the accreditation URL). Deduplicate; ensure each item is a complete URL with protocol.

Rules:
1) Do not invent or infer information; only extract what is explicitly present in the answer.
2) If any field is missing in the answer, set it to null (or empty array for all_urls).
3) For all_urls, include every URL reference present in the answer text (including markdown links). Do not include malformed entries.
4) If multiple universities are discussed, only extract details for the first one that appears in the answer.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_first_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    matches = re.findall(r"\d[\d,]*", s)
    if not matches:
        return None
    try:
        return int(matches[0].replace(",", ""))
    except Exception:
        return None


def normalize_setting(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s_low = s.strip().lower()
    # Normalize common phrases
    if "urban" in s_low:
        return "urban"
    if "suburban" in s_low:
        return "suburban"
    if "rural" in s_low:
        return "rural"
    return s_low


def detect_state_from_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    t = text.lower()
    # Look for state names/abbreviations
    if "virginia" in t or re.search(r"\bva\b", t):
        return "VA"
    if "north carolina" in t or re.search(r"\bnc\b", t):
        return "NC"
    if "tennessee" in t or re.search(r"\btn\b", t):
        return "TN"
    return None


def dedup_urls(urls: List[Optional[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def allowed_state_bool(extracted: UniversityExtraction) -> bool:
    state = extracted.stated_state
    addr = extracted.main_campus_address
    # Try explicit state first
    st_norm = None
    if state:
        st = state.strip().lower()
        if st in {"virginia", "va"}:
            st_norm = "VA"
        elif st in {"north carolina", "nc"}:
            st_norm = "NC"
        elif st in {"tennessee", "tn"}:
            st_norm = "TN"
    # Fallback to detecting from address
    if not st_norm:
        st_norm = detect_state_from_text(addr)
    return st_norm in {"VA", "NC", "TN"}


def term_is_fall_2024_or_2025(term: Optional[str]) -> bool:
    if not term:
        return False
    t = term.strip().lower()
    return "fall 2024" in t or "fall 2025" in t


# --------------------------------------------------------------------------- #
# Verification tree construction                                              #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, root_node, extracted: UniversityExtraction) -> None:
    identify_node = evaluator.add_parallel(
        id="Identify_University",
        desc="Identify one university that meets all specified criteria",
        parent=root_node,
        critical=True
    )

    # Gather sources
    all_urls = list(extracted.all_urls or [])
    # Ensure official and sacscoc URLs are included
    if extracted.official_website_url:
        all_urls.append(extracted.official_website_url)
    if extracted.sacscoc_accreditation_url:
        all_urls.append(extracted.sacscoc_accreditation_url)
    sources_general = dedup_urls(all_urls)

    # University Name
    name_node = evaluator.add_parallel(
        id="University_Name",
        desc="Provide the official name of the university",
        parent=identify_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.university_name and extracted.university_name.strip()),
        id="university_name_provided",
        desc="University name is provided",
        parent=name_node,
        critical=True
    )
    name_supported = evaluator.add_leaf(
        id="university_name_supported",
        desc="Official name matches the official website",
        parent=name_node,
        critical=True
    )
    name_claim = f"The official name of the university is '{extracted.university_name or ''}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_supported,
        sources=extracted.official_website_url if extracted.official_website_url else sources_general,
        additional_instruction=(
            "Verify that the university's official website reflects this official name. "
            "Branding on the homepage, title, header, or footer that matches or clearly indicates the same institution is acceptable. "
            "Allow minor punctuation and casing differences if it is clearly the same official name."
        )
    )

    # Physical Address
    addr_node = evaluator.add_parallel(
        id="Physical_Address",
        desc="Provide the complete physical address of the main campus",
        parent=identify_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.main_campus_address and extracted.main_campus_address.strip()),
        id="address_provided",
        desc="Main campus physical address is provided",
        parent=addr_node,
        critical=True
    )
    addr_supported = evaluator.add_leaf(
        id="address_supported",
        desc="Main campus physical address is supported by an official source",
        parent=addr_node,
        critical=True
    )
    addr_claim = f"The complete physical address of the main campus is '{extracted.main_campus_address or ''}'."
    await evaluator.verify(
        claim=addr_claim,
        node=addr_supported,
        sources=extracted.official_website_url if extracted.official_website_url else sources_general,
        additional_instruction=(
            "Check contact, visit, or footer sections on the official site for a full campus address. "
            "Minor formatting differences (e.g., abbreviations or punctuation) are acceptable as long as it is the same address."
        )
    )

    # State Location (VA/NC/TN)
    state_node = evaluator.add_parallel(
        id="State_Location",
        desc="The university is located in Virginia, North Carolina, or Tennessee",
        parent=identify_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=allowed_state_bool(extracted),
        id="state_in_allowed_detected",
        desc="State is one of VA, NC, or TN based on provided info",
        parent=state_node,
        critical=True
    )
    state_supported = evaluator.add_leaf(
        id="state_supported",
        desc="State location (VA/NC/TN) is supported by official source",
        parent=state_node,
        critical=True
    )
    state_supported_claim = (
        "The university's main campus is located in either Virginia, North Carolina, or Tennessee."
    )
    await evaluator.verify(
        claim=state_supported_claim,
        node=state_supported,
        sources=extracted.official_website_url if extracted.official_website_url else sources_general,
        additional_instruction=(
            "Confirm the state from the official website (e.g., address or About section). "
            "Accept Virginia (VA), North Carolina (NC), or Tennessee (TN)."
        )
    )

    # Undergraduate Enrollment
    enroll_node = evaluator.add_parallel(
        id="Undergraduate_Enrollment",
        desc="The university has total undergraduate enrollment between 7,000 and 25,000 students (fall 2024 or fall 2025)",
        parent=identify_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.undergrad_enrollment and extracted.undergrad_enrollment.strip()),
        id="enrollment_provided",
        desc="Undergraduate enrollment is provided",
        parent=enroll_node,
        critical=True
    )
    # Range check
    enrollment_value = parse_first_int(extracted.undergrad_enrollment)
    evaluator.add_custom_node(
        result=(enrollment_value is not None and 7000 <= enrollment_value <= 25000),
        id="enrollment_in_range",
        desc=f"Enrollment value is within 7,000–25,000 based on parsed integer {enrollment_value if enrollment_value is not None else 'None'}",
        parent=enroll_node,
        critical=True
    )
    # Term provided check
    evaluator.add_custom_node(
        result=term_is_fall_2024_or_2025(extracted.enrollment_term),
        id="enrollment_term_is_fall_2024_or_2025",
        desc="Enrollment term is Fall 2024 or Fall 2025",
        parent=enroll_node,
        critical=True
    )
    # Source support for enrollment value and term
    enroll_supported = evaluator.add_leaf(
        id="enrollment_supported",
        desc="Undergraduate enrollment (value and term) is supported by cited sources",
        parent=enroll_node,
        critical=True
    )
    enroll_claim = (
        f"The total undergraduate enrollment is '{extracted.undergrad_enrollment or ''}' "
        f"and this figure refers to '{extracted.enrollment_term or ''}'."
    )
    await evaluator.verify(
        claim=enroll_claim,
        node=enroll_supported,
        sources=extracted.official_website_url if extracted.official_website_url else sources_general,
        additional_instruction=(
            "Verify that the official site (e.g., facts, statistics, or institutional research pages) supports both the enrollment figure "
            "and that it corresponds to Fall 2024 or Fall 2025. "
            "Allow minor rounding (e.g., 10,000 vs. 10,050) as long as it's clearly the same figure."
        )
    )

    # Campus Size
    size_node = evaluator.add_parallel(
        id="Campus_Size",
        desc="The main campus is at least 90 acres in size",
        parent=identify_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.campus_size_acres and extracted.campus_size_acres.strip()),
        id="campus_size_provided",
        desc="Campus size (acres) is provided",
        parent=size_node,
        critical=True
    )
    size_value = parse_first_int(extracted.campus_size_acres)
    evaluator.add_custom_node(
        result=(size_value is not None and size_value >= 90),
        id="campus_size_at_least_90",
        desc=f"Campus size parsed integer {size_value if size_value is not None else 'None'} is at least 90 acres",
        parent=size_node,
        critical=True
    )
    size_supported = evaluator.add_leaf(
        id="campus_size_supported",
        desc="Campus size is supported by cited sources",
        parent=size_node,
        critical=True
    )
    size_claim = f"The main campus size is '{extracted.campus_size_acres or ''}'."
    await evaluator.verify(
        claim=size_claim,
        node=size_supported,
        sources=extracted.official_website_url if extracted.official_website_url else sources_general,
        additional_instruction=(
            "Verify a page on the official site (e.g., facts & figures, campus profile) that states the campus size (in acres). "
            "Minor unit formatting differences are acceptable if the numeric value matches."
        )
    )

    # Campus Setting
    setting_node = evaluator.add_parallel(
        id="Campus_Setting",
        desc="The campus setting is classified as urban or suburban (not rural)",
        parent=identify_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.campus_setting and extracted.campus_setting.strip()),
        id="campus_setting_provided",
        desc="Campus setting is provided",
        parent=setting_node,
        critical=True
    )
    setting_norm = normalize_setting(extracted.campus_setting)
    evaluator.add_custom_node(
        result=(setting_norm in {"urban", "suburban"}),
        id="campus_setting_allowed",
        desc=f"Campus setting '{setting_norm if setting_norm else 'None'}' is either urban or suburban",
        parent=setting_node,
        critical=True
    )
    setting_supported = evaluator.add_leaf(
        id="campus_setting_supported",
        desc="Campus setting classification is supported by cited sources",
        parent=setting_node,
        critical=True
    )
    setting_claim = f"The campus setting is classified as '{extracted.campus_setting or ''}' (urban or suburban)."
    await evaluator.verify(
        claim=setting_claim,
        node=setting_supported,
        sources=extracted.official_website_url if extracted.official_website_url else sources_general,
        additional_instruction=(
            "Confirm that the campus is characterized as urban or suburban. "
            "Accept reasonable official descriptions on the university site (e.g., 'located in an urban environment'). "
            "Do not accept 'rural'."
        )
    )

    # Official Website URL
    site_node = evaluator.add_parallel(
        id="Official_Website_URL",
        desc="Provide a direct link to the university's official website",
        parent=identify_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.official_website_url and extracted.official_website_url.strip()),
        id="official_site_provided",
        desc="Official website URL is provided",
        parent=site_node,
        critical=True
    )
    site_supported = evaluator.add_leaf(
        id="official_site_supported",
        desc="Provided URL is the university's official website",
        parent=site_node,
        critical=True
    )
    site_claim = (
        f"This URL is the official website of '{extracted.university_name or 'the university'}': "
        f"{extracted.official_website_url or ''}"
    )
    await evaluator.verify(
        claim=site_claim,
        node=site_supported,
        sources=extracted.official_website_url if extracted.official_website_url else sources_general,
        additional_instruction=(
            "Verify that the URL is an official site for the university (typically a .edu domain or clear official branding). "
            "Homepage is acceptable."
        )
    )

    # SACSCOC Accreditation URL
    sacscoc_node = evaluator.add_parallel(
        id="SACSCOC_Accreditation_URL",
        desc="Provide a direct link to a page confirming the university's SACSCOC accreditation",
        parent=identify_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(extracted.sacscoc_accreditation_url and extracted.sacscoc_accreditation_url.strip()),
        id="sacscoc_url_provided",
        desc="SACSCOC accreditation URL is provided",
        parent=sacscoc_node,
        critical=True
    )
    sacscoc_supported = evaluator.add_leaf(
        id="sacscoc_supported",
        desc="Accreditation page confirms SACSCOC accreditation for the university",
        parent=sacscoc_node,
        critical=True
    )
    sacscoc_claim = (
        f"This page confirms that '{extracted.university_name or 'the university'}' is accredited by the "
        f"Southern Association of Colleges and Schools Commission on Colleges (SACSCOC)."
    )
    await evaluator.verify(
        claim=sacscoc_claim,
        node=sacscoc_supported,
        sources=extracted.sacscoc_accreditation_url if extracted.sacscoc_accreditation_url else sources_general,
        additional_instruction=(
            "Look for explicit confirmation of SACSCOC accreditation. "
            "Accept official SACSCOC member listing pages, or official university accreditation pages that explicitly reference SACSCOC accreditation."
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
    model: str = "o4-mini"
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
        default_model=model
    )

    extracted = await evaluator.extract(
        prompt=prompt_extract_university(),
        template_class=UniversityExtraction,
        extraction_name="university_info"
    )

    # Optional: record some parsed/derived info for transparency
    evaluator.add_custom_info(
        info={
            "parsed_enrollment_int": parse_first_int(extracted.undergrad_enrollment),
            "parsed_campus_size_int": parse_first_int(extracted.campus_size_acres),
            "normalized_setting": normalize_setting(extracted.campus_setting),
            "detected_state": detect_state_from_text(extracted.main_campus_address) or (
                extracted.stated_state.strip() if extracted.stated_state else None
            )
        },
        info_type="derived_fields",
        info_name="parser_helpers"
    )

    await build_verification_tree(evaluator, root, extracted)

    return evaluator.get_summary()