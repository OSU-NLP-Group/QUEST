import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "public_universities_4_states"
TASK_DESCRIPTION = """
Find 4 public universities in the United States from at least 3 different states. For each university, provide: (1) The full official name of the university, (2) The U.S. state where it is located, (3) The official name and documented square footage of its student recreation center, (4) The official name and operating hours of its career services center, (5) The current undergraduate enrollment figure, and (6) Official URL references for each piece of information. All information must be verifiable through official university websites or government educational databases.
"""

# --------------------------------------------------------------------------- #
# US States mapping for diversity check                                       #
# --------------------------------------------------------------------------- #
US_STATE_ABBR_TO_NAME: Dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia"
}

def normalize_state_name(state_str: Optional[str]) -> Optional[str]:
    if not state_str:
        return None
    s = state_str.strip()
    if not s:
        return None
    # Normalize common punctuation and casing
    s_clean = s.replace(".", "").replace(",", "").strip()
    # Handle DC variants
    if s_clean.upper() in {"DC", "D C", "DISTRICT OF COLUMBIA", "WASHINGTON DC", "WASHINGTON D C"}:
        return "District of Columbia"
    # Abbreviation to full
    if len(s_clean) == 2 and s_clean.upper() in US_STATE_ABBR_TO_NAME:
        return US_STATE_ABBR_TO_NAME[s_clean.upper()]
    # Title case for normal names
    # Attempt to match known names by case-insensitive comparison
    for name in US_STATE_ABBR_TO_NAME.values():
        if s_clean.lower() == name.lower():
            return name
    # Try removing "state of " prefix
    lower = s_clean.lower()
    if lower.startswith("state of "):
        stripped = s_clean[9:].strip()
        for name in US_STATE_ABBR_TO_NAME.values():
            if stripped.lower() == name.lower():
                return name
        return stripped.title()
    return s_clean.title()


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class BasicInfo(BaseModel):
    university_name: Optional[str] = None
    state: Optional[str] = None
    public_status_urls: List[str] = Field(default_factory=list)


class RecreationInfo(BaseModel):
    rec_center_name: Optional[str] = None
    rec_center_sqft: Optional[str] = None
    rec_urls: List[str] = Field(default_factory=list)


class CareerInfo(BaseModel):
    career_center_name: Optional[str] = None
    career_hours: Optional[str] = None
    career_urls: List[str] = Field(default_factory=list)


class EnrollmentInfo(BaseModel):
    undergrad_enrollment: Optional[str] = None
    enrollment_urls: List[str] = Field(default_factory=list)


class UniversityExtraction(BaseModel):
    basic: Optional[BasicInfo] = None
    recreation: Optional[RecreationInfo] = None
    career: Optional[CareerInfo] = None
    enrollment: Optional[EnrollmentInfo] = None


class UniversitiesExtraction(BaseModel):
    universities: List[UniversityExtraction] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_universities() -> str:
    return """
    Extract information for up to the FIRST FOUR universities mentioned in the answer that are in the United States.
    For each university, return an object with the following nested fields. If something is missing from the answer, set it to null or an empty list as appropriate.
    Use arrays for URLs when multiple official sources are provided.

    Structure to return:
    {
      "universities": [
        {
          "basic": {
            "university_name": string or null,
            "state": string or null,
            "public_status_urls": [string, ...]   // official URLs confirming name/state/public status (university .edu, .gov, or official state system)
          },
          "recreation": {
            "rec_center_name": string or null,
            "rec_center_sqft": string or null,    // include units if present (e.g., "200,000 sq ft")
            "rec_urls": [string, ...]             // official URLs supporting rec center info and size
          },
          "career": {
            "career_center_name": string or null,
            "career_hours": string or null,       // preserve formatting as written (e.g., "Mon–Fri 8am–5pm")
            "career_urls": [string, ...]          // official URLs supporting name and hours
          },
          "enrollment": {
            "undergrad_enrollment": string or null,  // keep as provided (e.g., "31,500 undergraduates", "≈30k", "Fall 2024: 30,123")
            "enrollment_urls": [string, ...]         // official URLs supporting undergraduate enrollment
          }
        }
      ]
    }

    IMPORTANT REQUIREMENTS:
    - Only include the first four universities if more are listed.
    - For each piece of information, extract the official URLs cited in the answer text. They must be explicit URLs (plain or in markdown).
    - Prefer .edu or .gov domains or official state university system domains.
    - Do not invent any URLs or data not present in the answer.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def nonempty_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and any((u or "").strip() for u in urls or [])


async def verify_university(
    evaluator: Evaluator,
    parent_node,
    uni: UniversityExtraction,
    idx: int
) -> None:
    """
    Build verification subtree for a single university and execute verifications.
    """
    ulabel = f"University {idx + 1}"
    uni_node = evaluator.add_parallel(
        id=f"univ_{idx+1}",
        desc=f"{ulabel} verification",
        parent=parent_node,
        critical=False  # Each university contributes partial credit independently
    )

    # Safeguard empty university object
    basic = uni.basic or BasicInfo()
    recreation = uni.recreation or RecreationInfo()
    career = uni.career or CareerInfo()
    enrollment = uni.enrollment or EnrollmentInfo()

    # --------------------------- Basic Info -------------------------------- #
    basic_node = evaluator.add_parallel(
        id=f"u{idx+1}_basic",
        desc=f"{ulabel}: Basic identification and location information",
        parent=uni_node,
        critical=True
    )

    # URL existence (critical gate)
    basic_url_exist = evaluator.add_custom_node(
        result=nonempty_urls(basic.public_status_urls),
        id=f"u{idx+1}_basic_url",
        desc=f"{ulabel}: Official basic/public-status URL(s) provided",
        parent=basic_node,
        critical=True
    )

    # Name verification
    name_node = evaluator.add_leaf(
        id=f"u{idx+1}_name",
        desc=f"{ulabel}: Official full university name is correctly stated",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official full name of the university is '{basic.university_name or ''}'.",
        node=name_node,
        sources=basic.public_status_urls,
        additional_instruction=(
            "Accept as supported only if the page is an official university (.edu), government (.gov), or "
            "official state university system domain. Allow minor punctuation/casing variations. Prefer explicitly labeled 'official name'."
        )
    )

    # State verification
    state_node = evaluator.add_leaf(
        id=f"u{idx+1}_state",
        desc=f"{ulabel}: University state is correctly identified",
        parent=basic_node,
        critical=True
    )
    state_claim = f"The university is located in the U.S. state of '{basic.state or ''}'."
    await evaluator.verify(
        claim=state_claim,
        node=state_node,
        sources=basic.public_status_urls,
        additional_instruction=(
            "Accept as supported if the page shows the state directly or via city+state. "
            "Allow state abbreviations (e.g., CA) and full names to match."
        )
    )

    # Public status verification
    public_node = evaluator.add_leaf(
        id=f"u{idx+1}_public",
        desc=f"{ulabel}: Institution is verified as a public university",
        parent=basic_node,
        critical=True
    )
    await evaluator.verify(
        claim="This institution is a public university (publicly funded/state university).",
        node=public_node,
        sources=basic.public_status_urls,
        additional_instruction=(
            "Accept only if the official page clearly indicates that the institution is a 'public' university "
            "or is part of a public state system. Reject if the page is non-official (.edu/.gov/system domains required)."
        )
    )

    # ------------------------ Recreation Center ---------------------------- #
    rec_node = evaluator.add_parallel(
        id=f"u{idx+1}_rec",
        desc=f"{ulabel}: Recreation center facility information",
        parent=uni_node,
        critical=True
    )

    rec_url_exist = evaluator.add_custom_node(
        result=nonempty_urls(recreation.rec_urls),
        id=f"u{idx+1}_rec_url",
        desc=f"{ulabel}: Official recreation center URL(s) provided",
        parent=rec_node,
        critical=True
    )

    rec_name_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_rec_name",
        desc=f"{ulabel}: Official student recreation center name is correctly stated",
        parent=rec_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the student recreation center is '{recreation.rec_center_name or ''}'.",
        node=rec_name_leaf,
        sources=recreation.rec_urls,
        additional_instruction=(
            "Accept only if confirmed on an official university (.edu) or official campus/unit site. "
            "Allow minor formatting/abbreviation variations (e.g., 'Center' vs 'Ctr')."
        )
    )

    rec_size_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_rec_size",
        desc=f"{ulabel}: Recreation center square footage is correctly stated",
        parent=rec_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The student recreation center has a documented size of {recreation.rec_center_sqft or ''} (square feet).",
        node=rec_size_leaf,
        sources=recreation.rec_urls,
        additional_instruction=(
            "Verify that the page specifies the size in square feet (accept 'sq ft', 'SF', or 'square feet'). "
            "Allow comma separators and minor rounding if the page indicates 'about' or '~'. "
            "Accept only official university pages or official campus facility pages."
        )
    )

    # ------------------------ Career Services ------------------------------ #
    career_node = evaluator.add_parallel(
        id=f"u{idx+1}_career",
        desc=f"{ulabel}: Career services information",
        parent=uni_node,
        critical=True
    )

    career_url_exist = evaluator.add_custom_node(
        result=nonempty_urls(career.career_urls),
        id=f"u{idx+1}_career_url",
        desc=f"{ulabel}: Official career services URL(s) provided",
        parent=career_node,
        critical=True
    )

    career_name_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_career_name",
        desc=f"{ulabel}: Official career services center name is correctly stated",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The official name of the career services center is '{career.career_center_name or ''}'.",
        node=career_name_leaf,
        sources=career.career_urls,
        additional_instruction=(
            "Accept only if the center name is clearly presented on an official university (.edu) or official unit page. "
            "Allow minor abbreviation/branding variations."
        )
    )

    career_hours_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_career_hours",
        desc=f"{ulabel}: Career services operating hours are correctly stated",
        parent=career_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The operating hours of the career services center are: {career.career_hours or ''}.",
        node=career_hours_leaf,
        sources=career.career_urls,
        additional_instruction=(
            "Verify the posted office hours for the main career services office. "
            "Accept typical term-time schedules; format differences (e.g., 'Mon–Fri 8am–5pm' vs 'Monday-Friday, 8:00 AM to 5:00 PM') are acceptable."
        )
    )

    # ------------------------ Enrollment ---------------------------------- #
    enroll_node = evaluator.add_parallel(
        id=f"u{idx+1}_enroll",
        desc=f"{ulabel}: Undergraduate enrollment information",
        parent=uni_node,
        critical=True
    )

    enroll_url_exist = evaluator.add_custom_node(
        result=nonempty_urls(enrollment.enrollment_urls),
        id=f"u{idx+1}_enroll_url",
        desc=f"{ulabel}: Official enrollment URL(s) provided",
        parent=enroll_node,
        critical=True
    )

    enroll_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_enroll_num",
        desc=f"{ulabel}: Current undergraduate enrollment is correctly stated",
        parent=enroll_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"The current undergraduate enrollment is {enrollment.undergrad_enrollment or ''}.",
        node=enroll_leaf,
        sources=enrollment.enrollment_urls,
        additional_instruction=(
            "Verify the undergraduate enrollment figure as stated on an official university (.edu) page, institutional research/facts page, "
            "or government educational database (.gov). Allow minor rounding or 'about' qualifiers."
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
    """
    Evaluate an answer for the public universities multi-criteria task and return a structured summary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root: independent subtrees aggregated in parallel
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

    # Extract universities data
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Ensure exactly 4 entries (pad with empty ones if fewer, take first 4 if more)
    universities: List[UniversityExtraction] = list(extracted.universities[:4])
    while len(universities) < 4:
        universities.append(UniversityExtraction())

    # Add container nodes for each university
    uni_nodes = []
    for i in range(4):
        uni_nodes.append(
            evaluator.add_parallel(
                id=f"univ_container_{i+1}",
                desc=f"University {i+1} container",
                parent=root,
                critical=False
            )
        )

    # Verify each university subtree
    for i in range(4):
        await verify_university(evaluator, uni_nodes[i], universities[i], i)

    # Geographic diversity check (at least 3 different U.S. states among the four)
    # Use extracted states directly and normalize
    states_raw: List[Optional[str]] = [
        (universities[i].basic.state if universities[i].basic else None) for i in range(4)
    ]
    normalized_states: List[str] = [
        s for s in (normalize_state_name(sv) for sv in states_raw) if s is not None and s.strip() != ""
    ]
    unique_states = sorted(set(normalized_states))
    geo_diverse = len(unique_states) >= 3

    evaluator.add_custom_info(
        info={
            "extracted_states_raw": states_raw,
            "normalized_states": normalized_states,
            "unique_states": unique_states,
            "unique_state_count": len(unique_states)
        },
        info_type="intermediate",
        info_name="geographic_diversity_calc"
    )

    evaluator.add_custom_node(
        result=geo_diverse,
        id="geo_diversity",
        desc="At least 3 distinct U.S. states are represented among the 4 universities",
        parent=root,
        critical=True  # Critical requirement for the overall task
    )

    # Return summary
    return evaluator.get_summary()