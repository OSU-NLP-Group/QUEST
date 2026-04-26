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
TASK_ID = "career_centers_three_states"
TASK_DESCRIPTION = """I am researching university career centers to understand the comprehensive range of services they provide to students. Find three universities, each located in a different U.S. state, where the career center offers all of the following services:

1. Resume review services (in any format: in-person appointments, drop-in sessions, virtual meetings, or online submission)
2. Mock interview or interview preparation services (practice interviews offered in-person, virtually, or through scheduled appointments)
3. Access to at least one online career development platform (such as Handshake, Big Interview, VMock, or similar digital career tools)

For each of the three universities, provide:
- The university name and U.S. state location
- The career center's full contact information: physical address, email address, and phone number
- A description of how resume review services are delivered
- A description of how mock interview or interview preparation services are delivered
- The name(s) of online career platform(s) accessible to students
- Direct URL reference(s) to the official career center webpage(s) documenting these services

Ensure all three universities are in different U.S. states and all information is verifiable through official university career center websites.
"""

# --------------------------------------------------------------------------- #
# US States normalization utilities                                           #
# --------------------------------------------------------------------------- #

STATE_ABBR_TO_NAME: Dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "District of Columbia",
}

STATE_NAME_TO_ABBR: Dict[str, str] = {v: k for k, v in STATE_ABBR_TO_NAME.items()}

def normalize_state(state: Optional[str]) -> Optional[str]:
    if not state:
        return None
    s = state.strip()
    if not s:
        return None
    # Standardize capitalization for name matching
    title = s.replace(".", "").replace(",", "").strip()
    # Try exact full name match (case-insensitive)
    for name in STATE_NAME_TO_ABBR.keys():
        if title.lower() == name.lower():
            return name
    # Try common variants for DC
    if title.lower() in {"dc", "d c", "d.c", "d.c.", "district of columbia", "washington dc", "washington, dc"}:
        return "District of Columbia"
    # Try 2-letter abbreviation
    abbr = title.upper()
    if abbr in STATE_ABBR_TO_NAME:
        return STATE_ABBR_TO_NAME[abbr]
    return None

def is_valid_us_state(state: Optional[str]) -> bool:
    return normalize_state(state) is not None

def is_official_career_url(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    # Must be .edu domain; allow subdomains
    if ".edu" not in u:
        return False
    # Heuristic: contains career keywords
    career_keywords = [
        "career", "careers", "careercenter", "career-center", "career-services",
        "career_services", "career-development", "careerdevelopment"
    ]
    return any(k in u for k in career_keywords)

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #

class UniversityEntry(BaseModel):
    university_name: Optional[str] = None
    state: Optional[str] = None
    address: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    resume_review_description: Optional[str] = None
    mock_interview_description: Optional[str] = None
    online_platforms: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)

class UniversitiesExtraction(BaseModel):
    universities: List[UniversityEntry] = Field(default_factory=list)

# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #

def prompt_extract_universities() -> str:
    return """
    Extract up to five universities described in the answer that have career center information. For each university, extract the following fields exactly as stated in the answer (do not infer or invent):
    - university_name: The university's full name.
    - state: The U.S. state where the university is located (as provided, can be full name or abbreviation).
    - address: The career center’s physical address (street, city, state, and ZIP as provided).
    - email: The career center’s email address.
    - phone: The career center’s phone number.
    - resume_review_description: A brief phrase or sentence describing how resume review services are delivered (e.g., in-person appointments, drop-in hours, virtual review, or online submission).
    - mock_interview_description: A brief phrase or sentence describing how mock interview or interview preparation services are delivered (e.g., in-person or virtual practice interviews, appointment-based).
    - online_platforms: An array of the names of online career platforms accessible to students (e.g., Handshake, Big Interview, VMock, CareerShift, Focus2, GoinGlobal, etc.). Use exact names mentioned in the answer.
    - reference_urls: An array of the direct URL(s) to official university career center webpages that document these services. Only include URLs explicitly present in the answer (plain URLs or inside markdown links). Do not add or infer any URLs.

    Return a JSON object with:
    {
      "universities": [ { ... }, { ... }, { ... }, ... ]
    }

    If any field is missing for a given university in the answer, set it to null (or an empty list for arrays).
    Preserve the original text for addresses and descriptions; do not normalize or change formatting.
    """

# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #

def ordinal(n: int) -> str:
    return ["First", "Second", "Third", "Fourth", "Fifth"][n] if 0 <= n < 5 else f"#{n+1}"

def build_platforms_text(platforms: List[str]) -> str:
    if not platforms:
        return ""
    return ", ".join(platforms)

# --------------------------------------------------------------------------- #
# Verification per-university                                                 #
# --------------------------------------------------------------------------- #

async def verify_university(
    evaluator: Evaluator,
    root_node,
    uni: UniversityEntry,
    idx: int,
    prior_states_norm: List[str],
) -> None:
    """
    Build and evaluate the subtree for one university. Children are critical as per rubric.
    Parent (university_i) is non-critical under root to allow partial credit across universities.
    """
    uni_node = evaluator.add_parallel(
        id=f"university_{idx+1}",
        desc=f"{ordinal(idx)} university meeting all service requirements",
        parent=root_node,
        critical=False
    )

    # Identification (critical) - presence + state validity + distinctness constraint
    norm_state = normalize_state(uni.state)
    name_present = bool(uni.university_name and uni.university_name.strip())
    state_valid = is_valid_us_state(uni.state)
    unique_state_ok = norm_state is not None and norm_state not in prior_states_norm
    # For the first university, uniqueness is trivially true
    if idx == 0:
        unique_state_ok = state_valid

    id_desc = "University name and U.S. state location provided" if idx == 0 else (
        "University name and U.S. state location provided, confirmed to be in a different state than University 1"
        if idx == 1 else
        "University name and U.S. state location provided, confirmed to be in a different state than Universities 1 and 2"
    )
    evaluator.add_custom_node(
        result=(name_present and state_valid and unique_state_ok),
        id=f"u{idx+1}_identification",
        desc=id_desc,
        parent=uni_node,
        critical=True
    )

    # Reference URL(s) presence & basic officialness (critical)
    has_official = any(is_official_career_url(u) for u in (uni.reference_urls or []))
    evaluator.add_custom_node(
        result=(bool(uni.reference_urls) and has_official),
        id=f"u{idx+1}_reference_url",
        desc="Official career center webpage URL(s) provided to verify and document the services",
        parent=uni_node,
        critical=True
    )

    # Prepare sources for URL-grounded verification
    sources = uni.reference_urls if uni.reference_urls else None

    # Contact information verification (critical)
    contact_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_contact_info",
        desc="Career center contact information including physical address, email address, and phone number",
        parent=uni_node,
        critical=True
    )
    contact_claim = (
        f"On the official career center webpage(s) for {uni.university_name}, the career center's contact information includes: "
        f"physical address '{uni.address}', email address '{uni.email}', and phone number '{uni.phone}'."
    )
    contact_instruction = (
        "Verify that the provided career center page(s) list the stated contact information. "
        "Allow minor formatting differences (e.g., punctuation, area code formatting, or line breaks in addresses). "
        "It's acceptable if the address appears in a footer or contact section. "
        "Confirm that all three elements—address, email, and phone—are present and consistent with the claim."
    )

    # Resume review verification (critical)
    resume_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_resume_review",
        desc="Resume review service availability with description of delivery format (in-person, virtual, online submission, or drop-in)",
        parent=uni_node,
        critical=True
    )
    resume_claim = (
        f"On the official career center webpage(s) for {uni.university_name}, "
        f"the career center offers resume review services. The delivery format is described as: {uni.resume_review_description}."
    )
    resume_instruction = (
        "Look for terms like 'resume review', 'resume critique', 'CV review', 'resume feedback', or similar. "
        "Confirm the availability of the service and that the delivery format (e.g., in-person appointments, drop-ins, virtual, or online submission) matches or is reasonably equivalent to the claim."
    )

    # Mock interview verification (critical)
    mock_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_mock_interview",
        desc="Mock interview or interview preparation service availability with description of delivery format (in-person, virtual, or appointment-based)",
        parent=uni_node,
        critical=True
    )
    mock_claim = (
        f"On the official career center webpage(s) for {uni.university_name}, "
        f"the career center offers mock interview or interview preparation services. "
        f"The delivery format is described as: {uni.mock_interview_description}."
    )
    mock_instruction = (
        "Look for terms like 'mock interview', 'practice interview', 'interview coaching', or 'interview preparation'. "
        "Confirm that the service is available and that the delivery format (e.g., in-person, virtual, appointment-based) matches or is reasonably equivalent to the claim."
    )

    # Online platform access verification (critical)
    platform_leaf = evaluator.add_leaf(
        id=f"u{idx+1}_online_platform",
        desc="Name of at least one online career platform accessible to students (such as Handshake, Big Interview, VMock, or similar)",
        parent=uni_node,
        critical=True
    )
    platforms_text = build_platforms_text(uni.online_platforms)
    platform_claim = (
        f"On the official career center webpage(s) for {uni.university_name}, "
        f"students have access to the following online career platform(s): {platforms_text}."
    )
    platform_instruction = (
        "Verify that the page mentions the named platform(s) (e.g., Handshake, Big Interview, VMock, CareerShift, Focus2, GoinGlobal, etc.). "
        "Allow minor name variations (e.g., 'BigInterview' vs 'Big Interview'). "
        "If multiple platforms are listed in the claim, it is sufficient if all listed are present on the page(s)."
    )

    # Batch verify the four URL-grounded leaves; precondition logic will auto-skip if critical siblings failed
    claims_and_sources = [
        (contact_claim, sources, contact_leaf, contact_instruction),
        (resume_claim, sources, resume_leaf, resume_instruction),
        (mock_claim, sources, mock_leaf, mock_instruction),
        (platform_claim, sources, platform_leaf, platform_instruction),
    ]
    await evaluator.batch_verify(claims_and_sources)

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
    Evaluate an answer for the university career center services task.
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
        default_model=model
    )

    # Extract structured university info
    extracted = await evaluator.extract(
        prompt=prompt_extract_universities(),
        template_class=UniversitiesExtraction,
        extraction_name="universities_extraction"
    )

    # Normalize and select first three universities (pad with blanks if needed)
    universities: List[UniversityEntry] = list(extracted.universities[:3])
    while len(universities) < 3:
        universities.append(UniversityEntry())

    # Track prior normalized states for uniqueness checks
    prior_states_norm: List[str] = []
    for i in range(3):
        uni = universities[i]
        # Build per-university verification subtree
        await verify_university(evaluator, root, uni, i, prior_states_norm)
        # Update prior states list using normalization (only if valid)
        ns = normalize_state(uni.state)
        if ns and ns not in prior_states_norm:
            prior_states_norm.append(ns)

    return evaluator.get_summary()