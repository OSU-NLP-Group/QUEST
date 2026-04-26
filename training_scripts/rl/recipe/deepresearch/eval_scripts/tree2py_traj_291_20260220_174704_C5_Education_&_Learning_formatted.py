import asyncio
import logging
from typing import Any, List, Optional, Dict
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "tx_school_districts_70k_eval"
TASK_DESCRIPTION = """Find four public school districts in Texas, each with a student enrollment of at least 70,000 students. For each district, provide the following information that must be publicly available on the district's official website:

1. The official district name
2. The district's official website URL
3. The current superintendent's full name
4. The superintendent's official district email address
5. The regular board meeting schedule pattern (e.g., "second Thursday of each month at 6:00 PM")
6. A direct URL to the board meeting calendar, schedule, or agendas page
7. The main district office phone number

All information must be current as of February 2026 and verifiable through each district's official website.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class DistrictItem(BaseModel):
    """Structured info for one district extracted from the agent's answer."""
    official_name: Optional[str] = None
    website_url: Optional[str] = None
    enrollment: Optional[str] = None
    superintendent_name: Optional[str] = None
    superintendent_email: Optional[str] = None
    meeting_schedule: Optional[str] = None
    meeting_calendar_url: Optional[str] = None
    main_phone: Optional[str] = None
    additional_official_urls: List[str] = Field(default_factory=list)


class DistrictList(BaseModel):
    """Top-level extraction container holding up to 4 districts."""
    districts: List[DistrictItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_districts() -> str:
    return """
    Extract up to four public school districts in Texas that the answer provides. For each district, extract ONLY information explicitly present in the answer text. Do not infer or invent.

    For each district, extract the following fields:
    - official_name: The official district name exactly as stated in the answer (e.g., "Houston Independent School District" or "Houston ISD").
    - website_url: The primary official district website URL mentioned in the answer (e.g., https://www.houstonisd.org). Include full URL with protocol. If multiple are present, choose the main/homepage.
    - enrollment: The student enrollment number or string if presented (e.g., "198,000", "about 120,000"). If not provided in the answer, set null.
    - superintendent_name: The current superintendent's full name, as given in the answer. If absent, set null.
    - superintendent_email: The superintendent's official district email address, if provided (e.g., firstname.lastname@district.org). If absent, set null.
    - meeting_schedule: The regular board meeting schedule pattern text provided (e.g., "second Thursday each month at 6:00 PM"). If absent, set null.
    - meeting_calendar_url: A direct URL cited in the answer to the board meeting calendar, schedule, or agendas page. If absent, set null.
    - main_phone: The main district office phone number cited in the answer (e.g., "(713) 556-6000"). If absent, set null.
    - additional_official_urls: An array of any other official district URLs cited in the answer for this district (e.g., superintendent page, board page, contact page). Include only URLs from the district’s official domain(s). If none, return an empty array.

    Rules:
    - Return at most 4 district objects under the 'districts' array, in the same order as presented in the answer.
    - Include full URLs (with http:// or https://).
    - If a field is missing in the answer, set it to null (or [] for arrays).
    - Do not add any district not mentioned in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if not (u.startswith("http://") or u.startswith("https://")):
        return False
    try:
        parsed = urlparse(u)
        return bool(parsed.scheme) and bool(parsed.netloc)
    except Exception:
        return False


def _collect_official_sources(d: DistrictItem) -> List[str]:
    """Collect unique official sources for verification from the extracted item."""
    sources: List[str] = []
    if _is_valid_url(d.website_url):
        sources.append(d.website_url.strip())
    if _is_valid_url(d.meeting_calendar_url):
        sources.append(d.meeting_calendar_url.strip())
    for u in d.additional_official_urls or []:
        if _is_valid_url(u):
            sources.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_one_district(
    evaluator: Evaluator,
    parent_node,
    d: DistrictItem,
    index: int
) -> None:
    """
    Build verification sub-tree and run verifications for a single district.
    Follows the rubric structure:
      - identification (critical, parallel)
      - leadership (critical, parallel)
      - board_meetings (critical, parallel)
      - contact (critical leaf)
    """
    # Parent district node (non-critical to allow partial credit across districts)
    district_node = evaluator.add_parallel(
        id=f"district_{index+1}",
        desc=f"{['First','Second','Third','Fourth'][index]} qualifying Texas school district information",
        parent=parent_node,
        critical=False
    )

    # Gather official sources once
    official_sources = _collect_official_sources(d)

    # ---------------- Identification (critical) ---------------- #
    identification_node = evaluator.add_parallel(
        id=f"district_{index+1}_identification",
        desc="District has at least 70,000 students and basic information is correct",
        parent=district_node,
        critical=True
    )

    # Official district name – verify supported by official sources
    name_leaf = evaluator.add_leaf(
        id=f"district_{index+1}_name",
        desc="Official district name is provided",
        parent=identification_node,
        critical=True
    )
    name_claim = f"The official district name is '{d.official_name or ''}'."
    await evaluator.verify(
        claim=name_claim,
        node=name_leaf,
        sources=official_sources,
        additional_instruction=(
            "Verify on the district's official website that the provided name is the district's official name. "
            "Allow minor variations such as 'ISD' vs 'Independent School District' and casing. "
            "Ensure it is a public school district in Texas."
        )
    )

    # Official district website URL – existence check (critical)
    website_exists = _is_valid_url(d.website_url)
    evaluator.add_custom_node(
        result=website_exists,
        id=f"district_{index+1}_website",
        desc="Official district website URL is provided",
        parent=identification_node,
        critical=True
    )

    # Enrollment ≥ 70,000 – verify via official sources
    enrollment_leaf = evaluator.add_leaf(
        id=f"district_{index+1}_enrollment",
        desc="District has at least 70,000 students enrolled",
        parent=identification_node,
        critical=True
    )
    enrollment_claim = "The district has at least 70,000 students enrolled."
    await evaluator.verify(
        claim=enrollment_claim,
        node=enrollment_leaf,
        sources=official_sources,
        additional_instruction=(
            "Use the district's official website pages (e.g., Fast Facts, About, Statistics) to confirm student enrollment. "
            "Reasonable approximations (e.g., 'around 70,000' or '70,000+') count as meeting the threshold as of February 2026."
        )
    )

    # ---------------- Leadership (critical) ---------------- #
    leadership_node = evaluator.add_parallel(
        id=f"district_{index+1}_leadership",
        desc="Superintendent information is correct",
        parent=district_node,
        critical=True
    )

    # Superintendent name – verify via official sources
    sup_name_leaf = evaluator.add_leaf(
        id=f"district_{index+1}_superintendent_name",
        desc="Current superintendent's full name is provided",
        parent=leadership_node,
        critical=True
    )
    sup_name_claim = f"The current superintendent is '{d.superintendent_name or ''}'."
    await evaluator.verify(
        claim=sup_name_claim,
        node=sup_name_leaf,
        sources=official_sources,
        additional_instruction=(
            "Confirm on the district's official website (e.g., Superintendent page, Leadership, or Board pages) "
            "that this is the current superintendent as of February 2026. Allow minor name variations and titles."
        )
    )

    # Superintendent email – verify via official sources
    sup_email_leaf = evaluator.add_leaf(
        id=f"district_{index+1}_superintendent_email",
        desc="Superintendent's official district email address is provided",
        parent=leadership_node,
        critical=True
    )
    sup_email_claim = f"The superintendent's official district email address is '{d.superintendent_email or ''}'."
    await evaluator.verify(
        claim=sup_email_claim,
        node=sup_email_leaf,
        sources=official_sources,
        additional_instruction=(
            "Verify that the email address is shown on the district's official website (e.g., contact page, superintendent page). "
            "Accept 'mailto:' links or canonical obfuscated patterns if clearly indicating the same address. "
            "Prefer domain consistency with the district's official domain."
        )
    )

    # ---------------- Board Meetings (critical) ---------------- #
    board_node = evaluator.add_parallel(
        id=f"district_{index+1}_board_meetings",
        desc="Board meeting information is correct",
        parent=district_node,
        critical=True
    )

    # Meeting schedule pattern – verify against board calendar/agendas page if available
    schedule_leaf = evaluator.add_leaf(
        id=f"district_{index+1}_meeting_schedule",
        desc="Regular board meeting schedule pattern is provided",
        parent=board_node,
        critical=True
    )
    schedule_claim = f"The regular board meeting schedule pattern is '{d.meeting_schedule or ''}'."
    # Prefer meeting calendar URL if present; otherwise use other official sources
    schedule_sources = (
        [d.meeting_calendar_url.strip()] if _is_valid_url(d.meeting_calendar_url)
        else official_sources
    )
    await evaluator.verify(
        claim=schedule_claim,
        node=schedule_leaf,
        sources=schedule_sources,
        additional_instruction=(
            "Verify that the provided pattern matches the recurring schedule indicated on the district's official board calendar, "
            "board meetings schedule page, or agendas page. "
            "Allow minor textual variations if the recurring pattern is equivalent."
        )
    )

    # Meeting calendar URL – existence check (critical)
    calendar_exists = _is_valid_url(d.meeting_calendar_url)
    evaluator.add_custom_node(
        result=calendar_exists,
        id=f"district_{index+1}_meeting_calendar_url",
        desc="Direct URL to board meeting calendar/schedule page is provided",
        parent=board_node,
        critical=True
    )

    # ---------------- Contact (critical leaf under district) -------------- #
    contact_leaf = evaluator.add_leaf(
        id=f"district_{index+1}_contact",
        desc="Main district office phone number is provided",
        parent=district_node,
        critical=True
    )
    contact_claim = f"The main district office phone number is '{d.main_phone or ''}'."
    await evaluator.verify(
        claim=contact_claim,
        node=contact_leaf,
        sources=official_sources,
        additional_instruction=(
            "Verify on the district's official website (e.g., Contact Us or About page) that the provided phone number "
            "is the main district office phone. Accept standard formatting variations."
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
    Evaluate an answer for the Texas districts (>=70k enrollment) task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root-level: districts evaluated independently
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

    # Note: The framework enforces that a critical parent cannot have non-critical children.
    # The JSON marks the root as "critical", but district children are non-critical.
    # We therefore keep the root as non-critical (default in initialize) to comply with the framework.

    # Extract districts info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_districts(),
        template_class=DistrictList,
        extraction_name="districts_extraction"
    )

    # Keep only the first 4 districts; pad with empty if fewer
    districts: List[DistrictItem] = list(extracted.districts[:4])
    while len(districts) < 4:
        districts.append(DistrictItem())

    # Build verification subtrees for each district
    for idx, d in enumerate(districts):
        await verify_one_district(evaluator, root, d, idx)

    # Return standardized evaluation summary
    return evaluator.get_summary()