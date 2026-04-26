import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ai_ml_conf_2025_na"
TASK_DESCRIPTION = (
    "Identify three major peer-reviewed artificial intelligence or machine learning research conferences "
    "that will be held in North America between June 2025 and December 2025 (inclusive). Each conference must meet all criteria. "
    "For each, provide official conference name, full dates in YYYY-MM-DD (start to end), venue name, and complete venue address."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueAddress(BaseModel):
    street_address: Optional[str] = None  # street number + street name in one line
    city: Optional[str] = None
    state_province: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None


class ConferenceItem(BaseModel):
    conference_name: Optional[str] = None
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD
    venue_name: Optional[str] = None
    venue_address: VenueAddress = Field(default_factory=VenueAddress)
    source_urls: List[str] = Field(default_factory=list)  # Official conference and/or venue URLs


class ConferenceExtraction(BaseModel):
    conferences: List[ConferenceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
    Extract all conference entries mentioned in the answer that purport to satisfy the task.
    For each conference, return an object with the following fields:

    - conference_name: The complete official conference name, exactly as written in the answer.
    - start_date: The conference start date in YYYY-MM-DD format (return null if not provided in this exact format).
    - end_date: The conference end date in YYYY-MM-DD format (return null if not provided in this exact format).
    - venue_name: The official name of the physical venue (convention center) where the conference is held.
    - venue_address:
        - street_address: The street number and street name in one line (e.g., "800 Convention Center Blvd").
        - city
        - state_province
        - postal_code
        - country
      If any address component is missing from the answer, return null for that component.
    - source_urls: A list of URLs explicitly mentioned in the answer that point to official conference webpages
      (e.g., the conference site, program/schedule page) and/or official venue webpages (e.g., the venue site or listing page).
      Extract only URLs present in the answer; do not invent URLs. Include full URLs with protocol.

    Important:
    - Extract ALL conferences mentioned in the answer (not just the first three).
    - If a field is missing from the answer, set it to null.
    - Only include valid URLs that appear in the answer; ignore malformed URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _is_nonempty_str(s: Optional[str]) -> bool:
    return isinstance(s, str) and s.strip() != ""

def _is_iso_date(s: Optional[str]) -> bool:
    return bool(s and ISO_DATE_RE.match(s))

def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not _is_iso_date(s):
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None

def _duration_days(start: Optional[str], end: Optional[str]) -> Optional[int]:
    ds = _parse_date(start)
    de = _parse_date(end)
    if not ds or not de:
        return None
    return (de - ds).days + 1

def _address_is_complete(addr: VenueAddress) -> bool:
    if addr is None:
        return False
    street_ok = _is_nonempty_str(addr.street_address) and any(ch.isdigit() for ch in addr.street_address or "")
    city_ok = _is_nonempty_str(addr.city)
    sp_ok = _is_nonempty_str(addr.state_province)
    pc_ok = _is_nonempty_str(addr.postal_code)
    country_ok = _is_nonempty_str(addr.country)
    return street_ok and city_ok and sp_ok and pc_ok and country_ok

def _effective_conference_count(items: List[ConferenceItem]) -> int:
    """
    Count how many conferences are truly identified (having at least name, start_date, end_date, and venue_name).
    """
    count = 0
    for c in items:
        if _is_nonempty_str(c.conference_name) and _is_nonempty_str(c.venue_name) and _is_iso_date(c.start_date) and _is_iso_date(c.end_date):
            count += 1
    return count


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_conference(
    evaluator: Evaluator,
    parent_node,
    conf: ConferenceItem,
    index: int
) -> None:
    """
    Build verification subtree for a single conference entry (parallel aggregation).
    """
    conf_node = evaluator.add_parallel(
        id=f"Conference_{index+1}",
        desc=f"{index+1}st/nd/rd conference entry meets all criteria and required fields",
        parent=parent_node,
        critical=False  # allow partial credit per conference
    )

    # 1. Official Name Provided (critical)
    evaluator.add_custom_node(
        result=_is_nonempty_str(conf.conference_name),
        id=f"C{index+1}_Official_Name_Provided",
        desc="Provides the complete official conference name.",
        parent=conf_node,
        critical=True
    )

    # 2. Dates Provided in YYYY-MM-DD (critical)
    evaluator.add_custom_node(
        result=_is_iso_date(conf.start_date) and _is_iso_date(conf.end_date),
        id=f"C{index+1}_Dates_Provided_YYYY_MM_DD",
        desc="Provides a full start-to-end date range in YYYY-MM-DD format.",
        parent=conf_node,
        critical=True
    )

    # 3. Venue Name Provided (critical)
    evaluator.add_custom_node(
        result=_is_nonempty_str(conf.venue_name),
        id=f"C{index+1}_Venue_Name_Provided",
        desc="Provides the official name of the convention center venue.",
        parent=conf_node,
        critical=True
    )

    # 4. Venue Address Complete (critical)
    evaluator.add_custom_node(
        result=_address_is_complete(conf.venue_address),
        id=f"C{index+1}_Venue_Address_Complete",
        desc="Provides the complete venue street address including street number, street name, city, state/province, postal code, and country.",
        parent=conf_node,
        critical=True
    )

    # 5. Location US or Canada (critical) - verify against sources
    location_node = evaluator.add_leaf(
        id=f"C{index+1}_Location_US_or_Canada",
        desc="Conference is held in North America (United States or Canada).",
        parent=conf_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The conference venue '{conf.venue_name or ''}' is located in the United States or Canada."
        ),
        node=location_node,
        sources=conf.source_urls,
        additional_instruction=(
            "Determine whether the venue is in the US or Canada using official pages. "
            "Allow common variants like 'USA', 'U.S.', state names, Canadian provinces, etc."
        )
    )

    # 6. Date Window June–December 2025 inclusive (critical) - verify against sources
    window_node = evaluator.add_leaf(
        id=f"C{index+1}_Date_Window_Jun_to_Dec_2025",
        desc="Conference dates fall between June 2025 and December 2025 inclusive.",
        parent=conf_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "The official schedule places the conference between 2025-06-01 and 2025-12-31, inclusive."
        ),
        node=window_node,
        sources=conf.source_urls,
        additional_instruction=(
            "Check the official conference schedule or program page for the full date range. "
            "Tutorials/workshops included in the published program count toward the range."
        )
    )

    # 7. Duration at least 5 consecutive days (critical) - compute from provided dates
    dur = _duration_days(conf.start_date, conf.end_date)
    evaluator.add_custom_node(
        result=(dur is not None and dur >= 5),
        id=f"C{index+1}_Duration_At_Least_5_Consecutive_Days",
        desc="Conference spans at least 5 consecutive days.",
        parent=conf_node,
        critical=True
    )

    # 8. Established, recurring AI/ML/CV/NIPs focus (critical) - verify against sources
    recurring_node = evaluator.add_leaf(
        id=f"C{index+1}_Established_Recurring_AI_ML_Focus",
        desc="Conference is an established, recurring research conference series focused on AI, machine learning, computer vision, or neural information processing.",
        parent=conf_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "This conference is an established, recurring research conference series focused on artificial intelligence, "
            "machine learning, computer vision, or neural information processing."
        ),
        node=recurring_node,
        sources=conf.source_urls,
        additional_instruction=(
            "Look for evidence such as multiple past years, historical overview, and explicit research focus in AI/ML/CV/NIPs."
        )
    )

    # 9. In-person at a physical convention center (critical) - verify against sources
    inperson_node = evaluator.add_leaf(
        id=f"C{index+1}_In_Person_Convention_Center",
        desc="Conference is held in-person at a physical convention center venue (not virtual-only).",
        parent=conf_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            f"The conference is held in person at the physical convention center '{conf.venue_name or ''}'."
        ),
        node=inperson_node,
        sources=conf.source_urls,
        additional_instruction=(
            "Confirm that the event occurs physically at the named convention center. "
            "Hybrid is acceptable as long as there is an in-person component at the venue."
        )
    )

    # 10. Peer-reviewed paper presentations as core component (critical) - verify against sources
    peer_node = evaluator.add_leaf(
        id=f"C{index+1}_Peer_Reviewed_Paper_Presentations_Core",
        desc="Peer-reviewed paper presentations are a core component of the conference.",
        parent=conf_node,
        critical=True
    )
    await evaluator.verify(
        claim=(
            "Peer-reviewed paper presentations (papers accepted through a review process) are a core component of this conference."
        ),
        node=peer_node,
        sources=conf.source_urls,
        additional_instruction=(
            "Look for 'call for papers', 'peer review', 'accepted papers', 'program/technical papers' indicating reviewed papers are central."
        )
    )

    # 11. Official source URLs provided (critical) - existence check
    evaluator.add_custom_node(
        result=(isinstance(conf.source_urls, list) and len(conf.source_urls) > 0),
        id=f"C{index+1}_Official_Source_URLs",
        desc="Provides URLs to official conference and/or official venue websites sufficient to verify the stated facts.",
        parent=conf_node,
        critical=True
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2025 North America AI/ML conferences task using the Mind2Web2 evaluator.
    """
    # Initialize evaluator (root is non-critical to allow partial scoring across conferences)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=(
            "Identify exactly three qualifying peer-reviewed AI/ML research conferences held in North America between "
            "June 2025 and December 2025 inclusive, and provide required official details for each."
        ),
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract all conferences mentioned in the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferenceExtraction,
        extraction_name="conferences_extraction"
    )

    # Count Exactly Three (critical) — based on effective identified conferences in the answer
    total_identified = _effective_conference_count(extraction.conferences)
    evaluator.add_custom_node(
        result=(total_identified == 3),
        id="Count_Exactly_Three",
        desc="Exactly three conferences are identified (no fewer, no more).",
        parent=root,
        critical=True
    )

    # Prepare the three entries for verification: take the first three; pad if fewer
    selected: List[ConferenceItem] = list(extraction.conferences[:3])
    while len(selected) < 3:
        selected.append(ConferenceItem())

    # Build verification subtrees for each of the three conference entries
    for i in range(3):
        await verify_conference(evaluator, root, selected[i], i)

    # Return standardized summary
    return evaluator.get_summary()