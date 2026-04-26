import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_multi_genre_festivals_2025"
TASK_DESCRIPTION = (
    "A music industry publication is preparing a feature article on major multi-genre music festivals across the United States during the summer and fall of 2025. "
    "Identify 4 different major multi-genre music festivals in the United States that take place between June 1 and October 31, 2025, with each festival located in a different U.S. state. "
    "For each festival, provide: (1) The official festival name, (2) The exact dates of the festival (must be at least 2 consecutive days), (3) The host city and specific venue name, "
    "(4) At least two headlining artists from the 2025 lineup, (5) Evidence demonstrating the festival features at least 3 different music genres, and (6) A reference URL from the official festival website or a credible news source. "
    "The festivals should represent major events that attract large audiences and feature diverse lineups spanning multiple genres such as rock, pop, electronic, hip-hop, country, indie, or alternative music."
)

DATE_WINDOW_START = "2025-06-01"
DATE_WINDOW_END = "2025-10-31"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FestivalItem(BaseModel):
    name: Optional[str] = None
    # Prefer ISO format dates (YYYY-MM-DD). If unavailable, put best-effort string.
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    dates_text: Optional[str] = None  # Raw textual date expression (e.g., "Aug 2–4, 2025")
    city: Optional[str] = None
    state: Optional[str] = None  # Prefer full state name (e.g., "California") or postal code (e.g., "CA")
    venue: Optional[str] = None
    headliners: List[str] = Field(default_factory=list)
    genres_claimed: List[str] = Field(default_factory=list)  # e.g., ["rock", "pop", "hip-hop"]
    reference_urls: List[str] = Field(default_factory=list)  # official site and/or credible news links


class FestivalsExtraction(BaseModel):
    festivals: List[FestivalItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_festivals() -> str:
    return """
    Extract up to four (4) different major multi-genre U.S. music festivals described in the answer. For each festival, return:
    - name: The official festival name as stated in the answer (string).
    - start_date: The 2025 start date in ISO format YYYY-MM-DD if explicitly provided or can be unambiguously inferred from the answer (otherwise null).
    - end_date: The 2025 end date in ISO format YYYY-MM-DD if explicitly provided or can be unambiguously inferred from the answer (otherwise null).
    - dates_text: The exact date expression as written in the answer (e.g., "Aug 2–4, 2025" or "September 13-15, 2025"). Keep as a single string; do not invent.
    - city: Host city (string) as written in the answer (or null if missing).
    - state: U.S. state (full name or two-letter postal code) as written in the answer (or null if missing).
    - venue: Specific venue name (e.g., "Grant Park", "Golden Gate Park") as written (or null if missing).
    - headliners: At least two headlining artists from the 2025 lineup as listed in the answer (array of strings; if fewer than two provided, include those mentioned).
    - genres_claimed: Genres explicitly associated with the festival in the answer (e.g., ["rock", "pop", "hip-hop", "electronic", "country", "indie", "alternative"]). Include up to 5 key genres if they appear or are clearly implied in the answer. Empty array if none.
    - reference_urls: All URLs cited in the answer for this festival (official festival site and/or credible news sources). Return only URLs explicitly present in the answer text. If none, return an empty array.

    Notes:
    - Do NOT add, infer, or fabricate any fields. Extract exactly from the answer.
    - If multiple festivals are listed, keep the original order from the answer and extract the first four.
    - Keep strings verbatim (do not normalize city/state/venue spellings beyond trimming whitespace).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
_US_STATE_CODES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia", "PR": "Puerto Rico"
}


def canonicalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s_clean = s.strip()
    if not s_clean:
        return None
    up = s_clean.upper()
    if up in _US_STATE_CODES:
        return _US_STATE_CODES[up]
    # Title-case words (e.g., "new york" -> "New York")
    return " ".join(w.capitalize() for w in re.split(r"\s+", s_clean))


def canonicalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = re.sub(r"[^A-Za-z0-9]+", "", name).lower()
    return s or None


def ensure_len(lst: List[FestivalItem], k: int) -> List[FestivalItem]:
    padded = list(lst[:k])
    while len(padded) < k:
        padded.append(FestivalItem())
    return padded


def get_two_headliners(headliners: List[str]) -> List[str]:
    return [h for h in headliners if h and h.strip()][:2]


# --------------------------------------------------------------------------- #
# Verification per-festival                                                   #
# --------------------------------------------------------------------------- #
async def verify_single_festival(
    evaluator: Evaluator,
    parent_node,
    fest: FestivalItem,
    idx: int,
) -> None:
    """
    Build and verify all required leaf checks for one festival under its parent parallel node.
    """
    # Festival node (parallel; non-critical to allow partial credit across festivals)
    fest_node = evaluator.add_parallel(
        id=f"festival_{idx+1}",
        desc=f"Festival #{idx+1} (evaluate required fields and constraints)",
        parent=parent_node,
        critical=False,
    )

    urls = fest.reference_urls or []

    # Pre-create all leaf nodes as per rubric (IDs must match rubric)
    name_node = evaluator.add_leaf(
        id=f"festival_{idx+1}_name",
        desc="Provide the official festival name",
        parent=fest_node,
        critical=True,
    )
    dates_exact_node = evaluator.add_leaf(
        id=f"festival_{idx+1}_dates_exact",
        desc="Provide the exact festival dates",
        parent=fest_node,
        critical=True,
    )
    dates_in_window_node = evaluator.add_leaf(
        id=f"festival_{idx+1}_dates_in_window",
        desc="Festival dates occur between June 1 and Oct 31, 2025",
        parent=fest_node,
        critical=True,
    )
    duration_node = evaluator.add_leaf(
        id=f"festival_{idx+1}_duration_consecutive",
        desc="Festival spans at least 2 consecutive days",
        parent=fest_node,
        critical=True,
    )
    city_node = evaluator.add_leaf(
        id=f"festival_{idx+1}_city",
        desc="Provide the host city",
        parent=fest_node,
        critical=True,
    )
    state_node = evaluator.add_leaf(
        id=f"festival_{idx+1}_state",
        desc="Provide the U.S. state",
        parent=fest_node,
        critical=True,
    )
    venue_node = evaluator.add_leaf(
        id=f"festival_{idx+1}_venue",
        desc="Provide the specific venue name",
        parent=fest_node,
        critical=True,
    )
    headliners_node = evaluator.add_leaf(
        id=f"festival_{idx+1}_headliners",
        desc="Identify at least two headlining artists from the 2025 lineup",
        parent=fest_node,
        critical=True,
    )
    genres_node = evaluator.add_leaf(
        id=f"festival_{idx+1}_genre_diversity_evidence",
        desc="Provide evidence demonstrating the festival features at least 3 different music genres",
        parent=fest_node,
        critical=True,
    )
    major_event_node = evaluator.add_leaf(
        id=f"festival_{idx+1}_major_event_evidence",
        desc="Provide evidence supporting that the festival is a major event that attracts a large audience (e.g., attendance figures, capacity, or credible description of being large/major)",
        parent=fest_node,
        critical=True,
    )
    # Reference URL presence (critical). Implemented as a custom gating node.
    reference_url_node = evaluator.add_custom_node(
        result=bool(urls),
        id=f"festival_{idx+1}_reference_url",
        desc="Provide a reference URL from the official festival website or a credible news source",
        parent=fest_node,
        critical=True,
    )

    # Now conduct verifications, gating on the presence of at least one reference URL
    # 1) Name
    name_txt = fest.name or ""
    await evaluator.verify(
        claim=f"The official 2025 festival name is '{name_txt}'. Allow minor stylistic variations (e.g., punctuation/capitalization).",
        node=name_node,
        sources=urls,
        additional_instruction="Confirm the festival’s official name for the 2025 edition on the provided page(s). Minor stylistic differences are acceptable.",
        extra_prerequisites=[reference_url_node],
    )

    # 2) Exact dates
    sd = fest.start_date or (fest.dates_text or "")
    ed = fest.end_date or (fest.dates_text or "")
    # If dates_text only is available, the LLM should confirm those exact dates; otherwise prefer ISO start/end.
    dates_claim = (
        f"The 2025 festival dates are from '{fest.start_date}' to '{fest.end_date}'."
        if fest.start_date and fest.end_date
        else f"The 2025 festival dates are: '{fest.dates_text}'."
    )
    await evaluator.verify(
        claim=dates_claim,
        node=dates_exact_node,
        sources=urls,
        additional_instruction="Verify the exact 2025 dates stated for this festival on the provided page(s). If a range is provided (e.g., Aug 2–4, 2025), confirm it matches.",
        extra_prerequisites=[reference_url_node],
    )

    # 3) Dates fall in required window (logic check; inclusive)
    window_claim = (
        f"The festival’s 2025 dates ('{sd}' to '{ed}') occur between June 1, 2025 and October 31, 2025, inclusive."
    )
    await evaluator.verify(
        claim=window_claim,
        node=dates_in_window_node,
        additional_instruction="Treat the check as a logical evaluation of the given dates. Inclusive bounds: June 1, 2025 <= dates <= October 31, 2025.",
        extra_prerequisites=[reference_url_node, dates_exact_node],
    )

    # 4) Duration is at least 2 consecutive days (logic check)
    duration_claim = (
        f"The date range from '{sd}' to '{ed}' covers at least two consecutive days."
    )
    await evaluator.verify(
        claim=duration_claim,
        node=duration_node,
        additional_instruction="Interpret the provided start and end as a continuous festival period. Verify that the duration is >= 2 consecutive days.",
        extra_prerequisites=[reference_url_node, dates_exact_node],
    )

    # 5) City
    city_claim = f"The host city for the festival’s 2025 edition is '{fest.city or ''}'."
    await evaluator.verify(
        claim=city_claim,
        node=city_node,
        sources=urls,
        additional_instruction="Confirm the host city on the provided page(s). Allow reasonable variants like metropolitan area references if clearly equivalent.",
        extra_prerequisites=[reference_url_node],
    )

    # 6) State
    state_claim = f"The 2025 festival takes place in the U.S. state of '{fest.state or ''}'."
    await evaluator.verify(
        claim=state_claim,
        node=state_node,
        sources=urls,
        additional_instruction="Confirm the U.S. state on the provided page(s). Postal abbreviations and full names should be treated as equivalent.",
        extra_prerequisites=[reference_url_node],
    )

    # 7) Venue
    venue_claim = f"The specific venue for the 2025 festival is '{fest.venue or ''}'."
    await evaluator.verify(
        claim=venue_claim,
        node=venue_node,
        sources=urls,
        additional_instruction="Confirm the named venue (e.g., park, fairgrounds, stadium) for the 2025 festival on the provided page(s).",
        extra_prerequisites=[reference_url_node],
    )

    # 8) Headliners (at least two)
    two_heads = get_two_headliners(fest.headliners)
    if len(two_heads) >= 2:
        headliner_claim = f"The 2025 lineup includes at least two headliners: {two_heads}."
    else:
        # If fewer than two extracted, still form a claim (likely to fail)
        headliner_claim = f"The 2025 lineup includes at least two headliners: {fest.headliners}."
    await evaluator.verify(
        claim=headliner_claim,
        node=headliners_node,
        sources=urls,
        additional_instruction="Verify on the provided page(s) that the listed artists are headliners/top-billed for the 2025 edition. Minor naming variants acceptable.",
        extra_prerequisites=[reference_url_node],
    )

    # 9) Genre diversity evidence (>= 3 distinct genres)
    genres_list = fest.genres_claimed or []
    if genres_list:
        genre_claim = f"This festival features at least three different music genres. For example, the answer lists genres: {genres_list}."
    else:
        genre_claim = "This festival features at least three different music genres across its 2025 programming."
    await evaluator.verify(
        claim=genre_claim,
        node=genres_node,
        sources=urls,
        additional_instruction=(
            "On the provided page(s), look for explicit or strongly implied coverage of at least three distinct genres "
            "(e.g., rock, pop, electronic/EDM, hip-hop/rap, country, indie, alternative). Evidence may include lineup diversity or official descriptions."
        ),
        extra_prerequisites=[reference_url_node],
    )

    # 10) Major event evidence
    major_claim = (
        "This festival is a major event that attracts a large audience (e.g., tens of thousands) or is credibly described as large/major."
    )
    await evaluator.verify(
        claim=major_claim,
        node=major_event_node,
        sources=urls,
        additional_instruction=(
            "On the provided page(s), look for attendance figures, capacity, or credible descriptions like 'major', 'one of the largest', 'draws X attendees', etc."
        ),
        extra_prerequisites=[reference_url_node],
    )


# --------------------------------------------------------------------------- #
# Cross-festival checks                                                       #
# --------------------------------------------------------------------------- #
def compute_festival_distinctness(fests: List[FestivalItem]) -> bool:
    names = [canonicalize_name(f.name) for f in fests]
    if any(n is None for n in names):
        return False
    return len(set(names)) == 4


def compute_state_uniqueness(fests: List[FestivalItem]) -> bool:
    states = [canonicalize_state(f.state) for f in fests]
    if any(s is None for s in states):
        return False
    return len(set(states)) == 4


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the 2025 U.S. multi-genre festivals task.
    """
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

    # Extract festivals from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_festivals(),
        template_class=FestivalsExtraction,
        extraction_name="festivals_extraction",
    )

    # Keep only the first 4; pad if fewer
    festivals = ensure_len(extracted.festivals, 4)

    # Build festival nodes container at root (parallel children are added directly to root)
    # For each of the 4 festivals
    for i in range(4):
        await verify_single_festival(evaluator, root, festivals[i], i)

    # Cross-festival checks (critical parallel node)
    cross_node = evaluator.add_parallel(
        id="cross_festival_checks",
        desc="Cross-festival constraints across the set of 4 festivals",
        parent=root,
        critical=True,
    )

    # Distinct festival names (critical)
    distinct_ok = compute_festival_distinctness(festivals)
    evaluator.add_custom_node(
        result=distinct_ok,
        id="festival_distinctness",
        desc="Verify the response identifies 4 different festivals (no duplicates)",
        parent=cross_node,
        critical=True,
    )

    # Unique states (critical)
    unique_states_ok = compute_state_uniqueness(festivals)
    evaluator.add_custom_node(
        result=unique_states_ok,
        id="state_uniqueness",
        desc="Verify that all 4 festivals are located in different U.S. states",
        parent=cross_node,
        critical=True,
    )

    # Return structured summary
    return evaluator.get_summary()