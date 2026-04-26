import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "ncaa_2026_first_second_round_hosts"
TASK_DESCRIPTION = (
    "Identify four host cities for the first and second rounds of the 2026 NCAA Division I Men's Basketball Tournament. "
    "For each city, provide: (1) the complete name of the host venue, (2) the seating capacity of the venue, "
    "(3) the name of the host institution (university or conference), (4) the specific dates when first and second round "
    "games will be played at that location (in the format 'March DD & DD'), and (5) a direct link to an official source "
    "(NCAA.com, the venue's official website, or the host institution's athletics website) that confirms the city is hosting "
    "2026 tournament games. All four cities must be located in different states."
)

DATE_PATTERN_STRICT = r"(?i)^March\s+\d{1,2}\s*&\s*\d{1,2}$"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CityEntry(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    venue_name: Optional[str] = None
    venue_capacity: Optional[str] = None
    host_institution: Optional[str] = None
    dates: Optional[str] = None
    source_url: Optional[str] = None


class CitiesExtraction(BaseModel):
    cities: List[CityEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_cities() -> str:
    return (
        "Extract the first four host city entries mentioned in the answer for the 2026 NCAA Division I Men's Basketball "
        "Tournament first and second rounds. For each entry, extract EXACTLY the following fields from the answer text:\n"
        "- city: The city name.\n"
        "- state: The U.S. state for the city (full name or two-letter abbreviation as presented in the answer).\n"
        "- venue_name: The complete name of the host venue.\n"
        "- venue_capacity: The seating capacity number as given in the answer (keep punctuation like commas if present).\n"
        "- host_institution: The host institution (university or conference).\n"
        "- dates: The specific dates for first and second round games (as presented). The required format is 'March DD & DD'. "
        "If the answer uses another phrasing, still extract it verbatim.\n"
        "- source_url: A direct URL to an official source (NCAA.com, the venue's official website, or a host institution's "
        "athletics website) that confirms the city is hosting 2026 tournament games.\n\n"
        "Rules:\n"
        "1) Only extract information explicitly present in the answer; do not invent or infer.\n"
        "2) If the answer provides more than four entries, return only the first four.\n"
        "3) If fewer than four entries are provided, still return as many as present; missing fields should be null.\n"
        "4) For URLs, extract the actual URL text (plain or in markdown link form). If missing, set to null.\n"
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def is_nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def has_digits(s: Optional[str]) -> bool:
    return bool(s and re.search(r"\d", s))


def is_valid_http_url(s: Optional[str]) -> bool:
    return bool(s and str(s).strip().lower().startswith(("http://", "https://")))


def matches_march_dates_format(s: Optional[str]) -> bool:
    if not s:
        return False
    return bool(re.match(DATE_PATTERN_STRICT, s.strip()))


def unique_nonnull_states(states: List[Optional[str]]) -> bool:
    vals = [st.strip() for st in states if is_nonempty(st)]
    if len(vals) != 4:
        return False
    return len(set(v.lower() for v in vals)) == 4


# --------------------------------------------------------------------------- #
# Verification per-city                                                       #
# --------------------------------------------------------------------------- #
async def verify_city(
    evaluator: Evaluator,
    parent_node,
    city: CityEntry,
    idx: int,
) -> None:
    """
    Build the verification subtree for a single city.
    - Adds critical existence/format checks as custom nodes (as per rubric).
    - Adds non-critical factual verification leaves grounded by the provided source URL where applicable.
    """
    city_node = evaluator.add_parallel(
        id=f"city_{idx+1}",
        desc=f"{['First','Second','Third','Fourth'][idx]} host city information",
        parent=parent_node,
        critical=False,
    )

    # Critical existence/format checks (per rubric)
    evaluator.add_custom_node(
        result=is_nonempty(city.venue_name),
        id=f"city_{idx+1}_venue_name",
        desc="Complete name of the host venue provided",
        parent=city_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=is_nonempty(city.venue_capacity) and has_digits(city.venue_capacity),
        id=f"city_{idx+1}_venue_capacity",
        desc="Seating capacity of the venue provided as a specific number",
        parent=city_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=is_nonempty(city.host_institution),
        id=f"city_{idx+1}_host_institution",
        desc="Name of the host institution (university or conference) provided",
        parent=city_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=is_nonempty(city.dates) and matches_march_dates_format(city.dates),
        id=f"city_{idx+1}_dates",
        desc="Specific dates for first and second round games provided in the format 'March DD & DD'",
        parent=city_node,
        critical=True,
    )
    evaluator.add_custom_node(
        result=is_valid_http_url(city.source_url),
        id=f"city_{idx+1}_source_url",
        desc="Direct link to official source confirming the city is hosting 2026 tournament games",
        parent=city_node,
        critical=True,
    )

    # Non-critical verification leaves grounded by the provided source URL
    # Note: These will be auto-skipped if the source_url (or other critical siblings) fails.
    verifications: List[Dict[str, Any]] = []

    # Verify that the source is official AND confirms hosting for 2026 first/second rounds
    source_confirm_node = evaluator.add_leaf(
        id=f"city_{idx+1}_source_confirms_hosting",
        desc="Official source confirms the city is hosting 2026 first/second round games",
        parent=city_node,
        critical=False,
    )
    claim_source_confirm = (
        f"This webpage is an official source (NCAA.com or an official venue/host institution athletics site) and it "
        f"confirms that {city.city or 'the city'}{', ' + city.state if is_nonempty(city.state) else ''} is hosting "
        f"first and second round games of the 2026 NCAA Division I Men's Basketball Tournament."
    )
    verifications.append((claim_source_confirm, city.source_url, source_confirm_node, "Ensure the page explicitly confirms 2026 hosting at this city. Consider officialness by domain and branding (NCAA.com, venue official site, or host institution athletics site)."))

    # Verify venue name
    venue_check_node = evaluator.add_leaf(
        id=f"city_{idx+1}_venue_name_check",
        desc="Venue name is supported by the provided official source",
        parent=city_node,
        critical=False,
    )
    claim_venue = (
        f"The host venue for the 2026 first/second round games in {city.city or 'the city'}{', ' + city.state if is_nonempty(city.state) else ''} "
        f"is '{city.venue_name or ''}'."
    )
    verifications.append((claim_venue, city.source_url, venue_check_node, "Verify the venue name appears or is clearly indicated as the host for this location."))

    # Verify seating capacity (may not always be on the same page; still attempt)
    capacity_check_node = evaluator.add_leaf(
        id=f"city_{idx+1}_venue_capacity_check",
        desc="Venue capacity is supported by the provided official source",
        parent=city_node,
        critical=False,
    )
    claim_capacity = (
        f"The seating capacity of '{city.venue_name or 'the venue'}' is {city.venue_capacity or ''}."
    )
    verifications.append((claim_capacity, city.source_url, capacity_check_node, "Confirm the capacity figure on the page; allow reasonable formatting variations like commas."))

    # Verify host institution
    host_inst_check_node = evaluator.add_leaf(
        id=f"city_{idx+1}_host_institution_check",
        desc="Host institution is supported by the provided official source",
        parent=city_node,
        critical=False,
    )
    claim_host_inst = (
        f"The host institution (university or conference) for {city.city or 'the city'}{', ' + city.state if is_nonempty(city.state) else ''} "
        f"is '{city.host_institution or ''}'."
    )
    verifications.append((claim_host_inst, city.source_url, host_inst_check_node, "Verify the host institution attribution (host/sponsor) is explicitly mentioned."))

    # Verify dates
    dates_check_node = evaluator.add_leaf(
        id=f"city_{idx+1}_dates_check",
        desc="Specific dates are supported by the provided official source",
        parent=city_node,
        critical=False,
    )
    claim_dates = (
        f"The first and second round games at {city.city or 'the city'}{', ' + city.state if is_nonempty(city.state) else ''} "
        f"will be played on {city.dates or ''} in March 2026."
    )
    verifications.append((claim_dates, city.source_url, dates_check_node, "Verify the page lists these specific March dates for first/second rounds at this location."))

    # Execute all verification leaves in parallel
    await evaluator.batch_verify(verifications)


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 2026 NCAA first/second round host cities task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel: each city evaluated independently
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

    # 1) Extract structured city info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_cities(),
        template_class=CitiesExtraction,
        extraction_name="extracted_cities",
    )

    # Filter/pad to exactly 4 entries
    cities: List[CityEntry] = list(extracted.cities[:4])
    while len(cities) < 4:
        cities.append(CityEntry())

    # 2) Critical global constraint: All four cities are in different states
    states_list = [c.state for c in cities]
    evaluator.add_custom_node(
        result=unique_nonnull_states(states_list),
        id="all_different_states",
        desc="All four selected cities are located in different states (no two cities from the same state)",
        parent=root,
        critical=True,
    )

    # 3) Per-city verification
    for i, city in enumerate(cities):
        await verify_city(evaluator, root, city, i)

    # 4) Return structured result
    # Add small custom info (e.g., date regex used)
    evaluator.add_custom_info(
        info={"date_format_regex": DATE_PATTERN_STRICT, "cities_count_evaluated": 4},
        info_type="config",
        info_name="verification_config",
    )

    return evaluator.get_summary()