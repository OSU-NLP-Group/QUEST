import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task constants
# -----------------------------------------------------------------------------
TASK_ID = "awards_festivals_early_2026"
TASK_DESCRIPTION = """
I am planning to attend major entertainment awards ceremonies and film festivals in early 2026. Please provide detailed information about the following four events:

1. The 83rd Golden Globes
2. The 68th Grammy Awards
3. The 76th Berlin International Film Festival (Berlinale)
4. The 79th Cannes Film Festival

For each event, provide:
- The exact date or date range when the event takes place
- The city and country/state where it is held
- The specific venue name
- Additional key information: for awards ceremonies, specify the broadcast network(s); for film festivals, identify the jury president (if announced)
"""

# Expected values from the rubric (used for value-match checks)
EXPECTED = {
    "golden_globes": {
        "date": "January 11, 2026",
        "location": "Beverly Hills, California, U.S.",
        "venue": "The Beverly Hilton",
        "broadcast": "CBS and Paramount+ (streaming)",
    },
    "grammys": {
        "date": "February 1, 2026",
        "location": "Los Angeles, California, U.S.",
        "venue": "Crypto.com Arena",
        "broadcast": "CBS and Paramount+ (streaming)",
    },
    "berlinale": {
        "dates": "February 12 to 22, 2026",
        "location": "Berlin, Germany",
        # venue: any specific venue name is acceptable (must be provided + source-backed)
        # jury: optional (if not yet announced, that is also acceptable, with source)
    },
    "cannes": {
        "dates": "May 12 to 23, 2026",
        "location": "Cannes, France",
        "venue": "Palais des Festivals et des Congrès",
        "jury_president": "Park Chan-wook",
    }
}


# -----------------------------------------------------------------------------
# Pydantic models for extraction
# -----------------------------------------------------------------------------
class AwardEventInfo(BaseModel):
    date: Optional[str] = None
    date_sources: List[str] = Field(default_factory=list)

    location: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)

    venue: Optional[str] = None
    venue_sources: List[str] = Field(default_factory=list)

    broadcast: Optional[str] = None
    broadcast_sources: List[str] = Field(default_factory=list)


class FestivalEventInfo(BaseModel):
    dates: Optional[str] = None
    dates_sources: List[str] = Field(default_factory=list)

    location: Optional[str] = None
    location_sources: List[str] = Field(default_factory=list)

    venue: Optional[str] = None
    venue_sources: List[str] = Field(default_factory=list)

    jury_president: Optional[str] = None  # If not announced, use phrases like "TBA", "not announced", etc.
    jury_sources: List[str] = Field(default_factory=list)


class EventsExtraction(BaseModel):
    golden_globes: Optional[AwardEventInfo] = None
    grammys: Optional[AwardEventInfo] = None
    berlin: Optional[FestivalEventInfo] = None
    cannes: Optional[FestivalEventInfo] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_events() -> str:
    return """
Extract the requested structured information for the following four events exactly as stated in the provided answer text. Do NOT invent or infer any information not clearly present in the answer.

Events to extract:
1) 83rd Golden Globes (awards ceremony)
2) 68th Grammy Awards (awards ceremony)
3) 76th Berlin International Film Festival (Berlinale) (film festival)
4) 79th Cannes Film Festival (film festival)

For each event, extract the following fields, using EXACT strings as they appear in the answer:
- For awards (Golden Globes, Grammys):
  • date: The specific ceremony date (e.g., "January 11, 2026")
  • date_sources: A list of all URLs in the answer that support the date (extract actual URLs; if none, return [])
  • location: The city and state/country (e.g., "Beverly Hills, California, U.S.")
  • location_sources: URLs supporting the location
  • venue: The specific venue name (e.g., "The Beverly Hilton")
  • venue_sources: URLs supporting the venue
  • broadcast: The broadcast network(s) information as given (e.g., "CBS and Paramount+ (streaming)")
  • broadcast_sources: URLs supporting the broadcast information

- For festivals (Berlin, Cannes):
  • dates: The date range (e.g., "February 12 to 22, 2026")
  • dates_sources: URLs supporting the dates
  • location: The city and country (e.g., "Berlin, Germany")
  • location_sources: URLs supporting the location
  • venue: The specific venue (for Cannes it is commonly the Palais des Festivals; for Berlin, extract any specific venue name given)
  • venue_sources: URLs supporting the venue
  • jury_president: If the jury president is mentioned, extract the name; if the answer explicitly states it is not announced yet, extract that phrase (e.g., "not announced", "TBA"). If not mentioned at all, return null.
  • jury_sources: URLs supporting the jury president information (or supporting that it is not yet announced); if none, return []

Return a single JSON object with the following top-level fields:
{
  "golden_globes": {...},   // AwardEventInfo or null
  "grammys": {...},         // AwardEventInfo or null
  "berlin": {...},          // FestivalEventInfo or null
  "cannes": {...}           // FestivalEventInfo or null
}

Rules:
- Only include URLs that are explicitly present in the answer (plain URLs or in markdown).
- If any requested field is missing in the answer, set it to null; for lists of URLs, return [] if none.
- Preserve the original wording/formatting from the answer for string fields.
"""


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def _s(s: Optional[str]) -> str:
    return s or ""


def _lst(xs: Optional[List[str]]) -> List[str]:
    return xs if xs is not None else []


def _is_not_announced(text: Optional[str]) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    keywords = ["tba", "tbd", "not announced", "not yet announced", "unannounced", "pending", "to be announced"]
    return any(k in t for k in keywords)


# -----------------------------------------------------------------------------
# Reusable verification helpers
# -----------------------------------------------------------------------------
async def add_value_and_source_checks(
    evaluator: Evaluator,
    parent,
    *,
    id_prefix: str,
    group_desc: str,
    critical_group: bool,
    expected_value: Optional[str],
    extracted_value: Optional[str],
    sources: Optional[List[str]],
    value_desc_for_simple_check: str,
    url_claim_template: str,
    url_additional_instruction: str,
    value_equivalence_instruction: str,
):
    """
    Add a critical/non-critical group with:
      - value equivalence check against expected_value (if expected_value is provided)
      - sources existence check (critical if group is critical)
      - URL-backed verification that the claim (using the extracted value) is supported by the cited sources

    The URL-backed claim must be phrased using the EXTRACTED value to ensure the answer's claim is actually supported.
    """
    group_node = evaluator.add_parallel(
        id=id_prefix,
        desc=group_desc,
        parent=parent,
        critical=critical_group
    )

    # 1) Value equivalence (only if we have an expected value to compare against)
    if expected_value is not None:
        value_leaf = evaluator.add_leaf(
            id=f"{id_prefix}_value_match",
            desc=f"{value_desc_for_simple_check} matches expected",
            parent=group_node,
            critical=True if critical_group else False
        )
        await evaluator.verify(
            claim=f"The answer's value '{_s(extracted_value)}' means the same as '{expected_value}'.",
            node=value_leaf,
            additional_instruction=value_equivalence_instruction
        )

    # 2) Sources existence (critical within the group when group is critical; otherwise non-critical)
    srcs = _lst(sources)
    src_exist_leaf = evaluator.add_custom_node(
        result=(len(srcs) > 0),
        id=f"{id_prefix}_sources_provided",
        desc=f"At least one source URL is provided for {value_desc_for_simple_check}",
        parent=group_node,
        critical=True if critical_group else False
    )

    # 3) URL-backed verification using EXTRACTED value
    url_leaf = evaluator.add_leaf(
        id=f"{id_prefix}_url",
        desc=f"URL reference confirming {value_desc_for_simple_check}",
        parent=group_node,
        critical=True if critical_group else False
    )
    url_claim = url_claim_template.format(value=_s(extracted_value))
    await evaluator.verify(
        claim=url_claim,
        node=url_leaf,
        sources=srcs,
        additional_instruction=url_additional_instruction
    )


# -----------------------------------------------------------------------------
# Event-specific verification
# -----------------------------------------------------------------------------
async def verify_golden_globes(evaluator: Evaluator, root, info: Optional[AwardEventInfo]):
    node = evaluator.add_parallel(
        id="golden_globes_2026",
        desc="Information about the 83rd Golden Globes",
        parent=root,
        critical=False
    )
    data = info or AwardEventInfo()

    # Date
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="golden_globes_date",
        group_desc="The ceremony date is January 11, 2026",
        critical_group=True,
        expected_value=EXPECTED["golden_globes"]["date"],
        extracted_value=data.date,
        sources=data.date_sources,
        value_desc_for_simple_check="Golden Globes 2026 date",
        url_claim_template="The 83rd Golden Globes ceremony takes place on {value}.",
        url_additional_instruction="Verify the page explicitly states the 83rd Golden Globes (2026) date as specified. Minor formatting (e.g., Jan vs January) is OK only if it represents the same date.",
        value_equivalence_instruction="Allow date-format variations (e.g., Jan vs January, presence/absence of weekday) as long as they refer to January 11, 2026."
    )

    # Location
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="golden_globes_location",
        group_desc="The location is Beverly Hills, California, U.S.",
        critical_group=True,
        expected_value=EXPECTED["golden_globes"]["location"],
        extracted_value=data.location,
        sources=data.location_sources,
        value_desc_for_simple_check="Golden Globes 2026 location",
        url_claim_template="The 83rd Golden Globes ceremony takes place in {value}.",
        url_additional_instruction="Check that the location refers to Beverly Hills, California, United States (allow common abbreviations like CA, U.S., USA).",
        value_equivalence_instruction="Treat equivalents like 'Beverly Hills, CA', 'Beverly Hills, California, USA', and 'Beverly Hills, California, U.S.' as the same location."
    )

    # Venue
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="golden_globes_venue",
        group_desc="The venue is The Beverly Hilton",
        critical_group=True,
        expected_value=EXPECTED["golden_globes"]["venue"],
        extracted_value=data.venue,
        sources=data.venue_sources,
        value_desc_for_simple_check="Golden Globes 2026 venue",
        url_claim_template="The 83rd Golden Globes ceremony is held at {value}.",
        url_additional_instruction="Confirm the page identifies The Beverly Hilton as the venue for the 2026 Golden Globes.",
        value_equivalence_instruction="Allow minor variations like 'The Beverly Hilton Hotel' vs 'The Beverly Hilton'."
    )

    # Broadcast
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="golden_globes_broadcast",
        group_desc="The broadcast networks are CBS and Paramount+ (streaming)",
        critical_group=True,
        expected_value=EXPECTED["golden_globes"]["broadcast"],
        extracted_value=data.broadcast,
        sources=data.broadcast_sources,
        value_desc_for_simple_check="Golden Globes 2026 broadcast networks",
        url_claim_template="The 83rd Golden Globes will air on/through {value}.",
        url_additional_instruction="Verify the page indicates CBS as the broadcast network and Paramount+ as the streaming platform for the 2026 Golden Globes.",
        value_equivalence_instruction="Consider the value matching if it clearly indicates CBS (TV) and Paramount+ (streaming), allowing small wording differences."
    )


async def verify_grammys(evaluator: Evaluator, root, info: Optional[AwardEventInfo]):
    node = evaluator.add_parallel(
        id="grammy_awards_2026",
        desc="Information about the 68th Grammy Awards",
        parent=root,
        critical=False
    )
    data = info or AwardEventInfo()

    # Date
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="grammy_date",
        group_desc="The ceremony date is February 1, 2026",
        critical_group=True,
        expected_value=EXPECTED["grammys"]["date"],
        extracted_value=data.date,
        sources=data.date_sources,
        value_desc_for_simple_check="Grammy Awards 2026 date",
        url_claim_template="The 68th Annual Grammy Awards ceremony takes place on {value}.",
        url_additional_instruction="Verify the page explicitly states the 2026 Grammy ceremony date as specified.",
        value_equivalence_instruction="Allow common date-format variations if they still represent February 1, 2026."
    )

    # Location
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="grammy_location",
        group_desc="The location is Los Angeles, California, U.S.",
        critical_group=True,
        expected_value=EXPECTED["grammys"]["location"],
        extracted_value=data.location,
        sources=data.location_sources,
        value_desc_for_simple_check="Grammy Awards 2026 location",
        url_claim_template="The 68th Annual Grammy Awards ceremony takes place in {value}.",
        url_additional_instruction="Check that the location corresponds to Los Angeles, California, United States (allow abbreviations like LA, CA, U.S., USA).",
        value_equivalence_instruction="Treat equivalents like 'Los Angeles, CA', 'Los Angeles, California, U.S.' etc. as the same."
    )

    # Venue
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="grammy_venue",
        group_desc="The venue is Crypto.com Arena",
        critical_group=True,
        expected_value=EXPECTED["grammys"]["venue"],
        extracted_value=data.venue,
        sources=data.venue_sources,
        value_desc_for_simple_check="Grammy Awards 2026 venue",
        url_claim_template="The 68th Annual Grammy Awards ceremony is held at {value}.",
        url_additional_instruction="Confirm the page indicates Crypto.com Arena as the 2026 Grammy venue.",
        value_equivalence_instruction="Minor naming variations like 'Crypto.com Arena (formerly Staples Center)' are acceptable if clearly the same venue."
    )

    # Broadcast
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="grammy_broadcast",
        group_desc="The broadcast networks are CBS and Paramount+ (streaming)",
        critical_group=True,
        expected_value=EXPECTED["grammys"]["broadcast"],
        extracted_value=data.broadcast,
        sources=data.broadcast_sources,
        value_desc_for_simple_check="Grammy Awards 2026 broadcast networks",
        url_claim_template="The 68th Annual Grammy Awards will air on/through {value}.",
        url_additional_instruction="Verify CBS as the TV network and Paramount+ as the streaming platform for the 2026 Grammys.",
        value_equivalence_instruction="Consider it a match if both CBS and Paramount+ are clearly indicated (wording variations OK)."
    )


async def verify_berlinale(evaluator: Evaluator, root, info: Optional[FestivalEventInfo]):
    node = evaluator.add_parallel(
        id="berlin_film_festival_2026",
        desc="Information about the 76th Berlin International Film Festival",
        parent=root,
        critical=False
    )
    data = info or FestivalEventInfo()

    # Dates
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="berlin_dates",
        group_desc="The festival runs from February 12 to 22, 2026",
        critical_group=True,
        expected_value=EXPECTED["berlinale"]["dates"],
        extracted_value=data.dates,
        sources=data.dates_sources,
        value_desc_for_simple_check="Berlinale 2026 dates",
        url_claim_template="The 76th Berlin International Film Festival runs from {value}.",
        url_additional_instruction="Verify the date range for Berlinale 2026; allow minor formatting like en-dashes or abbreviated months if it's the same range.",
        value_equivalence_instruction="Consider 'Feb 12–22, 2026' as equivalent to 'February 12 to 22, 2026' if the range is the same."
    )

    # Location
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="berlin_location",
        group_desc="The location is Berlin, Germany",
        critical_group=True,
        expected_value=EXPECTED["berlinale"]["location"],
        extracted_value=data.location,
        sources=data.location_sources,
        value_desc_for_simple_check="Berlinale 2026 location",
        url_claim_template="The 76th Berlin International Film Festival takes place in {value}.",
        url_additional_instruction="Confirm that the festival is in Berlin, Germany.",
        value_equivalence_instruction="Allow minor variations like 'Berlin, DE' if it's clearly the same."
    )

    # Venue (any specific provided venue is acceptable; must be source-backed)
    venue_group = evaluator.add_parallel(
        id="berlin_venue",
        desc="A specific venue name is provided for the Berlin Film Festival",
        parent=node,
        critical=True
    )
    venue_exists = evaluator.add_custom_node(
        result=bool(_s(data.venue)),
        id="berlin_venue_exists",
        desc="A specific venue name is provided for Berlinale 2026",
        parent=venue_group,
        critical=True
    )
    venue_srcs = _lst(data.venue_sources)
    venue_src_exist = evaluator.add_custom_node(
        result=len(venue_srcs) > 0,
        id="berlin_venue_sources_provided",
        desc="At least one source URL is provided for the Berlinale 2026 venue",
        parent=venue_group,
        critical=True
    )
    venue_url = evaluator.add_leaf(
        id="berlin_venue_url",
        desc="URL reference confirming the Berlin Film Festival 2026 venue",
        parent=venue_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"{_s(data.venue)} is an official venue (or one of the official venues) used for the 76th Berlin International Film Festival (2026).",
        node=venue_url,
        sources=venue_srcs,
        additional_instruction="Accept that Berlinale uses multiple venues; confirm the cited venue is indeed an official festival venue in 2026."
    )

    # Jury president (non-critical): either a name with sources, or correctly noting not announced (with sources)
    jury_group = evaluator.add_parallel(
        id="berlin_jury_president",
        desc="Information about the jury president is provided (if announced) or it is noted that this information is not yet publicly announced",
        parent=node,
        critical=False
    )
    jury_srcs = _lst(data.jury_sources)
    # Whether any jury info is provided
    jury_info_present = evaluator.add_custom_node(
        result=bool(_s(data.jury_president)),
        id="berlin_jury_info_present",
        desc="Berlinale 2026 jury information is present in the answer (either a name or a 'not announced yet' note)",
        parent=jury_group,
        critical=False
    )
    # Jury sources existence (non-critical)
    jury_src_exist = evaluator.add_custom_node(
        result=len(jury_srcs) > 0,
        id="berlin_jury_sources_provided",
        desc="At least one source URL is provided for the Berlinale 2026 jury information",
        parent=jury_group,
        critical=False
    )
    jury_url = evaluator.add_leaf(
        id="berlin_jury_url",
        desc="URL reference for Berlin Film Festival 2026 jury information",
        parent=jury_group,
        critical=False
    )
    if _is_not_announced(data.jury_president):
        claim = "As of now, there has been no official announcement of the jury president for the 76th Berlin International Film Festival (2026)."
        add_ins = "Verify that the page indicates the jury president is not yet announced or is TBA."
    else:
        claim = f"The jury president for the 76th Berlin International Film Festival (2026) is {_s(data.jury_president)}."
        add_ins = "Verify that the page explicitly names this person as the jury president for Berlinale 2026."
    await evaluator.verify(
        claim=claim,
        node=jury_url,
        sources=jury_srcs,
        additional_instruction=add_ins
    )


async def verify_cannes(evaluator: Evaluator, root, info: Optional[FestivalEventInfo]):
    node = evaluator.add_parallel(
        id="cannes_film_festival_2026",
        desc="Information about the 79th Cannes Film Festival",
        parent=root,
        critical=False
    )
    data = info or FestivalEventInfo()

    # Dates
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="cannes_dates",
        group_desc="The festival runs from May 12 to 23, 2026",
        critical_group=True,
        expected_value=EXPECTED["cannes"]["dates"],
        extracted_value=data.dates,
        sources=data.dates_sources,
        value_desc_for_simple_check="Cannes 2026 dates",
        url_claim_template="The 79th Cannes Film Festival runs from {value}.",
        url_additional_instruction="Verify the date range for Cannes 2026; allow minor formatting differences if equivalent.",
        value_equivalence_instruction="Treat 'May 12–23, 2026' as equivalent to 'May 12 to 23, 2026' if the range is the same."
    )

    # Location
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="cannes_location",
        group_desc="The location is Cannes, France",
        critical_group=True,
        expected_value=EXPECTED["cannes"]["location"],
        extracted_value=data.location,
        sources=data.location_sources,
        value_desc_for_simple_check="Cannes 2026 location",
        url_claim_template="The 79th Cannes Film Festival takes place in {value}.",
        url_additional_instruction="Confirm that the festival is in Cannes, France.",
        value_equivalence_instruction="Minor variants like 'Cannes, FR' are okay if clearly equivalent."
    )

    # Venue
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="cannes_venue",
        group_desc="The venue is Palais des Festivals et des Congrès (Palais des Festivals)",
        critical_group=True,
        expected_value=EXPECTED["cannes"]["venue"],
        extracted_value=data.venue,
        sources=data.venue_sources,
        value_desc_for_simple_check="Cannes 2026 venue",
        url_claim_template="The 79th Cannes Film Festival is held at {value}.",
        url_additional_instruction="Verify the page indicates the Palais des Festivals et des Congrès as the main venue.",
        value_equivalence_instruction="Treat 'Palais des Festivals' as equivalent to 'Palais des Festivals et des Congrès' if clearly the same venue."
    )

    # Jury president (critical; expected Park Chan-wook)
    await add_value_and_source_checks(
        evaluator, node,
        id_prefix="cannes_jury_president",
        group_desc="The jury president is Park Chan-wook",
        critical_group=True,
        expected_value=EXPECTED["cannes"]["jury_president"],
        extracted_value=data.jury_president,
        sources=data.jury_sources,
        value_desc_for_simple_check="Cannes 2026 jury president",
        url_claim_template="The jury president for the 79th Cannes Film Festival (2026) is {value}.",
        url_additional_instruction="Confirm that Park Chan-wook is named as the 2026 Cannes jury president on the cited page(s).",
        value_equivalence_instruction="Allow minor transliteration or punctuation differences in the name if clearly the same person."
    )


# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
async def evaluate_answer(
    client: Any,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    # Initialize Evaluator (root is non-critical to allow partial credit across events)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Provide detailed information about four major entertainment events in early 2026",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_events(),
        template_class=EventsExtraction,
        extraction_name="events_extraction",
    )

    # Add ground truth (expected values per rubric) for transparency
    evaluator.add_ground_truth({
        "golden_globes_expected": EXPECTED["golden_globes"],
        "grammys_expected": EXPECTED["grammys"],
        "berlinale_expected": {"dates": EXPECTED["berlinale"]["dates"], "location": EXPECTED["berlinale"]["location"]},
        "cannes_expected": EXPECTED["cannes"],
    })

    # Build verification tree per event
    await verify_golden_globes(evaluator, root, extracted.golden_globes)
    await verify_grammys(evaluator, root, extracted.grammys)
    await verify_berlinale(evaluator, root, extracted.berlin)
    await verify_cannes(evaluator, root, extracted.cannes)

    # Return summary
    return evaluator.get_summary()