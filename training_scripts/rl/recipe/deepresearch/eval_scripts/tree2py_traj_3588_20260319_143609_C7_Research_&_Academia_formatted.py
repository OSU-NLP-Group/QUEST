import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# ------------------------------------------------------------------------------
# Task constants
# ------------------------------------------------------------------------------
TASK_ID = "planetary_conf_2026"
TASK_DESCRIPTION = (
    "Identify the academic conference in planetary science that meets ALL of the following criteria:\n\n"
    "- The conference is the 57th edition of this annual conference series\n"
    "- It takes place from March 16-20, 2026 (5 consecutive days)\n"
    "- The conference is held in The Woodlands, Texas\n"
    "- The venue is The Woodlands Waterway Marriott Hotel and Convention Center\n"
    "- The conference is offered in a hybrid format (both in-person and virtual attendance options)\n"
    "- It is jointly hosted by the Lunar and Planetary Institute (LPI) and NASA Johnson Space Center\n"
    "- The primary scientific focus is lunar and planetary science research\n"
    "- Registration for the conference opened by January 21, 2026\n"
    "- The location is approximately 31 miles north of Houston, Texas\n"
    "- The conference is described as a defining event in planetary research that brings together diverse international experts\n\n"
    "Provide the full name of the conference and its commonly used acronym."
)


# ------------------------------------------------------------------------------
# Extraction models
# ------------------------------------------------------------------------------
class ConferenceInfo(BaseModel):
    # Identification
    full_name: Optional[str] = None
    acronym: Optional[str] = None

    # Core constraints
    edition_number: Optional[str] = None  # e.g., "57th"
    start_date: Optional[str] = None      # e.g., "March 16, 2026"
    end_date: Optional[str] = None        # e.g., "March 20, 2026"
    city: Optional[str] = None            # e.g., "The Woodlands"
    state: Optional[str] = None           # e.g., "Texas"
    venue_name: Optional[str] = None      # e.g., "The Woodlands Waterway Marriott Hotel and Convention Center"
    format: Optional[str] = None          # e.g., "hybrid", "in-person and virtual"
    host_organizations: List[str] = Field(default_factory=list)  # e.g., ["Lunar and Planetary Institute (LPI)", "NASA Johnson Space Center"]
    scientific_focus: Optional[str] = None
    registration_open_date: Optional[str] = None  # any phrase/date the answer uses (e.g., "Jan 15, 2026")
    distance_from_houston_miles: Optional[str] = None  # e.g., "approximately 31 miles"
    significance_description: Optional[str] = None     # descriptive text

    # Sources explicitly cited in the answer (critical for verification)
    source_urls: List[str] = Field(default_factory=list)


# ------------------------------------------------------------------------------
# Extraction prompt
# ------------------------------------------------------------------------------
def prompt_extract_conference_info() -> str:
    return """
    Extract the conference identification and all relevant constraint details as explicitly stated in the answer text. 
    For each requested field, return the value exactly as written in the answer. 
    If the answer does not include a field, return null (or an empty array for lists).

    Required fields to extract:
    - full_name: The full official name of the conference provided in the answer (e.g., "57th Lunar and Planetary Science Conference").
    - acronym: The commonly used acronym provided (e.g., "LPSC").
    - edition_number: The edition identifier if stated (e.g., "57th").
    - start_date: The start date (e.g., "March 16, 2026").
    - end_date: The end date (e.g., "March 20, 2026").
    - city: The city (e.g., "The Woodlands").
    - state: The state (e.g., "Texas").
    - venue_name: The full venue name (e.g., "The Woodlands Waterway Marriott Hotel and Convention Center").
    - format: The conference format as described (e.g., "hybrid", "in-person and virtual").
    - host_organizations: A list of organizations listed as hosts or co-hosts (e.g., ["Lunar and Planetary Institute (LPI)", "NASA Johnson Space Center"]).
    - scientific_focus: A short phrase describing the primary scientific focus (e.g., "lunar and planetary science research").
    - registration_open_date: The date (or phrase) indicating registration opened (as provided in the answer).
    - distance_from_houston_miles: The approximate distance statement from Houston (e.g., "approximately 31 miles").
    - significance_description: A short phrase describing the conference significance if provided (e.g., "defining event in planetary research that brings together diverse international experts").
    - source_urls: An array of ALL URLs explicitly cited in the answer that support ANY of the claims above.
      Include URLs even if they appear in markdown link format; extract the actual URLs.
      Only include URLs explicitly present in the answer. Do not invent or infer URLs.

    If a URL is missing a protocol, prepend http:// as instructed.

    Return a single JSON object with these fields.
    """


# ------------------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------------------
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        u_stripped = u.strip()
        if not u_stripped:
            continue
        if u_stripped not in seen:
            seen.add(u_stripped)
            out.append(u_stripped)
    return out


def _conf_ref(info: ConferenceInfo) -> str:
    return (info.full_name or info.acronym or "the conference").strip()


# ------------------------------------------------------------------------------
# Verification builder
# ------------------------------------------------------------------------------
async def build_and_verify_conference_nodes(
    evaluator: Evaluator,
    parent_node,
    info: ConferenceInfo,
) -> None:
    """
    Build the Conference_Identification critical parallel node and all required critical leaf checks,
    then execute URL-grounded verifications where applicable.
    """
    # Create the critical parent node (parallel aggregation)
    conf_node = evaluator.add_parallel(
        id="Conference_Identification",
        desc="Correctly identify the academic conference in planetary science occurring in March 2026 that matches all specified criteria and provide the requested information",
        parent=parent_node,
        critical=True
    )

    # Always deduplicate sources; can be empty
    sources = _dedup_urls(info.source_urls)

    # 1) Edition_Number (critical, factual via sources)
    node_edition = evaluator.add_leaf(
        id="Edition_Number",
        desc="The conference is the 57th edition of this annual conference series",
        parent=conf_node,
        critical=True
    )
    claim_edition = f"The conference {_conf_ref(info)} is the 57th edition of its annual conference series (e.g., '57th LPSC')."
    # 2) Start_Date
    node_start = evaluator.add_leaf(
        id="Start_Date",
        desc="The conference begins on March 16, 2026",
        parent=conf_node,
        critical=True
    )
    claim_start = "The conference begins on March 16, 2026."
    # 3) End_Date
    node_end = evaluator.add_leaf(
        id="End_Date",
        desc="The conference ends on March 20, 2026",
        parent=conf_node,
        critical=True
    )
    claim_end = "The conference ends on March 20, 2026."
    # 4) Duration (5 consecutive days)
    node_duration = evaluator.add_leaf(
        id="Duration",
        desc="The conference spans 5 consecutive days",
        parent=conf_node,
        critical=True
    )
    claim_duration = "The conference spans five consecutive days from March 16 to March 20, 2026 (inclusive)."
    # 5) City_Location
    node_city = evaluator.add_leaf(
        id="City_Location",
        desc="The conference is held in The Woodlands, Texas",
        parent=conf_node,
        critical=True
    )
    claim_city = "The conference is held in The Woodlands, Texas."
    # 6) State_Location
    node_state = evaluator.add_leaf(
        id="State_Location",
        desc="The conference location is in the state of Texas",
        parent=conf_node,
        critical=True
    )
    claim_state = "The conference location is in the state of Texas."
    # 7) Venue_Name
    node_venue = evaluator.add_leaf(
        id="Venue_Name",
        desc="The conference venue is The Woodlands Waterway Marriott Hotel and Convention Center",
        parent=conf_node,
        critical=True
    )
    claim_venue = "The venue is The Woodlands Waterway Marriott Hotel and Convention Center."
    # 8) Conference_Format
    node_format = evaluator.add_leaf(
        id="Conference_Format",
        desc="The conference is offered in a hybrid format with both in-person and virtual attendance options",
        parent=conf_node,
        critical=True
    )
    claim_format = "The conference offers a hybrid format with both in-person and virtual attendance options."
    # 9) First_Host_Organization (LPI)
    node_host1 = evaluator.add_leaf(
        id="First_Host_Organization",
        desc="One of the hosting organizations is the Lunar and Planetary Institute (LPI)",
        parent=conf_node,
        critical=True
    )
    claim_host1 = "The Lunar and Planetary Institute (LPI) is one of the hosts or co-hosts of the conference."
    # 10) Second_Host_Organization (NASA JSC)
    node_host2 = evaluator.add_leaf(
        id="Second_Host_Organization",
        desc="The conference is co-hosted by NASA Johnson Space Center",
        parent=conf_node,
        critical=True
    )
    claim_host2 = "NASA Johnson Space Center is a co-host or co-organizer of the conference."
    # 11) Scientific_Focus
    node_focus = evaluator.add_leaf(
        id="Scientific_Focus",
        desc="The conference's primary scientific focus is lunar and planetary science research",
        parent=conf_node,
        critical=True
    )
    claim_focus = "The conference's primary scientific focus is lunar and planetary science research."
    # 12) Registration_Timeline
    node_reg = evaluator.add_leaf(
        id="Registration_Timeline",
        desc="Conference registration opened by January 21, 2026",
        parent=conf_node,
        critical=True
    )
    claim_reg = "Registration for the conference opened by January 21, 2026 (on or before that date)."
    # 13) Geographic_Context
    node_geo = evaluator.add_leaf(
        id="Geographic_Context",
        desc="The conference location is approximately 31 miles north of Houston, Texas",
        parent=conf_node,
        critical=True
    )
    claim_geo = "The Woodlands, Texas (the conference location) is approximately 31 miles north of Houston, Texas."
    # 14) Conference_Significance
    node_sig = evaluator.add_leaf(
        id="Conference_Significance",
        desc="The conference is described as a defining event in planetary research that brings together diverse international experts",
        parent=conf_node,
        critical=True
    )
    claim_sig = (
        "The conference is described as a defining or premier event in planetary research that brings together a diverse set of international experts."
    )
    # 15) Full_Name_Provided (existence in the answer)
    evaluator.add_custom_node(
        result=(info.full_name is not None and str(info.full_name).strip() != ""),
        id="Full_Name_Provided",
        desc="The solution provides the full name of the conference",
        parent=conf_node,
        critical=True
    )
    # 16) Acronym_Provided (existence in the answer)
    evaluator.add_custom_node(
        result=(info.acronym is not None and str(info.acronym).strip() != ""),
        id="Acronym_Provided",
        desc="The solution provides the commonly used acronym of the conference",
        parent=conf_node,
        critical=True
    )

    # Prepare batched verifications for factual criteria (URL‑grounded where sources exist)
    claims_and_nodes = [
        (
            claim_edition,
            sources,
            node_edition,
            "Accept phrasings like '57th Lunar and Planetary Science Conference' or '57th LPSC'. Minor formatting or punctuation differences are acceptable."
        ),
        (
            claim_start,
            sources,
            node_start,
            "Look for the explicit date range or start date. Accept formats like 'March 16–20, 2026' where the first date implies the start."
        ),
        (
            claim_end,
            sources,
            node_end,
            "Look for the explicit date range or end date. Accept formats like 'March 16–20, 2026' where the second date implies the end."
        ),
        (
            claim_duration,
            sources,
            node_duration,
            "If the page states 'March 16–20, 2026', that implies five consecutive days (Mon–Fri). Accept equivalent phrasing with en dashes or hyphens."
        ),
        (
            claim_city,
            sources,
            node_city,
            "Verify that the location is The Woodlands, Texas (city and state together)."
        ),
        (
            claim_state,
            sources,
            node_state,
            "Confirm that the event location is within the state of Texas."
        ),
        (
            claim_venue,
            sources,
            node_venue,
            "Allow minor variants such as 'The Woodlands Waterway Marriott' as long as the full venue (Hotel and Convention Center) is clearly indicated or equivalent."
        ),
        (
            claim_format,
            sources,
            node_format,
            "Look for wording such as 'hybrid', 'in-person and virtual', 'onsite and virtual', or similar clearly indicating both options."
        ),
        (
            claim_host1,
            sources,
            node_host1,
            "Look for LPI or 'Lunar and Planetary Institute' listed as host or co-host."
        ),
        (
            claim_host2,
            sources,
            node_host2,
            "Look for 'NASA Johnson Space Center' or 'NASA JSC' listed as host or co-host."
        ),
        (
            claim_focus,
            sources,
            node_focus,
            "The scope should emphasize lunar and planetary science research; allow paraphrases indicating planetary science focus."
        ),
        (
            claim_reg,
            sources,
            node_reg,
            "The evidence should indicate that registration opened on or before January 21, 2026. Any date earlier than or equal to Jan 21, 2026 that states 'registration opens' or 'registration is open' is acceptable."
        ),
        (
            claim_geo,
            sources,
            node_geo,
            "Allow approximate equivalents (e.g., ~30–33 miles). The statement must indicate it's north of Houston."
        ),
        (
            claim_sig,
            sources,
            node_sig,
            "Look for language describing the conference as premier/defining and bringing together a diverse/international community of experts."
        ),
    ]

    # Execute verifications in parallel
    await evaluator.batch_verify(claims_and_nodes)


# ------------------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------------------
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
    Entry point for evaluating an answer to the planetary science conference identification task.
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

    # Extract structured conference info from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_conference_info(),
        template_class=ConferenceInfo,
        extraction_name="conference_extraction"
    )

    # Optionally record a concise custom info block for debugging
    evaluator.add_custom_info(
        info={
            "full_name": extracted_info.full_name,
            "acronym": extracted_info.acronym,
            "edition_number": extracted_info.edition_number,
            "start_date": extracted_info.start_date,
            "end_date": extracted_info.end_date,
            "city": extracted_info.city,
            "state": extracted_info.state,
            "venue_name": extracted_info.venue_name,
            "format": extracted_info.format,
            "host_organizations": extracted_info.host_organizations,
            "registration_open_date": extracted_info.registration_open_date,
            "distance_from_houston_miles": extracted_info.distance_from_houston_miles,
            "num_source_urls": len(extracted_info.source_urls or []),
        },
        info_type="extraction_overview",
        info_name="extraction_overview"
    )

    # Build verification nodes and run checks
    await build_and_verify_conference_nodes(evaluator, root, extracted_info)

    # Return structured summary
    return evaluator.get_summary()