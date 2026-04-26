import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_jan2025_incidents"
TASK_DESCRIPTION = (
    "Identify four separate major fatal incidents that occurred in the United States during January 2025, where each "
    "incident resulted in at least 10 fatalities. For each incident, provide: (1) date or date range, (2) location "
    "(city+state or region+state), (3) confirmed fatalities, (4) nature/type, and (5) at least one reference URL from "
    "a reputable news source or official report. Ensure each of the four incidents is distinct and occurred at different "
    "times or locations."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class IncidentItem(BaseModel):
    # Keep fields as strings for robustness; do not force numeric/date parsing
    date_or_range: Optional[str] = None
    city_or_region: Optional[str] = None
    state: Optional[str] = None
    fatalities: Optional[str] = None
    nature: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class IncidentsExtraction(BaseModel):
    incidents: List[IncidentItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_incidents() -> str:
    return """
    Parse the answer into a list of discrete incident entries (ideally 4 or more if present). An "incident entry" should
    correspond to one distinct event described in the answer (e.g., a collision, attack, disaster, or similar).
    
    For each incident entry, extract the following fields exactly as stated in the answer (do not invent or infer):
    - date_or_range: The specific date or date range given (e.g., "Jan 5, 2025" or "Jan 7–9, 2025"). If not provided, set to null.
    - city_or_region: The city or named region indicated (for widespread incidents, the region name like "Southern California" or "Midwest"). If not provided, set to null.
    - state: The U.S. state abbreviation or full name (e.g., "CA" or "California"). If not provided, set to null.
    - fatalities: The number of confirmed fatalities the answer claims (string as written, e.g., "12", "at least 15", "10-12"). If not provided, set to null.
    - nature: The nature/type of the incident (e.g., "attack", "wildfire", "collision", "storm", "disaster"). If not provided, set to null.
    - sources: An array of URLs cited for this incident in the answer. Extract only valid URLs explicitly present (plain or markdown). If none given, return an empty array.

    Important:
    - Extract only what is explicitly present in the answer. Do not add information that is not written in the answer.
    - If there are more than four incidents in the answer, extract all of them (we will later consider only the first four).
    - If fewer than four are provided, extract whatever is present.
    - Do not merge multiple incidents together; keep each separate incident as its own object in the "incidents" array.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _summarize_incident_for_distinct_check(incident: IncidentItem, idx: int) -> str:
    d = incident.date_or_range or "N/A"
    loc = (incident.city_or_region or "N/A") + ", " + (incident.state or "N/A")
    fat = incident.fatalities or "N/A"
    nat = incident.nature or "N/A"
    return f"Incident {idx + 1}: date='{d}', location='{loc}', fatalities='{fat}', nature='{nat}'"


def _first_k_incidents(incidents: List[IncidentItem], k: int) -> List[IncidentItem]:
    # Return first k incidents; pad with empty placeholders if fewer
    selected = incidents[:k]
    while len(selected) < k:
        selected.append(IncidentItem())
    return selected


# --------------------------------------------------------------------------- #
# Per-incident verification                                                   #
# --------------------------------------------------------------------------- #
async def verify_one_incident(
    evaluator: Evaluator,
    parent_node,
    incident: IncidentItem,
    idx: int,
) -> None:
    """
    Build the sub-tree for one incident, with the five critical checks as leaf nodes.
    Per rubric, this incident node is non-critical under the root (partial credit allowed).
    """
    incident_node = evaluator.add_parallel(
        id=f"incident_{idx + 1}",
        desc=f"Incident #{idx + 1}: meets all per-incident constraints and includes all required fields",
        parent=parent_node,
        critical=False,
    )

    # Prepare info
    date_text = incident.date_or_range or ""
    city_or_region = incident.city_or_region or ""
    state = incident.state or ""
    fatalities_text = incident.fatalities or ""
    nature_text = incident.nature or ""
    urls = incident.sources if incident.sources else []

    # 1) Date within January 2025 (and provided)
    date_node = evaluator.add_leaf(
        id=f"incident_{idx + 1}_date",
        desc="Specific date or date range is provided and falls within January 1–31, 2025",
        parent=incident_node,
        critical=True,
    )
    date_claim = (
        f"The answer provides the date/date range as: '{date_text}'. "
        f"This incident occurred during that stated date/time and all relevant event dates fall within January 2025 "
        f"(i.e., January 1–31, 2025, considering U.S. local time zones). "
        f"If the provided date is missing or blank, this claim is incorrect."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=urls,  # Verify using the cited references
        additional_instruction=(
            "Check whether the referenced webpage(s) explicitly support that the incident took place on the stated "
            "date(s) and that those date(s) are within January 2025. If there is a range (e.g., Jan 7–9), the range "
            "must be wholly within January 2025. If the answer's date is missing, judge as Incorrect."
        ),
    )

    # 2) Location provided and within the U.S.
    location_node = evaluator.add_leaf(
        id=f"incident_{idx + 1}_location",
        desc="Location is provided (city+state or region+state for widespread incidents) and is within the United States",
        parent=incident_node,
        critical=True,
    )
    location_claim = (
        f"The answer provides the location as: '{city_or_region}, {state}'. "
        f"The incident occurred in the United States at that location (city+state or region+state). "
        f"If the provided location is missing (city/region or state is blank), this claim is incorrect."
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_node,
        sources=urls,
        additional_instruction=(
            "Verify that the provided location string is not blank and that the cited webpage(s) support that the "
            "incident occurred at that U.S. location. Allow reasonable variants of city/region naming; ensure a "
            "corresponding U.S. state is indicated. If the location fields are missing, mark as Incorrect."
        ),
    )

    # 3) Fatalities stated and at least 10
    fatalities_node = evaluator.add_leaf(
        id=f"incident_{idx + 1}_fatalities",
        desc="Total number of confirmed fatalities is stated and is at least 10",
        parent=incident_node,
        critical=True,
    )
    fatalities_claim = (
        f"The answer states the fatalities as: '{fatalities_text}'. Based on the provided source URL(s), "
        f"the confirmed death toll for this incident is at least 10. "
        f"If the fatalities value is missing or the sources indicate fewer than 10 confirmed deaths, "
        f"this claim is incorrect."
    )
    await evaluator.verify(
        claim=fatalities_claim,
        node=fatalities_node,
        sources=urls,
        additional_instruction=(
            "Use the cited sources to check the confirmed fatality count. The answer's fatalities may be written as a "
            "number or range (e.g., 'at least 12', '10-12'). This claim should pass only if the articles or official "
            "reports substantiate at least 10 fatalities. If the answer doesn't provide a fatalities value, or "
            "if sources support < 10 deaths, judge as Incorrect."
        ),
    )

    # 4) Nature/type clearly described and supported
    nature_node = evaluator.add_leaf(
        id=f"incident_{idx + 1}_nature",
        desc="Nature/type of incident is clearly described",
        parent=incident_node,
        critical=True,
    )
    nature_claim = (
        f"The answer describes the incident type/nature as: '{nature_text}'. "
        f"The referenced source(s) support that description (e.g., collision, attack, storm/disaster, wildfire, etc.). "
        f"If the nature/type is missing, this claim is incorrect."
    )
    await evaluator.verify(
        claim=nature_claim,
        node=nature_node,
        sources=urls,
        additional_instruction=(
            "Check that the incident type or nature stated in the answer is not blank and is supported by the cited "
            "webpage(s). Allow reasonable synonyms (e.g., 'crash' vs 'collision', 'storm' vs 'severe weather')."
        ),
    )

    # 5) At least one reputable reference URL is provided
    reference_node = evaluator.add_leaf(
        id=f"incident_{idx + 1}_reference",
        desc="At least one reference URL from a reputable news source or official report is provided",
        parent=incident_node,
        critical=True,
    )
    # Use a simple verification (no page retrieval needed) to judge reputability and presence
    urls_preview = urls if urls else []
    reference_claim = (
        f"The answer provides at least one reference URL for this incident, and at least one of these URLs is from a "
        f"reputable news outlet or an official source: {urls_preview}. "
        f"If there are no URLs listed, this claim is incorrect."
    )
    await evaluator.verify(
        claim=reference_claim,
        node=reference_node,
        additional_instruction=(
            "Judge reputability by domain. Consider reputable: .gov, .mil, .edu, official agency sites (FEMA, NTSB, DOT, "
            "NOAA, USGS, etc.), wire services (AP, Reuters), major national outlets (NYTimes, Washington Post, WSJ, "
            "Bloomberg, NPR, ABC, CBS, NBC, CNN), and credible local newspapers/tv stations. "
            "Do NOT consider personal blogs, low-quality aggregators, or unknown sites as reputable. "
            "If the list is empty, mark as Incorrect."
        ),
    )


# --------------------------------------------------------------------------- #
# Root-level verification                                                     #
# --------------------------------------------------------------------------- #
async def verify_root_constraints(
    evaluator: Evaluator,
    root_node,
    extracted: IncidentsExtraction,
    selected_incidents: List[IncidentItem],
) -> None:
    """
    Add the two critical root constraints:
    1) At least four incident entries are provided (not fewer).
    2) The four evaluated incidents are distinct events (different times and/or locations).
    """
    # 1) Provide at least four incidents (not fewer)
    count = len(extracted.incidents)
    has_four_or_more = count >= 4
    evaluator.add_custom_node(
        result=has_four_or_more,
        id="four_incidents_provided",
        desc="At least four incidents are provided as separate incident entries (not fewer)",
        parent=root_node,
        critical=True,
    )

    # 2) Distinctness check (simple verification)
    distinct_node = evaluator.add_leaf(
        id="incidents_distinct",
        desc="All four incidents are distinct events (not the same event or sub-events) and occur at different times and/or locations",
        parent=root_node,
        critical=True,
    )
    # Build a compact summary for LLM to assess distinctness
    summaries = [ _summarize_incident_for_distinct_check(inc, i) for i, inc in enumerate(selected_incidents) ]
    distinct_claim = (
        "Assess whether the following four incidents are distinct events (not sub-events/updates of the same overall "
        "incident) and occurred at different times and/or locations:\n"
        + "\n".join(summaries)
        + "\nIf any pair appears to be the same event (same place and date, only phrased differently), or one is just a "
        "sub-event/update of another, then this claim is incorrect."
    )
    await evaluator.verify(
        claim=distinct_claim,
        node=distinct_node,
        additional_instruction=(
            "Focus on dates/date ranges and locations to determine distinctness. Minor wording differences don't make a "
            "new event. If two have the same location and date window, likely the same incident."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the 'US January 2025 fatal incidents (>=10 fatalities) — four distinct incidents' task.
    """
    # Initialize evaluator with a parallel root (per rubric)
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

    # Extract incidents list from the answer
    extracted: IncidentsExtraction = await evaluator.extract(
        prompt=prompt_extract_incidents(),
        template_class=IncidentsExtraction,
        extraction_name="incidents_extraction",
    )

    # Only evaluate the first four incidents; pad if fewer than 4
    selected = _first_k_incidents(extracted.incidents, 4)

    # Add minimal reference info about the evaluation target (not strict GT, but constraints)
    evaluator.add_ground_truth(
        {
            "required_window": "January 1–31, 2025",
            "min_fatalities_per_incident": ">= 10",
            "required_incident_count": 4,
            "must_be_distinct": True,
            "scope": "United States only",
        },
        gt_type="task_constraints",
    )

    # Add critical root constraint checks
    await verify_root_constraints(evaluator, root, extracted, selected)

    # Build four per-incident subtrees (non-critical under root, but each has critical leaves)
    for i, inc in enumerate(selected):
        await verify_one_incident(evaluator, root, inc, i)

    # Return evaluation summary
    return evaluator.get_summary()