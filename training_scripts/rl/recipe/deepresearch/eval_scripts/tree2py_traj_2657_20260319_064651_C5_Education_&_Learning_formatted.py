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
TASK_ID = "ohio_school_weather_policies"
TASK_DESCRIPTION = """
Identify three Ohio public school districts and provide comprehensive information about their weather-related school closure and delay policies. For each district, you must include:

1. The official name of the school district
2. The URL of the district's official website
3. The specific time or time range when the district typically makes decisions about weather-related school closures or delays (e.g., "5:00 AM" or "between 4:30 AM and 5:30 AM")
4. At least two official communication channels the district uses to notify parents and staff of closures or delays (such as the district website, automated text/email alerts, social media platforms, local TV stations, or local radio stations)
5. Information about how the district implements two-hour delays, specifically whether school start times and bus pickup times are delayed by two hours
6. If the district publicly states a specific temperature or windchill threshold for closure decisions, provide this threshold
7. A reference URL (such as a page from the district's website or a documented policy) that supports the weather policy information you provide
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TwoHourDelayInfo(BaseModel):
    start_times_delayed_two_hours: Optional[str] = None  # expected values like: "yes", "no", or a short phrase
    bus_pickup_delayed_two_hours: Optional[str] = None   # expected values like: "yes", "no", or a short phrase
    dismissal_handling: Optional[str] = None             # e.g., "regular dismissal", "unchanged", or specific time


class DistrictPolicy(BaseModel):
    official_name: Optional[str] = None
    official_website_url: Optional[str] = None
    decision_maker: Optional[str] = None  # e.g., "Superintendent", "Superintendent/designee"
    transport_consultation: Optional[str] = None  # mention of transportation/operations consultation if claimed
    decision_time: Optional[str] = None  # specific time or range, as text
    communication_channels: List[str] = Field(default_factory=list)  # ["district website", "text/email alerts", ...]
    two_hour_delay: Optional[TwoHourDelayInfo] = None
    temperature_threshold: Optional[str] = None  # e.g., "-15°F wind chill", "0°F", or None if not claimed
    reference_urls: List[str] = Field(default_factory=list)  # policy or announcement pages that support info


class WeatherPoliciesExtraction(BaseModel):
    districts: List[DistrictPolicy] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_weather_policies() -> str:
    return """
    Extract up to three Ohio public school districts mentioned in the answer and their weather-related closure/delay policy details.

    For each district, extract the following fields exactly as stated in the answer:
    - official_name: The official name of the school district.
    - official_website_url: The URL of the district's official website, if present in the answer. Use the exact URL text.
    - decision_maker: Who is identified in the answer as the official decision-maker for weather-related closures/delays (e.g., "Superintendent", "Superintendent/designee"). If unspecified in the answer, return null.
    - transport_consultation: If the answer claims that transportation/operations staff are consulted (e.g., transportation supervisor), include the exact phrase or a short paraphrase; otherwise return null.
    - decision_time: The specific time or time range when the district typically makes closure/delay decisions (e.g., "5:00 AM", "between 4:30 AM and 5:30 AM"). If not provided, return null.
    - communication_channels: A list of at least zero strings listing the official communication channels explicitly mentioned in the answer (e.g., "district website", "text/email alerts", "robocall/phone", "Facebook", "Twitter/X", "local TV stations", "local radio", "parent portal"). Return exactly what is claimed in the answer; do not invent.
    - two_hour_delay:
        - start_times_delayed_two_hours: Extract a short value that captures the claim about school start times on a two-hour delay (e.g., "yes", "no", or a short phrase). Use "yes" if the answer states start times are delayed by two hours; use "no" if it states otherwise; if unclear, return a short phrase or null.
        - bus_pickup_delayed_two_hours: Similar rule as above but for bus pickup times. Use "yes" if the answer states bus pickups are delayed by two hours; "no" if otherwise; if unclear, return a short phrase or null.
        - dismissal_handling: A short phrase describing how dismissal times are handled on two-hour delay days (e.g., "regular dismissal", "unchanged", "dismissal as usual"). If not stated, return null.
    - temperature_threshold: If the answer claims a numeric temperature/windchill threshold (e.g., "-15°F wind chill"), extract it as a short text including unit; otherwise return null.
    - reference_urls: A list of one or more URLs that the answer cites for the district’s weather policy (policy page, handbook, news/alerts page on district site, etc.). If none are cited, return an empty list.

    Return a JSON object with a "districts" array of up to three DistrictPolicy objects. If the answer mentions more than three, include only the first three. If fewer are mentioned, include as many as available.
    Do not invent or infer details that do not explicitly appear in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s) and bool(s.strip())


def _gather_policy_sources(d: DistrictPolicy, require_any: bool = True) -> List[str]:
    # Merge unique URLs: reference_urls first, then official website
    urls = []
    for u in (d.reference_urls or []):
        if _nonempty(u) and u not in urls:
            urls.append(u)
    if _nonempty(d.official_website_url) and d.official_website_url not in urls:
        urls.append(d.official_website_url)  # sometimes the official site itself hosts policy
    if require_any:
        return urls
    return urls


def _first_or_none(urls: List[str]) -> Optional[str]:
    return urls[0] if urls else None


# --------------------------------------------------------------------------- #
# Verification per district                                                   #
# --------------------------------------------------------------------------- #
async def verify_single_district(
    evaluator: Evaluator,
    root_node,
    d: DistrictPolicy,
    idx: int,
) -> None:
    """
    Build the verification subtree for one district according to the rubric.
    """
    district_node = evaluator.add_parallel(
        id=f"district_{idx+1}",
        desc=f"District {idx+1} policy details",
        parent=root_node,
        critical=False,
    )

    # dX_official_name (critical): existence check
    if _nonempty(d.official_name):
        evaluator.add_custom_node(
            result=True,
            id=f"d{idx+1}_official_name",
            desc="Provides the official name of the school district",
            parent=district_node,
            critical=True,
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"d{idx+1}_official_name",
            desc="Provides the official name of the school district",
            parent=district_node,
            critical=True,
        )

    # dX_is_ohio_public_district (critical): verify via sources
    ohio_public_node = evaluator.add_leaf(
        id=f"d{idx+1}_is_ohio_public_district",
        desc="District is an actual Ohio public school district",
        parent=district_node,
        critical=True,
    )
    sources_for_ohio = _gather_policy_sources(d, require_any=True)
    if _nonempty(d.official_name) and sources_for_ohio:
        claim = f"The district named '{d.official_name}' is a public school district located in Ohio."
        await evaluator.verify(
            claim=claim,
            node=ohio_public_node,
            sources=sources_for_ohio,
            additional_instruction="Confirm that the district is a public K-12 school district in the State of Ohio using the provided webpage(s). Prefer explicit mentions like 'Ohio' or 'OH' and context that it's a public school district."
        )
    else:
        # No usable sources -> fail
        ohio_public_node.score = 0.0
        ohio_public_node.status = "failed"

    # dX_official_website_url (critical): verify that URL is the official district website
    if _nonempty(d.official_website_url) and _nonempty(d.official_name):
        site_node = evaluator.add_leaf(
            id=f"d{idx+1}_official_website_url",
            desc="Provides a valid URL to the district's official website",
            parent=district_node,
            critical=True,
        )
        claim = f"This URL is the official website of the school district named '{d.official_name}'."
        await evaluator.verify(
            claim=claim,
            node=site_node,
            sources=d.official_website_url,
            additional_instruction="Accept if the site branding, header/footer, or about/contact indicates it is the official website for the named school district."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"d{idx+1}_official_website_url",
            desc="Provides a valid URL to the district's official website",
            parent=district_node,
            critical=True,
        )

    # dX_decision_maker (critical): superintendent as decision maker
    dm_node = evaluator.add_leaf(
        id=f"d{idx+1}_decision_maker",
        desc="Identifies the superintendent as the official decision-maker for weather-related closures/delays (per constraints)",
        parent=district_node,
        critical=True,
    )
    dm_sources = _gather_policy_sources(d, require_any=True)
    if dm_sources:
        claim = "The superintendent is the official decision-maker (or primary authority/designee) for weather-related closures and delays."
        await evaluator.verify(
            claim=claim,
            node=dm_node,
            sources=dm_sources,
            additional_instruction="Look for explicit phrasing that the Superintendent (or Superintendent's designee) makes the final decision regarding weather-related closures or delays."
        )
    else:
        dm_node.score = 0.0
        dm_node.status = "failed"

    # dX_transport_consultation (non-critical): optional, only verify if claimed in answer; else pass
    if _nonempty(d.transport_consultation):
        trans_node = evaluator.add_leaf(
            id=f"d{idx+1}_transport_consultation",
            desc="Mentions consultation with transportation/operations staff (e.g., transportation supervisor) if stated in the cited district policy (aligning with constraints)",
            parent=district_node,
            critical=False,
        )
        trans_sources = _gather_policy_sources(d, require_any=True)
        if trans_sources:
            claim = "The district's policy mentions consultation with transportation/operations staff (e.g., transportation supervisor) as part of the weather closure/delay decision process."
            await evaluator.verify(
                claim=claim,
                node=trans_node,
                sources=trans_sources,
                additional_instruction="Accept if the page states that the Superintendent consults with transportation, operations, or similar staff before deciding closures/delays."
            )
        else:
            trans_node.score = 0.0
            trans_node.status = "failed"
    else:
        # Optional and not claimed -> pass without penalty
        evaluator.add_custom_node(
            result=True,
            id=f"d{idx+1}_transport_consultation",
            desc="Mentions consultation with transportation/operations staff (optional, not claimed)",
            parent=district_node,
            critical=False,
        )

    # dX_decision_time (critical): verify claimed time/range
    if _nonempty(d.decision_time):
        dt_node = evaluator.add_leaf(
            id=f"d{idx+1}_decision_time",
            desc="Provides the specific time or time range when the district typically makes closure/delay decisions",
            parent=district_node,
            critical=True,
        )
        dt_sources = _gather_policy_sources(d, require_any=True)
        if dt_sources:
            claim = f"The district typically makes weather closure/delay decisions at or by: {d.decision_time}."
            await evaluator.verify(
                claim=claim,
                node=dt_node,
                sources=dt_sources,
                additional_instruction="Accept reasonable variations, e.g., 'by 5:30 AM' vs 'between 5:00–5:30 AM'. The page should explicitly reference decision timing."
            )
        else:
            dt_node.score = 0.0
            dt_node.status = "failed"
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"d{idx+1}_decision_time",
            desc="Provides the specific time or time range when the district typically makes closure/delay decisions",
            parent=district_node,
            critical=True,
        )

    # dX_comm_channels (critical group, parallel)
    comm_group = evaluator.add_parallel(
        id=f"d{idx+1}_comm_channels",
        desc="Lists official communication channels used for notifications",
        parent=district_node,
        critical=True,
    )

    # dX_comm_includes_website (critical): verify via sources
    comm_website_node = evaluator.add_leaf(
        id=f"d{idx+1}_comm_includes_website",
        desc="Communication channels include the district official website (per constraints)",
        parent=comm_group,
        critical=True,
    )
    comm_sources = _gather_policy_sources(d, require_any=True)
    if comm_sources:
        claim = "The district uses its official website as one of the communication channels to announce weather-related closures or delays."
        await evaluator.verify(
            claim=claim,
            node=comm_website_node,
            sources=comm_sources,
            additional_instruction="Accept if the policy/notifications indicate closures/delays are posted on the district website."
        )
    else:
        comm_website_node.score = 0.0
        comm_website_node.status = "failed"

    # dX_comm_at_least_two_total (critical): simple count check from extraction
    evaluator.add_custom_node(
        result=len(d.communication_channels) >= 2,
        id=f"d{idx+1}_comm_at_least_two_total",
        desc="At least two official communication channels are provided in total",
        parent=comm_group,
        critical=True,
    )

    # dX_two_hour_delay (critical group, parallel)
    two_hr_group = evaluator.add_parallel(
        id=f"d{idx+1}_two_hour_delay",
        desc="Explains how a two-hour delay is implemented (per question + constraints)",
        parent=district_node,
        critical=True,
    )

    # Start times delayed by two hours (critical)
    if d.two_hour_delay and _nonempty(d.two_hour_delay.start_times_delayed_two_hours):
        start_node = evaluator.add_leaf(
            id=f"d{idx+1}_two_hour_delay_start_time",
            desc="States whether school start times are delayed by exactly two hours",
            parent=two_hr_group,
            critical=True,
        )
        th_sources = _gather_policy_sources(d, require_any=True)
        if th_sources:
            if d.two_hour_delay.start_times_delayed_two_hours.strip().lower() in ("yes", "true", "2 hours", "two hours", "2-hour", "two-hour"):
                claim = "On a two-hour delay day, school start times are delayed by two hours."
            elif d.two_hour_delay.start_times_delayed_two_hours.strip().lower() in ("no", "false"):
                claim = "On a two-hour delay day, school start times are not delayed by exactly two hours."
            else:
                claim = f"The district explains how start times change on a two-hour delay, consistent with: {d.two_hour_delay.start_times_delayed_two_hours}."
            await evaluator.verify(
                claim=claim,
                node=start_node,
                sources=th_sources,
                additional_instruction="Look for explicit 'two-hour delay' procedures regarding student start times."
            )
        else:
            start_node.score = 0.0
            start_node.status = "failed"
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"d{idx+1}_two_hour_delay_start_time",
            desc="States whether school start times are delayed by exactly two hours",
            parent=two_hr_group,
            critical=True,
        )

    # Bus pickup delayed by two hours (critical)
    if d.two_hour_delay and _nonempty(d.two_hour_delay.bus_pickup_delayed_two_hours):
        bus_node = evaluator.add_leaf(
            id=f"d{idx+1}_two_hour_delay_bus_pickup",
            desc="States whether bus pickup times are delayed by exactly two hours",
            parent=two_hr_group,
            critical=True,
        )
        th_sources = _gather_policy_sources(d, require_any=True)
        if th_sources:
            if d.two_hour_delay.bus_pickup_delayed_two_hours.strip().lower() in ("yes", "true", "2 hours", "two hours", "2-hour", "two-hour"):
                claim = "On a two-hour delay day, school bus pickup times are delayed by two hours."
            elif d.two_hour_delay.bus_pickup_delayed_two_hours.strip().lower() in ("no", "false"):
                claim = "On a two-hour delay day, school bus pickup times are not delayed by exactly two hours."
            else:
                claim = f"The district explains how bus pickup times change on a two-hour delay, consistent with: {d.two_hour_delay.bus_pickup_delayed_two_hours}."
            await evaluator.verify(
                claim=claim,
                node=bus_node,
                sources=th_sources,
                additional_instruction="Look for explicit 'two-hour delay' procedures for transportation/pickup schedules."
            )
        else:
            bus_node.score = 0.0
            bus_node.status = "failed"
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"d{idx+1}_two_hour_delay_bus_pickup",
            desc="States whether bus pickup times are delayed by exactly two hours",
            parent=two_hr_group,
            critical=True,
        )

    # Dismissal handling (critical)
    if d.two_hour_delay and _nonempty(d.two_hour_delay.dismissal_handling):
        dismiss_node = evaluator.add_leaf(
            id=f"d{idx+1}_two_hour_delay_dismissal",
            desc="States how dismissal times are handled on a two-hour delay day (per constraints or per district policy)",
            parent=two_hr_group,
            critical=True,
        )
        th_sources = _gather_policy_sources(d, require_any=True)
        if th_sources:
            claim = f"On two-hour delay days, dismissal times are handled as: {d.two_hour_delay.dismissal_handling}."
            await evaluator.verify(
                claim=claim,
                node=dismiss_node,
                sources=th_sources,
                additional_instruction="Accept typical policy such as 'regular dismissal'/'unchanged' if stated on the page."
            )
        else:
            dismiss_node.score = 0.0
            dismiss_node.status = "failed"
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"d{idx+1}_two_hour_delay_dismissal",
            desc="States how dismissal times are handled on a two-hour delay day (per constraints or per district policy)",
            parent=two_hr_group,
            critical=True,
        )

    # dX_temperature_threshold (non-critical): optional; verify if provided, otherwise pass
    if _nonempty(d.temperature_threshold):
        tt_node = evaluator.add_leaf(
            id=f"d{idx+1}_temperature_threshold",
            desc="If the answer claims a specific temperature/windchill threshold used for closure decisions, it provides a specific threshold (e.g., numeric value and unit) and it is supported by the provided reference URL(s); otherwise it may be omitted",
            parent=district_node,
            critical=False,
        )
        tt_sources = _gather_policy_sources(d, require_any=True)
        if tt_sources:
            claim = f"The district publicly states a temperature/windchill threshold for closures: {d.temperature_threshold}."
            await evaluator.verify(
                claim=claim,
                node=tt_node,
                sources=tt_sources,
                additional_instruction="Only accept if the threshold number and unit are present or clearly implied on the page (e.g., '-15°F wind chill')."
            )
        else:
            tt_node.score = 0.0
            tt_node.status = "failed"
    else:
        evaluator.add_custom_node(
            result=True,
            id=f"d{idx+1}_temperature_threshold",
            desc="Temperature/windchill threshold not claimed (optional)",
            parent=district_node,
            critical=False,
        )

    # dX_reference_url (critical): at least one reference URL that supports weather policy info
    ref_node = evaluator.add_leaf(
        id=f"d{idx+1}_reference_url",
        desc="Provides at least one reference URL that supports the weather policy information given for this district",
        parent=district_node,
        critical=True,
    )
    if d.reference_urls:
        claim = "This URL is an official or authoritative page that describes the district’s weather-related closure/delay policies or announcements."
        # Use multi-URL verification; success if any one supports the claim
        await evaluator.verify(
            claim=claim,
            node=ref_node,
            sources=d.reference_urls,
            additional_instruction="Prefer pages on the district's official website (policy page, student handbook, news/alerts page) that discuss closures/delays or two-hour delay procedures."
        )
    else:
        ref_node.score = 0.0
        ref_node.status = "failed"


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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Ohio public school district weather policy task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root as parallel; allow partial scoring across districts
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

    # Extract structured info
    extracted: WeatherPoliciesExtraction = await evaluator.extract(
        prompt=prompt_extract_weather_policies(),
        template_class=WeatherPoliciesExtraction,
        extraction_name="weather_policies_extraction",
    )

    # Normalize to exactly 3 slots (pad with empty if needed)
    districts: List[DistrictPolicy] = (extracted.districts or [])[:3]
    while len(districts) < 3:
        districts.append(DistrictPolicy())

    # Build verification tree for each district
    # The rubric defines each district as a parallel non-critical node under root
    tasks = []
    for i in range(3):
        tasks.append(verify_single_district(evaluator, root, districts[i], i))
    await asyncio.gather(*tasks)

    return evaluator.get_summary()