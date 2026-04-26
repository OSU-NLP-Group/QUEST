import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "outdoor_facilities_ops"
TASK_DESCRIPTION = (
    "I am planning outdoor recreation activities and need comprehensive operational information about specific facilities.\n\n"
    "Provide details about 2 outdoor recreation facilities located in Lagos, Nigeria and 1 national park in the Canary Islands, Spain that have documented operational policies:\n\n"
    "For the first Lagos facility:\n"
    "- Name of the facility\n- Operating hours\n- Gate closure policy (specify if the facility stops admitting new visitors before official closing time)\n"
    "- Adult entry fee (in Nigerian Naira)\n- Child or youth entry fee (in Nigerian Naira)\n"
    "- A notable infrastructure feature or significant structure at the facility\n- Reference URL\n\n"
    "For the second Lagos facility:\n"
    "- Name of the facility\n- Operating hours\n- Entry fee information\n- Visitor capacity or seating capacity\n- Reference URL\n\n"
    "For the Canary Islands national park:\n"
    "- Name of the national park\n- Which specific island it is located on\n"
    "- Advance permit or reservation requirements for accessing trails or summit areas\n"
    "- Daily visitor capacity limits for specific trails or restricted areas\n"
    "- The schedule for when permits are released, including the specific day of the week and time\n- Reference URL\n\n"
    "All information must be verifiable through the provided reference URLs."
)


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class FirstLagosFacility(BaseModel):
    name: Optional[str] = None
    operating_hours: Optional[str] = None
    gate_closure_policy: Optional[str] = None
    adult_entry_fee_ngn: Optional[str] = None
    child_entry_fee_ngn: Optional[str] = None
    notable_infrastructure: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class SecondLagosFacility(BaseModel):
    name: Optional[str] = None
    operating_hours: Optional[str] = None
    entry_fee_info: Optional[str] = None
    capacity: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class CanaryParkInfo(BaseModel):
    park_name: Optional[str] = None
    island: Optional[str] = None
    permit_requirements: Optional[str] = None
    daily_visitor_limit: Optional[str] = None
    permit_release_schedule: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class FacilitiesExtraction(BaseModel):
    first_lagos: Optional[FirstLagosFacility] = None
    second_lagos: Optional[SecondLagosFacility] = None
    canary_park: Optional[CanaryParkInfo] = None


# -----------------------------------------------------------------------------
# Extraction prompts
# -----------------------------------------------------------------------------
def prompt_extract_all() -> str:
    return """
    Extract exactly the following structured information from the answer text.

    IMPORTANT RULES:
    - Do NOT invent information. Extract only what is explicitly present in the answer.
    - If an item is missing, set it to null (or [] for URL lists).
    - For all URL fields, include only full URLs explicitly present in the answer (markdown links are okay—extract the actual URL).
    - Keep all values as strings as written, including currency symbols such as ₦ or NGN and textual ranges.
    - If multiple Lagos facilities or Canary parks are listed in the answer, take the first two Lagos facilities and the first Canary park only.

    For the first Lagos facility (first_lagos), extract:
    - name
    - operating_hours (verbatim as stated)
    - gate_closure_policy (verbatim; e.g., "last entry 5:00 pm" or "gates close to new visitors 30 minutes before closing"; if not mentioned, set to null)
    - adult_entry_fee_ngn (verbatim string, may include ₦/NGN and commas)
    - child_entry_fee_ngn (verbatim string, may include ₦/NGN and commas)
    - notable_infrastructure (verbatim short phrase, e.g., "suspension bridge", "boardwalk", "amphitheatre")
    - reference_urls (array of all URLs cited for this facility)

    For the second Lagos facility (second_lagos), extract:
    - name
    - operating_hours (verbatim)
    - entry_fee_info (verbatim string summarizing the entry fee(s))
    - capacity (visitor capacity or seating capacity; verbatim string including units or qualifiers like "up to", "maximum", etc.)
    - reference_urls (array of all URLs cited for this facility)

    For the Canary Islands national park (canary_park), extract:
    - park_name
    - island (specific island name, e.g., "Tenerife", "Lanzarote", etc.)
    - permit_requirements (verbatim summary of advance permit/reservation rules for trails/summit areas)
    - daily_visitor_limit (verbatim—put numeric and any qualifiers such as per-day, per-slot, etc.)
    - permit_release_schedule (verbatim and include day of the week and time if provided)
    - reference_urls (array of all URLs cited for this park)

    Return a JSON object with fields:
    {
      "first_lagos": {...},
      "second_lagos": {...},
      "canary_park": {...}
    }
    """


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _has_text(v: Optional[str]) -> bool:
    return isinstance(v, str) and v.strip() != ""


def _sources_or_empty(urls: Optional[List[str]]) -> List[str]:
    return urls if isinstance(urls, list) else []


# -----------------------------------------------------------------------------
# Verification subroutines
# -----------------------------------------------------------------------------
async def verify_first_lagos(evaluator: Evaluator, parent) -> None:
    """
    Build verification subtree for the first Lagos facility.
    """
    # Pull extracted info
    extraction_list = evaluator._extraction_results
    first: Optional[FirstLagosFacility] = None
    for item in extraction_list[::-1]:
        if "result" in item and "first_lagos" in item["result"]:
            try:
                first = FirstLagosFacility(**item["result"]["first_lagos"]) if item["result"]["first_lagos"] else None
                break
            except Exception:
                first = None
                break

    # Node group
    group = evaluator.add_parallel(
        id="First_Lagos_Facility",
        desc="Provides complete information about the first outdoor recreation facility in Lagos, Nigeria",
        parent=parent,
        critical=False
    )

    # If not provided at all, create failing leaves for all required items
    if first is None:
        evaluator.add_custom_node(
            result=False,
            id="first_lagos_reference_url",
            desc="Provides a valid reference URL containing verifiable information about the first Lagos facility",
            parent=group,
            critical=True
        )
        evaluator.add_custom_node(False, "first_lagos_name", "Identifies the name of the first Lagos facility", parent=group, critical=True)
        evaluator.add_custom_node(False, "first_lagos_hours", "Provides the documented operating hours for the first Lagos facility", parent=group, critical=True)
        evaluator.add_custom_node(False, "first_lagos_gate", "Specifies if the first Lagos facility has a gate closure time earlier than official closing time and states when gates close to new visitors", parent=group, critical=True)
        evaluator.add_custom_node(False, "first_lagos_adult_fee", "States the adult entry fee in Nigerian Naira for the first Lagos facility", parent=group, critical=True)
        evaluator.add_custom_node(False, "first_lagos_child_fee", "States the child or youth entry fee in Nigerian Naira for the first Lagos facility", parent=group, critical=True)
        evaluator.add_custom_node(False, "first_lagos_infra", "Describes a notable infrastructure feature or significant structure at the first Lagos facility", parent=group, critical=True)
        return

    urls = _sources_or_empty(first.reference_urls)

    # Reference URL existence (critical gate)
    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="first_lagos_reference_url",
        desc="Provides a valid reference URL containing verifiable information about the first Lagos facility",
        parent=group,
        critical=True
    )

    # Prepare verifications
    claims_and_nodes: List[tuple[str, List[str], Any, str]] = []

    # Facility Name
    if not _has_text(first.name):
        evaluator.add_custom_node(False, "first_lagos_name", "Identifies the name of the first Lagos facility", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="first_lagos_name",
            desc="Identifies the name of the first Lagos facility",
            parent=group,
            critical=True
        )
        claim = f"The page is about a facility named '{first.name}' located in Lagos, Nigeria (allow minor variants or formatting)."
        add_ins = "Confirm the page explicitly names the facility (or an equivalent/official variant) and that it is in Lagos, Nigeria."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Operating Hours
    if not _has_text(first.operating_hours):
        evaluator.add_custom_node(False, "first_lagos_hours", "Provides the documented operating hours for the first Lagos facility", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="first_lagos_hours",
            desc="Provides the documented operating hours for the first Lagos facility",
            parent=group,
            critical=True
        )
        nm = first.name or "the facility"
        claim = f"The operating hours for {nm} are: {first.operating_hours}."
        add_ins = "Match the stated schedule (days/times). Allow minor formatting differences and common abbreviations."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Gate Closure Policy
    if not _has_text(first.gate_closure_policy):
        evaluator.add_custom_node(False, "first_lagos_gate", "Specifies if the first Lagos facility has a gate closure time earlier than official closing time and states when gates close to new visitors", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="first_lagos_gate",
            desc="Specifies if the first Lagos facility has a gate closure time earlier than official closing time and states when gates close to new visitors",
            parent=group,
            critical=True
        )
        nm = first.name or "the facility"
        claim = f"{nm} has a gate closure/last-entry policy for new visitors: {first.gate_closure_policy}."
        add_ins = "Verify the page states last entry or gate closure time for new visitors (earlier than closing). If no such policy appears, mark incorrect."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Adult Entry Fee
    if not _has_text(first.adult_entry_fee_ngn):
        evaluator.add_custom_node(False, "first_lagos_adult_fee", "States the adult entry fee in Nigerian Naira for the first Lagos facility", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="first_lagos_adult_fee",
            desc="States the adult entry fee in Nigerian Naira for the first Lagos facility",
            parent=group,
            critical=True
        )
        nm = first.name or "the facility"
        claim = f"The adult entry fee for {nm} is: {first.adult_entry_fee_ngn}."
        add_ins = "Confirm fee on the page. Allow NGN/₦ symbols, commas, and minor formatting; ensure it is the adult price."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Child/Youth Entry Fee
    if not _has_text(first.child_entry_fee_ngn):
        evaluator.add_custom_node(False, "first_lagos_child_fee", "States the child or youth entry fee in Nigerian Naira for the first Lagos facility", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="first_lagos_child_fee",
            desc="States the child or youth entry fee in Nigerian Naira for the first Lagos facility",
            parent=group,
            critical=True
        )
        nm = first.name or "the facility"
        claim = f"The child/youth entry fee for {nm} is: {first.child_entry_fee_ngn}."
        add_ins = "Confirm the child/youth fee as stated. Allow NGN/₦ symbols and formatting variants."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Notable Infrastructure
    if not _has_text(first.notable_infrastructure):
        evaluator.add_custom_node(False, "first_lagos_infra", "Describes a notable infrastructure feature or significant structure at the first Lagos facility", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="first_lagos_infra",
            desc="Describes a notable infrastructure feature or significant structure at the first Lagos facility",
            parent=group,
            critical=True
        )
        nm = first.name or "the facility"
        claim = f"A notable infrastructure feature at {nm} is: {first.notable_infrastructure}."
        add_ins = "Confirm the page explicitly mentions this structure/feature as part of the facility."
        claims_and_nodes.append((claim, urls, node, add_ins))

    if claims_and_nodes:
        await evaluator.batch_verify(claims_and_nodes)


async def verify_second_lagos(evaluator: Evaluator, parent) -> None:
    """
    Build verification subtree for the second Lagos facility.
    """
    extraction_list = evaluator._extraction_results
    second: Optional[SecondLagosFacility] = None
    for item in extraction_list[::-1]:
        if "result" in item and "second_lagos" in item["result"]:
            try:
                second = SecondLagosFacility(**item["result"]["second_lagos"]) if item["result"]["second_lagos"] else None
                break
            except Exception:
                second = None
                break

    group = evaluator.add_parallel(
        id="Second_Lagos_Facility",
        desc="Provides complete information about the second outdoor recreation facility in Lagos, Nigeria",
        parent=parent,
        critical=False
    )

    if second is None:
        evaluator.add_custom_node(False, "second_lagos_reference_url", "Provides a valid reference URL containing verifiable information about the second Lagos facility", parent=group, critical=True)
        evaluator.add_custom_node(False, "second_lagos_name", "Identifies the name of the second Lagos facility", parent=group, critical=True)
        evaluator.add_custom_node(False, "second_lagos_hours", "Provides the documented operating hours for the second Lagos facility", parent=group, critical=True)
        evaluator.add_custom_node(False, "second_lagos_fee", "States the entry fee information for the second Lagos facility", parent=group, critical=True)
        evaluator.add_custom_node(False, "second_lagos_capacity", "Provides the visitor capacity or seating capacity for the second Lagos facility", parent=group, critical=True)
        return

    urls = _sources_or_empty(second.reference_urls)

    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="second_lagos_reference_url",
        desc="Provides a valid reference URL containing verifiable information about the second Lagos facility",
        parent=group,
        critical=True
    )

    claims_and_nodes: List[tuple[str, List[str], Any, str]] = []

    # Facility Name
    if not _has_text(second.name):
        evaluator.add_custom_node(False, "second_lagos_name", "Identifies the name of the second Lagos facility", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="second_lagos_name",
            desc="Identifies the name of the second Lagos facility",
            parent=group,
            critical=True
        )
        claim = f"The page is about a facility named '{second.name}' located in Lagos, Nigeria."
        add_ins = "Confirm the official/commonly used name appears and the location is Lagos, Nigeria."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Operating Hours
    if not _has_text(second.operating_hours):
        evaluator.add_custom_node(False, "second_lagos_hours", "Provides the documented operating hours for the second Lagos facility", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="second_lagos_hours",
            desc="Provides the documented operating hours for the second Lagos facility",
            parent=group,
            critical=True
        )
        nm = second.name or "the facility"
        claim = f"The operating hours for {nm} are: {second.operating_hours}."
        add_ins = "Match the days/times pattern on the page; minor formatting differences are acceptable."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Entry Fee Info
    if not _has_text(second.entry_fee_info):
        evaluator.add_custom_node(False, "second_lagos_fee", "States the entry fee information for the second Lagos facility", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="second_lagos_fee",
            desc="States the entry fee information for the second Lagos facility",
            parent=group,
            critical=True
        )
        nm = second.name or "the facility"
        claim = f"The entry fee information for {nm} is: {second.entry_fee_info}."
        add_ins = "Confirm that the page states this fee information (may include tiers, days, or categories)."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Capacity
    if not _has_text(second.capacity):
        evaluator.add_custom_node(False, "second_lagos_capacity", "Provides the visitor capacity or seating capacity for the second Lagos facility", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="second_lagos_capacity",
            desc="Provides the visitor capacity or seating capacity for the second Lagos facility",
            parent=group,
            critical=True
        )
        nm = second.name or "the facility"
        claim = f"The visitor/seating capacity for {nm} is: {second.capacity}."
        add_ins = "Verify that the page provides a numeric or clearly stated capacity (e.g., maximum, seating capacity)."
        claims_and_nodes.append((claim, urls, node, add_ins))

    if claims_and_nodes:
        await evaluator.batch_verify(claims_and_nodes)


async def verify_canary_park(evaluator: Evaluator, parent) -> None:
    """
    Build verification subtree for the Canary Islands national park.
    """
    extraction_list = evaluator._extraction_results
    park: Optional[CanaryParkInfo] = None
    for item in extraction_list[::-1]:
        if "result" in item and "canary_park" in item["result"]:
            try:
                park = CanaryParkInfo(**item["result"]["canary_park"]) if item["result"]["canary_park"] else None
                break
            except Exception:
                park = None
                break

    group = evaluator.add_parallel(
        id="Canary_Islands_National_Park",
        desc="Provides complete information about the national park in the Canary Islands, Spain",
        parent=parent,
        critical=False
    )

    if park is None:
        evaluator.add_custom_node(False, "canary_reference_url", "Provides a valid reference URL containing verifiable information about the Canary Islands park", parent=group, critical=True)
        evaluator.add_custom_node(False, "canary_park_name", "Identifies the name of the national park in the Canary Islands", parent=group, critical=True)
        evaluator.add_custom_node(False, "canary_island", "Specifies which specific island in the Canary Islands the park is located on", parent=group, critical=True)
        evaluator.add_custom_node(False, "canary_permit_req", "Describes the advance permit or reservation requirements for accessing trails or summit areas", parent=group, critical=True)
        evaluator.add_custom_node(False, "canary_daily_limit", "States the daily visitor capacity limits for specific trails or restricted areas", parent=group, critical=True)
        evaluator.add_custom_node(False, "canary_release_sched", "Explains the schedule for when permits are released, including the specific day of the week and time", parent=group, critical=True)
        return

    urls = _sources_or_empty(park.reference_urls)

    evaluator.add_custom_node(
        result=len(urls) > 0,
        id="canary_reference_url",
        desc="Provides a valid reference URL containing verifiable information about the Canary Islands park",
        parent=group,
        critical=True
    )

    claims_and_nodes: List[tuple[str, List[str], Any, str]] = []

    # Park Name
    if not _has_text(park.park_name):
        evaluator.add_custom_node(False, "canary_park_name", "Identifies the name of the national park in the Canary Islands", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="canary_park_name",
            desc="Identifies the name of the national park in the Canary Islands",
            parent=group,
            critical=True
        )
        claim = f"The national park's name is '{park.park_name}' and it is in the Canary Islands, Spain."
        add_ins = "Confirm the page explicitly names this national park and that it pertains to the Canary Islands."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Island Location
    if not _has_text(park.island):
        evaluator.add_custom_node(False, "canary_island", "Specifies which specific island in the Canary Islands the park is located on", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="canary_island",
            desc="Specifies which specific island in the Canary Islands the park is located on",
            parent=group,
            critical=True
        )
        claim = f"The park is located on the island of {park.island}."
        add_ins = "Verify the page states the specific island (e.g., Tenerife, Lanzarote, La Palma, etc.)."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Permit Requirements
    if not _has_text(park.permit_requirements):
        evaluator.add_custom_node(False, "canary_permit_req", "Describes the advance permit or reservation requirements for accessing trails or summit areas", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="canary_permit_req",
            desc="Describes the advance permit or reservation requirements for accessing trails or summit areas",
            parent=group,
            critical=True
        )
        claim = f"Advance permit/reservation requirements for restricted trails or summit areas are: {park.permit_requirements}."
        add_ins = "Confirm the page clearly states that advance permit or reservations are required (for specific trails/summit areas) and describes how/when to obtain them."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Daily Visitor Limit
    if not _has_text(park.daily_visitor_limit):
        evaluator.add_custom_node(False, "canary_daily_limit", "States the daily visitor capacity limits for specific trails or restricted areas", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="canary_daily_limit",
            desc="States the daily visitor capacity limits for specific trails or restricted areas",
            parent=group,
            critical=True
        )
        claim = f"The daily visitor capacity limit for the specified restricted areas/trails is: {park.daily_visitor_limit}."
        add_ins = "Verify that the page provides a daily quota/capacity (possibly with slot-based allocations)."
        claims_and_nodes.append((claim, urls, node, add_ins))

    # Permit Release Schedule
    if not _has_text(park.permit_release_schedule):
        evaluator.add_custom_node(False, "canary_release_sched", "Explains the schedule for when permits are released, including the specific day of the week and time", parent=group, critical=True)
    else:
        node = evaluator.add_leaf(
            id="canary_release_sched",
            desc="Explains the schedule for when permits are released, including the specific day of the week and time",
            parent=group,
            critical=True
        )
        claim = f"Permits are released on the following schedule: {park.permit_release_schedule}."
        add_ins = "Confirm the page provides the release schedule including day-of-week and time-of-day (with timezone if given). If those specifics are missing, mark incorrect."
        claims_and_nodes.append((claim, urls, node, add_ins))

    if claims_and_nodes:
        await evaluator.batch_verify(claims_and_nodes)


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the outdoor recreation facilities operational information task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Top-level: independent sections
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

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=FacilitiesExtraction,
        extraction_name="facilities_extraction"
    )

    # Build three subtrees according to the rubric
    await verify_first_lagos(evaluator, root)
    await verify_second_lagos(evaluator, root)
    await verify_canary_park(evaluator, root)

    # Return summary
    return evaluator.get_summary()