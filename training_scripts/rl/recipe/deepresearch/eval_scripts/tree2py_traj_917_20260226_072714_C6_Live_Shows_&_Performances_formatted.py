import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_chicago_apr2026"
TASK_DESCRIPTION = """I am a touring production manager for a Broadway show planning a Chicago run in April 2026. I need to identify three different theaters from Broadway in Chicago's network that meet the following requirements:

1. Each theater must be part of Broadway in Chicago's official network of major theaters
2. Each theater must have a seating capacity of at least 1,500 seats
3. Each theater must provide wheelchair accessible seating
4. For each theater, I need to know what show (if any) is currently scheduled there during April 2026

For each of the three theaters, please provide:
- Theater name and street address in Chicago
- Seating capacity
- Confirmation that it offers wheelchair accessible seating
- The show scheduled at that theater during April 2026 (or confirmation of availability)
- Reference URLs confirming all of the above information
"""

TARGET_YEAR = 2026
TARGET_MONTH = 4  # April


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TheaterItem(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    capacity_text: Optional[str] = None
    wheelchair_accessible_text: Optional[str] = None
    april_2026_show: Optional[str] = None

    id_refs: List[str] = Field(default_factory=list)
    capacity_refs: List[str] = Field(default_factory=list)
    accessibility_refs: List[str] = Field(default_factory=list)
    schedule_refs: List[str] = Field(default_factory=list)


class TheatersExtraction(BaseModel):
    theaters: List[TheaterItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theaters() -> str:
    return """
Extract up to three theaters mentioned in the answer. For each theater, extract the following fields strictly from the answer text:

- name: The theater name (string)
- address: The complete street address in Chicago, Illinois (string)
- capacity_text: The seating capacity as written in the answer (string; e.g., "2,300", "2,300 seats", "approx. 2,300")
- wheelchair_accessible_text: The text confirming wheelchair accessible seating, if stated (string; e.g., "wheelchair accessible seating", "ADA seating available", "yes"); if not clearly stated, set null
- april_2026_show: The show scheduled at the theater during April 2026, if provided. If the answer says no show scheduled, available, dark, TBD/TBA, or similar, write that text. If not provided, set null.

- id_refs: A list of URLs explicitly provided in the answer that confirm the identity of the theater (name and address) and/or its association with Broadway In Chicago (e.g., the venue's page on broadwayinchicago.com)
- capacity_refs: A list of URLs explicitly provided in the answer that confirm the seating capacity
- accessibility_refs: A list of URLs explicitly provided in the answer that confirm wheelchair accessible seating
- schedule_refs: A list of URLs explicitly provided in the answer that confirm what is scheduled (or not) during April 2026 for this theater

Rules:
- Only extract URLs that are explicitly present in the answer. Do not invent or infer URLs.
- Return a JSON object with a "theaters" array containing up to three objects with the fields above.
- If any field is missing in the answer for a theater, set it to null (or an empty list for URL lists).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _normalize(s: Optional[str]) -> str:
    return (s or "").strip()


def _is_valid_url(url: str) -> bool:
    return isinstance(url, str) and url.strip().lower().startswith(("http://", "https://"))


def _has_any_valid_url(urls: List[str]) -> bool:
    return any(_is_valid_url(u) for u in urls)


def pick_sources(*lists: List[str]) -> List[str]:
    for lst in lists:
        if lst and _has_any_valid_url(lst):
            # Filter to valid URLs only
            return [u for u in lst if _is_valid_url(u)]
    return []


def parse_capacity_to_int(capacity_text: Optional[str]) -> Optional[int]:
    """
    Extract an integer seat count from a free-form capacity string.
    Examples:
    - "2,300 seats" -> 2300
    - "Approximately 1800" -> 1800
    - "1,500–1,600" -> 1600 (choose the maximum found)
    """
    if not capacity_text:
        return None
    numbers = re.findall(r"(\d{1,3}(?:,\d{3})+|\d+)", capacity_text.replace("\u2009", "").replace("\xa0", " "))
    if not numbers:
        return None
    ints = []
    for n in numbers:
        try:
            ints.append(int(n.replace(",", "")))
        except Exception:
            continue
    if not ints:
        return None
    return max(ints)


def is_show_claim_none(show_text: Optional[str]) -> bool:
    st = _normalize(show_text).lower()
    if not st:
        return True
    none_markers = [
        "none", "no show", "no shows", "no event", "no events", "no performances",
        "available", "dark", "tbd", "tba", "to be determined", "to be announced",
        "unknown", "not announced", "n/a"
    ]
    return any(marker in st for marker in none_markers)


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_theater(
    evaluator: Evaluator,
    parent_node,
    item: TheaterItem,
    index: int,
) -> None:
    """
    Build the verification sub-tree for one theater and execute checks.
    """
    name = _normalize(item.name) or f"Theater #{index + 1}"
    address = _normalize(item.address)

    # Theater parent node (non-critical; allows partial credit across theaters)
    theater_node = evaluator.add_parallel(
        id=f"theater_{index + 1}",
        desc=f"{['First','Second','Third'][index] if index < 3 else f'#{index+1}th'} theater meeting all requirements",
        parent=parent_node,
        critical=False,
    )

    # 1) Identification (critical)
    ident_node = evaluator.add_parallel(
        id=f"theater_{index + 1}_identification",
        desc="Theater is correctly identified with all required details",
        parent=theater_node,
        critical=True,
    )

    # 1.a) Name + address leaf
    name_addr_leaf = evaluator.add_leaf(
        id=f"theater_{index + 1}_name_location",
        desc="Provide the theater name and complete street address in Chicago",
        parent=ident_node,
        critical=True,
    )
    name_addr_claim = (
        f"The theater named '{name}' has the street address '{address}' in Chicago, Illinois."
        if address else
        f"The theater named '{name}' is located in Chicago, Illinois, at the street address '{address}'."
    )
    await evaluator.verify(
        claim=name_addr_claim,
        node=name_addr_leaf,
        sources=pick_sources(item.id_refs),
        additional_instruction=(
            "Verify that the page shows the same theater name and its street address in Chicago, IL. "
            "Allow minor formatting differences (e.g., punctuation, abbreviations like 'St.' vs 'Street')."
        ),
    )

    # 1.b) Broadway In Chicago network membership leaf
    bic_leaf = evaluator.add_leaf(
        id=f"theater_{index + 1}_broadway_network",
        desc="Verify the theater is part of Broadway in Chicago's official network of major theaters",
        parent=ident_node,
        critical=True,
    )
    bic_claim = (
        f"The theater '{name}' is part of Broadway In Chicago's official network of major theaters."
    )
    await evaluator.verify(
        claim=bic_claim,
        node=bic_leaf,
        sources=pick_sources(item.id_refs),
        additional_instruction=(
            "Look for explicit mention on Broadway In Chicago or an official venue page indicating that the theater "
            "is one of the Broadway In Chicago venues (e.g., on broadwayinchicago.com 'Our Theatres' pages)."
        ),
    )

    # 1.c) Identification reference presence (custom existence check)
    evaluator.add_custom_node(
        result=_has_any_valid_url(item.id_refs),
        id=f"theater_{index + 1}_identification_reference",
        desc="Provide a reference URL confirming the theater's identity and Broadway in Chicago association",
        parent=ident_node,
        critical=True,
    )

    # 2) Capacity (critical)
    cap_node = evaluator.add_parallel(
        id=f"theater_{index + 1}_capacity",
        desc="Theater meets the minimum seating capacity requirement",
        parent=theater_node,
        critical=True,
    )

    # 2.a) Capacity value leaf (source-verified)
    cap_value_leaf = evaluator.add_leaf(
        id=f"theater_{index + 1}_capacity_value",
        desc="Provide the theater's exact seating capacity number",
        parent=cap_node,
        critical=True,
    )
    capacity_text = _normalize(item.capacity_text)
    cap_value_claim = f"The seating capacity of the theater '{name}' is '{capacity_text}'."
    await evaluator.verify(
        claim=cap_value_claim,
        node=cap_value_leaf,
        sources=pick_sources(item.capacity_refs, item.id_refs),
        additional_instruction=(
            "Check the page for the theater's stated seating capacity. Minor phrasing differences like including the word "
            "'seats' or commas in numbers are acceptable as long as the number matches."
        ),
    )

    # 2.b) Capacity minimum >= 1,500 (custom numeric check)
    parsed_capacity = parse_capacity_to_int(capacity_text)
    evaluator.add_custom_node(
        result=(parsed_capacity is not None and parsed_capacity >= 1500),
        id=f"theater_{index + 1}_capacity_minimum",
        desc="Verify the capacity is at least 1,500 seats",
        parent=cap_node,
        critical=True,
    )

    # 2.c) Capacity reference presence (custom existence check)
    evaluator.add_custom_node(
        result=_has_any_valid_url(item.capacity_refs),
        id=f"theater_{index + 1}_capacity_reference",
        desc="Provide a reference URL confirming the seating capacity",
        parent=cap_node,
        critical=True,
    )

    # 3) Accessibility (critical)
    acc_node = evaluator.add_parallel(
        id=f"theater_{index + 1}_accessibility",
        desc="Theater provides required accessibility features",
        parent=theater_node,
        critical=True,
    )

    # 3.a) Wheelchair accessible seating leaf
    wheelchair_leaf = evaluator.add_leaf(
        id=f"theater_{index + 1}_wheelchair_seating",
        desc="Confirm the theater offers wheelchair accessible seating",
        parent=acc_node,
        critical=True,
    )
    wheelchair_claim = f"The theater '{name}' offers wheelchair accessible seating."
    await evaluator.verify(
        claim=wheelchair_claim,
        node=wheelchair_leaf,
        sources=pick_sources(item.accessibility_refs, item.id_refs),
        additional_instruction=(
            "Look for accessibility details mentioning wheelchair accessible seating, ADA seating, "
            "companion seating, or similar terms on the venue or official pages."
        ),
    )

    # 3.b) Accessibility reference presence (custom existence check)
    evaluator.add_custom_node(
        result=_has_any_valid_url(item.accessibility_refs),
        id=f"theater_{index + 1}_accessibility_reference",
        desc="Provide a reference URL confirming accessibility features",
        parent=acc_node,
        critical=True,
    )

    # 4) Schedule for April 2026 (critical)
    sched_node = evaluator.add_parallel(
        id=f"theater_{index + 1}_april_schedule",
        desc="Information about April 2026 schedule",
        parent=theater_node,
        critical=True,
    )

    # 4.a) Schedule info leaf
    schedule_leaf = evaluator.add_leaf(
        id=f"theater_{index + 1}_schedule_info",
        desc="Identify what show (if any) is scheduled at the theater during April 2026",
        parent=sched_node,
        critical=True,
    )
    show_text = _normalize(item.april_2026_show)
    if is_show_claim_none(show_text):
        schedule_claim = (
            f"As of now, there is no show scheduled at '{name}' during April {TARGET_YEAR} "
            f"(the venue is available or listed as dark/TBA)."
        )
        schedule_instruction = (
            "Check the provided schedule or calendar page(s) for the venue for April 2026. "
            "If the page shows no events, is marked 'dark', 'TBA', or otherwise indicates no performances "
            "in April 2026, consider this claim supported."
        )
    else:
        schedule_claim = (
            f"In April {TARGET_YEAR}, the show '{show_text}' is scheduled to perform at '{name}'."
        )
        schedule_instruction = (
            "Verify the venue's schedule/calendar or the show's listing indicates performances in April 2026 at this venue. "
            "Accept if the listing shows a date range that includes April 2026 or specific April 2026 performance dates."
        )

    await evaluator.verify(
        claim=schedule_claim,
        node=schedule_leaf,
        sources=pick_sources(item.schedule_refs),
        additional_instruction=schedule_instruction,
    )

    # 4.b) Schedule reference presence (custom existence check)
    evaluator.add_custom_node(
        result=_has_any_valid_url(item.schedule_refs),
        id=f"theater_{index + 1}_schedule_reference",
        desc="Provide a reference URL confirming the April 2026 schedule information",
        parent=sched_node,
        critical=True,
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
    Evaluate an answer for the Broadway In Chicago April 2026 theater requirements.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Find three Broadway theaters in Chicago that meet all specified requirements for hosting a touring production in April 2026",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract up to three theaters from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_theaters(),
        template_class=TheatersExtraction,
        extraction_name="theaters_extraction",
    )

    # Keep only the first 3 theaters; pad if fewer
    theaters = list(extracted.theaters[:3])
    while len(theaters) < 3:
        theaters.append(TheaterItem())

    # Add a small piece of custom info for the evaluation context
    evaluator.add_custom_info(
        info={"target_year": TARGET_YEAR, "target_month": TARGET_MONTH, "network": "Broadway In Chicago"},
        info_type="context",
        info_name="evaluation_context",
    )

    # Build verification tree for each theater
    for idx in range(3):
        await verify_theater(evaluator, root, theaters[idx], idx)

    # Return final structured summary
    return evaluator.get_summary()