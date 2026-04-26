import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "caribbean_newest_ships_rc_celebrity"
TASK_DESCRIPTION = """
You are planning a Caribbean cruise vacation and want to sail on the newest ships from two major cruise lines. Identify one cruise ship from each of the following cruise lines that meets all the specified criteria:

Criteria:
- Royal Caribbean International ship: Must have a double occupancy passenger capacity of at least 5,000 passengers, must have debuted (entered service) in 2023 or 2024, and must homeport in either Miami, Florida or Fort Lauderdale, Florida
- Celebrity Cruises ship: Must have debuted (entered service) in 2023 or 2024 and must homeport in either Miami, Florida or Fort Lauderdale, Florida
- Both ships must offer Caribbean cruise itineraries

For each ship, provide:
1. Ship name
2. Year the ship debuted (entered service)
3. Double occupancy passenger capacity
4. Homeport city (Miami or Fort Lauderdale)
5. If the homeport is Miami, identify the specific PortMiami terminal the ship uses
6. One official source URL (from the cruise line or port authority) that verifies this information
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ShipInfo(BaseModel):
    """Information for a single ship."""
    name: Optional[str] = None
    debut_year: Optional[str] = None  # keep string to be robust to formats
    capacity_double_occupancy: Optional[str] = None  # string to allow ranges/text
    homeport_city: Optional[str] = None  # e.g., "Miami, Florida" or "Fort Lauderdale, Florida" or Port Everglades
    portmiami_terminal_if_miami: Optional[str] = None  # e.g., "Terminal A"
    source_url: Optional[str] = None  # exactly one official source URL as provided in the answer
    itinerary_regions: List[str] = Field(default_factory=list)  # include "Caribbean" if stated


class CruiseSelectionExtraction(BaseModel):
    """Ships extracted from the answer, one per cruise line."""
    rc_ship: Optional[ShipInfo] = None
    cc_ship: Optional[ShipInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_ships() -> str:
    return """
    Extract one Royal Caribbean International ship and one Celebrity Cruises ship as presented in the answer text.
    For each ship, extract the following fields exactly as stated:

    Royal Caribbean International ("rc_ship"):
    - name
    - debut_year (the year entered service)
    - capacity_double_occupancy (double-occupancy passenger capacity)
    - homeport_city (must be Miami, Florida or Fort Lauderdale, Florida; if answer uses "Port Everglades", treat it as Fort Lauderdale, Florida)
    - portmiami_terminal_if_miami (the terminal at PortMiami if homeport is Miami; otherwise return null)
    - source_url (exactly one official source URL from the cruise line OR a port authority page; if multiple URLs are given, return the first official one)
    - itinerary_regions (list of regions mentioned in itineraries for this ship; include "Caribbean" if stated)

    Celebrity Cruises ("cc_ship"):
    - name
    - debut_year (the year entered service)
    - capacity_double_occupancy (double-occupancy passenger capacity)
    - homeport_city (must be Miami, Florida or Fort Lauderdale, Florida; if answer uses "Port Everglades", treat it as Fort Lauderdale, Florida)
    - portmiami_terminal_if_miami (the terminal at PortMiami if homeport is Miami; otherwise return null)
    - source_url (exactly one official source URL from the cruise line OR a port authority page; if multiple URLs are given, return the first official one)
    - itinerary_regions (list of regions mentioned in itineraries for this ship; include "Caribbean" if stated)

    Rules:
    - Extract only information explicitly present in the answer. Do not invent or infer values.
    - If any field is not present for a ship, return null (or empty list for itinerary_regions).
    - For homeport_city, allow synonyms: "Port Everglades" corresponds to "Fort Lauderdale, Florida".
    - For source_url, prefer these official domains when available: royalcaribbean.com, celebritycruises.com, miamidade.gov/portmiami (or portmiami.miamidade.gov), porteverglades.net, broward.org/port (or broward.org/porteverglades).
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def parse_int_loose(text: Optional[str]) -> Optional[int]:
    """Extract the largest integer from a string (e.g., 'Capacity 5,734 (double)' -> 5734)."""
    if not text:
        return None
    nums = re.findall(r"\d{1,6}", text.replace(",", ""))
    if not nums:
        return None
    # choose the largest number; often capacity is the largest relevant integer
    return max(int(n) for n in nums)


def normalize_homeport(city_text: Optional[str]) -> Optional[str]:
    """Normalize homeport to 'Miami' or 'Fort Lauderdale' if possible."""
    if not city_text:
        return None
    s = city_text.lower()
    if "miami" in s or "portmiami" in s or "port miami" in s:
        return "Miami"
    if "fort lauderdale" in s or "port everglades" in s:
        return "Fort Lauderdale"
    return None


def is_miami_homeport(city_text: Optional[str]) -> bool:
    return normalize_homeport(city_text) == "Miami"


def official_source_instruction() -> str:
    return (
        "Treat the source as official only if the URL belongs to one of these authorities: "
        "royalcaribbean.com (Royal Caribbean), celebritycruises.com (Celebrity Cruises), "
        "miamidade.gov/portmiami or portmiami.miamidade.gov (PortMiami), "
        "porteverglades.net or broward.org/port (Port Everglades / Broward County). "
        "If the URL is not from these domains or an equivalent official authority site, "
        "then consider the source non-official. Verify that the page supports the claimed ship details."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_ship(
    evaluator: Evaluator,
    parent_node,
    ship: Optional[ShipInfo],
    cruise_line: str,
    require_min_capacity_5000: bool,
    node_id_prefix: str
) -> None:
    """
    Build verification nodes and run checks for one ship.

    cruise_line: "Royal Caribbean International" or "Celebrity Cruises"
    require_min_capacity_5000: True for RC, False for Celebrity
    node_id_prefix: "rc" or "cc"
    """
    # Create the ship group node (parallel aggregation)
    group_node = evaluator.add_parallel(
        id="royal_caribbean_ship" if node_id_prefix == "rc" else "celebrity_ship",
        desc=f"{cruise_line} ship meeting all specified criteria and required fields are provided",
        parent=parent_node,
        critical=False  # allow partial scoring per ship independently
    )

    # Extract fields robustly
    name = ship.name if ship else None
    debut_year = ship.debut_year if ship else None
    capacity_str = ship.capacity_double_occupancy if ship else None
    capacity_val = parse_int_loose(capacity_str)
    homeport_raw = ship.homeport_city if ship else None
    homeport_norm = normalize_homeport(homeport_raw)
    terminal_if_miami = ship.portmiami_terminal_if_miami if ship else None
    source_url = ship.source_url if ship else None
    itin_regions = ship.itinerary_regions if ship and ship.itinerary_regions else []

    # 1) Ship name is provided (critical)
    evaluator.add_custom_node(
        result=(name is not None and name.strip() != ""),
        id=f"{node_id_prefix}_ship_name",
        desc="Ship name is provided",
        parent=group_node,
        critical=True
    )

    # 2) Debut year is 2023 or 2024 (critical) - verify via source if available; otherwise simple check
    debut_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_debut_year",
        desc="Ship debuted (entered service) in 2023 or 2024",
        parent=group_node,
        critical=True
    )
    # Build claim: prefer specific year if provided; otherwise generic constraint
    if debut_year and re.search(r"2023|2024", debut_year):
        claim_debut = f"The ship '{name or 'UNKNOWN'}' debuted (entered service) in {debut_year}, which is in 2023 or 2024."
    else:
        claim_debut = f"The ship '{name or 'UNKNOWN'}' debuted (entered service) in 2023 or 2024."
    await evaluator.verify(
        claim=claim_debut,
        node=debut_leaf,
        sources=source_url,
        additional_instruction=(
            "Check the page for delivery, debut, entered service, or maiden voyage timing. "
            "Accept synonyms like 'entered service', 'debut', 'maiden voyage', or 'launched' if clearly tied to year. "
            "The year must be 2023 or 2024. " + official_source_instruction()
        )
    )

    # 3) Capacity provided and (for RC) at least 5,000 (critical) - treat as custom constraint check
    if require_min_capacity_5000:
        evaluator.add_custom_node(
            result=(capacity_str is not None and capacity_str.strip() != "" and (capacity_val is not None and capacity_val >= 5000)),
            id=f"{node_id_prefix}_capacity",
            desc="Ship’s double-occupancy passenger capacity is provided and is at least 5,000",
            parent=group_node,
            critical=True
        )
    else:
        evaluator.add_custom_node(
            result=(capacity_str is not None and capacity_str.strip() != ""),
            id=f"{node_id_prefix}_capacity",
            desc="Ship’s double-occupancy passenger capacity is provided",
            parent=group_node,
            critical=True
        )

    # 4) Homeport city provided and either Miami or Fort Lauderdale (critical)
    evaluator.add_custom_node(
        result=(homeport_raw is not None and homeport_raw.strip() != "" and homeport_norm in {"Miami", "Fort Lauderdale"}),
        id=f"{node_id_prefix}_homeport",
        desc="Ship homeport city is provided and is either Miami, Florida or Fort Lauderdale, Florida",
        parent=group_node,
        critical=True
    )

    # 5) If homeport is Miami, provide PortMiami terminal (critical)
    # Pass if homeport is Fort Lauderdale; require terminal if Miami
    evaluator.add_custom_node(
        result=(homeport_norm == "Fort Lauderdale") or (homeport_norm == "Miami" and terminal_if_miami is not None and terminal_if_miami.strip() != ""),
        id=f"{node_id_prefix}_terminal_if_miami",
        desc="If the ship’s homeport is Miami, the specific PortMiami terminal is identified",
        parent=group_node,
        critical=True
    )

    # 6) Ship offers Caribbean itineraries (critical) - verify via source if available
    itin_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_itinerary",
        desc="Ship offers Caribbean cruise itineraries",
        parent=group_node,
        critical=True
    )
    # Build claim
    if "caribbean" in [r.lower() for r in itin_regions]:
        claim_itin = f"The ship '{name or 'UNKNOWN'}' offers Caribbean cruise itineraries."
    else:
        claim_itin = f"The ship '{name or 'UNKNOWN'}' has itineraries that include the Caribbean region."
    await evaluator.verify(
        claim=claim_itin,
        node=itin_leaf,
        sources=source_url,
        additional_instruction=(
            "Look for itinerary pages or schedule indicating 'Caribbean' (Eastern/Western/Southern Caribbean). "
            "If the page clearly shows Caribbean itineraries, support the claim; otherwise do not. " + official_source_instruction()
        )
    )

    # 7) Exactly one official source URL provided that verifies stated information (critical)
    # If source is missing, fail this node explicitly; otherwise verify a summary of details.
    source_leaf = evaluator.add_leaf(
        id=f"{node_id_prefix}_source",
        desc="Exactly one official source URL (cruise line or port authority) is provided that verifies the stated ship information",
        parent=group_node,
        critical=True
    )

    if source_url and source_url.strip():
        # Build summary claim that the official page supports the key facts.
        # Only include terminal in the claim if Miami.
        parts = []
        if name:
            parts.append(f"ship name '{name}'")
        if debut_year:
            parts.append(f"debut year '{debut_year}'")
        if capacity_str:
            parts.append(f"double-occupancy capacity '{capacity_str}'")
        if homeport_norm:
            parts.append(f"homeport '{homeport_norm}'")
        if homeport_norm == "Miami" and terminal_if_miami:
            parts.append(f"PortMiami terminal '{terminal_if_miami}'")
        parts.append("Caribbean itineraries")

        summary = "; ".join(parts)
        claim_source = (
            f"The provided page is an official source and it supports the following information for the {cruise_line} ship: {summary}."
        )
        await evaluator.verify(
            claim=claim_source,
            node=source_leaf,
            sources=source_url,
            additional_instruction=(
                "Confirm the URL is official (cruise line or port authority). "
                "Then determine whether the page supports the listed facts (name, debut year, capacity, homeport, PortMiami terminal if applicable, and Caribbean itineraries). "
                "If some items are not supported on this page, mark as not supported. " + official_source_instruction()
            )
        )
    else:
        # No source provided -> fail
        source_leaf.score = 0.0
        source_leaf.status = "failed"


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
    Evaluate an answer for the Caribbean newest ships task:
    - One Royal Caribbean International ship (>=5000 capacity, debuted 2023/2024, homeport Miami or Fort Lauderdale, Caribbean itineraries)
    - One Celebrity Cruises ship (debuted 2023/2024, homeport Miami or Fort Lauderdale, Caribbean itineraries)
    - Provide required fields and one official source for each.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # independent checking of the two ships
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

    # Extract ships from the answer
    extraction: CruiseSelectionExtraction = await evaluator.extract(
        prompt=prompt_extract_ships(),
        template_class=CruiseSelectionExtraction,
        extraction_name="cruise_selection",
    )

    # Build verification for Royal Caribbean ship
    await verify_ship(
        evaluator=evaluator,
        parent_node=root,
        ship=extraction.rc_ship,
        cruise_line="Royal Caribbean International",
        require_min_capacity_5000=True,
        node_id_prefix="rc"
    )

    # Build verification for Celebrity Cruises ship
    await verify_ship(
        evaluator=evaluator,
        parent_node=root,
        ship=extraction.cc_ship,
        cruise_line="Celebrity Cruises",
        require_min_capacity_5000=False,
        node_id_prefix="cc"
    )

    # Return structured summary
    return evaluator.get_summary()