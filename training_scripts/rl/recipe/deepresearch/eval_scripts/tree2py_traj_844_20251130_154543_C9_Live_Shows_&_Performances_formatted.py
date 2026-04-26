import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "broadway_four_theaters"
TASK_DESCRIPTION = """Identify four distinct Broadway theaters that meet the following specific criteria. For each theater, provide its name, exact seating capacity, street address, and supporting reference URLs.

Theater 1 - Largest Broadway Theater:
Identify the Broadway theater with the highest seating capacity among all Broadway theaters. Provide:
- Theater name
- Exact seating capacity
- Street address confirming it is located within the Theater District (between 41st and 54th Streets, and between 6th and 8th Avenues in Manhattan)
- Name of the show currently playing at this theater as of November 2024
- Reference URL confirming the seating capacity
- Reference URL confirming the theater location
- Reference URL confirming the current show

Theater 2 - Smallest Broadway Theater:
Identify the Broadway theater with the lowest seating capacity while still meeting the Broadway definition (minimum 500 seats). Provide:
- Theater name
- Exact seating capacity
- Street address confirming it is located within the Theater District boundaries
- Confirmation that this capacity meets the minimum Broadway requirement
- Reference URL confirming the seating capacity
- Reference URL confirming the theater location
- Reference URL confirming this is identified as the smallest Broadway theater

Theater 3 - Second-Largest Broadway Theater:
Identify the Broadway theater with the second-highest seating capacity among all Broadway theaters. Provide:
- Theater name
- Exact seating capacity (expected to be between 1,700-1,800 seats)
- Street address confirming it is located within the Theater District boundaries
- Confirmation that this is the second-largest Broadway theater
- Reference URL confirming the seating capacity
- Reference URL confirming the theater location
- Reference URL confirming the theater's ranking as second-largest

Theater 4 - Theater Hosting Longest-Running Current Show:
Identify the Broadway theater currently hosting the show with the longest continuous run still playing on Broadway (the show that opened in 1996 and is still running). Provide:
- Theater name
- Name of the show currently running
- Opening date of the show (must be in 1996)
- Street address confirming the theater is located within the Theater District boundaries
- Confirmation that the theater meets the Broadway definition (500+ seats)
- Reference URL confirming the show is currently at this theater
- Reference URL confirming the opening date and longest-running status
- Reference URL confirming the theater location

All theaters must be located within the official Broadway Theater District boundaries (41st-54th Streets, 6th-8th Avenues in Manhattan) and must meet the Broadway theater definition of having 500 or more seats.
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TheaterItem(BaseModel):
    name: Optional[str] = None

    capacity: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)

    address: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)

    # Current show information (only required for Theater 1 and Theater 4, but allowed for all)
    show_name: Optional[str] = None
    show_urls: List[str] = Field(default_factory=list)

    # Ranking/evidence URLs (largest / smallest / second-largest)
    ranking_urls: List[str] = Field(default_factory=list)

    # For Theater 4 (longest-running current show)
    opening_date: Optional[str] = None
    longest_running_urls: List[str] = Field(default_factory=list)


class TheatersExtraction(BaseModel):
    theater1: Optional[TheaterItem] = None  # Largest by capacity
    theater2: Optional[TheaterItem] = None  # Smallest but >=500
    theater3: Optional[TheaterItem] = None  # Second-largest
    theater4: Optional[TheaterItem] = None  # Hosting longest-running current show (opened in 1996)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_theaters() -> str:
    return """
    Extract exactly four theater entries from the answer corresponding to these roles:
    - theater1: Largest Broadway theater by seating capacity among all Broadway theaters
    - theater2: Smallest Broadway theater by seating capacity that still meets Broadway minimum (>=500 seats)
    - theater3: Second-largest Broadway theater by seating capacity
    - theater4: The Broadway theater currently hosting the longest-running show still playing on Broadway (the show opened in 1996 and continues running)

    For each theater (theater1..theater4), extract the following fields (return null if missing and [] for missing URL lists):
    - name: Theater name as given
    - capacity: Exact seating capacity string as stated in the answer (e.g., "1,933", "1700")
    - capacity_urls: Array of URL(s) the answer cites to support the capacity
    - address: Full street address as stated
    - location_urls: Array of URL(s) the answer cites to support the address/location
    - show_name: If the answer specifies a current show for this theater, provide the show name; otherwise null
    - show_urls: Array of URL(s) to support the current show at this theater (e.g., the theater’s official page, Playbill, Broadway.org listing, etc.)
    - ranking_urls: Array of URL(s) that support any ranking claim (largest, smallest, second-largest) when applicable
    - opening_date: If the answer specifies an opening date for the show (for theater4), provide the date string from the answer; otherwise null
    - longest_running_urls: Array of URL(s) that support the claim that the show (for theater4) opened in 1996 and is the longest-running show still playing on Broadway

    Special rules:
    - Extract only URLs explicitly present in the answer (including within markdown links). Do not invent URLs.
    - If the answer lists multiple relevant URLs for a field, include them all.
    - Preserve capacity as a string; do not convert to a number.
    - Theaters should be Broadway houses (>=500 seats). The answer may already ensure this, but still extract what's provided.

    Return a JSON object with keys: theater1, theater2, theater3, theater4; each should be an object with the fields above.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_list(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]


def _combine_urls(*lists: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in _safe_list(lst):
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def _display_name(name: Optional[str]) -> str:
    return name.strip() if name else "the theater"


def _address_within_bounds_claim(address: Optional[str]) -> str:
    addr = address.strip() if address else "the provided address"
    return (
        f"The address '{addr}' is within Manhattan's Broadway Theater District boundaries: "
        f"between 41st and 54th Streets and between 6th and 8th Avenues."
    )


async def _verify_with_required_sources(
    evaluator: Evaluator,
    node,
    claim: str,
    sources: List[str],
    additional_instruction: str = "None",
) -> bool:
    """
    Verify a claim requiring URLs. If no sources are provided, mark node as failed and return False.
    """
    if not sources:
        node.score = 0.0
        node.status = "failed"
        return False
    return await evaluator.verify(
        claim=claim,
        node=node,
        sources=sources,
        additional_instruction=additional_instruction
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def _build_theater_1(evaluator: Evaluator, parent, t: Optional[TheaterItem]) -> None:
    """
    Theater 1: Largest Broadway theater by seating capacity; include current show (as of Nov 2024).
    """
    node_t1 = evaluator.add_parallel(
        id="theater_1",
        desc="Theater 1: Largest Broadway theater by seating capacity; include current show as of Nov 2024; include required attributes and references.",
        parent=parent,
        critical=False
    )

    # Name (critical)
    name_exists = bool(t and t.name and t.name.strip())
    evaluator.add_custom_node(
        result=name_exists,
        id="theater_1_name",
        desc="Provide the theater name.",
        parent=node_t1,
        critical=True
    )
    name_disp = _display_name(t.name if t else None)

    # Largest by capacity (critical) - try ranking URLs first, fall back to capacity URLs
    is_largest_leaf = evaluator.add_leaf(
        id="theater_1_is_largest_by_capacity",
        desc="The identified theater is the Broadway theater with the highest seating capacity among all Broadway theaters.",
        parent=node_t1,
        critical=True
    )
    largest_sources = _combine_urls(t.ranking_urls if t else None, t.capacity_urls if t else None)
    largest_claim = f"{name_disp} is the Broadway theater with the highest seating capacity among all Broadway theaters."
    await _verify_with_required_sources(
        evaluator,
        is_largest_leaf,
        largest_claim,
        largest_sources,
        additional_instruction="Focus on an explicit ranking statement (largest by seating capacity) among Broadway houses. Accept authoritative sources."
    )

    # Capacity group (critical)
    cap_group = evaluator.add_parallel(
        id="theater_1_capacity",
        desc="Provide the exact seating capacity and a reference URL confirming it.",
        parent=node_t1,
        critical=True
    )
    # Capacity value exists (critical)
    evaluator.add_custom_node(
        result=bool(t and t.capacity and t.capacity.strip()),
        id="theater_1_capacity_value",
        desc="State the exact seating capacity.",
        parent=cap_group,
        critical=True
    )
    # Capacity reference supports stated value (critical)
    cap_ref_leaf = evaluator.add_leaf(
        id="theater_1_capacity_reference",
        desc="Provide a reference URL confirming the seating capacity.",
        parent=cap_group,
        critical=True
    )
    cap_claim = f"The seating capacity of {name_disp} is {t.capacity.strip()} seats." if t and t.capacity else f"The seating capacity of {name_disp} is as stated."
    await _verify_with_required_sources(
        evaluator,
        cap_ref_leaf,
        cap_claim,
        _safe_list(t.capacity_urls if t else None),
        additional_instruction="Verify the capacity number stated in the answer matches the capacity shown on the source page."
    )

    # Location group (critical)
    loc_group = evaluator.add_parallel(
        id="theater_1_location",
        desc="Provide the street address and location evidence.",
        parent=node_t1,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.address and t.address.strip()),
        id="theater_1_address",
        desc="Provide the street address.",
        parent=loc_group,
        critical=True
    )
    loc_ref_leaf = evaluator.add_leaf(
        id="theater_1_location_reference",
        desc="Provide a reference URL confirming the theater address/location.",
        parent=loc_group,
        critical=True
    )
    loc_claim = f"The street address of {name_disp} is '{t.address.strip()}'." if t and t.address else f"The street address of {name_disp} is as stated."
    await _verify_with_required_sources(
        evaluator,
        loc_ref_leaf,
        loc_claim,
        _safe_list(t.location_urls if t else None),
        additional_instruction="The page should list the theater address or clearly identify the same location."
    )
    within_bounds_leaf = evaluator.add_leaf(
        id="theater_1_within_district_bounds",
        desc="Confirm (based on the provided address/location evidence) that the theater is within the Theater District boundaries (41st–54th Streets, 6th–8th Avenues, Manhattan).",
        parent=loc_group,
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        within_bounds_leaf,
        _address_within_bounds_claim(t.address if t else None),
        _safe_list(t.location_urls if t else None),
        additional_instruction="Use the explicit street (41st–54th) and avenues (6th–8th) boundaries. It is acceptable to infer from the address text (e.g., 'W 49th St' implies within 41st–54th). Do not rely on knowledge beyond the page content and the boundaries definition."
    )

    # Broadway min seats (critical) - verify 500+ via capacity reference
    min_seats_leaf = evaluator.add_leaf(
        id="theater_1_broadway_min_seats",
        desc="Confirm the theater meets the Broadway definition minimum of 500+ seats (can be verified from the stated capacity).",
        parent=node_t1,
        critical=True
    )
    min_claim = f"{name_disp} has at least 500 seats (meets Broadway definition)."  # The page should show capacity >= 500
    await _verify_with_required_sources(
        evaluator,
        min_seats_leaf,
        min_claim,
        _safe_list(t.capacity_urls if t else None),
        additional_instruction="Confirm from the capacity on the page that it is 500 or more."
    )

    # Current show as of Nov 2024 (critical)
    show_group = evaluator.add_parallel(
        id="theater_1_current_show",
        desc="Identify the show currently playing at this theater as of November 2024 and provide a reference URL.",
        parent=node_t1,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.show_name and t.show_name.strip()),
        id="theater_1_show_name",
        desc="Provide the current show name (as of Nov 2024).",
        parent=show_group,
        critical=True
    )
    show_ref_leaf = evaluator.add_leaf(
        id="theater_1_show_reference",
        desc="Provide a reference URL confirming the current show at this theater as of Nov 2024.",
        parent=show_group,
        critical=True
    )
    show_claim = (
        f"As of November 2024, the show '{t.show_name.strip()}' is currently playing at {name_disp}."
        if t and t.show_name else
        f"As of November 2024, the show mentioned is currently playing at {name_disp}."
    )
    await _verify_with_required_sources(
        evaluator,
        show_ref_leaf,
        show_claim,
        _safe_list(t.show_urls if t else None),
        additional_instruction="The source should clearly indicate that this show is currently at the specified theater around November 2024."
    )


async def _build_theater_2(evaluator: Evaluator, parent, t: Optional[TheaterItem]) -> None:
    """
    Theater 2: Smallest Broadway theater by seating capacity (but >=500).
    """
    node_t2 = evaluator.add_parallel(
        id="theater_2",
        desc="Theater 2: Smallest Broadway theater by seating capacity (but still 500+ seats); include required attributes and references.",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(t and t.name and t.name.strip()),
        id="theater_2_name",
        desc="Provide the theater name.",
        parent=node_t2,
        critical=True
    )
    name_disp = _display_name(t.name if t else None)

    # Capacity group (critical)
    cap_group = evaluator.add_parallel(
        id="theater_2_capacity",
        desc="Provide the exact seating capacity and a reference URL confirming it.",
        parent=node_t2,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.capacity and t.capacity.strip()),
        id="theater_2_capacity_value",
        desc="State the exact seating capacity.",
        parent=cap_group,
        critical=True
    )
    cap_ref_leaf = evaluator.add_leaf(
        id="theater_2_capacity_reference",
        desc="Provide a reference URL confirming the seating capacity.",
        parent=cap_group,
        critical=True
    )
    cap_claim = f"The seating capacity of {name_disp} is {t.capacity.strip()} seats." if t and t.capacity else f"The seating capacity of {name_disp} is as stated."
    await _verify_with_required_sources(
        evaluator,
        cap_ref_leaf,
        cap_claim,
        _safe_list(t.capacity_urls if t else None),
        additional_instruction="Verify the capacity number stated in the answer matches the capacity shown on the source page."
    )

    # Ranking: smallest by capacity (critical)
    smallest_leaf = evaluator.add_leaf(
        id="theater_2_smallest_by_capacity_reference",
        desc="Provide a reference URL supporting that this is the smallest Broadway theater by seating capacity (while still qualifying as Broadway).",
        parent=node_t2,
        critical=True
    )
    smallest_claim = f"{name_disp} is the smallest Broadway theater by seating capacity, while still meeting the 500+ seat requirement."
    await _verify_with_required_sources(
        evaluator,
        smallest_leaf,
        smallest_claim,
        _safe_list(t.ranking_urls if t else None),
        additional_instruction="Look for explicit statements that this Broadway house has the lowest seating capacity among Broadway theaters (>=500 seats)."
    )

    # Location group (critical)
    loc_group = evaluator.add_parallel(
        id="theater_2_location",
        desc="Provide the street address and location evidence.",
        parent=node_t2,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.address and t.address.strip()),
        id="theater_2_address",
        desc="Provide the street address.",
        parent=loc_group,
        critical=True
    )
    loc_ref_leaf = evaluator.add_leaf(
        id="theater_2_location_reference",
        desc="Provide a reference URL confirming the theater address/location.",
        parent=loc_group,
        critical=True
    )
    loc_claim = f"The street address of {name_disp} is '{t.address.strip()}'." if t and t.address else f"The street address of {name_disp} is as stated."
    await _verify_with_required_sources(
        evaluator,
        loc_ref_leaf,
        loc_claim,
        _safe_list(t.location_urls if t else None),
        additional_instruction="The page should list the theater address or clearly identify the same location."
    )
    within_bounds_leaf = evaluator.add_leaf(
        id="theater_2_within_district_bounds",
        desc="Confirm (based on the provided address/location evidence) that the theater is within the Theater District boundaries (41st–54th Streets, 6th–8th Avenues, Manhattan).",
        parent=loc_group,
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        within_bounds_leaf,
        _address_within_bounds_claim(t.address if t else None),
        _safe_list(t.location_urls if t else None),
        additional_instruction="Use the explicit street (41st–54th) and avenues (6th–8th) boundaries. It is acceptable to infer from the address text (e.g., 'W 45th St' implies within 41st–54th)."
    )

    # Broadway min seats (critical)
    min_seats_leaf = evaluator.add_leaf(
        id="theater_2_broadway_min_seats",
        desc="Confirm the theater meets the Broadway definition minimum of 500+ seats (can be verified from the stated capacity).",
        parent=node_t2,
        critical=True
    )
    min_claim = f"{name_disp} has at least 500 seats (meets Broadway definition)."
    await _verify_with_required_sources(
        evaluator,
        min_seats_leaf,
        min_claim,
        _safe_list(t.capacity_urls if t else None),
        additional_instruction="Confirm from the capacity on the page that it is 500 or more."
    )


async def _build_theater_3(evaluator: Evaluator, parent, t: Optional[TheaterItem]) -> None:
    """
    Theater 3: Second-largest Broadway theater by seating capacity; capacity expected 1,700–1,800 (non-critical range check).
    """
    node_t3 = evaluator.add_parallel(
        id="theater_3",
        desc="Theater 3: Second-largest Broadway theater by seating capacity; include required attributes and references.",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(t and t.name and t.name.strip()),
        id="theater_3_name",
        desc="Provide the theater name.",
        parent=node_t3,
        critical=True
    )
    name_disp = _display_name(t.name if t else None)

    # Capacity group (critical)
    cap_group = evaluator.add_parallel(
        id="theater_3_capacity",
        desc="Provide the exact seating capacity and a reference URL confirming it.",
        parent=node_t3,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.capacity and t.capacity.strip()),
        id="theater_3_capacity_value",
        desc="State the exact seating capacity.",
        parent=cap_group,
        critical=True
    )
    cap_ref_leaf = evaluator.add_leaf(
        id="theater_3_capacity_reference",
        desc="Provide a reference URL confirming the seating capacity.",
        parent=cap_group,
        critical=True
    )
    cap_claim = f"The seating capacity of {name_disp} is {t.capacity.strip()} seats." if t and t.capacity else f"The seating capacity of {name_disp} is as stated."
    await _verify_with_required_sources(
        evaluator,
        cap_ref_leaf,
        cap_claim,
        _safe_list(t.capacity_urls if t else None),
        additional_instruction="Verify the capacity number stated in the answer matches the capacity shown on the source page."
    )

    # Expected range (non-critical)
    range_leaf = evaluator.add_leaf(
        id="theater_3_capacity_expected_range",
        desc="If the answer uses the question’s noted expectation, the stated capacity should fall in the 1,700–1,800 range.",
        parent=node_t3,
        critical=False
    )
    range_claim = (
        f"The seating capacity for {name_disp} falls between 1,700 and 1,800 seats (inclusive)."
    )
    await evaluator.verify(
        claim=range_claim,
        node=range_leaf,
        sources=_safe_list(t.capacity_urls if t else None),
        additional_instruction="Use the capacity figure on the page to decide if it lies within [1700, 1800]. If the page shows a value just outside (e.g., 1,690 or 1,820), consider this a mismatch."
    )

    # Second-largest by capacity (critical)
    second_leaf = evaluator.add_leaf(
        id="theater_3_second_largest_by_capacity_reference",
        desc="Provide a reference URL supporting that this is the second-largest Broadway theater by seating capacity.",
        parent=node_t3,
        critical=True
    )
    second_claim = f"{name_disp} is the second-largest Broadway theater by seating capacity."
    await _verify_with_required_sources(
        evaluator,
        second_leaf,
        second_claim,
        _safe_list(t.ranking_urls if t else None),
        additional_instruction="Look for explicit statements that this house ranks second-largest by Broadway seating capacity."
    )

    # Location group (critical)
    loc_group = evaluator.add_parallel(
        id="theater_3_location",
        desc="Provide the street address and location evidence.",
        parent=node_t3,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.address and t.address.strip()),
        id="theater_3_address",
        desc="Provide the street address.",
        parent=loc_group,
        critical=True
    )
    loc_ref_leaf = evaluator.add_leaf(
        id="theater_3_location_reference",
        desc="Provide a reference URL confirming the theater address/location.",
        parent=loc_group,
        critical=True
    )
    loc_claim = f"The street address of {name_disp} is '{t.address.strip()}'." if t and t.address else f"The street address of {name_disp} is as stated."
    await _verify_with_required_sources(
        evaluator,
        loc_ref_leaf,
        loc_claim,
        _safe_list(t.location_urls if t else None),
        additional_instruction="The page should list the theater address or clearly identify the same location."
    )
    within_bounds_leaf = evaluator.add_leaf(
        id="theater_3_within_district_bounds",
        desc="Confirm (based on the provided address/location evidence) that the theater is within the Theater District boundaries (41st–54th Streets, 6th–8th Avenues, Manhattan).",
        parent=loc_group,
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        within_bounds_leaf,
        _address_within_bounds_claim(t.address if t else None),
        _safe_list(t.location_urls if t else None),
        additional_instruction="Use the explicit street (41st–54th) and avenues (6th–8th) boundaries. It is acceptable to infer from the address text (e.g., 'W 50th St' implies within 41st–54th)."
    )

    # Broadway min seats (critical)
    min_seats_leaf = evaluator.add_leaf(
        id="theater_3_broadway_min_seats",
        desc="Confirm the theater meets the Broadway definition minimum of 500+ seats (can be verified from the stated capacity).",
        parent=node_t3,
        critical=True
    )
    min_claim = f"{name_disp} has at least 500 seats (meets Broadway definition)."
    await _verify_with_required_sources(
        evaluator,
        min_seats_leaf,
        min_claim,
        _safe_list(t.capacity_urls if t else None),
        additional_instruction="Confirm from the capacity on the page that it is 500 or more."
    )


async def _build_theater_4(evaluator: Evaluator, parent, t: Optional[TheaterItem]) -> None:
    """
    Theater 4: Hosts the longest-running current Broadway show (show opened in 1996 and is still running).
    """
    node_t4 = evaluator.add_parallel(
        id="theater_4",
        desc="Theater 4: Theater hosting the longest-running current Broadway show (opened in 1996 and still running); include required attributes and references.",
        parent=parent,
        critical=False
    )

    evaluator.add_custom_node(
        result=bool(t and t.name and t.name.strip()),
        id="theater_4_name",
        desc="Provide the theater name.",
        parent=node_t4,
        critical=True
    )
    name_disp = _display_name(t.name if t else None)

    # Capacity group (critical)
    cap_group = evaluator.add_parallel(
        id="theater_4_capacity",
        desc="Provide the exact seating capacity and a reference URL confirming it (required by the overall prompt for each theater).",
        parent=node_t4,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.capacity and t.capacity.strip()),
        id="theater_4_capacity_value",
        desc="State the exact seating capacity.",
        parent=cap_group,
        critical=True
    )
    cap_ref_leaf = evaluator.add_leaf(
        id="theater_4_capacity_reference",
        desc="Provide a reference URL confirming the seating capacity.",
        parent=cap_group,
        critical=True
    )
    cap_claim = f"The seating capacity of {name_disp} is {t.capacity.strip()} seats." if t and t.capacity else f"The seating capacity of {name_disp} is as stated."
    await _verify_with_required_sources(
        evaluator,
        cap_ref_leaf,
        cap_claim,
        _safe_list(t.capacity_urls if t else None),
        additional_instruction="Verify the capacity number stated in the answer matches the capacity shown on the source page."
    )

    # Show group (critical)
    show_group = evaluator.add_parallel(
        id="theater_4_show",
        desc="Provide the name of the currently running show hosted at this theater and support it with a reference URL.",
        parent=node_t4,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.show_name and t.show_name.strip()),
        id="theater_4_show_name",
        desc="Provide the show name.",
        parent=show_group,
        critical=True
    )
    show_ref_leaf = evaluator.add_leaf(
        id="theater_4_show_reference",
        desc="Provide a reference URL confirming the show is currently at this theater.",
        parent=show_group,
        critical=True
    )
    show_claim = (
        f"The show '{t.show_name.strip()}' is currently playing at {name_disp}."
        if t and t.show_name else
        f"The show mentioned is currently playing at {name_disp}."
    )
    await _verify_with_required_sources(
        evaluator,
        show_ref_leaf,
        show_claim,
        _safe_list(t.show_urls if t else None),
        additional_instruction="The page should clearly list this show at the specified theater and indicate it's currently running."
    )

    # Longest-running + 1996 (critical)
    lr_group = evaluator.add_parallel(
        id="theater_4_longest_running_and_1996",
        desc="Verify the show is the longest-running show still playing on Broadway and that it opened in 1996, supported by a reference URL.",
        parent=node_t4,
        critical=True
    )
    opening_leaf = evaluator.add_leaf(
        id="theater_4_opening_date",
        desc="Provide the opening date (must be in 1996 per the question requirement).",
        parent=lr_group,
        critical=True
    )
    if t and t.show_name:
        opening_claim = (
            f"The Broadway show '{t.show_name.strip()}' opened in 1996."
            if not (t.opening_date and t.opening_date.strip())
            else f"The Broadway show '{t.show_name.strip()}' opened on {t.opening_date.strip()} (which is in 1996)."
        )
    else:
        opening_claim = "The referenced show opened in 1996."
    await _verify_with_required_sources(
        evaluator,
        opening_leaf,
        opening_claim,
        _safe_list(t.longest_running_urls if t else None),
        additional_instruction="Confirm the opening year is 1996. If an exact date was provided in the answer, verify it matches the page."
    )

    longest_leaf = evaluator.add_leaf(
        id="theater_4_longest_running_reference",
        desc="Provide a reference URL confirming the opening date and longest-running still-playing status.",
        parent=lr_group,
        critical=True
    )
    longest_claim = (
        f"'{t.show_name.strip()}' is the longest-running Broadway show that is still playing."
        if t and t.show_name else
        "The referenced show is the longest-running Broadway show that is still playing."
    )
    await _verify_with_required_sources(
        evaluator,
        longest_leaf,
        longest_claim,
        _safe_list(t.longest_running_urls if t else None),
        additional_instruction="The page should explicitly indicate that this show holds the longest-running status and is still running."
    )

    # Location group (critical)
    loc_group = evaluator.add_parallel(
        id="theater_4_location",
        desc="Provide the street address and location evidence.",
        parent=node_t4,
        critical=True
    )
    evaluator.add_custom_node(
        result=bool(t and t.address and t.address.strip()),
        id="theater_4_address",
        desc="Provide the street address.",
        parent=loc_group,
        critical=True
    )
    loc_ref_leaf = evaluator.add_leaf(
        id="theater_4_location_reference",
        desc="Provide a reference URL confirming the theater address/location.",
        parent=loc_group,
        critical=True
    )
    loc_claim = f"The street address of {name_disp} is '{t.address.strip()}'." if t and t.address else f"The street address of {name_disp} is as stated."
    await _verify_with_required_sources(
        evaluator,
        loc_ref_leaf,
        loc_claim,
        _safe_list(t.location_urls if t else None),
        additional_instruction="The page should list the theater address or clearly identify the same location."
    )
    within_bounds_leaf = evaluator.add_leaf(
        id="theater_4_within_district_bounds",
        desc="Confirm (based on the provided address/location evidence) that the theater is within the Theater District boundaries (41st–54th Streets, 6th–8th Avenues, Manhattan).",
        parent=loc_group,
        critical=True
    )
    await _verify_with_required_sources(
        evaluator,
        within_bounds_leaf,
        _address_within_bounds_claim(t.address if t else None),
        _safe_list(t.location_urls if t else None),
        additional_instruction="Use the explicit street (41st–54th) and avenues (6th–8th) boundaries. It is acceptable to infer from the address text."
    )

    # Broadway min seats (critical)
    min_seats_leaf = evaluator.add_leaf(
        id="theater_4_broadway_min_seats",
        desc="Confirm the theater meets the Broadway definition minimum of 500+ seats (can be verified from the stated capacity).",
        parent=node_t4,
        critical=True
    )
    min_claim = f"{name_disp} has at least 500 seats (meets Broadway definition)."
    await _verify_with_required_sources(
        evaluator,
        min_seats_leaf,
        min_claim,
        _safe_list(t.capacity_urls if t else None),
        additional_instruction="Confirm from the capacity on the page that it is 500 or more."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the 'Broadway theaters' task and return a structured summary.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent verification for each theater + global checks
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Extract structured information from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_theaters(),
        template_class=TheatersExtraction,
        extraction_name="theaters_extraction"
    )

    # Build theater-specific verification subtrees
    await _build_theater_1(evaluator, root, extraction.theater1)
    await _build_theater_2(evaluator, root, extraction.theater2)
    await _build_theater_3(evaluator, root, extraction.theater3)
    await _build_theater_4(evaluator, root, extraction.theater4)

    # Global checks (critical)
    global_node = evaluator.add_parallel(
        id="global_checks",
        desc="Global requirements that apply across the set of four theaters.",
        parent=root,
        critical=True
    )

    # Distinctness check (critical)
    distinct_leaf = evaluator.add_leaf(
        id="four_theaters_are_distinct",
        desc="Confirm the four identified theaters are distinct (no theater is repeated across Theater 1–4).",
        parent=global_node,
        critical=True
    )
    n1 = (extraction.theater1.name if extraction.theater1 and extraction.theater1.name else "").strip()
    n2 = (extraction.theater2.name if extraction.theater2 and extraction.theater2.name else "").strip()
    n3 = (extraction.theater3.name if extraction.theater3 and extraction.theater3.name else "").strip()
    n4 = (extraction.theater4.name if extraction.theater4 and extraction.theater4.name else "").strip()
    distinct_claim = (
        "The four identified Broadway theaters are all distinct venues with no duplicates: "
        f"{n1 or '[missing]'}, {n2 or '[missing]'}, {n3 or '[missing]'}, {n4 or '[missing]'}."
    )
    await evaluator.verify(
        claim=distinct_claim,
        node=distinct_leaf,
        additional_instruction="Determine if any two of the provided names refer to the same venue. Consider minor name variants (e.g., 'The Ambassador Theatre' vs 'Ambassador Theatre') as the same venue."
    )

    # Return the evaluation summary
    return evaluator.get_summary()