import asyncio
import logging
import re
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_city_performing_arts_venues"
TASK_DESCRIPTION = """
Identify four specific performing arts venues, one in each of the following cities: New York City, Los Angeles, Chicago, and Boston. Each venue must meet all city-specific criteria and provide the venue name, a specific seating capacity, and reference URLs confirming capacity, accessibility features, technical specifications, and historical/organizational information.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """Single venue deliverables for one city."""
    name: Optional[str] = None
    capacity: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VenuesExtraction(BaseModel):
    """Four venues, one per city."""
    nyc: Optional[VenueItem] = None
    los_angeles: Optional[VenueItem] = None
    chicago: Optional[VenueItem] = None
    boston: Optional[VenueItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venues() -> str:
    return """
    Extract one performing arts venue for each city: New York City, Los Angeles, Chicago, and Boston, as presented in the answer.
    For each city, extract:
    - name: The venue name (exactly as stated in the answer).
    - capacity: A specific numeric seating capacity mentioned for that venue (as a plain string like "2804"; if the answer uses words like "about 2800", return "2800"). If multiple capacities are mentioned, choose the main/adult capacity for the full auditorium.
    - sources: All URLs that the answer associates with that venue (include official sites, Wikipedia, reputable news or organizational pages, accessibility pages, technical/architectural pages, season calendars, etc.). Extract only valid URLs; also parse URLs embedded in markdown links. If none are provided, return an empty list.

    Important:
    - If the answer provides more than one venue for a given city, extract only the first one mentioned for that city.
    - If any field is missing, set it to null for name/capacity and an empty list for sources.
    - Do not invent URLs; extract only those present in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def parse_seat_count(capacity_str: Optional[str]) -> Optional[int]:
    """Parse a numeric seating capacity from a string like '2,804' or 'about 2800'."""
    if not capacity_str:
        return None
    s = capacity_str.strip()
    s = s.replace(",", "")
    m = re.search(r"\d{3,5}", s)
    try:
        return int(m.group(0)) if m else None
    except Exception:
        return None


def safe_sources(v: Optional[VenueItem]) -> List[str]:
    """Return sources list or [] if missing."""
    return v.sources if (v and v.sources) else []


# --------------------------------------------------------------------------- #
# City-specific verification builders                                          #
# --------------------------------------------------------------------------- #
async def build_nyc_verification(evaluator: Evaluator, root_node, v: Optional[VenueItem]) -> None:
    """Build verification subtree for the NYC venue."""
    city_node = evaluator.add_parallel(
        id="venue_1_new_york",
        desc="NYC venue (one item): satisfies all NYC-specific criteria and includes required deliverables with supporting references.",
        parent=root_node,
        critical=True
    )

    sources = safe_sources(v)
    name = v.name if v and v.name else None
    num_capacity = parse_seat_count(v.capacity if v else None)

    # Deliverables: Name provided
    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="nyc_name_provided",
        desc="Venue name is provided.",
        parent=city_node,
        critical=True
    )

    # Location in New York City (verify via sources)
    loc_node = evaluator.add_leaf(
        id="nyc_location_in_nyc",
        desc="Venue is located in New York City.",
        parent=city_node,
        critical=True
    )
    loc_claim = f"The venue '{name}' is located within New York City (one of the five boroughs: Manhattan, Brooklyn, Queens, The Bronx, or Staten Island)."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=sources,
        additional_instruction="Confirm by address or city listing from official/reputable sources. Do not accept locations outside NYC city limits."
    )

    # Capacity provided (specific numeric)
    evaluator.add_custom_node(
        result=(num_capacity is not None),
        id="nyc_capacity_provided",
        desc="A specific numeric seating capacity is provided.",
        parent=city_node,
        critical=True
    )

    # Capacity in required range
    evaluator.add_custom_node(
        result=(num_capacity is not None and 2500 <= num_capacity <= 3000),
        id="nyc_capacity_in_range",
        desc="Seating capacity is between 2,500 and 3,000 seats (inclusive).",
        parent=city_node,
        critical=True
    )

    # Accessibility criteria
    acc_node = evaluator.add_parallel(
        id="nyc_accessibility",
        desc="NYC accessibility criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    acc_wheel_node = evaluator.add_leaf(
        id="nyc_wheelchair_multiple_levels",
        desc="Wheelchair accessible seating is available on multiple levels.",
        parent=acc_node,
        critical=True
    )
    acc_wheel_claim = f"At '{name}', wheelchair accessible seating is available on multiple seating levels (e.g., orchestra and balcony/tiers)."
    await evaluator.verify(
        claim=acc_wheel_claim,
        node=acc_wheel_node,
        sources=sources,
        additional_instruction="Look for official accessibility pages or seating charts indicating accessible seating on more than one level."
    )

    acc_stepfree_node = evaluator.add_leaf(
        id="nyc_step_free_route",
        desc="There is a step-free accessible route from the entrance to seating areas.",
        parent=acc_node,
        critical=True
    )
    acc_stepfree_claim = f"'{name}' provides a step-free accessible route from the venue entrance to seating areas."
    await evaluator.verify(
        claim=acc_stepfree_claim,
        node=acc_stepfree_node,
        sources=sources,
        additional_instruction="Seek accessibility statements indicating ramps/elevators and step-free paths from entry to audience seating."
    )

    # Acoustics & configuration
    ac_node = evaluator.add_parallel(
        id="nyc_acoustics_and_configuration",
        desc="NYC acoustic and configuration criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    ac_design_node = evaluator.add_leaf(
        id="nyc_designed_or_renovated_for_acoustics",
        desc="A reputable source states the venue was specifically designed or renovated for acoustic performances.",
        parent=ac_node,
        critical=True
    )
    ac_design_claim = f"'{name}' was specifically designed or renovated to optimize acoustics for acoustic/classical performances."
    await evaluator.verify(
        claim=ac_design_claim,
        node=ac_design_node,
        sources=sources,
        additional_instruction="Evidence can include architectural notes, renovations focused on acoustics, or design intent for classical music."
    )

    ac_stage_node = evaluator.add_leaf(
        id="nyc_proscenium_or_concert_hall_suitable_for_orchestra",
        desc="A reputable source indicates the venue has a proscenium stage or concert-hall configuration suitable for orchestral performances.",
        parent=ac_node,
        critical=True
    )
    ac_stage_claim = f"'{name}' features a proscenium stage or concert hall configuration suitable for orchestral performances."
    await evaluator.verify(
        claim=ac_stage_claim,
        node=ac_stage_node,
        sources=sources,
        additional_instruction="Confirm stage/hall type; orchestral suitability may be described via stage, pit, shell, or concert-hall terminology."
    )

    # History & current status
    hist_node = evaluator.add_parallel(
        id="nyc_history_and_status",
        desc="NYC historical and current-status criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    hist_year_node = evaluator.add_leaf(
        id="nyc_established_before_1950",
        desc="A reputable source indicates the venue was established before 1950.",
        parent=hist_node,
        critical=True
    )
    hist_year_claim = f"'{name}' was established/opened before 1950."
    await evaluator.verify(
        claim=hist_year_claim,
        node=hist_year_node,
        sources=sources,
        additional_instruction="Use founding/opening year from official history pages, Wikipedia with citations, or reliable sources."
    )

    status_node = evaluator.add_leaf(
        id="nyc_operational_as_of_2026",
        desc="A reputable source indicates the venue is currently operational and hosting performances as of 2026.",
        parent=hist_node,
        critical=True
    )
    status_claim = f"As of 2026, '{name}' is operational and hosting performances (e.g., season calendar or current event listings)."
    await evaluator.verify(
        claim=status_claim,
        node=status_node,
        sources=sources,
        additional_instruction="Look for current (2025–2026) season pages, calendars, or announcements indicating ongoing performances."
    )

    # References aggregation
    refs_node = evaluator.add_parallel(
        id="nyc_references",
        desc="Reference URLs are provided that collectively substantiate the NYC venue’s required claims.",
        parent=city_node,
        critical=True
    )

    cap_ref_node = evaluator.add_leaf(
        id="nyc_reference_for_capacity",
        desc="At least one URL is provided that supports the stated seating capacity.",
        parent=refs_node,
        critical=True
    )
    cap_ref_claim = f"The seating capacity of '{name}' is {num_capacity} seats."
    await evaluator.verify(
        claim=cap_ref_claim,
        node=cap_ref_node,
        sources=sources,
        additional_instruction="At least one provided URL should explicitly state the total seating capacity matching the claimed number."
    )

    acc_ref_node = evaluator.add_leaf(
        id="nyc_reference_for_accessibility",
        desc="At least one URL is provided that supports the accessibility claims (wheelchair seating and/or step-free routes).",
        parent=refs_node,
        critical=True
    )
    acc_ref_claim = f"At least one provided source confirms accessibility features at '{name}' (wheelchair seating and/or step-free routes)."
    await evaluator.verify(
        claim=acc_ref_claim,
        node=acc_ref_node,
        sources=sources,
        additional_instruction="A single URL may confirm either wheelchair seating availability or step-free access; either suffices for this check."
    )

    tech_ref_node = evaluator.add_leaf(
        id="nyc_reference_for_technical_acoustics_config",
        desc="At least one URL is provided that supports the acoustic-design and configuration claims.",
        parent=refs_node,
        critical=True
    )
    tech_ref_claim = f"At least one provided source confirms that '{name}' was designed/renovated for acoustics and/or has orchestra-suitable hall/stage configuration."
    await evaluator.verify(
        claim=tech_ref_claim,
        node=tech_ref_node,
        sources=sources,
        additional_instruction="A single URL confirming either acoustic-focused design/renovation or concert-hall/proscenium orchestral suitability suffices."
    )

    hist_ref_node = evaluator.add_leaf(
        id="nyc_reference_for_history_status",
        desc="At least one URL is provided that supports establishment date and/or current operational status.",
        parent=refs_node,
        critical=True
    )
    hist_ref_claim = f"At least one provided source confirms either that '{name}' was established before 1950 or that it is currently operational as of 2026."
    await evaluator.verify(
        claim=hist_ref_claim,
        node=hist_ref_node,
        sources=sources,
        additional_instruction="A single URL confirming either the pre-1950 establishment or present-day operational activity suffices."
    )


async def build_la_verification(evaluator: Evaluator, root_node, v: Optional[VenueItem]) -> None:
    """Build verification subtree for the Los Angeles venue."""
    city_node = evaluator.add_parallel(
        id="venue_2_los_angeles",
        desc="Los Angeles venue (one item): satisfies all LA-specific criteria and includes required deliverables with supporting references.",
        parent=root_node,
        critical=True
    )

    sources = safe_sources(v)
    name = v.name if v and v.name else None
    num_capacity = parse_seat_count(v.capacity if v else None)

    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="la_name_provided",
        desc="Venue name is provided.",
        parent=city_node,
        critical=True
    )

    loc_node = evaluator.add_leaf(
        id="la_location_in_los_angeles",
        desc="Venue is located in Los Angeles.",
        parent=city_node,
        critical=True
    )
    loc_claim = f"The venue '{name}' is located in Los Angeles (city of Los Angeles)."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=sources,
        additional_instruction="Confirm by address or city listing from official/reputable sources; accept Downtown LA/Hollywood/etc within city limits."
    )

    evaluator.add_custom_node(
        result=(num_capacity is not None),
        id="la_capacity_provided",
        desc="A specific numeric seating capacity is provided.",
        parent=city_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(num_capacity is not None and 2000 <= num_capacity <= 2500),
        id="la_capacity_in_range",
        desc="Seating capacity is between 2,000 and 2,500 seats (inclusive).",
        parent=city_node,
        critical=True
    )

    acc_node = evaluator.add_parallel(
        id="la_accessibility",
        desc="LA accessibility criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    la_wc_node = evaluator.add_leaf(
        id="la_wheelchair_seating_provided",
        desc="Wheelchair accessible seating locations are provided.",
        parent=acc_node,
        critical=True
    )
    la_wc_claim = f"'{name}' provides wheelchair-accessible seating locations."
    await evaluator.verify(
        claim=la_wc_claim,
        node=la_wc_node,
        sources=sources,
        additional_instruction="Look for explicit mention of wheelchair accessible seating, ADA seating locations, or seating maps."
    )

    la_rest_node = evaluator.add_leaf(
        id="la_accessible_restrooms_and_concessions",
        desc="Accessible restrooms and concession areas are available.",
        parent=acc_node,
        critical=True
    )
    la_rest_claim = f"'{name}' provides accessible restrooms and accessible concession areas."
    await evaluator.verify(
        claim=la_rest_claim,
        node=la_rest_node,
        sources=sources,
        additional_instruction="Seek accessibility pages or ADA info that confirm accessible restrooms and concessions."
    )

    type_ac_node = evaluator.add_parallel(
        id="la_venue_type_and_acoustics",
        desc="LA venue-type and acoustics criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    la_type_node = evaluator.add_leaf(
        id="la_concert_hall_or_pac_not_arena",
        desc="A reputable source indicates the venue is specifically designed as a concert hall or performing arts center (not a multi-purpose arena).",
        parent=type_ac_node,
        critical=True
    )
    la_type_claim = f"'{name}' is a concert hall or performing arts center (not a multi-purpose arena)."
    await evaluator.verify(
        claim=la_type_claim,
        node=la_type_node,
        sources=sources,
        additional_instruction="Confirm venue classification from official descriptions or reputable sources."
    )

    la_ac_node = evaluator.add_leaf(
        id="la_acoustics_for_unamplified_orchestra",
        desc="A reputable source indicates professional acoustic treatment/design suitable for unamplified orchestral music.",
        parent=type_ac_node,
        critical=True
    )
    la_ac_claim = f"'{name}' has acoustic design/treatment suitable for unamplified orchestral music."
    await evaluator.verify(
        claim=la_ac_claim,
        node=la_ac_node,
        sources=sources,
        additional_instruction="Look for acoustics descriptions (e.g., reverberation design, acoustic panels, orchestral shell, etc.)."
    )

    org_prog_node = evaluator.add_parallel(
        id="la_resident_org_and_programming",
        desc="LA resident-organization and programming criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    la_home_node = evaluator.add_leaf(
        id="la_home_to_resident_symphony_or_major_classical_org",
        desc="A reputable source indicates the venue serves as home to a resident symphony orchestra or major classical music organization.",
        parent=org_prog_node,
        critical=True
    )
    la_home_claim = f"'{name}' serves as the home venue for a resident symphony orchestra or major classical music organization."
    await evaluator.verify(
        claim=la_home_claim,
        node=la_home_node,
        sources=sources,
        additional_instruction="Confirm via ensemble/organization pages or venue site that it is the resident/home venue."
    )

    la_reg_node = evaluator.add_leaf(
        id="la_regular_classical_throughout_year",
        desc="Evidence is provided (e.g., season/calendar or organizational description) supporting that the venue hosts regular classical performances throughout the year.",
        parent=org_prog_node,
        critical=True
    )
    la_reg_claim = f"'{name}' hosts regular classical music performances throughout the year."
    await evaluator.verify(
        claim=la_reg_claim,
        node=la_reg_node,
        sources=sources,
        additional_instruction="Season calendars, subscription series, or annual schedules should indicate recurring classical programming."
    )

    refs_node = evaluator.add_parallel(
        id="la_references",
        desc="Reference URLs are provided that collectively substantiate the LA venue’s required claims.",
        parent=city_node,
        critical=True
    )

    la_cap_ref = evaluator.add_leaf(
        id="la_reference_for_capacity",
        desc="At least one URL is provided that supports the stated seating capacity.",
        parent=refs_node,
        critical=True
    )
    la_cap_claim = f"The seating capacity of '{name}' is {num_capacity} seats."
    await evaluator.verify(
        claim=la_cap_claim,
        node=la_cap_ref,
        sources=sources,
        additional_instruction="At least one provided URL should explicitly state the total seating capacity matching the claimed number."
    )

    la_acc_ref = evaluator.add_leaf(
        id="la_reference_for_accessibility",
        desc="At least one URL is provided that supports the accessibility claims.",
        parent=refs_node,
        critical=True
    )
    la_acc_ref_claim = f"At least one provided source confirms accessibility features at '{name}' (wheelchair seating and/or accessible restrooms/concessions)."
    await evaluator.verify(
        claim=la_acc_ref_claim,
        node=la_acc_ref,
        sources=sources,
        additional_instruction="A single URL confirming any of the stated accessibility features suffices."
    )

    la_type_ac_ref = evaluator.add_leaf(
        id="la_reference_for_venue_type_acoustics",
        desc="At least one URL is provided that supports the concert-hall/PAC (not arena) and acoustics claims.",
        parent=refs_node,
        critical=True
    )
    la_type_ac_ref_claim = f"At least one provided source confirms '{name}' is a concert hall/PAC and/or has acoustics suitable for unamplified orchestral music."
    await evaluator.verify(
        claim=la_type_ac_ref_claim,
        node=la_type_ac_ref,
        sources=sources,
        additional_instruction="A single URL confirming either venue-type or acoustics suffices."
    )

    la_org_ref = evaluator.add_leaf(
        id="la_reference_for_resident_org_programming",
        desc="At least one URL is provided that supports resident organization and regular classical programming claims.",
        parent=refs_node,
        critical=True
    )
    la_org_ref_claim = f"At least one provided source confirms '{name}' is home to a resident symphony/major classical org and/or hosts regular classical performances."
    await evaluator.verify(
        claim=la_org_ref_claim,
        node=la_org_ref,
        sources=sources,
        additional_instruction="A single URL confirming either resident-org status or regular classical programming suffices."
    )


async def build_chicago_verification(evaluator: Evaluator, root_node, v: Optional[VenueItem]) -> None:
    """Build verification subtree for the Chicago venue."""
    city_node = evaluator.add_parallel(
        id="venue_3_chicago",
        desc="Chicago venue (one item): satisfies all Chicago-specific criteria and includes required deliverables with supporting references.",
        parent=root_node,
        critical=True
    )

    sources = safe_sources(v)
    name = v.name if v and v.name else None
    num_capacity = parse_seat_count(v.capacity if v else None)

    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="chicago_name_provided",
        desc="Venue name is provided.",
        parent=city_node,
        critical=True
    )

    loc_node = evaluator.add_leaf(
        id="chicago_location_in_chicago",
        desc="Venue is located in Chicago.",
        parent=city_node,
        critical=True
    )
    loc_claim = f"The venue '{name}' is located in Chicago."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=sources,
        additional_instruction="Confirm by address/city listing from official or reputable sources."
    )

    evaluator.add_custom_node(
        result=(num_capacity is not None),
        id="chicago_capacity_provided",
        desc="A specific numeric seating capacity is provided.",
        parent=city_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(num_capacity is not None and 2200 <= num_capacity <= 2800),
        id="chicago_capacity_in_range",
        desc="Seating capacity is between 2,200 and 2,800 seats (inclusive).",
        parent=city_node,
        critical=True
    )

    acc_node = evaluator.add_parallel(
        id="chicago_accessibility",
        desc="Chicago accessibility criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    ada_node = evaluator.add_leaf(
        id="chicago_ada_wheelchair_approx_1_percent",
        desc="A reputable source indicates the venue complies with ADA wheelchair seating expectations (~1% of total capacity).",
        parent=acc_node,
        critical=True
    )
    ada_claim = f"'{name}' complies with ADA wheelchair seating expectations (approximately 1% of total capacity)."
    await evaluator.verify(
        claim=ada_claim,
        node=ada_node,
        sources=sources,
        additional_instruction="Look for ADA compliance details; approximate 1% of capacity is acceptable."
    )

    assist_node = evaluator.add_leaf(
        id="chicago_assistive_listening_available",
        desc="A reputable source indicates assistive listening devices/systems are available.",
        parent=acc_node,
        critical=True
    )
    assist_claim = f"'{name}' offers assistive listening devices/systems for patrons with hearing impairments."
    await evaluator.verify(
        claim=assist_claim,
        node=assist_node,
        sources=sources,
        additional_instruction="Confirm assistive listening availability from official accessibility pages or reputable sources."
    )

    stage_node = evaluator.add_parallel(
        id="chicago_stage_and_rigging",
        desc="Chicago stage and rigging criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    stage_size_node = evaluator.add_leaf(
        id="chicago_stage_supports_full_orchestra_75_100",
        desc="A reputable source indicates the stage can accommodate a full symphony orchestra (75–100 musicians).",
        parent=stage_node,
        critical=True
    )
    stage_size_claim = f"The stage at '{name}' can accommodate a full symphony orchestra of approximately 75–100 musicians."
    await evaluator.verify(
        claim=stage_size_claim,
        node=stage_size_node,
        sources=sources,
        additional_instruction="Evidence may include stage dimensions, orchestral performance descriptions, or technical specs."
    )

    rigging_node = evaluator.add_leaf(
        id="chicago_rigging_or_fly_for_acoustic_shell",
        desc="A reputable source indicates professional rigging or a fly system is available for acoustic shells/concert configurations.",
        parent=stage_node,
        critical=True
    )
    rigging_claim = f"'{name}' has professional rigging or a fly system supporting acoustic shells or concert configurations."
    await evaluator.verify(
        claim=rigging_claim,
        node=rigging_node,
        sources=sources,
        additional_instruction="Confirm presence of rigging/fly systems via technical sheets or venue specs."
    )

    arch_node = evaluator.add_parallel(
        id="chicago_architecture_and_heritage",
        desc="Chicago architectural/heritage criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    landmark_node = evaluator.add_leaf(
        id="chicago_historic_landmark_or_documented_heritage",
        desc="Evidence is provided that the venue is a designated historic landmark OR a reputable source explicitly describes it as having significant architectural heritage.",
        parent=arch_node,
        critical=True
    )
    landmark_claim = f"'{name}' is designated a historic landmark or is described by reputable sources as having significant architectural heritage."
    await evaluator.verify(
        claim=landmark_claim,
        node=landmark_node,
        sources=sources,
        additional_instruction="Landmark designation pages or authoritative descriptions of heritage suffice."
    )

    era_node = evaluator.add_leaf(
        id="chicago_early_mid_20th_distinctive_design",
        desc="A reputable source indicates the venue features distinctive early-to-mid 20th century architectural design.",
        parent=arch_node,
        critical=True
    )
    era_claim = f"'{name}' features distinctive early-to-mid 20th century architectural design."
    await evaluator.verify(
        claim=era_claim,
        node=era_node,
        sources=sources,
        additional_instruction="Confirm era/style from architectural descriptions or historical references."
    )

    refs_node = evaluator.add_parallel(
        id="chicago_references",
        desc="Reference URLs are provided that collectively substantiate the Chicago venue’s required claims.",
        parent=city_node,
        critical=True
    )

    chi_cap_ref = evaluator.add_leaf(
        id="chicago_reference_for_capacity",
        desc="At least one URL is provided that supports the stated seating capacity.",
        parent=refs_node,
        critical=True
    )
    chi_cap_claim = f"The seating capacity of '{name}' is {num_capacity} seats."
    await evaluator.verify(
        claim=chi_cap_claim,
        node=chi_cap_ref,
        sources=sources,
        additional_instruction="At least one provided URL should explicitly state the total seating capacity matching the claimed number."
    )

    chi_acc_ref = evaluator.add_leaf(
        id="chicago_reference_for_accessibility",
        desc="At least one URL is provided that supports the accessibility claims.",
        parent=refs_node,
        critical=True
    )
    chi_acc_ref_claim = f"At least one provided source confirms accessibility features at '{name}' (ADA wheelchair seating and/or assistive listening)."
    await evaluator.verify(
        claim=chi_acc_ref_claim,
        node=chi_acc_ref,
        sources=sources,
        additional_instruction="A single URL confirming any of the stated accessibility features suffices."
    )

    chi_stage_ref = evaluator.add_leaf(
        id="chicago_reference_for_stage_rigging",
        desc="At least one URL is provided that supports the stage and rigging claims.",
        parent=refs_node,
        critical=True
    )
    chi_stage_ref_claim = f"At least one provided source confirms stage size for full orchestra and/or presence of rigging/fly systems at '{name}'."
    await evaluator.verify(
        claim=chi_stage_ref_claim,
        node=chi_stage_ref,
        sources=sources,
        additional_instruction="A single URL confirming either stage capacity or rigging suffices."
    )

    chi_heritage_ref = evaluator.add_leaf(
        id="chicago_reference_for_heritage_architecture",
        desc="At least one URL is provided that supports the landmark/heritage and architectural-era claims.",
        parent=refs_node,
        critical=True
    )
    chi_heritage_ref_claim = f"At least one provided source confirms landmark/heritage status and/or early-mid 20th architectural design for '{name}'."
    await evaluator.verify(
        claim=chi_heritage_ref_claim,
        node=chi_heritage_ref,
        sources=sources,
        additional_instruction="A single URL confirming either landmark/heritage or architectural era suffices."
    )


async def build_boston_verification(evaluator: Evaluator, root_node, v: Optional[VenueItem]) -> None:
    """Build verification subtree for the Boston venue."""
    city_node = evaluator.add_parallel(
        id="venue_4_boston",
        desc="Boston venue (one item): satisfies all Boston-specific criteria and includes required deliverables with supporting references.",
        parent=root_node,
        critical=True
    )

    sources = safe_sources(v)
    name = v.name if v and v.name else None
    num_capacity = parse_seat_count(v.capacity if v else None)

    evaluator.add_custom_node(
        result=bool(name and name.strip()),
        id="boston_name_provided",
        desc="Venue name is provided.",
        parent=city_node,
        critical=True
    )

    loc_node = evaluator.add_leaf(
        id="boston_location_in_boston",
        desc="Venue is located in Boston.",
        parent=city_node,
        critical=True
    )
    loc_claim = f"The venue '{name}' is located in Boston."
    await evaluator.verify(
        claim=loc_claim,
        node=loc_node,
        sources=sources,
        additional_instruction="Confirm by address/city listing from official or reputable sources."
    )

    evaluator.add_custom_node(
        result=(num_capacity is not None),
        id="boston_capacity_provided",
        desc="A specific numeric seating capacity is provided.",
        parent=city_node,
        critical=True
    )

    evaluator.add_custom_node(
        result=(num_capacity is not None and 2300 <= num_capacity <= 2700),
        id="boston_capacity_in_range",
        desc="Seating capacity is between 2,300 and 2,700 seats (inclusive).",
        parent=city_node,
        critical=True
    )

    acc_node = evaluator.add_parallel(
        id="boston_accessibility",
        desc="Boston accessibility criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    bos_wc_levels = evaluator.add_leaf(
        id="boston_wheelchair_various_price_levels",
        desc="Evidence supports that wheelchair accessible seating is offered throughout the auditorium at various price levels.",
        parent=acc_node,
        critical=True
    )
    bos_wc_levels_claim = f"'{name}' offers wheelchair accessible seating throughout the auditorium at various price levels."
    await evaluator.verify(
        claim=bos_wc_levels_claim,
        node=bos_wc_levels,
        sources=sources,
        additional_instruction="Look for seating maps or policy statements indicating accessible seating across different sections/price tiers."
    )

    bos_elev_all = evaluator.add_leaf(
        id="boston_elevator_to_all_levels",
        desc="Evidence supports elevator access to all seating levels.",
        parent=acc_node,
        critical=True
    )
    bos_elev_all_claim = f"'{name}' provides elevator access to all seating levels."
    await evaluator.verify(
        claim=bos_elev_all_claim,
        node=bos_elev_all,
        sources=sources,
        additional_instruction="Confirm by accessibility pages stating elevator coverage to all levels."
    )

    ac_node = evaluator.add_parallel(
        id="boston_acoustics_design",
        desc="Boston acoustic-design criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    bos_purpose = evaluator.add_leaf(
        id="boston_purpose_built_shoebox_or_vineyard",
        desc="A reputable source indicates the venue is purpose-built for symphonic music with shoebox or vineyard acoustic design.",
        parent=ac_node,
        critical=True
    )
    bos_purpose_claim = f"'{name}' is purpose-built for symphonic music and uses a shoebox or vineyard acoustic design."
    await evaluator.verify(
        claim=bos_purpose_claim,
        node=bos_purpose,
        sources=sources,
        additional_instruction="Confirm via architectural/acoustic descriptions."
    )

    bos_doc_ac = evaluator.add_leaf(
        id="boston_documented_acoustics_optimized",
        desc="A reputable source provides documented acoustic properties optimized for classical music performance.",
        parent=ac_node,
        critical=True
    )
    bos_doc_ac_claim = f"Documented acoustic properties of '{name}' indicate optimization for classical music performance."
    await evaluator.verify(
        claim=bos_doc_ac_claim,
        node=bos_doc_ac,
        sources=sources,
        additional_instruction="Look for documentation (reverberation times, design notes) indicating optimization for classical music."
    )

    inst_prest_node = evaluator.add_parallel(
        id="boston_institution_and_prestige",
        desc="Boston institutional-connection and prestige criteria are satisfied.",
        parent=city_node,
        critical=True
    )

    bos_home_major = evaluator.add_leaf(
        id="boston_primary_home_major_us_symphony",
        desc="A reputable source indicates the venue serves as the primary performance home for a major American symphony orchestra.",
        parent=inst_prest_node,
        critical=True
    )
    bos_home_major_claim = f"'{name}' serves as the primary performance home for a major American symphony orchestra."
    await evaluator.verify(
        claim=bos_home_major_claim,
        node=bos_home_major,
        sources=sources,
        additional_instruction="Confirm via orchestra or venue pages stating primary home venue."
    )

    bos_international = evaluator.add_leaf(
        id="boston_international_acoustic_reputation_and_touring",
        desc="Evidence is provided that reputable sources describe the venue as internationally renowned for acoustics AND that it hosts touring international orchestras.",
        parent=inst_prest_node,
        critical=True
    )
    bos_international_claim = f"'{name}' has an international reputation for acoustic excellence and hosts touring international orchestras."
    await evaluator.verify(
        claim=bos_international_claim,
        node=bos_international,
        sources=sources,
        additional_instruction="Seek reputable descriptions of international acoustic acclaim and examples of touring orchestras performing there."
    )

    refs_node = evaluator.add_parallel(
        id="boston_references",
        desc="Reference URLs are provided that collectively substantiate the Boston venue’s required claims.",
        parent=city_node,
        critical=True
    )

    bos_cap_ref = evaluator.add_leaf(
        id="boston_reference_for_capacity",
        desc="At least one URL is provided that supports the stated seating capacity.",
        parent=refs_node,
        critical=True
    )
    bos_cap_claim = f"The seating capacity of '{name}' is {num_capacity} seats."
    await evaluator.verify(
        claim=bos_cap_claim,
        node=bos_cap_ref,
        sources=sources,
        additional_instruction="At least one provided URL should explicitly state the total seating capacity matching the claimed number."
    )

    bos_acc_ref = evaluator.add_leaf(
        id="boston_reference_for_accessibility",
        desc="At least one URL is provided that supports the accessibility claims.",
        parent=refs_node,
        critical=True
    )
    bos_acc_ref_claim = f"At least one provided source confirms accessibility features at '{name}' (wheelchair seating across price levels and/or elevator access to all levels)."
    await evaluator.verify(
        claim=bos_acc_ref_claim,
        node=bos_acc_ref,
        sources=sources,
        additional_instruction="A single URL confirming either of the stated accessibility features suffices."
    )

    bos_acoustics_ref = evaluator.add_leaf(
        id="boston_reference_for_acoustics_design",
        desc="At least one URL is provided that supports the acoustic-design and documented-acoustics claims.",
        parent=refs_node,
        critical=True
    )
    bos_acoustics_ref_claim = f"At least one provided source confirms '{name}' is purpose-built with shoebox/vineyard design and/or documents acoustics optimized for classical performance."
    await evaluator.verify(
        claim=bos_acoustics_ref_claim,
        node=bos_acoustics_ref,
        sources=sources,
        additional_instruction="A single URL confirming either design type or documented optimized acoustics suffices."
    )

    bos_inst_ref = evaluator.add_leaf(
        id="boston_reference_for_institution_prestige",
        desc="At least one URL is provided that supports the symphony-home and international reputation/touring claims.",
        parent=refs_node,
        critical=True
    )
    bos_inst_ref_claim = f"At least one provided source confirms '{name}' is the primary home of a major US symphony and/or has international acoustics reputation with touring orchestras."
    await evaluator.verify(
        claim=bos_inst_ref_claim,
        node=bos_inst_ref,
        sources=sources,
        additional_instruction="A single URL confirming either home venue status or international acoustic reputation/touring suffices."
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
    Evaluate an answer for the four-city performing arts venues task.
    """
    # Initialize evaluator (root is non-critical by framework; we enforce city nodes critical)
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

    # Extract venues data from the answer
    venues = await evaluator.extract(
        prompt=prompt_extract_venues(),
        template_class=VenuesExtraction,
        extraction_name="venues_extraction",
    )

    # Optional: record target capacity ranges for clarity
    evaluator.add_custom_info(
        info={
            "nyc_required_capacity_range": [2500, 3000],
            "la_required_capacity_range": [2000, 2500],
            "chicago_required_capacity_range": [2200, 2800],
            "boston_required_capacity_range": [2300, 2700],
            "current_year_context": 2026
        },
        info_type="constraints",
        info_name="venue_constraints"
    )

    # Build verification trees for each city (critical children under root)
    await build_nyc_verification(evaluator, root, venues.nyc)
    await build_la_verification(evaluator, root, venues.los_angeles)
    await build_chicago_verification(evaluator, root, venues.chicago)
    await build_boston_verification(evaluator, root, venues.boston)

    # Return structured evaluation summary
    return evaluator.get_summary()