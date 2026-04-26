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
TASK_ID = "state_parks_summer_2026"
TASK_DESCRIPTION = (
    "Identify four state parks in the United States for a summer 2026 multi-state camping and hiking road trip. "
    "Each park must be in a different U.S. state and satisfy camping, hiking, accessibility, reservations, "
    "and amenities requirements, with appropriate official references and reservation URLs."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CampingInfo(BaseModel):
    electric_hookups_note: Optional[str] = None
    waterfront_note: Optional[str] = None
    restrooms_note: Optional[str] = None
    showers_note: Optional[str] = None
    pets_note: Optional[str] = None
    camping_fee_note: Optional[str] = None
    camping_reference_url: Optional[str] = None


class TrailInfo(BaseModel):
    name: Optional[str] = None
    length_miles: Optional[str] = None
    elevation_gain_ft: Optional[str] = None
    trail_reference_url: Optional[str] = None
    difficulty_variety_note: Optional[str] = None


class AccessibilityInfo(BaseModel):
    ada_note: Optional[str] = None
    accessibility_reference_url: Optional[str] = None


class ReservationInfo(BaseModel):
    reservation_url: Optional[str] = None
    summer_2026_availability_note: Optional[str] = None
    online_system_note: Optional[str] = None


class AmenitiesInfo(BaseModel):
    additional_facilities_note: Optional[str] = None
    visitor_services_note: Optional[str] = None


class ParkInfo(BaseModel):
    name: Optional[str] = None
    state: Optional[str] = None
    official_url: Optional[str] = None
    camping: CampingInfo = Field(default_factory=CampingInfo)
    trails: TrailInfo = Field(default_factory=TrailInfo)
    accessibility: AccessibilityInfo = Field(default_factory=AccessibilityInfo)
    reservations: ReservationInfo = Field(default_factory=ReservationInfo)
    amenities: AmenitiesInfo = Field(default_factory=AmenitiesInfo)


class ParksExtraction(BaseModel):
    parks: List[ParkInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_parks() -> str:
    return """
    Extract up to four U.S. state parks listed in the answer. If more than four are present, return only the first four. 
    For each park, extract the following fields exactly as presented in the answer (use null if missing):

    Park Identification:
    - name: The official name of the state park.
    - state: The U.S. state where the park is located (e.g., "Colorado").
    - official_url: The URL to the park's official state parks system webpage.

    Camping:
    - electric_hookups_note: Text or note indicating electric hookups (30 amp or 50 amp).
    - waterfront_note: Text indicating lakefront, riverfront, or oceanfront camping sites.
    - restrooms_note: Text indicating on-site restrooms with running water (flush toilets/modern restrooms).
    - showers_note: Text indicating shower facilities for campers.
    - pets_note: Text indicating pets are allowed in at least some camping areas.
    - camping_fee_note: Text indicating nightly fee for electric hookup sites (could be a range).
    - camping_reference_url: URL documenting camping facilities and fees.

    Trails:
    - trails.name: Name of a trail that is 3+ miles.
    - trails.length_miles: Length of that trail (e.g., "3.4 miles" or "5 mi").
    - trails.elevation_gain_ft: Elevation gain of that trail (e.g., "600 ft").
    - trails.trail_reference_url: URL documenting trail information.
    - trails.difficulty_variety_note: Text indicating the park offers trails of varying difficulty (at least Easy and Moderate).

    Accessibility:
    - accessibility.ada_note: Text indicating at least one ADA-accessible or wheelchair-friendly trail or facility.
    - accessibility.accessibility_reference_url: URL documenting accessibility features.

    Reservations:
    - reservations.reservation_url: URL to the park's reservation system (online).
    - reservations.summer_2026_availability_note: Text indicating the park is open and accepting reservations during June–August 2026.
    - reservations.online_system_note: Text indicating that the reservation system allows advance booking.

    Amenities:
    - amenities.additional_facilities_note: Text indicating picnic shelters, group camping areas, or day-use facilities.
    - amenities.visitor_services_note: Text indicating a visitor center or contact station along with operating hours.

    Special rules for URL extraction:
    - Extract only URLs explicitly present in the answer text. Do not invent.
    - Normalize links provided in markdown or embedded text as full URLs.
    - If any required field is not present, set it to null.

    Return JSON with:
    {
      "parks": [ParkInfo, ParkInfo, ParkInfo, ParkInfo]
    }
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _clean_urls(urls: List[Optional[str]]) -> List[str]:
    return [u for u in urls if isinstance(u, str) and u.strip() != ""]


def _collect_all_sources_for_park(p: ParkInfo) -> List[str]:
    return _clean_urls([
        p.official_url,
        p.camping.camping_reference_url,
        p.trails.trail_reference_url,
        p.accessibility.accessibility_reference_url,
        p.reservations.reservation_url,
    ])


async def _verify_with_sources_or_fail(
    evaluator: Evaluator,
    *,
    parent_node,
    node_id: str,
    desc: str,
    claim: str,
    primary_sources: List[str],
    fallback_sources: List[str],
    critical: bool = True,
    additional_instruction: str = "None",
) -> None:
    sources_to_use = primary_sources if primary_sources else fallback_sources
    if not sources_to_use:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=desc,
            parent=parent_node,
            critical=critical,
        )
        return

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=sources_to_use,
        additional_instruction=additional_instruction,
    )


# --------------------------------------------------------------------------- #
# Verification for one park                                                   #
# --------------------------------------------------------------------------- #
async def verify_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkInfo,
    park_index: int,
    all_states: List[Optional[str]],
) -> None:
    idx1 = park_index + 1
    park_node = evaluator.add_parallel(
        id=f"park_{idx1}",
        desc=[
            "First state park meeting all requirements",
            "Second state park meeting all requirements",
            "Third state park meeting all requirements",
            "Fourth state park meeting all requirements",
        ][park_index],
        parent=parent_node,
        critical=False,
    )

    # Identification
    ident_node = evaluator.add_parallel(
        id=f"park_{idx1}_identification",
        desc="Park identification and location verification",
        parent=park_node,
        critical=True,
    )

    # park_name leaf: verify against official or any references
    if park.name and park.name.strip():
        await _verify_with_sources_or_fail(
            evaluator,
            parent_node=ident_node,
            node_id=f"park_{idx1}_name",
            desc="Provide the official name of the state park",
            claim=f"The park's official name is '{park.name}'.",
            primary_sources=_clean_urls([park.official_url]),
            fallback_sources=_collect_all_sources_for_park(park),
            critical=True,
            additional_instruction="Confirm the park name shown on the official/reference page. Allow minor formatting differences or punctuation.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"park_{idx1}_name",
            desc="Provide the official name of the state park",
            parent=ident_node,
            critical=True,
        )

    # park_state leaf: verify state using official or any references
    if park.state and park.state.strip():
        await _verify_with_sources_or_fail(
            evaluator,
            parent_node=ident_node,
            node_id=f"park_{idx1}_state",
            desc="Specify the U.S. state where the park is located",
            claim=f"This park is located in {park.state}, United States.",
            primary_sources=_clean_urls([park.official_url]),
            fallback_sources=_collect_all_sources_for_park(park),
            critical=True,
            additional_instruction="Verify the state's name on the page or clear context on the official park system domain.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"park_{idx1}_state",
            desc="Specify the U.S. state where the park is located",
            parent=ident_node,
            critical=True,
        )

    # official_website leaf: URL existence
    evaluator.add_custom_node(
        result=bool(park.official_url and park.official_url.strip()),
        id=f"park_{idx1}_official_website",
        desc="Provide the URL to the park's official state parks system webpage",
        parent=ident_node,
        critical=True,
    )

    # unique_state leaf: ensure states are all distinct
    this_state = (park.state or "").strip().lower()
    other_states = [(s or "").strip().lower() for i, s in enumerate(all_states) if i != park_index]
    unique_state_result = bool(this_state) and (this_state not in set(other_states)) and (all_states.count(park.state) == 1)
    evaluator.add_custom_node(
        result=unique_state_result,
        id=f"park_{idx1}_unique_state",
        desc="Verify this park is in a different state than the other three parks",
        parent=ident_node,
        critical=True,
    )

    # Camping
    camp_node = evaluator.add_parallel(
        id=f"park_{idx1}_camping",
        desc="Camping facilities and requirements",
        parent=park_node,
        critical=True,
    )

    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=camp_node,
        node_id=f"park_{idx1}_electric_hookups",
        desc="Confirm the park offers campsites with electric hookups (30 amp or 50 amp)",
        claim="The park offers campsites with electric hookups (30 amp or 50 amp).",
        primary_sources=_clean_urls([park.camping.camping_reference_url]),
        fallback_sources=_clean_urls([park.official_url]),
        critical=True,
        additional_instruction="Look for 'electric', '30 amp', '50 amp' in campground amenities or site types. Accept RV sites with electric.",
    )

    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=camp_node,
        node_id=f"park_{idx1}_waterfront",
        desc="Confirm the park offers lakefront, riverfront, or oceanfront camping sites",
        claim="The park offers waterfront camping sites such as lakefront, riverfront, or oceanfront.",
        primary_sources=_clean_urls([park.camping.camping_reference_url]),
        fallback_sources=_clean_urls([park.official_url]),
        critical=True,
        additional_instruction="Accept mentions like 'waterfront', 'lakeside', 'riverside', 'beachside', or sites adjacent to a body of water.",
    )

    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=camp_node,
        node_id=f"park_{idx1}_restrooms",
        desc="Confirm the park has on-site restrooms with running water",
        claim="The campground provides on-site restrooms with running water (modern/flush toilets).",
        primary_sources=_clean_urls([park.camping.camping_reference_url]),
        fallback_sources=_clean_urls([park.official_url]),
        critical=True,
        additional_instruction="Look for 'modern restrooms', 'flush toilets', or 'running water' in campground amenities.",
    )

    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=camp_node,
        node_id=f"park_{idx1}_showers",
        desc="Confirm the park has shower facilities available to campers",
        claim="The campground has shower facilities available to campers.",
        primary_sources=_clean_urls([park.camping.camping_reference_url]),
        fallback_sources=_clean_urls([park.official_url]),
        critical=True,
        additional_instruction="Accept 'showers', 'shower house', or 'bathhouse with showers'.",
    )

    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=camp_node,
        node_id=f"park_{idx1}_pets",
        desc="Confirm the park allows pets in at least some camping areas",
        claim="Pets are allowed in at least some camping areas of the park.",
        primary_sources=_clean_urls([park.camping.camping_reference_url]),
        fallback_sources=_clean_urls([park.official_url]),
        critical=True,
        additional_instruction="Look for 'pets allowed', 'leashed dogs permitted', or designated pet-friendly loops/areas.",
    )

    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=camp_node,
        node_id=f"park_{idx1}_camping_fees",
        desc="Document the nightly camping fee for sites with electric hookups is between $25-$75",
        claim="The nightly camping fee for electric hookup sites is between $25 and $75.",
        primary_sources=_clean_urls([park.camping.camping_reference_url]),
        fallback_sources=_clean_urls([park.official_url]),
        critical=True,
        additional_instruction="Verify fee tables or rate info for electric/RV sites. Accept seasonal/range pricing if typical rates fall within $25–$75.",
    )

    evaluator.add_custom_node(
        result=bool(park.camping.camping_reference_url and park.camping.camping_reference_url.strip()),
        id=f"park_{idx1}_camping_reference",
        desc="Provide a reference URL documenting camping facilities and fees",
        parent=camp_node,
        critical=True,
    )

    # Trails
    trails_node = evaluator.add_parallel(
        id=f"park_{idx1}_trails",
        desc="Hiking trail requirements",
        parent=park_node,
        critical=True,
    )

    trail_3_node = evaluator.add_parallel(
        id=f"park_{idx1}_trail_3_miles",
        desc="Identify at least one hiking trail that is 3+ miles in length",
        parent=trails_node,
        critical=True,
    )

    # trail_name
    if park.trails.name and park.trails.name.strip():
        await _verify_with_sources_or_fail(
            evaluator,
            parent_node=trail_3_node,
            node_id=f"park_{idx1}_trail_name",
            desc="Provide the name of a trail that is 3+ miles long",
            claim=f"There is a hiking trail named '{park.trails.name}' at this park.",
            primary_sources=_clean_urls([park.trails.trail_reference_url]),
            fallback_sources=_clean_urls([park.official_url]),
            critical=True,
            additional_instruction="Confirm the trail name on the trail information page. Allow minor variants or alternate naming.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"park_{idx1}_trail_name",
            desc="Provide the name of a trail that is 3+ miles long",
            parent=trail_3_node,
            critical=True,
        )

    # trail_length (>= 3 miles)
    if park.trails.length_miles and park.trails.length_miles.strip():
        await _verify_with_sources_or_fail(
            evaluator,
            parent_node=trail_3_node,
            node_id=f"park_{idx1}_trail_length",
            desc="Document the trail length is 3 miles or greater",
            claim=f"The trail '{park.trails.name or 'the selected trail'}' has a documented length of at least 3 miles (claimed: {park.trails.length_miles}).",
            primary_sources=_clean_urls([park.trails.trail_reference_url]),
            fallback_sources=_clean_urls([park.official_url]),
            critical=True,
            additional_instruction="Check the documented mileage. Accept reasonable rounding (e.g., 2.9 counted as 3 if common convention supports). Prefer exact 3+ miles.",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"park_{idx1}_trail_length",
            desc="Document the trail length is 3 miles or greater",
            parent=trail_3_node,
            critical=True,
        )

    # trail_elevation (>= 500 ft)
    if park.trails.elevation_gain_ft and park.trails.elevation_gain_ft.strip():
        await _verify_with_sources_or_fail(
            evaluator,
            parent_node=trail_3_node,
            node_id=f"park_{idx1}_trail_elevation",
            desc="Document the trail has an elevation gain of at least 500 feet",
            claim=f"The trail '{park.trails.name or 'the selected trail'}' has a documented elevation gain of at least 500 feet (claimed: {park.trails.elevation_gain_ft}).",
            primary_sources=_clean_urls([park.trails.trail_reference_url]),
            fallback_sources=_clean_urls([park.official_url]),
            critical=True,
            additional_instruction="Check elevation gain values in trail descriptions or stats. Accept reasonable unit conversions (ft vs m).",
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id=f"park_{idx1}_trail_elevation",
            desc="Document the trail has an elevation gain of at least 500 feet",
            parent=trail_3_node,
            critical=True,
        )

    # difficulty variety (Easy and Moderate)
    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=trails_node,
        node_id=f"park_{idx1}_difficulty_variety",
        desc="Confirm the park offers trails of varying difficulty (at least Easy and Moderate options)",
        claim="The park offers trails of varying difficulty, including at least Easy and Moderate options.",
        primary_sources=_clean_urls([park.trails.trail_reference_url]),
        fallback_sources=_clean_urls([park.official_url]),
        critical=True,
        additional_instruction="Look for difficulty ratings or descriptors indicating 'Easy' and 'Moderate' trails in the system.",
    )

    evaluator.add_custom_node(
        result=bool(park.trails.trail_reference_url and park.trails.trail_reference_url.strip()),
        id=f"park_{idx1}_trail_reference",
        desc="Provide a reference URL documenting trail information",
        parent=trails_node,
        critical=True,
    )

    # Accessibility
    acc_node = evaluator.add_parallel(
        id=f"park_{idx1}_accessibility",
        desc="Accessibility features",
        parent=park_node,
        critical=True,
    )

    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=acc_node,
        node_id=f"park_{idx1}_ada_facility",
        desc="Confirm the park has at least one ADA-accessible or wheelchair-friendly trail or facility",
        claim="The park has at least one ADA-accessible or wheelchair-friendly trail or facility.",
        primary_sources=_clean_urls([park.accessibility.accessibility_reference_url]),
        fallback_sources=_clean_urls([park.official_url]),
        critical=True,
        additional_instruction="Look for ADA designations, accessible trails, accessible facilities, ramps, or accessible restrooms.",
    )

    evaluator.add_custom_node(
        result=bool(park.accessibility.accessibility_reference_url and park.accessibility.accessibility_reference_url.strip()),
        id=f"park_{idx1}_accessibility_reference",
        desc="Provide a reference URL documenting accessibility features",
        parent=acc_node,
        critical=True,
    )

    # Reservations
    res_node = evaluator.add_parallel(
        id=f"park_{idx1}_reservations",
        desc="Reservation system and availability",
        parent=park_node,
        critical=True,
    )

    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=res_node,
        node_id=f"park_{idx1}_online_system",
        desc="Confirm the park uses an online reservation system allowing advance booking",
        claim="The park uses an online reservation system that allows advance booking.",
        primary_sources=_clean_urls([park.reservations.reservation_url]),
        fallback_sources=_collect_all_sources_for_park(park),
        critical=True,
        additional_instruction="Accept official reservation portals, ReserveAmerica, or state park systems with online booking interfaces.",
    )

    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=res_node,
        node_id=f"park_{idx1}_summer_availability",
        desc="Confirm the park is open and accepting reservations during June-August 2026",
        claim="The park is open and accepting reservations during June, July, and August 2026.",
        primary_sources=_clean_urls([park.reservations.reservation_url]),
        fallback_sources=_collect_all_sources_for_ark(park) if False else _collect_all_sources_for_park(park),
        critical=True,
        additional_instruction="Verify that booking calendars or policies indicate availability for June–August 2026. Accept if the system shows summer 2026 dates or policies explicitly state summer availability.",
    )

    evaluator.add_custom_node(
        result=bool(park.reservations.reservation_url and park.reservations.reservation_url.strip()),
        id=f"park_{idx1}_reservation_url",
        desc="Provide the URL to the park's reservation system",
        parent=res_node,
        critical=True,
    )

    # Amenities
    amen_node = evaluator.add_parallel(
        id=f"park_{idx1}_amenities",
        desc="Additional facilities and services",
        parent=park_node,
        critical=True,
    )

    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=amen_node,
        node_id=f"park_{idx1}_additional_facilities",
        desc="Confirm the park has picnic shelters, group camping areas, or day-use facilities",
        claim="The park has picnic shelters, group camping areas, or day-use facilities.",
        primary_sources=_clean_urls([park.official_url]),
        fallback_sources=_collect_all_sources_for_park(park),
        critical=True,
        additional_instruction="Look for facility listings such as picnic shelters, group sites, or day-use areas on the official page.",
    )

    await _verify_with_sources_or_fail(
        evaluator,
        parent_node=amen_node,
        node_id=f"park_{idx1}_visitor_services",
        desc="Confirm the park has a visitor center or contact station with documented operating hours",
        claim="The park has a visitor center or contact station with documented operating hours.",
        primary_sources=_clean_urls([park.official_url]),
        fallback_sources=_collect_all_sources_for_park(park),
        critical=True,
        additional_instruction="Look for 'visitor center', 'contact station', and posted hours such as 'open daily 9am–5pm'.",
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

    extracted = await evaluator.extract(
        prompt=prompt_extract_parks(),
        template_class=ParksExtraction,
        extraction_name="parks_extraction",
    )

    # Normalize to exactly 4 parks
    parks: List[ParkInfo] = list(extracted.parks[:4])
    while len(parks) < 4:
        parks.append(ParkInfo())

    states_list = [p.state for p in parks]
    evaluator.add_custom_info(
        info={"extracted_states": states_list},
        info_type="auxiliary",
        info_name="state_distribution",
    )

    # Build subtrees for each park
    for i in range(4):
        await verify_park(
            evaluator=evaluator,
            parent_node=root,
            park=parks[i],
            park_index=i,
            all_states=states_list,
        )

    return evaluator.get_summary()