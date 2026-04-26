import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "rmnp_winter_permit_jan2026"
TASK_DESCRIPTION = (
    "I'm planning a 3-night wilderness backpacking trip in Rocky Mountain National Park during January 2026. "
    "I need to understand all the permit requirements and essential regulations for this trip. Please provide the "
    "following information with supporting reference URLs from the National Park Service:\n\n"
    "1. What is the total cost for the wilderness camping permit during January 2026, and what permit season does January fall under?\n\n"
    "2. How and when can I make a reservation for this winter wilderness permit? Are advance reservations available?\n\n"
    "3. Where must I pick up the physical permit, what are the office hours, and what is the complete address?\n\n"
    "4. What are the critical camping location rules for wilderness camping in RMNP (specifically regarding dispersed vs. designated camping)?\n\n"
    "5. What are the bear-related food storage requirements, including equipment specifications and distance requirements from campsite?\n\n"
    "Please ensure all answers are supported by official NPS or Recreation.gov URLs."
)


class RMNPRequirementsExtraction(BaseModel):
    january_permit_season: Optional[str] = None
    january_total_permit_cost: Optional[str] = None

    reservation_how: Optional[str] = None
    reservation_when: Optional[str] = None
    advance_reservations_available: Optional[str] = None

    pickup_location_name: Optional[str] = None
    pickup_office_hours: Optional[str] = None
    pickup_address: Optional[str] = None

    camping_rule_designated_vs_dispersed: Optional[str] = None

    food_storage_equipment: Optional[str] = None
    food_storage_distance_placement: Optional[str] = None

    urls_section1: List[str] = Field(default_factory=list)  # Season + total cost
    urls_section2: List[str] = Field(default_factory=list)  # Reservation process and availability
    urls_section3: List[str] = Field(default_factory=list)  # Pickup location, hours, address
    urls_section4: List[str] = Field(default_factory=list)  # Designated vs dispersed camping rule
    urls_section5: List[str] = Field(default_factory=list)  # Bear food storage equipment + distance
    all_urls: List[str] = Field(default_factory=list)       # (Optional) union of all URLs


def prompt_extract_requirements() -> str:
    return (
        "Extract the specific RMNP January 2026 winter wilderness permit requirements and key regulations exactly as stated "
        "in the provided answer. Return the following fields:\n"
        "1) january_permit_season: String stating which permit season January falls under (e.g., 'winter season', 'non-peak season').\n"
        "2) january_total_permit_cost: String stating the total cost for the wilderness camping permit relevant for January 2026. "
        "   If the answer specifies multiple components (e.g., reservation fee plus per-person-per-night fee), include the combined description "
        "   exactly as the answer states (do not compute; just extract the text).\n"
        "3) reservation_how: String explaining how the winter wilderness permit is obtained (e.g., 'walk-up only', 'in-person at Wilderness Office', "
        "   'online on Recreation.gov', etc.).\n"
        "4) reservation_when: String describing when the permit can be obtained (timing/window).\n"
        "5) advance_reservations_available: String stating whether advance reservations are available for winter wilderness permits (e.g., 'yes', 'no', 'not available').\n"
        "6) pickup_location_name: String naming where the physical permit must be picked up (office/location name).\n"
        "7) pickup_office_hours: String with office hours relevant to permit pickup.\n"
        "8) pickup_address: String with the complete address of the pickup office.\n"
        "9) camping_rule_designated_vs_dispersed: String stating whether wilderness camping must be at designated sites and whether dispersed camping is allowed or prohibited.\n"
        "10) food_storage_equipment: String describing the required bear-related food storage equipment (e.g., 'bear-resistant canister required' including specs/brands if stated).\n"
        "11) food_storage_distance_placement: String describing distance/placement requirements for storing the bear-resistant container relative to the campsite.\n"
        "\n"
        "Also extract supporting reference URLs cited in the answer for each numbered section. Only extract URLs explicitly present in the answer. "
        "Include them verbatim (plain URLs or markdown links):\n"
        "• urls_section1: URLs supporting the permit season classification and total cost.\n"
        "• urls_section2: URLs supporting the reservation method, timing, and advance reservations availability.\n"
        "• urls_section3: URLs supporting pickup location, office hours, and address.\n"
        "• urls_section4: URLs supporting designated vs. dispersed camping rules.\n"
        "• urls_section5: URLs supporting bear food storage requirements (equipment + distance/placement).\n"
        "• all_urls: Include a de-duplicated union of all URLs mentioned across sections.\n"
        "\n"
        "If any textual field is not mentioned in the answer, set it to null. If any section has no URLs cited in the answer, return an empty list for that section. "
        "Do not invent or infer any information or URLs. Extract exactly what the answer provides."
    )


def _is_official_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        host = parsed.netloc.lower()
        return host.endswith("nps.gov") or host.endswith("recreation.gov")
    except Exception:
        return False


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


async def _verify_permit_season_and_cost(
    evaluator: Evaluator,
    parent_node,
    data: RMNPRequirementsExtraction,
) -> None:
    group = evaluator.add_parallel(
        id="permit_season_and_total_cost",
        desc="Provide the permit season classification for January and the total wilderness permit cost.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: identify_permit_season_for_january
    season_leaf = evaluator.add_leaf(
        id="identify_permit_season_for_january",
        desc="State which wilderness permit season January falls under.",
        parent=group,
        critical=True,
    )
    if not data.january_permit_season or not data.urls_section1:
        season_leaf.score = 0.0
        season_leaf.status = "failed"
    else:
        claim = (
            f"In Rocky Mountain National Park (RMNP), January falls under the '{data.january_permit_season}' wilderness permit season."
        )
        await evaluator.verify(
            claim=claim,
            node=season_leaf,
            sources=data.urls_section1,
            additional_instruction=(
                "Verify the season classification for January on official RMNP/NPS/Recreation.gov pages. "
                "Allow reasonable synonyms (e.g., 'winter season', 'winter backcountry period', etc.), but ensure it matches RMNP official policy."
            ),
        )

    # Leaf: state_total_permit_cost
    cost_leaf = evaluator.add_leaf(
        id="state_total_permit_cost",
        desc="State the total cost for the wilderness camping permit for that season.",
        parent=group,
        critical=True,
    )
    if not data.january_total_permit_cost or not data.urls_section1:
        cost_leaf.score = 0.0
        cost_leaf.status = "failed"
    else:
        claim = (
            f"The total wilderness camping permit cost applicable for January 2026 is: {data.january_total_permit_cost}."
        )
        await evaluator.verify(
            claim=claim,
            node=cost_leaf,
            sources=data.urls_section1,
            additional_instruction=(
                "Confirm the stated fee/cost exactly as described on official RMNP/NPS/Recreation.gov sources. "
                "Fee descriptions may include reservation fees, per-person-per-night fees, seasonal differences, or walk-up conditions; "
                "judge correctness based on the official page(s)."
            ),
        )


async def _verify_reservation_process(
    evaluator: Evaluator,
    parent_node,
    data: RMNPRequirementsExtraction,
) -> None:
    group = evaluator.add_parallel(
        id="reservation_process_and_availability",
        desc="Explain how/when the winter wilderness permit is obtained and whether advance reservations are available.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: explain_how_to_obtain_permit
    how_leaf = evaluator.add_leaf(
        id="explain_how_to_obtain_permit",
        desc="Explain the method/process to obtain the winter wilderness permit (e.g., walk-up, in person, etc.).",
        parent=group,
        critical=True,
    )
    if not data.reservation_how or not data.urls_section2:
        how_leaf.score = 0.0
        how_leaf.status = "failed"
    else:
        claim = f"In January, the winter wilderness permit is obtained via: {data.reservation_how}."
        await evaluator.verify(
            claim=claim,
            node=how_leaf,
            sources=data.urls_section2,
            additional_instruction=(
                "Verify the process (e.g., walk-up/in-person at Wilderness Office, online, phone) for obtaining winter backcountry permits "
                "for RMNP January on official pages."
            ),
        )

    # Leaf: explain_when_to_obtain_permit
    when_leaf = evaluator.add_leaf(
        id="explain_when_to_obtain_permit",
        desc="State when the permit can be obtained (timing/availability window as described by official sources).",
        parent=group,
        critical=True,
    )
    if not data.reservation_when or not data.urls_section2:
        when_leaf.score = 0.0
        when_leaf.status = "failed"
    else:
        claim = f"The permit can be obtained during: {data.reservation_when}."
        await evaluator.verify(
            claim=claim,
            node=when_leaf,
            sources=data.urls_section2,
            additional_instruction=(
                "Verify timing/availability specifics (e.g., winter period dates, daily hours, which months are eligible, "
                "or walk-up windows) from RMNP/NPS/Recreation.gov official pages."
            ),
        )

    # Leaf: state_advance_reservation_availability
    adv_leaf = evaluator.add_leaf(
        id="state_advance_reservation_availability",
        desc="State whether advance reservations are available for winter wilderness permits.",
        parent=group,
        critical=True,
    )
    if not data.advance_reservations_available or not data.urls_section2:
        adv_leaf.score = 0.0
        adv_leaf.status = "failed"
    else:
        claim = f"Advance reservations for RMNP winter wilderness permits in January are: {data.advance_reservations_available}."
        await evaluator.verify(
            claim=claim,
            node=adv_leaf,
            sources=data.urls_section2,
            additional_instruction=(
                "Check official policy regarding whether advance reservations are offered or not for winter permits (January). "
                "Confirm using official RMNP/NPS/Recreation.gov sources."
            ),
        )


async def _verify_permit_pickup(
    evaluator: Evaluator,
    parent_node,
    data: RMNPRequirementsExtraction,
) -> None:
    group = evaluator.add_parallel(
        id="permit_pickup_logistics",
        desc="State where the physical permit must be picked up, the office hours, and the complete address.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: state_pickup_location
    loc_leaf = evaluator.add_leaf(
        id="state_pickup_location",
        desc="State where the physical permit must be picked up (which office/location).",
        parent=group,
        critical=True,
    )
    if not data.pickup_location_name or not data.urls_section3:
        loc_leaf.score = 0.0
        loc_leaf.status = "failed"
    else:
        claim = f"The physical permit must be picked up at: {data.pickup_location_name}."
        await evaluator.verify(
            claim=claim,
            node=loc_leaf,
            sources=data.urls_section3,
            additional_instruction=(
                "Verify the specified pickup office/location for RMNP winter backcountry permits using official RMNP/NPS/Recreation.gov pages."
            ),
        )

    # Leaf: state_office_hours
    hours_leaf = evaluator.add_leaf(
        id="state_office_hours",
        desc="State the Wilderness Office hours relevant to permit pickup.",
        parent=group,
        critical=True,
    )
    if not data.pickup_office_hours or not data.urls_section3:
        hours_leaf.score = 0.0
        hours_leaf.status = "failed"
    else:
        claim = f"Wilderness Office hours (permit pickup) are: {data.pickup_office_hours}."
        await evaluator.verify(
            claim=claim,
            node=hours_leaf,
            sources=data.urls_section3,
            additional_instruction=(
                "Confirm the office hours for picking up the permit from official RMNP/NPS sources."
            ),
        )

    # Leaf: state_complete_address
    addr_leaf = evaluator.add_leaf(
        id="state_complete_address",
        desc="Provide the complete address of the permit pickup office.",
        parent=group,
        critical=True,
    )
    if not data.pickup_address or not data.urls_section3:
        addr_leaf.score = 0.0
        addr_leaf.status = "failed"
    else:
        claim = f"The complete address for the permit pickup office is: {data.pickup_address}."
        await evaluator.verify(
            claim=claim,
            node=addr_leaf,
            sources=data.urls_section3,
            additional_instruction=(
                "Verify the full address for the specified office using official RMNP/NPS/Recreation.gov sources."
            ),
        )


async def _verify_camping_rules(
    evaluator: Evaluator,
    parent_node,
    data: RMNPRequirementsExtraction,
) -> None:
    group = evaluator.add_parallel(
        id="camping_location_rules_designated_vs_dispersed",
        desc="State camping location rules for RMNP wilderness camping regarding dispersed vs. designated-site camping.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: state_designated_vs_dispersed_rule
    rule_leaf = evaluator.add_leaf(
        id="state_designated_vs_dispersed_rule",
        desc="State whether wilderness camping must be at designated sites and whether dispersed camping is allowed.",
        parent=group,
        critical=True,
    )
    if not data.camping_rule_designated_vs_dispersed or not data.urls_section4:
        rule_leaf.score = 0.0
        rule_leaf.status = "failed"
    else:
        claim = f"RMNP wilderness camping rule: {data.camping_rule_designated_vs_dispersed}."
        await evaluator.verify(
            claim=claim,
            node=rule_leaf,
            sources=data.urls_section4,
            additional_instruction=(
                "Confirm whether RMNP allows only designated-site wilderness camping and whether dispersed camping is prohibited. "
                "Accept synonymous wording but ensure the rule matches official RMNP policy."
            ),
        )


async def _verify_bear_food_storage(
    evaluator: Evaluator,
    parent_node,
    data: RMNPRequirementsExtraction,
) -> None:
    group = evaluator.add_parallel(
        id="bear_food_storage_requirements",
        desc="State bear-related food storage requirements, including equipment specifications and distance/placement requirements.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: state_required_food_storage_equipment
    equip_leaf = evaluator.add_leaf(
        id="state_required_food_storage_equipment",
        desc="State the required bear-related food storage equipment specifications (what type of storage is required).",
        parent=group,
        critical=True,
    )
    if not data.food_storage_equipment or not data.urls_section5:
        equip_leaf.score = 0.0
        equip_leaf.status = "failed"
    else:
        claim = f"Required bear-related food storage equipment: {data.food_storage_equipment}."
        await evaluator.verify(
            claim=claim,
            node=equip_leaf,
            sources=data.urls_section5,
            additional_instruction=(
                "Verify the required equipment (e.g., approved bear-resistant canister) and any specification notes from official RMNP/NPS sources."
            ),
        )

    # Leaf: state_distance_or_placement_requirement
    place_leaf = evaluator.add_leaf(
        id="state_distance_or_placement_requirement",
        desc="State any required distance/placement requirement for storing the bear-resistant container relative to the campsite.",
        parent=group,
        critical=True,
    )
    if not data.food_storage_distance_placement or not data.urls_section5:
        place_leaf.score = 0.0
        place_leaf.status = "failed"
    else:
        claim = f"Placement/distance requirement for bear-resistant container: {data.food_storage_distance_placement}."
        await evaluator.verify(
            claim=claim,
            node=place_leaf,
            sources=data.urls_section5,
            additional_instruction=(
                "Confirm the stated distance/placement requirement (e.g., specified number of feet/yards from campsite, out of sight, etc.) "
                "from official RMNP/NPS sources."
            ),
        )


def _add_citations_checks(
    evaluator: Evaluator,
    parent_node,
    data: RMNPRequirementsExtraction,
) -> None:
    group = evaluator.add_parallel(
        id="citations_official_urls",
        desc="All requested answers are supported with official reference URLs from NPS and/or Recreation.gov.",
        parent=parent_node,
        critical=True,
    )

    # Leaf: urls_are_official_sources (custom binary)
    all_section_urls = (
        data.urls_section1
        + data.urls_section2
        + data.urls_section3
        + data.urls_section4
        + data.urls_section5
    )
    all_section_urls = _dedupe_urls(all_section_urls)
    all_official = bool(all_section_urls) and all(_is_official_url(u) for u in all_section_urls)

    evaluator.add_custom_node(
        result=all_official,
        id="urls_are_official_sources",
        desc="Provided reference URLs are from official NPS (nps.gov) and/or Recreation.gov domains.",
        parent=group,
        critical=True,
    )

    # Leaf: citations_cover_all_answer_sections (custom binary)
    sections_have_official = [
        (len(data.urls_section1) > 0) and any(_is_official_url(u) for u in data.urls_section1),
        (len(data.urls_section2) > 0) and any(_is_official_url(u) for u in data.urls_section2),
        (len(data.urls_section3) > 0) and any(_is_official_url(u) for u in data.urls_section3),
        (len(data.urls_section4) > 0) and any(_is_official_url(u) for u in data.urls_section4),
        (len(data.urls_section5) > 0) and any(_is_official_url(u) for u in data.urls_section5),
    ]
    coverage_ok = all(sections_have_official)

    evaluator.add_custom_node(
        result=coverage_ok,
        id="citations_cover_all_answer_sections",
        desc="Citations are provided such that each of the five requested sections (1–5) has supporting official URL evidence.",
        parent=group,
        critical=True,
    )

    evaluator.add_custom_info(
        info={
            "total_unique_urls": len(all_section_urls),
            "official_urls_count": sum(1 for u in all_section_urls if _is_official_url(u)),
            "sections_with_official": {
                "section1": sections_have_official[0],
                "section2": sections_have_official[1],
                "section3": sections_have_official[2],
                "section4": sections_have_official[3],
                "section5": sections_have_official[4],
            },
            "all_urls": all_section_urls,
        },
        info_type="url_statistics",
    )


async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
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

    extraction = await evaluator.extract(
        prompt=prompt_extract_requirements(),
        template_class=RMNPRequirementsExtraction,
        extraction_name="rmnp_requirements",
    )

    winter_node = evaluator.add_parallel(
        id="winter_backcountry_trip_requirements",
        desc="Provide January 2026 RMNP winter wilderness permit requirements and key wilderness regulations.",
        parent=root,
        critical=True,
    )

    await _verify_permit_season_and_cost(evaluator, winter_node, extraction)
    await _verify_reservation_process(evaluator, winter_node, extraction)
    await _verify_permit_pickup(evaluator, winter_node, extraction)
    await _verify_camping_rules(evaluator, winter_node, extraction)
    await _verify_bear_food_storage(evaluator, winter_node, extraction)
    _add_citations_checks(evaluator, winter_node, extraction)

    return evaluator.get_summary()