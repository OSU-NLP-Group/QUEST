import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "np_backcountry_permits_2026_group_8_10"
TASK_DESCRIPTION = (
    "A wilderness guide service is planning to offer multi-day backcountry camping trips for groups of 8-10 people across several U.S. national parks in summer 2026. "
    "They need comprehensive information about permit systems, regulations, and logistics to determine which parks are suitable for their group programs.\n\n"
    "Research and provide detailed information for the following 5 national parks: Rocky Mountain National Park (Colorado), Yosemite National Park (California), "
    "Grand Canyon National Park (Arizona), Sequoia & Kings Canyon National Parks (California), and Olympic National Park (Washington).\n\n"
    "For each park, provide:\n"
    "1. Group Size Regulations\n"
    "2. Advance Reservation System\n"
    "3. Complete Fee Structure\n"
    "4. Bear-Resistant Food Storage Requirements\n"
    "5. Group-Specific Restrictions for groups of 8–10\n"
    "6. Official Reference URL(s) (NPS or Recreation.gov)\n\n"
    "Answer must be based on current (2025–2026) regulations from official sources."
)

# Canonical park names and node ids
PARKS: List[Tuple[str, str]] = [
    ("rocky_mountain_np", "Rocky Mountain National Park"),
    ("yosemite_np", "Yosemite National Park"),
    ("grand_canyon_np", "Grand Canyon National Park"),
    ("sequoia_kings_canyon_np", "Sequoia & Kings Canyon National Parks"),
    ("olympic_np", "Olympic National Park"),
]
EXPECTED_PARK_SET = {label for _, label in PARKS}


# --------------------------------------------------------------------------- #
# Pydantic models for extraction                                              #
# --------------------------------------------------------------------------- #
class GroupSizeRegulations(BaseModel):
    max_group_size: Optional[str] = None
    site_type_distinction: Optional[str] = None
    status_8to10: Optional[str] = None


class AdvanceReservations(BaseModel):
    advance_window: Optional[str] = None
    process_type: Optional[str] = None
    key_dates_summer_2026: Optional[str] = None


class Fees(BaseModel):
    base_or_reservation_fees: Optional[str] = None
    per_person_charges: Optional[str] = None
    refundability: Optional[str] = None


class BearStorage(BaseModel):
    required_storage_type: Optional[str] = None
    applicable_season_or_time_period: Optional[str] = None
    approved_alternatives: Optional[str] = None


class GroupSpecificRestrictions(BaseModel):
    restrictions_or_none: Optional[str] = None


class ParkDetails(BaseModel):
    official_urls: List[str] = Field(default_factory=list)
    current_timeframe: Optional[str] = None
    group_size_regulations: Optional[GroupSizeRegulations] = None
    advance_reservations: Optional[AdvanceReservations] = None
    fees: Optional[Fees] = None
    bear_storage: Optional[BearStorage] = None
    group_specific_restrictions: Optional[GroupSpecificRestrictions] = None


class ParksCoverageExtraction(BaseModel):
    parks_covered: List[str] = Field(default_factory=list)
    extra_parks: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_coverage() -> str:
    allowed = [
        "Rocky Mountain National Park",
        "Yosemite National Park",
        "Grand Canyon National Park",
        "Sequoia & Kings Canyon National Parks",
        "Olympic National Park",
    ]
    allowed_str = "; ".join(allowed)
    return f"""
Identify which national parks the answer actually provides backcountry permit/regulation/logistics information for.

Return:
- parks_covered: a list of canonical park names drawn ONLY from this allowed set:
  {allowed_str}
  Notes:
  • Treat "Sequoia National Park" and "Kings Canyon National Park" (alone or together) as the combined canonical name "Sequoia & Kings Canyon National Parks".
  • Normalize obvious abbreviations (e.g., RMNP -> Rocky Mountain National Park, GCNP -> Grand Canyon National Park, etc.).
- extra_parks: any other U.S. national parks (if any) that the answer mentions or includes with permit/regulation details that are NOT in the allowed set. Use their common names.

Do not include duplicates. If none, return empty arrays.
"""


def prompt_extract_park_details(park_name: str) -> str:
    return f"""
Extract the following fields for: {park_name}

Rules:
- Extract only what is explicitly stated in the answer.
- For official_urls, include only official National Park Service (nps.gov) or Recreation.gov URLs that the answer cites for this park.
- If a field is not present in the answer, set it to null (or an empty list for URLs).
- Keep values as short factual phrases or sentences taken from the answer (e.g., "max group size is 12" or "bear canisters required May–Oct").

Fields to extract (JSON):
- official_urls: array of strings (only nps.gov or recreation.gov)
- current_timeframe: string (statement indicating info is current for 2025–2026, or that 2026 details are not yet published), if present.

- group_size_regulations:
  - max_group_size: string
  - site_type_distinction: string (distinguishes individual vs group sites, or explicitly states none)
  - status_8to10: string (whether groups of 8–10 are permitted and under what conditions)

- advance_reservations:
  - advance_window: string (how far in advance permits can be reserved)
  - process_type: string (e.g., lottery, first-come-first-served, online vs in-person)
  - key_dates_summer_2026: string (key dates for summer 2026 or explicitly 'not yet published')

- fees:
  - base_or_reservation_fees: string (base/reservation/processing fees; state 'none' if none)
  - per_person_charges: string (any per-person fee tiers; or 'none')
  - refundability: string (which fees are refundable vs non-refundable; or 'none')

- bear_storage:
  - required_storage_type: string (e.g., bear canister/box/cable/pole; or 'none required')
  - applicable_season_or_time_period: string (season/dates/areas where requirement applies)
  - approved_alternatives: string (approved alternatives, or 'none')

- group_specific_restrictions:
  - restrictions_or_none: string (restrictions specifically affecting groups of 8–10; or 'none beyond general rules')
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _value_or_placeholder(val: Optional[str]) -> str:
    return val if (val is not None and str(val).strip() != "") else "[not specified in the answer]"


def _park_common_additional_instruction(park_label: str) -> str:
    return (
        f"Judge only the stated claim against the provided official source URLs for {park_label}. "
        f"Accept minor phrasing differences. Consider only backcountry/wilderness backpacking permits and regulations for {park_label}. "
        f"Official sources are limited to National Park Service (nps.gov) and Recreation.gov. "
        f"If a required detail was not stated in the answer, you should treat the claim as unsupported."
    )


# --------------------------------------------------------------------------- #
# Verification for one park                                                   #
# --------------------------------------------------------------------------- #
async def verify_park(
    evaluator: Evaluator,
    root_node,
    park_node_id: str,
    park_label: str,
    details: ParkDetails,
) -> None:
    # Park node (parallel aggregate)
    park_node = evaluator.add_parallel(
        id=park_node_id,
        desc=f"{park_label}: all required information provided",
        parent=root_node,
        critical=False,
    )

    # Prepare sources
    sources = details.official_urls or []

    # 1) Official URLs (critical)
    official_urls_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_official_urls",
        desc="Provides official NPS and/or Recreation.gov URL(s) that support the stated information for this park.",
        parent=park_node,
        critical=True,
    )
    claim_urls = (
        f"The provided URLs are official National Park Service (nps.gov) and/or Recreation.gov pages for {park_label}, "
        f"and they contain information about backcountry or wilderness permits/regulations relevant to this park."
    )
    await evaluator.verify(
        claim=claim_urls,
        node=official_urls_leaf,
        sources=sources,
        additional_instruction="A URL counts as official only if it is on nps.gov or recreation.gov and is relevant to backcountry permits/regulations.",
    )

    # 2) Current timeframe (critical)
    current_tf_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_current_timeframe",
        desc="Information is presented as current for the 2025–2026 timeframe (or explicitly notes if 2026-specific details are not yet published, with official source supporting that status).",
        parent=park_node,
        critical=True,
    )
    claim_tf = (
        f"In the answer, the information for {park_label} is presented as current for the 2025–2026 timeframe, "
        f"OR it explicitly notes that 2026-specific details are not yet published; and the provided official URLs are consistent with that status."
    )
    await evaluator.verify(
        claim=claim_tf,
        node=current_tf_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 3) Group size regulations (critical group)
    gsz_node = evaluator.add_parallel(
        id=f"{park_node_id}_group_size_regulations",
        desc="Group size regulations for backcountry camping",
        parent=park_node,
        critical=True,
    )

    # 3.1 max group size (critical)
    max_gsz_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_max_group_size",
        desc="States the maximum allowed group size for backcountry camping.",
        parent=gsz_node,
        critical=True,
    )
    max_group_size_val = _value_or_placeholder(
        details.group_size_regulations.max_group_size if details.group_size_regulations else None
    )
    claim_max_gsz = (
        f"According to the official sources, the maximum allowed backcountry camping group size in {park_label} is: '{max_group_size_val}'."
    )
    await evaluator.verify(
        claim=claim_max_gsz,
        node=max_gsz_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 3.2 site type distinction (critical)
    site_type_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_site_type_distinction",
        desc="Distinguishes between individual sites vs designated group sites (or explicitly states no distinction exists) where applicable.",
        parent=gsz_node,
        critical=True,
    )
    site_type_val = _value_or_placeholder(
        details.group_size_regulations.site_type_distinction if details.group_size_regulations else None
    )
    claim_site_type = (
        f"For {park_label}, the official sources indicate the following about individual vs. designated group sites (or no distinction): '{site_type_val}'."
    )
    await evaluator.verify(
        claim=claim_site_type,
        node=site_type_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 3.3 groups of 8–10 (critical)
    eight_to_ten_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_8to10_status",
        desc="States whether groups of 8–10 are permitted (and under what conditions, if any).",
        parent=gsz_node,
        critical=True,
    )
    eight_to_ten_val = _value_or_placeholder(
        details.group_size_regulations.status_8to10 if details.group_size_regulations else None
    )
    claim_eight_to_ten = (
        f"Official sources show whether groups of 8–10 people are permitted for backcountry camping in {park_label}, and under what conditions: '{eight_to_ten_val}'."
    )
    await evaluator.verify(
        claim=claim_eight_to_ten,
        node=eight_to_ten_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 4) Advance reservations (critical group)
    adv_node = evaluator.add_parallel(
        id=f"{park_node_id}_advance_reservations",
        desc="Advance reservation system details",
        parent=park_node,
        critical=True,
    )

    # 4.1 advance window (critical)
    adv_window_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_advance_window",
        desc="Specifies how far in advance permits can be reserved (days/months).",
        parent=adv_node,
        critical=True,
    )
    adv_window_val = _value_or_placeholder(
        details.advance_reservations.advance_window if details.advance_reservations else None
    )
    claim_adv_window = (
        f"According to official sources, permits for {park_label} can be reserved with the following advance window: '{adv_window_val}'."
    )
    await evaluator.verify(
        claim=claim_adv_window,
        node=adv_window_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 4.2 process type (critical)
    process_type_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_process_type",
        desc="Describes the reservation process (e.g., lottery vs first-come-first-served, online vs in-person pickup if relevant).",
        parent=adv_node,
        critical=True,
    )
    process_type_val = _value_or_placeholder(
        details.advance_reservations.process_type if details.advance_reservations else None
    )
    claim_process_type = (
        f"The official sources describe the permit reservation process for {park_label} as: '{process_type_val}'."
    )
    await evaluator.verify(
        claim=claim_process_type,
        node=process_type_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 4.3 key dates summer 2026 (critical)
    key_dates_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_key_dates_summer_2026",
        desc="Provides key dates relevant to the 2026 summer season for obtaining reservations (or explicitly states dates are not yet published, supported by official source).",
        parent=adv_node,
        critical=True,
    )
    key_dates_val = _value_or_placeholder(
        details.advance_reservations.key_dates_summer_2026 if details.advance_reservations else None
    )
    claim_key_dates = (
        f"The answer states the following for summer 2026 key reservation dates/status for {park_label}: '{key_dates_val}', "
        f"and this is supported by the provided official URLs (e.g., explicit dates or a statement that 2026 dates are not yet published)."
    )
    await evaluator.verify(
        claim=claim_key_dates,
        node=key_dates_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 5) Fees (critical group)
    fees_node = evaluator.add_parallel(
        id=f"{park_node_id}_fees",
        desc="Complete fee structure",
        parent=park_node,
        critical=True,
    )

    # 5.1 base/reservation fees (critical)
    base_fee_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_base_or_reservation_fees",
        desc="States any base and/or reservation/processing fees (or explicitly states none).",
        parent=fees_node,
        critical=True,
    )
    base_fee_val = _value_or_placeholder(details.fees.base_or_reservation_fees if details.fees else None)
    claim_base_fee = (
        f"Official sources show the base and/or reservation/processing fees for {park_label} backcountry permits as: '{base_fee_val}'. "
        f"Ignore park entrance fees; consider only permit/reservation fees."
    )
    await evaluator.verify(
        claim=claim_base_fee,
        node=base_fee_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 5.2 per-person charges (critical)
    per_person_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_per_person_charges",
        desc="States any per-person charges (and any tiers that exist), or explicitly states there are no per-person charges.",
        parent=fees_node,
        critical=True,
    )
    per_person_val = _value_or_placeholder(details.fees.per_person_charges if details.fees else None)
    claim_per_person = (
        f"Official sources show the per-person charges (if any) for {park_label} backcountry permits as: '{per_person_val}'. "
        f"Ignore entrance fees; evaluate only permit-related per-person charges."
    )
    await evaluator.verify(
        claim=claim_per_person,
        node=per_person_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 5.3 refundability (critical)
    refund_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_refundability",
        desc="Specifies which fees are refundable vs non-refundable (or explicitly states the refund policy is not offered/none).",
        parent=fees_node,
        critical=True,
    )
    refund_val = _value_or_placeholder(details.fees.refundability if details.fees else None)
    claim_refund = (
        f"Official sources specify the refundability policy for {park_label} backcountry permit/reservation fees as: '{refund_val}'."
    )
    await evaluator.verify(
        claim=claim_refund,
        node=refund_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 6) Bear-resistant food storage (critical group; alternatives non-critical)
    bear_node = evaluator.add_parallel(
        id=f"{park_node_id}_bear_storage",
        desc="Bear-resistant food storage requirements",
        parent=park_node,
        critical=True,
    )

    # 6.1 required storage type (critical)
    storage_type_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_required_storage_type",
        desc="Specifies required food-storage method(s) (e.g., canister, bear box, cable/pole), or states that no special bear-resistant method is required.",
        parent=bear_node,
        critical=True,
    )
    storage_type_val = _value_or_placeholder(details.bear_storage.required_storage_type if details.bear_storage else None)
    claim_storage_type = (
        f"Official sources specify the required backcountry food storage method(s) in {park_label} as: '{storage_type_val}'."
    )
    await evaluator.verify(
        claim=claim_storage_type,
        node=storage_type_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 6.2 applicable season/time period (critical)
    storage_period_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_applicable_season_or_time_period",
        desc="Specifies when/where the requirement applies (season, dates, or areas).",
        parent=bear_node,
        critical=True,
    )
    storage_period_val = _value_or_placeholder(
        details.bear_storage.applicable_season_or_time_period if details.bear_storage else None
    )
    claim_storage_period = (
        f"Official sources state when/where the {park_label} food storage requirement applies (season/dates/areas) as: '{storage_period_val}'."
    )
    await evaluator.verify(
        claim=claim_storage_period,
        node=storage_period_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 6.3 approved alternatives (non-critical)
    storage_alt_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_approved_alternatives",
        desc="Mentions approved alternatives if the park allows alternatives (or states none).",
        parent=bear_node,
        critical=False,
    )
    storage_alt_val = _value_or_placeholder(details.bear_storage.approved_alternatives if details.bear_storage else None)
    claim_storage_alt = (
        f"Official sources mention the following approved alternatives (if any) for {park_label} food storage: '{storage_alt_val}'."
    )
    await evaluator.verify(
        claim=claim_storage_alt,
        node=storage_alt_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )

    # 7) Group-specific restrictions for 8–10 (critical)
    grp_node = evaluator.add_parallel(
        id=f"{park_node_id}_group_specific_restrictions",
        desc="Group-specific restrictions affecting groups of 8–10",
        parent=park_node,
        critical=True,
    )
    grp_leaf = evaluator.add_leaf(
        id=f"{park_node_id}_restrictions_or_none",
        desc="Identifies additional regulations specifically affecting groups of 8–10 (e.g., required use of group sites, minimum distances between groups, site-specific limitations), or explicitly states no additional group-specific restrictions beyond the general rules.",
        parent=grp_node,
        critical=True,
    )
    grp_val = _value_or_placeholder(
        details.group_specific_restrictions.restrictions_or_none if details.group_specific_restrictions else None
    )
    claim_grp = (
        f"Official sources identify the following additional regulations (if any) specifically affecting groups of 8–10 in {park_label}: '{grp_val}'."
    )
    await evaluator.verify(
        claim=claim_grp,
        node=grp_leaf,
        sources=sources,
        additional_instruction=_park_common_additional_instruction(park_label),
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    # Initialize
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

    # Extract coverage and per-park details in parallel
    coverage_task = evaluator.extract(
        prompt=prompt_extract_coverage(),
        template_class=ParksCoverageExtraction,
        extraction_name="parks_coverage",
    )

    # Prepare per-park detail extraction tasks
    detail_tasks = []
    for park_id, park_label in PARKS:
        task = evaluator.extract(
            prompt=prompt_extract_park_details(park_label),
            template_class=ParkDetails,
            extraction_name=f"{park_id}_details",
        )
        detail_tasks.append(task)

    coverage_result, *park_details_list = await asyncio.gather(coverage_task, *detail_tasks)
    park_details_map = {pid: details for (pid, _), details in zip(PARKS, park_details_list)}

    # Add ground truth info (the canonical park list)
    evaluator.add_ground_truth({"expected_parks": list(EXPECTED_PARK_SET)}, gt_type="expected_park_set")

    # Coverage check (critical)
    provided_set = set(coverage_result.parks_covered or [])
    extra_set = set(coverage_result.extra_parks or [])
    coverage_ok = (provided_set == EXPECTED_PARK_SET) and (len(extra_set) == 0)
    evaluator.add_custom_node(
        result=coverage_ok,
        id="coverage_exact_parks",
        desc="Covers exactly the 5 specified parks (Rocky Mountain, Yosemite, Grand Canyon, Sequoia & Kings Canyon, Olympic) with no missing parks and no extra parks.",
        parent=root,
        critical=True,
    )

    # Add custom info for transparency
    evaluator.add_custom_info(
        {
            "parks_covered": list(provided_set),
            "extra_parks_detected": list(extra_set),
        },
        info_type="coverage_check",
    )

    # Build and verify each park subtree
    for park_id, park_label in PARKS:
        details = park_details_map.get(park_id, ParkDetails())
        await verify_park(
            evaluator=evaluator,
            root_node=root,
            park_node_id=park_id,
            park_label=park_label,
            details=details,
        )

    # Return summary
    return evaluator.get_summary()