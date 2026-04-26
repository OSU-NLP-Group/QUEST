import asyncio
import logging
import math
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "glacier_wilderness_trip_guide_2026"
TASK_DESCRIPTION = (
    "A family group of 7 people (including 2 children ages 12 and 14, and 1 senior citizen age 65 who is a US resident) "
    "is planning a 4-night wilderness camping trip to Glacier National Park in July 2026. They are all US residents and "
    "this will be their only national park visit of the year.\n\n"
    "Provide a comprehensive trip planning guide that includes:\n"
    "1. All wilderness permit requirements, including fees, party size limits, and pickup procedures\n"
    "2. All entrance fee options and the most cost-effective pass strategy for this group\n"
    "3. Mandatory equipment requirements for wilderness camping at Glacier\n"
    "4. Timing constraints for permit acquisition and pickup\n"
    "5. Any other applicable regulations\n\n"
    "For each requirement, provide the specific details (fees, limits, procedures) and include supporting reference URLs from official sources."
)

# Group / trip facts (used for calculations and internal checks)
GROUP_SIZE = 7
NUM_CHILDREN_UNDER_15 = 2  # ages 12 and 14
NUM_SENIORS_62_PLUS = 1    # one senior aged 65
NUM_NON_SENIOR_ADULTS = GROUP_SIZE - NUM_CHILDREN_UNDER_15 - NUM_SENIORS_62_PLUS  # 4
NIGHTS = 4

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ClaimWithSources(BaseModel):
    statement: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class FeeBreakdown(BaseModel):
    base_fee: Optional[str] = None                   # e.g., "$10" or "10 dollars"
    per_person_per_night_fee: Optional[str] = None   # e.g., "$7" or "7"
    urls: List[str] = Field(default_factory=list)


class EntrancePassPrice(BaseModel):
    price: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class EntranceStrategyInfo(BaseModel):
    strategy: Optional[str] = None                       # The recommended pass/fee strategy text
    total_cost: Optional[str] = None                     # Reported total entrance cost for the group
    breakdown: Optional[str] = None                      # If the answer includes a cost breakdown
    alternatives: List[str] = Field(default_factory=list)  # Alternatives mentioned (titles/short descriptions)
    justification: Optional[str] = None                  # Rationale comparing costs
    urls: List[str] = Field(default_factory=list)        # Any URLs cited for entrance strategy (e.g., NPS passes page)


class ExtractedGlacierGuide(BaseModel):
    # Wilderness permit requirements
    permit_required: Optional[ClaimWithSources] = None
    permit_fees: Optional[FeeBreakdown] = None
    permit_total_reported: Optional[str] = None
    max_party_size: Optional[ClaimWithSources] = None
    max_people_per_campsite: Optional[ClaimWithSources] = None
    min_campsites_needed_reported: Optional[str] = None
    pickup_in_person: Optional[ClaimWithSources] = None

    # Timing constraints
    timing_walkup: Optional[ClaimWithSources] = None
    timing_pickup_windows: Optional[ClaimWithSources] = None
    timing_no_after_430: Optional[ClaimWithSources] = None

    # Entrance fees / pass strategy
    annual_pass: Optional[EntrancePassPrice] = None
    senior_annual: Optional[EntrancePassPrice] = None
    senior_lifetime: Optional[EntrancePassPrice] = None
    children_free: Optional[ClaimWithSources] = None
    entrance_strategy: Optional[EntranceStrategyInfo] = None

    # Mandatory equipment
    igbc_requirement: Optional[ClaimWithSources] = None
    show_container: Optional[ClaimWithSources] = None
    storage_distance: Optional[ClaimWithSources] = None

    # Other applicable regulations
    permit_validity_scope: Optional[ClaimWithSources] = None
    max_stay_limit: Optional[ClaimWithSources] = None
    additional_regulation: Optional[ClaimWithSources] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_glacier_trip() -> str:
    return """
Extract the specific, explicitly stated items and supporting URLs from the answer for the Glacier National Park 4-night wilderness camping guide. Return exactly and only what is present in the answer. For each item that requires URLs, include every cited URL relevant to that item.

You must extract the following fields (return null or empty array when missing):

1) Wilderness permit requirements:
- permit_required.statement: The exact statement that a wilderness permit is required for all overnight backcountry camping in Glacier.
- permit_required.urls: All URLs the answer cites for that requirement.
- permit_fees.base_fee: The base permit fee for summer season (May 1–Oct 31) as stated in the answer (e.g., "$10").
- permit_fees.per_person_per_night_fee: The per-person-per-night fee for summer season as stated (e.g., "$7").
- permit_fees.urls: All URLs the answer cites for the fee rules.
- permit_total_reported: The answer’s calculated total wilderness permit fee for 7 people over 4 nights (if the answer provides a numeric total).
- max_party_size.statement: The stated maximum party size (e.g., "12 people").
- max_party_size.urls: All URLs cited for max party size.
- max_people_per_campsite.statement: The stated max people per campsite (e.g., "4 people per campsite").
- max_people_per_campsite.urls: All URLs cited for campsite capacity.
- min_campsites_needed_reported: The answer’s stated minimum number of campsites needed for a 7-person group given the per-site limit (if provided).
- pickup_in_person.statement: The stated rule that permits must be picked up in person at a wilderness center (or equivalent phrasing).
- pickup_in_person.urls: All URLs cited for pickup procedure.

2) Timing constraints:
- timing_walkup.statement: The exact statement about walk-up permits availability (day before or day of the trip) and approximate share held for walk-up (e.g., ~30%).
- timing_walkup.urls: All URLs cited for walk-up details.
- timing_pickup_windows.statement: The exact statement of pickup windows (e.g., day before 8 AM–5 PM, same day 8 AM–11 AM).
- timing_pickup_windows.urls: All URLs cited for pickup window times.
- timing_no_after_430.statement: The exact statement that permits are not issued after 4:30 PM at any location.
- timing_no_after_430.urls: All URLs cited for that cutoff rule.

3) Entrance fees / pass strategy:
- annual_pass.price: The stated price of the America the Beautiful Annual Pass (e.g., "$80").
- annual_pass.urls: All URLs cited for that price/eligibility.
- senior_annual.price: The stated price of the Senior Annual Pass (e.g., "$20").
- senior_annual.urls: All URLs cited for that price/eligibility.
- senior_lifetime.price: The stated price of the Senior Lifetime Pass (e.g., "$80").
- senior_lifetime.urls: All URLs cited for that price/eligibility.
- children_free.statement: The statement that children aged 15 and under are free for national park admission.
- children_free.urls: All URLs cited for that rule.
- entrance_strategy.strategy: The recommended entrance fee/pass strategy text as stated in the answer.
- entrance_strategy.total_cost: The reported total entrance cost for this 7-person group (if present).
- entrance_strategy.breakdown: Any cost breakdown text used by the answer (if present).
- entrance_strategy.alternatives: A list of at least one alternative option title/description mentioned in the answer (if any).
- entrance_strategy.justification: The text the answer uses to justify cost-effectiveness by comparing the recommended strategy against an alternative (if present).
- entrance_strategy.urls: Any URLs the answer cites about entrance passes (if present).

4) Mandatory equipment:
- igbc_requirement.statement: The statement that an IGBC-approved bear-resistant container is required for undesignated camping in the Nyack/Coal Creek zone.
- igbc_requirement.urls: All URLs cited for that requirement.
- show_container.statement: The statement that the container must be shown to a ranger before an undesignated permit is issued (when applicable).
- show_container.urls: All URLs cited for that rule.
- storage_distance.statement: The statement that the bear container must be secured/stored at least 100 feet from the campsite (when applicable).
- storage_distance.urls: All URLs cited for that distance requirement.

5) Other applicable regulations:
- permit_validity_scope.statement: The statement that the wilderness permit is valid only for specified dates, locations, and party size.
- permit_validity_scope.urls: All URLs cited for that validity scope.
- max_stay_limit.statement: The statement that the maximum backcountry stay is 14 nights during the July–August period.
- max_stay_limit.urls: All URLs cited for that limit.
- additional_regulation.statement: At least one OTHER regulation relevant to wilderness camping at Glacier (e.g., food storage rules, campfires, human waste, pets)—beyond the items above.
- additional_regulation.urls: All URLs cited for that additional regulation.

SPECIAL RULES FOR URL EXTRACTION:
- Extract only URLs explicitly present in the answer text. Include URLs from markdown links. If a URL lacks a protocol, prepend http://.
- Do not invent URLs. If none are provided by the answer for a particular item, return an empty list for that item’s urls.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _sanitize_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    cleaned = []
    seen = set()
    for u in urls:
        if not u:
            continue
        u = u.strip()
        if not u:
            continue
        if not (u.startswith("http://") or u.startswith("https://")):
            u = "http://" + u
        if u not in seen:
            seen.add(u)
            cleaned.append(u)
    return cleaned


def _official_source_instruction() -> str:
    return (
        "Important: The claim must be supported by an official source. "
        "Accept only if the provided URL is an official U.S. National Park Service or U.S. government source "
        "(e.g., nps.gov, recreation.gov, doi.gov, usgs.gov) or an official Glacier NP page. "
        "If the URL is not official or the webpage does not clearly support the claim, judge as NOT SUPPORTED. "
        "Allow minor phrasing variations (e.g., '8am-5pm' vs '8 AM–5 PM')."
    )


def _parse_first_number(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", value.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_first_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    m = re.search(r"(\d+)", value.replace(",", ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


async def _verify_claim_with_sources(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    urls: Optional[List[str]],
    critical: bool = True,
    additional_instruction: Optional[str] = None,
) -> None:
    """
    Create a leaf and verify the claim against provided URLs. If URLs are missing, auto-fail the leaf.
    """
    srcs = _sanitize_urls(urls)
    if len(srcs) == 0:
        evaluator.add_custom_node(
            result=False,
            id=node_id,
            desc=f"{desc} (failed: no supporting URLs provided in the answer)",
            parent=parent,
            critical=critical
        )
        return

    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=srcs,
        additional_instruction=additional_instruction or _official_source_instruction(),
    )


async def _verify_simple_claim(
    evaluator: Evaluator,
    *,
    parent,
    node_id: str,
    desc: str,
    claim: str,
    critical: bool = True,
    additional_instruction: Optional[str] = None,
) -> None:
    leaf = evaluator.add_leaf(
        id=node_id,
        desc=desc,
        parent=parent,
        critical=critical
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=None,
        additional_instruction=additional_instruction or "None"
    )


# --------------------------------------------------------------------------- #
# Section verifications                                                       #
# --------------------------------------------------------------------------- #
async def verify_wilderness_permit_requirements(
    evaluator: Evaluator,
    parent,
    data: ExtractedGlacierGuide
) -> None:
    """
    Wilderness permit requirements section.
    NOTE: To allow non-critical leaves under this section (per rubric), this section node is configured as non-critical
    in the tree construction, so these leaves may include a mix of critical and non-critical checks.
    """
    # 1) Permit required for overnight (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="permit_required_for_overnight_with_citation",
        desc="State that a wilderness permit is required for all overnight backcountry camping at Glacier National Park AND include an official source URL.",
        claim="A wilderness permit is required for all overnight backcountry camping in Glacier National Park.",
        urls=(data.permit_required.urls if data.permit_required else []),
        critical=True
    )

    # 2) Permit fee rules (with citation): $10 base + $7 per person per night, May 1–Oct 31
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="permit_fee_rules_with_citation",
        desc="State the summer permit fee rules (May 1–Oct 31): $10 base fee + $7 per person per night AND include an official source URL.",
        claim="During the summer season (May 1–October 31), the backcountry permit fee is $10 per permit plus $7 per person per night.",
        urls=(data.permit_fees.urls if data.permit_fees else []),
        critical=True
    )

    # 3) Permit fee total for this trip (non-critical) -> pure computation check
    # Attempt to parse the numbers from the answer
    base = _parse_first_number(data.permit_fees.base_fee if data.permit_fees else None)
    pppn = _parse_first_number(data.permit_fees.per_person_per_night_fee if data.permit_fees else None)
    reported_total = _parse_first_number(data.permit_total_reported)

    result = False
    expected_total = None
    if base is not None and pppn is not None and reported_total is not None:
        expected_total = base + pppn * GROUP_SIZE * NIGHTS
        # Consider rounding to nearest whole dollar; accept small rounding noise
        result = abs(reported_total - expected_total) <= 1.0

    evaluator.add_custom_node(
        result=result,
        id="permit_fee_total_for_this_trip",
        desc=(
            f"Correctly calculate total wilderness permit fee for 7 people over 4 nights: "
            f"base_fee({base}) + per_person_per_night({pppn}) × 7 × 4 = {expected_total}. "
            f"Answer reported total = {reported_total}."
        ),
        parent=parent,
        critical=False  # per rubric, non-critical
    )

    # 4) Max party size 12 (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="max_party_size_with_citation",
        desc="State the maximum party size is 12 people AND include an official source URL.",
        claim="The maximum party size allowed is 12 people.",
        urls=(data.max_party_size.urls if data.max_party_size else []),
        critical=True
    )

    # 5) Max people per campsite 4 (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="max_people_per_campsite_with_citation",
        desc="State that each campsite is limited to 4 people maximum AND include an official source URL.",
        claim="Each individual backcountry campsite is limited to a maximum of 4 people.",
        urls=(data.max_people_per_campsite.urls if data.max_people_per_campsite else []),
        critical=True
    )

    # 6) Minimum campsites needed for 7 people given 4 per site (non-critical) => ceil(7/4)=2
    reported_min_sites = _parse_first_int(data.min_campsites_needed_reported)
    expected_sites = math.ceil(GROUP_SIZE / 4)
    evaluator.add_custom_node(
        result=(reported_min_sites == expected_sites),
        id="min_campsites_needed_for_7",
        desc=f"Minimum number of campsites for 7 people with 4-person cap per site is {expected_sites}. Answer reported: {reported_min_sites}.",
        parent=parent,
        critical=False  # per rubric, non-critical
    )

    # 7) Pickup in person at wilderness center (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="pickup_in_person_requirement_with_citation",
        desc="State that the permit must be picked up in person at a wilderness center AND include an official source URL.",
        claim="Backcountry permits must be picked up in person at a designated wilderness/permit center.",
        urls=(data.pickup_in_person.urls if data.pickup_in_person else []),
        critical=True
    )


async def verify_timing_constraints(
    evaluator: Evaluator,
    parent,
    data: ExtractedGlacierGuide
) -> None:
    # Walk-up permits: day before or day-of; approx 30% held for walk-up (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="walkup_permit_availability_and_share_with_citation",
        desc="State that walk-up permits are available the day before or day of the trip and ~30% of sites are held for walk-up AND include an official source URL.",
        claim="Walk-up backcountry permits at Glacier are available the day before or the day of the trip, and approximately 30% of sites are held for walk-up distribution.",
        urls=(data.timing_walkup.urls if data.timing_walkup else []),
        critical=True
    )

    # Pickup time windows: day before 8 AM–5 PM; same day 8 AM–11 AM (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="pickup_time_windows_with_citation",
        desc="State the pickup windows: day before (8 AM–5 PM) or same day (8 AM–11 AM) AND include an official source URL.",
        claim="Permit pickup windows are 8 AM–5 PM on the day before the trip or 8 AM–11 AM on the day of the trip.",
        urls=(data.timing_pickup_windows.urls if data.timing_pickup_windows else []),
        critical=True
    )

    # No issuance after 4:30 PM (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="no_issuance_after_430pm_with_citation",
        desc="State that permits are not issued after 4:30 PM at any location AND include an official source URL.",
        claim="Permits are not issued after 4:30 PM at any location.",
        urls=(data.timing_no_after_430.urls if data.timing_no_after_430 else []),
        critical=True
    )


async def verify_entrance_fees_and_pass_strategy(
    evaluator: Evaluator,
    parent,
    data: ExtractedGlacierGuide
) -> None:
    # Annual Pass (America the Beautiful) $80 with citation
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="annual_pass_price_eligibility_with_citation",
        desc="State the America the Beautiful Annual Pass price is $80 for US citizens/residents AND include an official source URL.",
        claim="The America the Beautiful Annual Pass costs $80 for U.S. citizens or residents.",
        urls=(data.annual_pass.urls if data.annual_pass else []),
        critical=True
    )

    # Senior Annual Pass $20 with citation
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="senior_annual_pass_price_eligibility_with_citation",
        desc="State the Senior Annual Pass price is $20 for US citizens/residents age 62+ AND include an official source URL.",
        claim="The Senior Annual Pass costs $20 for U.S. citizens or permanent residents age 62 and older.",
        urls=(data.senior_annual.urls if data.senior_annual else []),
        critical=True
    )

    # Senior Lifetime Pass $80 with citation
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="senior_lifetime_pass_price_eligibility_with_citation",
        desc="State the Senior Lifetime Pass price is $80 for US citizens/residents age 62+ AND include an official source URL.",
        claim="The Senior Lifetime Pass costs $80 for U.S. citizens or permanent residents age 62 and older.",
        urls=(data.senior_lifetime.urls if data.senior_lifetime else []),
        critical=True
    )

    # Children free (15 and under) with citation
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="children_free_admission_with_citation",
        desc="State that children age 15 and under are free admission to national parks AND include an official source URL.",
        claim="Children aged 15 and under are admitted free to U.S. national parks.",
        urls=(data.children_free.urls if data.children_free else []),
        critical=True
    )

    # Use correct group composition in the entrance-fee reasoning (simple verify)
    await _verify_simple_claim(
        evaluator,
        parent=parent,
        node_id="use_correct_group_composition",
        desc="Use the given group composition correctly (7 total; 2 children under 15; 1 senior age 65; 4 non-senior adults) in entrance-fee reasoning.",
        claim=(
            "Within the entrance-fee reasoning in the answer, the group composition is used correctly: "
            "7 total people; 2 children under 15 (ages 12 and 14) treated as free; "
            "1 senior age 65 eligible for Senior Pass; remaining 4 are non-senior adults."
        ),
        critical=True,
        additional_instruction=(
            "Judge based on the answer text. The answer should explicitly or implicitly reflect this composition in its cost reasoning."
        )
    )

    # Recommended pass strategy (simple verify but fail if absent)
    rec = (data.entrance_strategy.strategy if data.entrance_strategy else None)
    if rec and rec.strip():
        await _verify_simple_claim(
            evaluator,
            parent=parent,
            node_id="recommend_pass_strategy",
            desc="State a concrete recommended pass strategy for the group for their single national-park visit.",
            claim=f"The answer provides a concrete recommended entrance pass strategy tailored to this group: '{rec}'.",
            critical=True,
            additional_instruction=(
                "Assess whether this is a concrete, actionable recommendation (e.g., which pass(es) to buy) rather than a vague description."
            )
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="recommend_pass_strategy",
            desc="Recommended entrance pass strategy is missing or not stated concretely in the answer.",
            parent=parent,
            critical=True
        )

    # Entrance cost calculation consistent with recommendation and stated prices (simple verify; require existence)
    total_cost = (data.entrance_strategy.total_cost if data.entrance_strategy else None)
    if rec and rec.strip() and total_cost and total_cost.strip():
        bk = (data.entrance_strategy.breakdown if data.entrance_strategy else None)
        await _verify_simple_claim(
            evaluator,
            parent=parent,
            node_id="entrance_cost_calculation_consistent_with_recommendation",
            desc="Provide a numerically consistent total entrance-cost calculation that matches the recommended strategy and stated pass prices/eligibility.",
            claim=(
                f"The entrance total cost stated as '{total_cost}' is numerically consistent with the recommended strategy "
                f"('{rec}') and with the pass prices/eligibility the answer itself presents. "
                f"Breakdown (if provided): '{bk}'."
            ),
            critical=True,
            additional_instruction=(
                "Judge solely from the answer text. Check internal consistency: the numbers used in the total must align with the stated pass prices "
                "and the group composition (children free; one senior eligible). If inconsistent or missing, judge as incorrect."
            )
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="entrance_cost_calculation_consistent_with_recommendation",
            desc="Entrance total cost calculation is missing or cannot be evaluated due to absent strategy/total in the answer.",
            parent=parent,
            critical=True
        )

    # Cost-effectiveness justification vs at least one alternative (simple verify; require existence)
    alts = (data.entrance_strategy.alternatives if data.entrance_strategy else [])
    just = (data.entrance_strategy.justification if data.entrance_strategy else None)
    if alts and len(alts) > 0 and just and just.strip():
        await _verify_simple_claim(
            evaluator,
            parent=parent,
            node_id="cost_effectiveness_justification",
            desc="Justify cost-effectiveness by comparing the recommendation against at least one alternative; recommended total is not higher.",
            claim=(
                f"The answer compares the recommended strategy ('{rec}') against at least one alternative ({alts[0] if alts else ''}) "
                f"and provides justification showing the recommended total ('{total_cost}') is not higher based on prices/fees presented."
            ),
            critical=True,
            additional_instruction=(
                "Judge from the answer text: There should be an explicit comparison to at least one alternative option and a clear justification "
                "that the recommended strategy is as cheap or cheaper than the alternative(s) using the prices cited."
            )
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="cost_effectiveness_justification",
            desc="Cost-effectiveness comparison/justification is missing or lacks at least one alternative in the answer.",
            parent=parent,
            critical=True
        )


async def verify_mandatory_equipment_requirements(
    evaluator: Evaluator,
    parent,
    data: ExtractedGlacierGuide
) -> None:
    # IGBC-required in Nyack/Coal Creek undesignated (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="igbc_container_requirement_with_citation",
        desc="State that an IGBC-approved bear-resistant container is required for undesignated camping in the Nyack/Coal Creek zone AND include an official source URL.",
        claim="An IGBC-approved bear-resistant container is required for undesignated camping in the Nyack/Coal Creek zone of Glacier National Park.",
        urls=(data.igbc_requirement.urls if data.igbc_requirement else []),
        critical=True
    )

    # Container must be shown to ranger before undesignated permit is issued (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="container_must_be_shown_to_ranger_with_citation",
        desc="State that the bear-resistant container must be shown to a ranger before an undesignated permit is issued (when applicable) AND include an official source URL.",
        claim="Before an undesignated backcountry permit is issued (where applicable), the bear-resistant container must be shown to a ranger.",
        urls=(data.show_container.urls if data.show_container else []),
        critical=True
    )

    # Storage distance at least 100 feet (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="container_storage_distance_rule_with_citation",
        desc="State that the bear container must be secured at least 100 feet from the campsite (when applicable) AND include an official source URL.",
        claim="Bear-resistant containers must be stored or secured at least 100 feet from the campsite (when applicable).",
        urls=(data.storage_distance.urls if data.storage_distance else []),
        critical=True
    )


async def verify_other_applicable_regulations(
    evaluator: Evaluator,
    parent,
    data: ExtractedGlacierGuide
) -> None:
    # Permit validity scope (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="permit_validity_scope_with_citation",
        desc="State that the wilderness permit is valid only for specified dates, locations, and party size AND include an official source URL.",
        claim="A Glacier backcountry wilderness permit is valid only for the specified dates, locations (itinerary), and party size listed on the permit.",
        urls=(data.permit_validity_scope.urls if data.permit_validity_scope else []),
        critical=True
    )

    # Max backcountry stay limit: 14 nights in July–August (with citation)
    await _verify_claim_with_sources(
        evaluator,
        parent=parent,
        node_id="max_backcountry_stay_limit_with_citation",
        desc="State the maximum backcountry stay limit is 14 nights during the July–August period AND include an official source URL.",
        claim="The maximum backcountry stay during the July–August period is 14 nights.",
        urls=(data.max_stay_limit.urls if data.max_stay_limit else []),
        critical=True
    )

    # At least one additional regulation beyond the others (with citation)
    # Use the extracted additional_regulation statement and URLs
    add_stmt = data.additional_regulation.statement if data.additional_regulation else None
    add_urls = data.additional_regulation.urls if data.additional_regulation else []
    if add_stmt and add_stmt.strip():
        await _verify_claim_with_sources(
            evaluator,
            parent=parent,
            node_id="at_least_one_additional_regulation_with_citation",
            desc="Include at least one additional regulation relevant to wilderness camping at Glacier NP beyond permits/fees/timing/equipment AND include an official source URL.",
            claim=f"Additional regulation: {add_stmt}",
            urls=add_urls,
            critical=True
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="at_least_one_additional_regulation_with_citation",
            desc="Additional regulation is missing or has no supporting URL in the answer.",
            parent=parent,
            critical=True
        )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
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
    Evaluate an answer for the Glacier NP 4-night wilderness camping planning guide.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates top-level sections in parallel
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

    # Extract structured info
    extracted: ExtractedGlacierGuide = await evaluator.extract(
        prompt=prompt_extract_glacier_trip(),
        template_class=ExtractedGlacierGuide,
        extraction_name="glacier_trip_extraction"
    )

    # Add ground truth / context info (for summary)
    evaluator.add_custom_info(
        info={
            "group_size": GROUP_SIZE,
            "children_under_15": NUM_CHILDREN_UNDER_15,
            "seniors_62_plus": NUM_SENIORS_62_PLUS,
            "non_senior_adults": NUM_NON_SENIOR_ADULTS,
            "nights": NIGHTS,
            "trip_month_year": "July 2026",
        },
        info_type="group_trip_context",
    )

    # Build top-level sections
    # Important: The 'wilderness_permit_requirements' section contains a couple of non-critical leaves per rubric.
    # To allow mixed criticality children, we set this parent as non-critical (framework requires all children of a critical node be critical).
    permit_node = evaluator.add_parallel(
        id="wilderness_permit_requirements",
        desc="State wilderness permit requirements (requirement, fees, party size limits, pickup procedure), with official citation(s).",
        parent=root,
        critical=False
    )

    timing_node = evaluator.add_parallel(
        id="timing_constraints_for_acquisition_and_pickup",
        desc="State timing constraints for permit acquisition and pickup with official citation(s).",
        parent=root,
        critical=True
    )

    entrance_node = evaluator.add_parallel(
        id="entrance_fees_and_pass_strategy",
        desc="Provide entrance fee/pass options and the most cost-effective strategy with official citation(s).",
        parent=root,
        critical=True
    )

    equipment_node = evaluator.add_parallel(
        id="mandatory_equipment_requirements",
        desc="State mandatory equipment requirements (e.g., IGBC bear container) with official citation(s).",
        parent=root,
        critical=True
    )

    other_regs_node = evaluator.add_parallel(
        id="other_applicable_regulations",
        desc="State other applicable regulations with official citation(s).",
        parent=root,
        critical=True
    )

    # Perform verifications per section
    await verify_wilderness_permit_requirements(evaluator, permit_node, extracted)
    await verify_timing_constraints(evaluator, timing_node, extracted)
    await verify_entrance_fees_and_pass_strategy(evaluator, entrance_node, extracted)
    await verify_mandatory_equipment_requirements(evaluator, equipment_node, extracted)
    await verify_other_applicable_regulations(evaluator, other_regs_node, extracted)

    # Return structured result
    return evaluator.get_summary()