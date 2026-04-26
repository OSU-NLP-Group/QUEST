import asyncio
import logging
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# =============================================================================
# Task constants
# =============================================================================
TASK_ID = "sierra_nevada_reg_compliance"
TASK_DESCRIPTION = (
    "You are planning a 5-day backpacking trip in California's Sierra Nevada wilderness areas, "
    "specifically entering via the Kearsarge Pass trailhead at Onion Valley and hiking through the Rae Lakes Loop area "
    "in Sequoia-Kings Canyon National Parks. Your group consists of 8 people, and you plan to depart in mid-July 2026. "
    "For this trip, you need to document comprehensive regulatory compliance information. Provide detailed information "
    "for each of the following requirements: (1) The wilderness entry trailhead name, managing agency, and daily trailhead "
    "quota system; (2) The complete permit reservation fee structure including both the non-refundable reservation fee and "
    "per-person fees; (3) The permit reservation advance booking window showing when permits become available relative to the entry date; "
    "(4) Bear canister requirements specifying which areas require them and the seasonal applicability; "
    "(5) Bear canister placement regulations including minimum distance from campsites; "
    "(6) Camping distance requirements from water sources including the minimum prohibited distance and the preferred distance; "
    "(7) Designated campsite requirements for the Paradise Valley area including night limits; "
    "(8) Night stay limits for the individual Rae Lakes and Charlotte Lake; "
    "(9) Maximum group size limits applicable to this area for both on-trail and off-trail travel; "
    "(10) Campfire restriction information for nearby wilderness areas including specific water bodies with proximity-based bans; "
    "(11) Cathole requirements including depth, distance from water/camp/trails, and disposal procedures; "
    "(12) Water treatment requirements and procedures for washing away from water sources; "
    "(13) Permit possession and presentation requirements during the trip. Each piece of information must be accurate, "
    "specific to the stated wilderness areas, and include proper reference URLs from official land management agency sources."
)

ALLOWED_OFFICIAL_DOMAINS = {"nps.gov", "fs.usda.gov", "recreation.gov"}


# =============================================================================
# Data Models for Extraction
# =============================================================================
class EntryTrailheadQuota(BaseModel):
    trailhead_name: Optional[str] = None
    managing_agency: Optional[str] = None
    daily_quota_number: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PermitFeeStructure(BaseModel):
    reservation_fee: Optional[str] = None
    per_person_fee: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PermitAdvanceWindow(BaseModel):
    release_description: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BearCanisterRequirements(BaseModel):
    areas_required_description: Optional[str] = None  # which areas require canisters and mention Rae Lakes applicability
    seasonal_applicability: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class BearCanisterPlacement(BaseModel):
    min_distance_from_camp_feet: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CampingDistanceFromWater(BaseModel):
    min_prohibited_distance_feet: Optional[str] = None  # must prohibit camping within 25 feet
    preferred_distance_feet: Optional[str] = None       # must state 100 feet preferred
    well_established_site_rule: Optional[str] = None    # mention the well-established-site rule regarding the 100-foot distance
    sources: List[str] = Field(default_factory=list)


class ParadiseValleyDesignatedSites(BaseModel):
    designated_sites_required: Optional[str] = None  # should indicate designated-only camping in Paradise Valley
    night_limit: Optional[str] = None                # number of nights allowed in Paradise Valley
    sources: List[str] = Field(default_factory=list)


class RaeLakesCharlotteNightLimits(BaseModel):
    rae_lakes_night_limit: Optional[str] = None      # one night per individual Rae Lake (Lower, Middle, Upper)
    charlotte_lake_night_limit: Optional[str] = None # two-night limit
    sources: List[str] = Field(default_factory=list)


class GroupSizeLimits(BaseModel):
    on_trail_limit: Optional[str] = None  # must be 15
    off_trail_limit: Optional[str] = None # must be 12
    sources: List[str] = Field(default_factory=list)


class CampfireRestrictionsAnselAdams(BaseModel):
    named_water_bodies: List[str] = Field(default_factory=list)  # at least one named water body covered by the ban
    proximity_ban_distance: Optional[str] = None                 # must be 1/4 mile from lake outlets
    sources: List[str] = Field(default_factory=list)


class CatholeRequirements(BaseModel):
    depth_inches: Optional[str] = None                            # must be 6–8 inches
    distance_from_water_camp_trails_feet: Optional[str] = None    # must be at least 200 feet
    disposal_procedures: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WaterTreatmentAndWashing(BaseModel):
    treat_all_drinking_water: Optional[str] = None
    washing_distance_feet: Optional[str] = None                   # must be 200 feet
    washing_procedure: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PermitPossessionPresentation(BaseModel):
    must_carry_permit: Optional[str] = None
    must_present_to_rangers: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ComplianceExtraction(BaseModel):
    entry_trailhead_quota: Optional[EntryTrailheadQuota] = None
    permit_fee_structure: Optional[PermitFeeStructure] = None
    permit_advance_booking_window: Optional[PermitAdvanceWindow] = None
    bear_canister_requirements: Optional[BearCanisterRequirements] = None
    bear_canister_placement_distance: Optional[BearCanisterPlacement] = None
    camping_distance_from_water: Optional[CampingDistanceFromWater] = None
    paradise_valley_designated_sites_and_night_limit: Optional[ParadiseValleyDesignatedSites] = None
    rae_lakes_and_charlotte_night_limits: Optional[RaeLakesCharlotteNightLimits] = None
    group_size_limits: Optional[GroupSizeLimits] = None
    campfire_restrictions_ansel_adams_lakes: Optional[CampfireRestrictionsAnselAdams] = None
    cathole_requirements: Optional[CatholeRequirements] = None
    water_treatment_and_washing_distance: Optional[WaterTreatmentAndWashing] = None
    permit_possession_and_presentation: Optional[PermitPossessionPresentation] = None


# =============================================================================
# Extraction Prompt
# =============================================================================
def prompt_extract_compliance() -> str:
    return """
Extract, from the provided answer, structured compliance information specifically for:
- The Kearsarge Pass trailhead at Onion Valley entry (Inyo National Forest) and the Rae Lakes Loop area in Sequoia-Kings Canyon National Parks (SEKI).
- Only extract what is explicitly stated in the answer. Do not infer or invent.
- Always extract the supporting official URLs cited in the answer (NPS, USFS, Recreation.gov). If none are provided, return an empty array for sources.

Return a JSON object of type ComplianceExtraction with these fields:

1) entry_trailhead_quota:
   - trailhead_name: The entry trailhead name (e.g., "Kearsarge Pass (Onion Valley)").
   - managing_agency: The managing agency (e.g., "Inyo National Forest" or similar).
   - daily_quota_number: The daily entry quota number for this trailhead (as written).
   - sources: Array of official URLs from the answer that support this info.

2) permit_fee_structure:
   - reservation_fee: The non-refundable reservation fee text/amount (as written).
   - per_person_fee: The per-person fee text/amount (as written).
   - sources: URLs.

3) permit_advance_booking_window:
   - release_description: The description of the advance booking window and any split/portion release scheme (as written).
   - sources: URLs.

4) bear_canister_requirements:
   - areas_required_description: Text describing where canisters are required, explicitly covering applicability to Rae Lakes area.
   - seasonal_applicability: Text describing any seasonal date applicability (as written).
   - sources: URLs.

5) bear_canister_placement_distance:
   - min_distance_from_camp_feet: The minimum distance canisters must be placed from campsites (as written).
   - sources: URLs.

6) camping_distance_from_water:
   - min_prohibited_distance_feet: The minimum prohibited distance from water sources (as written).
   - preferred_distance_feet: The preferred distance from water sources (as written).
   - well_established_site_rule: The rule text explaining exceptions or the well-established-site aspect regarding the 100-foot preference (as written).
   - sources: URLs.

7) paradise_valley_designated_sites_and_night_limit:
   - designated_sites_required: Text stating camping is only allowed in designated sites in Paradise Valley (as written).
   - night_limit: Night limit in Paradise Valley (as written).
   - sources: URLs.

8) rae_lakes_and_charlotte_night_limits:
   - rae_lakes_night_limit: Night limit for each individual Rae Lake (as written).
   - charlotte_lake_night_limit: Night limit for Charlotte Lake (as written).
   - sources: URLs.

9) group_size_limits:
   - on_trail_limit: On-trail maximum group size (as written).
   - off_trail_limit: Off-trail maximum group size (as written).
   - sources: URLs.

10) campfire_restrictions_ansel_adams_lakes:
   - named_water_bodies: Array of the specific water body names mentioned (e.g., "Thousand Island Lake") covered by the proximity-based ban.
   - proximity_ban_distance: The proximity-based ban distance (as written).
   - sources: URLs.

11) cathole_requirements:
   - depth_inches: Depth range for catholes (as written).
   - distance_from_water_camp_trails_feet: Required distance from water, camp, and trails (as written).
   - disposal_procedures: Disposal procedure for waste and toilet paper (as written).
   - sources: URLs.

12) water_treatment_and_washing_distance:
   - treat_all_drinking_water: Statement indicating all drinking water must be treated (as written).
   - washing_distance_feet: Required distance from water sources for washing (as written).
   - washing_procedure: Procedures for washing and disposing greywater (as written).
   - sources: URLs.

13) permit_possession_and_presentation:
   - must_carry_permit: Statement that permit must be carried/possessed (as written).
   - must_present_to_rangers: Statement that permit must be presented to rangers upon request (as written).
   - sources: URLs.
"""


# =============================================================================
# Helper utilities
# =============================================================================
def listify(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Filter out obvious malformed empty strings
    return [u for u in urls if isinstance(u, str) and u.strip() != ""]


def is_official_url(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        # Strip leading www.
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return any(netloc.endswith(d) for d in ALLOWED_OFFICIAL_DOMAINS)
    except Exception:
        return False


def all_sources_official(sources: List[str]) -> bool:
    if not sources:
        return False
    return all(is_official_url(s) for s in sources)


async def add_official_sources_check(
    evaluator: Evaluator,
    parent_node,
    sources: List[str],
    node_id: str,
    desc: str,
    critical: bool = True
):
    evaluator.add_custom_node(
        result=all_sources_official(sources),
        id=node_id,
        desc=desc,
        parent=parent_node,
        critical=critical
    )


# =============================================================================
# Verification builders for each rubric item
# =============================================================================
async def build_entry_trailhead_quota(
    evaluator: Evaluator,
    root,
    data: Optional[EntryTrailheadQuota]
):
    parent = evaluator.add_parallel(
        id="entry_trailhead_quota",
        desc="Entry trailhead info and daily quota supported by official URL(s)",
        parent=root,
        critical=True
    )

    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.trailhead_name is not None and data.trailhead_name.strip() != "") and
        (data.managing_agency is not None and data.managing_agency.strip() != "") and
        (data.daily_quota_number is not None and data.daily_quota_number.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="entry_trailhead_quota_exists",
        desc="Trailhead name, managing agency, and daily quota number are provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="entry_trailhead_quota_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)",
        critical=True
    )

    # Verify trailhead identity and managing agency
    node_trailhead_agency = evaluator.add_leaf(
        id="entry_trailhead_name_agency_supported",
        desc="Trailhead (Kearsarge Pass via Onion Valley) and managing agency are supported by sources",
        parent=parent,
        critical=True,
    )
    claim_trailhead_agency = (
        f"The cited source(s) describe the Kearsarge Pass (Onion Valley) wilderness entry trailhead and "
        f"identify the managing agency as {data.managing_agency if data else ''}."
    )
    await evaluator.verify(
        claim=claim_trailhead_agency,
        node=node_trailhead_agency,
        sources=sources,
        additional_instruction="Allow minor naming variants like 'Kearsarge Pass (Onion Valley)' or 'Onion Valley to Kearsarge Pass'. Confirm agency name (e.g., Inyo National Forest)."
    )

    # Verify daily quota number
    node_quota = evaluator.add_leaf(
        id="entry_trailhead_quota_number_supported",
        desc="Daily trailhead quota number is supported by sources",
        parent=parent,
        critical=True,
    )
    claim_quota = (
        f"The daily entry quota for the Kearsarge Pass (Onion Valley) trailhead is {data.daily_quota_number if data else ''}."
    )
    await evaluator.verify(
        claim=claim_quota,
        node=node_quota,
        sources=sources,
        additional_instruction="Check the exact daily quota number on the official page. Match numbers even if formatted differently."
    )


async def build_permit_fee_structure(
    evaluator: Evaluator,
    root,
    data: Optional[PermitFeeStructure]
):
    parent = evaluator.add_parallel(
        id="permit_fee_structure",
        desc="Permit fee structure (reservation fee and per-person fee) supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.reservation_fee is not None and data.reservation_fee.strip() != "") and
        (data.per_person_fee is not None and data.per_person_fee.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="permit_fee_structure_exists",
        desc="Reservation fee and per-person fee are provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="permit_fee_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )

    # Reservation fee leaf
    node_res_fee = evaluator.add_leaf(
        id="reservation_fee_supported",
        desc="Reservation fee amount is supported by sources",
        parent=parent,
        critical=True
    )
    claim_res_fee = f"The non-refundable reservation fee is {data.reservation_fee if data else ''}."
    await evaluator.verify(
        claim=claim_res_fee,
        node=node_res_fee,
        sources=sources,
        additional_instruction="Confirm the listed reservation fee on the official permit page (often Recreation.gov)."
    )

    # Per-person fee leaf
    node_pp_fee = evaluator.add_leaf(
        id="per_person_fee_supported",
        desc="Per-person fee amount is supported by sources",
        parent=parent,
        critical=True
    )
    claim_pp_fee = f"The per-person fee is {data.per_person_fee if data else ''}."
    await evaluator.verify(
        claim=claim_pp_fee,
        node=node_pp_fee,
        sources=sources,
        additional_instruction="Confirm the per-person fee as stated on the official permit information page."
    )


async def build_permit_advance_window(
    evaluator: Evaluator,
    root,
    data: Optional[PermitAdvanceWindow]
):
    parent = evaluator.add_parallel(
        id="permit_advance_booking_window",
        desc="Permit advance booking window and quota release scheme supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.release_description is not None and data.release_description.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="advance_window_exists",
        desc="Advance booking window and release description are provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="advance_window_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )
    # Verification leaf
    node_window = evaluator.add_leaf(
        id="advance_window_supported",
        desc="Advance booking window (timing/portion releases) supported by sources",
        parent=parent,
        critical=True
    )
    claim_window = (
        f"The advance booking window and any quota release timing/portion details are as follows: "
        f"{data.release_description if data else ''}"
    )
    await evaluator.verify(
        claim=claim_window,
        node=node_window,
        sources=sources,
        additional_instruction="Verify when permits become available relative to the entry date, and any split release schedule (e.g., a portion 6 months out, remaining 2 weeks out)."
    )


async def build_bear_canister_requirements(
    evaluator: Evaluator,
    root,
    data: Optional[BearCanisterRequirements]
):
    parent = evaluator.add_parallel(
        id="bear_canister_requirements",
        desc="Bear canister requirement coverage (including Rae Lakes area) and seasonality supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.areas_required_description is not None and data.areas_required_description.strip() != "") and
        (data.seasonal_applicability is not None and data.seasonal_applicability.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="canister_reqs_exist",
        desc="Areas requiring canisters and seasonal applicability provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="canister_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )

    # Leaf: Rae Lakes applicability
    node_rae_lakes = evaluator.add_leaf(
        id="canister_applies_rae_lakes",
        desc="Bear canisters applicability to Rae Lakes area is supported by sources",
        parent=parent,
        critical=True
    )
    claim_rae = (
        "The cited sources indicate that bear-resistant food storage (bear canisters or provided bear boxes) is required for travel/camping in the Rae Lakes area."
    )
    await evaluator.verify(
        claim=claim_rae,
        node=node_rae_lakes,
        sources=sources,
        additional_instruction="Allow that some areas have fixed bear boxes; otherwise canisters required. Confirm explicit applicability to Rae Lakes area."
    )

    # Leaf: Seasonality
    node_season = evaluator.add_leaf(
        id="canister_seasonality_supported",
        desc="Seasonal applicability dates/statement supported by sources",
        parent=parent,
        critical=True
    )
    claim_season = f"The seasonal applicability regarding bear canisters is as stated: {data.seasonal_applicability if data else ''}"
    await evaluator.verify(
        claim=claim_season,
        node=node_season,
        sources=sources,
        additional_instruction="Confirm whether the requirement is seasonal or year-round and match any date ranges if provided."
    )


async def build_bear_canister_placement(
    evaluator: Evaluator,
    root,
    data: Optional[BearCanisterPlacement]
):
    parent = evaluator.add_parallel(
        id="bear_canister_placement_distance",
        desc="Bear canister placement minimum distance from camp supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.min_distance_from_camp_feet is not None and data.min_distance_from_camp_feet.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="canister_placement_exists",
        desc="Minimum placement distance from camp provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="canister_placement_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )
    node_place = evaluator.add_leaf(
        id="canister_placement_distance_supported",
        desc="Minimum canister placement distance (>= 100 feet) supported by sources",
        parent=parent,
        critical=True
    )
    claim_place = (
        f"Bear canisters (or stored food) must be placed at least {data.min_distance_from_camp_feet if data else ''} from campsites."
    )
    await evaluator.verify(
        claim=claim_place,
        node=node_place,
        sources=sources,
        additional_instruction="Confirm that the minimum placement distance is at least 100 feet; allow variants like '100 ft' or '100 feet'."
    )


async def build_camping_distance_from_water(
    evaluator: Evaluator,
    root,
    data: Optional[CampingDistanceFromWater]
):
    parent = evaluator.add_parallel(
        id="camping_distance_from_water",
        desc="Camping distance requirements from water supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.min_prohibited_distance_feet is not None and data.min_prohibited_distance_feet.strip() != "") and
        (data.preferred_distance_feet is not None and data.preferred_distance_feet.strip() != "") and
        (data.well_established_site_rule is not None and data.well_established_site_rule.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="camping_distance_water_exists",
        desc="Min prohibited distance, preferred distance, and well-established-site rule provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="camping_distance_water_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )

    # Prohibited within 25 ft
    node_min25 = evaluator.add_leaf(
        id="camping_min_25ft_supported",
        desc="Prohibits camping within 25 feet of water supported by sources",
        parent=parent,
        critical=True
    )
    claim_25 = f"Camping is prohibited within {data.min_prohibited_distance_feet if data else ''} of water sources."
    await evaluator.verify(
        claim=claim_25,
        node=node_min25,
        sources=sources,
        additional_instruction="Confirm that the minimum prohibition includes 25 feet (e.g., 'do not camp within 25 feet of water')."
    )

    # Preferred 100 ft
    node_pref100 = evaluator.add_leaf(
        id="camping_preferred_100ft_supported",
        desc="States preferred camping distance of 100 feet from water supported by sources",
        parent=parent,
        critical=True
    )
    claim_100 = f"The preferred camping distance from water sources is {data.preferred_distance_feet if data else ''}."
    await evaluator.verify(
        claim=claim_100,
        node=node_pref100,
        sources=sources,
        additional_instruction="Confirm '100 feet' preferred distance language; allow '100 ft'."
    )

    # Well-established-site rule
    node_established = evaluator.add_leaf(
        id="camping_well_established_rule_supported",
        desc="Includes well-established-site rule regarding 100 ft preference supported by sources",
        parent=parent,
        critical=True
    )
    claim_established = f"The regulation/guidance includes the well-established-site rule: {data.well_established_site_rule if data else ''}"
    await evaluator.verify(
        claim=claim_established,
        node=node_established,
        sources=sources,
        additional_instruction="Verify that guidance mentions established sites and/or exceptions while still stating the 100-foot preference."
    )


async def build_paradise_valley_rules(
    evaluator: Evaluator,
    root,
    data: Optional[ParadiseValleyDesignatedSites]
):
    parent = evaluator.add_parallel(
        id="paradise_valley_designated_sites_and_night_limit",
        desc="Paradise Valley designated-sites-only and night limit supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.designated_sites_required is not None and data.designated_sites_required.strip() != "") and
        (data.night_limit is not None and data.night_limit.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="paradise_valley_rules_exist",
        desc="Paradise Valley 'designated sites only' and night limit are provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="paradise_valley_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )

    # Designated-only
    node_designated = evaluator.add_leaf(
        id="paradise_valley_designated_supported",
        desc="Paradise Valley requires camping only in designated sites (supported)",
        parent=parent,
        critical=True
    )
    claim_designated = f"Paradise Valley requires camping only in designated sites: {data.designated_sites_required if data else ''}"
    await evaluator.verify(
        claim=claim_designated,
        node=node_designated,
        sources=sources,
        additional_instruction="Verify a rule for Paradise Valley specifying designated-site-only camping."
    )

    # Night limit
    node_pv_night = evaluator.add_leaf(
        id="paradise_valley_night_limit_supported",
        desc="Paradise Valley night limit is supported by sources",
        parent=parent,
        critical=True
    )
    claim_pv_night = f"The night limit for Paradise Valley is {data.night_limit if data else ''}."
    await evaluator.verify(
        claim=claim_pv_night,
        node=node_pv_night,
        sources=sources,
        additional_instruction="Confirm the stated night limit for Paradise Valley."
    )


async def build_rae_charlotte_limits(
    evaluator: Evaluator,
    root,
    data: Optional[RaeLakesCharlotteNightLimits]
):
    parent = evaluator.add_parallel(
        id="rae_lakes_and_charlotte_night_limits",
        desc="Rae Lakes and Charlotte Lake night limits supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.rae_lakes_night_limit is not None and data.rae_lakes_night_limit.strip() != "") and
        (data.charlotte_lake_night_limit is not None and data.charlotte_lake_night_limit.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="rae_charlotte_limits_exist",
        desc="Night limits for Rae Lakes and Charlotte Lake provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="rae_charlotte_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )
    # Rae Lakes
    node_rae = evaluator.add_leaf(
        id="rae_lakes_night_limit_supported",
        desc="Each individual Rae Lake one-night limit supported by sources",
        parent=parent,
        critical=True
    )
    claim_rae = f"The night limit for each individual Rae Lake is {data.rae_lakes_night_limit if data else ''}."
    await evaluator.verify(
        claim=claim_rae,
        node=node_rae,
        sources=sources,
        additional_instruction="Confirm a one-night limit at each of Lower, Middle, and Upper Rae Lakes if stated."
    )
    # Charlotte Lake
    node_char = evaluator.add_leaf(
        id="charlotte_lake_night_limit_supported",
        desc="Charlotte Lake two-night limit supported by sources",
        parent=parent,
        critical=True
    )
    claim_char = f"The night limit for Charlotte Lake is {data.charlotte_lake_night_limit if data else ''}."
    await evaluator.verify(
        claim=claim_char,
        node=node_char,
        sources=sources,
        additional_instruction="Confirm a two-night camping limit for Charlotte Lake if stated."
    )


async def build_group_size_limits(
    evaluator: Evaluator,
    root,
    data: Optional[GroupSizeLimits]
):
    parent = evaluator.add_parallel(
        id="group_size_limits",
        desc="Maximum group sizes (on-trail and off-trail) supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.on_trail_limit is not None and data.on_trail_limit.strip() != "") and
        (data.off_trail_limit is not None and data.off_trail_limit.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="group_size_limits_exist",
        desc="On-trail and off-trail limits are provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="group_size_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )
    # On-trail
    node_on = evaluator.add_leaf(
        id="group_size_on_trail_supported",
        desc="On-trail maximum group size is supported by sources",
        parent=parent,
        critical=True
    )
    claim_on = f"The maximum on-trail group size in this area is {data.on_trail_limit if data else ''}."
    await evaluator.verify(
        claim=claim_on,
        node=node_on,
        sources=sources,
        additional_instruction="Confirm the on-trail limit (must be 15 per rubric)."
    )
    # Off-trail
    node_off = evaluator.add_leaf(
        id="group_size_off_trail_supported",
        desc="Off-trail maximum group size is supported by sources",
        parent=parent,
        critical=True
    )
    claim_off = f"The maximum off-trail group size in this area is {data.off_trail_limit if data else ''}."
    await evaluator.verify(
        claim=claim_off,
        node=node_off,
        sources=sources,
        additional_instruction="Confirm the off-trail limit (must be 12 per rubric)."
    )


async def build_campfire_restrictions_ansel_adams(
    evaluator: Evaluator,
    root,
    data: Optional[CampfireRestrictionsAnselAdams]
):
    parent = evaluator.add_parallel(
        id="campfire_restrictions_ansel_adams_lakes",
        desc="Campfire restrictions in Ansel Adams Wilderness (specific lakes and 1/4 mile outlet rule) supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    named_ok = data is not None and isinstance(data.named_water_bodies, list) and len(data.named_water_bodies) > 0
    exists = (
        data is not None and
        named_ok and
        (data.proximity_ban_distance is not None and data.proximity_ban_distance.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="campfire_aa_exists",
        desc="Includes at least one named water body and the proximity-based ban distance with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="campfire_aa_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )

    # Proximity rule distance
    node_distance = evaluator.add_leaf(
        id="campfire_aa_distance_supported",
        desc="Proximity-based ban distance (1/4 mile from lake outlets) supported by sources",
        parent=parent,
        critical=True
    )
    claim_distance = f"The campfire restriction includes a proximity-based ban of {data.proximity_ban_distance if data else ''} from lake outlets."
    await evaluator.verify(
        claim=claim_distance,
        node=node_distance,
        sources=sources,
        additional_instruction="Confirm language such as 'no wood fires within 1/4 mile of lake outlets' for Ansel Adams Wilderness regulations."
    )

    # Named water body covered by the rule (use first one)
    first_water_body = (data.named_water_bodies[0] if named_ok else "")
    node_named = evaluator.add_leaf(
        id="campfire_aa_named_water_supported",
        desc="Names at least one specific water body covered by the proximity-based ban",
        parent=parent,
        critical=True
    )
    claim_named = f"The cited source(s) state that the campfire ban applies to {first_water_body} in Ansel Adams Wilderness."
    await evaluator.verify(
        claim=claim_named,
        node=node_named,
        sources=sources,
        additional_instruction="Confirm that the named water body (e.g., Thousand Island Lake, Garnet Lake, Shadow Lake) is included in the no-fire zone."
    )


async def build_cathole_requirements(
    evaluator: Evaluator,
    root,
    data: Optional[CatholeRequirements]
):
    parent = evaluator.add_parallel(
        id="cathole_requirements",
        desc="Cathole depth, distances, and disposal procedures supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.depth_inches is not None and data.depth_inches.strip() != "") and
        (data.distance_from_water_camp_trails_feet is not None and data.distance_from_water_camp_trails_feet.strip() != "") and
        (data.disposal_procedures is not None and data.disposal_procedures.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="cathole_exists",
        desc="Cathole depth, distance, and disposal procedures are provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="cathole_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )
    # Depth 6–8 inches
    node_depth = evaluator.add_leaf(
        id="cathole_depth_supported",
        desc="Cathole depth (6–8 inches) supported by sources",
        parent=parent,
        critical=True
    )
    claim_depth = f"Catholes must be {data.depth_inches if data else ''} deep."
    await evaluator.verify(
        claim=claim_depth,
        node=node_depth,
        sources=sources,
        additional_instruction="Confirm the stated depth range includes 6–8 inches."
    )
    # Distance 200 ft from water/camp/trails
    node_dist = evaluator.add_leaf(
        id="cathole_distance_supported",
        desc="Cathole placement at least 200 feet from water, camp, and trails supported by sources",
        parent=parent,
        critical=True
    )
    claim_dist = f"Catholes must be at least {data.distance_from_water_camp_trails_feet if data else ''} from water, camp, and trails."
    await evaluator.verify(
        claim=claim_dist,
        node=node_dist,
        sources=sources,
        additional_instruction="Confirm distance requirement equals or exceeds 200 feet; allow phrasing like '200 ft'."
    )
    # Disposal procedures
    node_disposal = evaluator.add_leaf(
        id="cathole_disposal_supported",
        desc="Cathole disposal procedures (e.g., cover, pack out TP) supported by sources",
        parent=parent,
        critical=True
    )
    claim_disposal = f"The disposal procedure is: {data.disposal_procedures if data else ''}"
    await evaluator.verify(
        claim=claim_disposal,
        node=node_disposal,
        sources=sources,
        additional_instruction="Confirm proper disposal procedures (e.g., cover and disguise catholes; pack out toilet paper/hygiene products where required)."
    )


async def build_water_treatment_and_washing(
    evaluator: Evaluator,
    root,
    data: Optional[WaterTreatmentAndWashing]
):
    parent = evaluator.add_parallel(
        id="water_treatment_and_washing_distance",
        desc="Water treatment and washing distance procedures supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.treat_all_drinking_water is not None and data.treat_all_drinking_water.strip() != "") and
        (data.washing_distance_feet is not None and data.washing_distance_feet.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="water_treatment_washing_exists",
        desc="Statement to treat all water and washing distance are provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="water_treatment_washing_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )
    # Treat all drinking water
    node_treat = evaluator.add_leaf(
        id="water_treatment_supported",
        desc="All drinking water must be treated confirmed by sources",
        parent=parent,
        critical=True
    )
    claim_treat = f"It is required that {data.treat_all_drinking_water if data else ''}"
    await evaluator.verify(
        claim=claim_treat,
        node=node_treat,
        sources=sources,
        additional_instruction="Confirm the requirement to treat or properly purify all drinking water."
    )
    # Washing 200 feet away
    node_wash = evaluator.add_leaf(
        id="washing_distance_supported",
        desc="Washing activities must be at least 200 feet from water sources",
        parent=parent,
        critical=True
    )
    claim_wash = f"Washing must be conducted at least {data.washing_distance_feet if data else ''} from water sources."
    await evaluator.verify(
        claim=claim_wash,
        node=node_wash,
        sources=sources,
        additional_instruction="Confirm the 200-foot distance rule for washing and disposal of greywater; allow '200 ft' phrasing."
    )


async def build_permit_possession_and_presentation(
    evaluator: Evaluator,
    root,
    data: Optional[PermitPossessionPresentation]
):
    parent = evaluator.add_parallel(
        id="permit_possession_and_presentation",
        desc="Permit possession and presentation requirements supported by official URL(s)",
        parent=root,
        critical=True
    )
    sources = listify(data.sources if data else [])
    exists = (
        data is not None and
        (data.must_carry_permit is not None and data.must_carry_permit.strip() != "") and
        (data.must_present_to_rangers is not None and data.must_present_to_rangers.strip() != "") and
        len(sources) > 0
    )
    evaluator.add_custom_node(
        result=exists,
        id="permit_possession_presentation_exists",
        desc="Must carry and present permit statements provided with sources",
        parent=parent,
        critical=True
    )
    await add_official_sources_check(
        evaluator, parent, sources,
        node_id="permit_possession_presentation_sources_official",
        desc="Sources are official (NPS/USFS/Recreation.gov)"
    )
    # Carry/possess
    node_carry = evaluator.add_leaf(
        id="permit_must_carry_supported",
        desc="Must carry/possess permit requirement supported by sources",
        parent=parent,
        critical=True
    )
    claim_carry = f"It is required that {data.must_carry_permit if data else ''}"
    await evaluator.verify(
        claim=claim_carry,
        node=node_carry,
        sources=sources,
        additional_instruction="Confirm language stating that the wilderness permit must be carried/possessed during the trip."
    )
    # Present to rangers
    node_present = evaluator.add_leaf(
        id="permit_must_present_supported",
        desc="Must present permit to ranger upon request supported by sources",
        parent=parent,
        critical=True
    )
    claim_present = f"It is required that {data.must_present_to_rangers if data else ''}"
    await evaluator.verify(
        claim=claim_present,
        node=node_present,
        sources=sources,
        additional_instruction="Confirm language requiring presentation of the permit to rangers upon request."
    )


# =============================================================================
# Main evaluation function
# =============================================================================
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
    Evaluate an answer for the Sierra Nevada regulatory compliance task and return a structured result dict.
    """

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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_compliance(),
        template_class=ComplianceExtraction,
        extraction_name="compliance_extraction"
    )

    # Record allowed official domains for transparency
    evaluator.add_custom_info(
        info={"allowed_official_domains": sorted(list(ALLOWED_OFFICIAL_DOMAINS))},
        info_type="metadata",
        info_name="allowed_domains"
    )

    # Build each rubric item subtree (all children will be marked critical under a critical-like gating via parent nodes)
    await build_entry_trailhead_quota(evaluator, root, extraction.entry_trailhead_quota)
    await build_permit_fee_structure(evaluator, root, extraction.permit_fee_structure)
    await build_permit_advance_window(evaluator, root, extraction.permit_advance_booking_window)
    await build_bear_canister_requirements(evaluator, root, extraction.bear_canister_requirements)
    await build_bear_canister_placement(evaluator, root, extraction.bear_canister_placement_distance)
    await build_camping_distance_from_water(evaluator, root, extraction.camping_distance_from_water)
    await build_paradise_valley_rules(evaluator, root, extraction.paradise_valley_designated_sites_and_night_limit)
    await build_rae_charlotte_limits(evaluator, root, extraction.rae_lakes_and_charlotte_night_limits)
    await build_group_size_limits(evaluator, root, extraction.group_size_limits)
    await build_campfire_restrictions_ansel_adams(evaluator, root, extraction.campfire_restrictions_ansel_adams_lakes)
    await build_cathole_requirements(evaluator, root, extraction.cathole_requirements)
    await build_water_treatment_and_washing(evaluator, root, extraction.water_treatment_and_washing_distance)
    await build_permit_possession_and_presentation(evaluator, root, extraction.permit_possession_and_presentation)

    # Return evaluation summary
    return evaluator.get_summary()