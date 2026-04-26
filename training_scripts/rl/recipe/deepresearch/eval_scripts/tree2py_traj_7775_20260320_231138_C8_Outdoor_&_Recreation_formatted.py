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
TASK_ID = "sll_backpacking_traverse_plan"
TASK_DESCRIPTION = """
Plan a 3-day, 2-night backpacking traverse of the Sierra de la Laguna Biosphere Reserve in Baja California Sur, Mexico, starting from the San Dionisio trailhead and ending at the La Burrera trailhead. The plan must include and comply with all regulatory and logistical requirements, with supporting reference URLs.
"""

EXPECTED_DAILY_FEE_MXN = 125.0
TRIP_DAYS = 3
EXPECTED_TOTAL_MXN = EXPECTED_DAILY_FEE_MXN * TRIP_DAYS  # 375.0


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class RouteSection(BaseModel):
    start_trailhead: Optional[str] = None
    end_trailhead: Optional[str] = None
    designated_access_compliance: Optional[bool] = None
    established_trail_compliance: Optional[bool] = None
    route_urls: List[str] = Field(default_factory=list)


class PermitsFees(BaseModel):
    conanp_camping_permit_required_mentioned: Optional[bool] = None
    entry_fee_rate_per_day_mxn: Optional[str] = None
    total_entry_fee_per_person_mxn: Optional[str] = None
    discounts_noted: Optional[str] = None
    permit_fee_urls: List[str] = Field(default_factory=list)


class CampingSection(BaseModel):
    night1_location: Optional[str] = None
    night1_designated_area: Optional[bool] = None
    night1_60m_from_water: Optional[bool] = None
    night2_location: Optional[str] = None
    night2_designated_area: Optional[bool] = None
    night2_60m_from_water: Optional[bool] = None
    no_permanent_facilities: Optional[bool] = None
    no_excavate_level: Optional[bool] = None
    no_alter_natural: Optional[bool] = None
    camping_urls: List[str] = Field(default_factory=list)


class WaterSection(BaseModel):
    sources: List[str] = Field(default_factory=list)
    purification_method: Optional[str] = None
    no_entry_into_water: Optional[bool] = None
    no_contamination: Optional[bool] = None
    water_urls: List[str] = Field(default_factory=list)


class WasteSection(BaseModel):
    pack_out_trash: Optional[bool] = None
    cathole_depth: Optional[str] = None
    cathole_distance_from_water: Optional[str] = None
    waste_urls: List[str] = Field(default_factory=list)


class FireSection(BaseModel):
    fire_only_designated_sites: Optional[bool] = None
    fire_completely_extinguished: Optional[bool] = None
    no_combustible_materials_left: Optional[bool] = None
    primary_cooking_method: Optional[str] = None
    fire_urls: List[str] = Field(default_factory=list)


class ProhibitedSection(BaseModel):
    no_firearms_axes_machetes: Optional[bool] = None
    no_alcohol: Optional[bool] = None
    no_paints_unless_justified: Optional[bool] = None
    prohibited_urls: List[str] = Field(default_factory=list)


class EnvironmentalSection(BaseModel):
    no_cutting_plants: Optional[bool] = None
    no_disturb_animals: Optional[bool] = None
    no_collect_artifacts_fossils: Optional[bool] = None
    no_exotic_species: Optional[bool] = None
    minimize_noise: Optional[bool] = None
    no_alter_historical_cultural_natural_sites: Optional[bool] = None
    environmental_urls: List[str] = Field(default_factory=list)


class GearSection(BaseModel):
    cold_high_elevation_gear: Optional[bool] = None
    sun_protection_gear: Optional[bool] = None
    lightweight_backpacking_equipment: Optional[bool] = None
    gear_list: List[str] = Field(default_factory=list)


class SafetyNavSection(BaseModel):
    navigation_challenges_addressed: Optional[bool] = None
    navigation_strategy: Optional[str] = None
    emergency_contact_info: Optional[str] = None
    emergency_contact_urls: List[str] = Field(default_factory=list)
    trip_timing_oct_may: Optional[bool] = None


class RegulatoryRefsSection(BaseModel):
    official_conanp_urls: List[str] = Field(default_factory=list)
    supporting_urls: List[str] = Field(default_factory=list)
    ack_authorized_service_providers: Optional[bool] = None
    ack_capacity_limit_600: Optional[bool] = None


class PlanExtraction(BaseModel):
    route: Optional[RouteSection] = None
    permits_fees: Optional[PermitsFees] = None
    camping: Optional[CampingSection] = None
    water: Optional[WaterSection] = None
    waste: Optional[WasteSection] = None
    fire: Optional[FireSection] = None
    prohibited: Optional[ProhibitedSection] = None
    environmental: Optional[EnvironmentalSection] = None
    gear: Optional[GearSection] = None
    safety_nav: Optional[SafetyNavSection] = None
    regulatory_refs: Optional[RegulatoryRefsSection] = None
    all_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract structured information from the backpacking plan answer for Sierra de la Laguna Biosphere Reserve (Baja California Sur, Mexico). Follow these rules:
    - Only extract what is explicitly stated in the answer.
    - For boolean fields: set true only if the plan clearly commits or confirms that item; set false only if it explicitly denies; otherwise return null.
    - For URLs: extract actual URLs present in the answer (plain or markdown).
    - Return null for fields not present. Do not invent.

    Return a JSON object with this schema:

    {
      "route": {
        "start_trailhead": str|null,
        "end_trailhead": str|null,
        "designated_access_compliance": bool|null,   // plan confirms any Core Zone entry uses designated access routes only
        "established_trail_compliance": bool|null,   // plan commits to staying on established/interpretive trails
        "route_urls": [urls...]
      },
      "permits_fees": {
        "conanp_camping_permit_required_mentioned": bool|null,  // plan states a CONANP camping permit may be required for multi-day trips
        "entry_fee_rate_per_day_mxn": str|null,                 // e.g., "$125 MXN per person per day"
        "total_entry_fee_per_person_mxn": str|null,             // total for entire 3-day trip per person, e.g., "375 MXN"
        "discounts_noted": str|null,                            // any discounts mentioned (or "none"/"no discounts")
        "permit_fee_urls": [urls...]
      },
      "camping": {
        "night1_location": str|null,
        "night1_designated_area": bool|null,
        "night1_60m_from_water": bool|null,
        "night2_location": str|null,
        "night2_designated_area": bool|null,
        "night2_60m_from_water": bool|null,
        "no_permanent_facilities": bool|null,
        "no_excavate_level": bool|null,
        "no_alter_natural": bool|null,
        "camping_urls": [urls...]
      },
      "water": {
        "sources": [str...],                    // specific named water sources along route if given
        "purification_method": str|null,        // filter/boil/chemical etc.
        "no_entry_into_water": bool|null,       // plan commits to not entering water bodies
        "no_contamination": bool|null,          // plan commits to not contaminating water
        "water_urls": [urls...]
      },
      "waste": {
        "pack_out_trash": bool|null,            // plan commits to packing out all trash
        "cathole_depth": str|null,              // e.g., "6-8 inches", "15–20 cm"
        "cathole_distance_from_water": str|null,// e.g., "200 feet", "60 m"
        "waste_urls": [urls...]
      },
      "fire": {
        "fire_only_designated_sites": bool|null,
        "fire_completely_extinguished": bool|null,
        "no_combustible_materials_left": bool|null,
        "primary_cooking_method": str|null,     // e.g., "canister stove"
        "fire_urls": [urls...]
      },
      "prohibited": {
        "no_firearms_axes_machetes": bool|null,
        "no_alcohol": bool|null,
        "no_paints_unless_justified": bool|null,
        "prohibited_urls": [urls...]
      },
      "environmental": {
        "no_cutting_plants": bool|null,
        "no_disturb_animals": bool|null,
        "no_collect_artifacts_fossils": bool|null,
        "no_exotic_species": bool|null,
        "minimize_noise": bool|null,
        "no_alter_historical_cultural_natural_sites": bool|null,
        "environmental_urls": [urls...]
      },
      "gear": {
        "cold_high_elevation_gear": bool|null,          // thermal/cold-weather gear for up to 2,100m
        "sun_protection_gear": bool|null,
        "lightweight_backpacking_equipment": bool|null,
        "gear_list": [str...]
      },
      "safety_nav": {
        "navigation_challenges_addressed": bool|null,   // mentions trails not well-marked AND provides navigation strategy
        "navigation_strategy": str|null,                // e.g., GPS/GPX/map+compass/guide
        "emergency_contact_info": str|null,             // any phone/email/address or named office contact info
        "emergency_contact_urls": [urls...],
        "trip_timing_oct_may": bool|null                // plan confirms travel window within Oct–May
      },
      "regulatory_refs": {
        "official_conanp_urls": [urls...],              // URLs that appear to be official CONANP pages for Sierra de la Laguna
        "supporting_urls": [urls...],                   // other reference URLs supporting regulations/logistics (fees, permits, rules, maps)
        "ack_authorized_service_providers": bool|null,  // plan acknowledges coordination with authorized service providers
        "ack_capacity_limit_600": bool|null             // plan acknowledges 600-person capacity limit
      },
      "all_urls": [urls...]                              // every URL found in the answer (deduplicate)
    }

    URL classification guidelines:
    - official_conanp_urls: URLs from Mexican government/CONANP domains that explicitly cover Sierra de la Laguna (Spanish pages acceptable).
    - supporting_urls: any other URLs that support rules, fees, permits, maps, contacts, logistics mentioned in the plan.
    """


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def parse_mxn_amount(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    s = text.strip().lower()
    # Replace common currency words/symbols and commas
    s = s.replace(",", " ")
    # Find first number (integer or decimal)
    m = re.search(r"(\d+[.,]?\d*)", s)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return float(num)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Section verification builders                                               #
# --------------------------------------------------------------------------- #
async def build_route_and_access(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Route_and_Access",
        desc="Route details match required start/end and comply with access/trail rules.",
        parent=parent,
        critical=True
    )
    route = plan.route or RouteSection()

    # Start and End Trailheads
    start_end_leaf = evaluator.add_leaf(
        id="Start_and_End_Trailheads",
        desc="Plan explicitly starts at San Dionisio trailhead and ends at La Burrera trailhead.",
        parent=node,
        critical=True
    )
    start_txt = route.start_trailhead or ""
    end_txt = route.end_trailhead or ""
    start_end_claim = (
        f"The trip plan explicitly starts at the San Dionisio trailhead and ends at the La Burrera trailhead. "
        f"In the plan, the start was given as '{start_txt}' and the end as '{end_txt}'. "
        f"Treat common variants or Spanish phrasing (e.g., 'San Dionisio', 'Cañón de San Dionisio', 'La Burrera') as matches."
    )
    await evaluator.verify(
        claim=start_end_claim,
        node=start_end_leaf,
        additional_instruction="Verify only against the answer text. Allow minor spelling/casing/punctuation variants."
    )

    # Designated Access Route Compliance
    access_leaf = evaluator.add_leaf(
        id="Designated_Access_Route_Compliance",
        desc="Plan confirms any Core Zone entry uses one (or more) of the designated access routes listed in the constraints.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan confirms that any entry to the Core Zone will be via designated access routes only (as per CONANP rules).",
        node=access_leaf,
        additional_instruction="Look for an explicit commitment in the plan text."
    )

    # Established Trail Compliance
    established_leaf = evaluator.add_leaf(
        id="Established_Trail_Compliance",
        desc="Plan commits to staying on established routes and interpretive trails designated by the Directorate.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan commits to staying on established routes/interpretive trails designated by the Reserve Directorate.",
        node=established_leaf,
        additional_instruction="Check for an explicit commitment in the plan text."
    )


async def build_permits_and_fees(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Permits_and_Fees",
        desc="Plan identifies required permits and correctly computes fees for the full 3-day trip, including discounts if available.",
        parent=parent,
        critical=True
    )
    pf = plan.permits_fees or PermitsFees()

    # CONANP Camping Permit Requirement Stated
    permit_leaf = evaluator.add_leaf(
        id="CONANP_Camping_Permit_Requirement_Stated",
        desc="Plan identifies whether/that a CONANP permit may be required for multi-day hikes involving camping (per constraints).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan states that a CONANP permit may be required for multi-day hikes that include camping in the Sierra de la Laguna Biosphere Reserve.",
        node=permit_leaf,
        additional_instruction="Verify this from the answer text only."
    )

    # Entry Fee Total Calculation (125 MXN/day/person for 3 days = 375 MXN)
    rate_num = parse_mxn_amount(pf.entry_fee_rate_per_day_mxn)
    total_num = parse_mxn_amount(pf.total_entry_fee_per_person_mxn)
    correct_calc = (rate_num == EXPECTED_DAILY_FEE_MXN) and (total_num == EXPECTED_TOTAL_MXN)

    evaluator.add_custom_node(
        result=bool(correct_calc),
        id="Entry_Fee_Total_Calculation",
        desc=f"Plan calculates total entry fees per person for the full {TRIP_DAYS}-day trip using the stated rate ($125 MXN/day). Expected total: {int(EXPECTED_TOTAL_MXN)} MXN.",
        parent=node,
        critical=True
    )

    # Discounts Noted If Available
    discounts_leaf = evaluator.add_leaf(
        id="Discounts_Noted_If_Available",
        desc="Plan notes any available entry-fee discounts (or explicitly states none identified).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan either lists available entry-fee discounts or explicitly states that no discounts were identified.",
        node=discounts_leaf,
        additional_instruction="Accept statements like 'no discounts available/identified' as satisfying this requirement."
    )


async def build_camping_plan(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Camping_Plan",
        desc="Plan specifies compliant designated camping locations for both nights and commits to camping-site rules.",
        parent=parent,
        critical=True
    )
    c = plan.camping or CampingSection()

    # Night 1 Camping Location
    n1_leaf = evaluator.add_leaf(
        id="Night_1_Camping_Location",
        desc="Night 1 camping location is specified and designated; plan confirms at least 60 meters from water bodies.",
        parent=node,
        critical=True
    )
    n1_claim = (
        "The plan specifies the Night 1 camping location by name, confirms it is a designated camping area, "
        "and states it will be at least 60 meters from water bodies."
    )
    await evaluator.verify(
        claim=n1_claim,
        node=n1_leaf,
        additional_instruction="Verify from the plan text. Accept equivalents like '60 m' or '200 ft' as meeting the 60 meters rule."
    )

    # Night 2 Camping Location
    n2_leaf = evaluator.add_leaf(
        id="Night_2_Camping_Location",
        desc="Night 2 camping location is specified and designated; plan confirms at least 60 meters from water bodies.",
        parent=node,
        critical=True
    )
    n2_claim = (
        "The plan specifies the Night 2 camping location by name, confirms it is a designated camping area, "
        "and states it will be at least 60 meters from water bodies."
    )
    await evaluator.verify(
        claim=n2_claim,
        node=n2_leaf,
        additional_instruction="Verify from the plan text. Accept equivalents like '60 m' or '200 ft' as meeting the 60 meters rule."
    )

    # No Permanent Camping Facilities
    evaluator.add_leaf(
        id="No_Permanent_Camping_Facilities",
        desc="Plan commits to not erecting permanent camping facilities.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan commits to not erecting or installing permanent camping facilities.",
        node=evaluator.find_node("No_Permanent_Camping_Facilities")
    )

    # No Excavating or Leveling Campsite
    evaluator.add_leaf(
        id="No_Excavating_or_Leveling_Campsite",
        desc="Plan commits to not excavating or leveling the land where camping.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan commits to not excavating or leveling the land at the camping site.",
        node=evaluator.find_node("No_Excavating_or_Leveling_Campsite")
    )

    # No Altering Natural Conditions
    evaluator.add_leaf(
        id="No_Altering_Natural_Conditions_At_Campsite",
        desc="Plan commits to not altering natural conditions of the camping site.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan commits to not altering the natural conditions of the camping site.",
        node=evaluator.find_node("No_Altering_Natural_Conditions_At_Campsite")
    )


async def build_water_management(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Water_Management",
        desc="Plan identifies water sources, purification/filtration approach, and waterbody protection compliance.",
        parent=parent,
        critical=True
    )
    w = plan.water or WaterSection()

    # Water Source Identification
    src_leaf = evaluator.add_leaf(
        id="Water_Source_Identification",
        desc="Plan identifies available water sources along the route.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan identifies one or more natural water sources along the intended route (e.g., arroyos, springs, tanks, or named water points).",
        node=src_leaf,
        additional_instruction="Verify presence of at least one named or described water source in the plan."
    )

    # Purification/Filtration System
    pur_leaf = evaluator.add_leaf(
        id="Water_Purification_or_Filtration_System",
        desc="Plan specifies a water purification/filtration method/system to be carried and used for natural sources.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan specifies a water purification or filtration method (e.g., filter, boil, chemical) that will be used for natural sources.",
        node=pur_leaf
    )

    # No Waterbody Entry
    no_entry_leaf = evaluator.add_leaf(
        id="No_Waterbody_Entry",
        desc="Plan commits to not entering bodies of water.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan commits to not entering bodies of water.",
        node=no_entry_leaf
    )

    # No Water Contamination
    no_contam_leaf = evaluator.add_leaf(
        id="No_Water_Contamination",
        desc="Plan commits to not contaminating bodies of water with organic or inorganic waste.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan commits to not contaminating water bodies with organic or inorganic waste.",
        node=no_contam_leaf
    )


async def build_waste_disposal(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Waste_Disposal",
        desc="Plan provides complete trash and human-waste handling compliant with stated rules.",
        parent=parent,
        critical=True
    )

    # Trash Pack Out
    trash_leaf = evaluator.add_leaf(
        id="Trash_Pack_Out",
        desc="Plan commits to packing out all garbage (no trash left, deposited, or thrown).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan commits to packing out all trash (no litter or leaving garbage).",
        node=trash_leaf
    )

    # Cat-hole Technique And Conditions
    cathole_leaf = evaluator.add_leaf(
        id="Cat_Hole_Technique_And_Conditions",
        desc="Plan specifies cat-hole technique when no toilets are available, including required depth (6–8 inches) and minimum distance from water (at least 200 feet).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="When toilets are unavailable, the plan specifies cat-holes 6–8 inches (15–20 cm) deep and at least 200 feet (60 meters) from water sources.",
        node=cathole_leaf,
        additional_instruction="Allow metric/imperial equivalents. Verify from the answer text."
    )


async def build_fire_management(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Fire_Management",
        desc="Plan addresses fire rules and states primary cooking method.",
        parent=parent,
        critical=True
    )
    f = plan.fire or FireSection()

    # Fire only in designated sites
    fire_only_leaf = evaluator.add_leaf(
        id="Fire_Only_in_Designated_Sites",
        desc="Plan confirms any fires (if made) will be only in designated sites and under specified conditions.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan confirms that any fires, if made, will be only in designated sites and under required conditions.",
        node=fire_only_leaf
    )

    # Fire completely extinguished
    fire_out_leaf = evaluator.add_leaf(
        id="Fire_Completely_Extinguished",
        desc="Plan confirms all campfires will be completely extinguished before leaving.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan confirms that all campfires will be completely extinguished before leaving camp.",
        node=fire_out_leaf
    )

    # No combustible materials left
    no_comb_leaf = evaluator.add_leaf(
        id="No_Combustible_Materials_Left",
        desc="Plan commits to not leaving combustible materials that create fire risks.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan commits to not leaving combustible materials that could create fire risks.",
        node=no_comb_leaf
    )

    # Primary cooking method specified (existence check)
    evaluator.add_custom_node(
        result=bool((f.primary_cooking_method or "").strip()),
        id="Primary_Cooking_Method_Specified",
        desc="Plan specifies the primary cooking method.",
        parent=node,
        critical=True
    )


async def build_prohibited_items(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Prohibited_Items",
        desc="Plan confirms prohibited items are excluded from the gear list.",
        parent=parent,
        critical=True
    )

    # No Firearms/Axes/Machetes
    fam_leaf = evaluator.add_leaf(
        id="No_Firearms_Axes_Machetes",
        desc="Plan confirms no firearms, axes, or machetes will be brought.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan confirms that no firearms, axes, or machetes will be brought.",
        node=fam_leaf
    )

    # No Alcoholic Beverages
    alcohol_leaf = evaluator.add_leaf(
        id="No_Alcoholic_Beverages",
        desc="Plan confirms no alcoholic beverages will be brought.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan confirms that no alcoholic beverages will be brought.",
        node=alcohol_leaf
    )

    # No Paints Unless Justified and Controlled
    paints_leaf = evaluator.add_leaf(
        id="No_Paints_Unless_Justified_and_Controlled",
        desc="Plan confirms no paints of any kind will be introduced unless justified and under control.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan confirms no paints of any kind will be introduced unless clearly justified and strictly controlled.",
        node=paints_leaf
    )


async def build_environmental_protection(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Environmental_Protection",
        desc="Plan commits to the listed environmental protection rules.",
        parent=parent,
        critical=True
    )

    async def add_simple_commitment(leaf_id: str, desc: str, claim: str):
        leaf = evaluator.add_leaf(id=leaf_id, desc=desc, parent=node, critical=True)
        await evaluator.verify(claim=claim, node=leaf)

    await add_simple_commitment(
        "No_Cutting_Plants",
        "Plan commits to not cutting or mutilating plants.",
        "The plan commits to not cutting, mutilating, or damaging plants."
    )
    await add_simple_commitment(
        "No_Disturbing_Animals",
        "Plan commits to not disturbing animals.",
        "The plan commits to not disturbing or harassing wildlife."
    )
    await add_simple_commitment(
        "No_Collecting_Artifacts_or_Fossils",
        "Plan commits to not taking fossils or archaeological artifacts.",
        "The plan commits to not collecting fossils or archaeological artifacts."
    )
    await add_simple_commitment(
        "No_Introducing_Exotic_Species",
        "Plan commits to not introducing exotic species of any kind.",
        "The plan commits to not introducing exotic/non-native species of any kind."
    )
    await add_simple_commitment(
        "Minimize_Noise",
        "Plan commits to not making unnecessary noise.",
        "The plan commits to minimizing noise and avoiding unnecessary disturbance."
    )
    await add_simple_commitment(
        "No_Altering_Historical_Cultural_Natural_Sites",
        "Plan commits to not altering sites of historical, cultural, or natural value.",
        "The plan commits to not altering sites of historical, cultural, or natural value."
    )


async def build_gear_preparation(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Gear_Preparation",
        desc="Plan includes appropriate gear for high-elevation conditions and lightweight backpacking.",
        parent=parent,
        critical=True
    )

    # Cold/high elevation gear
    cold_leaf = evaluator.add_leaf(
        id="Cold_and_High_Elevation_Gear",
        desc="Gear list includes thermal/cold-weather gear suitable for camping up to 2,100 meters.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The gear list includes thermal/cold-weather gear suitable for camping up to approximately 2,100 meters elevation.",
        node=cold_leaf
    )

    # Sun protection gear
    sun_leaf = evaluator.add_leaf(
        id="Sun_Protection_Gear",
        desc="Gear list includes sun protection.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The gear list includes sun protection (e.g., hat, sunscreen, sunglasses, UPF clothing).",
        node=sun_leaf
    )

    # Lightweight backpacking equipment
    light_leaf = evaluator.add_leaf(
        id="Lightweight_Backpacking_Equipment",
        desc="Gear list includes lightweight backpacking equipment suitable for mountain terrain.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The gear list includes lightweight backpacking equipment suitable for mountain terrain.",
        node=light_leaf
    )


async def build_safety_and_navigation(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Safety_and_Navigation",
        desc="Plan addresses navigation challenges, emergency contact info, and recommended seasonal timing.",
        parent=parent,
        critical=True
    )

    # Navigation challenges addressed
    nav_leaf = evaluator.add_leaf(
        id="Navigation_Challenges_Addressed",
        desc="Plan notes that not all trails are well-marked and provides a navigation strategy.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan notes that not all trails are well-marked and provides a concrete navigation strategy (e.g., GPX, GPS, map+compass, or guide).",
        node=nav_leaf
    )

    # Emergency contact info
    contact_leaf = evaluator.add_leaf(
        id="Emergency_Contact_Info_For_Reserve_Office",
        desc="Plan provides emergency contact information for the reserve office.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan provides emergency contact information for the Sierra de la Laguna Biosphere Reserve office (such as a phone number, email, or official office URL).",
        node=contact_leaf,
        additional_instruction="Verify from the answer text only."
    )

    # Trip timing Oct–May
    timing_leaf = evaluator.add_leaf(
        id="Trip_Timing_October_to_May",
        desc="Plan confirms trip timing falls within the recommended October–May visiting period.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan confirms that the trip timing falls within the recommended October to May visiting period.",
        node=timing_leaf
    )


async def build_regulatory_references(evaluator: Evaluator, parent, plan: PlanExtraction):
    node = evaluator.add_parallel(
        id="Regulatory_References_and_Key_Acknowledgments",
        desc="Plan includes official regulatory references and acknowledges key regulatory requirements.",
        parent=parent,
        critical=True
    )
    r = plan.regulatory_refs or RegulatoryRefsSection()

    # Official CONANP Regulations URL(s)
    conanp_leaf = evaluator.add_leaf(
        id="Official_CONANP_Regulations_URL",
        desc="Plan includes reference URL(s) to official CONANP regulations for Sierra La Laguna.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This page is an official CONANP (Comisión Nacional de Áreas Naturales Protegidas) page that provides regulations or official visitor rules for the Sierra de la Laguna Biosphere Reserve.",
        node=conanp_leaf,
        sources=r.official_conanp_urls,
        additional_instruction="Spanish-language CONANP pages are acceptable. The page should clearly pertain to Sierra de la Laguna or its regulations/visitor rules."
    )

    # Supporting URLs for regulatory/logistical claims
    supp_leaf = evaluator.add_leaf(
        id="Supporting_URLs_For_Regulatory_and_Logistical_Claims",
        desc="Plan provides supporting reference URLs for the regulatory requirements and key logistical information it relies on.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="This page supports at least one regulatory requirement or key logistical detail cited in the plan (e.g., entry fees, permits, camping rules, fire restrictions, maps, or official contact information) for the Sierra de la Laguna Biosphere Reserve.",
        node=supp_leaf,
        sources=r.supporting_urls,
        additional_instruction="It's sufficient if any one of the provided URLs substantiates a cited regulatory or logistical detail."
    )

    # Authorized Service Provider Coordination Acknowledged
    auth_leaf = evaluator.add_leaf(
        id="Authorized_Service_Provider_Coordination_Acknowledged",
        desc="Plan acknowledges the requirement to coordinate with authorized service providers.",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan acknowledges the requirement to coordinate with authorized service providers for the reserve.",
        node=auth_leaf
    )

    # Reserve Capacity Limit Acknowledged
    cap_leaf = evaluator.add_leaf(
        id="Reserve_Capacity_Limit_Acknowledged",
        desc="Plan acknowledges the reserve capacity limitation (600 people).",
        parent=node,
        critical=True
    )
    await evaluator.verify(
        claim="The plan acknowledges that the reserve has a capacity limitation of approximately 600 people.",
        node=cap_leaf,
        additional_instruction="Allow minor phrasing differences as long as the 600-person capacity limit is acknowledged."
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
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregates all sections in parallel
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

    # 1) Extract structured plan information
    plan: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction"
    )

    # 2) Build verification tree (all sections are critical under a critical root)
    # Multi_Day_Backpacking_Plan root wrapper (critical)
    plan_node = evaluator.add_parallel(
        id="Multi_Day_Backpacking_Plan",
        desc="Plan a 3-day, 2-night backpacking traverse in Sierra de la Laguna Biosphere Reserve starting at San Dionisio and ending at La Burrera, covering all regulatory/logistics with URLs.",
        parent=root,
        critical=True
    )

    # Sections (all critical and parallel under the plan node)
    await build_route_and_access(evaluator, plan_node, plan)
    await build_permits_and_fees(evaluator, plan_node, plan)
    await build_camping_plan(evaluator, plan_node, plan)
    await build_water_management(evaluator, plan_node, plan)
    await build_waste_disposal(evaluator, plan_node, plan)
    await build_fire_management(evaluator, plan_node, plan)
    await build_prohibited_items(evaluator, plan_node, plan)
    await build_environmental_protection(evaluator, plan_node, plan)
    await build_gear_preparation(evaluator, plan_node, plan)
    await build_safety_and_navigation(evaluator, plan_node, plan)
    await build_regulatory_references(evaluator, plan_node, plan)

    # 3) Return structured evaluation summary
    # Add small custom info for fee calculation expectation
    evaluator.add_custom_info(
        {
            "expected_daily_fee_mxn": EXPECTED_DAILY_FEE_MXN,
            "trip_days": TRIP_DAYS,
            "expected_total_fee_mxn_per_person": EXPECTED_TOTAL_MXN
        },
        info_type="calculation_expectations",
        info_name="fee_expectations"
    )

    return evaluator.get_summary()