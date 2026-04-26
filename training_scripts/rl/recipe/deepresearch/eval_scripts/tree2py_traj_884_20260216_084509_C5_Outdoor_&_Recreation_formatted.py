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
TASK_ID = "trip_planning_blm_2026"
TASK_DESCRIPTION = """
A U.S. resident family is planning a 20-day outdoor recreation trip in July 2026. Their itinerary includes visiting 5 different national parks that each charge a $35 per-vehicle entrance fee. They also plan to spend 12 consecutive days dispersed camping on BLM-managed public lands in the Intermountain Region, using their vehicle to access sites along designated forest roads. During the trip, they expect to encounter various wildlife including deer, elk, and potentially bears.

Provide a comprehensive trip planning guide that addresses:

1. Whether purchasing an America the Beautiful Annual Pass is more cost-effective than paying individual entrance fees, including your calculation and recommendation

2. How they must manage their 12-day dispersed camping period to comply with BLM regulations regarding stay limits, property attendance rules, and site selection practices, as well as motorized access requirements in the Intermountain Region

3. What minimum distances they must maintain from different types of wildlife during viewing opportunities

4. What types of surfaces are appropriate for establishing camp under Leave No Trace Principle 2, and what site selection priority they should follow
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class PassSection(BaseModel):
    total_entrance_fees_stated: Optional[str] = None  # e.g., "$175"
    pass_cost_stated: Optional[str] = None            # e.g., "$80"
    pass_cost_effective_date: Optional[str] = None    # e.g., "January 1, 2026"
    pass_sources: List[str] = Field(default_factory=list)
    comparison_recommendation_text: Optional[str] = None  # e.g., "Buy the annual pass"


class BLMSection(BaseModel):
    stay_limit_statement_text: Optional[str] = None
    stay_limit_sources: List[str] = Field(default_factory=list)
    property_attendance_statement_text: Optional[str] = None
    property_attendance_sources: List[str] = Field(default_factory=list)
    site_selection_statement_text: Optional[str] = None
    site_selection_sources: List[str] = Field(default_factory=list)
    motor_access_statement_text: Optional[str] = None
    motor_access_sources: List[str] = Field(default_factory=list)


class WildlifeSection(BaseModel):
    general_distance_statement_text: Optional[str] = None  # e.g., "25 yards from most wildlife"
    general_distance_sources: List[str] = Field(default_factory=list)
    bear_distance_statement_text: Optional[str] = None     # e.g., "100 yards from bears & wolves"
    bear_distance_sources: List[str] = Field(default_factory=list)


class LNTSection(BaseModel):
    durable_surfaces_list: List[str] = Field(default_factory=list)  # extracted from answer
    surfaces_sources: List[str] = Field(default_factory=list)
    prioritize_existing_statement_text: Optional[str] = None
    priority_sources: List[str] = Field(default_factory=list)


class TripGuideExtraction(BaseModel):
    pass_section: Optional[PassSection] = None
    blm_section: Optional[BLMSection] = None
    wildlife_section: Optional[WildlifeSection] = None
    lnt_section: Optional[LNTSection] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_guide() -> str:
    return """
Extract the following structured information from the answer. Only extract facts or URLs explicitly present in the answer text.

1) Pass and fees analysis:
- pass_section.total_entrance_fees_stated: The total cost (as written, including $ if present) that the answer claims for paying individual entrance fees for 5 park visits at $35 per vehicle; if not explicitly stated, return null.
- pass_section.pass_cost_stated: The dollar amount for the America the Beautiful Annual Pass, as stated in the answer; if not stated, return null.
- pass_section.pass_cost_effective_date: Any effective date the answer mentions for that pass price (e.g., "effective January 1, 2026"); if not present, return null.
- pass_section.pass_sources: All URLs the answer cites to support the annual pass price or pass details. Return an empty list if none are present.
- pass_section.comparison_recommendation_text: The answer’s explicit recommendation comparing paying individual fees vs buying the pass (e.g., "buy the annual pass as it's cheaper"); if none, return null.

2) BLM dispersed camping compliance:
- blm_section.stay_limit_statement_text: The statement in the answer about the BLM dispersed camping stay limit (e.g., "14 days in any 28-day period"); if none, return null.
- blm_section.stay_limit_sources: All URLs cited to support BLM stay limits.
- blm_section.property_attendance_statement_text: The statement in the answer about unattended personal property limits on BLM lands (e.g., "no more than 10 days"); if none, return null.
- blm_section.property_attendance_sources: All URLs cited to support the unattended property rule.
- blm_section.site_selection_statement_text: The statement about using existing/disturbed sites to avoid creating new disturbances; if none, return null.
- blm_section.site_selection_sources: All URLs cited to support that site selection practice.
- blm_section.motor_access_statement_text: The statement about motorized vehicle access/parking distance for dispersed camping in the Intermountain Region (e.g., "within 150 feet of designated routes"); if none, return null.
- blm_section.motor_access_sources: All URLs cited to support the Intermountain Region motorized access distance rule.

3) Wildlife viewing distances:
- wildlife_section.general_distance_statement_text: The stated minimum distance for most wildlife (e.g., "stay at least 25 yards from deer and elk"); if none, return null.
- wildlife_section.general_distance_sources: All URLs cited to support the general wildlife distance.
- wildlife_section.bear_distance_statement_text: The stated minimum distance for bears/wolves (e.g., "stay at least 100 yards"); if none, return null.
- wildlife_section.bear_distance_sources: All URLs cited to support the bear/wolf distance.

4) Leave No Trace Principle 2 (durable surfaces and priority):
- lnt_section.durable_surfaces_list: A list of the surfaces the answer claims are durable (each item as a string, exactly as written), such as "rock", "gravel", "dry grasses", "sand", "snow", "established trails/campsites". Return an empty list if none are given.
- lnt_section.surfaces_sources: All URLs cited to support durable surfaces guidance.
- lnt_section.prioritize_existing_statement_text: The statement noting that using existing established campsites is preferred to minimize impact; if none, return null.
- lnt_section.priority_sources: All URLs cited to support the "prioritize existing sites" guidance.

Return a single JSON object with fields: pass_section, blm_section, wildlife_section, lnt_section.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_pass_cost_effectiveness(evaluator: Evaluator, parent_node, data: TripGuideExtraction) -> None:
    """
    Build and verify the 'Pass_Cost_Effectiveness' subtree (sequential).
    """
    psec = data.pass_section or PassSection()

    pass_node = evaluator.add_sequential(
        id="Pass_Cost_Effectiveness",
        desc="Correctly analyzes whether the America the Beautiful Annual Pass is cost-effective for the specified trip",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: Calculate_Total_Entrance_Fees (simple logic check based on answer content)
    calc_leaf = evaluator.add_leaf(
        id="Calculate_Total_Entrance_Fees",
        desc="Correctly calculates the total cost of individual entrance fees for 5 park visits at $35 per vehicle",
        parent=pass_node,
        critical=True
    )
    # Build claim
    if psec.total_entrance_fees_stated:
        calc_claim = (
            f"The answer calculates the total cost for five ($35) park entrances as {psec.total_entrance_fees_stated}, "
            f"and this equals $175."
        )
    else:
        calc_claim = (
            "The answer explicitly calculates the total cost for five national park entrances at $35 each as $175."
        )
    await evaluator.verify(
        claim=calc_claim,
        node=calc_leaf,
        additional_instruction="Check the answer text for the stated total and verify that 5 × $35 = $175. "
                               "If the answer's stated total is not $175 or is missing, mark this incorrect."
    )

    # Gate: ensure pass cost sources are provided
    pass_cost_src_exists = evaluator.add_custom_node(
        result=bool(psec.pass_sources),
        id="Identify_Pass_Cost_Sources_Provided",
        desc="Sources are provided to support the annual pass cost claim",
        parent=pass_node,
        critical=True
    )

    # Leaf 2: Identify_Pass_Cost (verify by provided URLs)
    pass_cost_leaf = evaluator.add_leaf(
        id="Identify_Pass_Cost",
        desc="Correctly identifies the 2026 America the Beautiful Resident Annual Pass cost as $80 (effective January 1, 2026)",
        parent=pass_node,
        critical=True
    )
    pass_cost_claim = "The America the Beautiful Annual Pass (standard annual pass) costs $80."
    await evaluator.verify(
        claim=pass_cost_claim,
        node=pass_cost_leaf,
        sources=psec.pass_sources if psec.pass_sources else None,
        additional_instruction=(
            "Verify that the page refers to the standard 'America the Beautiful' Annual Pass at $80 (not the Senior, "
            "Access, Volunteer, Military, or 4th Grade passes). If the page does not explicitly mention the 2026 "
            "effective date, ignore that detail as long as the price is $80."
        )
    )

    # Leaf 3: Provide_Valid_Comparison (simple logical check on answer)
    comparison_leaf = evaluator.add_leaf(
        id="Provide_Valid_Comparison",
        desc="Provides a valid comparison between total entrance fees and annual pass cost with a logical recommendation",
        parent=pass_node,
        critical=True
    )
    comp_claim = (
        "The answer compares a total of $175 for five individual entrances versus an $80 annual pass and recommends "
        "purchasing the annual pass as the more cost-effective option."
    )
    await evaluator.verify(
        claim=comp_claim,
        node=comparison_leaf,
        additional_instruction=(
            "Check that the answer explicitly makes a comparison between the $175 total and the $80 annual pass and "
            "draws a logical conclusion (i.e., recommends the annual pass as cheaper). Minor wording differences are acceptable."
        )
    )


async def verify_dispersed_camping_compliance(evaluator: Evaluator, parent_node, data: TripGuideExtraction) -> None:
    """
    Build and verify the 'Dispersed_Camping_Compliance' subtree (parallel).
    """
    bsec = data.blm_section or BLMSection()

    blm_node = evaluator.add_parallel(
        id="Dispersed_Camping_Compliance",
        desc="Correctly addresses BLM dispersed camping regulations for the 12-day camping period",
        parent=parent_node,
        critical=True
    )

    # Stay limit: source existence
    stay_src_exists = evaluator.add_custom_node(
        result=bool(bsec.stay_limit_sources),
        id="Stay_Limit_Sources_Provided",
        desc="Sources are provided to support BLM dispersed camping stay limits",
        parent=blm_node,
        critical=True
    )

    # Leaf: Stay_Limit_Compliance
    stay_leaf = evaluator.add_leaf(
        id="Stay_Limit_Compliance",
        desc="Correctly states that dispersed camping is limited to 14 days within any 28 consecutive day period, and confirms that 12 consecutive days complies with this limit",
        parent=blm_node,
        critical=True
    )
    stay_claim = (
        "On BLM-managed public lands, dispersed camping is limited to a maximum of 14 days within any 28 consecutive "
        "day period; therefore, a 12-day consecutive stay complies with this limit."
    )
    await evaluator.verify(
        claim=stay_claim,
        node=stay_leaf,
        sources=bsec.stay_limit_sources if bsec.stay_limit_sources else None,
        additional_instruction="Accept equivalent wording like '14 days in any 28-day period'. Logical implication that 12 < 14 is acceptable."
    )

    # Property attendance: source existence
    prop_src_exists = evaluator.add_custom_node(
        result=bool(bsec.property_attendance_sources),
        id="Property_Attendance_Sources_Provided",
        desc="Sources are provided to support BLM unattended property rules",
        parent=blm_node,
        critical=True
    )

    # Leaf: Property_Attendance_Rule
    prop_leaf = evaluator.add_leaf(
        id="Property_Attendance_Rule",
        desc="Correctly states that personal property cannot be left unattended for more than 10 days on BLM lands",
        parent=blm_node,
        critical=True
    )
    prop_claim = "On BLM lands, personal property may not be left unattended for more than 10 days."
    await evaluator.verify(
        claim=prop_claim,
        node=prop_leaf,
        sources=bsec.property_attendance_sources if bsec.property_attendance_sources else None,
        additional_instruction="Accept phrasing like 'unattended property' and 'no more than 10 days'. Ignore Alaska-specific exceptions if the cited source pertains to the Lower 48."
    )

    # Site selection: source existence
    site_src_exists = evaluator.add_custom_node(
        result=bool(bsec.site_selection_sources),
        id="Site_Selection_Sources_Provided",
        desc="Sources are provided to support site selection practices",
        parent=blm_node,
        critical=True
    )

    # Leaf: Site_Selection_Practices
    site_leaf = evaluator.add_leaf(
        id="Site_Selection_Practices",
        desc="Correctly states that campers should use existing disturbed sites when possible to avoid creating new disturbances",
        parent=blm_node,
        critical=True
    )
    site_claim = "Campers should use existing, previously disturbed or established sites when possible to avoid creating new disturbances."
    await evaluator.verify(
        claim=site_claim,
        node=site_leaf,
        sources=bsec.site_selection_sources if bsec.site_selection_sources else None,
        additional_instruction="Accept equivalent guidance such as 'use existing campsites', 'camp on previously impacted areas', or 'avoid creating new fire rings or campsites'."
    )

    # Motorized access: source existence
    motor_src_exists = evaluator.add_custom_node(
        result=bool(bsec.motor_access_sources),
        id="Motorized_Access_Sources_Provided",
        desc="Sources are provided to support Intermountain Region motorized access distance rules",
        parent=blm_node,
        critical=True
    )

    # Leaf: Motorized_Access_Distance
    motor_leaf = evaluator.add_leaf(
        id="Motorized_Access_Distance",
        desc="Correctly states that in the Intermountain Region, motorized vehicle use for dispersed camping is only allowed within 150 feet of designated routes",
        parent=blm_node,
        critical=True
    )
    motor_claim = "In the BLM Intermountain Region, motorized vehicle travel or parking for dispersed camping is only allowed within 150 feet of designated routes."
    await evaluator.verify(
        claim=motor_claim,
        node=motor_leaf,
        sources=bsec.motor_access_sources if bsec.motor_access_sources else None,
        additional_instruction="Accept phrasing like 'within 150 feet of designated roads/routes' or 'parking within 150 feet of the route'."
    )


async def verify_wildlife_safety_distances(evaluator: Evaluator, parent_node, data: TripGuideExtraction) -> None:
    """
    Build and verify the 'Wildlife_Safety_Distances' subtree (parallel).
    """
    wsec = data.wildlife_section or WildlifeSection()

    wildlife_node = evaluator.add_parallel(
        id="Wildlife_Safety_Distances",
        desc="Correctly specifies minimum safe viewing distances for different types of wildlife",
        parent=parent_node,
        critical=True
    )

    # General wildlife: source existence
    general_src_exists = evaluator.add_custom_node(
        result=bool(wsec.general_distance_sources),
        id="General_Wildlife_Sources_Provided",
        desc="Sources are provided to support general wildlife viewing distance",
        parent=wildlife_node,
        critical=True
    )

    # Leaf: General_Wildlife_Distance
    general_leaf = evaluator.add_leaf(
        id="General_Wildlife_Distance",
        desc="Correctly states that visitors must maintain at least 25 yards distance from most wildlife (such as deer and elk)",
        parent=wildlife_node,
        critical=True
    )
    general_claim = "Visitors must stay at least 25 yards (23 meters) from most wildlife such as deer and elk."
    await evaluator.verify(
        claim=general_claim,
        node=general_leaf,
        sources=wsec.general_distance_sources if wsec.general_distance_sources else None,
        additional_instruction="Accept equivalent phrasing from NPS or park guidelines. Minor unit conversions (yards/meters) are acceptable."
    )

    # Bears/wolves: source existence
    bear_src_exists = evaluator.add_custom_node(
        result=bool(wsec.bear_distance_sources),
        id="Bear_Wolf_Sources_Provided",
        desc="Sources are provided to support bear/wolf viewing distances",
        parent=wildlife_node,
        critical=True
    )

    # Leaf: Bear_Distance
    bear_leaf = evaluator.add_leaf(
        id="Bear_Distance",
        desc="Correctly states that visitors must maintain at least 100 yards distance from bears and wolves",
        parent=wildlife_node,
        critical=True
    )
    bear_claim = "Visitors must stay at least 100 yards (91 meters) from bears and wolves."
    await evaluator.verify(
        claim=bear_claim,
        node=bear_leaf,
        sources=wsec.bear_distance_sources if wsec.bear_distance_sources else None,
        additional_instruction="Accept equivalent phrasing from NPS or park guidelines (e.g., 'about a football field'). Minor unit conversions are acceptable."
    )


async def verify_lnt_camping_surfaces(evaluator: Evaluator, parent_node, data: TripGuideExtraction) -> None:
    """
    Build and verify the 'Leave_No_Trace_Camping_Surfaces' subtree (parallel).
    """
    lsec = data.lnt_section or LNTSection()

    lnt_node = evaluator.add_parallel(
        id="Leave_No_Trace_Camping_Surfaces",
        desc="Correctly identifies appropriate durable surfaces for camping under Leave No Trace Principle 2",
        parent=parent_node,
        critical=True
    )

    # Durable surfaces: source existence
    surfaces_src_exists = evaluator.add_custom_node(
        result=bool(lsec.surfaces_sources),
        id="Durable_Surfaces_Sources_Provided",
        desc="Sources are provided to support Leave No Trace durable surfaces guidance",
        parent=lnt_node,
        critical=True
    )

    # Leaf: Identify_Durable_Surfaces
    surfaces_leaf = evaluator.add_leaf(
        id="Identify_Durable_Surfaces",
        desc="Correctly identifies at least three types of durable surfaces from: rock, gravel, dry grasses, sand, snow, or established trails and campsites",
        parent=lnt_node,
        critical=True
    )
    listed = lsec.durable_surfaces_list or []
    listed_preview = ", ".join(listed[:6]) if listed else "none"
    surfaces_claim = (
        "Under Leave No Trace Principle 2, durable surfaces include examples such as rock, gravel, dry grasses, sand, snow, "
        "and established trails/campsites. The answer identifies at least three of these durable surfaces: "
        f"{listed_preview}."
    )
    await evaluator.verify(
        claim=surfaces_claim,
        node=surfaces_leaf,
        sources=lsec.surfaces_sources if lsec.surfaces_sources else None,
        additional_instruction="Accept reasonable synonyms (e.g., 'dry grass' for 'dry grasses'; 'established sites' for 'established campsites'). The durable-surface list must be supported by the cited Leave No Trace guidance."
    )

    # Prioritize existing sites: source existence
    priority_src_exists = evaluator.add_custom_node(
        result=bool(lsec.priority_sources),
        id="Prioritize_Existing_Sites_Sources_Provided",
        desc="Sources are provided to support the 'prioritize existing sites' guidance",
        parent=lnt_node,
        critical=True
    )

    # Leaf: Prioritize_Existing_Sites
    priority_leaf = evaluator.add_leaf(
        id="Prioritize_Existing_Sites",
        desc="Correctly notes that using existing established campsites is preferred to minimize environmental impact",
        parent=lnt_node,
        critical=True
    )
    priority_claim = (
        "Using existing established campsites is preferred to minimize environmental impact, consistent with Leave No Trace guidance."
    )
    await evaluator.verify(
        claim=priority_claim,
        node=priority_leaf,
        sources=lsec.priority_sources if lsec.priority_sources else None,
        additional_instruction="Accept equivalent phrasing such as 'use existing campsites where possible' or 'concentrate use on durable surfaces'."
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
    Evaluate an answer for the comprehensive outdoor recreation trip planning task.
    """
    # Initialize evaluator and root
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_trip_guide(),
        template_class=TripGuideExtraction,
        extraction_name="trip_guide_extraction"
    )

    # Add ground truth references (for transparency in the summary)
    evaluator.add_ground_truth({
        "expected_parks_count": 5,
        "per_vehicle_fee": "$35",
        "expected_total_fees": "$175",
        "annual_pass_cost": "$80 (standard America the Beautiful Annual Pass)",
        "blm_stay_limit": "14 days within any 28-day period",
        "blm_unattended_property": "No more than 10 days",
        "motorized_access_intermountain": "Within 150 feet of designated routes",
        "wildlife_distances": {
            "general": "25 yards (23 meters) from most wildlife such as deer and elk",
            "bears_wolves": "100 yards (91 meters) from bears and wolves"
        },
        "lnt_durable_surfaces_examples": ["rock", "gravel", "dry grasses", "sand", "snow", "established trails/campsites"]
    }, gt_type="ground_truth")

    # Build Trip Planning Guide root node (critical parallel aggregator)
    guide_node = evaluator.add_parallel(
        id="Trip_Planning_Guide",
        desc="Comprehensive evaluation of outdoor recreation trip planning compliance across all required dimensions",
        parent=root,
        critical=True
    )

    # Subtrees
    await verify_pass_cost_effectiveness(evaluator, guide_node, extracted)
    await verify_dispersed_camping_compliance(evaluator, guide_node, extracted)
    await verify_wildlife_safety_distances(evaluator, guide_node, extracted)
    await verify_lnt_camping_surfaces(evaluator, guide_node, extracted)

    # Return summary
    return evaluator.get_summary()