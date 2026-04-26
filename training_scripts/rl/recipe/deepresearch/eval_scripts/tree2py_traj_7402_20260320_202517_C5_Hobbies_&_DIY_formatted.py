import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "macys_2024_inflation_and_diy_plan"
TASK_DESCRIPTION = """
You are planning to attend the 2024 Macy's Thanksgiving Day Parade balloon inflation event in New York City with your family. As part of this experience, you want to create DIY parade balloon crafts as a family activity before attending the event. Develop a comprehensive plan that includes: (1) Event Logistics - When and where does the balloon inflation event take place? Specify the date, time range, entry point location, and the viewing area location. (2) DIY Craft Materials - What are the four core materials required to create DIY parade balloon crafts? What specific type of adhesive tool must be used (noting the temperature specification)? (3) Safety Requirements - What allergy-related safety consideration must be addressed before starting the craft project? If allergies are present, what alternative materials can be used? (4) Event Attendance Preparation - What is the maximum bag size allowed in spectator areas? (5) Timeline Coordination - Explain how you will coordinate the timing of completing the DIY crafts with attending the inflation event, accounting for any necessary preparation or drying time. Your answer must be based on official 2024 event information and standard DIY craft safety practices.
"""

# Expected official facts for 2024 balloon inflation event
EXPECTED_FACTS = {
    "event_date": "Wednesday, November 27, 2024",
    "event_time_range": "1:00 PM to 6:00 PM",
    "entry_point_location": "West 72nd Street and Columbus Avenue",
    "viewing_area_location": "around the American Museum of Natural History between West 77th and West 81st Streets",
    "line_open_time": "12:00 PM",
    "entry_close_time": "6:00 PM",
    "ticketing": "The event is free and no tickets are required",
    "maximum_bag_size": "12 inches × 6 inches × 12 inches",
}


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class PlanExtraction(BaseModel):
    # 1) Event Logistics (as stated in the answer)
    event_date: Optional[str] = None
    event_time_range: Optional[str] = None
    entry_point_location: Optional[str] = None
    viewing_area_location: Optional[str] = None
    line_open_time: Optional[str] = None
    entry_close_time: Optional[str] = None
    ticketing: Optional[str] = None

    # URLs explicitly cited in the answer for official 2024 event info
    official_sources: List[str] = Field(default_factory=list)

    # 2) DIY Craft Materials & Methods
    materials_list: List[str] = Field(default_factory=list)
    adhesive_tool: Optional[str] = None
    stabilization_method: Optional[str] = None

    # 3) Safety Requirements
    allergy_statement: Optional[str] = None
    latex_free_alternatives: List[str] = Field(default_factory=list)

    # 4) Event Attendance Preparation
    max_bag_size: Optional[str] = None
    clothing_recommendation: Optional[str] = None

    # 5) Timeline Coordination
    craft_completion_timing: Optional[str] = None
    drying_time_details: Optional[str] = None
    arrival_time_window: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
Extract the plan details exactly as they appear in the answer. Do not invent anything. Return null where the answer does not state the required information.

You must extract:
1) Event Logistics (the balloon inflation public viewing event for 2024 in NYC):
   - event_date: the date (e.g., "Wednesday, November 27, 2024")
   - event_time_range: the viewing hours (e.g., "1:00 PM to 6:00 PM")
   - entry_point_location: entrance location (e.g., "West 72nd Street and Columbus Avenue")
   - viewing_area_location: viewing path/area (e.g., "around the American Museum of Natural History between West 77th and West 81st Streets")
   - line_open_time: when the public viewing line opens (e.g., "12:00 PM")
   - entry_close_time: when entry closes (e.g., "6:00 PM")
   - ticketing: a short statement on whether it is free/no tickets required, as stated in the answer text.

2) Official source URLs:
   - official_sources: extract ALL URLs explicitly cited in the answer that are used as evidence for the 2024 balloon inflation event and attendee rules (e.g., Macy’s official 2024 pages, NYC/NYPD guidance). Accept URLs in plain form or markdown. Only include valid URLs; do not infer.

3) DIY Craft Materials & Methods:
   - materials_list: a list of all core materials listed in the answer for the DIY balloon crafts.
   - adhesive_tool: the adhesive tool specified (e.g., "low-temperature hot glue gun").
   - stabilization_method: how foam/balls are stabilized during painting (e.g., "mounted on toothpicks" or "set in plastic cups").

4) Safety Requirements:
   - allergy_statement: extract the exact line(s) where the plan addresses latex allergy safety before starting.
   - latex_free_alternatives: list all latex-free alternatives mentioned (e.g., "paper plates", "cardstock templates").

5) Event Attendance Preparation:
   - max_bag_size: the maximum allowed bag size as stated (e.g., "12 inches × 6 inches × 12 inches")
   - clothing_recommendation: any line recommending warm layers/clothing.

6) Timeline Coordination:
   - craft_completion_timing: when the crafts will be completed relative to the event date (e.g., "before Nov 27")
   - drying_time_details: details on acrylic paint drying time before assembly/transport.
   - arrival_time_window: the intended arrival/queue/attendance time window for the inflation event as stated in the answer.

Follow URL extraction rules strictly. Return all fields in a single JSON object.
"""


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def build_event_logistics_nodes(evaluator: Evaluator, parent_node, extracted: PlanExtraction) -> None:
    """
    Build the event logistics subtree with verification grounded by the cited official sources.
    All children here are critical.
    """
    event_node = evaluator.add_parallel(
        id="inflation_event_logistics",
        desc="Provide correct official logistics for the balloon inflation event.",
        parent=parent_node,
        critical=True
    )

    sources = extracted.official_sources or []

    # Event Date
    leaf_date = evaluator.add_leaf(
        id="event_date",
        desc="States the event occurs on Wednesday, November 27, 2024.",
        parent=event_node,
        critical=True
    )
    claim_date = "The 2024 Macy's Thanksgiving Day Parade balloon inflation public viewing event occurs on Wednesday, November 27, 2024 (Thanksgiving Eve) in New York City."
    await evaluator.verify(
        claim=claim_date,
        node=leaf_date,
        sources=sources,
        additional_instruction="Only pass if the provided webpage(s) explicitly refer to the 2024 balloon inflation public viewing date. Reject pages about other years."
    )

    # Event Time Range
    leaf_time = evaluator.add_leaf(
        id="event_time_range",
        desc="States the event runs from 1:00 PM to 6:00 PM.",
        parent=event_node,
        critical=True
    )
    claim_time = "The public viewing hours for the 2024 balloon inflation are from 1:00 PM to 6:00 PM."
    await evaluator.verify(
        claim=claim_time,
        node=leaf_time,
        sources=sources,
        additional_instruction="Accept minor formatting like '1 pm–6 pm'. Must clearly be 2024 balloon inflation viewing hours."
    )

    # Entry Point Location
    leaf_entry = evaluator.add_leaf(
        id="entry_point_location",
        desc="States the entry point is at West 72nd Street and Columbus Avenue.",
        parent=event_node,
        critical=True
    )
    claim_entry = "For the 2024 balloon inflation public viewing, the entrance is at West 72nd Street and Columbus Avenue."
    await evaluator.verify(
        claim=claim_entry,
        node=leaf_entry,
        sources=sources,
        additional_instruction="Allow minor variants like 'W 72nd St & Columbus Ave'. Must be the entrance for the 2024 balloon inflation viewing."
    )

    # Viewing Area Location
    leaf_viewing = evaluator.add_leaf(
        id="viewing_area_location",
        desc="States the viewing path is around the American Museum of Natural History between West 77th and West 81st Streets.",
        parent=event_node,
        critical=True
    )
    claim_viewing = "The viewing area/path for the 2024 balloon inflation is around the American Museum of Natural History between West 77th and West 81st Streets."
    await evaluator.verify(
        claim=claim_viewing,
        node=leaf_viewing,
        sources=sources,
        additional_instruction="Must clearly mention the AMNH and the W 77th–W 81st Streets segment for 2024 balloon inflation viewing."
    )

    # Line Open Time
    leaf_line_open = evaluator.add_leaf(
        id="line_open_time",
        desc="States the public viewing line opens at 12:00 PM (noon).",
        parent=event_node,
        critical=True
    )
    claim_line_open = "For the 2024 balloon inflation public viewing, the public viewing line opens at 12:00 PM (noon)."
    await evaluator.verify(
        claim=claim_line_open,
        node=leaf_line_open,
        sources=sources,
        additional_instruction="Must be explicitly stated for 2024 balloon inflation. Accept 'noon' for 12:00 PM."
    )

    # Entry Close Time
    leaf_entry_close = evaluator.add_leaf(
        id="entry_close_time",
        desc="States entry closes at 6:00 PM.",
        parent=event_node,
        critical=True
    )
    claim_entry_close = "For the 2024 balloon inflation public viewing, entry closes at 6:00 PM."
    await evaluator.verify(
        claim=claim_entry_close,
        node=leaf_entry_close,
        sources=sources,
        additional_instruction="Confirm that 6:00 PM is the entry close time for 2024 balloon inflation viewing."
    )

    # Ticketing
    leaf_ticketing = evaluator.add_leaf(
        id="ticketing",
        desc="States the event is free and no tickets are required.",
        parent=event_node,
        critical=True
    )
    claim_ticket = "The 2024 balloon inflation public viewing event is free and does not require tickets."
    await evaluator.verify(
        claim=claim_ticket,
        node=leaf_ticketing,
        sources=sources,
        additional_instruction="Verify the official page(s) clearly state free admission and no tickets for the 2024 balloon inflation viewing."
    )


async def build_diy_materials_methods_nodes(evaluator: Evaluator, parent_node, extracted: PlanExtraction) -> None:
    """
    DIY craft materials, adhesive tool, and stabilization method checks.
    All children here are critical and verified against the answer content (simple checks).
    """
    diy_node = evaluator.add_parallel(
        id="diy_craft_materials_and_methods",
        desc="Provide required DIY craft materials/tools and required handling/stabilization method.",
        parent=parent_node,
        critical=True
    )

    # Four core materials
    leaf_materials = evaluator.add_leaf(
        id="four_core_materials",
        desc="Lists all four core materials: styrofoam/foam balls; wooden skewers; card stock OR foam sheets; acrylic paint.",
        parent=diy_node,
        critical=True
    )
    claim_materials = (
        "The plan lists all four core materials for the DIY parade balloon crafts: "
        "(1) styrofoam or foam balls, (2) wooden skewers, (3) either card stock or foam sheets, and (4) acrylic paint."
    )
    await evaluator.verify(
        claim=claim_materials,
        node=leaf_materials,
        additional_instruction="Pass only if the answer text explicitly includes all four categories. For #3, accept either 'card stock' or 'foam sheets' (or both)."
    )

    # Adhesive tool: low-temperature hot glue gun
    leaf_adhesive = evaluator.add_leaf(
        id="adhesive_tool_requirement",
        desc="Specifies a low-temperature hot glue gun is required (not regular hot glue).",
        parent=diy_node,
        critical=True
    )
    claim_adhesive = "The plan specifies using a low-temperature hot glue gun (not a regular/high-temperature glue gun)."
    await evaluator.verify(
        claim=claim_adhesive,
        node=leaf_adhesive,
        additional_instruction="Look for explicit 'low-temperature' language tied to a hot glue gun."
    )

    # Stabilization method for painting
    leaf_stabilize = evaluator.add_leaf(
        id="painting_stabilization_method",
        desc="States foam balls must be stabilized for painting (mounted on toothpicks or set in plastic cups).",
        parent=diy_node,
        critical=True
    )
    claim_stabilize = (
        "The plan states stabilizing foam/balls for painting by either mounting them on toothpicks/skewers or setting them in plastic cups (or similar holders)."
    )
    await evaluator.verify(
        claim=claim_stabilize,
        node=leaf_stabilize,
        additional_instruction="Accept reasonable equivalents for stabilization (e.g., toothpicks, skewers, plastic cups). Must be explicit."
    )


async def build_safety_nodes(evaluator: Evaluator, parent_node, extracted: PlanExtraction) -> None:
    """
    Safety requirements related to latex allergy and alternatives.
    All children critical and verified within answer content.
    """
    safety_node = evaluator.add_parallel(
        id="safety_requirements",
        desc="Address the allergy-related safety requirement and alternatives.",
        parent=parent_node,
        critical=True
    )

    # Latex allergy consideration
    leaf_allergy = evaluator.add_leaf(
        id="latex_allergy_consideration",
        desc="Explicitly addresses latex allergy safety before starting the craft (confirm none or note risk/need to check).",
        parent=safety_node,
        critical=True
    )
    claim_allergy = (
        "Before starting the craft, the plan explicitly addresses latex allergy safety by confirming no one has a latex allergy "
        "or by noting the risk and the need to check for latex allergies."
    )
    await evaluator.verify(
        claim=claim_allergy,
        node=leaf_allergy,
        additional_instruction="Look for explicit mention of 'latex' and allergy checking prior to crafting."
    )

    # Latex-free alternatives
    leaf_alts = evaluator.add_leaf(
        id="latex_free_alternatives",
        desc="Provides latex-free alternatives to use if allergies are present (paper plates or cardstock templates).",
        parent=safety_node,
        critical=True
    )
    claim_alts = (
        "If latex allergies are present, the plan provides latex-free alternatives such as paper plates or cardstock templates."
    )
    await evaluator.verify(
        claim=claim_alts,
        node=leaf_alts,
        additional_instruction="Accept either or both: 'paper plates' and 'cardstock templates'. Must explicitly present as alternatives for allergies."
    )


async def build_attendance_rules_nodes(evaluator: Evaluator, parent_node, extracted: PlanExtraction) -> None:
    """
    Attendance preparation (critical rule) - maximum bag size, grounded by official sources.
    """
    attend_node = evaluator.add_parallel(
        id="event_attendance_preparation",
        desc="Include required spectator-area preparation constraints.",
        parent=parent_node,
        critical=True
    )

    # Maximum bag size (critical)
    leaf_bag = evaluator.add_leaf(
        id="maximum_bag_size",
        desc="States the maximum allowed bag size is 12 inches × 6 inches × 12 inches.",
        parent=attend_node,
        critical=True
    )
    claim_bag = (
        "For the 2024 balloon inflation/spectator areas, the maximum allowed bag size is 12 inches × 6 inches × 12 inches."
    )
    await evaluator.verify(
        claim=claim_bag,
        node=leaf_bag,
        sources=extracted.official_sources or [],
        additional_instruction="Verify this dimension limit on an official 2024 page (Macy’s/City/NYPD). Accept minor formatting such as 12in x 6in x 12in."
    )


async def build_timeline_nodes(evaluator: Evaluator, parent_node, extracted: PlanExtraction) -> None:
    """
    Timeline coordination checks. All children critical and verified against the answer content.
    """
    tl_node = evaluator.add_parallel(
        id="timeline_coordination",
        desc="Explain how craft completion timing is coordinated with attending the inflation event, including drying/prep time.",
        parent=parent_node,
        critical=True
    )

    # Craft completion before event date
    leaf_complete = evaluator.add_leaf(
        id="craft_completion_before_event",
        desc="Plans to complete DIY crafts before November 27 to be ready for the event.",
        parent=tl_node,
        critical=True
    )
    claim_complete = "The plan schedules completing the DIY crafts before November 27, 2024, so they are ready for the inflation event."
    await evaluator.verify(
        claim=claim_complete,
        node=leaf_complete,
        additional_instruction="Look for explicit timing that finishes the crafts before Nov 27, 2024."
    )

    # Drying time accounted for
    leaf_dry = evaluator.add_leaf(
        id="drying_time_accounted_for",
        desc="Accounts for acrylic paint drying time before assembly can begin.",
        parent=tl_node,
        critical=True
    )
    claim_dry = "The plan accounts for acrylic paint drying time before assembly/transport."
    await evaluator.verify(
        claim=claim_dry,
        node=leaf_dry,
        additional_instruction="Must mention drying time or waiting period for acrylic paint before proceeding."
    )

    # Attendance timing within official hours
    leaf_within_hours = evaluator.add_leaf(
        id="event_attendance_timing_within_official_hours",
        desc="States an intended arrival/queue or attendance time window for the inflation event that is consistent with official line open time (12:00 PM), event hours (1:00–6:00 PM), and entry closing time (6:00 PM).",
        parent=tl_node,
        critical=True
    )
    claim_within_hours = (
        "The plan states an intended arrival/queue or attendance time window for the balloon inflation that is consistent with "
        "the line opening at 12:00 PM, public viewing hours of 1:00–6:00 PM, and entry closing at 6:00 PM."
    )
    await evaluator.verify(
        claim=claim_within_hours,
        node=leaf_within_hours,
        additional_instruction="Check the times mentioned in the answer text and ensure they fall within those official windows."
    )


async def add_optional_warm_clothing_node(evaluator: Evaluator, root_node, extracted: PlanExtraction) -> None:
    """
    Optional, non-critical recommendation node for warm clothing (under root as non-critical).
    """
    leaf_warm = evaluator.add_leaf(
        id="warm_clothing_recommendation",
        desc="Recommends warm clothing for November weather in NYC.",
        parent=root_node,
        critical=False
    )
    claim_warm = "The plan recommends wearing warm clothing or layers appropriate for November weather in NYC."
    await evaluator.verify(
        claim=claim_warm,
        node=leaf_warm,
        additional_instruction="Look for explicit mention of warm clothing, layers, coats, hats, gloves, or similar."
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
    """
    Evaluate an answer for the 2024 Macy's balloon inflation planning and DIY crafts task.
    """
    # Initialize evaluator and root node
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

    # Extract structured plan details from the answer
    extracted: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction",
    )

    # Add reference ground truth of expected official facts (for transparency)
    evaluator.add_ground_truth(
        {
            "expected_official_2024_inflation_facts": EXPECTED_FACTS,
            "notes": "Used for reference when constructing verification claims; actual verification grounded by cited official URLs where applicable.",
        },
        gt_type="expected_facts",
    )

    # Build core planning subtree (critical)
    plan_core = evaluator.add_parallel(
        id="balloon_inflation_event_planning",
        desc="Plan to attend the 2024 Macy's Thanksgiving Day Parade balloon inflation event in NYC and prepare DIY parade balloon crafts, satisfying all stated constraints.",
        parent=root,
        critical=True,
    )

    # Build subtrees under the critical core
    await build_event_logistics_nodes(evaluator, plan_core, extracted)
    await build_diy_materials_methods_nodes(evaluator, plan_core, extracted)
    await build_safety_nodes(evaluator, plan_core, extracted)
    await build_attendance_rules_nodes(evaluator, plan_core, extracted)
    await build_timeline_nodes(evaluator, plan_core, extracted)

    # Add the optional non-critical recommendation node under root
    await add_optional_warm_clothing_node(evaluator, root, extracted)

    # Return evaluation summary
    return evaluator.get_summary()