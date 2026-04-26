import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "eclipse_2024_trip_planning"
TASK_DESCRIPTION = (
    "On April 8, 2024, a total solar eclipse crossed over the United States. Only two national parks were in the complete path of totality for this eclipse. "
    "For a visitor planning to experience this eclipse: "
    "1. What is the name of the national park in Ohio that was in the path of totality? "
    "2. What is the name of the national park in Arkansas that was in the path of totality? "
    "3. What is the name of the campground located within the boundaries of the Arkansas national park? "
    "4. How many campsites does this campground have? "
    "5. What are the operating hours for the Boston Mill Visitor Center at the Ohio national park during the months of March through December? "
    "6. What is the name of the wheelchair-accessible trail that runs through the Ohio national park? "
    "7. As of January 1, 2026, what is the cost of the America the Beautiful Annual Pass for U.S. residents?"
)

RUBRIC_ROOT_DESC = "Complete and accurate planning information for visiting national parks during the April 8, 2024 solar eclipse"

# Ground truth expectations (for reporting; verification relies on sources and simple checks)
EXPECTED_OHIO_PARK = "Cuyahoga Valley National Park"
EXPECTED_ARKANSAS_PARK = "Hot Springs National Park"
EXPECTED_CAMPGROUND = "Gulpha Gorge Campground"
EXPECTED_CAMPSITE_COUNT = "44"
EXPECTED_BOSTON_MILL_HOURS_PHRASE = "9:30 AM to 5:00 PM daily during March through December"
EXPECTED_ACCESSIBLE_TRAIL = "Ohio & Erie Canal Towpath Trail"
EXPECTED_ANNUAL_PASS_COST = "$80"

APRIL_8_2024 = "April 8, 2024"

# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class EclipsePlanningExtraction(BaseModel):
    ohio_park: Optional[str] = None
    ohio_park_sources: List[str] = Field(default_factory=list)

    arkansas_park: Optional[str] = None
    arkansas_park_sources: List[str] = Field(default_factory=list)

    campground_name: Optional[str] = None
    campground_sources: List[str] = Field(default_factory=list)

    campsite_count: Optional[str] = None
    campsite_sources: List[str] = Field(default_factory=list)

    boston_mill_hours: Optional[str] = None
    boston_mill_hours_sources: List[str] = Field(default_factory=list)

    accessible_trail_name: Optional[str] = None
    accessible_trail_sources: List[str] = Field(default_factory=list)

    annual_pass_cost: Optional[str] = None
    annual_pass_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_planning_info() -> str:
    return """
Extract the requested planning information exactly as stated in the answer and also extract the URLs the answer cites as sources for each item. Return the following JSON fields:

1) ohio_park: The name of the national park in Ohio that the answer claims was in the path of totality for the April 8, 2024 eclipse.
   ohio_park_sources: An array of URLs explicitly cited in the answer as sources supporting that Ohio park claim.

2) arkansas_park: The name of the national park in Arkansas that the answer claims was in the path of totality for the April 8, 2024 eclipse.
   arkansas_park_sources: An array of URLs explicitly cited in the answer as sources supporting that Arkansas park claim.

3) campground_name: The name of the campground the answer claims is within the boundaries of the Arkansas national park.
   campground_sources: An array of URLs explicitly cited as sources supporting that campground location/identity claim.

4) campsite_count: The number of campsites at that campground as stated in the answer (return it as a string exactly as written, e.g., '44', '44 sites', etc.).
   campsite_sources: An array of URLs explicitly cited as sources supporting the campsite count.

5) boston_mill_hours: The operating hours for the Boston Mill Visitor Center (Ohio park) for March through December, exactly as written in the answer (e.g., '9:30 AM to 5:00 PM daily').
   boston_mill_hours_sources: An array of URLs explicitly cited as sources supporting those hours.

6) accessible_trail_name: The name of the wheelchair-accessible trail that runs through the Ohio national park, exactly as written in the answer.
   accessible_trail_sources: An array of URLs explicitly cited as sources supporting the accessibility of that trail.

7) annual_pass_cost: The cost of the America the Beautiful Annual Pass for U.S. residents as of January 1, 2026, exactly as written in the answer (e.g., '$80' or '80 USD').
   annual_pass_sources: An array of URLs explicitly cited as sources supporting that 2026 price.

Rules:
- Extract only what appears in the answer. If any requested item is missing, set its value to null and its sources to an empty array.
- For sources, extract only actual URLs explicitly present in the answer (plain links or markdown links). Do not invent or infer URLs.
- Do not normalize or transform the extracted text values; preserve the wording from the answer.
"""


# --------------------------------------------------------------------------- #
# Helper: add a standard trio of checks for a claimed value                   #
# --------------------------------------------------------------------------- #
async def add_value_checks(
    evaluator: Evaluator,
    parent,
    id_base: str,
    existence_desc: str,
    provided_value: Optional[str],
    provided_sources: List[str],
    expected_value_desc: str,
    expected_value: str,
    support_desc: str,
    support_claim: str,
    match_instruction: str,
    support_instruction: str,
):
    # 1) Existence + source presence (critical)
    evaluator.add_custom_node(
        result=bool(provided_value and str(provided_value).strip()) and bool(provided_sources),
        id=f"{id_base}_provided",
        desc=existence_desc,
        parent=parent,
        critical=True,
    )

    # 2) Match to expected canonical value (critical, simple logical check)
    match_node = evaluator.add_leaf(
        id=f"{id_base}_match",
        desc=expected_value_desc,
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided value '{provided_value}' refers to the same entity or value as '{expected_value}'.",
        node=match_node,
        additional_instruction=match_instruction,
    )

    # 3) Supported by cited sources (critical, URL-grounded)
    support_node = evaluator.add_leaf(
        id=f"{id_base}_supported",
        desc=support_desc,
        parent=parent,
        critical=True,
    )
    await evaluator.verify(
        claim=support_claim,
        node=support_node,
        sources=provided_sources,
        additional_instruction=support_instruction,
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
    # Initialize evaluator (root is non-critical to allow partial credit across items)
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

    # Add a top-level planning node to mirror rubric
    planning_node = evaluator.add_parallel(
        id="eclipse_trip_planning",
        desc=RUBRIC_ROOT_DESC,
        parent=root,
        critical=False,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_planning_info(),
        template_class=EclipsePlanningExtraction,
        extraction_name="planning_info",
    )

    # Add ground truth info (for transparency in the final report)
    evaluator.add_ground_truth(
        {
            "expected_ohio_park": EXPECTED_OHIO_PARK,
            "expected_arkansas_park": EXPECTED_ARKANSAS_PARK,
            "expected_campground": EXPECTED_CAMPGROUND,
            "expected_campsite_count": EXPECTED_CAMPSITE_COUNT,
            "expected_boston_mill_hours": EXPECTED_BOSTON_MILL_HOURS_PHRASE,
            "expected_accessible_trail": EXPECTED_ACCESSIBLE_TRAIL,
            "expected_annual_pass_cost": EXPECTED_ANNUAL_PASS_COST,
            "eclipse_date": APRIL_8_2024,
        },
        gt_type="ground_truth",
    )

    # ----------------------------- Ohio park --------------------------------
    ohio_node = evaluator.add_parallel(
        id="ohio_park_identification",
        desc="Correctly identifies Cuyahoga Valley National Park as the national park in Ohio that was in the path of totality",
        parent=planning_node,
        critical=False,
    )
    await add_value_checks(
        evaluator=evaluator,
        parent=ohio_node,
        id_base="ohio_park",
        existence_desc="Ohio national park name is provided with at least one source",
        provided_value=extracted.ohio_park,
        provided_sources=extracted.ohio_park_sources,
        expected_value_desc="Provided Ohio park matches 'Cuyahoga Valley National Park'",
        expected_value=EXPECTED_OHIO_PARK,
        support_desc="Source(s) support that Cuyahoga Valley National Park was in the path of totality on April 8, 2024",
        support_claim=f"{EXPECTED_OHIO_PARK} was in the path of totality for the {APRIL_8_2024} total solar eclipse.",
        match_instruction="Consider common abbreviations and minor formatting (e.g., 'CVNP') as matching the full official name.",
        support_instruction="Verify that the cited page(s) explicitly indicate CVNP was in or experienced totality during the April 8, 2024 eclipse (maps, NPS notices, NASA pages, etc.).",
    )

    # --------------------------- Arkansas park ------------------------------
    ar_node = evaluator.add_parallel(
        id="arkansas_park_identification",
        desc="Correctly identifies Hot Springs National Park as the national park in Arkansas that was in the path of totality",
        parent=planning_node,
        critical=False,
    )
    await add_value_checks(
        evaluator=evaluator,
        parent=ar_node,
        id_base="arkansas_park",
        existence_desc="Arkansas national park name is provided with at least one source",
        provided_value=extracted.arkansas_park,
        provided_sources=extracted.arkansas_park_sources,
        expected_value_desc="Provided Arkansas park matches 'Hot Springs National Park'",
        expected_value=EXPECTED_ARKANSAS_PARK,
        support_desc="Source(s) support that Hot Springs National Park was in the path of totality on April 8, 2024",
        support_claim=f"{EXPECTED_ARKANSAS_PARK} was in the path of totality for the {APRIL_8_2024} total solar eclipse.",
        match_instruction="Allow minor variations like 'Hot Springs NP'; treat them as matching the official name.",
        support_instruction="Verify that the cited page(s) explicitly indicate Hot Springs National Park was in or experienced totality during the April 8, 2024 eclipse.",
    )

    # ---------------------- Campground within HSNP --------------------------
    cg_node = evaluator.add_parallel(
        id="hot_springs_campground",
        desc="Correctly identifies Gulpha Gorge as the campground within Hot Springs National Park boundaries",
        parent=planning_node,
        critical=False,
    )
    await add_value_checks(
        evaluator=evaluator,
        parent=cg_node,
        id_base="campground",
        existence_desc="Campground name is provided with at least one source",
        provided_value=extracted.campground_name,
        provided_sources=extracted.campground_sources,
        expected_value_desc="Provided campground matches 'Gulpha Gorge Campground' (or 'Gulpha Gorge')",
        expected_value=EXPECTED_CAMPGROUND,
        support_desc="Source(s) support that Gulpha Gorge Campground is located within Hot Springs National Park",
        support_claim=f"{EXPECTED_CAMPGROUND} is located within the boundaries of {EXPECTED_ARKANSAS_PARK}.",
        match_instruction="Treat 'Gulpha Gorge' and 'Gulpha Gorge Campground' as the same campground; allow minor formatting differences.",
        support_instruction="Verify that the cited page(s) indicate Gulpha Gorge Campground is an official campground within Hot Springs National Park.",
    )

    # ------------------------ Campsite capacity (44) ------------------------
    cap_node = evaluator.add_parallel(
        id="campsite_capacity",
        desc="Correctly states that Gulpha Gorge Campground has 44 campsites",
        parent=planning_node,
        critical=False,
    )
    # Existence + sources
    evaluator.add_custom_node(
        result=bool(extracted.campsite_count and str(extracted.campsite_count).strip()) and bool(extracted.campsite_sources),
        id="campsite_capacity_provided",
        desc="Campsite count is provided with at least one source",
        parent=cap_node,
        critical=True,
    )
    # Match to expected count (simple verify)
    cap_match = evaluator.add_leaf(
        id="campsite_capacity_match",
        desc="Provided campsite count matches '44' (allow phrasings like '44 sites')",
        parent=cap_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The provided campsite count '{extracted.campsite_count}' indicates there are 44 campsites at Gulpha Gorge Campground.",
        node=cap_match,
        additional_instruction="Accept as matching if the number 44 is clearly present (e.g., '44', '44 sites', '44 campsites'). Ignore minor wording.",
    )
    # Supported by sources
    cap_support = evaluator.add_leaf(
        id="campsite_capacity_supported",
        desc="Source(s) support that Gulpha Gorge Campground has 44 campsites",
        parent=cap_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"{EXPECTED_CAMPGROUND} has 44 campsites.",
        node=cap_support,
        sources=extracted.campsite_sources,
        additional_instruction="Confirm that the cited page(s) explicitly state the total number of sites as 44.",
    )

    # -------------------- Boston Mill Visitor Center hours ------------------
    hours_node = evaluator.add_parallel(
        id="boston_mill_hours",
        desc="Correctly provides Boston Mill Visitor Center operating hours as 9:30 AM to 5:00 PM daily during March through December",
        parent=planning_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=bool(extracted.boston_mill_hours and str(extracted.boston_mill_hours).strip()) and bool(extracted.boston_mill_hours_sources),
        id="boston_mill_hours_provided",
        desc="Boston Mill Visitor Center hours (Mar–Dec) are provided with at least one source",
        parent=hours_node,
        critical=True,
    )
    hours_match = evaluator.add_leaf(
        id="boston_mill_hours_match",
        desc="Provided hours correspond to '9:30 AM to 5:00 PM daily' for March through December",
        parent=hours_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stated Boston Mill hours '{extracted.boston_mill_hours}' match '9:30 AM to 5:00 PM daily' for March through December.",
        node=hours_match,
        additional_instruction="Allow minor formatting variations (e.g., '9:30AM–5:00PM', 'daily 9:30 to 5'). Ensure the months March–December are covered.",
    )
    hours_support = evaluator.add_leaf(
        id="boston_mill_hours_supported",
        desc="Source(s) support that Boston Mill Visitor Center is open 9:30 AM–5:00 PM daily during March–December",
        parent=hours_node,
        critical=True,
    )
    await evaluator.verify(
        claim="During March through December, the Boston Mill Visitor Center is open daily from 9:30 AM to 5:00 PM.",
        node=hours_support,
        sources=extracted.boston_mill_hours_sources,
        additional_instruction="Confirm the seasonal schedule on the official park site or authoritative source that clearly states these hours for Mar–Dec.",
    )

    # ------------------- Wheelchair-accessible trail (Towpath) -------------
    trail_node = evaluator.add_parallel(
        id="towpath_accessibility",
        desc="Correctly identifies the Ohio & Erie Canal Towpath Trail as wheelchair accessible",
        parent=planning_node,
        critical=False,
    )
    await add_value_checks(
        evaluator=evaluator,
        parent=trail_node,
        id_base="accessible_trail",
        existence_desc="Wheelchair-accessible trail name is provided with at least one source",
        provided_value=extracted.accessible_trail_name,
        provided_sources=extracted.accessible_trail_sources,
        expected_value_desc="Provided trail matches 'Ohio & Erie Canal Towpath Trail'",
        expected_value=EXPECTED_ACCESSIBLE_TRAIL,
        support_desc="Source(s) support that the Ohio & Erie Canal Towpath Trail is wheelchair accessible (at least key sections)",
        support_claim=f"The {EXPECTED_ACCESSIBLE_TRAIL} in Cuyahoga Valley National Park is wheelchair-accessible (fully or in significant accessible segments).",
        match_instruction="Allow minor naming variations (e.g., 'Towpath Trail', 'Ohio and Erie Canal Towpath').",
        support_instruction="Confirm the cited page(s) explicitly mention accessibility or wheelchair access for the Towpath Trail within the park.",
    )

    # -------------------- America the Beautiful Annual Pass -----------------
    pass_node = evaluator.add_parallel(
        id="annual_pass_cost_2026",
        desc="Correctly states the America the Beautiful Annual Pass cost as $80 for U.S. residents effective January 1, 2026",
        parent=planning_node,
        critical=False,
    )
    evaluator.add_custom_node(
        result=bool(extracted.annual_pass_cost and str(extracted.annual_pass_cost).strip()) and bool(extracted.annual_pass_sources),
        id="annual_pass_cost_provided",
        desc="Annual Pass price is provided with at least one source",
        parent=pass_node,
        critical=True,
    )
    pass_match = evaluator.add_leaf(
        id="annual_pass_cost_match",
        desc="Provided price matches '$80' (e.g., '$80', '80 USD')",
        parent=pass_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The stated annual pass price '{extracted.annual_pass_cost}' is equivalent to $80.",
        node=pass_match,
        additional_instruction="Accept formats like '$80', '80 USD', '$80.00'. Focus on the numeric value 80.",
    )
    pass_support = evaluator.add_leaf(
        id="annual_pass_cost_supported",
        desc="Source(s) support that the America the Beautiful Annual Pass costs $80 (as of Jan 1, 2026)",
        parent=pass_node,
        critical=True,
    )
    await evaluator.verify(
        claim="The America the Beautiful Annual Pass price is $80 (as of January 1, 2026).",
        node=pass_support,
        sources=extracted.annual_pass_sources,
        additional_instruction="Confirm the listed price on an official or authoritative source (e.g., NPS/USGS). If a page lists the current price as $80 and is contemporaneous with 2025–2026, treat it as valid for Jan 1, 2026.",
    )

    # Return summary
    return evaluator.get_summary()