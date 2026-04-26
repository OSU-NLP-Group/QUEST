import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nissan_stadium_research"
TASK_DESCRIPTION = """
Country music artist Lainey Wilson is scheduled to perform at Nissan Stadium in Nashville, Tennessee on May 23, 2026 as part of her Whirlwind World Tour. Research this venue and provide the following information: (1) The name of the professional football team that calls Nissan Stadium their home stadium, (2) The approximate seating capacity of the current Nissan Stadium, (3) The planned seating capacity of the new replacement stadium currently under construction adjacent to the current facility, (4) The expected completion year for the new stadium, and (5) The location of Nissan Stadium relative to downtown Nashville.
"""

# Ground truth targets for verification
GROUND_TRUTH = {
    "home_team": "Tennessee Titans",
    "current_capacity_approx": "69,143",
    "new_capacity_approx": "60,000",
    "completion_year": "2027",
    "location_relative": "on the east bank of the Cumberland River, across from downtown Nashville",
}


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class TeamInfo(BaseModel):
    team_name: Optional[str] = None
    team_urls: List[str] = Field(default_factory=list)


class CapacityInfo(BaseModel):
    capacity_value_text: Optional[str] = None
    capacity_urls: List[str] = Field(default_factory=list)


class LocationInfo(BaseModel):
    location_description: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)


class NewCapacityInfo(BaseModel):
    new_capacity_value_text: Optional[str] = None
    new_capacity_urls: List[str] = Field(default_factory=list)


class TimelineInfo(BaseModel):
    completion_year_text: Optional[str] = None
    timeline_urls: List[str] = Field(default_factory=list)


class NissanStadiumExtraction(BaseModel):
    team: Optional[TeamInfo] = None
    current_capacity: Optional[CapacityInfo] = None
    location: Optional[LocationInfo] = None
    new_capacity: Optional[NewCapacityInfo] = None
    timeline: Optional[TimelineInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nissan_stadium_info() -> str:
    return """
    Extract structured information about Nissan Stadium in Nashville, Tennessee as presented in the answer text.
    You must only extract details explicitly mentioned in the answer and the URLs the answer cites.

    Extract the following fields:

    1) team:
       - team_name: the professional football team specified as calling Nissan Stadium their home stadium.
       - team_urls: all URLs cited that support the home team identification (list of full URLs).

    2) current_capacity:
       - capacity_value_text: the seating capacity of the current Nissan Stadium, as a text string (e.g., "69,143", "about 69,000").
       - capacity_urls: all URLs cited that support the capacity figure (list of full URLs).

    3) location:
       - location_description: the stadium’s location relative to downtown Nashville, as a text string (e.g., "on the east bank of the Cumberland River, across from downtown Nashville").
       - location_urls: all URLs cited that support the location description (list of full URLs).

    4) new_capacity:
       - new_capacity_value_text: the planned seating capacity of the new replacement stadium adjacent to the current facility, as a text string (e.g., "approximately 60,000").
       - new_capacity_urls: all URLs cited that support the planned capacity (list of full URLs).

    5) timeline:
       - completion_year_text: the expected completion year for the new stadium (e.g., "2027").
       - timeline_urls: all URLs cited that support the completion timeline (list of full URLs).

    Rules:
    - Return null for any field not mentioned in the answer.
    - Only include URLs explicitly present in the answer. Do not invent URLs.
    - URLs may be plain or inside markdown; extract the actual URL string.
    - Keep values as strings; do not convert to numbers.
    """


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_current_stadium_information(
    evaluator: Evaluator,
    parent_node,
    extracted: NissanStadiumExtraction,
) -> None:
    # Create "Current_Stadium_Information" node
    current_node = evaluator.add_parallel(
        id="Current_Stadium_Information",
        desc="Information about the current Nissan Stadium must be provided",
        parent=parent_node,
        critical=False
    )

    # ---------------- Home Team Details ----------------
    home_team_node = evaluator.add_parallel(
        id="Home_Team_Details",
        desc="The professional football team that calls Nissan Stadium home must be identified",
        parent=current_node,
        critical=False
    )

    team_name = extracted.team.team_name if extracted.team else None
    team_urls = extracted.team.team_urls if extracted.team else []

    # Team_Name: simple match check against ground truth
    team_name_leaf = evaluator.add_leaf(
        id="Team_Name",
        desc="The team name must be Tennessee Titans",
        parent=home_team_node,
        critical=True
    )
    team_match_claim = f"'{team_name}' and '{GROUND_TRUTH['home_team']}' refer to the same NFL team."
    await evaluator.verify(
        claim=team_match_claim,
        node=team_name_leaf,
        additional_instruction="Evaluate whether the extracted team name refers to the Tennessee Titans. Allow minor variations like 'Titans' or letter casing."
    )

    # Team_Reference_URL: existence check (critical)
    evaluator.add_custom_node(
        result=bool(team_urls),
        id="Team_Reference_URL",
        desc="A reference URL supporting the home team identification must be provided",
        parent=home_team_node,
        critical=True
    )

    # Additional leaf to verify source support for home team
    team_support_leaf = evaluator.add_leaf(
        id="Team_Source_Support",
        desc="Home team identification is supported by cited sources",
        parent=home_team_node,
        critical=True
    )
    team_support_claim = "Nissan Stadium is the home stadium of the Tennessee Titans."
    await evaluator.verify(
        claim=team_support_claim,
        node=team_support_leaf,
        sources=team_urls,
        additional_instruction="Verify the provided source(s) explicitly state or clearly imply that Nissan Stadium is the home stadium of the Tennessee Titans. If sources are irrelevant or inaccessible, mark as not supported."
    )

    # ---------------- Capacity Details ----------------
    capacity_node = evaluator.add_parallel(
        id="Capacity_Details",
        desc="The seating capacity of the current stadium must be provided",
        parent=current_node,
        critical=False
    )

    capacity_text = extracted.current_capacity.capacity_value_text if extracted.current_capacity else None
    capacity_urls = extracted.current_capacity.capacity_urls if extracted.current_capacity else []

    capacity_value_leaf = evaluator.add_leaf(
        id="Capacity_Value",
        desc="The capacity must be approximately 69,143 seats",
        parent=capacity_node,
        critical=True
    )
    capacity_claim = "The seating capacity of the current Nissan Stadium is approximately 69,143."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_value_leaf,
        sources=capacity_urls,
        additional_instruction="Check the page(s) for Nissan Stadium's seating capacity. Accept reasonable rounding (e.g., ~69,000) or the precise figure 69,143; consider variations within ±1,000 as approximate equivalence."
    )

    evaluator.add_custom_node(
        result=bool(capacity_urls),
        id="Capacity_Reference_URL",
        desc="A reference URL supporting the capacity figure must be provided",
        parent=capacity_node,
        critical=True
    )

    capacity_support_leaf = evaluator.add_leaf(
        id="Capacity_Source_Support",
        desc="Current stadium capacity is supported by cited sources",
        parent=capacity_node,
        critical=True
    )
    capacity_support_claim = "Nissan Stadium's listed seating capacity is around 69,143 seats."
    await evaluator.verify(
        claim=capacity_support_claim,
        node=capacity_support_leaf,
        sources=capacity_urls,
        additional_instruction="Verify the capacity figure on the provided source(s). Accept approximate wording such as 'about 69,000' if consistent with 69,143."
    )

    # ---------------- Location Details ----------------
    location_node = evaluator.add_parallel(
        id="Location_Details",
        desc="The location of the stadium relative to downtown Nashville must be described",
        parent=current_node,
        critical=False
    )

    location_text = extracted.location.location_description if extracted.location else None
    location_urls = extracted.location.location_urls if extracted.location else []

    location_desc_leaf = evaluator.add_leaf(
        id="Location_Description",
        desc="The stadium location must be described as on the east bank of the Cumberland River, across from downtown Nashville",
        parent=location_node,
        critical=True
    )
    location_claim = "Nissan Stadium is on the east bank of the Cumberland River, across from downtown Nashville."
    await evaluator.verify(
        claim=location_claim,
        node=location_desc_leaf,
        sources=location_urls,
        additional_instruction="Verify that the source(s) describe Nissan Stadium's location relative to downtown Nashville—specifically, on the east bank of the Cumberland River across from downtown."
    )

    evaluator.add_custom_node(
        result=bool(location_urls),
        id="Location_Reference_URL",
        desc="A reference URL supporting the location description must be provided",
        parent=location_node,
        critical=True
    )

    location_support_leaf = evaluator.add_leaf(
        id="Location_Source_Support",
        desc="Stadium location relative to downtown Nashville is supported by cited sources",
        parent=location_node,
        critical=True
    )
    location_support_claim = "Nissan Stadium sits on Nashville's east bank of the Cumberland River, opposite downtown."
    await evaluator.verify(
        claim=location_support_claim,
        node=location_support_leaf,
        sources=location_urls,
        additional_instruction="Confirm the location phrasing or equivalent wording on the provided source(s): e.g., 'east bank', 'across the river from downtown', 'across from downtown Nashville'."
    )


async def build_new_stadium_information(
    evaluator: Evaluator,
    parent_node,
    extracted: NissanStadiumExtraction,
) -> None:
    # Create "New_Stadium_Information" node
    new_node = evaluator.add_parallel(
        id="New_Stadium_Information",
        desc="Information about the new replacement stadium currently under construction must be provided",
        parent=parent_node,
        critical=False
    )

    # ---------------- Capacity Planning ----------------
    new_capacity_node = evaluator.add_parallel(
        id="Capacity_Planning",
        desc="The planned capacity of the new stadium must be provided",
        parent=new_node,
        critical=False
    )

    new_capacity_text = extracted.new_capacity.new_capacity_value_text if extracted.new_capacity else None
    new_capacity_urls = extracted.new_capacity.new_capacity_urls if extracted.new_capacity else []

    new_capacity_leaf = evaluator.add_leaf(
        id="New_Capacity_Value",
        desc="The new stadium capacity must be approximately 60,000 seats",
        parent=new_capacity_node,
        critical=True
    )
    new_capacity_claim = "The planned seating capacity of the new Nissan Stadium replacement is approximately 60,000 seats."
    await evaluator.verify(
        claim=new_capacity_claim,
        node=new_capacity_leaf,
        sources=new_capacity_urls,
        additional_instruction="Verify that the provided source(s) indicate a planned capacity near 60,000 seats (e.g., 60,000–62,000). Minor rounding or ranges are acceptable."
    )

    evaluator.add_custom_node(
        result=bool(new_capacity_urls),
        id="New_Capacity_Reference_URL",
        desc="A reference URL supporting the new stadium capacity must be provided",
        parent=new_capacity_node,
        critical=True
    )

    new_capacity_support_leaf = evaluator.add_leaf(
        id="New_Capacity_Source_Support",
        desc="New stadium capacity is supported by cited sources",
        parent=new_capacity_node,
        critical=True
    )
    new_capacity_support_claim = "The new enclosed stadium adjacent to the current facility is planned at about 60,000 seats."
    await evaluator.verify(
        claim=new_capacity_support_claim,
        node=new_capacity_support_leaf,
        sources=new_capacity_urls,
        additional_instruction="Confirm the planned capacity figure or range on the provided source(s). Accept approximate wording consistent with ~60,000."
    )

    # ---------------- Construction Timeline ----------------
    timeline_node = evaluator.add_parallel(
        id="Construction_Timeline",
        desc="The expected completion year for the new stadium must be provided",
        parent=new_node,
        critical=False
    )

    completion_text = extracted.timeline.completion_year_text if extracted.timeline else None
    timeline_urls = extracted.timeline.timeline_urls if extracted.timeline else []

    completion_leaf = evaluator.add_leaf(
        id="Completion_Year",
        desc="The completion year must be 2027",
        parent=timeline_node,
        critical=True
    )
    completion_claim = "The expected completion year for the new Nissan Stadium is 2027."
    await evaluator.verify(
        claim=completion_claim,
        node=completion_leaf,
        sources=timeline_urls,
        additional_instruction="Verify any phrasing indicating completion/opening/target year as 2027 for the new stadium. Accept equivalent wording such as 'opening in 2027'."
    )

    evaluator.add_custom_node(
        result=bool(timeline_urls),
        id="Timeline_Reference_URL",
        desc="A reference URL supporting the completion timeline must be provided",
        parent=timeline_node,
        critical=True
    )

    timeline_support_leaf = evaluator.add_leaf(
        id="Timeline_Source_Support",
        desc="New stadium completion timeline is supported by cited sources",
        parent=timeline_node,
        critical=True
    )
    timeline_support_claim = "The new stadium's expected completion or opening is scheduled for 2027."
    await evaluator.verify(
        claim=timeline_support_claim,
        node=timeline_support_leaf,
        sources=timeline_urls,
        additional_instruction="Confirm on the provided source(s) that 2027 is the expected completion/opening/target year."
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
    Evaluate an answer for the Nissan Stadium research task.
    """
    # Initialize evaluator
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

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_nissan_stadium_info(),
        template_class=NissanStadiumExtraction,
        extraction_name="nissan_stadium_info",
    )

    # Record ground truth expectations for transparency
    evaluator.add_ground_truth({
        "expected_home_team": GROUND_TRUTH["home_team"],
        "expected_current_capacity_approx": GROUND_TRUTH["current_capacity_approx"],
        "expected_new_capacity_approx": GROUND_TRUTH["new_capacity_approx"],
        "expected_completion_year": GROUND_TRUTH["completion_year"],
        "expected_location_relative": GROUND_TRUTH["location_relative"],
    })

    # Create top-level research node (non-critical to allow mixed child criticalities)
    research_node = evaluator.add_parallel(
        id="Nissan_Stadium_Research",
        desc="Comprehensive research about Nissan Stadium in Nashville, Tennessee must be provided covering all requested information categories",
        parent=root,
        critical=False
    )

    # Build subtrees
    await build_current_stadium_information(evaluator, research_node, extracted)
    await build_new_stadium_information(evaluator, research_node, extracted)

    # Optional: record custom info that we adjusted top-level criticality due to framework constraints
    evaluator.add_custom_info(
        info={"note": "Top-level node set non-critical to satisfy framework constraint that critical parents must have all-critical children."},
        info_type="design_note"
    )

    # Return structured evaluation summary
    return evaluator.get_summary()