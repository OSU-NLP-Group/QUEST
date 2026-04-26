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
TASK_ID = "ca_outdoor_recreation_plan"
TASK_DESCRIPTION = (
    "I'm planning a Southern California outdoor recreation trip. I want to visit the California theme park that holds "
    "the United States record for having the most roller coasters at a single amusement park. I also want to go hiking at "
    "a California State Natural Reserve in San Diego County that features coastal hiking trails with ocean views. For the "
    "hiking location, I need you to recommend a specific trail within that reserve that offers views of the Pacific Ocean "
    "or coastal bluffs. Please provide: (1) the name and location of the theme park, along with its roller coaster count; "
    "(2) the name of the State Natural Reserve; and (3) the name of a specific hiking trail within that reserve. Include "
    "reference URLs for each."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ThemeParkInfo(BaseModel):
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    roller_coaster_count: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class ReserveInfo(BaseModel):
    name: Optional[str] = None
    county: Optional[str] = None
    designation: Optional[str] = None  # Expected to include "State Natural Reserve"
    coastal_features: Optional[str] = None  # e.g., "coastal hiking trails with ocean views"
    urls: List[str] = Field(default_factory=list)


class TrailInfo(BaseModel):
    name: Optional[str] = None
    reserve_name: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class PlanExtraction(BaseModel):
    theme_park: Optional[ThemeParkInfo] = None
    reserve: Optional[ReserveInfo] = None
    trail: Optional[TrailInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan() -> str:
    return """
    Extract the key items from the answer for a Southern California outdoor recreation plan. The answer should include:
    1) A California theme park that holds the United States record for the most roller coasters at a single amusement park.
    2) A California State Natural Reserve in San Diego County that features coastal hiking trails with ocean views.
    3) A specific hiking trail within that reserve that offers views of the Pacific Ocean or coastal bluffs.

    Extract the following fields:

    theme_park:
      - name: The theme park’s official name.
      - city: The city where the park is located (e.g., Valencia or Santa Clarita).
      - state: The U.S. state (should be "California" or "CA").
      - roller_coaster_count: The reported number of roller coasters at the park, as stated in the answer.
      - urls: A list of one or more reference URLs cited in the answer that support the park’s coaster count and/or record status.

    reserve:
      - name: The specific name of the State Natural Reserve (e.g., "Torrey Pines State Natural Reserve").
      - county: The county in which it is located (should be "San Diego County" or "San Diego").
      - designation: The formal designation string (should include "State Natural Reserve").
      - coastal_features: A short phrase summarizing the coastal trail/ocean-view aspect mentioned in the answer (e.g., "coastal hiking trails with ocean views").
      - urls: A list of one or more reference URLs cited in the answer that confirm the designation and coastal trail features.

    trail:
      - name: The specific name of a hiking trail within the identified reserve that offers ocean or coastal bluff views.
      - reserve_name: The name of the reserve in which this trail is located.
      - urls: A list of one or more reference URLs cited in the answer that describe this trail and its features.

    Rules:
    - Extract ONLY what is explicitly present in the answer. Do not invent or infer facts or URLs.
    - If a field is missing in the answer, set it to null (or an empty list for urls).
    - If multiple candidates are listed, extract the one that best matches the described requirements as presented by the answer (usually the first clearly suitable one).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def combine_sources(*lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for lst in lists:
        if not lst:
            continue
        for u in lst:
            if not isinstance(u, str):
                continue
            if u not in seen:
                merged.append(u)
                seen.add(u)
    return merged


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_theme_park_checks(evaluator: Evaluator, parent_node, theme: ThemeParkInfo) -> None:
    theme_node = evaluator.add_parallel(
        id="Theme_Park_Identification",
        desc="Correctly identify the California theme park that holds the US record for most roller coasters",
        parent=parent_node,
        critical=False
    )

    # Existence/reference checks to gate downstream verification
    ref_present = evaluator.add_custom_node(
        result=bool(theme and theme.urls and len(theme.urls) > 0),
        id="Theme_Park_Reference",
        desc="Provide a reference URL that confirms the park's roller coaster count and record status",
        parent=theme_node,
        critical=True
    )
    info_present = evaluator.add_custom_node(
        result=bool(theme and theme.name and theme.name.strip()),
        id="Theme_Park_Provided",
        desc="Theme park name is provided",
        parent=theme_node,
        critical=True
    )

    sources = theme.urls if theme else []

    # Park_Location
    park_loc_node = evaluator.add_leaf(
        id="Park_Location",
        desc="The theme park is located in California",
        parent=theme_node,
        critical=True
    )
    park_loc_claim = f"The theme park {theme.name or ''} is located in the state of California."
    await evaluator.verify(
        claim=park_loc_claim,
        node=park_loc_node,
        sources=sources,
        additional_instruction="Confirm the page indicates the park is in California. Accept reasonable variants like 'CA' or a California city."
    )

    # Record_Status
    record_node = evaluator.add_leaf(
        id="Record_Status",
        desc="The theme park holds the US record for most roller coasters at a single amusement park",
        parent=theme_node,
        critical=True
    )
    record_claim = f"{theme.name or ''} holds the United States record for the most roller coasters at a single amusement park."
    await evaluator.verify(
        claim=record_claim,
        node=record_node,
        sources=sources,
        additional_instruction="Look for phrases indicating 'most roller coasters' record. If the page states 'world record' or 'most in North America' and the park is in the US, treat it as supporting the US record."
    )

    # Roller_Coaster_Count
    count_node = evaluator.add_leaf(
        id="Roller_Coaster_Count",
        desc="Provide the number of roller coasters at the park",
        parent=theme_node,
        critical=True
    )
    count_claim = f"{theme.name or ''} has {theme.roller_coaster_count or ''} roller coasters."
    await evaluator.verify(
        claim=count_claim,
        node=count_node,
        sources=sources,
        additional_instruction="Verify the page explicitly states the same coaster count. Allow minor textual variants (e.g., spelled-out numbers)."
    )

    # Park_Name_and_City
    name_city_node = evaluator.add_leaf(
        id="Park_Name_and_City",
        desc="Provide the specific name of the theme park and the city where it is located",
        parent=theme_node,
        critical=True
    )
    name_city_claim = f"The theme park is named '{theme.name or ''}' and is located in {theme.city or ''}, California."
    await evaluator.verify(
        claim=name_city_claim,
        node=name_city_node,
        sources=sources,
        additional_instruction="Confirm both the official park name and city on the referenced page. Accept reasonable formatting or city variants (e.g., Santa Clarita/Valencia)."
    )


async def build_reserve_checks(evaluator: Evaluator, parent_node, reserve: ReserveInfo) -> None:
    reserve_node = evaluator.add_parallel(
        id="State_Natural_Reserve_Identification",
        desc="Correctly identify a California State Natural Reserve in San Diego County with coastal hiking trails",
        parent=parent_node,
        critical=False
    )

    ref_present = evaluator.add_custom_node(
        result=bool(reserve and reserve.urls and len(reserve.urls) > 0),
        id="Reserve_Reference",
        desc="Provide a reference URL that confirms the reserve's designation and coastal hiking features",
        parent=reserve_node,
        critical=True
    )
    info_present = evaluator.add_custom_node(
        result=bool(reserve and reserve.name and reserve.name.strip()),
        id="Reserve_Info_Provided",
        desc="Reserve name is provided",
        parent=reserve_node,
        critical=True
    )

    sources = reserve.urls if reserve else []

    # Reserve_Location
    reserve_loc_node = evaluator.add_leaf(
        id="Reserve_Location",
        desc="The natural reserve is located in San Diego County, California",
        parent=reserve_node,
        critical=True
    )
    reserve_loc_claim = f"{reserve.name or ''} is located in San Diego County, California."
    await evaluator.verify(
        claim=reserve_loc_claim,
        node=reserve_loc_node,
        sources=sources,
        additional_instruction="Confirm the page mentions 'San Diego County' or a locality clearly within San Diego County."
    )

    # Reserve_Designation
    reserve_desig_node = evaluator.add_leaf(
        id="Reserve_Designation",
        desc="The location is designated as a California State Natural Reserve (not just a state park or regional park)",
        parent=reserve_node,
        critical=True
    )
    reserve_desig_claim = f"{reserve.name or ''} is designated as a California State Natural Reserve."
    await evaluator.verify(
        claim=reserve_desig_claim,
        node=reserve_desig_node,
        sources=sources,
        additional_instruction="The text should explicitly indicate 'State Natural Reserve'. Do not accept only 'State Park' or generic 'park'."
    )

    # Coastal_Features
    coastal_node = evaluator.add_leaf(
        id="Coastal_Features",
        desc="The reserve features coastal hiking trails with ocean views",
        parent=reserve_node,
        critical=True
    )
    coastal_claim = f"{reserve.name or ''} features coastal hiking trails with ocean views."
    await evaluator.verify(
        claim=coastal_claim,
        node=coastal_node,
        sources=sources,
        additional_instruction="Look for mentions of 'coastal trails', 'ocean views', 'bluffs', or similar wording indicating ocean-view hikes."
    )

    # Reserve_Name
    reserve_name_node = evaluator.add_leaf(
        id="Reserve_Name",
        desc="Provide the specific name of the State Natural Reserve",
        parent=reserve_node,
        critical=True
    )
    reserve_name_claim = f"There is a California State Natural Reserve named '{reserve.name or ''}'."
    await evaluator.verify(
        claim=reserve_name_claim,
        node=reserve_name_node,
        sources=sources,
        additional_instruction="Confirm the reserve's official name on the provided reference page."
    )


async def build_trail_checks(evaluator: Evaluator, parent_node, trail: TrailInfo, reserve: ReserveInfo) -> None:
    trail_node = evaluator.add_parallel(
        id="Hiking_Trail_Selection",
        desc="Select and describe a specific hiking trail within the identified State Natural Reserve",
        parent=parent_node,
        critical=False
    )

    ref_present = evaluator.add_custom_node(
        result=bool(trail and trail.urls and len(trail.urls) > 0),
        id="Trail_Reference",
        desc="Provide a reference URL that describes the trail and its features",
        parent=trail_node,
        critical=True
    )
    info_present = evaluator.add_custom_node(
        result=bool(trail and trail.name and trail.name.strip() and trail.reserve_name and trail.reserve_name.strip()),
        id="Trail_Info_Provided",
        desc="Trail name and associated reserve name are provided",
        parent=trail_node,
        critical=True
    )

    sources = combine_sources(trail.urls if trail else [], reserve.urls if reserve else [])

    # Trail_Location
    trail_loc_node = evaluator.add_leaf(
        id="Trail_Location",
        desc="The trail is located within a California State Natural Reserve in San Diego County that has coastal features",
        parent=trail_node,
        critical=True
    )
    trail_loc_claim = (
        f"The trail '{trail.name or ''}' is located within {trail.reserve_name or reserve.name or ''}, "
        f"a California State Natural Reserve in San Diego County."
    )
    await evaluator.verify(
        claim=trail_loc_claim,
        node=trail_loc_node,
        sources=sources,
        additional_instruction="Verify the trail is inside the named State Natural Reserve; the reserve should be in San Diego County."
    )

    # Ocean_Views
    ocean_views_node = evaluator.add_leaf(
        id="Ocean_Views",
        desc="The trail provides views of the Pacific Ocean or coastal bluffs",
        parent=trail_node,
        critical=True
    )
    ocean_views_claim = f"The trail '{trail.name or ''}' provides views of the Pacific Ocean or coastal bluffs."
    await evaluator.verify(
        claim=ocean_views_claim,
        node=ocean_views_node,
        sources=sources,
        additional_instruction="Look for explicit mentions of ocean views, coastal bluffs, coastline overlooks, etc."
    )

    # Trail_Name
    trail_name_node = evaluator.add_leaf(
        id="Trail_Name",
        desc="Provide the specific name of the hiking trail",
        parent=trail_node,
        critical=True
    )
    trail_name_claim = f"There is a hiking trail named '{trail.name or ''}' within {trail.reserve_name or reserve.name or ''}."
    await evaluator.verify(
        claim=trail_name_claim,
        node=trail_name_node,
        sources=sources,
        additional_instruction="Confirm the trail's official or commonly used name on the reference page."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
        default_model=model
    )

    # Extract structured information
    extracted = await evaluator.extract(
        prompt=prompt_extract_plan(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction"
    )

    # Record extracted info as custom info for debugging
    evaluator.add_custom_info(
        info={
            "theme_park": extracted.theme_park.dict() if extracted.theme_park else None,
            "reserve": extracted.reserve.dict() if extracted.reserve else None,
            "trail": extracted.trail.dict() if extracted.trail else None
        },
        info_type="extraction_overview"
    )

    # Create a top-level plan node (non-critical to allow partial credit)
    plan_node = evaluator.add_parallel(
        id="California_Outdoor_Recreation_Plan",
        desc="Evaluate a Southern California outdoor recreation plan that identifies the theme park with the most roller coasters in the US and a coastal hiking location in San Diego County",
        parent=root,
        critical=False
    )

    theme = extracted.theme_park or ThemeParkInfo()
    reserve = extracted.reserve or ReserveInfo()
    trail = extracted.trail or TrailInfo()

    # Build verification subtrees
    await build_theme_park_checks(evaluator, plan_node, theme)
    await build_reserve_checks(evaluator, plan_node, reserve)
    await build_trail_checks(evaluator, plan_node, trail, reserve)

    # Return evaluation summary
    return evaluator.get_summary()