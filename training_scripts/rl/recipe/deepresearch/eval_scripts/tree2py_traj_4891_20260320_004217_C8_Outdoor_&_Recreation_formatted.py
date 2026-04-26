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
TASK_ID = "accessible_rv_trip_plan"
TASK_DESCRIPTION = """
You are planning a 4-week accessible RV camping trip to national parks in the United States during summer 2026. You will spend one week at each of four different parks. Your RV is 28 feet long, and you require campgrounds with electric hookups for medical equipment. You also want to explore parks that offer good wheelchair accessibility for trail hiking.

Identify 4 national parks that meet ALL of the following criteria:
1. The park has campgrounds that can accommodate RVs of at least 28 feet in length
2. The park has at least one campground with electric hookups available
3. The park has wheelchair-accessible trails for exploration

For each of the 4 parks, provide the following information:
• Park name and state location
• Wheelchair accessibility information (either the percentage of wheelchair-accessible trails or a description of the accessible features available)
• The maximum RV length that can be accommodated at the park's campgrounds
• Confirmation of electric hookup availability, including the specific campground name(s) that offer these hookups
• Names of at least 2 specific wheelchair-accessible trails in the park
• The wildlife viewing distance requirements that apply in the park (state both the minimum distance visitors must maintain from most wildlife and the minimum distance from predators such as bears and wolves)
• A reference URL from an official or reliable source (such as National Park Service, Recreation.gov, or reputable travel sites) that supports your information
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkInfo(BaseModel):
    park_name: Optional[str] = None
    state: Optional[str] = None
    accessibility_info: Optional[str] = None  # percentage or descriptive features
    rv_max_length: Optional[str] = None       # keep as string to allow ranges/text
    electric_campgrounds: List[str] = Field(default_factory=list)
    accessible_trails: List[str] = Field(default_factory=list)
    wildlife_distance_general: Optional[str] = None    # e.g., "25 yards" or "23 meters"
    wildlife_distance_predators: Optional[str] = None  # e.g., "100 yards" or "91 meters"
    reference_urls: List[str] = Field(default_factory=list)


class TripPlanExtraction(BaseModel):
    parks: List[ParkInfo] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_trip_plan() -> str:
    return """
    Extract the first four national parks (as they appear in the answer) intended for a 4-week accessible RV trip plan.
    For each park, extract the following fields:
    - park_name: The park's official name (e.g., "Yellowstone National Park")
    - state: The U.S. state or states where the park is located (e.g., "Wyoming", "Utah/Arizona")
    - accessibility_info: A short description of wheelchair accessibility for trails or a stated percentage of accessible trails
    - rv_max_length: The stated maximum RV length accommodated at any campground in the park (text as shown, e.g., "up to 35 feet", "32–40 ft", etc.)
    - electric_campgrounds: An array of the names of campground(s) in the park that have electric hookups (e.g., ["Madison Campground", "Fishing Bridge RV Park"])
    - accessible_trails: An array of named wheelchair-accessible trails in the park (include at least two when provided)
    - wildlife_distance_general: The stated minimum distance visitors must maintain from most wildlife (e.g., "25 yards", "23 meters")
    - wildlife_distance_predators: The stated minimum distance visitors must maintain from predators such as bears and wolves (e.g., "100 yards", "91 meters")
    - reference_urls: An array of source URLs explicitly mentioned in the answer (NPS, Recreation.gov, or other reputable sites). Extract only URLs present in the answer text (plain links or markdown). Do not invent URLs.

    Rules:
    - Do NOT fabricate any information. Only extract what is explicitly present in the answer.
    - If a field is missing, set it to null (for strings) or [] (for arrays).
    - For URLs, extract only valid, complete URLs mentioned in the answer. If a URL is missing a protocol, prepend http://
    - Return a JSON object with a single field: "parks": [ParkInfo, ...] (up to the first four parks).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def take_first_n(items: List[str], n: int) -> List[str]:
    return items[:n] if items else []


def non_empty_str(s: Optional[str]) -> bool:
    return s is not None and isinstance(s, str) and s.strip() != ""


# --------------------------------------------------------------------------- #
# Verification logic per-park                                                 #
# --------------------------------------------------------------------------- #
async def verify_park(
    evaluator: Evaluator,
    parent_node,
    park: ParkInfo,
    park_index_1based: int,
) -> None:
    """
    Build verification nodes and run checks for a single park.
    The rubric requires 7 critical checks per park under a parallel aggregator.
    """
    # Create the park node (parallel; non-critical at the root level allows partial credit across parks)
    park_node = evaluator.add_parallel(
        id=f"park_{park_index_1based}",
        desc=(
            f"{['First','Second','Third','Fourth'][park_index_1based-1]} national park meeting all "
            f"accessibility, RV accommodation, and electric hookup criteria"
        ),
        parent=parent_node,
        critical=False
    )

    # 1) Identification: Park name and state location provided (existence check)
    evaluator.add_custom_node(
        result=(non_empty_str(park.park_name) and non_empty_str(park.state)),
        id=f"park_{park_index_1based}_identification",
        desc="Park name and state location provided",
        parent=park_node,
        critical=True
    )

    # 2) Wheelchair accessibility info provided (existence check)
    evaluator.add_custom_node(
        result=non_empty_str(park.accessibility_info),
        id=f"park_{park_index_1based}_wheelchair_accessibility",
        desc="Wheelchair accessibility information provided (percentage of accessible trails or description of accessible features)",
        parent=park_node,
        critical=True
    )

    # Reference URLs will be used by all factual verifications
    sources = park.reference_urls if park.reference_urls else []

    # 3) RV accommodation: at least one campground >= 28 ft (verify with sources)
    rv_node = evaluator.add_leaf(
        id=f"park_{park_index_1based}_rv_accommodation",
        desc="Maximum RV length that can be accommodated is stated and meets the minimum requirement of 28 feet",
        parent=park_node,
        critical=True
    )
    park_name_for_claim = park.park_name or "the park"
    rv_claim = (
        f"At least one campground in {park_name_for_claim} accommodates RVs of at least 28 feet in length."
    )
    await evaluator.verify(
        claim=rv_claim,
        node=rv_node,
        sources=sources,
        additional_instruction=(
            "Use official NPS or Recreation.gov campground pages if available. "
            "The claim is correct if any campground within the park lists RV/trailer/rig or total vehicle length "
            "limits of 28 ft or greater. Accept synonyms such as 'max vehicle length', 'site length', 'pad length', "
            "and allow ranges. If the answer provides a specific maximum like "
            f"'{park.rv_max_length or 'N/A'}', it's sufficient that it is ≥ 28 ft."
        )
    )

    # 4) Electric hookups availability with specific campground name(s) (verify with sources)
    elec_node = evaluator.add_leaf(
        id=f"park_{park_index_1based}_electric_hookups",
        desc="Electric hookup availability confirmed with specific campground name(s) that offer hookups",
        parent=park_node,
        critical=True
    )
    named_cg = ", ".join(park.electric_campgrounds) if park.electric_campgrounds else "none provided"
    elec_claim = (
        f"The following campground(s) in {park_name_for_claim} offer electric hookups: {named_cg}."
    )
    await evaluator.verify(
        claim=elec_claim,
        node=elec_node,
        sources=sources,
        additional_instruction=(
            "Confirm that the specified campground(s) explicitly list electric hookups (e.g., 30/50 amp). "
            "If no specific campground names were provided in the answer, treat this claim as not supported even "
            "if the park has electric hookups somewhere."
        )
    )

    # 5) At least 2 specific wheelchair-accessible trail names (verify with sources)
    trails_node = evaluator.add_leaf(
        id=f"park_{park_index_1based}_trail_names",
        desc="At least 2 specific wheelchair-accessible trail names provided",
        parent=park_node,
        critical=True
    )
    first_two_trails = take_first_n(park.accessible_trails, 2)
    if len(first_two_trails) >= 2:
        trails_list_text = ", ".join(first_two_trails)
    else:
        trails_list_text = ", ".join(first_two_trails) if first_two_trails else "none"
    trails_claim = (
        f"The following named trails in {park_name_for_claim} are wheelchair-accessible: {trails_list_text}."
    )
    await evaluator.verify(
        claim=trails_claim,
        node=trails_node,
        sources=sources,
        additional_instruction=(
            "Verify that at least two named trails are in the park and designated as wheelchair-accessible "
            "(e.g., paved, boardwalk, ADA accessible, or explicitly described as wheelchair accessible). "
            "If fewer than two trail names were provided in the answer, mark this as incorrect."
        )
    )

    # 6) Wildlife viewing distance requirements (verify with sources)
    wildlife_node = evaluator.add_leaf(
        id=f"park_{park_index_1based}_wildlife_distance",
        desc="Wildlife viewing distance requirements stated (minimum distance from most wildlife and from predators)",
        parent=park_node,
        critical=True
    )
    general_dist = park.wildlife_distance_general or "unspecified"
    predator_dist = park.wildlife_distance_predators or "unspecified"
    wildlife_claim = (
        f"In {park_name_for_claim}, visitors must stay at least {general_dist} from most wildlife and at least "
        f"{predator_dist} from predators such as bears and wolves."
    )
    await evaluator.verify(
        claim=wildlife_claim,
        node=wildlife_node,
        sources=sources,
        additional_instruction=(
            "Check the park's regulations/safety page(s). Accept if the park states the generic NPS guidance "
            "(25 yards/23 meters from most wildlife and 100 yards/91 meters from bears/wolves) or park-specific "
            "equivalents. Allow reasonable unit conversions and rounding."
        )
    )

    # 7) Reference URL existence (existence check only)
    evaluator.add_custom_node(
        result=(len(park.reference_urls) > 0),
        id=f"park_{park_index_1based}_reference_url",
        desc="Reference URL from an official or reliable source provided to support the information",
        parent=park_node,
        critical=True
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
    Evaluate an answer for the accessible RV trip plan task and return a structured dictionary.
    """
    # Initialize evaluator (root should be non-critical to allow partial credit across parks)
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

    # Root node description mirrors the rubric; keep root non-critical to avoid forcing all children critical
    accessible_plan_root = evaluator.add_parallel(
        id="accessible_rv_trip_plan",
        desc="Complete 4-week accessible RV trip planning information for 4 national parks meeting specified accessibility and accommodation criteria",
        parent=root,
        critical=False
    )

    # Extract structured trip plan info
    extraction = await evaluator.extract(
        prompt=prompt_extract_trip_plan(),
        template_class=TripPlanExtraction,
        extraction_name="trip_plan_extraction"
    )

    # Normalize to 4 parks (pad with empty placeholders if necessary)
    parks = (extraction.parks or [])[:4]
    while len(parks) < 4:
        parks.append(ParkInfo())

    # Verify each of the four parks according to the rubric
    for idx in range(4):
        await verify_park(
            evaluator=evaluator,
            parent_node=accessible_plan_root,
            park=parks[idx],
            park_index_1based=idx + 1
        )

    # Optionally record some custom stats
    evaluator.add_custom_info(
        {
            "parks_extracted": len(extraction.parks) if extraction and extraction.parks is not None else 0,
            "parks_evaluated": 4
        },
        info_type="stats",
        info_name="extraction_stats"
    )

    # Return structured summary with verification tree and scores
    return evaluator.get_summary()