import asyncio
import logging
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "wi_state_park_dells_criteria_2023"
TASK_DESCRIPTION = (
    "Identify the Wisconsin state park that meets ALL of the following criteria as of December 31, 2023: "
    "1. The park is located within 2 miles of Wisconsin Dells, "
    "2. The park has at least 80 campsites, "
    "3. The park offers campsites with electrical hookups, "
    "4. The campground provides flush toilets and shower facilities, "
    "5. The park features a self-guided nature trail. "
    "Provide the state park name and reference URLs supporting each criterion."
)


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class ParkExtraction(BaseModel):
    park_name: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    campsite_count_urls: List[str] = Field(default_factory=list)
    electrical_hookups_urls: List[str] = Field(default_factory=list)
    bathroom_facilities_urls: List[str] = Field(default_factory=list)
    nature_trail_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_park_info() -> str:
    return """
    Extract the state park identification and the categorized reference URLs exactly as provided in the answer.

    You must extract:
    - park_name: The specific Wisconsin state park name provided in the answer text.
    - location_urls: URLs the answer cites for proving the park is within 2 miles of Wisconsin Dells.
    - campsite_count_urls: URLs the answer cites for proving the park has at least 80 campsites.
    - electrical_hookups_urls: URLs the answer cites for proving the park offers campsites with electrical hookups.
    - bathroom_facilities_urls: URLs the answer cites for proving the campground provides flush toilets and showers.
    - nature_trail_urls: URLs the answer cites for proving the park features a self-guided nature trail.

    Important:
    - Only include URLs explicitly present in the answer. Do not fabricate or infer any URLs.
    - If the answer gives one URL intended to support multiple criteria, include that same URL in each relevant list.
    - If a category has no URLs in the answer, return an empty list for that category.
    - Keep URLs as full valid links (prepend http:// if missing).

    Return a JSON object with fields:
    park_name, location_urls, campsite_count_urls, electrical_hookups_urls, bathroom_facilities_urls, nature_trail_urls.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def has_valid_url(urls: Optional[List[str]]) -> bool:
    if not urls or not isinstance(urls, list):
        return False
    for u in urls:
        if isinstance(u, str) and u.strip().lower().startswith(("http://", "https://")):
            return True
    return False


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    extracted: ParkExtraction
) -> None:
    """
    Build the verification tree per rubric and perform verifications.
    """
    # Create a critical task node under the root to enforce all‑or‑nothing
    main_node = evaluator.add_parallel(
        id="StateParkIdentification",
        desc="Correctly identify a Wisconsin state park meeting all specified criteria",
        parent=evaluator.root,
        critical=True
    )

    # 1) Park name provided (existence check)
    park_name_provided = evaluator.add_custom_node(
        result=bool(extracted.park_name and extracted.park_name.strip()),
        id="ParkNameProvided",
        desc="The answer provides a specific Wisconsin state park name",
        parent=main_node,
        critical=True
    )

    # 2) DocumentationReferences (existence of URLs for each criterion)
    refs_parent = evaluator.add_parallel(
        id="DocumentationReferences",
        desc="Provide valid reference URLs supporting all claims",
        parent=main_node,
        critical=True
    )

    location_ref = evaluator.add_custom_node(
        result=has_valid_url(extracted.location_urls),
        id="LocationReference",
        desc="URL confirming the park's location within 2 miles of Wisconsin Dells",
        parent=refs_parent,
        critical=True
    )

    campsite_count_ref = evaluator.add_custom_node(
        result=has_valid_url(extracted.campsite_count_urls),
        id="CampsiteCountReference",
        desc="URL confirming the number of campsites (at least 80)",
        parent=refs_parent,
        critical=True
    )

    electrical_ref = evaluator.add_custom_node(
        result=has_valid_url(extracted.electrical_hookups_urls),
        id="ElectricalHookupsReference",
        desc="URL confirming availability of electrical hookups",
        parent=refs_parent,
        critical=True
    )

    bathroom_ref = evaluator.add_custom_node(
        result=has_valid_url(extracted.bathroom_facilities_urls),
        id="BathroomFacilitiesReference",
        desc="URL confirming flush toilets and showers",
        parent=refs_parent,
        critical=True
    )

    nature_trail_ref = evaluator.add_custom_node(
        result=has_valid_url(extracted.nature_trail_urls),
        id="NatureTrailReference",
        desc="URL confirming the self-guided nature trail",
        parent=refs_parent,
        critical=True
    )

    # 3) Geographic proximity verification (factual leaf, URL-grounded)
    geo_leaf = evaluator.add_leaf(
        id="GeographicProximity",
        desc="The identified park is located within 2 miles of Wisconsin Dells",
        parent=main_node,
        critical=True
    )
    park_display = extracted.park_name or "the park"
    geo_claim = f"The state park '{extracted.park_name}' is located within 2 miles of Wisconsin Dells, Wisconsin."
    await evaluator.verify(
        claim=geo_claim,
        node=geo_leaf,
        sources=extracted.location_urls,
        additional_instruction=(
            "As of Dec 31, 2023, verify the webpage explicitly supports that the park is within 2 miles of "
            "Wisconsin Dells (e.g., '1.5 miles', 'about 2 miles', 'within two miles'). Generic statements like "
            "'near Wisconsin Dells' without distance are insufficient. Prefer explicit mileage."
        ),
        extra_prerequisites=[location_ref, park_name_provided]
    )

    # 4) Camping Facilities group (all critical)
    camping_parent = evaluator.add_parallel(
        id="CampingFacilities",
        desc="Verify the park offers all required camping facilities",
        parent=main_node,
        critical=True
    )

    # 4.a) Campsite capacity >= 80
    capacity_leaf = evaluator.add_leaf(
        id="CampsiteCapacity",
        desc="The park has at least 80 campsites available",
        parent=camping_parent,
        critical=True
    )
    capacity_claim = (
        f"As of Dec 31, 2023, the park '{extracted.park_name}' has at least 80 total campsites "
        "(sum across its campgrounds if applicable)."
    )
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_leaf,
        sources=extracted.campsite_count_urls,
        additional_instruction=(
            "Confirm the total number of campsites is >= 80. Accept phrasing like 'more than 80', 'at least 80', "
            "'90 sites', etc. If multiple campground areas are within the park, you may sum their counts if the "
            "page makes this clear."
        ),
        extra_prerequisites=[campsite_count_ref, park_name_provided]
    )

    # 4.b) Electrical hookups available
    electric_leaf = evaluator.add_leaf(
        id="ElectricalHookups",
        desc="The park offers campsites with electrical hookups",
        parent=camping_parent,
        critical=True
    )
    electric_claim = f"The park '{extracted.park_name}' offers campsites with electrical hookups (electric sites)."
    await evaluator.verify(
        claim=electric_claim,
        node=electric_leaf,
        sources=extracted.electrical_hookups_urls,
        additional_instruction=(
            "Look for explicit mentions such as 'electric sites', 'sites with electricity', '30/50 amp service', "
            "or similar wording indicating electrical hookups at campsites."
        ),
        extra_prerequisites=[electrical_ref, park_name_provided]
    )

    # 4.c) Flush toilets and showers
    bathroom_leaf = evaluator.add_leaf(
        id="BathroomFacilities",
        desc="The campground provides both flush toilets and shower facilities",
        parent=camping_parent,
        critical=True
    )
    bathroom_claim = (
        f"The campground at '{extracted.park_name}' provides both flush toilets and shower facilities "
        "(seasonal showers count as showers)."
    )
    await evaluator.verify(
        claim=bathroom_claim,
        node=bathroom_leaf,
        sources=extracted.bathroom_facilities_urls,
        additional_instruction=(
            "Both features must be present: flush toilets AND showers. Seasonal availability is acceptable "
            "if explicitly indicated."
        ),
        extra_prerequisites=[bathroom_ref, park_name_provided]
    )

    # 5) Self-guided nature trail
    nature_leaf = evaluator.add_leaf(
        id="NatureTrail",
        desc="The park features a self-guided nature trail",
        parent=main_node,
        critical=True
    )
    nature_claim = f"The park '{extracted.park_name}' features a self-guided nature trail."
    await evaluator.verify(
        claim=nature_claim,
        node=nature_leaf,
        sources=extracted.nature_trail_urls,
        additional_instruction=(
            "Confirm there is a self-guided nature trail. Accept phrasing like 'self-guided trail', "
            "'self-guided interpretive trail', or similar. Guided-only tours are not sufficient."
        ),
        extra_prerequisites=[nature_trail_ref, park_name_provided]
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
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Evaluate an answer for the Wisconsin Dells state park criteria task.

    Returns a structured summary with the verification tree and final score.
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

    # Extract structured info from the answer
    extracted: ParkExtraction = await evaluator.extract(
        prompt=prompt_extract_park_info(),
        template_class=ParkExtraction,
        extraction_name="park_extraction"
    )

    # Optional: record as-of date as custom info
    evaluator.add_custom_info(
        info={"as_of_date": "2023-12-31"},
        info_type="metadata",
        info_name="evaluation_context"
    )

    # Build and verify
    await build_and_verify_tree(evaluator, extracted)

    # Return evaluator summary
    return evaluator.get_summary()