import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "colorado_state_park_selection"
TASK_DESCRIPTION = (
    "I am planning a camping trip to Colorado and need to find a state park that accommodates my specific requirements. "
    "I am looking for a park that is located at an elevation of 8,000 feet or higher, has wheelchair-accessible camping facilities, "
    "allows leashed dogs at the campsites, and offers fishing opportunities. Can you identify a Colorado state park that meets all of these criteria "
    "and provide the official park information including a reference URL?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ParkSelection(BaseModel):
    """
    Extracted fields for the identified Colorado state park.
    """
    park_name: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_park_selection() -> str:
    return """
    Extract the identified Colorado state park and the reference URL(s) cited in the answer.

    Return a JSON object with the following fields:
    1. park_name: The name of the Colorado state park identified in the answer (e.g., "Mueller State Park").
    2. reference_urls: An array of all URLs explicitly provided in the answer that serve as references for the park information.
       - Include any official park pages, policy pages, camping pages, brochures, or other URLs mentioned by the answer.
       - Extract actual URLs (including those inside markdown links). Do not infer URLs not present in the answer.
       - Only include valid URLs. If a URL lacks protocol (http/https), prepend http://.

    If the park name is not explicitly provided, return `null` for park_name.
    If no reference URLs are provided, return an empty array for reference_urls.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_colorado_state_park(
    evaluator: Evaluator,
    root_node,
    selection: ParkSelection,
) -> None:
    """
    Build and execute the verification tree based on the rubric.
    """
    # Create the main critical node (parent gate for all criteria)
    main_node = evaluator.add_parallel(
        id="Colorado_State_Park_Identification",
        desc="Identify a Colorado state park that meets the elevation, accessibility, pet, and fishing requirements, and provide an official reference URL.",
        parent=root_node,
        critical=True
    )

    # 1) Park name provided (existence check) - critical
    park_name_provided = evaluator.add_custom_node(
        result=bool(selection.park_name and selection.park_name.strip()),
        id="Park_Name_Provided",
        desc="The name of the identified Colorado state park is provided.",
        parent=main_node,
        critical=True
    )

    # 2) Official reference URL(s) - split into existence + officialness under a critical group
    official_group = evaluator.add_parallel(
        id="Official_Reference_URL",
        desc="Official reference URL(s) are provided and at least one is an official source.",
        parent=main_node,
        critical=True
    )

    urls_provided = evaluator.add_custom_node(
        result=bool(selection.reference_urls),
        id="Official_Reference_URL_Provided",
        desc="At least one reference URL is provided in the answer.",
        parent=official_group,
        critical=True
    )

    urls_official_leaf = evaluator.add_leaf(
        id="Official_Reference_URL_Is_Official",
        desc="At least one provided reference URL is an official source for the park (Colorado Parks & Wildlife or Colorado government domain).",
        parent=official_group,
        critical=True
    )
    official_claim = (
        f"At least one of these URLs is an official page for {selection.park_name or 'the identified park'} "
        f"from Colorado Parks & Wildlife (cpw.state.co.us or *.state.co.us) or a Colorado government domain (e.g., *.colorado.gov). "
        f"The page should provide official park information."
    )
    await evaluator.verify(
        claim=official_claim,
        node=urls_official_leaf,
        sources=selection.reference_urls,
        additional_instruction=(
            "Judge officialness using the domain and page content. Accept pages under cpw.state.co.us, *.state.co.us, or *.colorado.gov, "
            "and pages that clearly indicate 'Colorado Parks & Wildlife' or an official Colorado government entity in branding or header/footer. "
            "Do NOT accept third-party aggregators (e.g., AllTrails, Hipcamp, Wikipedia, private blogs) as official sources."
        ),
    )

    # Convenience handle for prerequisites: ensure subsequent verifications depend on official page success
    extra_prereqs = [urls_official_leaf]

    # 3) Location in Colorado (critical)
    location_leaf = evaluator.add_leaf(
        id="Location_Colorado",
        desc="The identified park is located in Colorado.",
        parent=main_node,
        critical=True
    )
    location_claim = (
        f"This page confirms that {selection.park_name or 'the identified park'} is located in Colorado, "
        f"and it is a Colorado State Park."
    )
    await evaluator.verify(
        claim=location_claim,
        node=location_leaf,
        sources=selection.reference_urls,
        additional_instruction=(
            "Look for explicit mentions such as 'Colorado', 'Colorado Parks & Wildlife', or 'Colorado State Park'. "
            "The evidence should make it clear that the park is in Colorado."
        ),
        extra_prerequisites=extra_prereqs
    )

    # 4) Elevation requirement (critical: >= 8,000 feet)
    elevation_leaf = evaluator.add_leaf(
        id="Elevation_Requirement",
        desc="The park is at an elevation of 8,000 feet or higher above sea level (as supported by the cited official source).",
        parent=main_node,
        critical=True
    )
    elevation_claim = (
        f"The elevation of {selection.park_name or 'the identified park'} is at least 8,000 feet above sea level. "
        f"Statements like 'over 8,000 feet', '~8,000 ft', or values ≥ 8,000 ft (including metric equivalents ≥ 2,438 meters) qualify."
    )
    await evaluator.verify(
        claim=elevation_claim,
        node=elevation_leaf,
        sources=selection.reference_urls,
        additional_instruction=(
            "Verify that the official page(s) explicitly state an elevation ≥ 8,000 ft, or provide a metric elevation ≥ 2,438 m. "
            "Allow reasonable rounding. If only a range is given, accept if the minimum is ≥ 8,000 ft. "
            "Do not rely on non-official pages."
        ),
        extra_prerequisites=extra_prereqs
    )

    # 5) Wheelchair-accessible camping facilities (critical)
    accessible_leaf = evaluator.add_leaf(
        id="Wheelchair_Accessible_Camping",
        desc="The park has at least one wheelchair-accessible campsite/camping facility (as supported by the cited official source).",
        parent=main_node,
        critical=True
    )
    accessible_claim = (
        f"The official source indicates that {selection.park_name or 'the identified park'} has at least one wheelchair-accessible "
        f"campsite or ADA-accessible camping facilities."
    )
    await evaluator.verify(
        claim=accessible_claim,
        node=accessible_leaf,
        sources=selection.reference_urls,
        additional_instruction=(
            "Look for terms such as 'ADA accessible campsite', 'wheelchair-accessible', 'accessible camping', or similar. "
            "Accept if any campground or campsite in the park is described as accessible. "
            "General accessibility for restrooms or trails alone is insufficient unless it specifically includes camping."
        ),
        extra_prerequisites=extra_prereqs
    )

    # 6) Leashed dogs allowed at campsites (critical)
    dogs_leaf = evaluator.add_leaf(
        id="Leashed_Dogs_Allowed",
        desc="The park allows leashed dogs at campsites (as supported by the cited official source).",
        parent=main_node,
        critical=True
    )
    dogs_claim = (
        f"The official source indicates that leashed dogs are allowed at campsites/campgrounds in {selection.park_name or 'the identified park'}."
    )
    await evaluator.verify(
        claim=dogs_claim,
        node=dogs_leaf,
        sources=selection.reference_urls,
        additional_instruction=(
            "Check the park's pet policy or camping page for statements like 'pets allowed', 'dogs must be leashed', "
            "specifically permitting pets in campgrounds/campsites. "
            "Policies may restrict pets in buildings or some areas; this does not negate allowance at campsites."
        ),
        extra_prerequisites=extra_prereqs
    )

    # 7) Fishing opportunities (critical)
    fishing_leaf = evaluator.add_leaf(
        id="Fishing_Opportunities",
        desc="The park offers fishing opportunities (as supported by the cited official source).",
        parent=main_node,
        critical=True
    )
    fishing_claim = (
        f"The official source indicates that fishing opportunities exist within {selection.park_name or 'the identified park'} "
        f"(e.g., lakes, reservoirs, rivers, or designated fishing areas)."
    )
    await evaluator.verify(
        claim=fishing_claim,
        node=fishing_leaf,
        sources=selection.reference_urls,
        additional_instruction=(
            "Look for 'fishing', 'angling', or specific waterbodies where fishing is permitted. "
            "Program pages, brochures, or official activity lists that include fishing are acceptable."
        ),
        extra_prerequisites=extra_prereqs
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
    Evaluate the agent's answer for the Colorado state park selection task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root container; main rubric node added beneath
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

    # Extract park selection info from the answer
    selection = await evaluator.extract(
        prompt=prompt_extract_park_selection(),
        template_class=ParkSelection,
        extraction_name="park_selection",
    )

    # Build verification tree according to rubric and run checks
    await verify_colorado_state_park(evaluator, root, selection)

    # Return structured result summary
    return evaluator.get_summary()