import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy
from mind2web2.llm_client.base_client import LLMClient

TASK_ID = "spinning_class"
TASK_DESCRIPTION = """
Find me four studios offering indoor cycling/spinning in Seattle, WA that have a Google review rating of 4.5 stars or higher. For each studio, provide its name, physical address, a link of its Google Maps page, google review rating, a link to their class schedule or timetable page (from its official website).
"""

EVAL_NOTES = ""
GROUND_TRUTH = {}


class StudioInfo(BaseModel):
    """Data model for extracted studio information."""
    name: Optional[str] = Field(default=None, description="Studio name")
    address: Optional[str] = Field(default=None, description="Physical address")
    google_maps_url: Optional[str] = Field(default=None, description="Google Maps page URL")
    google_rating: Optional[str] = Field(default=None, description="Google review rating")
    schedule_url: Optional[str] = Field(default=None, description="Class schedule/timetable page URL")


class ExtractedStudios(BaseModel):
    """Container for all extracted studio information."""
    studios: List[StudioInfo] = Field(default_factory=list, description="List of spinning studios")


def prompt_extract_studios() -> str:
    """
    Extraction prompt for getting structured studio information from the answer.
    """
    return """
    Extract information about spinning/indoor cycling studios from the answer.

    Look for:
    - Studio names
    - Physical addresses in Seattle, WA
    - Google Maps page URLs
    - Google review ratings (numerical values)
    - Class schedule or timetable page URLs from official websites

    Extract all studios mentioned in the answer, even if there are more or fewer than 4.
    Extract information exactly as it appears in the text.
    If any field is not mentioned for a studio, set it to null.
    """


def create_placeholder_studio(index: int) -> StudioInfo:
    """Create a placeholder studio for missing items."""
    return StudioInfo(
        name=f"[Missing Studio {index + 1}]",
        address=None,
        google_maps_url=None,
        google_rating=None,
        schedule_url=None
    )


async def verify_studio_requirements(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        studio: StudioInfo,
        studio_index: int,
) -> None:
    """
    Verify all requirements for a single studio.
    """

    # Create studio container node
    studio_node = evaluator.add_parallel(
        id=f"studio_{studio_index + 1}",
        desc=f"Studio {studio_index + 1}: {studio.name or '[Missing]'}",
        parent=parent_node,
        critical=False,  # Allow partial credit for studios
    )

    # Check if studio exists (has name)
    studio_exists = bool(studio.name and studio.name.strip() and studio.address and studio.google_rating and studio.google_maps_url and studio.schedule_url and not studio.name.startswith("[Missing"))

    existence_node = evaluator.add_custom_node(
        result=studio_exists,
        id=f"studio_{studio_index + 1}_exists",
        desc=f"Studio {studio_index + 1} information provided",
        parent=studio_node,
        critical=True,  # If studio doesn't exist, other checks are meaningless
    )

    if not studio_exists:
        # Create placeholder nodes for missing studio
        for field in ["name", "address", "google_maps_url", "google_rating", "schedule_url"]:
            evaluator.add_leaf(
                id=f"studio_{studio_index + 1}_{field}",
                desc=f"Studio {studio_index + 1} {field.replace('_', ' ')} verification",
                parent=studio_node,
                critical=True,
                score=0.0,
                status="skipped"
            )
        return

    # Verify each required field
    # await verify_studio_name(evaluator, studio_node, studio, studio_index)
    await verify_studio_address(evaluator, studio_node, studio, studio_index)
    await verify_google_maps_url(evaluator, studio_node, studio, studio_index)
    await verify_google_rating(evaluator, studio_node, studio, studio_index)
    await verify_schedule_url(evaluator, studio_node, studio, studio_index)


async def verify_studio_name(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        studio: StudioInfo,
        studio_index: int,
) -> None:
    """Verify studio name is provided and reasonable."""

    name_node = evaluator.add_leaf(
        id=f"studio_{studio_index + 1}_name",
        desc=f"Studio {studio_index + 1} has valid name and is related to spinning",
        parent=parent_node,
        critical=True,  # Name is essential for identification
    )

    claim = f"This page is related to a studio named '{studio.name}',and we can infer from the page or the name that this studio is related to spinning or indoor cycling"

    await evaluator.verify(
        claim=claim,
        node=name_node,
        sources=None,
        additional_instruction="Verify that this appears to be a legitimate business name for a fitness studio, not placeholder text or invalid content"
    )


async def verify_studio_address(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        studio: StudioInfo,
        studio_index: int,
) -> None:
    """Verify studio address is in Seattle, WA."""

    address_node = evaluator.add_leaf(
        id=f"studio_{studio_index + 1}_address",
        desc=f"Studio {studio_index + 1} address is in Seattle, WA",
        parent=parent_node,
        critical=True,  # Must be in Seattle as specified
    )

    claim = f"Use your kowledge or check from the address details, judge whether this address '{studio.address}' is located in Seattle, WA"

    await evaluator.verify(
        claim=claim,
        node=address_node,
        sources=None,
        additional_instruction="Verify that the address contains Seattle, WA or clear Seattle location indicators. Allow reasonable variations in address format."
    )


async def verify_google_maps_url(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        studio: StudioInfo,
        studio_index: int,
) -> None:
    """Verify Google Maps URL is valid and matches the studio."""

    maps_url_node = evaluator.add_leaf(
        id=f"studio_{studio_index + 1}_google_maps",
        desc=f"Studio {studio_index + 1} Google Maps URL is valid and accurate for the studio name or the address",
        parent=parent_node,
        critical=True,  # Google Maps link is required
    )
    # if not studio.google_maps_url:
    #     claim = "No Google Maps URL was provided"
    #     await evaluator.verify(
    #         claim=claim,
    #         node=maps_url_node,
    #         sources=None,
    #         additional_instruction="This should fail since Google Maps URL is required"
    #     )
    #     return
    #

    claim = f"Check: 1)This must be a google map page 2) It's either 2.1) for the studio '{studio.name}' or 2.2) for this address '{studio.address}', which may not be showing the studio name."

    await evaluator.verify(
        claim=claim,
        node=maps_url_node,
        sources=studio.google_maps_url,
        additional_instruction="Allow reasonable variations in how the business name or address appears on Google Maps."
    )


async def verify_google_rating(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        studio: StudioInfo,
        studio_index: int,
) -> None:
    """Verify Google rating meets the 4.5+ requirement."""

    rating_node = evaluator.add_leaf(
        id=f"studio_{studio_index + 1}_rating_requirement",
        desc=f"Studio {studio_index + 1} Google rating is 4.5 stars or higher",
        parent=parent_node,
        critical=True,  # Rating requirement is mandatory
    )

    if not studio.google_rating:
        claim = "No Google rating was provided"
        await evaluator.verify(
            claim=claim,
            node=rating_node,
            sources=None,
            additional_instruction="This should fail since Google rating is required"
        )
        return

    claim = f"This page shows a google rating of '{studio.google_rating}' that is 4.5 stars or higher。 Plz ignore the potential inconsistency of the number of reviewers, and focus on the rating itself. As long as it shows a rating score of at least 4.5, it's acceptable."

    # Also verify with Google Maps URL if available
    sources = [studio.google_maps_url] if studio.google_maps_url else None

    await evaluator.verify(
        claim=claim,
        node=rating_node,
        sources=sources,
        additional_instruction="Verify that the rating is 4.5 or above. Accept ratings like '4.5', '4.6', '4.8', etc. "
    )


async def verify_schedule_url(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        studio: StudioInfo,
        studio_index: int,
) -> None:
    """Verify class schedule URL is from official website."""

    schedule_node = evaluator.add_leaf(
        id=f"studio_{studio_index + 1}_schedule_url",
        desc=f"Studio {studio_index + 1} schedule URL is from official website and it is a spinning studio",
        parent=parent_node,
        critical=True,  # Schedule URL is required
    )

    # if not studio.schedule_url:
    #     claim = "No class schedule URL was provided"
    #     await evaluator.verify(
    #         claim=claim,
    #         node=schedule_node,
    #         sources=None,
    #         additional_instruction="This should fail since schedule URL is required"
    #     )
    #     return

    claim = f"The page shows  a class schedule or timetable page from the official website of '{studio.name}', and the classes are related to spinning or indoor cycling"

    await evaluator.verify(
        claim=claim,
        node=schedule_node,
        sources=studio.schedule_url,
        additional_instruction="Verify that this URL leads to a legitimate class schedule or timetable page from the studio's official website. The page should show spinning/cycling class schedules."
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                               #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
        client: LLMClient,
        answer: str,
        agent_name: str,
        answer_name: str,
        cache: CacheFileSys,
        semaphore: asyncio.Semaphore,
        logger: logging.Logger,
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Main evaluation function for spinning class studio recommendations.

    Evaluates whether the answer provides 4 valid spinning studios in Seattle, WA
    with all required information and meeting the 4.5+ Google rating requirement.
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Allow partial credit for studios
        agent_name=agent_name,
        answer_name=answer_name,
        # Evaluator creation parameters
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # -------- 2. Extract structured information ------------------- #
    parsed_studios = await evaluator.extract(
        prompt=prompt_extract_studios(),
        template_class=ExtractedStudios,
        extraction_name="studio_extraction",
        source=None,  # Extract from answer text
    )

    # -------- 3. Prepare studios for evaluation ------------------- #
    # Take first 4 studios or create placeholders for missing ones
    studios_to_evaluate = []

    # Add extracted studios (up to 4)
    for i, studio in enumerate(parsed_studios.studios[:4]):
        studios_to_evaluate.append(studio)

    # Add placeholders for missing studios
    while len(studios_to_evaluate) < 4:
        studios_to_evaluate.append(create_placeholder_studio(len(studios_to_evaluate)))

    # Record extraction summary
    evaluator.add_custom_info({
        "total_studios_extracted": len(parsed_studios.studios),
        "studios_evaluated": 4,
        "studios_with_placeholders": 4 - min(len(parsed_studios.studios), 4)
    }, "extraction_summary")

    # -------- 4. Build verification tree -------------------------- #

    # Create nodes for each of the 4 required studios
    for i, studio in enumerate(studios_to_evaluate):
        await verify_studio_requirements(evaluator, root, studio, i)

    # -------- 5. Return evaluation results ------------------------ #
    return evaluator.get_summary()