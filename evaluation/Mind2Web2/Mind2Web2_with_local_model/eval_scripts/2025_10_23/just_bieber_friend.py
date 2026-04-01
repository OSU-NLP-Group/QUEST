import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "just_bieber_friend"
TASK_DESCRIPTION = """
Could you remind me of the name of the YouTube influencer who is a friend of Justin Bieber and plays a supporting role in a DC superhero movie? Please provide his name, the title of the movie along with its information page listing the cast, a link to his YouTube channel, and a link to an article or webpage that mentions his relationship with Justin Bieber.
"""

EVAL_NOTES = ""  # No additional evaluation notes provided
GROUND_TRUTH = {}  # No ground truth provided


class PersonBieberInfo(BaseModel):
    """Extracted information about the person and their relationship with Justin Bieber."""
    name: Optional[str] = Field(default=None, description="Name of the YouTube influencer")
    bieber_relationship_urls: Optional[List[str]] = Field(default_factory=list,
                                                          description="All URLs that might mention their relationship with Justin Bieber")


class MovieInfo(BaseModel):
    """Extracted information about the movie."""
    movie_title: Optional[str] = Field(default=None, description="Title of the DC superhero movie")
    movie_info_urls: Optional[List[str]] = Field(default_factory=list,
                                                 description="All URLs that contain movie cast information")


class YoutubeInfo(BaseModel):
    """Extracted YouTube channel information."""
    youtube_channel_urls: Optional[List[str]] = Field(default_factory=list,
                                                      description="All YouTube channel URLs mentioned")


def prompt_extract_person_bieber() -> str:
    """Extraction prompt for person name and Bieber relationship."""
    return """
    Extract information about the YouTube influencer who is a friend of Justin Bieber.

    Look for:
    - name: The full name of the person mentioned as Justin Bieber's friend
    - bieber_relationship_urls: ALL URLs provided that might mention, discuss, or provide evidence of their relationship/friendship with Justin Bieber

    Extract information exactly as it appears in the text.
    Include ALL relevant URLs, even if they seem redundant.
    If any field is not mentioned, set it to null.
    """


def prompt_extract_movie_info(person_name: str) -> str:
    """Extraction prompt for movie information."""
    return f"""
    Extract information about the DC superhero movie that {person_name} appears in.

    Look for:
    - movie_title: The exact title of the DC superhero movie
    - movie_info_urls: ALL URLs provided that show movie information, cast lists, or movie details

    Extract information exactly as it appears in the text.
    Include ALL relevant URLs that might contain cast information.
    If any field is not mentioned, set it to null.
    """


def prompt_extract_youtube() -> str:
    """Extraction prompt for YouTube channel."""
    return """
    Extract YouTube channel information for the person.

    Look for:
    - youtube_channel_urls: ALL YouTube channel URLs mentioned in the answer

    Include every YouTube channel link provided.
    If no YouTube URLs are found, return empty list.
    """


async def verify_step1_core_requirements(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        person_info: PersonBieberInfo,
        movie_info: MovieInfo,
) -> None:
    """Step 1: Verify core requirements - person identity and key relationships."""

    step1_node = evaluator.add_parallel(
        id="step1_core_requirements",
        desc="Core requirements: Person identity and key relationships",
        parent=parent_node,
        critical=False,  # Non-critical as specified
    )

    # Existence check for person and URLs
    existence_check = evaluator.add_custom_node(
        result=bool(
            person_info.name and
            person_info.name.strip() and
            person_info.bieber_relationship_urls and movie_info.movie_title
        ),
        id="person_and_urls_exist",
        desc="Person name and relationship URLs are provided",
        parent=step1_node,
        critical=True,  # Critical within this step
    )

    # Verify Bieber friendship
    bieber_friend_node = evaluator.add_leaf(
        id="bieber_friendship_verification",
        desc=f"{person_info.name or 'The person'} is a friend of Justin Bieber",
        parent=step1_node,
        critical=True,
    )

    # if person_info.name and person_info.bieber_relationship_urls:
    claim = f"{person_info.name} has a friendship or close relationship with Justin Bieber"
    await evaluator.verify(
        claim=claim,
        node=bieber_friend_node,
        sources=person_info.bieber_relationship_urls,
        additional_instruction="Look for evidence of friendship or close relationship. This could include mentions of friendship, collaborations, hanging out together, social media interactions, or being described as friends."
    )

    # Verify DC superhero movie supporting role
    dc_movie_node = evaluator.add_leaf(
        id="dc_movie_supporting_role",
        desc=f"{person_info.name or 'The person'} plays a supporting role in a DC superhero movie",
        parent=step1_node,
        critical=True,
    )

    # if person_info.name and movie_info.movie_info_urls:
    claim = f"{person_info.name} appears in '{movie_info.movie_title}' in a supporting role (not a main/lead role)"
    await evaluator.verify(
        claim=claim,
        node=dc_movie_node,
        sources=movie_info.movie_info_urls,
        additional_instruction="Verify the person appears in a SUPPORTING role, not as a main character or lead. The role should be more than just a cameo but not a primary character. Also confirm this is a DC superhero movie."
    )


async def verify_step2_additional_info(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        person_info: PersonBieberInfo,
        movie_info: MovieInfo,
        youtube_info: YoutubeInfo,
) -> None:
    """Step 2: Verify additional information - YouTube and movie details."""

    step2_node = evaluator.add_parallel(
        id="step2_additional_info",
        desc="Additional information: YouTube channel and movie details",
        parent=parent_node,
        critical=False,  # Non-critical as specified
    )

    # Existence check for all required information
    # all_info_exists = evaluator.add_custom_node(
    #     result=bool(
    #         youtube_info.youtube_channel_urls and
    #         movie_info.movie_info_urls and
    #         movie_info.movie_title and person_info.name
    #     ),
    #     node_id="additional_info_exists",
    #     description="YouTube channel and movie information are provided",
    #     parent=step2_node,
    #     critical=True,
    # )

    # Verify YouTube channel
    if youtube_info.youtube_channel_urls:
        youtube_node = evaluator.add_leaf(
            id="youtube_channel_verification",
            desc=f"YouTube channel is provided and belongs to {person_info.name or 'the person'}",
            parent=step2_node,
            critical=False,
        )

        # if person_info.name and youtube_info.youtube_channel_urls:
        claim = f"This is a YouTube channel page and it belongs to {person_info.name}"
        await evaluator.verify(
            claim=claim,
            node=youtube_node,
            sources=youtube_info.youtube_channel_urls,
            additional_instruction="Verify this is the official YouTube channel of the person. Look for channel name, about section, or any identifying information."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="youtube_channel_verification",
            desc=f"YouTube channel is provided and belongs to {person_info.name or 'the person'}",
            parent=step2_node,
            critical=False
        )

    # Verify movie cast page and DC movie status
    if movie_info.movie_title and movie_info.movie_info_urls:
        movie_cast_node = evaluator.add_leaf(
            id="movie_cast_page_dc_verification",
            desc=f"Movie cast page shows {person_info.name or 'the person'} in {movie_info.movie_title or 'the movie'}, which is a DC superhero movie",
            parent=step2_node,
            critical=False,
        )

        # if person_info.name and movie_info.movie_title and movie_info.movie_info_urls:
        claim = f"This is a cast information page, and it shows {person_info.name} in '{movie_info.movie_title}', and '{movie_info.movie_title}' is a DC superhero movie"
        await evaluator.verify(
            claim=claim,
            node=movie_cast_node,
            sources=movie_info.movie_info_urls,
            additional_instruction="First verify the person appears in the cast list. Then use your knowledge or the information on the webpage to confirm this is genuinely a DC superhero movie (from DC Comics, DCEU, or DC-related productions)."
        )
    else:
        evaluator.add_custom_node(
            result=False,
            id="movie_cast_page_dc_verification",
            desc=f"Movie cast page shows {person_info.name or 'the person'} in {movie_info.movie_title or 'the movie'}, which is a DC superhero movie",
            parent=step2_node,
            critical=False
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
        model: str = "o4-mini"
) -> Dict[str, Any]:
    """
    Main evaluation function for the Justin Bieber friend task.

    Sequential evaluation in two steps:
    1. Core requirements: Person identity and key relationships
    2. Additional information: YouTube channel and movie details
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # Sequential as specified
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

    # First extraction: Person name and Bieber relationship
    person_bieber_info = await evaluator.extract(
        prompt=prompt_extract_person_bieber(),
        template_class=PersonBieberInfo,
        extraction_name="person_bieber_extraction",
    )

    # Second extraction: Movie information (using person name if available)
    movie_info = await evaluator.extract(
        prompt=prompt_extract_movie_info(person_bieber_info.name or "the person"),
        template_class=MovieInfo,
        extraction_name="movie_extraction",
    )

    # Third extraction: YouTube channel
    youtube_info = await evaluator.extract(
        prompt=prompt_extract_youtube(),
        template_class=YoutubeInfo,
        extraction_name="youtube_extraction",
    )

    # -------- 3. Build verification tree -------------------------- #

    # Step 1: Core requirements (non-critical)
    await verify_step1_core_requirements(evaluator, root, person_bieber_info, movie_info)

    # Step 2: Additional information (non-critical)
    await verify_step2_additional_info(evaluator, root, person_bieber_info, movie_info, youtube_info)

    # -------- 4. Return evaluation results ------------------------ #
    return evaluator.get_summary()