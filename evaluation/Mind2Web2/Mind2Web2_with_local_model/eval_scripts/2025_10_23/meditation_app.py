import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from mind2web2 import CacheFileSys, Evaluator, VerificationNode, AggregationStrategy, LLMClient

TASK_ID = "meditation_app"
TASK_DESCRIPTION = """
Identify two meditation apps from the U.S. Google Play Store that have more than 300,000 user ratings and job listings for software engineering-related positions online. For each app, provide its overall Play Store rating, a direct link to its Play Store page, the app's official website, and a direct link to a software engineering-related job listing.
"""

EVAL_NOTES = ""  # No additional notes provided
GROUND_TRUTH = {}  # No ground truth provided


class AppNameList(BaseModel):
    """List of app names extracted from the answer"""
    app_names: List[str] = Field(default_factory=list, description="Names of meditation apps mentioned")


class SingleAppInfo(BaseModel):
    """Detailed information about a single meditation app"""
    name: Optional[str] = Field(default=None, description="Name of the meditation app")
    play_store_rating: Optional[str] = Field(default=None, description="Overall Play Store rating")
    play_store_url: Optional[str] = Field(default=None, description="Direct link to Play Store page")
    official_website: Optional[str] = Field(default=None, description="App's official website")
    job_listing_url: Optional[str] = Field(default=None, description="Direct link to software engineering job listing")


def prompt_extract_app_names() -> str:
    """Extraction prompt for getting just the app names"""
    return """
    Extract ONLY the names of meditation apps mentioned in the answer.

    The task asked for two meditation apps from the U.S. Google Play Store.

    Return a list of app names exactly as they appear in the answer.
    Do not include any other information - just the app names.
    """


def prompt_extract_single_app_info(app_name: str) -> str:
    """Extraction prompt for getting detailed info about a specific app"""
    return f"""
    Extract detailed information about the meditation app "{app_name}" from the answer.

    Look for and extract:
    - name: The exact name of the app (should be "{app_name}" or very similar)
    - play_store_rating: The overall Play Store rating (as a string, e.g., "4.5", "4.5/5", "4.5 stars")
    - play_store_url: The direct link to its Play Store page
    - official_website: The app's official website URL
    - job_listing_url: The direct link to a software engineering related job listing

    Extract information exactly as it appears in the text.
    If any field is not mentioned, set it to null.
    Focus ONLY on information related to "{app_name}".
    """


async def verify_single_app(
        evaluator: Evaluator,
        parent_node: VerificationNode,
        app_info: SingleAppInfo,
        app_index: int,
) -> None:
    """Verify all requirements for a single meditation app"""

    # Create a parent node for this app
    app_node = evaluator.add_parallel(
        id=f"app_{app_index}",
        desc=f"Meditation app #{app_index}: {app_info.name or 'Unknown'}",
        parent=parent_node,
        critical=False,  # Non-critical to allow partial credit
    )

    # 1. Combined existence check for all required information
    all_info_exists = (
            bool(app_info.name and app_info.name.strip()) and
            bool(app_info.play_store_rating and app_info.play_store_rating.strip()) and
            bool(app_info.play_store_url and app_info.play_store_url.strip()) and
            bool(app_info.official_website and app_info.official_website.strip()) and
            bool(app_info.job_listing_url and app_info.job_listing_url.strip())
    )

    info_exists_node = evaluator.add_custom_node(
        result=all_info_exists,
        id=f"app_{app_index}_all_info_exists",
        desc=f"All required information exists for {app_info.name or 'app'} (name, rating, Play Store URL, website, job URL)",
        parent=app_node,
        critical=True,  # Critical - missing any info fails the app
    )

    # 2. Comprehensive Google Play Store page verification
    play_store_verification_node = evaluator.add_leaf(
        id=f"app_{app_index}_play_store_verification",
        desc=f"Google Play Store page verification for {app_info.name or 'app'}",
        parent=app_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"""Verify the following about the Google Play Store page:
        1. The page is a valid Google Play Store page for an app
        2. The page is for the app '{app_info.name}'
        3. The app is meditation-related (check app description, category, or title, or use your own knowledge if it's a well-known app)
        4. The app has more than 300,000 user ratings (look for ratings count like '300K+', '500,000', '1M+', etc.)
        5. The overall rating on the page matches '{app_info.play_store_rating}' (allow ±0.1 difference, focus only on the rating number, not review count)
        """,
        node=play_store_verification_node,
        sources=app_info.play_store_url,
        additional_instruction="Verify ALL 5 points. For point 5, if the answer says '4.5' and the page shows '4.4' or '4.6', that's acceptable. Only compare the numerical rating value, ignore the number of ratings because that is irrelevant and we don't care."
    )

    # official website verification
    official_website_verification_node = evaluator.add_leaf(
        id=f"app_{app_index}_official_website_verification",
        desc=f"Official website verification for {app_info.name or 'app'}",
        parent=app_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"""Verify this webpage ({app_info.official_website}) is the official website of the app '{app_info.name}'""",
        node=official_website_verification_node,
        sources=app_info.official_website,
    )

    # 3. Comprehensive job listing verification
    job_listing_verification_node = evaluator.add_leaf(
        id=f"app_{app_index}_job_listing_verification",
        desc=f"Job listing verification for {app_info.name or 'app'}",
        parent=app_node,
        critical=True,
    )

    await evaluator.verify(
        claim=f"""Verify the following about the job listing page:
        1. The job listing is for the company/app '{app_info.name}' (check company name, about section, or context)
        2. The position is software engineering related (e.g., software engineer, developer, programmer, engineering manager, tech lead, QA engineer, DevOps, etc.)
        3. This page shows a specific job position — not a job search results page, job listings overview page, or general company careers page.
        """,
        node=job_listing_verification_node,
        sources=app_info.job_listing_url,
        additional_instruction="Be flexible about software engineering roles - accept any technical/programming related position. The company association can be direct or through parent company."
    )


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
    Main evaluation function for the meditation_app task.

    Evaluates whether the answer correctly identifies 2 meditation apps meeting all criteria:
    - From U.S. Google Play Store
    - More than 300,000 user ratings
    - Has software engineering job listings
    - Provides all required information (rating, Play Store link, website, job link)
    """

    # -------- 1. Initialize evaluator ----------------------------- #
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Apps are evaluated independently
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

    # -------- 2. Extract app names first -------------------------- #
    app_names = await evaluator.extract(
        prompt=prompt_extract_app_names(),
        template_class=AppNameList,
        extraction_name="app_names_extraction",
    )

    # -------- 3. Extract detailed info for each app --------------- #
    NUM_REQUIRED_APPS = 2
    detailed_apps = []

    # Extract details for each app found (up to NUM_REQUIRED_APPS)
    for i, app_name in enumerate(app_names.app_names[:NUM_REQUIRED_APPS]):
        app_info = await evaluator.extract(
            prompt=prompt_extract_single_app_info(app_name),
            template_class=SingleAppInfo,
            extraction_name=f"app_{i + 1}_details",
        )
        detailed_apps.append(app_info)

    # Pad with empty apps if fewer than required were found
    while len(detailed_apps) < NUM_REQUIRED_APPS:
        detailed_apps.append(SingleAppInfo())

    # -------- 4. Build verification tree -------------------------- #

    # Verify each app
    for i, app_info in enumerate(detailed_apps[:NUM_REQUIRED_APPS], 1):
        await verify_single_app(evaluator, root, app_info, i)

    # -------- 5. Return evaluation results ------------------------ #
    return evaluator.get_summary()