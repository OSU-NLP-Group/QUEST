import asyncio
import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from mind2web2.evaluator import Evaluator, AggregationStrategy
from mind2web2.llm_client.base_client import LLMClient
from mind2web2.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "cat_adoption"
TASK_DESCRIPTION = """
Find female short-haired cats available for adoption near Redmond, WA (other locations in Washington State are also acceptable). Cats should be either kittens or young (generally 0-2 years old). Use the following pet adoption platforms to find three suitable cats from each:

- Petfinder
- Adopt-a-Pet

Provide direct links to each pet's adoption profile.
"""
JUDGE_MODEL = "o4-mini"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CatInfo(BaseModel):
    """Information about a single cat adoption profile."""
    name: Optional[str] = None
    description: Optional[str] = None


class PlatformCats(BaseModel):
    """Cats from a specific platform."""
    cats: List[CatInfo] = Field(default_factory=list)


class CatUrlInfo(BaseModel):
    """Links for a specific cat."""
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_platform_cats(platform: str) -> str:
    return f"""
    Extract information about all cats mentioned from the {platform} platform in the answer.

    For each cat from {platform}, extract:
    - name: The cat's name (if mentioned)
    - description: Any description provided about the cat

    Include all cats that are explicitly stated to be from {platform}.
    If no cat names are mentioned but there are links to {platform}, create entries with just null names.

    Return empty list if no cats from {platform} are mentioned.
    """


def prompt_extract_cat_urls(platform: str, cat_name: Optional[str] = None) -> str:
    if cat_name:
        return f"""
        Extract all URLs that are mentioned as links to adoption profiles for the cat named "{cat_name}" on the "{platform}" platform.
        Focus only on links that are explicitly associated with this specific cat on {platform}.
        Include all URLs that might refer to this cat, even if they're just labeled as "link" or "here", or merely cited in the introduction of this cat.
        Return the URLs as a list of strings.
        """
    else:
        return f"""
        Extract all URLs that are mentioned as links to cat adoption profiles on the "{platform}" platform.
        Include all links that are explicitly associated with cats on {platform}.
        Return the URLs as a list of strings.
        """


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #

async def verify_cat_meets_criteria(
        evaluator: Evaluator,
        parent_node,
        cat_index: int,
        platform: str,
        cat_info: CatInfo,
        urls: List[str],
) -> None:
    """
    Verify two aspects:
    1. That any non-None information extracted from the answer matches what's on the webpage
    2. That the cat on the webpage meets all required criteria
    """
    cat_identifier = cat_info.name if cat_info.name else f"Cat #{cat_index + 1}"

    evaluator.add_custom_node(
        result=bool(len(urls)>0),
        id=f"cat{cat_index + 1}_{platform}_exists",
        desc=f"Verify that the cat is provided with necessary url",
        parent=parent_node,
        critical=True
    )

    # 1. First verification: Text information matches webpage (if provided)
    text_info_node = evaluator.add_leaf(
        id=f"text_info_verification_cat{cat_index + 1}_{platform}",
        desc=f"Verify that provided information about {cat_identifier} matches the webpage",
        parent=parent_node,
        critical=True,
    )

    # Only perform this check if we have actual information to verify
    has_text_info = cat_info.name is not None or cat_info.description is not None

    if has_text_info:
        info_claims = []
        if cat_info.name:
            info_claims.append(f"The cat is named '{cat_info.name}'.")
        if cat_info.description:
            info_claims.append(f"Cat description: {cat_info.description}")

        text_info_claim = " ".join(info_claims)

        await evaluator.verify(
            claim=text_info_claim,
            node=text_info_node,
            sources=urls,
            additional_instruction="""
            Verify only that the provided information matches what's on the webpage.
            For the cat's name, check if the name matches exactly or with minor variations (e.g., capitalization).
            For descriptions, check if the key points match, not necessarily word-for-word.
            """
        )
    else:
        # If no text information was provided, consider this check passed
        text_info_node.score = 1.0
        text_info_node.status = "passed"

    # 2. Second verification: Cat meets all criteria
    cat_criteria_node = evaluator.add_leaf(
        id=f"cat_criteria_verification_cat{cat_index + 1}_{platform}",
        desc=f"Verify that {cat_identifier} meets all required criteria (female, short-haired, young/kitten, in Washington State)",
        parent=parent_node,
        critical=True,
    )

    criteria_claim = f"""
    The cat on this webpage meets ALL of the following criteria:
    1. The cat is female
    2. The cat has short hair
    3. The cat is a kitten or young cat (0-2 years old)
    4. The cat is located in Washington State
    5. The cat's profile is on the {platform} platform
    """

    await evaluator.verify(
        claim=criteria_claim,
        node=cat_criteria_node,
        sources=urls,
        additional_instruction="""
        Check if the cat meets ALL five criteria. The cat must:
        1. Be explicitly identified as female (F)
        2. Have short hair (may be described as short-haired, short coat, etc.)
        3. Be a kitten or young cat (typically described as baby, kitten, young, or with an age between 0-2 years)
        4. Be located in Washington State (WA, or a specific city in Washington)
        5. Be on a webpage that belongs to the specified platform

        The cat must meet ALL criteria to be considered verified.
        """
    )


async def verify_platform_cats(
        evaluator: Evaluator,
        platform: str,
        cats_info: List[CatInfo],
        cats_urls: List[List[str]],
) -> None:
    """
    Verify cats from a specific platform, ensuring exactly 3 cat nodes are created.
    """
    platform_node = evaluator.add_parallel(
        id=f"{platform}_platform",
        desc=f"Evaluation of cats from {platform}",
        critical=False,  # Not critical to allow partial credit
    )

    # Always create exactly 3 cat nodes
    for i in range(3):
        cat_name = cats_info[i].name if i < len(cats_info) and cats_info[i].name else None
        cat_identifier = cat_name if cat_name else f"Cat #{i + 1}"

        cat_node = evaluator.add_parallel(
            id=f"cat_{i + 1}_{platform}",
            desc=f"Evaluation of cat #{i + 1} from {platform}: {cat_identifier}",
            parent=platform_node,
            critical=False,  # Not critical to allow partial credit
        )

        # Check if we have a cat and URLs for this position
        if i < len(cats_info) and i < len(cats_urls) and len(cats_urls[i]) > 0:
            # We have a cat with URLs, perform normal verification
            await verify_cat_meets_criteria(
                evaluator=evaluator,
                parent_node=cat_node,
                cat_index=i,
                platform=platform,
                cat_info=cats_info[i],
                urls=cats_urls[i],
            )
        else:
            await verify_cat_meets_criteria(
                evaluator=evaluator,
                parent_node=cat_node,
                cat_index=i,
                platform=platform,
                cat_info=CatInfo(),
                urls=[],
            )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict:
    """
    Evaluate the answer for the cat adoption task.
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
        default_model=model if model else JUDGE_MODEL
    )

    # Define platforms
    platforms = ["Petfinder", "Adopt-a-Pet"]
    platform_cats = {}
    platform_urls = {}

    # Extract cats for each platform
    for platform in platforms:
        # Extract cats for this platform
        platform_cats_data = await evaluator.extract(
            prompt=prompt_extract_platform_cats(platform),
            template_class=PlatformCats,
            extraction_name=f"{platform}_cats"
        )
        platform_cats[platform] = platform_cats_data.cats

        # Extract URLs for each cat
        urls_for_platform = []

        # If we have cats, get URLs for each one
        if platform_cats[platform]:
            for cat in platform_cats[platform]:
                cat_urls = await evaluator.extract(
                    prompt=prompt_extract_cat_urls(platform, cat.name),
                    template_class=CatUrlInfo,
                    extraction_name=f"{platform}_{cat.name or 'unknown'}_urls"
                )
                urls_for_platform.append(cat_urls.urls if cat_urls and cat_urls.urls else [])
        else:
            # If no cats were extracted for this platform, URLs list will be empty
            # The verification will handle creating skipped nodes
            pass

        platform_urls[platform] = urls_for_platform

    # Add custom info for debugging
    evaluator.add_custom_info(
        {
            "platforms": platforms,
            "platform_cats": {p: [cat.dict() for cat in platform_cats[p]] for p in platforms},
            "platform_urls": platform_urls
        },
        "extraction_summary"
    )

    # Verify cats from each platform
    for platform in platforms:
        await verify_platform_cats(
            evaluator=evaluator,
            platform=platform,
            cats_info=platform_cats[platform],
            cats_urls=platform_urls[platform],
        )

    # Return the result using the new summary format
    return evaluator.get_summary()