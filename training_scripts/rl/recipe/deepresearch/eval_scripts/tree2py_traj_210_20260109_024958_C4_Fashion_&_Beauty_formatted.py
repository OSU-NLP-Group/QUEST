import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "ulta_brand_launch_2024"
TASK_DESCRIPTION = (
    "In 2024, a British luxury makeup brand founded by a professional makeup artist launched at Ulta Beauty for the first time. "
    "This brand was founded in 2013 and entered 600 Ulta stores. Identify this brand and provide: "
    "(1) The brand name, (2) The founder's full name, (3) The year the brand was founded, "
    "(4) The exact date the brand launched online at Ulta Beauty, (5) The exact date the brand launched in physical Ulta Beauty stores, "
    "(6) The number of Ulta Beauty stores the brand entered, (7) One product category the brand offers, and (8) The country where the brand originated."
)


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class BrandLaunchInfo(BaseModel):
    brand_name: Optional[str] = None
    founder_full_name: Optional[str] = None
    founded_year: Optional[str] = None
    ulta_online_launch_date: Optional[str] = None
    ulta_in_store_launch_date: Optional[str] = None
    ulta_store_count: Optional[str] = None
    product_category: Optional[str] = None
    origin_country: Optional[str] = None

    # Source URLs explicitly mentioned in the answer
    brand_profile_sources: List[str] = Field(default_factory=list)  # For: luxury/prestige, origin country, founded year, founder identity (if applicable)
    founder_profile_sources: List[str] = Field(default_factory=list)  # For: founder is a professional makeup artist
    ulta_launch_sources: List[str] = Field(default_factory=list)  # For: Ulta debut, online date, in-store date, store count


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_brand_launch_info() -> str:
    return """
    Extract the following information exactly as presented in the answer text. Do not infer or invent any information:
    1. brand_name: The brand's name.
    2. founder_full_name: The full name of the brand's founder.
    3. founded_year: The year the brand was founded (return as a string; e.g., "2013").
    4. ulta_online_launch_date: The exact date the brand launched online at Ulta Beauty (return as seen; e.g., "February 4, 2024" or "Feb 4, 2024" or "2/4/2024").
    5. ulta_in_store_launch_date: The exact date the brand launched in physical Ulta Beauty stores (return as seen; e.g., "February 18, 2024" or "Feb 18, 2024" or "2/18/2024").
    6. ulta_store_count: The number of Ulta Beauty stores the brand entered (return as a string; e.g., "600").
    7. product_category: Provide one product category the brand offers (e.g., "makeup", "skincare", "lipstick"). If multiple categories are mentioned, choose one commonly recognized category from the answer.
    8. origin_country: The country where the brand originated (return as a string; e.g., "United Kingdom", "Britain", or "UK").

    Also extract the URL sources explicitly cited in the answer:
    - brand_profile_sources: URLs that support the brand's core profile details (e.g., founded year, origin country, luxury/prestige classification, founder identity if applicable). Only include URLs explicitly present in the answer.
    - founder_profile_sources: URLs specifically supporting that the founder is a professional makeup artist. Only include URLs explicitly present in the answer.
    - ulta_launch_sources: URLs that support the Ulta launch details (first time at Ulta, online launch date, in-store launch date, store count). Only include URLs explicitly present in the answer.

    Rules:
    - Return null for any field not mentioned in the answer.
    - For URLs: extract only valid URLs that are explicitly present in the answer (including markdown links); do not infer URLs.
    - Keep all fields as strings (dates, numbers) exactly as shown.
    """


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
def safe_sources(urls: Optional[List[str]]) -> List[str]:
    """Normalize sources to a clean list of non-empty strings."""
    if not urls:
        return []
    return [u.strip() for u in urls if isinstance(u, str) and u.strip()]


# -----------------------------------------------------------------------------
# Verification builders (subtrees)
# -----------------------------------------------------------------------------
async def build_basic_fields_checks(evaluator: Evaluator, parent_node, info: BrandLaunchInfo) -> None:
    """
    Basic required attributes presence and the founded year requirement.
    All children are critical to satisfy the rubric's critical root.
    """
    group = evaluator.add_parallel(
        id="basic_fields",
        desc="Basic required attributes provided",
        parent=parent_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.brand_name and info.brand_name.strip()),
        id="BrandName",
        desc="Provides the brand name for the brand described in the prompt.",
        parent=group,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.founder_full_name and info.founder_full_name.strip()),
        id="FounderFullName",
        desc="Provides the founder's full name for the brand named in the answer.",
        parent=group,
        critical=True,
    )

    evaluator.add_custom_node(
        result=bool(info.product_category and info.product_category.strip()),
        id="ProductCategoryProvided",
        desc="Provides at least one valid product category offered by the brand (e.g., makeup, skincare).",
        parent=group,
        critical=True,
    )

    # FoundedYearIs2013 (verification). Not gated by sources; will use sources if available.
    node_founded_2013 = evaluator.add_leaf(
        id="FoundedYearIs2013",
        desc="States that the brand named in the answer was founded in 2013.",
        parent=group,
        critical=True,
    )
    claim_2013 = f"The brand '{info.brand_name or ''}' was founded in 2013."
    await evaluator.verify(
        claim=claim_2013,
        node=node_founded_2013,
        sources=safe_sources(info.brand_profile_sources),
        additional_instruction="Confirm the founded year is explicitly given as 2013. If the cited page states a different year, mark incorrect.",
    )


async def build_brand_profile_support(evaluator: Evaluator, parent_node, info: BrandLaunchInfo) -> None:
    """
    Brand profile support checks: Luxury/Prestige and Origin Country (UK).
    Gated by presence of brand_profile_sources.
    """
    group = evaluator.add_parallel(
        id="profile_support",
        desc="Brand profile supported by cited sources",
        parent=parent_node,
        critical=True,
    )

    # Gate: sources provided
    evaluator.add_custom_node(
        result=len(safe_sources(info.brand_profile_sources)) > 0,
        id="brand_profile_sources_provided",
        desc="Brand profile sources are provided in the answer.",
        parent=group,
        critical=True,
    )

    # Luxury/Prestige verification
    node_luxury = evaluator.add_leaf(
        id="LuxuryPrestige",
        desc="Indicates (with support) that the brand named in the answer is a luxury/prestige beauty brand.",
        parent=group,
        critical=True,
    )
    claim_luxury = f"Brand '{info.brand_name or ''}' is a luxury or prestige beauty brand."
    await evaluator.verify(
        claim=claim_luxury,
        node=node_luxury,
        sources=safe_sources(info.brand_profile_sources),
        additional_instruction="Confirm the source(s) explicitly describe the brand as 'luxury' or 'prestige'. Accept synonyms like 'high-end', 'premium', 'prestige', 'luxury'.",
    )

    # Origin Country: UK/Britain verification
    node_origin = evaluator.add_leaf(
        id="OriginCountryUK",
        desc="States the country where the brand originated and it is the United Kingdom (Britain).",
        parent=group,
        critical=True,
    )
    claim_origin = f"Brand '{info.brand_name or ''}' originated in the United Kingdom (Britain)."
    await evaluator.verify(
        claim=claim_origin,
        node=node_origin,
        sources=safe_sources(info.brand_profile_sources),
        additional_instruction="Treat 'British brand' or 'UK-based' as equivalent to 'originated in the United Kingdom (Britain)'.",
    )


async def build_founder_profile_support(evaluator: Evaluator, parent_node, info: BrandLaunchInfo) -> None:
    """
    Founder background support: founder is a professional makeup artist.
    Gated by founder_profile_sources presence.
    """
    group = evaluator.add_parallel(
        id="founder_support",
        desc="Founder professional background supported by cited sources",
        parent=parent_node,
        critical=True,
    )

    # Gate: sources provided
    evaluator.add_custom_node(
        result=len(safe_sources(info.founder_profile_sources)) > 0,
        id="founder_profile_sources_provided",
        desc="Founder professional background sources are provided in the answer.",
        parent=group,
        critical=True,
    )

    node_founder_mua = evaluator.add_leaf(
        id="FounderIsProfessionalMakeupArtist",
        desc="Indicates (with support) that the founder named in the answer is a professional makeup artist.",
        parent=group,
        critical=True,
    )
    claim_mua = f"The founder '{info.founder_full_name or ''}' is a professional makeup artist."
    await evaluator.verify(
        claim=claim_mua,
        node=node_founder_mua,
        sources=safe_sources(info.founder_profile_sources),
        additional_instruction="Treat 'makeup artist', 'MUA', 'celebrity makeup artist', or similar phrasing as equivalent to 'professional makeup artist'.",
    )


async def build_ulta_launch_support(evaluator: Evaluator, parent_node, info: BrandLaunchInfo) -> None:
    """
    Ulta Beauty 2024 launch details: first time at Ulta, online date Feb 4, 2024,
    in-store date Feb 18, 2024, and entered exactly 600 stores.
    Gated by ulta_launch_sources presence.
    """
    group = evaluator.add_parallel(
        id="ulta_launch_support",
        desc="Ulta Beauty 2024 launch details supported by cited sources",
        parent=parent_node,
        critical=True,
    )

    # Gate: sources provided
    evaluator.add_custom_node(
        result=len(safe_sources(info.ulta_launch_sources)) > 0,
        id="ulta_launch_sources_provided",
        desc="Ulta launch sources are provided in the answer.",
        parent=group,
        critical=True,
    )

    # First-time at Ulta
    node_first_time = evaluator.add_leaf(
        id="FirstTimeAtUlta",
        desc="Indicates (with support) that the 2024 launch was the brand's first launch at Ulta Beauty.",
        parent=group,
        critical=True,
    )
    claim_first_time = f"In 2024, {info.brand_name or 'the brand'} launched at Ulta Beauty for the first time."
    await evaluator.verify(
        claim=claim_first_time,
        node=node_first_time,
        sources=safe_sources(info.ulta_launch_sources),
        additional_instruction="The source should mention that this is the brand's first presence at Ulta Beauty (e.g., 'first time at Ulta', 'Ulta debut'). If the source indicates prior availability at Ulta, mark incorrect.",
    )

    # Online launch date: February 4, 2024
    node_online_date = evaluator.add_leaf(
        id="UltaOnlineLaunchDateFeb4_2024",
        desc="States that the brand launched online at Ulta Beauty on February 4, 2024.",
        parent=group,
        critical=True,
    )
    claim_online_date = "The brand launched online at Ulta Beauty on February 4, 2024."
    await evaluator.verify(
        claim=claim_online_date,
        node=node_online_date,
        sources=safe_sources(info.ulta_launch_sources),
        additional_instruction="Accept 'February 4, 2024', 'Feb 4, 2024', or '2/4/2024'. The page must clearly tie this date to the online launch at Ulta Beauty.",
    )

    # In-store launch date: February 18, 2024
    node_instore_date = evaluator.add_leaf(
        id="UltaInStoreLaunchDateFeb18_2024",
        desc="States that the brand launched in physical Ulta Beauty stores on February 18, 2024.",
        parent=group,
        critical=True,
    )
    claim_instore_date = "The brand launched in physical Ulta Beauty stores on February 18, 2024."
    await evaluator.verify(
        claim=claim_instore_date,
        node=node_instore_date,
        sources=safe_sources(info.ulta_launch_sources),
        additional_instruction="Accept 'February 18, 2024', 'Feb 18, 2024', or '2/18/2024'. The page must clearly tie this date to the in-store launch.",
    )

    # Entered exactly 600 Ulta stores
    node_store_count = evaluator.add_leaf(
        id="EnteredExactly600Stores",
        desc="States that the brand entered exactly 600 Ulta Beauty stores.",
        parent=group,
        critical=True,
    )
    claim_store_count = "The brand entered exactly 600 Ulta Beauty stores."
    await evaluator.verify(
        claim=claim_store_count,
        node=node_store_count,
        sources=safe_sources(info.ulta_launch_sources),
        additional_instruction="Require the exact number '600'. Do not accept 'over 600', 'around 600', or a different number.",
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for the Ulta Beauty brand launch task using the Mind2Web2 framework.
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
    info: BrandLaunchInfo = await evaluator.extract(
        prompt=prompt_extract_brand_launch_info(),
        template_class=BrandLaunchInfo,
        extraction_name="brand_launch_info",
    )

    # Build main critical node as per rubric
    main_node = evaluator.add_parallel(
        id="UltaBeautyBrandLaunch2024",
        desc="Evaluate the identified brand and required attributes against the question and stated constraints.",
        parent=root,
        critical=True,
    )

    # Build subtrees according to rubric items
    await build_basic_fields_checks(evaluator, main_node, info)
    await build_brand_profile_support(evaluator, main_node, info)
    await build_founder_profile_support(evaluator, main_node, info)
    await build_ulta_launch_support(evaluator, main_node, info)

    # Return structured summary
    return evaluator.get_summary()