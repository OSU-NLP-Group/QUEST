import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "chef_dc_2star_restaurant"
TASK_DESCRIPTION = """
A chef won the James Beard Humanitarian of the Year award in 2018. This chef founded a nonprofit disaster relief organization in 2010 in response to the Haiti earthquake. In 2017, this organization responded to Hurricane Maria in Puerto Rico and served nearly 4 million meals to residents affected by the disaster. The chef also operates a restaurant group that includes a restaurant with 2 Michelin stars located in Washington, DC, specifically at 855 E St NW in the Penn Quarter neighborhood. What is the name of this 2 Michelin-starred restaurant?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RestaurantTaskExtraction(BaseModel):
    # Core answer fields
    restaurant_name: Optional[str] = None
    address_string: Optional[str] = None  # e.g., "855 E St NW, Washington, DC 20004"
    neighborhood: Optional[str] = None  # e.g., "Penn Quarter"

    # URLs explicitly cited in the answer (by category, if discernible)
    restaurant_urls: List[str] = Field(default_factory=list)  # Michelin, official site, maps, articles about the restaurant
    address_urls: List[str] = Field(default_factory=list)     # pages that explicitly show address/neighborhood
    group_urls: List[str] = Field(default_factory=list)       # chef's restaurant group pages or profiles
    nonprofit_urls: List[str] = Field(default_factory=list)   # nonprofit (e.g., WCK) pages supporting founding year, mission, responses
    other_urls: List[str] = Field(default_factory=list)       # any other URLs cited in the answer

    # Entities (if mentioned in the answer)
    chef_name: Optional[str] = None
    chef_group_name: Optional[str] = None
    nonprofit_name: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_core() -> str:
    return """
    Extract the final restaurant answer and any supporting URLs explicitly cited in the answer text.

    Required fields:
    1) restaurant_name: The single, specific restaurant name that the answer claims is the 2‑Michelin‑star restaurant in Washington, DC.
    2) address_string: If the answer mentions a specific street address for that restaurant (e.g., "855 E St NW, Washington, DC"), extract it exactly as written; otherwise null.
    3) neighborhood: If the answer mentions the neighborhood for that restaurant (e.g., "Penn Quarter"), extract it; otherwise null.

    URL fields (ONLY include URLs explicitly present in the answer; do not invent):
    4) restaurant_urls: URLs that specifically reference the named restaurant (e.g., the Michelin Guide page, the restaurant’s official site, Google Maps listing, news articles about the restaurant).
    5) address_urls: URLs that can verify the restaurant’s address or neighborhood if provided (can overlap with restaurant_urls).
    6) group_urls: URLs about the chef’s restaurant group or profiles that connect the chef and the restaurant group.
    7) nonprofit_urls: URLs about the nonprofit disaster relief organization (e.g., founding year, Haiti earthquake response, Hurricane Maria response, meal counts).
    8) other_urls: Any other URLs the answer cites that are relevant context (avoid duplicates if possible).

    Optional entity fields (extract only if explicitly mentioned in the answer):
    9) chef_name: The chef’s name associated with the restaurant group that includes the named restaurant.
    10) chef_group_name: The name of that restaurant group.
    11) nonprofit_name: The name of the nonprofit disaster relief organization founded by the chef.

    Rules:
    - Return null for any missing string fields.
    - Return [] for any missing URL arrays.
    - Extract ONLY URLs that are explicitly present in the answer text (plain URLs or markdown links).
    - Do not infer or add URLs that the answer did not provide.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(*url_lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for urls in url_lists:
        for u in urls:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _display_or_fallback(value: Optional[str], fallback: str) -> str:
    v = (value or "").strip()
    return v if v else fallback


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extraction: RestaurantTaskExtraction) -> None:
    # Root (non-leaf) under evaluator.root; we make a critical sequential block that mirrors the rubric's "Root"
    main = evaluator.add_sequential(
        id="Root",
        desc="Provide the name of the 2 Michelin-starred restaurant that satisfies all given constraints (chef/organization history and DC address details).",
        parent=evaluator.root,
        critical=True,
    )

    # Normalize sources
    sources_restaurant = _dedup_urls(extraction.restaurant_urls, extraction.address_urls)
    sources_group = _dedup_urls(extraction.group_urls)
    sources_nonprofit = _dedup_urls(extraction.nonprofit_urls)
    sources_all = _dedup_urls(sources_restaurant, sources_group, sources_nonprofit, extraction.other_urls)

    restaurant_name = _display_or_fallback(extraction.restaurant_name, "<missing>")
    address_in_prompt = "855 E St NW, Washington, DC"
    neighborhood_in_prompt = "Penn Quarter"

    chef_display = _display_or_fallback(extraction.chef_name, "the chef associated with the named restaurant")
    group_display = _display_or_fallback(extraction.chef_group_name, "the restaurant group associated with that chef")
    nonprofit_display = _display_or_fallback(extraction.nonprofit_name, "the nonprofit disaster relief organization founded by the chef")

    # 1) RestaurantNameProvided (leaf)
    node_name_provided = evaluator.add_leaf(
        id="RestaurantNameProvided",
        desc="Answer provides a single, specific restaurant name (not a chef name, organization name, or restaurant group name).",
        parent=main,
        critical=True,
    )
    claim_name_provided = (
        f"The answer explicitly provides a single, specific restaurant name rather than a chef, nonprofit, or restaurant group; "
        f"the extracted restaurant name is '{restaurant_name}'."
    )
    await evaluator.verify(
        claim=claim_name_provided,
        node=node_name_provided,
        sources=None,
        additional_instruction="Judge only whether the provided name is a single, specific restaurant (not a person, nonprofit, or group). Ignore correctness of other constraints here."
    )

    # 2) RestaurantMeetsAllConstraints (parallel, all children critical)
    all_constraints = evaluator.add_parallel(
        id="RestaurantMeetsAllConstraints",
        desc="The named restaurant satisfies all constraints stated in the prompt/constraints.",
        parent=main,
        critical=True,
    )

    # Prepare leaves
    node_two_stars = evaluator.add_leaf(
        id="HasTwoMichelinStars",
        desc="The named restaurant has 2 Michelin stars.",
        parent=all_constraints,
        critical=True,
    )
    node_address = evaluator.add_leaf(
        id="LocatedAtSpecifiedAddressAndNeighborhood",
        desc="The named restaurant is located in Washington, DC at 855 E St NW in the Penn Quarter neighborhood.",
        parent=all_constraints,
        critical=True,
    )
    node_intimate = evaluator.add_leaf(
        id="RestaurantNameReflectsIntimateSmallScaleDiningConcept",
        desc="The named restaurant's name reflects an intimate, small-scale dining concept.",
        parent=all_constraints,
        critical=True,
    )
    node_jbf_2018 = evaluator.add_leaf(
        id="ChefWonJamesBeardHumanitarianOfTheYear2018",
        desc="The chef associated with operating the restaurant group that includes the named restaurant won the James Beard Humanitarian of the Year award in 2018.",
        parent=all_constraints,
        critical=True,
    )
    node_founded_2010 = evaluator.add_leaf(
        id="ChefFoundedNonprofitIn2010",
        desc="That chef founded a nonprofit disaster relief organization in 2010.",
        parent=all_constraints,
        critical=True,
    )
    node_haiti = evaluator.add_leaf(
        id="NonprofitFoundedInResponseToHaitiEarthquake2010",
        desc="That nonprofit was founded in response to the 2010 Haiti earthquake.",
        parent=all_constraints,
        critical=True,
    )
    node_maria_2017 = evaluator.add_leaf(
        id="NonprofitRespondedToHurricaneMariaInPuertoRico2017",
        desc="That nonprofit responded to Hurricane Maria in Puerto Rico in 2017.",
        parent=all_constraints,
        critical=True,
    )
    node_maria_meals = evaluator.add_leaf(
        id="NonprofitServedNearlyFourMillionMealsDuringMariaResponse",
        desc="During the Hurricane Maria response, that nonprofit served nearly 4 million meals to affected residents.",
        parent=all_constraints,
        critical=True,
    )
    node_group_approx40 = evaluator.add_leaf(
        id="ChefOperatesRestaurantGroupApprox40Restaurants",
        desc="That chef operates a restaurant group with approximately 40 restaurants.",
        parent=all_constraints,
        critical=True,
    )
    node_in_group = evaluator.add_leaf(
        id="NamedRestaurantIncludedInChefRestaurantGroup",
        desc="The named restaurant is included in (i.e., operated under) that chef's restaurant group.",
        parent=all_constraints,
        critical=True,
    )

    # Build claims and trigger verification (parallelized where appropriate)
    claims_and_sources: List[tuple[str, List[str] | None, Any, Optional[str]]] = []

    # HasTwoMichelinStars
    claim_two_stars = f"The restaurant '{restaurant_name}' has 2 Michelin stars."
    claims_and_sources.append((
        claim_two_stars,
        sources_restaurant if sources_restaurant else sources_all,
        node_two_stars,
        "Use only the provided URLs. Prefer Michelin Guide or other credible sources. Accept clear statements like 'two Michelin stars' or '2 stars'."
    ))

    # LocatedAtSpecifiedAddressAndNeighborhood
    claim_address = (
        f"The restaurant '{restaurant_name}' is located at {address_in_prompt} in Washington, DC, "
        f"and is in the {neighborhood_in_prompt} neighborhood."
    )
    claims_and_sources.append((
        claim_address,
        _dedup_urls(sources_restaurant, extraction.address_urls) or sources_all,
        node_address,
        "Allow minor formatting variants (e.g., 'Street' vs 'St', presence/absence of ZIP). The neighborhood must match 'Penn Quarter' in DC."
    ))

    # RestaurantNameReflectsIntimateSmallScaleDiningConcept
    # Note: We verify the restaurant is an intimate, small-scale dining concept (e.g., very limited seating / chef's counter),
    # which is what the prompt implies. This is better grounded than analyzing semantics of the name alone.
    claim_intimate = (
        f"The restaurant '{restaurant_name}' offers an intimate, small-scale dining experience "
        f"(e.g., highly limited seating like a small counter or chef's table)."
    )
    claims_and_sources.append((
        claim_intimate,
        sources_restaurant if sources_restaurant else sources_all,
        node_intimate,
        "Look for phrases indicating a very small number of seats, intimate chef's counter, or otherwise small-scale concept."
    ))

    # ChefWonJamesBeardHumanitarianOfTheYear2018
    claim_jbf_2018 = (
        f"In 2018, {chef_display} won the James Beard Foundation's Humanitarian of the Year award."
    )
    claims_and_sources.append((
        claim_jbf_2018,
        sources_group if sources_group else sources_all,
        node_jbf_2018,
        "Verify award year (2018) and award name. Any credible biography or award page among the provided URLs is acceptable."
    ))

    # ChefFoundedNonprofitIn2010
    claim_founded_2010 = (
        f"{chef_display} founded a nonprofit disaster relief organization "
        f"{'' if extraction.nonprofit_name else ''} in 2010."
        if extraction.nonprofit_name is None
        else f"{chef_display} founded the nonprofit {nonprofit_display} in 2010."
    )
    claims_and_sources.append((
        claim_founded_2010,
        sources_nonprofit if sources_nonprofit else sources_all,
        node_founded_2010,
        "Confirm that the chef founded the nonprofit in 2010."
    ))

    # NonprofitFoundedInResponseToHaitiEarthquake2010
    claim_haiti = f"{nonprofit_display} was founded in response to the 2010 Haiti earthquake."
    claims_and_sources.append((
        claim_haiti,
        sources_nonprofit if sources_nonprofit else sources_all,
        node_haiti,
        "Look for language indicating the founding was sparked by or in response to the 2010 Haiti earthquake."
    ))

    # NonprofitRespondedToHurricaneMariaInPuertoRico2017
    claim_maria_2017 = f"{nonprofit_display} responded to Hurricane Maria in Puerto Rico in 2017."
    claims_and_sources.append((
        claim_maria_2017,
        sources_nonprofit if sources_nonprofit else sources_all,
        node_maria_2017,
        "Confirm a response/relief effort in Puerto Rico following Hurricane Maria in 2017."
    ))

    # NonprofitServedNearlyFourMillionMealsDuringMariaResponse
    claim_maria_meals = (
        f"During its Hurricane Maria response in Puerto Rico, {nonprofit_display} served nearly 4 million meals."
    )
    claims_and_sources.append((
        claim_maria_meals,
        sources_nonprofit if sources_nonprofit else sources_all,
        node_maria_meals,
        "Allow approximate wording like 'nearly 4 million' or 'around 4 million'; counts like ~3.7–4.0 million should pass."
    ))

    # ChefOperatesRestaurantGroupApprox40Restaurants
    claim_group_approx40 = (
        f"{chef_display} operates a restaurant group {group_display} with approximately 40 restaurants."
    )
    claims_and_sources.append((
        claim_group_approx40,
        sources_group if sources_group else sources_all,
        node_group_approx40,
        "Treat 'approximately 40' flexibly (e.g., 'around 40', 'about 40'). If the page indicates high-30s to about-40, consider it supported."
    ))

    # NamedRestaurantIncludedInChefRestaurantGroup
    claim_in_group = (
        f"The restaurant '{restaurant_name}' is included in and operated under {group_display}."
    )
    claims_and_sources.append((
        claim_in_group,
        _dedup_urls(sources_group, sources_restaurant) or sources_all,
        node_in_group,
        "Look for the chef's group listing this restaurant or a clear statement that it is part of the group."
    ))

    # Run all constraint verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # root container; we add a critical sequential child node "Root"
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_core(),
        template_class=RestaurantTaskExtraction,
        extraction_name="core_extraction",
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, extraction)

    # Return structured summary
    return evaluator.get_summary()