import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bestbuy_highest_state_store"
TASK_DESCRIPTION = (
    "Identify the U.S. state that has the highest number of Best Buy store locations. "
    "Once you've identified this state, find one Best Buy store that is located in this state but is NOT within the city limits of either Los Angeles or San Francisco. "
    "Provide the complete street address, city, ZIP code, and include a reference URL from the official Best Buy store locator website confirming this location. "
    "Additionally, if available, provide the store number."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StateInfo(BaseModel):
    state_name: Optional[str] = None
    supporting_urls: List[str] = Field(default_factory=list)


class StoreInfo(BaseModel):
    state: Optional[str] = None
    city: Optional[str] = None
    street_address: Optional[str] = None
    zip_code: Optional[str] = None
    official_store_locator_url: Optional[str] = None
    store_number: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_state() -> str:
    return """
    Extract the specific U.S. state that the answer claims has the highest number of Best Buy store locations.
    Return:
    - state_name: The U.S. state named as having the highest number of Best Buy store locations. If not explicitly stated, return null.
    - supporting_urls: Any URLs provided in the answer that claim or support which state has the highest number of Best Buy locations. Extract only explicit URLs mentioned; if none are provided, return an empty list.
    """


def prompt_extract_store() -> str:
    return """
    Extract one Best Buy store referenced in the answer (if multiple are given, use the first one mentioned).
    For this single store, return:
    - state: The state (e.g., CA, California) as stated in the answer for the store.
    - city: The city name of the store as stated in the answer.
    - street_address: The complete street address line as stated in the answer (e.g., '1234 Example Rd.').
    - zip_code: The ZIP code as stated in the answer.
    - official_store_locator_url: The URL to the official Best Buy store locator page for this exact store (e.g., a page on bestbuy.com or stores.bestbuy.com). If the answer provides multiple URLs, choose the one that most clearly corresponds to the store’s official listing. If the answer does not provide such a URL, return null.
    - store_number: The store number if the answer includes it (e.g., 'Store #123'). If not provided, return null.

    Important:
    - Extract only what is explicitly present in the answer. Do not invent or infer any fields.
    - If a field is missing in the answer, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(val: Optional[str]) -> str:
    return val or ""


def _domain_hint_for_bestbuy(url: Optional[str]) -> str:
    """
    Provide a short hint to the verifier about expected domains.
    """
    return (
        "The official Best Buy store locator pages are typically hosted on 'bestbuy.com' (e.g., '/site/store-locator/') "
        "or 'stores.bestbuy.com'. If the URL is not on these domains, it is likely not an official store locator listing."
    )


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_state_identification(
    evaluator: Evaluator,
    parent_node,
    state_info: StateInfo,
) -> None:
    """
    Build and verify the 'state identification' part:
    - Ensure the answer names a state for 'highest number of Best Buy locations'.
    - Verify the answer itself claims that this named state has the highest number (we only check that the claim is made).
    """
    state_node = evaluator.add_parallel(
        id="state_identification",
        desc="Names the U.S. state that has the highest number of Best Buy store locations.",
        parent=parent_node,
        critical=True,  # Critical step in the overall task
    )

    # Existence check (critical)
    has_state = bool(state_info and state_info.state_name and state_info.state_name.strip())
    evaluator.add_custom_node(
        result=has_state,
        id="state_name_provided",
        desc="A specific U.S. state is named in the answer for 'highest number of Best Buy locations'.",
        parent=state_node,
        critical=True,
    )

    # Verify the answer actually claims this state has the highest number
    # (We only check that the claim is present in the answer; not external factual correctness.)
    claim_node = evaluator.add_leaf(
        id="state_highest_claim_in_answer",
        desc="The answer explicitly claims the named state has the highest number of Best Buy locations.",
        parent=state_node,
        critical=True,
    )
    claim_text = (
        f"The answer states or implies that '{_safe_str(state_info.state_name)}' "
        f"has the highest number of Best Buy store locations among U.S. states."
    )
    await evaluator.verify(
        claim=claim_text,
        node=claim_node,
        additional_instruction="Only determine whether the provided answer text claims this; do not verify factual correctness."
    )


async def verify_store_selection(
    evaluator: Evaluator,
    parent_node,
    identified_state: Optional[str],
    store_info: StoreInfo,
) -> None:
    """
    Build and verify the 'store selection' part:
    - The store must be in the identified state.
    - It must not be within Los Angeles or San Francisco city limits.
    - Provide complete store details and confirm them via official Best Buy store locator URL.
    """
    store_node = evaluator.add_parallel(
        id="store_selection",
        desc="Selects one Best Buy store in the identified state that is not within Los Angeles or San Francisco city limits, and provides required store details and confirmation URL.",
        parent=parent_node,
        critical=True,  # Core part must be correct to succeed the overall task
    )

    # Group: Store details (address, city, zip, official URL) - critical
    details_node = evaluator.add_parallel(
        id="store_details",
        desc="Provides complete and verifiable store information required by the question.",
        parent=store_node,
        critical=True,
    )

    # 1) Official store locator URL must be verifiable first (critical)
    official_url_leaf = evaluator.add_leaf(
        id="official_store_locator_url",
        desc="Provides a URL from the official Best Buy store locator website that confirms the store location details.",
        parent=details_node,
        critical=True,
    )
    official_url = store_info.official_store_locator_url
    official_url_claim = (
        "This webpage is an official Best Buy store locator listing for a specific store location, "
        "hosted on either bestbuy.com (e.g., '/site/store-locator/') or stores.bestbuy.com."
    )
    await evaluator.verify(
        claim=official_url_claim,
        node=official_url_leaf,
        sources=official_url,
        additional_instruction=(
            "Judge this claim by checking the URL and page content. "
            + _domain_hint_for_bestbuy(official_url)
        ),
    )

    # 2) Street address (critical)
    street_leaf = evaluator.add_leaf(
        id="street_address",
        desc="Provides the complete street address of the store.",
        parent=details_node,
        critical=True,
    )
    street_claim = f"The store’s street address shown on the official page is '{_safe_str(store_info.street_address)}'."
    await evaluator.verify(
        claim=street_claim,
        node=street_leaf,
        sources=official_url,
        additional_instruction="Match the address as displayed on the page. Minor formatting differences are acceptable (e.g., abbreviations like Rd vs. Road).",
    )

    # 3) City (critical)
    city_leaf = evaluator.add_leaf(
        id="city",
        desc="Provides the city of the store.",
        parent=details_node,
        critical=True,
    )
    city_claim = f"The store’s city shown on the official page is '{_safe_str(store_info.city)}'."
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        sources=official_url,
        additional_instruction="Verify that the city name on the page matches the provided city.",
    )

    # 4) ZIP code (critical)
    zip_leaf = evaluator.add_leaf(
        id="zip_code",
        desc="Provides the ZIP code of the store.",
        parent=details_node,
        critical=True,
    )
    zip_claim = f"The store’s ZIP/postal code shown on the official page is '{(_safe_str(store_info.zip_code))}'."
    await evaluator.verify(
        claim=zip_claim,
        node=zip_leaf,
        sources=official_url,
        additional_instruction="Verify that the ZIP code on the page matches the provided ZIP code.",
    )

    # Location must be in the identified state (critical)
    loc_state_leaf = evaluator.add_leaf(
        id="location_in_correct_state",
        desc="The selected store is located in the identified state.",
        parent=store_node,
        critical=True,
    )
    # Prefer store_info.state if provided; otherwise use identified_state
    expected_state = _safe_str(store_info.state) or _safe_str(identified_state)
    loc_state_claim = f"This store’s state shown on the official page is '{expected_state}'."
    await evaluator.verify(
        claim=loc_state_claim,
        node=loc_state_leaf,
        sources=official_url,
        additional_instruction="Check the state listed on the page. Minor variations (e.g., 'CA' vs 'California') should be considered equivalent.",
    )

    # Not within Los Angeles city limits (critical)
    not_la_leaf = evaluator.add_leaf(
        id="not_in_los_angeles_city_limits",
        desc="The selected store is NOT within Los Angeles city limits.",
        parent=store_node,
        critical=True,
    )
    not_la_claim = (
        "The store’s city as shown on the official page is not 'Los Angeles'. "
        f"It is listed as '{_safe_str(store_info.city)}'."
    )
    await evaluator.verify(
        claim=not_la_claim,
        node=not_la_leaf,
        sources=official_url,
        additional_instruction="If the page lists 'Los Angeles' as the city, the claim is not supported. Otherwise, the claim is supported.",
    )

    # Not within San Francisco city limits (critical)
    not_sf_leaf = evaluator.add_leaf(
        id="not_in_san_francisco_city_limits",
        desc="The selected store is NOT within San Francisco city limits.",
        parent=store_node,
        critical=True,
    )
    not_sf_claim = (
        "The store’s city as shown on the official page is not 'San Francisco'. "
        f"It is listed as '{_safe_str(store_info.city)}'."
    )
    await evaluator.verify(
        claim=not_sf_claim,
        node=not_sf_leaf,
        sources=official_url,
        additional_instruction="If the page lists 'San Francisco' as the city, the claim is not supported. Otherwise, the claim is supported.",
    )


async def verify_store_optional_info(
    evaluator: Evaluator,
    parent_node,
    store_info: StoreInfo,
) -> None:
    """
    Optional information: store number if available (non-critical).
    """
    optional_node = evaluator.add_parallel(
        id="store_optional_info",
        desc="Optional store information.",
        parent=parent_node,
        critical=False,
    )

    # Single leaf as per rubric: store number if available (non-critical)
    store_num_leaf = evaluator.add_leaf(
        id="store_number_if_available",
        desc="Provides the Best Buy store number if it is available from the official store-locator listing used.",
        parent=optional_node,
        critical=False,
    )
    store_num_claim = (
        f"The official store listing shows the store number as '{_safe_str(store_info.store_number)}'. "
        "If the page does not list any store number, this claim should be judged as not supported."
    )
    await evaluator.verify(
        claim=store_num_claim,
        node=store_num_leaf,
        sources=store_info.official_store_locator_url,
        additional_instruction="Confirm whether the page explicitly shows a store number. If not present, mark the claim incorrect.",
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate the answer for the Best Buy task:
    1) Identify the state with the highest number of Best Buy locations.
    2) Provide and verify one store in that state, excluding Los Angeles and San Francisco city limits.
    3) Include required details and official store locator URL; store number if available.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,  # First identify the state, then verify the store
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

    # Extract state identification information
    state_info: StateInfo = await evaluator.extract(
        prompt=prompt_extract_state(),
        template_class=StateInfo,
        extraction_name="state_identification",
    )

    # Extract store information
    store_info: StoreInfo = await evaluator.extract(
        prompt=prompt_extract_store(),
        template_class=StoreInfo,
        extraction_name="store_selection",
    )

    # Build and verify the tree according to the rubric (with minor structural adjustments to satisfy framework constraints)
    await verify_state_identification(evaluator, root, state_info)
    await verify_store_selection(evaluator, root, state_info.state_name, store_info)
    await verify_store_optional_info(evaluator, root, store_info)

    # Return summary with verification tree and overall score
    return evaluator.get_summary()