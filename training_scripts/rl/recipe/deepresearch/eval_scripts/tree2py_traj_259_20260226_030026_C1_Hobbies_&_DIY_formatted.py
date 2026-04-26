import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "christmas_eve_2025_earliest_closing"
TASK_DESCRIPTION = "Among the major national home improvement and craft store chains (Home Depot, Lowe's, Hobby Lobby, and Michaels), which one closes earliest on Christmas Eve 2025, and what is its closing time?"
CHRISTMAS_EVE_DATE_TEXT = "December 24, 2025 (Christmas Eve)"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class HoursEvidenceExtraction(BaseModel):
    """
    Extracted structured information from the agent's answer:
    - predicted_store: which chain the answer claims closes earliest.
    - predicted_closing_time: the claimed closing time for that chain on Christmas Eve 2025.
    - Per-store URLs: all URLs cited in the answer that relate to Christmas Eve 2025 hours for each chain.
    - comparative_urls: URLs that compare multiple chains or list holiday hours across many stores.
    """
    predicted_store: Optional[str] = None
    predicted_closing_time: Optional[str] = None

    home_depot_urls: List[str] = Field(default_factory=list)
    lowes_urls: List[str] = Field(default_factory=list)
    hobby_lobby_urls: List[str] = Field(default_factory=list)
    michaels_urls: List[str] = Field(default_factory=list)

    comparative_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hours_evidence() -> str:
    return """
    From the provided answer, extract the following fields exactly as stated:

    1) predicted_store: The store chain the answer claims closes earliest on Christmas Eve 2025. It should be one of
       "Home Depot", "Lowe's", "Hobby Lobby", or "Michaels" (allow minor variants like "The Home Depot", "Lowes").
    2) predicted_closing_time: The specific closing time the answer claims for that store on Christmas Eve 2025
       (e.g., "5 PM", "6 p.m.", "5:30 pm").
    3) home_depot_urls: All URLs explicitly provided in the answer that relate to Home Depot's Christmas Eve 2025 hours.
    4) lowes_urls: All URLs explicitly provided in the answer that relate to Lowe's Christmas Eve 2025 hours.
    5) hobby_lobby_urls: All URLs explicitly provided in the answer that relate to Hobby Lobby's Christmas Eve 2025 hours.
    6) michaels_urls: All URLs explicitly provided in the answer that relate to Michaels' Christmas Eve 2025 hours.
    7) comparative_urls: All URLs explicitly provided in the answer that discuss or compare multiple stores' holiday/Christmas Eve hours
       (e.g., a news/guide page listing hours for several chains; include any page that references more than one of the four chains).

    IMPORTANT URL RULES:
    - Only include URLs that are explicitly present in the answer text (plain URLs, markdown links, etc.).
    - Return full URLs; if a URL is missing a protocol, prepend "http://".
    - If a field is not mentioned, set it to null (for strings) or an empty list (for arrays).

    Return a single JSON object with these exact fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_store_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    if "home depot" in s:
        return "Home Depot"
    if "lowe" in s:  # matches "lowe's" or "lowes"
        return "Lowe's"
    if "hobby lobby" in s:
        return "Hobby Lobby"
    if "michael" in s:
        return "Michaels"
    return name.strip()


def get_store_urls(extracted: HoursEvidenceExtraction, store: Optional[str]) -> List[str]:
    if not store:
        return []
    if store == "Home Depot":
        return extracted.home_depot_urls or []
    if store == "Lowe's":
        return extracted.lowes_urls or []
    if store == "Hobby Lobby":
        return extracted.hobby_lobby_urls or []
    if store == "Michaels":
        return extracted.michaels_urls or []
    return []


def build_all_chain_urls(extracted: HoursEvidenceExtraction) -> List[str]:
    urls = []
    urls.extend(extracted.home_depot_urls or [])
    urls.extend(extracted.lowes_urls or [])
    urls.extend(extracted.hobby_lobby_urls or [])
    urls.extend(extracted.michaels_urls or [])
    urls.extend(extracted.comparative_urls or [])
    # Deduplicate while preserving order
    seen = set()
    unique_urls: List[str] = []
    for u in urls:
        if u and u not in seen:
            unique_urls.append(u)
            seen.add(u)
    return unique_urls


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def verify_answer_correctness(
    evaluator: Evaluator,
    parent_node,
    extracted: HoursEvidenceExtraction,
) -> None:
    """
    Build the verification sub-tree for 'Answer_Correctness' and run the checks.
    """
    # Create the critical parent node for this rubric section
    correctness_node = evaluator.add_parallel(
        id="Answer_Correctness",
        desc="The answer correctly identifies which major national home improvement/craft store chain closes earliest on Christmas Eve 2025 and provides the accurate closing time",
        parent=parent_node,
        critical=True,
    )

    # Prepare normalized store and sources
    norm_store = normalize_store_name(extracted.predicted_store)
    store_time = extracted.predicted_closing_time or ""
    store_specific_urls = get_store_urls(extracted, norm_store)
    all_urls = build_all_chain_urls(extracted)

    # Leaf 1: Earliest_Closing_Store (critical)
    earliest_leaf = evaluator.add_leaf(
        id="Earliest_Closing_Store",
        desc="The identified store has the earliest closing time on Christmas Eve 2025 among Home Depot, Lowe's, Hobby Lobby, and Michaels",
        parent=correctness_node,
        critical=True,
    )
    earliest_claim = (
        f"Among Home Depot, Lowe's, Hobby Lobby, and Michaels, {norm_store} closes earliest on Christmas Eve 2025."
        if norm_store else
        "Among Home Depot, Lowe's, Hobby Lobby, and Michaels, [unspecified store] closes earliest on Christmas Eve 2025."
    )

    # Additional instruction for earliest-closing verification
    earliest_add_ins = (
        "To support this comparative claim, the single webpage you are checking must itself explicitly present "
        "Christmas Eve 2025 closing hours for multiple chains or clearly state which chain closes earliest. "
        "If the page only shows hours for a single store without comparing to the others, that page alone does NOT support the comparative claim. "
        "Allow minor time-format variations (e.g., '5 PM' vs '5:00 p.m.'). Focus on December 24, 2025. "
    )
    if not all_urls:
        earliest_add_ins += "No source URLs were provided in the answer; therefore, you should deem the claim unsupported and mark it as incorrect."

    await evaluator.verify(
        claim=earliest_claim,
        node=earliest_leaf,
        sources=all_urls if all_urls else None,
        additional_instruction=earliest_add_ins,
    )

    # Leaf 2: Correct_Closing_Time (critical)
    closing_time_leaf = evaluator.add_leaf(
        id="Correct_Closing_Time",
        desc="The stated closing time matches the verified Christmas Eve 2025 hours for the identified store",
        parent=correctness_node,
        critical=True,
    )
    closing_time_claim = (
        f"On {CHRISTMAS_EVE_DATE_TEXT}, {norm_store} closes at {store_time}."
        if norm_store else
        f"On {CHRISTMAS_EVE_DATE_TEXT}, [unspecified store] closes at {store_time}."
    )

    closing_time_add_ins = (
        "Verify that the page states the store's closing time specifically for Christmas Eve 2025. "
        "If the page mentions 'local time' or 'hours may vary by location' but also lists a typical or stated closing time for Christmas Eve 2025, that can count as support. "
        "Allow minor formatting variations (e.g., '5 PM' vs '5:00 p.m.'), but ensure the time itself matches. "
        "Generic holiday pages without Christmas Eve specifics or pages for other years do not suffice."
    )
    if not store_specific_urls:
        closing_time_add_ins += " No source URLs for the identified store were provided; treat the claim as unsupported and incorrect."

    await evaluator.verify(
        claim=closing_time_claim,
        node=closing_time_leaf,
        sources=store_specific_urls if store_specific_urls else None,
        additional_instruction=closing_time_add_ins,
    )

    # Record additional custom info to aid debugging
    evaluator.add_custom_info(
        info={
            "predicted_store_raw": extracted.predicted_store,
            "predicted_store_normalized": norm_store,
            "predicted_closing_time": store_time,
            "home_depot_urls_count": len(extracted.home_depot_urls),
            "lowes_urls_count": len(extracted.lowes_urls),
            "hobby_lobby_urls_count": len(extracted.hobby_lobby_urls),
            "michaels_urls_count": len(extracted.michaels_urls),
            "comparative_urls_count": len(extracted.comparative_urls),
            "all_urls_count": len(all_urls),
        },
        info_type="extraction_summary",
        info_name="extraction_summary",
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
    Evaluate the agent's answer for: which chain closes earliest on Christmas Eve 2025 and what is its closing time.
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
    extracted = await evaluator.extract(
        prompt=prompt_extract_hours_evidence(),
        template_class=HoursEvidenceExtraction,
        extraction_name="hours_evidence",
    )

    # Build the verification tree for the rubric and run verifications
    await verify_answer_correctness(evaluator, root, extracted)

    # Return the structured evaluation summary
    return evaluator.get_summary()