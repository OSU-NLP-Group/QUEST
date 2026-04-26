import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "illinois_indoor_venue"
TASK_DESCRIPTION = (
    "Identify an indoor concert venue in Illinois with a seating capacity between 1,000 and 2,500 people that provides "
    "wheelchair-accessible seating and has a bag size restriction policy that permits bags up to 12 inches by 6 inches "
    "by 12 inches. Provide the venue name, location, seating capacity (with reference URL), wheelchair accessibility "
    "information (with reference URL), and bag policy details (with reference URL)."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueExtraction(BaseModel):
    """Structured data for a single venue as presented in the answer."""
    name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None  # Accept full state name (Illinois) or abbreviation (IL)
    is_indoor: Optional[bool] = None  # True if answer explicitly states indoor, else null/False
    indoor_support_text: Optional[str] = None  # optional text snippet related to "indoor"

    capacity: Optional[str] = None  # Extract verbatim (e.g., "2,100", "about 2,000", "1,800–2,000")
    capacity_source_url: Optional[str] = None

    accessibility_info: Optional[str] = None  # e.g., "ADA seating available", "wheelchair-accessible seating"
    accessibility_source_url: Optional[str] = None

    bag_policy_details: Optional[str] = None  # verbatim description containing allowed sizes
    bag_policy_source_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_venue() -> str:
    return """
    Extract the details for the single venue that the answer proposes (choose the first venue if multiple are listed).
    You must only extract information explicitly present in the answer text. Do not infer or invent details.

    Return a JSON object with the following fields:
    - name: The venue name (string).
    - city: The city for the venue location (string), if provided; else null.
    - state: The state for the venue location (string). Prefer "Illinois" or "IL" if explicitly provided; else extract what is given; if absent, null.
    - is_indoor: A boolean indicating whether the answer explicitly states the venue is an indoor concert venue. True if explicitly stated, False if explicitly stated otherwise, null if not stated.
    - indoor_support_text: A short text snippet or phrase from the answer that mentions the venue being indoor, if available; else null.

    Seating capacity (with source):
    - capacity: The seating capacity as written in the answer (string). If a range or approximate value is given, extract it verbatim. If absent, null.
    - capacity_source_url: The single URL explicitly provided in the answer that verifies the seating capacity. If multiple URLs are provided, pick the one most directly tied to capacity. If none are provided, null.

    Wheelchair-accessible seating (with source):
    - accessibility_info: The answer's statement about wheelchair-accessible seating (string), e.g., "wheelchair-accessible seating available", "ADA seating". If absent, null.
    - accessibility_source_url: The single URL explicitly provided that verifies wheelchair-accessible seating. If none are provided, null.

    Bag policy (with source):
    - bag_policy_details: The answer's statement about bag policy (string). This should include any mentioned size limit such as "12 x 6 x 12 inches" if present in the answer. If absent, null.
    - bag_policy_source_url: The single URL explicitly provided that verifies the bag policy. If none are provided, null.

    Strict rules for URLs:
    - Only include valid URLs explicitly mentioned in the answer. If a source is referenced without an actual URL, return null for that source field.
    - If a URL is missing protocol, prepend http://.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_nonempty(value: Optional[str]) -> bool:
    return bool(value and isinstance(value, str) and value.strip())


def _parse_ints_from_text(text: str) -> List[int]:
    """
    Extract integers from text, preserving large numbers like "2,100".
    Returns list of ints. Handles commas and mixed formats like "1,800–2,000".
    """
    if not _is_nonempty(text):
        return []
    nums = re.findall(r"\d{1,3}(?:,\d{3})+|\d+", text)
    cleaned = []
    for n in nums:
        try:
            cleaned.append(int(n.replace(",", "")))
        except Exception:
            continue
    return cleaned


def _capacity_in_range(text: Optional[str], min_val: int = 1000, max_val: int = 2500) -> bool:
    """
    Check if any number found in the capacity text lies within [min_val, max_val].
    If no numbers found, return False.
    """
    if not _is_nonempty(text):
        return False
    nums = _parse_ints_from_text(text)
    if not nums:
        return False
    return any(min_val <= n <= max_val for n in nums)


def _is_illinois(state: Optional[str]) -> bool:
    """
    Check if the provided state string indicates Illinois (case-insensitive).
    Accept "Illinois", "IL", or variants like "Ill.".
    """
    if not _is_nonempty(state):
        return False
    s = state.strip().lower()
    return s in {"illinois", "il", "ill.", "il."} or "illinois" in s


def _bag_policy_allows_12x6x12(text: Optional[str]) -> bool:
    """
    Check whether the bag policy text mentions the dimension '12 x 6 x 12' (with variations).
    This is a heuristic: we look for the dimensions and permissive language.
    """
    if not _is_nonempty(text):
        return False

    t = text.lower()
    # Dimension pattern: 12 x 6 x 12, allowing optional quotes or "inches" and "by" variants
    dim_pattern = re.compile(
        r"(?:12\s*(?:inches|\"|”|’)?\s*(?:x|by)\s*6\s*(?:inches|\"|”|’)?\s*(?:x|by)\s*12\s*(?:inches|\"|”|’)?)",
        re.IGNORECASE,
    )
    has_dims = bool(dim_pattern.search(t))

    # Permissive wording near the text
    # We just check global text for any permissive words since answer is short
    permissive_words = ["allow", "allowed", "permit", "permitted", "up to", "not exceeding", "no larger than"]
    has_perm = any(word in t for word in permissive_words)

    return has_dims and has_perm


# --------------------------------------------------------------------------- #
# Verification building                                                       #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, venue: VenueExtraction) -> None:
    """
    Build the verification tree per rubric and run the corresponding checks.
    Root is critical, and all its children must be critical in Mind2Web2.
    """
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,
        agent_name=evaluator._agent_name or "unknown_agent",
        answer_name=evaluator._answer_name or "unknown_answer",
        client=evaluator.verifier.client,  # evaluator already created via initialize; reuse clients/models
        task_description=TASK_DESCRIPTION,
        answer=evaluator.verifier.task_description,  # This is not correct: we need the actual answer
        global_cache=evaluator.verifier.cache,
        global_semaphore=evaluator.verifier.semaphore,
        logger=evaluator.verifier.logger,
        default_model=evaluator.verifier.MODEL_NAME,
    )
    # The above initialize would reinitialize evaluator incorrectly; we should not reinitialize inside builder.
    # Instead, use the already-initialized root.
    # To avoid confusion, we will find current root and use it.
    # But evaluator.initialize() must be called only once outside; So we do nothing here.
    pass  # Placeholder to avoid redefining; We'll construct below in evaluate_answer()


# --------------------------------------------------------------------------- #
# Main verification logic                                                     #
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
    Evaluate the answer for the Illinois indoor concert venue task.
    """
    # Initialize evaluator once
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

    # Extract venue info
    venue_info = await evaluator.extract(
        prompt=prompt_extract_venue(),
        template_class=VenueExtraction,
        extraction_name="venue_extraction",
    )

    # ---------------- Root children (all critical) ---------------- #

    # 1) Venue Name Provided (Critical)
    evaluator.add_custom_node(
        result=_is_nonempty(venue_info.name),
        id="Venue_Name_Provided",
        desc="Venue name is provided.",
        parent=root,
        critical=True,
    )

    # 2) Venue Location (Critical Sequential: city/state provided -> located in Illinois)
    loc_node = evaluator.add_sequential(
        id="Venue_Location",
        desc="Venue location is provided and the venue is in Illinois.",
        parent=root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty(venue_info.city) and _is_nonempty(venue_info.state),
        id="Location_Provided",
        desc="Venue location is provided (at least city and state).",
        parent=loc_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_illinois(venue_info.state),
        id="Located_In_Illinois",
        desc="The provided location confirms the venue is in Illinois.",
        parent=loc_node,
        critical=True,
    )

    # 3) Indoor Concert Venue (Critical)
    # Use a simple verification based on the answer context.
    indoor_leaf = evaluator.add_leaf(
        id="Indoor_Concert_Venue",
        desc="Venue is confirmed to be an indoor concert venue.",
        parent=root,
        critical=True,
    )
    indoor_claim = "The venue is an indoor concert venue."
    await evaluator.verify(
        claim=indoor_claim,
        node=indoor_leaf,
        additional_instruction=(
            "Judge based on the answer content only. Do not use external knowledge. "
            "Consider the claim true only if the answer explicitly indicates that the venue is indoors."
        ),
    )

    # 4) Seating Capacity With Source (Critical Parallel)
    cap_node = evaluator.add_parallel(
        id="Seating_Capacity_With_Source",
        desc="Seating capacity is stated, is between 1,000 and 2,500 (inclusive), and includes a reference URL verifying the capacity.",
        parent=root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty(venue_info.capacity),
        id="Capacity_Provided",
        desc="Seating capacity is provided.",
        parent=cap_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_capacity_in_range(venue_info.capacity, 1000, 2500),
        id="Capacity_In_Range_1000_2500",
        desc="Seating capacity is between 1,000 and 2,500 (inclusive).",
        parent=cap_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty(venue_info.capacity_source_url),
        id="Capacity_Source_Provided",
        desc="A reference URL for seating capacity is provided.",
        parent=cap_node,
        critical=True,
    )

    capacity_source_support_leaf = evaluator.add_leaf(
        id="Capacity_Source_Supports",
        desc="The provided capacity source supports the stated seating capacity.",
        parent=cap_node,
        critical=True,
    )
    capacity_claim = f"The venue's seating capacity is {venue_info.capacity or ''}."
    await evaluator.verify(
        claim=capacity_claim,
        node=capacity_source_support_leaf,
        sources=venue_info.capacity_source_url,
        additional_instruction=(
            "Verify that the cited webpage explicitly states the seating capacity matching the answer. "
            "Allow minor variations (e.g., punctuation or formatting). If the page shows a range and "
            "the answer mentions a single value within that range, consider it acceptable."
        ),
    )

    # 5) Wheelchair-Accessible Seating With Source (Critical Parallel)
    ada_node = evaluator.add_parallel(
        id="Wheelchair_Accessible_Seating_With_Source",
        desc="Wheelchair-accessible seating availability is stated and includes a reference URL verifying it.",
        parent=root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty(venue_info.accessibility_info),
        id="Wheelchair_Info_Provided",
        desc="Wheelchair-accessible seating information is provided.",
        parent=ada_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty(venue_info.accessibility_source_url),
        id="Wheelchair_Source_Provided",
        desc="A reference URL for wheelchair-accessible seating is provided.",
        parent=ada_node,
        critical=True,
    )

    ada_support_leaf = evaluator.add_leaf(
        id="Wheelchair_Source_Supports",
        desc="The provided accessibility source supports wheelchair-accessible seating availability.",
        parent=ada_node,
        critical=True,
    )
    ada_claim = "Wheelchair-accessible seating is available at this venue."
    await evaluator.verify(
        claim=ada_claim,
        node=ada_support_leaf,
        sources=venue_info.accessibility_source_url,
        additional_instruction=(
            "Confirm the page mentions wheelchair-accessible seating or ADA-accessible seating/sections. "
            "Synonyms such as 'ADA seating', 'accessible seating', or 'wheelchair seating' are acceptable."
        ),
    )

    # 6) Bag Policy 12x6x12 With Source (Critical Parallel)
    bag_node = evaluator.add_parallel(
        id="Bag_Policy_12x6x12_With_Source",
        desc="Bag policy details state that bags up to 12 inches by 6 inches by 12 inches are permitted and include a reference URL verifying the policy.",
        parent=root,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty(venue_info.bag_policy_details),
        id="Bag_Policy_Provided",
        desc="Bag policy details are provided.",
        parent=bag_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_is_nonempty(venue_info.bag_policy_source_url),
        id="Bag_Policy_Source_Provided",
        desc="A reference URL for bag policy is provided.",
        parent=bag_node,
        critical=True,
    )

    evaluator.add_custom_node(
        result=_bag_policy_allows_12x6x12(venue_info.bag_policy_details),
        id="Bag_Size_Allows_12x6x12",
        desc="Bag policy explicitly permits up to 12 x 6 x 12 inches.",
        parent=bag_node,
        critical=True,
    )

    bag_support_leaf = evaluator.add_leaf(
        id="Bag_Policy_Source_Supports_12x6x12",
        desc="The provided bag policy source explicitly supports the 12 x 6 x 12 inch allowance.",
        parent=bag_node,
        critical=True,
    )
    bag_claim = "The venue's bag policy permits bags up to 12 inches by 6 inches by 12 inches."
    await evaluator.verify(
        claim=bag_claim,
        node=bag_support_leaf,
        sources=venue_info.bag_policy_source_url,
        additional_instruction=(
            "Check if the page states clear/standard bags up to 12 x 6 x 12 inches are allowed, or equivalent phrasing. "
            "Accept phrasing like 'up to', 'not exceeding', or 'no larger than' 12 x 6 x 12 inches."
        ),
    )

    # Return structured evaluation summary
    return evaluator.get_summary()