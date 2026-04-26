import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "us_venue_capacities"
TASK_DESCRIPTION = """Based on venue capacity standards and specifications for major performing arts venues in the United States, provide the following information:

1. What is the minimum seating capacity required for a theater to be classified as a Broadway theater?
2. What is the seating capacity of the Gershwin Theatre, which is the largest Broadway theater?
3. What is the seating capacity of the Hayes Theater, which is the smallest Broadway theater?
4. What is the seating capacity of the Dorothy Chandler Pavilion in Los Angeles?
5. What is the capacity of the Greek Theatre in Los Angeles?
"""

EXPECTED_VALUES = {
    "broadway_minimum": "500",
    "gershwin": "1,933",
    "hayes": "597",
    "dorothy_chandler": "3,197",
    "greek_theatre": "5,900",
}

# Node descriptions (from rubric JSON)
NODE_DESCRIPTIONS = {
    "broadway_minimum_capacity": "Correctly state the minimum seating capacity required for a theater to be classified as a Broadway theater (500 seats)",
    "gershwin_theatre_capacity": "Correctly state the seating capacity of the Gershwin Theatre as 1,933 seats",
    "hayes_theater_capacity": "Correctly state the seating capacity of the Hayes Theater as 597 seats",
    "dorothy_chandler_capacity": "Correctly state the seating capacity of the Dorothy Chandler Pavilion in Los Angeles as 3,197 seats",
    "greek_theatre_capacity": "Correctly state the capacity of the Greek Theatre in Los Angeles as 5,900",
}

TOPIC_TEXT = {
    "broadway_minimum": "the minimum seating capacity required for a theater to be classified as a Broadway theater",
    "gershwin": "the seating capacity of the Gershwin Theatre",
    "hayes": "the seating capacity of the Hayes Theater",
    "dorothy_chandler": "the seating capacity of the Dorothy Chandler Pavilion in Los Angeles",
    "greek_theatre": "the capacity of the Greek Theatre in Los Angeles",
}


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class VenueItem(BaseModel):
    """A single item: the value stated in the answer and any URLs cited for it."""
    value: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


class VenueCapacitiesExtraction(BaseModel):
    """Extracted capacities and thresholds for the requested five items."""
    broadway_minimum: Optional[VenueItem] = None
    gershwin: Optional[VenueItem] = None
    hayes: Optional[VenueItem] = None
    dorothy_chandler: Optional[VenueItem] = None
    greek_theatre: Optional[VenueItem] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_venue_capacities() -> str:
    return """
    Extract from the provided answer the five requested items. For each item, return:
    - value: the capacity (or required capacity threshold) exactly as stated in the answer (keep the original formatting in the answer, e.g., include commas, words like "seats", "at least", "or more", "about", etc.).
    - source_urls: a list of all URLs explicitly cited in the answer that directly support that item's value. Only include URLs that are actually present in the answer text (plain URLs or Markdown links). If no URLs are cited, return an empty list.

    The five items to extract (use these exact keys in your JSON):
    - broadway_minimum: the minimum seating capacity for a theater to be classified as a Broadway theater.
    - gershwin: the seating capacity of the Gershwin Theatre.
    - hayes: the seating capacity of the Hayes Theater.
    - dorothy_chandler: the seating capacity of the Dorothy Chandler Pavilion in Los Angeles.
    - greek_theatre: the capacity of the Greek Theatre in Los Angeles.

    Notes:
    - Do not invent any URLs. Only extract URLs explicitly included in the answer.
    - If an item is not provided in the answer, set its 'value' to null and 'source_urls' to an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def canonicalize_number_str(num_str: Optional[str]) -> Optional[str]:
    """
    Take a possibly formatted capacity string from the answer and produce a simple, comma-grouped number string.
    - Keep only digits from the first numeric token found.
    - Return a string with thousand separators (commas), e.g., "3197" -> "3,197".
    - If no digits found, return None.
    """
    if not num_str:
        return None
    # Find the first numeric token (allow comma-formatted or plain digits)
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+)", num_str)
    if not m:
        return None
    digits_only = re.sub(r"[^\d]", "", m.group(1))
    if not digits_only:
        return None
    # Add commas for thousands grouping
    rev = digits_only[::-1]
    grouped = ",".join(rev[i:i+3] for i in range(0, len(rev), 3))[::-1]
    return grouped


def _value_present(item: Optional[VenueItem]) -> bool:
    return bool(item and item.value and item.value.strip())


def _sources_present(item: Optional[VenueItem]) -> bool:
    return bool(item and item.source_urls and len(item.source_urls) > 0)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def verify_capacity_item(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    topic_key: str,
    expected_value: str,
    item: Optional[VenueItem],
) -> None:
    """
    Build verification subtree for a single capacity item.
    Structure (parallel):
      - value_provided (custom, critical)
      - value_correct (leaf, critical)    --> checks the answer states the expected value
      - source_supported (leaf, non-critical)  --> checks cited URLs (if any) support the stated value from the answer
    """
    topic_text = TOPIC_TEXT[topic_key]

    item_node = evaluator.add_parallel(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=False
    )

    # 1) Existence check: did the answer provide a value?
    value_exists = _value_present(item)
    evaluator.add_custom_node(
        result=value_exists,
        id=f"{node_id}_value_provided",
        desc=f"The answer provides a value for {topic_text}",
        parent=item_node,
        critical=True
    )

    # 2) Correctness against expected value (based solely on the answer text)
    correct_leaf = evaluator.add_leaf(
        id=f"{node_id}_value_correct",
        desc=f"The answer states the correct value for {topic_text}: {expected_value}",
        parent=item_node,
        critical=True
    )
    # We assert the answer states the expected value. Allow minor formatting variants.
    correct_claim = (
        f"In the provided answer, {topic_text} is stated as {expected_value}. "
        f"Treat minor formatting variants as equivalent (e.g., '1933' vs '1,933', "
        f"including or omitting the word 'seats', or phrasing like 'at least 500', '500+', or '500 or more' for thresholds). "
        f"If the answer gives a different number, judge this claim as incorrect."
    )
    await evaluator.verify(
        claim=correct_claim,
        node=correct_leaf,
        additional_instruction="Judge correctness strictly against the answer text; do not use your own knowledge."
    )

    # 3) Source support (non-critical). If URLs are present, verify the answer's stated value is supported by the cited sources.
    # If no URLs, add a skipped leaf to make the tree explicit.
    source_leaf = None
    if _sources_present(item):
        source_leaf = evaluator.add_leaf(
            id=f"{node_id}_source_supported",
            desc=f"The cited source(s) support the stated value for {topic_text}",
            parent=item_node,
            critical=False
        )
        # Use the value stated in the answer for source verification (check internal consistency of the answer with its sources)
        stated_value = item.value or expected_value
        canonical_value = canonicalize_number_str(stated_value) or stated_value

        source_claim = (
            f"The cited page(s) explicitly support that {topic_text} is {canonical_value}."
        )

        add_ins = (
            "Verify directly from the page text and/or screenshot that the numeric capacity is stated. "
            "Allow minor formatting variants (e.g., 1933 vs 1,933; inclusion of 'seats'). "
            "If multiple numbers appear, ensure the one corresponding to capacity matches the answer's stated number."
        )
        await evaluator.verify(
            claim=source_claim,
            node=source_leaf,
            sources=item.source_urls,  # list of URLs
            additional_instruction=add_ins
        )
    else:
        evaluator.add_leaf(
            id=f"{node_id}_source_supported",
            desc=f"The cited source(s) support the stated value for {topic_text} (no sources provided → skipped)",
            parent=item_node,
            critical=False,
            score=0.0,
            status="skipped"
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the U.S. venue capacities task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks for each item
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description="Correctly answer all questions about U.S. performing arts venue capacities and classification standards",
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model,
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_venue_capacities(),
        template_class=VenueCapacitiesExtraction,
        extraction_name="venue_capacities_extraction",
    )

    # Record ground truth values for transparency
    evaluator.add_ground_truth({
        "expected_values": {
            "broadway_minimum": "500",
            "gershwin": "1,933",
            "hayes": "597",
            "dorothy_chandler": "3,197",
            "greek_theatre": "5,900",
        }
    })

    # Build verification subtrees for each requested item
    await verify_capacity_item(
        evaluator=evaluator,
        parent_node=root,
        node_id="broadway_minimum_capacity",
        node_desc=NODE_DESCRIPTIONS["broadway_minimum_capacity"],
        topic_key="broadway_minimum",
        expected_value=EXPECTED_VALUES["broadway_minimum"],
        item=extracted.broadway_minimum,
    )

    await verify_capacity_item(
        evaluator=evaluator,
        parent_node=root,
        node_id="gershwin_theatre_capacity",
        node_desc=NODE_DESCRIPTIONS["gershwin_theatre_capacity"],
        topic_key="gershwin",
        expected_value=EXPECTED_VALUES["gershwin"],
        item=extracted.gershwin,
    )

    await verify_capacity_item(
        evaluator=evaluator,
        parent_node=root,
        node_id="hayes_theater_capacity",
        node_desc=NODE_DESCRIPTIONS["hayes_theater_capacity"],
        topic_key="hayes",
        expected_value=EXPECTED_VALUES["hayes"],
        item=extracted.hayes,
    )

    await verify_capacity_item(
        evaluator=evaluator,
        parent_node=root,
        node_id="dorothy_chandler_capacity",
        node_desc=NODE_DESCRIPTIONS["dorothy_chandler_capacity"],
        topic_key="dorothy_chandler",
        expected_value=EXPECTED_VALUES["dorothy_chandler"],
        item=extracted.dorothy_chandler,
    )

    await verify_capacity_item(
        evaluator=evaluator,
        parent_node=root,
        node_id="greek_theatre_capacity",
        node_desc=NODE_DESCRIPTIONS["greek_theatre_capacity"],
        topic_key="greek_theatre",
        expected_value=EXPECTED_VALUES["greek_theatre"],
        item=extracted.greek_theatre,
    )

    # Return evaluation summary
    return evaluator.get_summary()