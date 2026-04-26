import asyncio
import logging
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "belmont_concert_capacity"
TASK_DESCRIPTION = "What is the concert seating capacity of the arena that opened at Belmont Park in Elmont, New York in November 2021?"


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class CapacityAnswerExtraction(BaseModel):
    """
    Structured extraction of key elements from the agent's answer.
    """
    arena_name: Optional[str] = None
    concert_capacity: Optional[str] = None
    capacity_context: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_capacity_info() -> str:
    return """
    You must extract the key information the answer provides to solve:
    "What is the concert seating capacity of the arena that opened at Belmont Park in Elmont, New York in November 2021?"

    Extract the following fields from the answer text:
    - arena_name: The name of the arena the answer identifies as the one fitting the description (located at Belmont Park in Elmont, NY; opened in November 2021). If multiple venues are mentioned, choose the one the answer uses for the capacity.
    - concert_capacity: The seating capacity value the answer claims specifically for concerts for that arena. Return exactly as written in the answer (allow commas, "approximately", "up to", or ranges).
    - capacity_context: Any short phrase around the capacity clarifying that it is specifically for concerts (e.g., "concert capacity", "for concerts", "maximum concert capacity"). If the answer does not specify that the capacity is for concerts, return a short note indicating ambiguity (e.g., "not explicitly labeled for concerts").
    - sources: An array of all URLs the answer cites as references or sources (include any URL that appears in the answer; include official sites, Wikipedia, ticketing pages, reputable news, etc.). Extract actual URLs even if embedded in markdown.

    Rules:
    - Do not invent information. If a field is missing, return null (for strings) or an empty array (for sources).
    - Extract only URLs actually present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def has_digits(s: Optional[str]) -> bool:
    if not s:
        return False
    return any(ch.isdigit() for ch in s)


def normalize_capacity_digits(capacity_str: Optional[str]) -> Optional[str]:
    """
    Return the digit-only form of the capacity, e.g., "19,500" -> "19500".
    If no digits are present, return None.
    """
    if not capacity_str:
        return None
    digits = re.findall(r"\d+", capacity_str)
    return "".join(digits) if digits else None


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extraction: CapacityAnswerExtraction) -> None:
    """
    Build and run verification according to the rubric tree.

    JSON rubric (interpreted into nodes):
    - arena_concert_capacity_answer (critical, sequential)
        1) arena_identification (critical, leaf)
        2) concert_capacity_provided (critical, expanded into parallel subchecks)
            - capacity_value_is_numeric (critical, custom)
            - capacity_is_specifically_for_concerts (critical, leaf)
        3) capacity_verifiable_sources (critical, expanded into parallel subchecks)
            - capacity_has_source_url (critical, custom)
            - capacity_supported_by_source (critical, leaf; verify by provided URLs)
    """
    # Create top-level critical sequential node mirroring the rubric's root
    arena_answer_node = evaluator.add_sequential(
        id="arena_concert_capacity_answer",
        desc="Answer correctly identifies the arena described in the question and provides its concert seating capacity with reliable verification.",
        parent=evaluator.root,
        critical=True
    )

    arena_name = extraction.arena_name or ""
    capacity_str = extraction.concert_capacity or ""
    capacity_digits = normalize_capacity_digits(capacity_str)
    sources_list = extraction.sources if extraction.sources else []

    # ------------------------------------------------------------------ #
    # 1) Arena identification                                            #
    # ------------------------------------------------------------------ #
    arena_identification_node = evaluator.add_leaf(
        id="arena_identification",
        desc="Identifies the arena that is located at Belmont Park in Elmont, New York and opened in November 2021 (i.e., the correct venue for the described constraints).",
        parent=arena_answer_node,
        critical=True
    )

    arena_identification_claim = (
        f"The answer identifies the arena as '{arena_name}', which is the arena located at Belmont Park in "
        f"Elmont, New York and opened in November 2021."
    )

    await evaluator.verify(
        claim=arena_identification_claim,
        node=arena_identification_node,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Treat this as correct only if the answer text or at least one provided URL explicitly supports that the named venue "
            "is at Belmont Park in Elmont, NY and opened in November 2021. If URLs are irrelevant or do not support the location "
            "and opening date, mark as incorrect. Do not rely on the judge's own knowledge."
        )
    )

    # ------------------------------------------------------------------ #
    # 2) Concert capacity provided                                       #
    #     (expanded into two critical checks)                            #
    # ------------------------------------------------------------------ #
    capacity_provided_parent = evaluator.add_parallel(
        id="concert_capacity_provided",
        desc="Provides a numeric seating capacity specifically for concerts (not a different event configuration) for the identified arena.",
        parent=arena_answer_node,
        critical=True
    )

    # 2.a) Capacity value is numeric-like (existence of digits)
    capacity_value_is_numeric_node = evaluator.add_custom_node(
        result=has_digits(capacity_str),
        id="capacity_value_is_numeric",
        desc="A numeric capacity value is provided (contains digits).",
        parent=capacity_provided_parent,
        critical=True
    )

    # 2.b) Capacity is specifically for concerts (based on answer text, not external inference)
    capacity_for_concerts_leaf = evaluator.add_leaf(
        id="capacity_is_specifically_for_concerts",
        desc="The provided capacity is explicitly labeled as the concert seating capacity (not for sports or generic capacity).",
        parent=capacity_provided_parent,
        critical=True
    )

    capacity_for_concerts_claim = (
        f"In the answer text, the capacity value '{capacity_str}' is explicitly described as being for concerts (e.g., "
        f"phrases like 'concert capacity', 'for concerts', 'maximum concert capacity'). It is not a sports configuration "
        f"like hockey or basketball."
    )

    await evaluator.verify(
        claim=capacity_for_concerts_claim,
        node=capacity_for_concerts_leaf,
        sources=None,
        additional_instruction=(
            "Use only the answer text for this check. If the answer does not explicitly specify that the capacity is for concerts, "
            "mark as incorrect even if a cited source would imply it."
        )
    )

    # ------------------------------------------------------------------ #
    # 3) Capacity verifiable by sources                                  #
    #     (expanded into two critical checks)                            #
    # ------------------------------------------------------------------ #
    verifiable_parent = evaluator.add_parallel(
        id="capacity_verifiable_sources",
        desc="Provides at least one reference URL from a reliable source that supports the stated concert seating capacity for the identified arena.",
        parent=arena_answer_node,
        critical=True
    )

    # 3.a) At least one URL is present
    has_source_url_node = evaluator.add_custom_node(
        result=(len(sources_list) > 0),
        id="capacity_has_source_url",
        desc="At least one reference URL is provided.",
        parent=verifiable_parent,
        critical=True
    )

    # 3.b) The sources support the stated concert capacity for the identified arena
    capacity_supported_leaf = evaluator.add_leaf(
        id="capacity_supported_by_sources",
        desc="At least one provided URL (reliable) explicitly supports the stated concert seating capacity for the identified arena.",
        parent=verifiable_parent,
        critical=True
    )

    # Build the verification claim against sources
    # Keep the raw value to allow 'approximately', 'up to', or formatted numbers
    supported_claim = (
        f"The concert seating capacity of {arena_name} is {capacity_str}."
    )

    # Additional instruction emphasizes: reliable sources and correct configuration
    additional_instruction = (
        "Verify this exact claim on the provided webpage: it must clearly state a concert seating capacity matching the answer. "
        "Accept minor formatting differences (commas/spaces) or equivalent phrasing like 'up to {answer_value}' or 'approximately {answer_value}'. "
        "If the page shows a range (e.g., 19,000–19,500) and the answer picks one bound or a value clearly within the stated range, accept. "
        "Reject capacities that are clearly for hockey/basketball or other configurations. "
        "Only accept as 'supported' if the page appears reliable (e.g., official venue site, major ticketing platform such as Ticketmaster, "
        "Wikipedia, or a reputable news/industry outlet). If none of the URLs support the claim, mark as not supported."
    )

    # Include a hint about numeric normalization if we have digits
    if capacity_digits:
        additional_instruction += (
            f" For numeric equivalence, treat '{capacity_digits}' (digit-only form) as equivalent to common formatted variants like "
            f"'{capacity_str}'."
        )

    await evaluator.verify(
        claim=supported_claim,
        node=capacity_supported_leaf,
        sources=sources_list if sources_list else None,
        additional_instruction=additional_instruction
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
    Entry point for evaluating an answer to the Belmont Park concert capacity question.
    """
    # Initialize evaluator with a sequential root (task has ordered critical checks)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extraction
    extraction = await evaluator.extract(
        prompt=prompt_extract_capacity_info(),
        template_class=CapacityAnswerExtraction,
        extraction_name="capacity_answer_extraction",
    )

    # Optional: record contextual info (non-scoring)
    evaluator.add_custom_info(
        info={
            "notes": "This evaluation checks (1) correct arena identification per question constraints, "
                     "(2) that a numeric concert capacity is provided in the answer, and "
                     "(3) that the capacity is supported by at least one reliable source URL.",
        },
        info_type="evaluation_notes",
    )

    # Build and execute verification tree
    await build_verification_tree(evaluator, extraction)

    # Produce summary
    return evaluator.get_summary()