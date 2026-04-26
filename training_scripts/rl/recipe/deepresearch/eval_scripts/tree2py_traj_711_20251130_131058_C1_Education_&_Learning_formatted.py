import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "jennifer_garner_university_location"
TASK_DESCRIPTION = """
Jennifer Garner graduated in 1994 with a Bachelor of Fine Arts degree in theater performance. Identify the name of the university she attended and provide the city and state where this university is located.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class UniversityIdentification(BaseModel):
    """Extraction model for university name and its location, plus cited sources."""
    university_name: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_university_info() -> str:
    return """
    From the provided answer, extract the following information (as explicitly stated in the answer):
    1. university_name: The name of the university where Jennifer Garner earned her Bachelor of Fine Arts (BFA) degree in theater performance in 1994.
    2. city: The city where this university is located.
    3. state: The state (or equivalent region) where this university is located.
    4. sources: All URLs explicitly mentioned in the answer that support either the degree-at-university fact or the university’s location (include both general references like the university homepage or Wikipedia, and any specific pages used as evidence).

    Rules for sources:
    - Extract only URLs that appear in the answer (plain URLs or markdown links).
    - Do not fabricate URLs. If none are present, return an empty list.

    If any field is missing in the answer, set it to null (for strings) or an empty list (for sources).
    """


# --------------------------------------------------------------------------- #
# Helper for additional instructions                                          #
# --------------------------------------------------------------------------- #
def build_additional_instruction_for_university(has_sources: bool) -> str:
    if has_sources:
        return (
            "Use the provided URLs to verify that Jennifer Garner earned a Bachelor of Fine Arts (BFA) degree in "
            "theater performance in 1994 at the specified university. Allow minor wording variations (e.g., 'theatre' "
            "vs 'theater'). If the URLs are irrelevant, invalid, or do not explicitly support the statement, judge it as not supported."
        )
    else:
        return (
            "No URLs were extracted from the answer. Focus only on the claim and the answer text provided. "
            "Judge 'Correct' only if the answer itself clearly and explicitly states the fact as claimed; otherwise judge 'Incorrect'."
        )


def build_additional_instruction_for_location(has_sources: bool) -> str:
    if has_sources:
        return (
            "Use the provided URLs to verify the university’s location (city and state). Allow minor or reasonable naming "
            "variants (e.g., city suffixes or alternate spellings). If the URLs are irrelevant, invalid, or do not clearly "
            "support the location, judge it as not supported."
        )
    else:
        return (
            "No URLs were extracted from the answer. Focus only on the claim and the answer text provided. "
            "Judge 'Correct' only if the answer itself clearly and explicitly states the location as claimed; otherwise judge 'Incorrect'."
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
) -> Dict[str, Any]:
    """
    Evaluate the agent's answer for identifying Jennifer Garner's university and its location.
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

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_university_info(),
        template_class=UniversityIdentification,
        extraction_name="university_identification",
    )

    # Build the verification subtree corresponding to the rubric root
    rubric_root = evaluator.add_parallel(
        id="University_and_Location_Identification",
        desc="Correctly identify the university where Jennifer Garner earned her Bachelor of Fine Arts degree in theater in 1994, and provide its location",
        parent=root,
        critical=True
    )

    # Leaf 1: University Name verification
    uni_name_leaf = evaluator.add_leaf(
        id="University_Name",
        desc="The correct university name is provided",
        parent=rubric_root,
        critical=True
    )
    uni_name = extracted.university_name or ""
    has_sources = bool(extracted.sources)
    uni_claim = (
        f"Jennifer Garner earned a Bachelor of Fine Arts (BFA) degree in theater performance in 1994 at {uni_name}."
    )
    await evaluator.verify(
        claim=uni_claim,
        node=uni_name_leaf,
        sources=extracted.sources if has_sources else None,
        additional_instruction=build_additional_instruction_for_university(has_sources)
    )

    # Leaf 2: Location verification
    location_leaf = evaluator.add_leaf(
        id="Location",
        desc="The correct city and state where the university is located are provided",
        parent=rubric_root,
        critical=True
    )
    city = extracted.city or ""
    state = extracted.state or ""
    loc_claim = f"The university {uni_name} is located in {city}, {state}."
    await evaluator.verify(
        claim=loc_claim,
        node=location_leaf,
        sources=extracted.sources if has_sources else None,
        additional_instruction=build_additional_instruction_for_location(has_sources)
    )

    # Return evaluation summary
    return evaluator.get_summary()