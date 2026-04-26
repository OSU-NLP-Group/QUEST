import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "jfk_t5_rooftop"
TASK_DESCRIPTION = (
    "What is the name of the free outdoor rooftop recreational space at JFK Airport Terminal 5 that passengers can access "
    "after going through security, what are its daily operating hours, and approximately how large is it in square feet?"
)

# Ground-truth references for informational purposes in summary
GROUND_TRUTH = {
    "accepted_names": [
        "T5 Rooftop",
        "T5 Rooftop & Wooftop Lounge",
        "T5 Rooftop and Wooftop Lounge",
    ],
    "operating_hours": "6:00 AM to 10:00 PM",
    "approx_size_sqft": "4,046",
}

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class FacilityExtraction(BaseModel):
    facility_name: Optional[str] = None
    operating_hours: Optional[str] = None
    size_sqft: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_info() -> str:
    return """
    From the answer, extract the following fields about the free outdoor rooftop recreational space at JFK Airport Terminal 5:

    - facility_name: The name the answer uses for the space (e.g., "T5 Rooftop", "T5 Rooftop & Wooftop Lounge").
    - operating_hours: The daily operating hours mentioned in the answer (e.g., "6:00 AM to 10:00 PM"). Keep the formatting as in the answer.
    - size_sqft: The approximate size in square feet as mentioned in the answer (e.g., "4,046 square feet", "~4,000 sq ft"). Preserve the text as written.
    - sources: All URLs cited in the answer that support any of the above information. Include complete URLs. If none are provided, return an empty list.

    Rules:
    - Do not invent information. If any field is not present in the answer, set it to null (or [] for sources).
    - Extract URLs only if explicitly present (plain URL or within markdown).
    """


# --------------------------------------------------------------------------- #
# Verification helper functions                                               #
# --------------------------------------------------------------------------- #
async def verify_facility_identification(
    evaluator: Evaluator,
    parent_node,
    extracted: FacilityExtraction,
) -> None:
    """
    Build and verify the Facility_Identification subtree:
      - Confirms the name as presented in the answer is supported by sources.
      - Confirms it is at JFK Terminal 5.
      - Confirms it is an outdoor rooftop space.
      - Confirms it is accessible post-security.
      - Confirms it is free to access.
    All checks are critical under this critical parent.
    """
    sources = extracted.sources if extracted and extracted.sources else None
    name_text = extracted.facility_name or ""

    fi_node = evaluator.add_parallel(
        id="Facility_Identification",
        desc="Correctly identifies the facility as the T5 Rooftop (or T5 Rooftop & Wooftop Lounge), verifying location, outdoor nature, post-security access, and free access",
        parent=parent_node,
        critical=True,
    )

    # Name supported by sources
    leaf_name = evaluator.add_leaf(
        id="facility_name_supported",
        desc="The facility name used in the answer is supported by sources",
        parent=fi_node,
        critical=True,
    )
    claim_name = (
        f"The free outdoor rooftop recreational space at JFK Airport Terminal 5 is called '{name_text}'. "
        f"This facility is also commonly referred to as 'T5 Rooftop' or 'T5 Rooftop & Wooftop Lounge'."
    )
    await evaluator.verify(
        claim=claim_name,
        node=leaf_name,
        sources=sources,
        additional_instruction=(
            "Verify on the provided web pages that the rooftop space at JFK Terminal 5 is referred to by the given name. "
            "Minor variations in punctuation or conjunctions (e.g., '&' vs 'and') should be accepted."
        ),
    )

    # Located at JFK Terminal 5
    leaf_loc = evaluator.add_leaf(
        id="facility_location_t5",
        desc="Facility is located at JFK Airport Terminal 5",
        parent=fi_node,
        critical=True,
    )
    claim_loc = "This facility is located at John F. Kennedy International Airport (JFK), Terminal 5."
    await evaluator.verify(
        claim=claim_loc,
        node=leaf_loc,
        sources=sources,
        additional_instruction="Confirm that the referenced rooftop space specifically belongs to Terminal 5 at JFK.",
    )

    # Outdoor rooftop space
    leaf_outdoor = evaluator.add_leaf(
        id="facility_is_outdoor_rooftop",
        desc="Facility is an outdoor rooftop space",
        parent=fi_node,
        critical=True,
    )
    claim_outdoor = "This facility is an outdoor rooftop space (open-air)."
    await evaluator.verify(
        claim=claim_outdoor,
        node=leaf_outdoor,
        sources=sources,
        additional_instruction="Look for explicit wording indicating 'outdoor', 'open-air', or rooftop terrace/garden.",
    )

    # Accessible after security screening (post-security)
    leaf_postsec = evaluator.add_leaf(
        id="facility_post_security",
        desc="Facility is accessible after security screening (post-security)",
        parent=fi_node,
        critical=True,
    )
    claim_postsec = "This facility is located in the post-security area (after TSA/security screening)."
    await evaluator.verify(
        claim=claim_postsec,
        node=leaf_postsec,
        sources=sources,
        additional_instruction="Accept synonyms like 'post-security', 'airside', or 'beyond security'.",
    )

    # Free to access
    leaf_free = evaluator.add_leaf(
        id="facility_free_access",
        desc="Facility is free to access",
        parent=fi_node,
        critical=True,
    )
    claim_free = "This facility is free to access for passengers (no additional admission fee required)."
    await evaluator.verify(
        claim=claim_free,
        node=leaf_free,
        sources=sources,
        additional_instruction="Look for wording like 'free', 'no fee', or 'complimentary'.",
    )


async def verify_operating_hours(
    evaluator: Evaluator,
    parent_node,
    extracted: FacilityExtraction,
) -> None:
    """
    Verifies the operating hours information.
    We use a sequential critical node to (1) check the answer's stated hours match the expected schedule,
    and (2) confirm via sources that the schedule is indeed correct.
    """
    sources = extracted.sources if extracted and extracted.sources else None
    hours_text = extracted.operating_hours or ""

    # Parent node for hours (critical)
    hours_node = evaluator.add_sequential(
        id="Operating_Hours",
        desc="Correctly provides the daily operating hours as 6:00 AM to 10:00 PM",
        parent=parent_node,
        critical=True,
    )

    # Step 1: Match check between answer and expected hours (simple reasoning)
    leaf_match = evaluator.add_leaf(
        id="operating_hours_match_expected",
        desc="Answer's stated hours are effectively equivalent to 6:00 AM to 10:00 PM",
        parent=hours_node,
        critical=True,
    )
    claim_match = (
        f"The operating hours stated in the answer ('{hours_text}') are effectively equivalent to "
        f"'6:00 AM to 10:00 PM' (accept stylistic variants like '6 AM–10 PM', '6am-10pm', or '06:00–22:00')."
    )
    await evaluator.verify(
        claim=claim_match,
        node=leaf_match,
        additional_instruction=(
            "Judge semantic equivalence, not exact formatting. If the answer omitted hours, treat this as not equivalent."
        ),
    )

    # Step 2: Source-supported truth of hours
    leaf_supported = evaluator.add_leaf(
        id="operating_hours_supported",
        desc="Operating hours 6:00 AM to 10:00 PM are supported by sources",
        parent=hours_node,
        critical=True,
    )
    claim_supported = "The T5 Rooftop at JFK Terminal 5 is open daily from 6:00 AM to 10:00 PM."
    await evaluator.verify(
        claim=claim_supported,
        node=leaf_supported,
        sources=sources,
        additional_instruction=(
            "Confirm that the webpage explicitly supports a daily schedule of 6 AM to 10 PM. Accept minor formatting variations."
        ),
    )


async def verify_size_information(
    evaluator: Evaluator,
    parent_node,
    extracted: FacilityExtraction,
) -> None:
    """
    Verifies size information. Non-critical overall, but we still perform two checks:
      (1) the answer's stated size is approximately 4,046 sq ft,
      (2) sources support the ~4,046 sq ft figure.
    """
    sources = extracted.sources if extracted and extracted.sources else None
    size_text = extracted.size_sqft or ""

    size_node = evaluator.add_parallel(
        id="Size_Information",
        desc="Provides the approximate size as 4,046 square feet",
        parent=parent_node,
        critical=False,
    )

    # Check the answer's stated size for approx equivalence
    leaf_match = evaluator.add_leaf(
        id="size_matches_expected",
        desc="Answer's stated size is approximately 4,046 square feet",
        parent=size_node,
        critical=False,
    )
    claim_size_match = (
        f"The size stated in the answer ('{size_text}') is approximately equal to 4,046 square feet. "
        f"Allow typical approximation phrasing like '~4,000 sq ft' or 'about 4,046 sq ft'."
    )
    await evaluator.verify(
        claim=claim_size_match,
        node=leaf_match,
        additional_instruction=(
            "Treat values within roughly ±5% of 4,046 sq ft as approximately equal. "
            "If the answer omitted a size, consider this not approximately equal."
        ),
    )

    # Source-supported claim that it's ~4,046 sq ft
    leaf_source = evaluator.add_leaf(
        id="size_supported_by_sources",
        desc="Sources support that the facility is approximately 4,046 square feet",
        parent=size_node,
        critical=False,
    )
    claim_size_truth = "The T5 Rooftop at JFK Terminal 5 is approximately 4,046 square feet in size."
    await evaluator.verify(
        claim=claim_size_truth,
        node=leaf_source,
        sources=sources,
        additional_instruction=(
            "Confirm that the webpage cites a size around 4,046 square feet. Accept minor approximation language."
        ),
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
    Evaluate an answer for the JFK T5 Rooftop information task.
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
        prompt=prompt_extract_facility_info(),
        template_class=FacilityExtraction,
        extraction_name="facility_info",
    )

    # Record ground-truth reference info for transparency (not used to auto-pass/fail)
    evaluator.add_ground_truth(
        {
            "accepted_names": GROUND_TRUTH["accepted_names"],
            "expected_operating_hours": GROUND_TRUTH["operating_hours"],
            "expected_approx_size_sqft": GROUND_TRUTH["approx_size_sqft"],
        },
        gt_type="reference_truth",
    )

    # Build the rubric root node (as per provided JSON)
    rubric_root = evaluator.add_parallel(
        id="Outdoor_Rooftop_Facility",
        desc="Identification of the free outdoor rooftop recreational space at JFK Airport Terminal 5 that is accessible after security, including its operating hours and size",
        parent=root,
        critical=False,
    )

    # 1) Facility Identification subtree (critical)
    await verify_facility_identification(evaluator, rubric_root, extracted)

    # 2) Operating Hours subtree (critical)
    await verify_operating_hours(evaluator, rubric_root, extracted)

    # 3) Size Information subtree (non-critical)
    await verify_size_information(evaluator, rubric_root, extracted)

    return evaluator.get_summary()