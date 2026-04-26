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
TASK_ID = "tn_veto_override_minimum_combined"
TASK_DESCRIPTION = """
In the Tennessee General Assembly, what is the minimum combined number of legislators from both chambers required to successfully override a gubernatorial veto?
"""

# Ground truth and arithmetic expectations (used for transparent reference only)
SENATE_SIZE_GT = 33
HOUSE_SIZE_GT = 99
SENATE_MIN_GT = SENATE_SIZE_GT // 2 + 1  # floor(33/2) + 1 = 17
HOUSE_MIN_GT = HOUSE_SIZE_GT // 2 + 1    # floor(99/2) + 1 = 50
COMBINED_MIN_GT = SENATE_MIN_GT + HOUSE_MIN_GT  # 17 + 50 = 67


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class OverrideExtraction(BaseModel):
    """
    Structure information we try to extract from the agent's answer.
    All fields are optional; if not present in the answer, they can remain None.
    """
    rule_statement: Optional[str] = None  # any description of the override rule noted by the agent
    both_chambers_statement: Optional[str] = None  # explicit mention that both chambers must independently meet thresholds
    one_of_six_statement: Optional[str] = None  # any sentence claiming TN is one of six states with simple-majority override
    senate_size: Optional[str] = None
    house_size: Optional[str] = None
    senate_minimum_votes: Optional[str] = None
    house_minimum_votes: Optional[str] = None
    combined_minimum: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)  # any URLs the answer cited


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_override_info() -> str:
    return """
    Extract information from the answer about Tennessee's gubernatorial veto override requirement.

    Fields to extract (return null if not present):
    1) rule_statement: Any sentence/phrase that describes how a veto override works in Tennessee.
    2) both_chambers_statement: If the answer explicitly states that both chambers (Senate and House) must each meet their own threshold for the override to succeed, copy that sentence/phrase. Otherwise null.
    3) one_of_six_statement: If the answer states (or implies) that Tennessee is one of six U.S. states where a simple majority of elected members suffices to override a veto, copy that sentence/phrase. Otherwise null.
    4) senate_size: The number of elected members in the Tennessee Senate (as mentioned in the answer), e.g., "33".
    5) house_size: The number of elected members in the Tennessee House of Representatives (as mentioned in the answer), e.g., "99".
    6) senate_minimum_votes: The minimum number of votes in the Tennessee Senate needed to override the veto (as mentioned in the answer), e.g., "17".
    7) house_minimum_votes: The minimum number of votes in the Tennessee House needed to override the veto (as mentioned in the answer), e.g., "50".
    8) combined_minimum: The final minimum combined number from both chambers that the answer reports (e.g., "67").
    9) source_urls: Extract all URLs (in any reasonable format, including Markdown links) that the answer provides as citations and list them. If none are given, return an empty list.

    IMPORTANT:
    - Do not invent information that is not explicitly in the answer.
    - If a value is numeric, still return it as a string exactly as shown in the answer.
    - For source_urls, include only actual URLs you can find in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper functions and references                                             #
# --------------------------------------------------------------------------- #
def get_reference_urls() -> Dict[str, List[str]]:
    """
    Provide a small set of fallback authoritative URLs for verification, used when
    the answer does not provide sources or provides incomplete ones.

    Note: These are general references known to discuss state veto override thresholds
    and Tennessee legislature composition.
    """
    rule_urls = [
        # National Conference of State Legislatures overview of veto override thresholds
        "https://www.ncsl.org/legislators-staff/legislative-veto-override-requirements.aspx",
        # Ballotpedia coverage on veto overrides by state
        "https://ballotpedia.org/Veto_override",
    ]
    senate_urls = [
        # Membership size is typically stated on the chamber page
        "https://en.wikipedia.org/wiki/Tennessee_Senate",
    ]
    house_urls = [
        "https://en.wikipedia.org/wiki/Tennessee_House_of_Representatives",
    ]
    return {
        "rule": rule_urls,
        "senate": senate_urls,
        "house": house_urls,
    }


def merge_sources(primary: List[str], fallbacks: List[str]) -> List[str]:
    """Merge and deduplicate URL lists while preserving order."""
    seen = set()
    merged = []
    for url in list(primary) + list(fallbacks):
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            merged.append(url)
    return merged


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def add_and_verify_override_rule(evaluator: Evaluator, parent, extracted: OverrideExtraction) -> None:
    """
    OverrideRule (parallel, adjusted to non-critical to avoid unfairly failing answers
    that don't include broader context facts not strictly required by the question).
    """
    refs = get_reference_urls()
    combined_rule_sources = merge_sources(extracted.source_urls, refs["rule"])

    rule_node = evaluator.add_parallel(
        id="OverrideRule",
        desc="Identify the override voting rule and success condition",
        parent=parent,
        critical=False  # adjusted from JSON to avoid over-penalizing
    )

    # BothChambersRequired
    both_node = evaluator.add_leaf(
        id="BothChambersRequired",
        desc="States that both chambers must independently meet their thresholds for the override to succeed",
        parent=rule_node,
        critical=False  # adjusted to non-critical
    )
    both_claim = (
        "In Tennessee, overriding a gubernatorial veto requires each chamber (the Senate and the House) "
        "to individually meet the required threshold; it is not a single combined vote across both chambers."
    )
    await evaluator.verify(
        claim=both_claim,
        node=both_node,
        sources=combined_rule_sources,
        additional_instruction="Look for phrasing such as 'each House' or 'each chamber' needing a majority to repass the bill over the veto."
    )

    # OneOfSixStatesFact
    six_node = evaluator.add_leaf(
        id="OneOfSixStatesFact",
        desc="States that Tennessee is one of six U.S. states that require only a simple majority (of elected members) to override a gubernatorial veto",
        parent=rule_node,
        critical=False  # adjusted to non-critical
    )
    six_claim = (
        "Tennessee is one of six U.S. states that allow a gubernatorial veto to be overridden by a simple majority "
        "of the elected members in each chamber."
    )
    await evaluator.verify(
        claim=six_claim,
        node=six_node,
        sources=combined_rule_sources,
        additional_instruction="Check if Tennessee is listed among states with a simple-majority override requirement. Allow minor wording variations."
    )


async def add_and_verify_chamber_sizes(evaluator: Evaluator, parent, extracted: OverrideExtraction) -> None:
    """
    ChamberSizes (parallel, kept critical as in JSON since sizes underpin the computation).
    """
    refs = get_reference_urls()
    senate_sources = merge_sources(extracted.source_urls, refs["senate"])
    house_sources = merge_sources(extracted.source_urls, refs["house"])

    sizes_node = evaluator.add_parallel(
        id="ChamberSizes",
        desc="Determine the number of elected members in each chamber",
        parent=parent,
        critical=True
    )

    senate_leaf = evaluator.add_leaf(
        id="SenateSize",
        desc="Tennessee Senate has 33 elected members",
        parent=sizes_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Tennessee Senate has 33 elected members.",
        node=senate_leaf,
        sources=senate_sources,
        additional_instruction="Verify the membership size (number of seats) of the Tennessee Senate."
    )

    house_leaf = evaluator.add_leaf(
        id="HouseSize",
        desc="Tennessee House has 99 elected members",
        parent=sizes_node,
        critical=True
    )
    await evaluator.verify(
        claim="The Tennessee House of Representatives has 99 elected members.",
        node=house_leaf,
        sources=house_sources,
        additional_instruction="Verify the membership size (number of seats) of the Tennessee House of Representatives."
    )


async def add_and_verify_minimums(evaluator: Evaluator, parent, extracted: OverrideExtraction) -> None:
    """
    MinimumVotesPerChamber (parallel, critical), combining definitional logic and arithmetic
    with evidence for the underlying facts (sizes and rule).
    """
    refs = get_reference_urls()
    combined_sources = merge_sources(extracted.source_urls, refs["rule"] + refs["senate"] + refs["house"])

    min_node = evaluator.add_parallel(
        id="MinimumVotesPerChamber",
        desc="Compute the minimum votes required in each chamber given 'simple majority' means more than half",
        parent=parent,
        critical=True
    )

    # SimpleMajorityDefinition
    def_leaf = evaluator.add_leaf(
        id="SimpleMajorityDefinition",
        desc="Uses/acknowledges that a simple majority is more than half of the elected members in a chamber",
        parent=min_node,
        critical=True
    )
    await evaluator.verify(
        claim="A simple majority means strictly more than half of the elected members in a chamber.",
        node=def_leaf,
        additional_instruction="This is a standard definition; confirm the logical meaning of 'simple majority' as > 50%."
    )

    # SenateMinimumComputation
    senate_min_leaf = evaluator.add_leaf(
        id="SenateMinimumComputation",
        desc="Correctly derives the Senate minimum as floor(33/2) + 1",
        parent=min_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In Tennessee, the minimum number of votes needed in the Senate to override a veto is {SENATE_MIN_GT} (which equals floor(33/2)+1).",
        node=senate_min_leaf,
        sources=combined_sources,
        additional_instruction=(
            "You may derive this from two supported facts: (1) the Senate has 33 members; "
            "(2) the override requires a simple majority of elected members in each chamber. "
            "Therefore, a majority of 33 is 17."
        )
    )

    # HouseMinimumComputation
    house_min_leaf = evaluator.add_leaf(
        id="HouseMinimumComputation",
        desc="Correctly derives the House minimum as floor(99/2) + 1",
        parent=min_node,
        critical=True
    )
    await evaluator.verify(
        claim=f"In Tennessee, the minimum number of votes needed in the House to override a veto is {HOUSE_MIN_GT} (which equals floor(99/2)+1).",
        node=house_min_leaf,
        sources=combined_sources,
        additional_instruction=(
            "You may derive this from two supported facts: (1) the House has 99 members; "
            "(2) the override requires a simple majority of elected members in each chamber. "
            "Therefore, a majority of 99 is 50."
        )
    )


async def add_and_verify_combined_minimum(evaluator: Evaluator, parent, extracted: OverrideExtraction) -> None:
    """
    CombinedMinimum (leaf, critical): verify the final combined minimum as the sum of the chamber minimums.
    """
    refs = get_reference_urls()
    combined_sources = merge_sources(extracted.source_urls, refs["rule"] + refs["senate"] + refs["house"])

    combined_leaf = evaluator.add_leaf(
        id="CombinedMinimum",
        desc="Final answer is the sum of the chamber minimums (Senate minimum + House minimum) as the minimum combined number required",
        parent=parent,
        critical=True
    )
    await evaluator.verify(
        claim=f"The minimum combined number of legislators from both chambers required to successfully override a Tennessee gubernatorial veto is {COMBINED_MIN_GT}.",
        node=combined_leaf,
        sources=combined_sources,
        additional_instruction=(
            "This is not a single joint vote; rather, it reflects the simultaneous minimums that must be met in each chamber. "
            f"Use the supported minima (Senate {SENATE_MIN_GT} and House {HOUSE_MIN_GT}) to accept {COMBINED_MIN_GT} as the combined figure."
        )
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
    Build the verification tree and evaluate the agent's answer.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level aggregator stays non-critical
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

    # 1) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_override_info(),
        template_class=OverrideExtraction,
        extraction_name="override_extraction",
    )

    # 2) Ground truth info for transparency
    evaluator.add_ground_truth({
        "senate_size": SENATE_SIZE_GT,
        "house_size": HOUSE_SIZE_GT,
        "senate_minimum": SENATE_MIN_GT,
        "house_minimum": HOUSE_MIN_GT,
        "combined_minimum": COMBINED_MIN_GT,
        "notes": "Computed as simple majority (> half) of elected members in each chamber."
    }, gt_type="expected_values")

    # 3) Build the OverrideVoteRequirement subtree (sequential flow), adjusted to non-critical to allow partial scoring
    main = evaluator.add_sequential(
        id="OverrideVoteRequirement",
        desc="Minimum combined number of legislators needed to override a Tennessee gubernatorial veto",
        parent=root,
        critical=False  # adjusted from JSON to avoid total failure on ancillary misses
    )

    # 3.1 Override rule and success condition
    await add_and_verify_override_rule(evaluator, main, extracted)

    # 3.2 Chamber sizes (critical)
    await add_and_verify_chamber_sizes(evaluator, main, extracted)

    # 3.3 Minimum votes per chamber (critical)
    await add_and_verify_minimums(evaluator, main, extracted)

    # 3.4 Final combined minimum (critical)
    await add_and_verify_combined_minimum(evaluator, main, extracted)

    # 4) Return the summary
    return evaluator.get_summary()