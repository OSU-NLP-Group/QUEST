import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "anthropic_cmu_grand_challenge_2025"
TASK_DESCRIPTION = (
    "In 2025, Anthropic partnered with Carnegie Mellon University's Scott Institute for Energy Innovation as a "
    "Grand Challenge Partner. What is the total funding amount committed by Anthropic, the duration of this partnership, "
    "and the primary research focus area that this funding will support?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PartnershipExtraction(BaseModel):
    partnership_parties: Optional[str] = None  # e.g., "Anthropic and Carnegie Mellon University's Scott Institute for Energy Innovation"
    role_designation: Optional[str] = None     # e.g., "Grand Challenge Partner"
    funding_amount: Optional[str] = None       # e.g., "$1 million", "US$1,000,000", "1 million dollars"
    duration: Optional[str] = None             # e.g., "three years", "3 years"
    research_focus: Optional[str] = None       # e.g., "AI for electric grid modernization and sustainability..."
    announcement_date: Optional[str] = None    # e.g., "July 15, 2025"
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_partnership() -> str:
    return """
    Extract the specific details about Anthropic's Grand Challenge partnership with Carnegie Mellon University's Scott Institute for Energy Innovation as explicitly stated in the answer text.

    Return a JSON object with the following fields:
    - partnership_parties: The parties of the partnership as stated in the answer (e.g., "Anthropic and Carnegie Mellon University's Scott Institute for Energy Innovation"). If not stated, return null.
    - role_designation: The designation or role of Anthropic in the partnership (e.g., "Grand Challenge Partner"). If not stated, return null.
    - funding_amount: The total funding amount committed by Anthropic as stated in the answer (e.g., "$1 million", "US$1,000,000", "1 million dollars"). If not stated, return null.
    - duration: The duration of the funding/partnership as stated in the answer (e.g., "three years", "3 years", "three-year"). If not stated, return null.
    - research_focus: The primary research focus area as stated in the answer (e.g., "AI for electric grid modernization and sustainability, emphasizing energy efficiency and resilience"). If not stated, return null.
    - announcement_date: The announcement date as stated in the answer (e.g., "July 15, 2025" or "2025-07-15"). If not stated, return null.
    - source_urls: An array of all URLs explicitly cited in the answer that are relevant to this partnership (e.g., press releases, official announcements, university pages). Include each URL exactly as presented; if a URL is missing a protocol, prepend http://. If no URLs are mentioned, return an empty array.

    Important:
    - Do not infer or add information that is not explicitly present in the answer.
    - Preserve the exact phrasing from the answer for string fields when possible.
    - Include all relevant URLs mentioned in the answer (plain or markdown links).
    """


# --------------------------------------------------------------------------- #
# Helper for constructing verification subtrees                               #
# --------------------------------------------------------------------------- #
async def add_two_step_check(
    evaluator: Evaluator,
    parent_node,
    base_id: str,
    node_desc: str,
    stated_desc: str,
    stated_claim: str,
    supported_desc: str,
    supported_claim: str,
    sources: List[str],
    stated_instruction: str,
    supported_instruction: str,
) -> None:
    """
    Add a sequential two-step verification:
    1) Check that the answer explicitly states the fact (simple verification).
    2) Check that the cited sources support the fact (URL verification).
    If step 1 fails, step 2 is skipped automatically.
    """
    seq_node = evaluator.add_sequential(
        id=base_id,
        desc=node_desc,
        parent=parent_node,
        critical=True
    )

    # Step 1: stated in answer
    stated_leaf = evaluator.add_leaf(
        id=f"{base_id}_stated",
        desc=stated_desc,
        parent=seq_node,
        critical=True
    )
    await evaluator.verify(
        claim=stated_claim,
        node=stated_leaf,
        additional_instruction=stated_instruction
    )

    # Step 2: supported by sources
    supported_leaf = evaluator.add_leaf(
        id=f"{base_id}_supported",
        desc=supported_desc,
        parent=seq_node,
        critical=True
    )
    await evaluator.verify(
        claim=supported_claim,
        node=supported_leaf,
        sources=sources,
        additional_instruction=supported_instruction
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
    Evaluate an answer for the Anthropic–CMU Scott Institute Grand Challenge partnership details.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root aggregation
        agent_name=agent_name,
        answer_name=answer_name,
        client=client,
        task_description=TASK_DESCRIPTION,
        answer=answer,
        global_cache=cache,
        global_semaphore=semaphore,
        logger=logger,
        default_model=model
    )

    # Main critical node to mirror the rubric root
    main_node = evaluator.add_parallel(
        id="Anthropic_CMU_Grand_Challenge_Partnership",
        desc=(
            "Evaluate whether the answer satisfies all stated constraints and provides the requested details about "
            "Anthropic's Grand Challenge Partnership with Carnegie Mellon University's Scott Institute for Energy Innovation."
        ),
        parent=root,
        critical=True
    )

    # Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_partnership(),
        template_class=PartnershipExtraction,
        extraction_name="partnership_extraction"
    )

    # Add expected ground truth info (for context in output, not used for scoring directly)
    evaluator.add_ground_truth({
        "expected_total_funding": "$1 million",
        "expected_duration": "three years",
        "expected_primary_research_focus": "AI for electric grid modernization and sustainability (energy efficiency and resilience)",
        "expected_announcement_date": "July 15, 2025",
        "expected_role_designation": "Grand Challenge Partner",
        "expected_parties": "Anthropic and Carnegie Mellon University's Scott Institute for Energy Innovation"
    })

    # Use the extracted URLs (can be empty list if none provided)
    urls = extracted.source_urls if extracted and extracted.source_urls else []

    # 1) Partnership Parties
    await add_two_step_check(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="Partnership_Parties",
        node_desc="States that the partnership is between Anthropic and Carnegie Mellon University's Scott Institute for Energy Innovation.",
        stated_desc="Answer explicitly states the partnership is between Anthropic and CMU's Scott Institute for Energy Innovation.",
        stated_claim=(
            "In the answer text, the partnership is described as between Anthropic and Carnegie Mellon University's "
            "Scott Institute for Energy Innovation (also known as the Wilton E. Scott Institute for Energy Innovation or 'Scott Institute')."
        ),
        supported_desc="This partnership pairing is supported by the cited sources.",
        supported_claim=(
            "Anthropic partnered with Carnegie Mellon University's Scott Institute for Energy Innovation."
        ),
        sources=urls,
        stated_instruction=(
            "Check only the answer text. Allow reasonable name variants such as 'CMU's Scott Institute', "
            "'Scott Institute for Energy Innovation', or 'Wilton E. Scott Institute for Energy Innovation'."
        ),
        supported_instruction=(
            "Verify that at least one cited source explicitly states that Anthropic partnered with Carnegie Mellon University's "
            "Scott Institute for Energy Innovation (allowing common variants of the institute's name)."
        )
    )

    # 2) Role Designation: Grand Challenge Partner
    await add_two_step_check(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="Partnership_Role_Designation",
        node_desc='States that Anthropic is designated as a "Grand Challenge Partner" in this partnership.',
        stated_desc='Answer explicitly states Anthropic is a "Grand Challenge Partner".',
        stated_claim=(
            "In the answer text, Anthropic is designated as a 'Grand Challenge Partner' in this partnership."
        ),
        supported_desc='The "Grand Challenge Partner" designation is supported by the cited sources.',
        supported_claim=(
            "Anthropic is designated as a 'Grand Challenge Partner' in this partnership."
        ),
        sources=urls,
        stated_instruction=(
            "Check only the answer text. Allow minor phrasing variations such as 'Grand Challenge partnership partner' "
            "or inclusion of quotes/capitalization differences."
        ),
        supported_instruction=(
            "Confirm that at least one cited source clearly describes Anthropic as a 'Grand Challenge Partner'. "
            "Minor capitalization or punctuation differences are acceptable."
        )
    )

    # 3) Total Funding Amount: $1 million
    await add_two_step_check(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="Total_Funding_Amount",
        node_desc="States the total Anthropic funding commitment is $1 million.",
        stated_desc="Answer explicitly states the total funding commitment is $1 million (or equivalent phrasing).",
        stated_claim=(
            "In the answer text, the total funding commitment by Anthropic is stated as $1 million (accept equivalent forms such as "
            "'US$1,000,000', '1 million dollars', 'USD 1M', or 'US$1M')."
        ),
        supported_desc="The $1 million funding commitment is supported by the cited sources.",
        supported_claim="Anthropic committed $1 million (about USD 1,000,000) in funding for this partnership.",
        sources=urls,
        stated_instruction=(
            "Check only the answer text and confirm it states $1 million or an equivalent expression (e.g., '1 million dollars', "
            "'US$1,000,000', 'US$1M', 'USD 1M')."
        ),
        supported_instruction=(
            "Verify that at least one cited source explicitly states the total Anthropic funding commitment is $1 million. "
            "Allow numeric and currency-format variations that clearly indicate one million US dollars."
        )
    )

    # 4) Funding/Partnership Duration: three years
    await add_two_step_check(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="Funding_Duration",
        node_desc="States the funding/partnership duration is three years.",
        stated_desc="Answer explicitly states the duration is three years.",
        stated_claim=(
            "In the answer text, the partnership/funding duration is stated as three years (accept '3 years' or 'three-year')."
        ),
        supported_desc="The three-year duration is supported by the cited sources.",
        supported_claim="The partnership/funding duration is three years.",
        sources=urls,
        stated_instruction=(
            "Check only the answer text. Accept minor phrasing like 'three-year' or '3 years'."
        ),
        supported_instruction=(
            "Verify that at least one cited source explicitly states the partnership/funding duration is three years. "
            "Accept phrasing variants like '3-year'."
        )
    )

    # 5) Primary Research Focus Area
    await add_two_step_check(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="Primary_Research_Focus_Area",
        node_desc="Describes the primary research focus as AI for electric grid modernization and sustainability, emphasizing energy efficiency and resilience.",
        stated_desc="Answer explicitly describes the primary research focus accordingly.",
        stated_claim=(
            "In the answer text, the primary research focus is described as applying AI to electric grid modernization and sustainability, "
            "with emphasis on energy efficiency and grid resilience. Allow equivalent phrasing conveying the same meaning."
        ),
        supported_desc="The stated primary research focus is supported by the cited sources.",
        supported_claim=(
            "The funding will support research applying AI to electric grid modernization and sustainability, emphasizing energy efficiency "
            "and grid resilience."
        ),
        sources=urls,
        stated_instruction=(
            "Check only the answer text. Allow equivalent wording that clearly conveys AI applied to electric grid modernization and sustainability, "
            "with emphasis on energy efficiency and resilience."
        ),
        supported_instruction=(
            "Confirm that at least one cited source describes the research focus as applying AI to modernize the electric grid and improve "
            "sustainability, including energy efficiency and grid resilience."
        )
    )

    # 6) Announcement Date: July 15, 2025
    await add_two_step_check(
        evaluator=evaluator,
        parent_node=main_node,
        base_id="Announcement_Date",
        node_desc="States the announcement date is July 15, 2025.",
        stated_desc="Answer explicitly states the announcement date as July 15, 2025.",
        stated_claim=(
            "In the answer text, the announcement date is stated as July 15, 2025 (accept 'July 15th, 2025' or '2025-07-15')."
        ),
        supported_desc="The July 15, 2025 announcement date is supported by the cited sources.",
        supported_claim="The announcement date of this partnership was July 15, 2025.",
        sources=urls,
        stated_instruction=(
            "Check only the answer text. Accept minor date formatting variants (e.g., 'July 15th, 2025' or ISO '2025-07-15')."
        ),
        supported_instruction=(
            "Verify that at least one cited source clearly indicates the announcement date was July 15, 2025. "
            "Accept minor formatting variants."
        )
    )

    # Return the evaluation summary
    return evaluator.get_summary()