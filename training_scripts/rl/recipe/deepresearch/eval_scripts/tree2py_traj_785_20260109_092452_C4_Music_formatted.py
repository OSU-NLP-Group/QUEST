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
TASK_ID = "festival_2024_april_california"
TASK_DESCRIPTION = (
    "In 2024, a major music festival in California took place over two consecutive weekends in April. "
    "Provide the following information about this festival: (1) The exact date ranges for both weekends "
    "(including month, starting day, and ending day for each weekend), (2) The specific city in California "
    "where the festival was held, (3) The names of all three headlining artists for the festival, and (4) "
    "A reference URL from an official source or credible music publication that confirms this information."
)

GROUND_TRUTH = {
    "first_weekend": "April 12–14, 2024",   # Accept formatting variants (Apr 12-14, 2024, April 12 to 14, 2024, etc.)
    "second_weekend": "April 19–21, 2024",  # Accept formatting variants
    "city": "Indio, California",
    "headliners": ["Lana Del Rey", "Tyler, the Creator", "Doja Cat"],
}

# Credible sources guidance for the judge model (used in additional_instruction)
CREDIBLE_DOMAIN_HINT = (
    "Only consider the claim as supported if the URL is either an official festival source or a major credible "
    "music/news publication. Examples of credible domains include (but are not limited to): "
    "coachella.com, goldenvoice.com, billboard.com, rollingstone.com, pitchfork.com, variety.com, nme.com, "
    "theguardian.com, latimes.com, nytimes.com, forbes.com, bbc.com, cnn.com, apnews.com. "
    "If the URL is not from a recognizable official or major credible publication domain, treat it as NOT supported."
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FestivalExtraction(BaseModel):
    first_weekend: Optional[str] = None
    second_weekend: Optional[str] = None
    city: Optional[str] = None
    headliners: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_festival_info() -> str:
    return """
    Extract the festival information explicitly stated in the answer text. Return the following fields:

    - first_weekend: The date range of the first weekend, as written in the answer (e.g., "April 12–14, 2024" or "April 12-14, 2024" or "Apr 12 to 14, 2024").
    - second_weekend: The date range of the second weekend, as written in the answer.
    - city: The host city as written in the answer (e.g., "Indio, California" or "Indio, CA").
    - headliners: The list of all headlining artists named in the answer. Do not include non-headlining performers.
    - reference_urls: A list of all URLs explicitly mentioned in the answer as references/sources for the festival info.

    Rules:
    - Do not invent or infer any information not explicitly present in the answer.
    - If a field is missing in the answer, return null for single-value fields or an empty array for lists.
    - For reference_urls, only extract actual URLs present in the answer (including markdown links). Do not add your own.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def format_list_for_claim(items: List[str]) -> str:
    if not items:
        return "[]"
    if len(items) == 1:
        return f"['{items[0]}']"
    return "[" + ", ".join(f"'{x}'" for x in items) + "]"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_festival_tree(evaluator: Evaluator, extracted: FestivalExtraction) -> None:
    """
    Build the verification tree according to the rubric and execute verifications.
    """
    # Create the critical parent node (parallel aggregation)
    festival_node = evaluator.add_parallel(
        id="festival_information",
        desc="Provide the required information about the April 2024 California music festival held over two consecutive weekends, including exact weekend date ranges, host city, three headliners, and a credible confirming reference URL.",
        parent=evaluator.root,
        critical=True
    )

    # 1) First weekend dates exact
    first_weekend_leaf = evaluator.add_leaf(
        id="first_weekend_dates_exact",
        desc="First weekend date range is exactly April 12–14, 2024 (month and start/end days included).",
        parent=festival_node,
        critical=True
    )
    fw_val = extracted.first_weekend or ""
    fw_claim = (
        f"The first weekend date range stated in the answer is '{fw_val}'. "
        f"This is equivalent to 'April 12–14, 2024' (allowing minor formatting variants such as 'April 12-14, 2024', "
        f"'Apr 12–14, 2024', or 'April 12 to 14, 2024')."
    )
    await evaluator.verify(
        claim=fw_claim,
        node=first_weekend_leaf,
        additional_instruction="Judge equivalence flexibly: accept en dash vs hyphen, 'Apr' vs 'April', and 'to' phrasing as equivalent to April 12–14, 2024."
    )

    # 2) Second weekend dates exact
    second_weekend_leaf = evaluator.add_leaf(
        id="second_weekend_dates_exact",
        desc="Second weekend date range is exactly April 19–21, 2024 (month and start/end days included).",
        parent=festival_node,
        critical=True
    )
    sw_val = extracted.second_weekend or ""
    sw_claim = (
        f"The second weekend date range stated in the answer is '{sw_val}'. "
        f"This is equivalent to 'April 19–21, 2024' (allowing minor formatting variants such as 'April 19-21, 2024', "
        f"'Apr 19–21, 2024', or 'April 19 to 21, 2024')."
    )
    await evaluator.verify(
        claim=sw_claim,
        node=second_weekend_leaf,
        additional_instruction="Judge equivalence flexibly: accept en dash vs hyphen, 'Apr' vs 'April', and 'to' phrasing as equivalent to April 19–21, 2024."
    )

    # 3) City exact
    city_leaf = evaluator.add_leaf(
        id="city_exact",
        desc="Host city is identified as Indio, California.",
        parent=festival_node,
        critical=True
    )
    city_val = extracted.city or ""
    city_claim = (
        f"The host city stated in the answer is '{city_val}', which is equivalent to 'Indio, California'. "
        f"Accept variants like 'Indio, CA' or mentioning the venue 'Empire Polo Club in Indio, California' as equivalent."
    )
    await evaluator.verify(
        claim=city_claim,
        node=city_leaf,
        additional_instruction="Consider 'Indio, CA' or mentions of 'Empire Polo Club in Indio, California' as equivalent to 'Indio, California'."
    )

    # 4) Exactly three headliners with specific names
    headliners_leaf = evaluator.add_leaf(
        id="headliners_exact_three",
        desc="Exactly three headlining artists are provided and they are Lana Del Rey, Tyler, the Creator, and Doja Cat (no missing or extra headliners).",
        parent=festival_node,
        critical=True
    )
    extracted_headliners = extracted.headliners or []
    hl_str = format_list_for_claim(extracted_headliners)
    expected_hl = format_list_for_claim(GROUND_TRUTH["headliners"])
    headliners_claim = (
        f"The answer lists exactly three headlining artists and they are Lana Del Rey, Tyler, the Creator, and Doja Cat "
        f"(with no additional headliners). The extracted headliners list is: {hl_str}. "
        f"Minor punctuation/casing variations are acceptable; 'Tyler, the Creator' must be recognized as the same artist."
    )
    await evaluator.verify(
        claim=headliners_claim,
        node=headliners_leaf,
        additional_instruction="Ensure there are exactly three names and they correspond to Lana Del Rey, Tyler, the Creator, and Doja Cat; allow minor casing/punctuation variants."
    )

    # 5) Reference URL credible and confirms all information
    reference_leaf = evaluator.add_leaf(
        id="reference_url_credible_and_confirms",
        desc="At least one reference URL from an official source or credible music publication is provided, and it confirms the weekend dates, host city, and the three headliners.",
        parent=festival_node,
        critical=True
    )
    refs = extracted.reference_urls or []
    confirm_claim = (
        "This source confirms that the festival takes place in Indio, California over two consecutive weekends "
        "in April 2024 with the exact date ranges April 12–14 and April 19–21, 2024, and that the three headliners "
        "are Lana Del Rey, Tyler, the Creator, and Doja Cat."
    )
    await evaluator.verify(
        claim=confirm_claim,
        node=reference_leaf,
        sources=refs,
        additional_instruction=(
            "Verify that the page explicitly states the two weekend ranges (April 12–14 and April 19–21, 2024), "
            "the location (Indio, California), and the three headliners (Lana Del Rey, Tyler, the Creator, Doja Cat). "
            + CREDIBLE_DOMAIN_HINT
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
    Evaluate an answer for the April 2024 California music festival task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    evaluator.initialize(
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
        prompt=prompt_extract_festival_info(),
        template_class=FestivalExtraction,
        extraction_name="festival_extraction",
    )

    # Add ground truth to summary for reference
    evaluator.add_ground_truth(
        {
            "first_weekend": GROUND_TRUTH["first_weekend"],
            "second_weekend": GROUND_TRUTH["second_weekend"],
            "city": GROUND_TRUTH["city"],
            "headliners": GROUND_TRUTH["headliners"],
        },
        gt_type="ground_truth_festival",
    )

    # Build verification nodes and run checks
    await build_and_verify_festival_tree(evaluator, extracted)

    # Return final structured summary
    return evaluator.get_summary()