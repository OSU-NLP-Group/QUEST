import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "aa_carry_on_policy"
TASK_DESCRIPTION = (
    "A traveler is preparing for a flight on American Airlines and wants to verify if their carry-on bag meets the airline's size requirements. "
    "According to American Airlines' official policy, what are the maximum allowed dimensions (length × width × height) for a carry-on bag, "
    "and must these measurements include the bag's handles and wheels?"
)

# Optional: Known official AA policy URL(s) for reference (not used as gating sources in this rubric)
AA_OFFICIAL_POLICY_URLS = [
    "https://www.aa.com/i18n/travel-info/baggage/carry-on-baggage.jsp"
]

# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class CarryOnPolicyExtraction(BaseModel):
    """
    Extract from the answer:
    - Any stated maximum carry-on dimensions as raw text (e.g., '22 x 14 x 9 inches')
    - Any explicitly labeled length/width/height values as separate strings, if present
    - The unit used for dimensions (e.g., 'in', 'inches', 'cm')
    - Whether the answer explicitly states that measurements include handles and wheels
    - Any source URLs included in the answer
    """
    dimensions_raw: Optional[str] = None
    length: Optional[str] = None
    width: Optional[str] = None
    height: Optional[str] = None
    unit: Optional[str] = None
    labels_order_lwh: Optional[bool] = None
    includes_handles_wheels: Optional[bool] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_policy_from_answer() -> str:
    return (
        "From the answer text, extract the following fields about American Airlines' carry-on policy:\n"
        "1) dimensions_raw: The maximum carry-on size text as presented (e.g., '22 x 14 x 9 inches', '22×14×9 in').\n"
        "2) length: The value that corresponds to the length if explicitly labeled (string; else null).\n"
        "3) width: The value that corresponds to the width if explicitly labeled (string; else null).\n"
        "4) height: The value that corresponds to the height if explicitly labeled (string; else null).\n"
        "5) unit: The unit used for the dimensions as stated (e.g., 'inches', 'in', 'cm', or null if not stated).\n"
        "6) labels_order_lwh: true if the answer explicitly labels the order as length × width × height or clearly maps each value to L/W/H; false otherwise.\n"
        "7) includes_handles_wheels: true if the answer explicitly says measurements must include handles and wheels (exterior dimensions); false otherwise.\n"
        "8) source_urls: All URLs cited in the answer (extract actual URL strings as-is)."
    )


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    extraction: CarryOnPolicyExtraction
) -> None:
    """
    Build the verification tree per rubric:
    Root (non-critical, created by Evaluator.initialize)
     └── AA_CarryOn_Policy_Answer_Correctness (critical, parallel)
         ├── Max_Dimensions_LengthWidthHeight (critical leaf)
         └── Handles_And_Wheels_Included (critical leaf)
    """

    # Create the rubric root node as a critical parallel node under evaluator.root
    rubric_root = evaluator.add_parallel(
        id="AA_CarryOn_Policy_Answer_Correctness",
        desc="Evaluate whether the response correctly states American Airlines’ maximum carry-on dimensions and whether dimensions include handles and wheels.",
        parent=evaluator.root,
        critical=True
    )

    # Leaf 1: Max dimensions stated as 22 × 14 × 9 inches, L × W × H (or explicitly labeled which is which)
    dims_leaf = evaluator.add_leaf(
        id="Max_Dimensions_LengthWidthHeight",
        desc="Answer states the maximum allowed carry-on dimensions are 22 × 14 × 9 inches in the order length × width × height (or explicitly labels which dimension is which).",
        parent=rubric_root,
        critical=True
    )
    dims_claim = (
        "The answer text states that the maximum allowed American Airlines carry-on dimensions are 22 × 14 × 9 inches. "
        "This can be written with 'x', '×', or 'by' and in any reasonable spacing. "
        "It either uses the conventional order (length × width × height) or explicitly labels which value is length, which is width, and which is height."
    )
    dims_additional_instruction = (
        "Judge based solely on the answer text. Accept variations like '22x14x9 in', '22 by 14 by 9 inches', or the '×' symbol. "
        "Accept if the answer explicitly labels length/width/height even if not in that exact order. "
        "Require the three numeric values 22, 14, and 9 with inch units; metric equivalents may appear additionally but aren't required. "
        "If the answer uses only centimeters without the inch values 22, 14, and 9, treat it as not satisfying this check."
    )
    await evaluator.verify(
        claim=dims_claim,
        node=dims_leaf,
        additional_instruction=dims_additional_instruction
    )

    # Leaf 2: Explicit statement that measurements include handles and wheels
    haw_leaf = evaluator.add_leaf(
        id="Handles_And_Wheels_Included",
        desc="Answer explicitly states that the measurements must include the bag’s handles and wheels (i.e., are exterior dimensions).",
        parent=rubric_root,
        critical=True
    )
    haw_claim = (
        "The answer text explicitly states that the carry-on measurements must include the bag’s handles and wheels "
        "(i.e., the exterior dimensions include wheels and handles)."
    )
    haw_additional_instruction = (
        "Judge based solely on the answer text. Look for explicit language such as 'including handles and wheels', "
        "'include wheels and handles', 'measurements include wheels/handles', or equivalent phrasing. "
        "Implicit hints without explicitly mentioning both handles and wheels should not be accepted."
    )
    await evaluator.verify(
        claim=haw_claim,
        node=haw_leaf,
        additional_instruction=haw_additional_instruction
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
    Evaluate an answer for American Airlines carry-on policy sizing.
    Returns a structured summary with the verification tree and final score.
    """
    # Initialize evaluator with a parallel root
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
        default_model=model
    )

    # Extraction from the answer (recorded for transparency)
    extraction = await evaluator.extract(
        prompt=prompt_extract_policy_from_answer(),
        template_class=CarryOnPolicyExtraction,
        extraction_name="carry_on_policy_extraction"
    )

    # Add ground truth info for context (not used for gating)
    evaluator.add_ground_truth({
        "expected_dimensions_inches": "22 × 14 × 9 inches (length × width × height)",
        "handles_and_wheels_included": True,
        "official_source_hint": AA_OFFICIAL_POLICY_URLS
    }, gt_type="ground_truth_policy")

    # Build verification as per rubric and run checks
    await build_verification_tree(evaluator, extraction)

    # Return final structured evaluation summary
    return evaluator.get_summary()