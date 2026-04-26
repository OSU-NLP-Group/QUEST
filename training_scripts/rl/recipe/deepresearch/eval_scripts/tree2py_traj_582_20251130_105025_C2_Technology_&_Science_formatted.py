import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "apple_bf2025_laptop_175gc"
TASK_DESCRIPTION = (
    "During Apple's Black Friday 2025 Shopping Event (November 28 - December 1, 2025), "
    "Apple offered Apple Gift Cards with eligible purchases, with gift card amounts varying by product. "
    "Identify which specific Apple laptop model qualified for exactly a $175 Apple Gift Card during this promotional event. "
    "Provide the laptop model's name, its starting price as advertised during the event, and its screen size. "
    "Include a reference URL from Apple's official Shopping Event page or Apple Store that confirms your answer."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AppleLaptopAnswer(BaseModel):
    """
    Structured extraction from the agent's answer.
    """
    model_name: Optional[str] = None
    starting_price: Optional[str] = None
    screen_size: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop_info() -> str:
    return """
    From the answer, extract the single Apple laptop model that the answer claims qualified for exactly a $175 Apple Gift Card during Apple's Black Friday 2025 Shopping Event (November 28 – December 1, 2025).
    Return the following fields:

    - model_name: The exact official model name as stated (e.g., "MacBook Air 13‑inch (M3)" or similar).
    - starting_price: The "Starting at" price as shown/advertised during the event (a string, keep any currency symbols or formatting, e.g., "$999").
    - screen_size: The screen size of the identified model, as written (e.g., "13-inch", "15-inch", "13.6‑inch").
    - reference_urls: All URLs explicitly provided in the answer that are meant to support this identification or its specifications. Include any Apple Shopping Event page(s) and/or Apple Store product page(s) mentioned. Do not create or infer URLs; only include URLs explicitly present in the answer text. Keep full URLs with protocol. Include all relevant URLs in the order they appear.

    If a field is missing in the answer, set it to null (or an empty array for reference_urls).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _apple_first(urls: List[str]) -> List[str]:
    """
    Reorder URLs to try Apple domains first (apple.com/*), preserving relative order otherwise.
    """
    apple = [u for u in urls if "apple.com" in (u or "").lower()]
    non_apple = [u for u in urls if "apple.com" not in (u or "").lower()]
    return apple + non_apple


# --------------------------------------------------------------------------- #
# Verification subroutine                                                     #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, extracted: AppleLaptopAnswer) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    """
    # Create a critical sequential node to represent the rubric root:
    # "Apple_Laptop_Identification_Task"
    task_node = evaluator.add_sequential(
        id="Apple_Laptop_Identification_Task",
        desc="Evaluate the complete identification of the Apple laptop qualifying for a $175 gift card during the Black Friday 2025 Shopping Event, along with its specifications",
        parent=evaluator.root,
        critical=True
    )

    # Child 1 (leaf): Product_Identification (critical)
    product_ident_leaf = evaluator.add_leaf(
        id="Product_Identification",
        desc="Correctly identify which Apple laptop model qualifies for exactly a $175 Apple Gift Card during Apple's Black Friday 2025 Shopping Event (November 28 - December 1, 2025)",
        parent=task_node,
        critical=True
    )

    # Verify the model and $175 gift card qualification
    model_name = extracted.model_name or ""
    refs = _apple_first(extracted.reference_urls or [])
    identification_claim = (
        f"The Apple laptop model '{model_name}' qualified for exactly a $175 Apple Gift Card during Apple's "
        f"Black Friday 2025 Shopping Event (November 28 – December 1, 2025)."
    )
    await evaluator.verify(
        claim=identification_claim,
        node=product_ident_leaf,
        sources=refs,  # may be empty; if empty, verification will be simple (see additional_instruction)
        additional_instruction=(
            "Decide only based on official Apple webpages when URLs are provided. "
            "To mark Correct, at least one provided URL must clearly show that the exact identified laptop model "
            "qualified for a $175 Apple Gift Card specifically during Apple's Shopping Event 2025 (Nov 28–Dec 1, 2025). "
            "Accept Apple Shopping Event pages or Apple Store product pages that visibly show the event banner/terms "
            "for this model with the $175 amount. If no URLs are provided, or if the provided pages are not on apple.com "
            "or do not explicitly state the $175 amount for the identified model, mark Incorrect."
        )
    )

    # Child 2 (parallel, critical): Product_Specifications
    specs_node = evaluator.add_parallel(
        id="Product_Specifications",
        desc="Provide accurate specifications for the identified laptop",
        parent=task_node,
        critical=True
    )

    # Leaf under specs: Specifications_Reference (critical)
    specs_ref_leaf = evaluator.add_leaf(
        id="Specifications_Reference",
        desc="Provide a valid URL from Apple's official Black Friday Shopping Event page or Apple Store that confirms the product specifications",
        parent=specs_node,
        critical=True
    )
    specs_ref_claim = (
        "At least one of the provided URLs is an official Apple website (apple.com) and it is either the "
        "Apple Black Friday 2025 Shopping Event page (Nov 28 – Dec 1, 2025) or an Apple Store product page "
        "that mentions the identified laptop model and includes product details (such as price or screen size)."
    )
    await evaluator.verify(
        claim=specs_ref_claim,
        node=specs_ref_leaf,
        sources=refs,
        additional_instruction=(
            "Mark Correct only if at least one URL is on apple.com and is clearly relevant: either the Shopping Event 2025 page "
            "or an Apple Store product page that references the identified model and shows key details (price or screen size). "
            "If no URLs are provided, or none are on apple.com, or none mention the identified model or specs, mark Incorrect."
        )
    )

    # Leaf under specs: Starting_Price (critical)
    starting_price_leaf = evaluator.add_leaf(
        id="Starting_Price",
        desc="Provide the correct starting price for the identified laptop model as stated during the Black Friday 2025 Shopping Event",
        parent=specs_node,
        critical=True
    )
    starting_price = extracted.starting_price or ""
    starting_price_claim = (
        f"During Apple's Black Friday 2025 Shopping Event, the starting price for '{model_name}' was '{starting_price}'."
    )
    await evaluator.verify(
        claim=starting_price_claim,
        node=starting_price_leaf,
        sources=refs,
        additional_instruction=(
            "Verify the 'Starting at' price for the identified model using the provided Apple URLs. "
            "Accept exact matches or clearly equivalent phrasing like 'From $X'. Currency formatting should be treated leniently "
            "(e.g., $999 vs 999 USD). If the provided Apple pages do not show this price for the model during the event, "
            "or no Apple URL is provided, mark Incorrect."
        )
    )

    # Leaf under specs: Screen_Size (critical)
    screen_size_leaf = evaluator.add_leaf(
        id="Screen_Size",
        desc="Provide the correct screen size specification for the identified laptop model",
        parent=specs_node,
        critical=True
    )
    screen_size = extracted.screen_size or ""
    screen_size_claim = f"The screen size of '{model_name}' is {screen_size}."
    await evaluator.verify(
        claim=screen_size_claim,
        node=screen_size_leaf,
        sources=refs,
        additional_instruction=(
            "Verify the model's display size on the Apple page(s). Allow reasonable formatting variants (e.g., 13-inch vs 13 in vs 13.6‑inch). "
            "If the size shown on the Apple page(s) does not match the stated size, or no Apple URL is provided, mark Incorrect."
        )
    )


# --------------------------------------------------------------------------- #
# Main evaluation entry point                                                 #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Apple Black Friday 2025 laptop ($175 gift card) task.
    """
    # Initialize evaluator (root is non-critical by framework design)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Outer root strategy; rubric root is added as a child critical node
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
        prompt=prompt_extract_laptop_info(),
        template_class=AppleLaptopAnswer,
        extraction_name="extracted_laptop_info",
    )

    # Optional: record some custom context for transparency
    evaluator.add_custom_info(
        {
            "event_window": "Nov 28 – Dec 1, 2025",
            "target_gift_card_amount": "$175",
            "note": "All verifications should prioritize official Apple pages (apple.com).",
        },
        info_type="context",
        info_name="evaluation_context",
    )

    # Build verification tree and run verifications
    await build_and_verify_tree(evaluator, extracted)

    # Return standard summary
    return evaluator.get_summary()