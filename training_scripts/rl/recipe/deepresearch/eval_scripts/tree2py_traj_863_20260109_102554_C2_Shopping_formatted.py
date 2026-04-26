import asyncio
import logging
import re
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "bestbuy_laptop_selection"
TASK_DESCRIPTION = """
I'm looking for a portable laptop for business use from Best Buy. Find one laptop that meets all of the following requirements:

- Weighs 3.5 pounds or less
- Has a 14-inch display
- Has at least 16GB of RAM
- Has at least 512GB SSD storage
- Has Full HD (1920×1080) or higher resolution display

For the laptop you identify, provide:
1. The laptop's brand and model name
2. The exact weight in pounds
3. The RAM capacity
4. The SSD storage capacity
5. The display resolution
6. A link to the product page on Best Buy's website showing these specifications
7. The standard return policy duration (in days) that Best Buy offers for laptops
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LaptopItem(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    weight_lbs: Optional[str] = None
    ram: Optional[str] = None
    ssd: Optional[str] = None
    resolution: Optional[str] = None
    display_size_inch: Optional[str] = None
    bestbuy_url: Optional[str] = None


class LaptopSelectionExtraction(BaseModel):
    laptops: List[LaptopItem] = Field(default_factory=list)
    return_policy_days: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop_selection() -> str:
    return """
    Extract the details for up to three laptops mentioned in the answer (take them in the order they appear).
    For each laptop, extract:
    - brand: The brand/manufacturer name (e.g., Lenovo, HP, Dell).
    - model: The model name/series (e.g., ThinkPad T14 Gen 3).
    - weight_lbs: The exact weight in pounds as stated (include the numeric value and unit, e.g., "3.0 lb", "2.9 pounds").
    - ram: The memory capacity as stated (e.g., "16GB", "32 GB").
    - ssd: The SSD storage capacity as stated (e.g., "512GB", "1TB SSD").
    - resolution: The display resolution (e.g., "1920×1080", "2560x1600", "FHD", "UHD", "4K").
    - display_size_inch: The display size (e.g., "14-inch", "14.0 in", "14").
    - bestbuy_url: The Best Buy product page URL for the laptop (must be a valid URL; if missing protocol, prepend http://).

    Return a JSON with:
    {
      "laptops": [
         { brand, model, weight_lbs, ram, ssd, resolution, display_size_inch, bestbuy_url },
         ...
      ],
      "return_policy_days": "<the number of days for Best Buy's standard return window for laptops>"
    }

    SPECIAL RULES:
    - Extract only what is explicitly stated in the answer.
    - If any field is missing for a laptop, set it to null.
    - For URLs: extract only actual URLs shown in the answer; include full URLs; if protocol missing, prepend http://.
    - For "return_policy_days": extract the numeric duration (e.g., "15") if stated; otherwise null.
    """


# --------------------------------------------------------------------------- #
# Simple validators for existence-format checks                               #
# --------------------------------------------------------------------------- #
def _has_weight_in_pounds(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    return bool(re.search(r"\d+(\.\d+)?\s*(lb|lbs|pound|pounds)\b", text, flags=re.IGNORECASE))


def _has_ram_capacity(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    return bool(re.search(r"\b\d+(\.\d+)?\s*GB\b", text, flags=re.IGNORECASE))


def _has_ssd_capacity(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    return bool(re.search(r"\b\d+(\.\d+)?\s*(GB|TB)\b", text, flags=re.IGNORECASE))


def _has_resolution(text: Optional[str]) -> bool:
    if not text or not isinstance(text, str):
        return False
    patterns = [
        r"\b\d{3,4}\s*[x×]\s*\d{3,4}\b",
        r"\bFHD\b", r"\bFull\s*HD\b", r"\bUHD\b", r"\b4K\b",
        r"\bQHD\b", r"\bWQHD\b", r"\bWUXGA\b", r"\bWUXGA\s*1920\s*[x×]\s*1200\b",
        r"\b1080p\b"
    ]
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def _is_bestbuy_url(url: Optional[str]) -> bool:
    if not url or not isinstance(url, str):
        return False
    return "bestbuy.com" in url.lower()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root_node,
    extraction: LaptopSelectionExtraction
) -> None:
    """
    Build the verification tree according to the rubric and perform checks.
    """

    # Create top-level critical node representing the whole task
    task_node = evaluator.add_parallel(
        id="Laptop_Selection_Task",
        desc="Identify one Best Buy laptop that satisfies all constraints and report the required specifications and return-window duration.",
        parent=root_node,
        critical=True
    )

    # Choose the first laptop (if multiple were extracted)
    selected: LaptopItem = extraction.laptops[0] if extraction.laptops else LaptopItem()

    # Subtree: Required Laptop Details Reported
    details_node = evaluator.add_parallel(
        id="Required_Laptop_Details_Reported",
        desc="The response reports all requested laptop details for the selected laptop.",
        parent=task_node,
        critical=True
    )

    # Brand and model provided
    brand_model_leaf = evaluator.add_custom_node(
        result=bool(selected.brand and selected.brand.strip()) and bool(selected.model and selected.model.strip()),
        id="Brand_And_Model_Name",
        desc="Provide the laptop's brand and model name.",
        parent=details_node,
        critical=True
    )

    # Exact weight value in pounds provided (numeric + unit)
    weight_detail_leaf = evaluator.add_custom_node(
        result=_has_weight_in_pounds(selected.weight_lbs),
        id="Exact_Weight_In_Pounds",
        desc="Provide the exact weight value in pounds (a numeric value with unit).",
        parent=details_node,
        critical=True
    )

    # RAM capacity stated
    ram_detail_leaf = evaluator.add_custom_node(
        result=_has_ram_capacity(selected.ram),
        id="RAM_Capacity_Stated",
        desc="State the RAM capacity (in GB).",
        parent=details_node,
        critical=True
    )

    # SSD capacity stated
    ssd_detail_leaf = evaluator.add_custom_node(
        result=_has_ssd_capacity(selected.ssd),
        id="SSD_Capacity_Stated",
        desc="State the SSD storage capacity (in GB or TB).",
        parent=details_node,
        critical=True
    )

    # Display resolution stated
    resolution_detail_leaf = evaluator.add_custom_node(
        result=_has_resolution(selected.resolution),
        id="Display_Resolution_Stated",
        desc="State the display resolution (e.g., 1920×1080 or higher).",
        parent=details_node,
        critical=True
    )

    # Best Buy product page link provided
    bestbuy_link_leaf = evaluator.add_custom_node(
        result=_is_bestbuy_url(selected.bestbuy_url),
        id="BestBuy_Product_Page_Link",
        desc="Provide a link to the Best Buy product page for the laptop that shows the relevant specifications.",
        parent=details_node,
        critical=True
    )

    # Subtree: Laptop Meets Constraints
    constraints_node = evaluator.add_parallel(
        id="Laptop_Meets_Constraints",
        desc="The chosen laptop satisfies all stated selection constraints.",
        parent=task_node,
        critical=True
    )

    # Listed on BestBuy and corresponds to identified laptop
    listed_leaf = evaluator.add_leaf(
        id="Listed_On_BestBuy",
        desc="A Best Buy product-page URL is provided and it is on bestbuy.com for the identified laptop.",
        parent=constraints_node,
        critical=True
    )
    brand_for_claim = selected.brand or ""
    model_for_claim = selected.model or ""
    claim_listed = (
        f"This page is on bestbuy.com and shows a laptop identified as {brand_for_claim} {model_for_claim}, "
        f"or a very close variant of that model."
    )
    await evaluator.verify(
        claim=claim_listed,
        node=listed_leaf,
        sources=selected.bestbuy_url,
        extra_prerequisites=[bestbuy_link_leaf, brand_model_leaf],
        additional_instruction=(
            "Confirm that the provided URL belongs to Best Buy (bestbuy.com) and that the page title/specs mention "
            "the brand and model (allow minor suffixes or series variants such as 'Gen' or small differences)."
        ),
    )

    # Weight ≤ 3.5 lb
    weight_constraint_leaf = evaluator.add_leaf(
        id="Weight_At_Most_3_5_lb",
        desc="The laptop's weight is ≤ 3.5 pounds.",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The weight of this laptop is less than or equal to 3.5 pounds.",
        node=weight_constraint_leaf,
        sources=selected.bestbuy_url,
        extra_prerequisites=[bestbuy_link_leaf],
        additional_instruction=(
            "Check the product specifications section (e.g., 'Weight'). If weight is shown in pounds, compare directly; "
            "if shown in grams, convert to pounds and ensure it is ≤ 3.5 lb (approximately ≤ 1587.6 g)."
        ),
    )

    # Display size 14-inch
    display_size_leaf = evaluator.add_leaf(
        id="Display_Size_14_in",
        desc="The laptop has a 14-inch display.",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The laptop's screen size is 14 inches.",
        node=display_size_leaf,
        sources=selected.bestbuy_url,
        extra_prerequisites=[bestbuy_link_leaf],
        additional_instruction=(
            "Look for 'Screen Size' or similar in specifications; allow minor decimal representations like '14.0' or "
            "'14-inch class' to qualify."
        ),
    )

    # RAM ≥ 16GB
    ram_constraint_leaf = evaluator.add_leaf(
        id="RAM_At_Least_16GB",
        desc="The laptop has ≥ 16GB of RAM.",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The system memory (RAM) is at least 16 GB.",
        node=ram_constraint_leaf,
        sources=selected.bestbuy_url,
        extra_prerequisites=[bestbuy_link_leaf],
        additional_instruction=(
            "Check 'Memory' or 'System Memory' in the specs. If the page shows 16GB, 32GB, 64GB etc., consider it sufficient. "
            "If memory is upgradable, verify the current configuration meets ≥16GB."
        ),
    )

    # SSD ≥ 512GB
    ssd_constraint_leaf = evaluator.add_leaf(
        id="SSD_At_Least_512GB",
        desc="The laptop has ≥ 512GB SSD storage.",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The internal storage SSD capacity is at least 512 GB.",
        node=ssd_constraint_leaf,
        sources=selected.bestbuy_url,
        extra_prerequisites=[bestbuy_link_leaf],
        additional_instruction=(
            "Check 'Storage' or 'Solid State Drive Capacity'. Values like 512GB, 1TB, 2TB qualify as ≥512GB."
        ),
    )

    # Resolution ≥ Full HD (1920×1080)
    resolution_constraint_leaf = evaluator.add_leaf(
        id="Resolution_At_Least_FHD",
        desc="The laptop display resolution is Full HD (1920×1080) or higher.",
        parent=constraints_node,
        critical=True
    )
    await evaluator.verify(
        claim="The display resolution is 1920×1080 or higher.",
        node=resolution_constraint_leaf,
        sources=selected.bestbuy_url,
        extra_prerequisites=[bestbuy_link_leaf],
        additional_instruction=(
            "Check 'Resolution' in specifications. Resolutions like 1920×1080, 1920×1200, 2560×1600, 2880×1800, 3840×2160 (4K), etc., "
            "are all ≥ 1920×1080."
        ),
    )

    # Subtree: Return Policy Duration Reported
    return_node = evaluator.add_parallel(
        id="Return_Policy_Duration_Reported",
        desc="Provide Best Buy's standard return policy duration (in days) for laptops.",
        parent=task_node,
        critical=True
    )

    # Return window duration stated (verify from answer content; optionally page also mentions it)
    return_days_leaf = evaluator.add_leaf(
        id="Return_Window_Days",
        desc="State the standard return window duration in days for laptop purchases at Best Buy.",
        parent=return_node,
        critical=True
    )
    return_days_text = extraction.return_policy_days or ""
    claim_return = (
        f"The answer states that Best Buy's standard return window for laptops is {return_days_text} days."
    )
    await evaluator.verify(
        claim=claim_return,
        node=return_days_leaf,
        sources=None,
        additional_instruction=(
            "Verify within the provided answer text whether a numeric return-window duration (in days) is stated for laptops."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate the agent's answer for the Best Buy laptop selection task.
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
    extraction = await evaluator.extract(
        prompt=prompt_extract_laptop_selection(),
        template_class=LaptopSelectionExtraction,
        extraction_name="laptop_selection_extraction"
    )

    # Add ground truth constraints info (for transparency)
    evaluator.add_ground_truth({
        "constraints": {
            "max_weight_lbs": 3.5,
            "required_display_size_inch": 14,
            "min_ram_gb": 16,
            "min_ssd_gb": 512,
            "min_resolution": "1920×1080 (FHD)"
        }
    }, gt_type="selection_constraints")

    # Build and execute verification tree
    await build_verification_tree(evaluator, root, extraction)

    # Return full summary
    return evaluator.get_summary()