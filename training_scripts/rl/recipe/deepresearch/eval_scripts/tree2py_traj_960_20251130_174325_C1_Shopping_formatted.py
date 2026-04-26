import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "thanksgiving_blackfriday_retailer_2024"
TASK_DESCRIPTION = (
    "Which major national sporting goods retailer was closed on Thanksgiving Day 2024 "
    "(Thursday, November 28, 2024) and opened at 6:00 AM local time on Black Friday 2024 "
    "(Friday, November 29, 2024)?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class RetailerExtraction(BaseModel):
    """
    Structured information extracted from the agent's answer.
    """
    retailer_name: Optional[str] = None
    thanksgiving_2024_closure_mention: Optional[str] = None
    black_friday_2024_opening_time_mention: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_retailer_info() -> str:
    return (
        "From the provided answer, extract the following fields about the single identified retailer:\n"
        "1) retailer_name: The exact name of the single retailer identified in the answer. "
        "   If multiple retailers are mentioned, choose the one the answer clearly identifies as the match. "
        "   If no single specific retailer name is given, return null.\n"
        "2) thanksgiving_2024_closure_mention: If the answer explicitly mentions the retailer being closed on "
        "   Thanksgiving Day 2024 (Thursday, November 28, 2024), extract the sentence or phrase. Otherwise, null.\n"
        "3) black_friday_2024_opening_time_mention: If the answer explicitly mentions the retailer opening at "
        "   6:00 AM local time on Black Friday 2024 (Friday, November 29, 2024), extract the sentence or phrase. "
        "   Otherwise, null.\n"
        "4) source_urls: Extract all URLs cited in the answer as evidence. Include URLs presented in plain form "
        "   or markdown links. Return a list (can be empty if none).\n"
        "Only extract information explicitly present in the answer; do not invent or infer."
    )


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _safe_retailer_name(name: Optional[str]) -> str:
    return name.strip() if isinstance(name, str) and name.strip() else "the retailer named in the answer"


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_retailer_tree(
    evaluator: Evaluator,
    root_node,
    extracted: RetailerExtraction,
) -> None:
    """
    Build the verification tree under the critical Retailer_Identification node
    and run verifications according to the rubric.
    """
    # Create the critical parent node for retailer identification
    retailer_node = evaluator.add_parallel(
        id="Retailer_Identification",
        desc="Identifies the retailer that matches the Thanksgiving 2024 closure and Black Friday 2024 opening-time constraints",
        parent=root_node,
        critical=True,
    )

    # Prepare reusable data
    rname = _safe_retailer_name(extracted.retailer_name)
    sources_list = extracted.source_urls if extracted.source_urls else []

    # 1) Retailer_Name_Provided (Critical) - existence check
    name_provided = bool(extracted.retailer_name and extracted.retailer_name.strip())
    evaluator.add_custom_node(
        result=name_provided,
        id="Retailer_Name_Provided",
        desc="The response names a specific retailer (not a category or multiple retailers).",
        parent=retailer_node,
        critical=True,
    )

    # 2) Major_National_US_Retailer (Critical) - verify claim (use sources if available)
    major_nat_leaf = evaluator.add_leaf(
        id="Major_National_US_Retailer",
        desc="The identified retailer is a major national retailer in the United States.",
        parent=retailer_node,
        critical=True,
    )
    major_nat_claim = f"The retailer {rname} is a major national retailer in the United States."
    await evaluator.verify(
        claim=major_nat_claim,
        node=major_nat_leaf,
        sources=sources_list,
        additional_instruction=(
            "Verify that the retailer operates nationally across the United States with a substantial presence "
            "(e.g., many stores across multiple states, widely recognized national brand). "
            "Use authoritative sources such as the retailer's official 'About' page, Wikipedia, or credible news/press releases."
        ),
    )

    # 3) Primarily_Sporting_Goods_Retailer (Critical) - verify claim (use sources if available)
    sporting_leaf = evaluator.add_leaf(
        id="Primarily_Sporting_Goods_Retailer",
        desc="The identified retailer is primarily a sporting goods retailer.",
        parent=retailer_node,
        critical=True,
    )
    sporting_claim = f"The retailer {rname} is primarily a sporting goods retailer."
    await evaluator.verify(
        claim=sporting_claim,
        node=sporting_leaf,
        sources=sources_list,
        additional_instruction=(
            "Verify the retailer's primary business/category is sporting goods (e.g., sports equipment, athletic gear, "
            "outdoor sporting goods). Apparel-only brands or general big-box retailers should not count unless the "
            "primary category is sporting goods."
        ),
    )

    # 4) Thanksgiving_Closure (Critical) - verify closure on Thanksgiving Day 2024
    tg_leaf = evaluator.add_leaf(
        id="Thanksgiving_Closure",
        desc="The retailer was closed on Thanksgiving Day (Thursday, November 28, 2024).",
        parent=retailer_node,
        critical=True,
    )
    tg_claim = (
        f"The retailer {rname} was closed on Thanksgiving Day 2024 (Thursday, November 28, 2024)."
    )
    await evaluator.verify(
        claim=tg_claim,
        node=tg_leaf,
        sources=sources_list,
        additional_instruction=(
            "Confirm via store hours pages, official announcements, or reputable news that the retailer's stores "
            "were closed on Thanksgiving Day 2024 (Nov 28, 2024). Accept clear statements like 'Closed on Thanksgiving Day'."
        ),
    )

    # 5) Black_Friday_Opening_Time (Critical) - verify 6:00 AM local time opening on Black Friday 2024
    bf_leaf = evaluator.add_leaf(
        id="Black_Friday_Opening_Time",
        desc="The retailer opened at 6:00 AM local time on Black Friday (Friday, November 29, 2024).",
        parent=retailer_node,
        critical=True,
    )
    bf_claim = (
        f"The retailer {rname} opened at 6:00 AM local time on Black Friday 2024 (Friday, November 29, 2024)."
    )
    await evaluator.verify(
        claim=bf_claim,
        node=bf_leaf,
        sources=sources_list,
        additional_instruction=(
            "Check Black Friday 2024 store hours, official announcements, or credible news to verify opening at "
            "6:00 AM local time on Friday, Nov 29, 2024. Minor format variations such as '6 AM' are acceptable."
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
    Evaluate the agent's answer for the Thanksgiving/Black Friday 2024 retailer identification task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Only one main branch; parallel is fine
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

    # Extract structured retailer info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_retailer_info(),
        template_class=RetailerExtraction,
        extraction_name="retailer_extraction",
    )

    # Optionally record some custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "retailer_name": extracted.retailer_name,
            "source_count": len(extracted.source_urls),
            "sample_sources": extracted.source_urls[:3],
            "thanksgiving_mention": extracted.thanksgiving_2024_closure_mention,
            "black_friday_opening_mention": extracted.black_friday_2024_opening_time_mention,
        },
        info_type="extraction_overview",
        info_name="extraction_overview",
    )

    # Build tree and run verifications
    await build_and_verify_retailer_tree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()