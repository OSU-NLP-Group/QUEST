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
TASK_ID = "genesis_wolf_ranch_plan_selection"
TASK_DESCRIPTION = """
I am interested in purchasing a 3D-printed home at the Genesis Collection in Wolf Ranch, Georgetown, Texas. I need a floor plan with exactly 4 bedrooms, at least 2,000 square feet of living space, and a maximum budget of $565,000. Please identify a floor plan that meets these requirements and provide the floor plan name, square footage, number of bedrooms and bathrooms, and price. Include a reference URL from the developer's official website that confirms these specifications.
"""

BUDGET_LIMIT = 565000


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FloorPlanSelection(BaseModel):
    """
    Information for a single floor plan selection.
    Keep fields as strings to maximize compatibility with varied answer formats.
    """
    floor_plan_name: Optional[str] = None
    bedrooms: Optional[str] = None
    bathrooms: Optional[str] = None
    square_footage: Optional[str] = None
    price: Optional[str] = None
    reference_url: Optional[str] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_floor_plan() -> str:
    return """
    Extract one specific floor plan recommended in the answer for the Genesis Collection at Wolf Ranch (Georgetown, Texas).
    If multiple plans are mentioned, pick the first one that appears to match the user's constraints (4 bedrooms, >= 2,000 sq ft, price <= $565,000). If constraints are not clearly checked, still extract the very first explicit floor plan presented.

    Return a JSON object with the following fields (use strings where appropriate):
    - floor_plan_name: The name of the selected floor plan.
    - bedrooms: The number of bedrooms as written (e.g., "4", "4 bedrooms").
    - bathrooms: The number of bathrooms as written (e.g., "3", "3 bathrooms" or "3.5").
    - square_footage: The living area as written (e.g., "2,069 sq ft", "2,100 square feet").
    - price: The price as written (e.g., "$559,990", "from $560,000", "$560k").
    - reference_url: A single URL to a page on the official developer/builder or community website that supports these details (for example Lennar.com, ICON website, or the official Wolf Ranch community website). If multiple URLs are provided in the answer, choose the one most clearly from the official builder/developer/community site; otherwise choose the first URL mentioned. If no URL is present, set this field to null.

    Important:
    - Extract exactly what is written in the answer; do not invent values.
    - Do not include third-party aggregator/listing/broker sites as the reference URL if an official site URL is available in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper for additional instruction snippets                                  #
# --------------------------------------------------------------------------- #
def instr_name_match() -> str:
    return (
        "Judge by the page content whether the floor plan name matches. Allow minor variations "
        "(case, punctuation, hyphenation, or inclusion of collection/community labels)."
    )


def instr_official_site() -> str:
    return (
        "Decide whether this URL is an official developer/builder/community site (e.g., Lennar.com, ICON website, "
        "or the official Wolf Ranch community site). Use branding, logos, footer/company info, and domain cues. "
        "Do NOT consider third-party listing or aggregator sites as official."
    )


def instr_number_match() -> str:
    return (
        "Verify the statement exactly against the webpage. Allow reasonable formatting differences "
        "(e.g., 'sq ft' vs 'square feet', commas in numbers, 'from $X' vs '$X')."
    )


def instr_requirement_text() -> str:
    return (
        "Focus on whether the page supports the requirement as stated. If the page clearly indicates the requirement "
        "is met, pass; otherwise, fail."
    )


# --------------------------------------------------------------------------- #
# Verification logic                                                          #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, root, data: FloorPlanSelection) -> None:
    """
    Build the verification tree as specified by the rubric (with fine-grained leaf nodes),
    and run the necessary verifications.
    """
    # Group 1: Reference URL (Critical)
    ref_group = evaluator.add_parallel(
        id="reference_url_group",
        desc="Provide a reference URL from the developer's official website that confirms the floor plan specifications",
        parent=root,
        critical=True
    )

    url_exists_node = evaluator.add_custom_node(
        result=bool(data.reference_url and str(data.reference_url).strip()),
        id="reference_url_provided",
        desc="A reference URL is provided",
        parent=ref_group,
        critical=True
    )

    url_official_node = evaluator.add_leaf(
        id="reference_url_official_site",
        desc="The provided URL is an official developer/builder/community website page (not a third-party listing)",
        parent=ref_group,
        critical=True
    )
    await evaluator.verify(
        claim="This page is part of the official developer/builder/community website for the Genesis Collection/Wolf Ranch (e.g., Lennar, ICON, or the official Wolf Ranch site). It is not a third-party listing or aggregator.",
        node=url_official_node,
        sources=data.reference_url,
        additional_instruction=instr_official_site(),
        extra_prerequisites=[url_exists_node]
    )

    # Group 2: Floor Plan Identification (Critical)
    id_group = evaluator.add_parallel(
        id="floor_plan_identification",
        desc="Identify the floor plan and confirm it is from the Genesis Collection at Wolf Ranch, Georgetown, Texas",
        parent=root,
        critical=True
    )

    # Floor plan name presence and accuracy
    name_present_node = evaluator.add_custom_node(
        result=bool(data.floor_plan_name and str(data.floor_plan_name).strip()),
        id="floor_plan_name_provided",
        desc="Provide the name of the floor plan (value present in the answer)",
        parent=id_group,
        critical=True
    )

    name_accuracy_node = evaluator.add_leaf(
        id="floor_plan_name_accuracy",
        desc="The listed floor plan name matches what the reference page shows",
        parent=id_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The floor plan name is '{data.floor_plan_name}'.",
        node=name_accuracy_node,
        sources=data.reference_url,
        additional_instruction=instr_name_match(),
        extra_prerequisites=[url_exists_node, name_present_node]
    )

    # Location verification against the page (Genesis Collection at Wolf Ranch, Georgetown, TX)
    location_node = evaluator.add_leaf(
        id="location_verification",
        desc="Confirm the floor plan is from the Genesis Collection at Wolf Ranch in Georgetown, Texas",
        parent=id_group,
        critical=True
    )
    await evaluator.verify(
        claim="This webpage is for a floor plan in the Genesis Collection at Wolf Ranch in Georgetown, Texas.",
        node=location_node,
        sources=data.reference_url,
        additional_instruction="Minor wording variations (e.g., 'Wolf Ranch - Genesis Collection' or 'Georgetown, TX') are acceptable.",
        extra_prerequisites=[url_exists_node]
    )

    # Group 3: Required Specifications (Critical)
    req_group = evaluator.add_parallel(
        id="required_specifications",
        desc="Provide all required specifications including those that are selection criteria",
        parent=root,
        critical=True
    )

    # Bedrooms (must be exactly 4)
    bedrooms_group = evaluator.add_parallel(
        id="bedroom_count_main",
        desc="Provide the number of bedrooms, which must be exactly 4",
        parent=req_group,
        critical=True
    )

    bedrooms_present_node = evaluator.add_custom_node(
        result=bool(data.bedrooms and str(data.bedrooms).strip()),
        id="bedroom_value_present",
        desc="Bedroom count value is provided in the answer",
        parent=bedrooms_group,
        critical=True
    )

    bedrooms_accuracy_node = evaluator.add_leaf(
        id="bedroom_value_accurate",
        desc="The provided bedroom count matches the reference page",
        parent=bedrooms_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The floor plan has {data.bedrooms} bedrooms.",
        node=bedrooms_accuracy_node,
        sources=data.reference_url,
        additional_instruction=instr_number_match(),
        extra_prerequisites=[url_exists_node, bedrooms_present_node]
    )

    bedrooms_requirement_node = evaluator.add_leaf(
        id="bedroom_requirement_four",
        desc="The floor plan has exactly 4 bedrooms",
        parent=bedrooms_group,
        critical=True
    )
    await evaluator.verify(
        claim="This floor plan has exactly 4 bedrooms.",
        node=bedrooms_requirement_node,
        sources=data.reference_url,
        additional_instruction=instr_requirement_text(),
        extra_prerequisites=[url_exists_node]
    )

    # Square Footage (must be at least 2,000)
    sqft_group = evaluator.add_parallel(
        id="square_footage_main",
        desc="Provide the square footage, which must be at least 2,000 square feet",
        parent=req_group,
        critical=True
    )

    sqft_present_node = evaluator.add_custom_node(
        result=bool(data.square_footage and str(data.square_footage).strip()),
        id="square_footage_value_present",
        desc="Square footage value is provided in the answer",
        parent=sqft_group,
        critical=True
    )

    sqft_accuracy_node = evaluator.add_leaf(
        id="square_footage_value_accurate",
        desc="The provided square footage matches the reference page",
        parent=sqft_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The floor plan has {data.square_footage} of living space.",
        node=sqft_accuracy_node,
        sources=data.reference_url,
        additional_instruction=instr_number_match(),
        extra_prerequisites=[url_exists_node, sqft_present_node]
    )

    sqft_requirement_node = evaluator.add_leaf(
        id="square_footage_requirement_2000_plus",
        desc="The floor plan has at least 2,000 square feet of living space",
        parent=sqft_group,
        critical=True
    )
    await evaluator.verify(
        claim="This floor plan has at least 2,000 square feet of living space.",
        node=sqft_requirement_node,
        sources=data.reference_url,
        additional_instruction=instr_requirement_text(),
        extra_prerequisites=[url_exists_node]
    )

    # Bathrooms (must be provided; no specific threshold in rubric)
    baths_group = evaluator.add_parallel(
        id="bathroom_count_main",
        desc="Provide the number of bathrooms",
        parent=req_group,
        critical=True
    )

    baths_present_node = evaluator.add_custom_node(
        result=bool(data.bathrooms and str(data.bathrooms).strip()),
        id="bathroom_value_present",
        desc="Bathroom count value is provided in the answer",
        parent=baths_group,
        critical=True
    )

    baths_accuracy_node = evaluator.add_leaf(
        id="bathroom_value_accurate",
        desc="The provided bathroom count matches the reference page",
        parent=baths_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The floor plan has {data.bathrooms} bathrooms.",
        node=baths_accuracy_node,
        sources=data.reference_url,
        additional_instruction=instr_number_match(),
        extra_prerequisites=[url_exists_node, baths_present_node]
    )

    # Price (must be provided and at most $565,000)
    price_group = evaluator.add_parallel(
        id="price_main",
        desc="Provide the price, which must be at most $565,000",
        parent=req_group,
        critical=True
    )

    price_present_node = evaluator.add_custom_node(
        result=bool(data.price and str(data.price).strip()),
        id="price_value_present",
        desc="Price value is provided in the answer",
        parent=price_group,
        critical=True
    )

    price_accuracy_node = evaluator.add_leaf(
        id="price_value_accurate",
        desc="The provided price matches the reference page",
        parent=price_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The price of the floor plan is {data.price}.",
        node=price_accuracy_node,
        sources=data.reference_url,
        additional_instruction=(
            "Consider equivalent expressions like 'from $X' or '$X starting price' as matching X. "
            "Allow minor formatting differences like commas, currency symbols, and 'k' notation. "
            "If the page only shows a range or 'pricing from', treat that as the price point shown."
        ),
        extra_prerequisites=[url_exists_node, price_present_node]
    )

    price_requirement_node = evaluator.add_leaf(
        id="price_requirement_within_budget",
        desc=f"The price is at most ${BUDGET_LIMIT:,}",
        parent=price_group,
        critical=True
    )
    await evaluator.verify(
        claim=f"The price shown on this page is less than or equal to ${BUDGET_LIMIT:,}.",
        node=price_requirement_node,
        sources=data.reference_url,
        additional_instruction=(
            "If the page shows 'from $X' or a price range, use the minimum stated number to evaluate. "
            "Pass only if the stated price (or starting price) is <= the threshold."
        ),
        extra_prerequisites=[url_exists_node]
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
    Evaluate an answer for the Genesis Collection floor plan selection task.
    """
    # Initialize evaluator with a parallel root (non-critical)
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_floor_plan(),
        template_class=FloorPlanSelection,
        extraction_name="selected_floor_plan"
    )

    # Add custom info snapshot
    evaluator.add_custom_info(
        {
            "extracted_floor_plan_name": extracted.floor_plan_name,
            "extracted_bedrooms": extracted.bedrooms,
            "extracted_bathrooms": extracted.bathrooms,
            "extracted_square_footage": extracted.square_footage,
            "extracted_price": extracted.price,
            "reference_url": extracted.reference_url
        },
        info_type="extraction_summary",
        info_name="extraction_overview"
    )

    # Build verification tree and run checks
    await build_and_verify_tree(evaluator, root, extracted)

    # Return final structured summary
    return evaluator.get_summary()