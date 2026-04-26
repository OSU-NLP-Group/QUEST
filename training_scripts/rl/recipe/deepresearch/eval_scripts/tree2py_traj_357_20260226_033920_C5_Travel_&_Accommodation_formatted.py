import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "senior_pass_priority_pass_2026"
TASK_DESCRIPTION = (
    "Maria, a 65-year-old U.S. citizen, is planning a 3-week road trip in summer 2026 to visit multiple national parks "
    "and federal recreation sites, including several national forest campgrounds. She will be departing from Atlanta's "
    "Hartsfield-Jackson International Airport and returning there at the end of her trip. Maria has a Capital One Venture X "
    "credit card that provides Priority Pass membership, and she wants to use an airport lounge before her departure flight.\n\n"
    "Based on the current 2026 information:\n\n"
    "1. What type of America the Beautiful Senior Pass should Maria purchase (Annual or Lifetime), and what is the cost?\n"
    "2. Provide a cost-benefit justification for your recommendation, specifically mentioning the camping discount benefit.\n"
    "3. What are the two main eligibility requirements Maria must meet to purchase a Senior Pass, and does she meet them?\n"
    "4. For her Priority Pass lounge access at Atlanta airport, what is the key requirement she must have on the day of her departure?\n"
    "5. Provide reference URLs for: (a) Senior Pass eligibility and pricing information, and (b) Priority Pass lounge access requirements."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PlanExtraction(BaseModel):
    # Senior Pass recommendation and pricing
    recommended_pass_type: Optional[str] = None  # e.g., "Annual", "Lifetime"
    pass_price: Optional[str] = None             # e.g., "$20", "20 USD", "$80"
    justification: Optional[str] = None
    mentions_camping_discount: Optional[bool] = None

    # Eligibility info
    age: Optional[str] = None                    # e.g., "65"
    citizenship: Optional[str] = None            # e.g., "U.S. citizen", "permanent resident"

    # Sources
    senior_pass_urls: List[str] = Field(default_factory=list)  # URLs in the answer for Senior Pass eligibility/pricing
    lounge_urls: List[str] = Field(default_factory=list)       # URLs in the answer for Priority Pass requirements/lounges

    # Lounge info mentioned in the answer
    lounge_location: Optional[str] = None                     # e.g., "The Club at ATL (Concourse F)"
    mentions_boarding_pass_requirement: Optional[bool] = None
    guest_fee_text: Optional[str] = None                      # e.g., "$35 per guest after Feb 1, 2026"


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_plan_info() -> str:
    return """
    Extract the following structured information from the answer text.

    Senior Pass recommendation:
    - recommended_pass_type: Which Senior Pass type is recommended (Annual or Lifetime)? Return exactly 'Annual' or 'Lifetime' if mentioned; otherwise null.
    - pass_price: The price stated for the recommended pass type (e.g., '$20' or '$80'). If not stated, return null.
    - justification: The cost-benefit justification text for the recommendation, if provided; otherwise null.
    - mentions_camping_discount: Return true if the answer explicitly mentions a camping discount benefit for the Senior Pass (e.g., '50% off camping'), otherwise false.

    Eligibility:
    - age: The traveler's age mentioned in the answer (e.g., '65'). If not mentioned, return null.
    - citizenship: The traveler's citizenship or residency status mentioned (e.g., 'U.S. citizen' or 'permanent resident'). If not mentioned, return null.

    Sources:
    - senior_pass_urls: A list of all URLs in the answer that reference Senior Pass eligibility and/or pricing information. Only include actual URLs present in the answer.
    - lounge_urls: A list of all URLs in the answer that reference Priority Pass lounge access requirements and/or specific lounge pages at Atlanta airport. Only include actual URLs present in the answer.

    Lounge info:
    - lounge_location: If the answer mentions the Priority Pass lounge location at Atlanta (e.g., 'The Club at ATL in Concourse F'), extract it; otherwise null.
    - mentions_boarding_pass_requirement: Return true if the answer explicitly states that a same-day boarding pass is required for Priority Pass lounge access; otherwise false.
    - guest_fee_text: If the answer mentions a guest fee (e.g., '$35 per guest after February 1, 2026'), extract that text; otherwise null.

    Return a single JSON object with exactly these fields.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_pass_type(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().lower()
    if "annual" in s:
        return "Annual"
    if "lifetime" in s:
        return "Lifetime"
    return None


def extract_price_value(raw: Optional[str]) -> Optional[str]:
    """
    Extract a numeric price value ('20' or '80') from a string like '$20', 'USD 80', etc.
    Returns None if not found.
    """
    if not raw:
        return None
    nums = re.findall(r"\d+", raw)
    if not nums:
        return None
    # Prefer two-digit recognizable amounts
    for n in nums:
        if n in {"20", "80"}:
            return n
    # Fallback to the first number
    return nums[0]


def fmt_price(price_value: Optional[str]) -> Optional[str]:
    if not price_value:
        return None
    # Normalize to $XX format
    return f"${price_value}"


# --------------------------------------------------------------------------- #
# Verification builder functions                                              #
# --------------------------------------------------------------------------- #
async def build_supporting_docs_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: PlanExtraction
) -> Dict[str, Any]:
    """
    Build SupportingDocumentation section first to serve as prerequisites for other URL-verification leaves.
    Returns a dict with references to the created leaf nodes for convenience.
    """
    docs_node = evaluator.add_parallel(
        id="SupportingDocumentation",
        desc="Provide proper source references for key information",
        parent=parent_node,
        critical=True  # Both children are critical; acceptable since all are critical leaves below
    )

    senior_source_exists = bool(extracted.senior_pass_urls)
    lounge_source_exists = bool(extracted.lounge_urls)

    senior_ref_node = evaluator.add_custom_node(
        result=senior_source_exists,
        id="SeniorPassSourceReference",
        desc="Provide a valid URL reference for Senior Pass eligibility and pricing information",
        parent=docs_node,
        critical=True
    )

    lounge_ref_node = evaluator.add_custom_node(
        result=lounge_source_exists,
        id="LoungeAccessSourceReference",
        desc="Provide a valid URL reference for Priority Pass lounge access requirements",
        parent=docs_node,
        critical=True
    )

    return {
        "docs_node": docs_node,
        "senior_ref_leaf": senior_ref_node,
        "lounge_ref_leaf": lounge_ref_node
    }


async def build_pass_eligibility_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: PlanExtraction
) -> None:
    """
    Build PassEligibilityVerification with two critical leaves: AgeRequirement and CitizenshipRequirement.
    """
    elig_node = evaluator.add_parallel(
        id="PassEligibilityVerification",
        desc="Verify that the traveler meets all eligibility requirements for the America the Beautiful Senior Pass",
        parent=parent_node,
        critical=True  # Both children critical; acceptable
    )

    # AgeRequirement
    age_leaf = evaluator.add_leaf(
        id="AgeRequirement",
        desc="Confirm the traveler is 62 years of age or older",
        parent=elig_node,
        critical=True
    )
    age_claim = "The traveler meets the Senior Pass age requirement (62 years or older)."
    await evaluator.verify(
        claim=age_claim,
        node=age_leaf,
        additional_instruction=(
            f"Use the age mentioned in the answer if present. Extracted age: {extracted.age or 'unknown'}."
            " If age is at least 62, the requirement is met."
        )
    )

    # CitizenshipRequirement
    citizenship_leaf = evaluator.add_leaf(
        id="CitizenshipRequirement",
        desc="Confirm the traveler is a U.S. citizen or permanent resident",
        parent=elig_node,
        critical=True
    )
    citizenship_claim = "The traveler meets the Senior Pass citizenship/residency requirement (U.S. citizen or U.S. permanent resident)."
    await evaluator.verify(
        claim=citizenship_claim,
        node=citizenship_leaf,
        additional_instruction=(
            f"Use the citizenship/residency mentioned in the answer if present. Extracted: {extracted.citizenship or 'unknown'}."
            " If she is a U.S. citizen or U.S. permanent resident, the requirement is met."
        )
    )


async def build_pass_type_selection_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: PlanExtraction,
    senior_ref_leaf
) -> None:
    """
    Build PassTypeSelection with three critical children: identification, cost justification, price accuracy.
    """
    pts_node = evaluator.add_parallel(
        id="PassTypeSelection",
        desc="Evaluate whether the appropriate Senior Pass type (Annual $20 or Lifetime $80) is selected based on the traveler's usage plans",
        parent=parent_node,
        critical=True  # All children critical
    )

    # PassTypeIdentification: ensure recommended type is recognized
    normalized_type = normalize_pass_type(extracted.recommended_pass_type)
    type_identified = normalized_type in {"Annual", "Lifetime"}
    evaluator.add_custom_node(
        result=bool(type_identified),
        id="PassTypeIdentification",
        desc="Identify which pass type (Senior Annual Pass or Senior Lifetime Pass) is recommended",
        parent=pts_node,
        critical=True
    )

    # CostJustification: presence and mentions camping discount
    cj_leaf = evaluator.add_leaf(
        id="CostJustification",
        desc="Provide clear cost-benefit justification for the selected pass type, specifically mentioning the camping discount benefit",
        parent=pts_node,
        critical=True
    )
    cj_claim = (
        "The answer includes a clear cost-benefit justification for the selected pass type"
        " and explicitly mentions the camping discount benefit (e.g., 50% off camping fees)."
    )
    await evaluator.verify(
        claim=cj_claim,
        node=cj_leaf,
        additional_instruction=(
            f"Look for justification text and explicit mention of camping discount. Extracted justification: "
            f"{(extracted.justification or 'none')}. Extracted 'mentions_camping_discount': "
            f"{extracted.mentions_camping_discount if extracted.mentions_camping_discount is not None else 'unknown'}."
            " The justification should connect benefits (like discounted camping) to the travel plan."
        )
    )

    # PriceAccuracy: verify price against official sources
    price_leaf = evaluator.add_leaf(
        id="PriceAccuracy",
        desc="State the correct price for the selected pass type ($20 for Annual or $80 for Lifetime)",
        parent=pts_node,
        critical=True
    )
    price_value = extract_price_value(extracted.pass_price)
    price_str = fmt_price(price_value) if price_value else None
    if normalized_type and price_str:
        price_claim = f"The price for the Senior {normalized_type} Pass is {price_str}."
    else:
        # Fallback claim to still allow verification attempt (likely to fail if missing)
        price_claim = (
            "The answer correctly states the official price for the recommended Senior Pass type (Annual $20 or Lifetime $80)."
        )
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=extracted.senior_pass_urls,  # requires sources; will be skipped if prerequisite fails
        additional_instruction=(
            "Verify the price against the official Senior Pass information page(s)."
            " Accept minor formatting differences, but the numeric amount must match."
        ),
        extra_prerequisites=[senior_ref_leaf]
    )


async def build_pass_benefits_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: PlanExtraction,
    senior_ref_leaf
) -> None:
    """
    Build PassBenefitsIdentification with one critical child (CampingDiscountRate) and two non-critical children.
    """
    benefits_node = evaluator.add_parallel(
        id="PassBenefitsIdentification",
        desc="Accurately identify the benefits provided by the Senior Pass",
        parent=parent_node,
        critical=False  # Mixed criticality among children
    )

    # CampingDiscountRate (critical)
    camp_leaf = evaluator.add_leaf(
        id="CampingDiscountRate",
        desc="State that the pass provides 50% discount on camping fees at applicable sites",
        parent=benefits_node,
        critical=True
    )
    camp_claim = "The Senior Pass provides a 50% discount on camping fees at applicable sites."
    await evaluator.verify(
        claim=camp_claim,
        node=camp_leaf,
        sources=extracted.senior_pass_urls,
        additional_instruction=(
            "Confirm that the 50% discount applies to camping fees at eligible federal recreation sites/campgrounds."
            " This is commonly stated on the official pass benefits page."
        ),
        extra_prerequisites=[senior_ref_leaf]
    )

    # EntranceFeesCoverage (non-critical)
    entrance_leaf = evaluator.add_leaf(
        id="EntranceFeesCoverage",
        desc="Confirm that the pass covers entrance fees at federal recreation sites",
        parent=benefits_node,
        critical=False
    )
    entrance_claim = "The Senior Pass covers entrance fees at federal recreation sites."
    await evaluator.verify(
        claim=entrance_claim,
        node=entrance_leaf,
        sources=extracted.senior_pass_urls,
        additional_instruction="Verify that the pass admits the holder to federal recreation sites that charge entrance fees.",
        extra_prerequisites=[senior_ref_leaf]
    )

    # FederalAgenciesCoverage (non-critical)
    agencies_leaf = evaluator.add_leaf(
        id="FederalAgenciesCoverage",
        desc="Acknowledge that the pass covers sites managed by six federal agencies",
        parent=benefits_node,
        critical=False
    )
    agencies_claim = "The Senior Pass covers sites managed by six federal agencies."
    await evaluator.verify(
        claim=agencies_claim,
        node=agencies_leaf,
        sources=extracted.senior_pass_urls,
        additional_instruction=(
            "Verify that the Senior Pass applies to sites managed by six agencies"
            " (e.g., NPS, USFS, USFWS, BLM, BOR, and USACE). Minor wording variations are acceptable."
        ),
        extra_prerequisites=[senior_ref_leaf]
    )


async def build_lounge_access_nodes(
    evaluator: Evaluator,
    parent_node,
    extracted: PlanExtraction,
    lounge_ref_leaf
) -> None:
    """
    Build LoungeAccessPlanning with one critical child (BoardingPassRequirement) and two non-critical children.
    """
    lounge_node = evaluator.add_parallel(
        id="LoungeAccessPlanning",
        desc="Evaluate the lounge access plan for the departure from Atlanta airport",
        parent=parent_node,
        critical=False  # Mixed criticality among children
    )

    # BoardingPassRequirement (critical)
    bpass_leaf = evaluator.add_leaf(
        id="BoardingPassRequirement",
        desc="Identify that a same-day boarding pass is required for lounge access",
        parent=lounge_node,
        critical=True
    )
    bpass_claim = "Priority Pass lounge access requires a same-day boarding pass."
    await evaluator.verify(
        claim=bpass_claim,
        node=bpass_leaf,
        sources=extracted.lounge_urls,
        additional_instruction="Verify the boarding pass requirement in Priority Pass terms or the specific lounge access rules.",
        extra_prerequisites=[lounge_ref_leaf]
    )

    # LoungeLocationIdentification (non-critical)
    lounge_loc_leaf = evaluator.add_leaf(
        id="LoungeLocationIdentification",
        desc="Identify the correct Priority Pass lounge location at Atlanta airport (The Club at ATL in Concourse F)",
        parent=lounge_node,
        critical=False
    )
    lounge_loc_claim = "At Atlanta airport, the Priority Pass lounge is The Club at ATL located in Concourse F."
    await evaluator.verify(
        claim=lounge_loc_claim,
        node=lounge_loc_leaf,
        sources=extracted.lounge_urls,
        additional_instruction=(
            "Confirm location details on Priority Pass or the lounge's official page; allow minor wording variations."
        ),
        extra_prerequisites=[lounge_ref_leaf]
    )

    # GuestPolicyAwareness (non-critical)
    guest_leaf = evaluator.add_leaf(
        id="GuestPolicyAwareness",
        desc="Acknowledge that guests incur a fee ($35 per guest for Capital One Venture X Priority Pass after February 1, 2026)",
        parent=lounge_node,
        critical=False
    )
    guest_claim = "Guests incur a fee of $35 per guest for Capital One Venture X Priority Pass after February 1, 2026."
    await evaluator.verify(
        claim=guest_claim,
        node=guest_leaf,
        sources=extracted.lounge_urls,
        additional_instruction=(
            "Verify guest pricing policy specific to Capital One Venture X Priority Pass benefit starting February 1, 2026."
            " Accept small wording differences but the amount and effective date should match."
        ),
        extra_prerequisites=[lounge_ref_leaf]
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
# --------------------------------------------------------------------------- #
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the Senior Pass recommendation and Priority Pass lounge access requirements task (2026).
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
        default_model=model
    )

    # Extract structured plan info
    extracted: PlanExtraction = await evaluator.extract(
        prompt=prompt_extract_plan_info(),
        template_class=PlanExtraction,
        extraction_name="plan_extraction"
    )

    # Record a short custom info summary
    evaluator.add_custom_info(
        info={
            "recommended_pass_type": extracted.recommended_pass_type,
            "pass_price": extracted.pass_price,
            "age": extracted.age,
            "citizenship": extracted.citizenship,
            "senior_pass_urls_count": len(extracted.senior_pass_urls),
            "lounge_urls_count": len(extracted.lounge_urls),
            "lounge_location": extracted.lounge_location,
            "mentions_boarding_pass_requirement": extracted.mentions_boarding_pass_requirement,
            "mentions_camping_discount": extracted.mentions_camping_discount,
            "guest_fee_text": extracted.guest_fee_text
        },
        info_type="extracted_overview"
    )

    # Build the top-level evaluation node representing the rubric root
    travel_node = evaluator.add_parallel(
        id="TravelPlanningCompliance",
        desc="Evaluate whether the complete travel plan meets all requirements for pass eligibility, appropriate pass selection, lounge access logistics, cost calculations, and supporting documentation",
        parent=root,
        critical=False  # Set to non-critical to allow mixed critical children per framework rules
    )

    # 1) Supporting Documentation first (to serve as prerequisites for URL verifications)
    docs_refs = await build_supporting_docs_nodes(evaluator, travel_node, extracted)
    senior_ref_leaf = docs_refs["senior_ref_leaf"]
    lounge_ref_leaf = docs_refs["lounge_ref_leaf"]

    # 2) Pass Eligibility Verification
    await build_pass_eligibility_nodes(evaluator, travel_node, extracted)

    # 3) Pass Type Selection
    await build_pass_type_selection_nodes(evaluator, travel_node, extracted, senior_ref_leaf)

    # 4) Pass Benefits Identification
    await build_pass_benefits_nodes(evaluator, travel_node, extracted, senior_ref_leaf)

    # 5) Lounge Access Planning
    await build_lounge_access_nodes(evaluator, travel_node, extracted, lounge_ref_leaf)

    # Add ground truth expectations for reference (non-binding)
    evaluator.add_ground_truth({
        "senior_pass_prices": {"Annual": "$20", "Lifetime": "$80"},
        "camping_discount": "50% off applicable camping fees",
        "federal_agencies_count": 6,
        "atl_priority_pass_lounge": "The Club at ATL (Concourse F)",
        "priority_pass_boarding_pass_requirement": "same-day boarding pass",
        "venture_x_guest_fee_after_2026_02_01": "$35 per guest"
    }, gt_type="expected_facts_2026")

    # Return structured result
    return evaluator.get_summary()