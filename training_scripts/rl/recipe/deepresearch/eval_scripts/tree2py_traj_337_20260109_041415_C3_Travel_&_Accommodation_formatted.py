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
TASK_ID = "travel_planning_schengen_united_marriott"
TASK_DESCRIPTION = """
You are planning an international trip to Europe for business and need to select a destination that meets specific travel requirements. Your task is to identify a European country that satisfies both of the following criteria:

1. The country must be a member of the Schengen Area
2. The country must be subject to United Airlines' checked baggage weight restriction of 70 pounds (32 kilograms)

Once you have identified a qualifying country, provide the following information:

A. State the passport validity requirement for Schengen Area countries (how many months your U.S. passport must be valid beyond your intended departure date from the Schengen Area)

B. Confirm the specific checked baggage weight limit in pounds (or kilograms) that United Airlines enforces for flights to/from your selected country

C. Identify the minimum Marriott Bonvoy elite status tier that provides the "priority late checkout" benefit to members

D. State the minimum number of qualifying nights required annually to achieve the elite status tier you identified in part C

E. Provide a valid URL reference that documents the qualification requirements (specifically the qualifying nights threshold) for the elite status tier you identified

Your answer should demonstrate accurate understanding of international travel requirements, airline baggage policies, and hotel loyalty program structures.
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class TravelInfoExtraction(BaseModel):
    # Selected destination
    selected_country: Optional[str] = None

    # Evidence URLs for Schengen membership (from the answer)
    schengen_sources: List[str] = Field(default_factory=list)

    # Evidence URLs for United baggage policy / 70-lb restriction (from the answer)
    ua_baggage_sources: List[str] = Field(default_factory=list)

    # Passport validity requirement as stated in the answer (e.g., "3 months", "three months")
    passport_validity_months: Optional[str] = None

    # Baggage weight text as stated (e.g., "70 pounds (32 kg)")
    baggage_weight_limit: Optional[str] = None

    # Marriott status information
    marriott_min_status_tier: Optional[str] = None  # e.g., "Silver Elite", "Gold Elite"
    marriott_qualifying_nights: Optional[str] = None  # e.g., "10", "25", etc.

    # Documentation URLs provided in the answer
    marriott_doc_url: Optional[str] = None  # The URL that documents qualifying nights threshold (required by task E)
    marriott_benefits_urls: List[str] = Field(default_factory=list)  # Any benefits pages cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_travel_info() -> str:
    return """
    From the answer text, extract the following fields exactly as they are presented. If an item is missing, return null (for single values) or [] (for lists). Do not fabricate anything.

    Required fields:
    1) selected_country: The name of the single European country chosen to meet the criteria.

    2) schengen_sources: A list of all URLs cited in the answer that support or indicate that the selected country is a Schengen Area member. Include any general EU/Schengen list pages if cited. Return [] if none.

    3) ua_baggage_sources: A list of URLs cited that support United Airlines' 70-pound (32 kg) checked baggage restriction (e.g., United baggage policy pages). Return [] if none.

    4) passport_validity_months: The passport validity requirement for Schengen Area countries as stated in the answer (e.g., "3 months"). If spelled out (e.g., "three months"), return it as written.

    5) baggage_weight_limit: The specific checked baggage weight limit stated for United Airlines for flights to/from the selected country (e.g., "70 pounds (32 kilograms)"). Return it as written in the answer.

    6) marriott_min_status_tier: The minimum Marriott Bonvoy elite status tier that provides "priority late checkout" as stated in the answer. Examples: "Silver Elite", "Gold Elite", "Platinum Elite", "Titanium Elite", "Ambassador Elite".

    7) marriott_qualifying_nights: The minimum number of qualifying nights required annually to achieve the tier in (6), exactly as stated in the answer. Return digits if available (e.g., "10", "25", "50", "75", "100").

    8) marriott_doc_url: A single URL that the answer provides which documents the qualifying nights threshold for the tier in (6). If multiple URLs are provided, return the most directly relevant one for the nights threshold. Return null if none.

    9) marriott_benefits_urls: Any additional URLs that the answer cites which describe Marriott Bonvoy elite benefits by tier (these may help confirm "priority late checkout"). Return [] if none.

    Return a single JSON object with these fields.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _normalize_marriott_urls(extracted: TravelInfoExtraction) -> List[str]:
    urls: List[str] = []
    if extracted.marriott_doc_url:
        urls.append(extracted.marriott_doc_url)
    if extracted.marriott_benefits_urls:
        urls.extend([u for u in extracted.marriott_benefits_urls if isinstance(u, str) and u.strip()])
    # De-duplicate while preserving order
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique_urls.append(u)
    return unique_urls


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: TravelInfoExtraction) -> None:
    """
    Build the rubric tree and run verifications according to the provided rubric JSON.
    """

    # Root: Create a top-level sequential critical node for the whole task
    task_root = evaluator.add_sequential(
        id="Travel_Planning_Task",
        desc="Complete all required travel planning verifications for a European destination meeting specified criteria",
        parent=evaluator.root,
        critical=True
    )

    # ------------------------------------------------------------------- #
    # Destination_Selection (parallel, critical)                          #
    # ------------------------------------------------------------------- #
    dest_node = evaluator.add_parallel(
        id="Destination_Selection",
        desc="Identify a European country that is both a Schengen Area member and subject to United Airlines' 70-pound checked baggage weight restriction",
        parent=task_root,
        critical=True
    )

    # Existence check: A destination country should be specified (critical; gates other checks)
    evaluator.add_custom_node(
        result=bool(extracted.selected_country and extracted.selected_country.strip()),
        id="Destination_Specified",
        desc="A destination country is specified in the answer",
        parent=dest_node,
        critical=True
    )

    # Leaf: Schengen_Area_Membership
    schengen_leaf = evaluator.add_leaf(
        id="Schengen_Area_Membership",
        desc="The selected country must be a member of the Schengen Area",
        parent=dest_node,
        critical=True
    )
    schengen_claim = f"{(extracted.selected_country or '').strip()} is a member of the Schengen Area."
    await evaluator.verify(
        claim=schengen_claim,
        node=schengen_leaf,
        sources=extracted.schengen_sources,  # May be [], which routes to simple verify
        additional_instruction=(
            "Use the provided URL(s) to confirm that the selected country is in the Schengen Area. "
            "Accept official EU/Schengen or reputable governmental sources. "
            "If the URL(s) are missing, irrelevant, or do not support the claim, judge as not supported."
        )
    )

    # Leaf: United_Baggage_Restriction_Applicable
    ua_limit_leaf = evaluator.add_leaf(
        id="United_Baggage_Restriction_Applicable",
        desc="The selected country must be subject to United Airlines' checked baggage weight restriction of 70 pounds (32 kilograms)",
        parent=dest_node,
        critical=True
    )
    ua_claim = (
        f"United Airlines enforces a maximum checked baggage weight of 70 pounds (32 kilograms) per bag; "
        f"flights to/from {(extracted.selected_country or '').strip()} are subject to this 70-pound (32 kg) restriction."
    )
    await evaluator.verify(
        claim=ua_claim,
        node=ua_limit_leaf,
        sources=extracted.ua_baggage_sources,
        additional_instruction=(
            "Check the provided United Airlines baggage policy page(s) for explicit mention of a 70 lb (32 kg) maximum per checked bag, or language that bags over 70 lb (32 kg) are not accepted. "
            "General policy pages that apply network-wide, including Europe, are acceptable evidence. "
            "If the page(s) do not state the 70 lb (32 kg) restriction, judge as not supported."
        )
    )

    # ------------------------------------------------------------------- #
    # Travel_Information_Requirements (parallel, critical)                #
    # ------------------------------------------------------------------- #
    info_node = evaluator.add_parallel(
        id="Travel_Information_Requirements",
        desc="Provide all required travel information for the selected destination",
        parent=task_root,
        critical=True
    )

    # Leaf: Passport_Validity_Requirement
    passport_leaf = evaluator.add_leaf(
        id="Passport_Validity_Requirement",
        desc="State the passport validity requirement for Schengen Area countries (months U.S. passport must be valid beyond intended departure date from the Schengen Area)",
        parent=info_node,
        critical=True
    )

    # We verify correctness of the stated requirement; if the answer is correct, it should indicate at least 3 months beyond Schengen departure.
    # We allow the verifier to use general knowledge for this check.
    passport_claim = (
        "For Schengen Area countries, a U.S. passport must be valid at least 3 months beyond the intended date of departure from the Schengen Area."
    )
    await evaluator.verify(
        claim=passport_claim,
        node=passport_leaf,
        additional_instruction=(
            "For this specific check, you MAY use your general knowledge about Schengen passport validity rules. "
            "Judge as Correct only if the answer aligns with the canonical Schengen requirement (at least 3 months beyond departure). "
            "If the answer asserts a different number of months or does not clearly state the requirement, judge as Incorrect."
        )
    )

    # Leaf: Baggage_Weight_Limit_Confirmation
    bw_leaf = evaluator.add_leaf(
        id="Baggage_Weight_Limit_Confirmation",
        desc="Confirm the specific United Airlines checked baggage weight limit (in pounds or kilograms) for flights to/from the selected country",
        parent=info_node,
        critical=True
    )
    # We confirm the specific limit of 70 lb (32 kg) using the United sources supplied in the answer.
    bw_claim = (
        "United Airlines' maximum checked baggage weight per bag is 70 pounds (32 kilograms)."
    )
    await evaluator.verify(
        claim=bw_claim,
        node=bw_leaf,
        sources=extracted.ua_baggage_sources,
        additional_instruction=(
            "Use the provided United Airlines baggage policy page(s) to confirm the 70 lb (32 kg) maximum per checked bag. "
            "Synonyms like '70 lbs', '32 kg', 'bags over 70 lb not accepted' should be treated as equivalent. "
            "If the URL(s) do not support 70 lb (32 kg) explicitly, judge as not supported."
        )
    )

    # ------------------------------------------------------------------- #
    # Elite_Status_Requirements (parallel, critical)                      #
    # ------------------------------------------------------------------- #
    elite_node = evaluator.add_parallel(
        id="Elite_Status_Requirements",
        desc="Provide required information about the Marriott Bonvoy elite status tier that provides 'priority late checkout' and its qualifying-night requirements, including a supporting URL",
        parent=info_node,
        critical=True
    )

    # Optional but helpful gating: Ensure a documentation URL is provided (critical existence check).
    evaluator.add_custom_node(
        result=bool(extracted.marriott_doc_url and extracted.marriott_doc_url.strip()),
        id="Marriott_Documentation_URL_Provided",
        desc="A documentation URL for the qualifying nights threshold is provided",
        parent=elite_node,
        critical=True
    )

    marriott_urls = _normalize_marriott_urls(extracted)

    # Leaf: Minimum_Status_Tier_Identification
    tier_leaf = evaluator.add_leaf(
        id="Minimum_Status_Tier_Identification",
        desc="Identify the minimum Marriott Bonvoy elite status tier that provides the 'priority late checkout' benefit",
        parent=elite_node,
        critical=True
    )
    stated_tier = (extracted.marriott_min_status_tier or "").strip()
    tier_claim = (
        f"The minimum Marriott Bonvoy elite status tier that includes a benefit described as 'Priority Late Checkout' (or equivalent phrasing) is '{stated_tier}'."
    )
    await evaluator.verify(
        claim=tier_claim,
        node=tier_leaf,
        sources=marriott_urls,  # Try documentation page and any benefits page(s) provided
        additional_instruction=(
            "Use the provided Marriott page(s). Treat 'Priority Late Checkout' as equivalent to phrases like 'Late checkout (subject to availability)' or similar wording. "
            "Confirm that the stated tier indeed includes that benefit, and that no strictly lower tier is shown to include it. "
            "If the page(s) do not show benefits by tier or do not support the claim, judge as not supported."
        )
    )

    # Leaf: Annual_Qualifying_Nights_Threshold
    nights_leaf = evaluator.add_leaf(
        id="Annual_Qualifying_Nights_Threshold",
        desc="State the minimum number of qualifying nights required annually to achieve the identified elite status tier",
        parent=elite_node,
        critical=True
    )
    stated_nights = (extracted.marriott_qualifying_nights or "").strip()
    nights_claim = (
        f"To achieve the Marriott Bonvoy tier '{stated_tier}', a member must complete at least {stated_nights} qualifying nights per calendar year."
    )
    await evaluator.verify(
        claim=nights_claim,
        node=nights_leaf,
        sources=extracted.marriott_doc_url,  # This should be the authoritative threshold documentation URL
        additional_instruction=(
            "Focus on the qualifying/elite nights requirement. Accept synonyms like 'Elite Night Credits', 'nights required', or a table that clearly specifies nights per tier. "
            "Verify that the stated nights number matches the documentation for the same tier."
        )
    )

    # Leaf: Documentation_URL
    doc_leaf = evaluator.add_leaf(
        id="Documentation_URL",
        desc="Provide a valid URL reference that documents the qualifying nights threshold for the identified elite status tier",
        parent=elite_node,
        critical=True
    )
    doc_claim = (
        f"This page documents the minimum qualifying nights requirement for the Marriott Bonvoy tier '{stated_tier}'."
    )
    await evaluator.verify(
        claim=doc_claim,
        node=doc_leaf,
        sources=extracted.marriott_doc_url,
        additional_instruction=(
            "Judge as supported only if the page explicitly includes the qualifying nights requirement (or a table/list that unambiguously specifies nights per tier), "
            "and it covers the stated tier."
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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate a single answer for the Travel Planning task and return a structured result dictionary.
    """
    # Initialize evaluator with a sequential root strategy (we'll attach our critical task node under it)
    evaluator = Evaluator()
    evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.SEQUENTIAL,
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

    # Extract structured information from the answer
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_travel_info(),
        template_class=TravelInfoExtraction,
        extraction_name="travel_info_extraction"
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, extracted_info)

    # Return the aggregated evaluation summary
    return evaluator.get_summary()