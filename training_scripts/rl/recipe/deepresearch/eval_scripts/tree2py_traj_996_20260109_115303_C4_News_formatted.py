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
TASK_ID = "walmart_vizio_acq_2024"
TASK_DESCRIPTION = (
    "Regarding the acquisition of VIZIO by Walmart that was completed in December 2024, provide the following information: "
    "(1) The exact date the acquisition was completed, "
    "(2) The date the acquisition was initially announced, "
    "(3) The purchase price per share (in cash), "
    "(4) The total equity value of the transaction, "
    "(5) The city and state where VIZIO's headquarters is located, "
    "(6) The name of the operating system that was a key part of the acquisition, "
    "(7) The name of the person who continues as VIZIO's CEO after the acquisition, "
    "(8) The name of the Walmart executive to whom VIZIO's CEO reports, and "
    "(9) The Walmart business segment under which VIZIO's operations are now reported."
)

# Ground truth reference values for verification
GROUND_TRUTH = {
    "completion_date": "December 3, 2024",
    "announcement_date": "February 20, 2024",
    "purchase_price_per_share": "$11.50 per share (cash)",
    "total_equity_value": "approximately $2.3 billion",
    "hq_combined": "Irvine, California",
    "operating_system_name": "SmartCast Operating System (also known as VIZIO OS)",
    "continuing_ceo_name": "William Wang",
    "walmart_exec_name": "Seth Dallaire",
    "reporting_segment": "Walmart U.S. segment",
}

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class FieldWithSources(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class HQInfo(BaseModel):
    city: Optional[str] = None
    state: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OversightInfo(BaseModel):
    executive_name: Optional[str] = None
    title: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AcquisitionDetailsExtraction(BaseModel):
    completion_date: Optional[FieldWithSources] = None
    announcement_date: Optional[FieldWithSources] = None
    purchase_price_per_share: Optional[FieldWithSources] = None
    total_equity_value: Optional[FieldWithSources] = None
    headquarters: Optional[HQInfo] = None
    operating_system_name: Optional[FieldWithSources] = None
    continuing_ceo_name: Optional[FieldWithSources] = None
    walmart_executive: Optional[OversightInfo] = None
    reporting_segment: Optional[FieldWithSources] = None
    all_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_acquisition_details() -> str:
    return """
    Extract the Walmart–VIZIO acquisition details exactly as presented in the answer.

    Return a JSON object matching this schema:
    {
      "completion_date": {"value": <string or null>, "sources": [<url> ...]},
      "announcement_date": {"value": <string or null>, "sources": [<url> ...]},
      "purchase_price_per_share": {"value": <string or null>, "sources": [<url> ...]},
      "total_equity_value": {"value": <string or null>, "sources": [<url> ...]},
      "headquarters": {"city": <string or null>, "state": <string or null>, "sources": [<url> ...]},
      "operating_system_name": {"value": <string or null>, "sources": [<url> ...]},
      "continuing_ceo_name": {"value": <string or null>, "sources": [<url> ...]},
      "walmart_executive": {"executive_name": <string or null>, "title": <string or null>, "sources": [<url> ...]},
      "reporting_segment": {"value": <string or null>, "sources": [<url> ...]},
      "all_sources": [<url> ...]
    }

    Field-specific guidance:
    1) completion_date: the exact closing/completion date of the acquisition.
    2) announcement_date: the date the acquisition was first announced.
    3) purchase_price_per_share: cash purchase price per share; preserve currency symbols if present (e.g., "$11.50 per share").
    4) total_equity_value: the total equity value of the transaction; preserve the approximate wording if present (e.g., "approximately $2.3 billion", "$2.3B").
    5) headquarters: city and state of VIZIO's headquarters (e.g., Irvine, California).
    6) operating_system_name: the OS name highlighted as a key part of the acquisition (e.g., "SmartCast Operating System", "VIZIO OS", "SmartCast OS"). Extract the name as written.
    7) continuing_ceo_name: the person who continues as VIZIO's CEO after the acquisition.
    8) walmart_executive: the Walmart executive to whom VIZIO (or VIZIO's CEO) reports after closing, and the title if provided.
    9) reporting_segment: the Walmart business segment under which VIZIO’s operations are reported post‑acquisition.

    IMPORTANT for sources:
    - Extract only URLs explicitly present in the answer (plain or markdown links).
    - For each field, add the specific URLs that support the statement to the field’s "sources" list.
    - Also include an "all_sources" array that contains every URL mentioned anywhere in the answer.
    - If a field is mentioned but has no specific URL next to it, leave its field-level "sources" array empty; do NOT invent URLs.
    - If the answer provides no URLs at all, set "all_sources" to [].

    If any field is not mentioned in the answer, set its value to null and its "sources" to [].
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _unique_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen and isinstance(x, str) and x.strip():
            seen.add(x)
            out.append(x)
    return out


def _combine_city_state(city: Optional[str], state: Optional[str]) -> Optional[str]:
    city = city.strip() if city else None
    state = state.strip() if state else None
    if city and state:
        return f"{city}, {state}"
    return city or state


# --------------------------------------------------------------------------- #
# Generic verification builder                                                #
# --------------------------------------------------------------------------- #
async def verify_field_with_sources(
    evaluator: Evaluator,
    parent_node,
    node_id: str,
    node_desc: str,
    extracted_value: Optional[str],
    expected_value: str,
    field_sources: List[str],
    fallback_sources: List[str],
    value_match_instruction: str,
    source_claim: str,
    source_instruction: str,
) -> None:
    """
    Build a sequential critical node with:
      1) Existence check (value + some sources exist)
      2) Value match check (extracted vs expected)
      3) Source support check (expected claim supported by provided URLs)
    """
    # Create the item node (critical sequential)
    item_node = evaluator.add_sequential(
        id=node_id,
        desc=node_desc,
        parent=parent_node,
        critical=True,
    )

    # Resolve sources to use (field-specific first, else fall back to all_sources)
    sources_to_use = _unique_preserve_order(field_sources or fallback_sources)

    # 1) Existence check (critical)
    exists = bool(extracted_value and extracted_value.strip()) and len(sources_to_use) > 0
    evaluator.add_custom_node(
        result=exists,
        id=f"{node_id}_exists",
        desc=f"{node_desc} — value and sources are provided",
        parent=item_node,
        critical=True,
    )

    # 2) Value match (critical)
    match_node = evaluator.add_leaf(
        id=f"{node_id}_match",
        desc=f"{node_desc} — extracted value matches expected",
        parent=item_node,
        critical=True,
    )
    claim_match = (
        f"The extracted value '{extracted_value or ''}' and the expected value '{expected_value}' "
        f"refer to the same fact."
    )
    await evaluator.verify(
        claim=claim_match,
        node=match_node,
        additional_instruction=value_match_instruction,
    )

    # 3) Source support (critical)
    source_node = evaluator.add_leaf(
        id=f"{node_id}_source_support",
        desc=f"{node_desc} — claim is supported by cited sources",
        parent=item_node,
        critical=True,
    )
    await evaluator.verify(
        claim=source_claim,
        node=source_node,
        sources=sources_to_use,
        additional_instruction=source_instruction,
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
    Evaluate an answer for Walmart's acquisition of VIZIO (December 2024).
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # The nine items are verified independently
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

    # Extract details from the answer
    extracted: AcquisitionDetailsExtraction = await evaluator.extract(
        prompt=prompt_extract_acquisition_details(),
        template_class=AcquisitionDetailsExtraction,
        extraction_name="walmart_vizio_acquisition_details",
    )

    # Add ground truth for reference
    evaluator.add_ground_truth(
        {
            "completion_date": GROUND_TRUTH["completion_date"],
            "announcement_date": GROUND_TRUTH["announcement_date"],
            "purchase_price_per_share": GROUND_TRUTH["purchase_price_per_share"],
            "total_equity_value": GROUND_TRUTH["total_equity_value"],
            "hq_location": GROUND_TRUTH["hq_combined"],
            "operating_system_name": GROUND_TRUTH["operating_system_name"],
            "continuing_ceo_name": GROUND_TRUTH["continuing_ceo_name"],
            "walmart_exec_name": GROUND_TRUTH["walmart_exec_name"],
            "reporting_segment": GROUND_TRUTH["reporting_segment"],
        },
        gt_type="ground_truth",
    )

    # Create the critical parent node for all details
    main_node = evaluator.add_parallel(
        id="Walmart_VIZIO_Acquisition_Details",
        desc="Verify all required details about the Walmart–VIZIO acquisition completed in December 2024",
        parent=root,
        critical=True,
    )

    # Value-match instructions for robustness
    date_value_ins = (
        "Treat different date formats as equivalent (e.g., 'December 3, 2024', 'Dec. 3, 2024', '2024-12-03'). "
        "Focus on whether they refer to the same calendar date."
    )
    price_value_ins = (
        "Treat currency formatting variations as equivalent. '$11.50 per share', '11.50 USD per share', or "
        "'$11.50/share' are equivalent. Note that the price is in cash."
    )
    equity_value_ins = (
        "Allow reasonable approximations and formatting variants like '$2.3B', '$2.30 billion', or 'approximately $2.3 billion'. "
        "They all refer to the same approximate value."
    )
    hq_value_ins = (
        "Treat 'Irvine, California' and 'Irvine, CA' as equivalent; minor variations or abbreviations are acceptable."
    )
    os_value_ins = (
        "Treat 'SmartCast Operating System', 'SmartCast OS', 'VIZIO OS', and 'VIZIO Operating System' as equivalent descriptions "
        "of the same operating system."
    )
    name_value_ins = (
        "Allow minor variations like middle initials, casing, and spacing. Focus on whether the names refer to the same person."
    )
    segment_value_ins = (
        "Treat wording variants like 'Walmart U.S.', 'Walmart US segment', and 'U.S. segment' as equivalent."
    )

    # Source verification instruction (general)
    general_source_ins = (
        "Verify that the provided webpages explicitly support the claim. Focus on clear statements or official press releases. "
        "Allow minor naming variations, but the substance must match the claim."
    )

    # Resolve all_sources fallback
    all_sources = _unique_preserve_order(extracted.all_sources if extracted and extracted.all_sources else [])

    # 1) Completion Date
    completion_val = extracted.completion_date.value if extracted.completion_date else None
    completion_sources = extracted.completion_date.sources if extracted.completion_date else []
    await verify_field_with_sources(
        evaluator=evaluator,
        parent_node=main_node,
        node_id="Completion_Date",
        node_desc="Acquisition completion date is December 3, 2024",
        extracted_value=completion_val,
        expected_value=GROUND_TRUTH["completion_date"],
        field_sources=completion_sources,
        fallback_sources=all_sources,
        value_match_instruction=date_value_ins,
        source_claim="The acquisition of VIZIO by Walmart was completed on December 3, 2024.",
        source_instruction=general_source_ins,
    )

    # 2) Announcement Date
    announcement_val = extracted.announcement_date.value if extracted.announcement_date else None
    announcement_sources = extracted.announcement_date.sources if extracted.announcement_date else []
    await verify_field_with_sources(
        evaluator=evaluator,
        parent_node=main_node,
        node_id="Announcement_Date",
        node_desc="Initial acquisition announcement date is February 20, 2024",
        extracted_value=announcement_val,
        expected_value=GROUND_TRUTH["announcement_date"],
        field_sources=announcement_sources,
        fallback_sources=all_sources,
        value_match_instruction=date_value_ins,
        source_claim="Walmart announced its intent to acquire VIZIO on February 20, 2024.",
        source_instruction=general_source_ins,
    )

    # 3) Purchase Price per Share
    price_val = extracted.purchase_price_per_share.value if extracted.purchase_price_per_share else None
    price_sources = extracted.purchase_price_per_share.sources if extracted.purchase_price_per_share else []
    await verify_field_with_sources(
        evaluator=evaluator,
        parent_node=main_node,
        node_id="Purchase_Price_Per_Share",
        node_desc="Purchase price is $11.50 per share in cash",
        extracted_value=price_val,
        expected_value=GROUND_TRUTH["purchase_price_per_share"],
        field_sources=price_sources,
        fallback_sources=all_sources,
        value_match_instruction=price_value_ins,
        source_claim="The agreed purchase price was $11.50 per share, in cash.",
        source_instruction=general_source_ins,
    )

    # 4) Total Equity Value
    equity_val = extracted.total_equity_value.value if extracted.total_equity_value else None
    equity_sources = extracted.total_equity_value.sources if extracted.total_equity_value else []
    await verify_field_with_sources(
        evaluator=evaluator,
        parent_node=main_node,
        node_id="Total_Equity_Value",
        node_desc="Total equity value is approximately $2.3 billion",
        extracted_value=equity_val,
        expected_value=GROUND_TRUTH["total_equity_value"],
        field_sources=equity_sources,
        fallback_sources=all_sources,
        value_match_instruction=equity_value_ins,
        source_claim="The total equity value of the transaction was approximately $2.3 billion.",
        source_instruction=general_source_ins,
    )

    # 5) Headquarters Location
    hq_city = extracted.headquarters.city if extracted.headquarters else None
    hq_state = extracted.headquarters.state if extracted.headquarters else None
    hq_combined_val = _combine_city_state(hq_city, hq_state)
    hq_sources = extracted.headquarters.sources if extracted.headquarters else []
    await verify_field_with_sources(
        evaluator=evaluator,
        parent_node=main_node,
        node_id="VIZIO_Headquarters_Location",
        node_desc="VIZIO headquarters is in Irvine, California (city and state)",
        extracted_value=hq_combined_val,
        expected_value=GROUND_TRUTH["hq_combined"],
        field_sources=hq_sources,
        fallback_sources=all_sources,
        value_match_instruction=hq_value_ins,
        source_claim="VIZIO is headquartered in Irvine, California.",
        source_instruction=general_source_ins,
    )

    # 6) Operating System Name
    os_val = extracted.operating_system_name.value if extracted.operating_system_name else None
    os_sources = extracted.operating_system_name.sources if extracted.operating_system_name else []
    await verify_field_with_sources(
        evaluator=evaluator,
        parent_node=main_node,
        node_id="Operating_System_Name",
        node_desc="Key operating system is SmartCast Operating System (also known as VIZIO OS)",
        extracted_value=os_val,
        expected_value=GROUND_TRUTH["operating_system_name"],
        field_sources=os_sources,
        fallback_sources=all_sources,
        value_match_instruction=os_value_ins,
        source_claim="A key part of the acquisition was VIZIO's SmartCast Operating System (also referred to as VIZIO OS).",
        source_instruction=general_source_ins,
    )

    # 7) Continuing CEO
    ceo_val = extracted.continuing_ceo_name.value if extracted.continuing_ceo_name else None
    ceo_sources = extracted.continuing_ceo_name.sources if extracted.continuing_ceo_name else []
    await verify_field_with_sources(
        evaluator=evaluator,
        parent_node=main_node,
        node_id="VIZIO_Continuing_CEO",
        node_desc="Continuing VIZIO CEO post-acquisition is William Wang",
        extracted_value=ceo_val,
        expected_value=GROUND_TRUTH["continuing_ceo_name"],
        field_sources=ceo_sources,
        fallback_sources=all_sources,
        value_match_instruction=name_value_ins,
        source_claim="After the acquisition, William Wang continues as VIZIO's CEO.",
        source_instruction=general_source_ins,
    )

    # 8) Walmart Executive Oversight (to whom VIZIO/CEO reports)
    exec_name_val = extracted.walmart_executive.executive_name if extracted.walmart_executive else None
    exec_sources = extracted.walmart_executive.sources if extracted.walmart_executive else []
    await verify_field_with_sources(
        evaluator=evaluator,
        parent_node=main_node,
        node_id="Walmart_Executive_Oversight",
        node_desc="VIZIO CEO reports to Seth Dallaire (EVP and Chief Growth Officer, Walmart U.S.)",
        extracted_value=exec_name_val,
        expected_value=GROUND_TRUTH["walmart_exec_name"],
        field_sources=exec_sources,
        fallback_sources=all_sources,
        value_match_instruction=name_value_ins,
        source_claim="Following the acquisition, VIZIO (or VIZIO's leadership) reports to Walmart executive Seth Dallaire.",
        source_instruction=general_source_ins,
    )

    # 9) Reporting Segment
    segment_val = extracted.reporting_segment.value if extracted.reporting_segment else None
    segment_sources = extracted.reporting_segment.sources if extracted.reporting_segment else []
    await verify_field_with_sources(
        evaluator=evaluator,
        parent_node=main_node,
        node_id="Reporting_Segment",
        node_desc="VIZIO operations are reported under the Walmart U.S. segment",
        extracted_value=segment_val,
        expected_value=GROUND_TRUTH["reporting_segment"],
        field_sources=segment_sources,
        fallback_sources=all_sources,
        value_match_instruction=segment_value_ins,
        source_claim="Post‑acquisition, VIZIO’s operations are reported under the Walmart U.S. segment.",
        source_instruction=general_source_ins,
    )

    # Return structured evaluation summary
    return evaluator.get_summary()