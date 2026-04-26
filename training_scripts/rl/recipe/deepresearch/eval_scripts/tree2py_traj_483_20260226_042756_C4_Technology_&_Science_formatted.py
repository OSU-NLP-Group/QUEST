import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "verizon_outage_2026_01_14"
TASK_DESCRIPTION = (
    "On January 14, 2026, Verizon Wireless experienced a major service outage affecting customers across the "
    "United States. Create a comprehensive outage summary report that includes the following information: "
    "the exact date of the outage, the time when service was fully restored, the approximate total duration of the outage, "
    "the number of customers affected, the geographic scope of the disruption, the technical root cause as stated by Verizon, "
    "the monetary compensation amount offered to affected customers, the method for customers to redeem this compensation, and "
    "the guidance provided to customers for reconnecting to the network after service restoration."
)

# Ground truth expectations (used to judge matches to requested criteria)
GROUND_TRUTH = {
    "outage_date": "January 14, 2026",
    "resolution_time": "10:15 PM ET on January 14, 2026",
    "outage_duration": "approximately 10 hours",
    "customers_affected": "more than 1.5 million customers",
    "geographic_scope": "nationwide across the United States",
    "root_cause": "a software issue (as stated by Verizon)",
    "compensation_amount": "$20 account credits",
    "compensation_method": "redeemable via the myVerizon app",
    "restoration_guidance": "advised customers to restart their devices to reconnect",
}


# --------------------------------------------------------------------------- #
# Data models for answer extraction                                           #
# --------------------------------------------------------------------------- #
class FieldValue(BaseModel):
    value: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class OutageReportExtraction(BaseModel):
    outage_date: Optional[FieldValue] = None
    resolution_time: Optional[FieldValue] = None
    outage_duration: Optional[FieldValue] = None
    customers_affected: Optional[FieldValue] = None
    geographic_scope: Optional[FieldValue] = None
    root_cause: Optional[FieldValue] = None
    compensation_amount: Optional[FieldValue] = None
    compensation_method: Optional[FieldValue] = None
    restoration_guidance: Optional[FieldValue] = None
    global_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_outage_report() -> str:
    return """
Extract the requested outage summary details exactly as stated in the answer and, for each item, extract the URLs that the answer cites as sources supporting that specific item.

Return a JSON object matching this schema:
- outage_date: { value: string|null, sources: string[] }  // exact calendar date (e.g., "January 14, 2026")
- resolution_time: { value: string|null, sources: string[] }  // when service was fully restored (include timezone if present)
- outage_duration: { value: string|null, sources: string[] }  // approximate total duration (e.g., "about 10 hours")
- customers_affected: { value: string|null, sources: string[] }  // number or magnitude (e.g., "more than 1.5 million")
- geographic_scope: { value: string|null, sources: string[] }  // scope (e.g., "nationwide across the United States")
- root_cause: { value: string|null, sources: string[] }  // cause as stated by Verizon (e.g., "software issue")
- compensation_amount: { value: string|null, sources: string[] }  // the account credit amount (e.g., "$20")
- compensation_method: { value: string|null, sources: string[] }  // redemption method/channel (e.g., "myVerizon app")
- restoration_guidance: { value: string|null, sources: string[] }  // guidance for reconnecting (e.g., "restart devices")
- global_sources: string[]  // all URLs cited anywhere in the answer (e.g., in a general Sources section)

Extraction rules:
1) Extract values exactly as written in the answer (do not paraphrase).
2) For each item's 'sources', include only URLs that the answer appears to cite to support that specific item. If the answer does not map URLs item-by-item but provides a single general list of sources, leave the per-item 'sources' arrays empty and instead populate 'global_sources' with all cited URLs.
3) If the answer gives no sources at all, return empty arrays for 'sources' and 'global_sources'.
4) If an item is not mentioned, set its 'value' to null and 'sources' to an empty array.
5) Only include valid URLs. If a URL is missing a protocol, prepend http://.
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_text(val: Optional[str]) -> str:
    return (val or "").strip()


def _pick_sources(item: Optional[FieldValue], global_sources: List[str]) -> List[str]:
    if item and item.sources:
        return [s for s in item.sources if _safe_text(s)]
    if global_sources:
        return [s for s in global_sources if _safe_text(s)]
    return []


async def _verify_single_item(
    evaluator: Evaluator,
    parent_node,
    *,
    node_id: str,
    expected_desc: str,
    extracted_item: Optional[FieldValue],
    expected_criterion_text: str,
    url_claim_text_builder,
    match_instruction: str,
    url_instruction: str,
    field_label_for_value_check: str,
):
    """
    Build verification sub-tree for one required field.

    Structure (all children critical under this item's node to ensure item fails if any fails):
    - custom: value_present
    - custom: sources_present
    - leaf:   value_matches_expected (simple_verify)
    - leaf:   supported_by_sources (verify_by_urls)
    """
    item_node = evaluator.add_parallel(
        id=node_id,
        desc=expected_desc,
        parent=parent_node,
        critical=False
    )

    value_str = _safe_text(extracted_item.value if extracted_item else None)
    sources_list = _pick_sources(extracted_item, extracted_item.sources if extracted_item else [])
    # If the item has no per-item sources, try global fallback later in the calling function by passing correct list.
    # To keep logic localized, we will override sources_list in the caller when necessary.

    # We'll re-pick with global_sources in caller; but keep here as placeholder.

    # 1) Value present
    evaluator.add_custom_node(
        result=bool(value_str),
        id=f"{node_id}_value_present",
        desc=f"The answer provides a {field_label_for_value_check} value.",
        parent=item_node,
        critical=True
    )

    # 2) Sources present (we'll rely on caller to pass final_sources via closure in url_instruction/verification call)
    # For now, just create the node; we'll set actual result in a second custom call inside caller.
    # Instead, compute here via closure: We'll let caller supply final_sources in url instruction call; to keep cohesion,
    # we compute sources_present at the call site and pass as parameter. To avoid complexity, we do not compute here.

    # We will return item_node so caller can add sources_present and run verifications after preparing final sources.
    return item_node, value_str


async def _add_sources_and_run_verifications(
    evaluator: Evaluator,
    item_node,
    *,
    node_id: str,
    value_str: str,
    final_sources: List[str],
    expected_desc: str,
    expected_criterion_text: str,
    url_claim_text_builder,
    match_instruction: str,
    url_instruction: str,
):
    # 2) Sources present (critical)
    evaluator.add_custom_node(
        result=bool(final_sources),
        id=f"{node_id}_sources_present",
        desc="The answer provides at least one cited URL source for this item (directly or via a general sources list).",
        parent=item_node,
        critical=True
    )

    # 3) Value matches expected criterion (critical)
    match_leaf = evaluator.add_leaf(
        id=f"{node_id}_matches_expected",
        desc=expected_desc,
        parent=item_node,
        critical=True
    )
    match_claim = (
        f"The agent-stated value '{value_str}' satisfies the target criterion: {expected_criterion_text}."
    )
    await evaluator.verify(
        claim=match_claim,
        node=match_leaf,
        additional_instruction=match_instruction
    )

    # 4) Supported by cited sources (critical)
    support_leaf = evaluator.add_leaf(
        id=f"{node_id}_supported_by_sources",
        desc="The reported value is supported by the cited sources.",
        parent=item_node,
        critical=True
    )
    url_claim_text = url_claim_text_builder(value_str)
    await evaluator.verify(
        claim=url_claim_text,
        node=support_leaf,
        sources=final_sources,
        additional_instruction=url_instruction
    )


# --------------------------------------------------------------------------- #
# Main verification builder                                                   #
# --------------------------------------------------------------------------- #
async def build_outage_verification_tree(evaluator: Evaluator, root, extracted: OutageReportExtraction):
    # Create the main report node under root
    main_node = evaluator.add_parallel(
        id="Outage_Summary_Report",
        desc="Complete and accurate summary report of the January 14, 2026 Verizon wireless network outage including outage characteristics, impact assessment, and remediation measures",
        parent=root,
        critical=False
    )

    # Resolve per-item + global sources fallback
    global_sources = extracted.global_sources or []

    # 1) Outage Date
    item_node, value_str = await _verify_single_item(
        evaluator, main_node,
        node_id="Outage_Date",
        expected_desc="Report correctly identifies the outage date as January 14, 2026",
        extracted_item=extracted.outage_date,
        expected_criterion_text=GROUND_TRUTH["outage_date"],
        url_claim_text_builder=lambda v: f"The Verizon Wireless outage occurred on {v}.",
        match_instruction="Treat calendar formatting variants (e.g., 'Jan 14, 2026') as equivalent. The date must be January 14, 2026.",
        url_instruction="Verify the sources explicitly indicate the outage date as January 14, 2026. Accept formatting variants of the same date.",
        field_label_for_value_check="date of the outage"
    )
    final_sources = _pick_sources(extracted.outage_date, global_sources)
    await _add_sources_and_run_verifications(
        evaluator, item_node,
        node_id="Outage_Date",
        value_str=value_str,
        final_sources=final_sources,
        expected_desc="Report correctly identifies the outage date as January 14, 2026",
        expected_criterion_text=GROUND_TRUTH["outage_date"],
        url_claim_text_builder=lambda v: f"The Verizon Wireless outage occurred on {v}.",
        match_instruction="Treat calendar formatting variants (e.g., 'Jan 14, 2026') as equivalent. The date must be January 14, 2026.",
        url_instruction="Verify the sources explicitly indicate the outage date as January 14, 2026. Accept formatting variants of the same date."
    )

    # 2) Resolution Time
    item_node, value_str = await _verify_single_item(
        evaluator, main_node,
        node_id="Resolution_Time",
        expected_desc="Report correctly identifies that service was fully restored by 10:15 PM ET on January 14, 2026",
        extracted_item=extracted.resolution_time,
        expected_criterion_text=GROUND_TRUTH["resolution_time"],
        url_claim_text_builder=lambda v: f"Service was fully restored by {v}.",
        match_instruction=(
            "Judge whether the stated time satisfies 'by 10:15 PM ET on January 14, 2026'. "
            "Allow minor variations in formatting (e.g., lowercase 'pm'), but it must be Eastern Time and the same date."
        ),
        url_instruction=(
            "Check the sources for an explicit statement that service was fully restored by 10:15 PM Eastern Time on January 14, 2026. "
            "Do not accept partial restoration."
        ),
        field_label_for_value_check="service restoration time"
    )
    final_sources = _pick_sources(extracted.resolution_time, global_sources)
    await _add_sources_and_run_verifications(
        evaluator, item_node,
        node_id="Resolution_Time",
        value_str=value_str,
        final_sources=final_sources,
        expected_desc="Report correctly identifies that service was fully restored by 10:15 PM ET on January 14, 2026",
        expected_criterion_text=GROUND_TRUTH["resolution_time"],
        url_claim_text_builder=lambda v: f"Service was fully restored by {v}.",
        match_instruction=(
            "Judge whether the stated time satisfies 'by 10:15 PM ET on January 14, 2026'. "
            "Allow minor formatting variants; must explicitly denote Eastern Time and the correct date."
        ),
        url_instruction=(
            "Verify the sources explicitly state that full service was restored by 10:15 PM ET on January 14, 2026."
        )
    )

    # 3) Outage Duration
    item_node, value_str = await _verify_single_item(
        evaluator, main_node,
        node_id="Outage_Duration",
        expected_desc="Report correctly states the outage lasted approximately 10 hours",
        extracted_item=extracted.outage_duration,
        expected_criterion_text=GROUND_TRUTH["outage_duration"],
        url_claim_text_builder=lambda v: f"The outage lasted {v}.",
        match_instruction=(
            "Judge whether the stated duration satisfies 'approximately 10 hours'. "
            "Allow reasonable approximations (e.g., ~9 to ~11 hours) as matching the criterion."
        ),
        url_instruction=(
            "Verify the sources indicate an outage duration close to ~10 hours (approximate phrasing acceptable)."
        ),
        field_label_for_value_check="outage duration"
    )
    final_sources = _pick_sources(extracted.outage_duration, global_sources)
    await _add_sources_and_run_verifications(
        evaluator, item_node,
        node_id="Outage_Duration",
        value_str=value_str,
        final_sources=final_sources,
        expected_desc="Report correctly states the outage lasted approximately 10 hours",
        expected_criterion_text=GROUND_TRUTH["outage_duration"],
        url_claim_text_builder=lambda v: f"The outage lasted {v}.",
        match_instruction=(
            "Judge whether the stated duration satisfies 'approximately 10 hours'. "
            "Allow reasonable approximations within about ±1 hour."
        ),
        url_instruction="Confirm that the sources indicate an approximate duration of around 10 hours."
    )

    # 4) Customer Impact
    item_node, value_str = await _verify_single_item(
        evaluator, main_node,
        node_id="Customer_Impact",
        expected_desc="Report correctly indicates more than 1.5 million customers were affected",
        extracted_item=extracted.customers_affected,
        expected_criterion_text=GROUND_TRUTH["customers_affected"],
        url_claim_text_builder=lambda v: f"The outage affected {v}.",
        match_instruction=(
            "Judge whether the stated magnitude satisfies 'more than 1.5 million customers'. "
            "Accept phrasing like 'over 1.5 million' or specific numbers greater than 1.5 million."
        ),
        url_instruction="Verify the sources indicate that over 1.5 million customers were affected.",
        field_label_for_value_check="customer impact"
    )
    final_sources = _pick_sources(extracted.customers_affected, global_sources)
    await _add_sources_and_run_verifications(
        evaluator, item_node,
        node_id="Customer_Impact",
        value_str=value_str,
        final_sources=final_sources,
        expected_desc="Report correctly indicates more than 1.5 million customers were affected",
        expected_criterion_text=GROUND_TRUTH["customers_affected"],
        url_claim_text_builder=lambda v: f"The outage affected {v}.",
        match_instruction=(
            "Judge whether the stated number or phrase satisfies 'more than 1.5 million customers'. "
            "Phrases like 'over 1.5 million' or specific numbers above 1.5 million are acceptable."
        ),
        url_instruction="Confirm that the sources report over 1.5 million affected customers."
    )

    # 5) Geographic Scope
    item_node, value_str = await _verify_single_item(
        evaluator, main_node,
        node_id="Geographic_Scope",
        expected_desc="Report correctly identifies the outage as nationwide across the United States",
        extracted_item=extracted.geographic_scope,
        expected_criterion_text=GROUND_TRUTH["geographic_scope"],
        url_claim_text_builder=lambda v: f"The outage was {v}.",
        match_instruction=(
            "Judge whether the stated scope satisfies 'nationwide across the United States'. "
            "Accept equivalent phrases (e.g., 'national', 'across the U.S.', 'U.S.-wide')."
        ),
        url_instruction="Verify the sources indicate the outage was nationwide across the United States.",
        field_label_for_value_check="geographic scope"
    )
    final_sources = _pick_sources(extracted.geographic_scope, global_sources)
    await _add_sources_and_run_verifications(
        evaluator, item_node,
        node_id="Geographic_Scope",
        value_str=value_str,
        final_sources=final_sources,
        expected_desc="Report correctly identifies the outage as nationwide across the United States",
        expected_criterion_text=GROUND_TRUTH["geographic_scope"],
        url_claim_text_builder=lambda v: f"The outage was {v}.",
        match_instruction=(
            "Judge whether the stated scope matches 'nationwide across the United States'. "
            "Allow reasonable synonyms like 'nationwide', 'U.S.-wide', or 'across the U.S.'."
        ),
        url_instruction="Confirm the sources describe the outage as nationwide across the United States."
    )

    # 6) Root Cause
    item_node, value_str = await _verify_single_item(
        evaluator, main_node,
        node_id="Root_Cause",
        expected_desc="Report correctly attributes the outage to a software issue",
        extracted_item=extracted.root_cause,
        expected_criterion_text=GROUND_TRUTH["root_cause"],
        url_claim_text_builder=lambda v: f"The outage was caused by {v}.",
        match_instruction=(
            "Judge whether the stated cause satisfies 'a software issue (as stated by Verizon)'. "
            "Accept equivalent phrases like 'software error', 'software bug', or 'software issue'."
        ),
        url_instruction="Verify the sources (preferably Verizon statements) attribute the outage to a software issue.",
        field_label_for_value_check="root cause"
    )
    final_sources = _pick_sources(extracted.root_cause, global_sources)
    await _add_sources_and_run_verifications(
        evaluator, item_node,
        node_id="Root_Cause",
        value_str=value_str,
        final_sources=final_sources,
        expected_desc="Report correctly attributes the outage to a software issue",
        expected_criterion_text=GROUND_TRUTH["root_cause"],
        url_claim_text_builder=lambda v: f"The outage was caused by {v}.",
        match_instruction=(
            "Judge whether the stated cause matches 'a software issue (as stated by Verizon)'. "
            "Allow synonymous phrasing like 'software bug/error/issue'."
        ),
        url_instruction="Confirm that sources, ideally an official Verizon statement, attribute the outage to a software issue."
    )

    # 7) Compensation Amount
    item_node, value_str = await _verify_single_item(
        evaluator, main_node,
        node_id="Compensation_Amount",
        expected_desc="Report correctly states Verizon offered $20 account credits to affected customers",
        extracted_item=extracted.compensation_amount,
        expected_criterion_text=GROUND_TRUTH["compensation_amount"],
        url_claim_text_builder=lambda v: f"Verizon offered {v} to affected customers.",
        match_instruction=(
            "Judge whether the stated amount satisfies '$20 account credits'. "
            "Formatting variants (e.g., '20 dollars', '$20 credit') are acceptable if unambiguously equivalent."
        ),
        url_instruction="Verify the sources state that Verizon offered $20 credits to affected customers.",
        field_label_for_value_check="compensation amount"
    )
    final_sources = _pick_sources(extracted.compensation_amount, global_sources)
    await _add_sources_and_run_verifications(
        evaluator, item_node,
        node_id="Compensation_Amount",
        value_str=value_str,
        final_sources=final_sources,
        expected_desc="Report correctly states Verizon offered $20 account credits to affected customers",
        expected_criterion_text=GROUND_TRUTH["compensation_amount"],
        url_claim_text_builder=lambda v: f"Verizon offered {v} to affected customers.",
        match_instruction=(
            "Judge whether the stated amount matches '$20 account credits'. "
            "Accept equivalent phrasing like '$20 credit(s)'."
        ),
        url_instruction="Confirm the sources state that affected customers were offered $20 account credits."
    )

    # 8) Compensation Method
    item_node, value_str = await _verify_single_item(
        evaluator, main_node,
        node_id="Compensation_Method",
        expected_desc="Report correctly identifies the myVerizon app as the redemption channel for credits",
        extracted_item=extracted.compensation_method,
        expected_criterion_text=GROUND_TRUTH["compensation_method"],
        url_claim_text_builder=lambda v: f"Customers could redeem the compensation via {v}.",
        match_instruction=(
            "Judge whether the stated method satisfies 'redeemable via the myVerizon app'. "
            "Accept close variants like 'via the My Verizon app' or 'in the myVerizon app'."
        ),
        url_instruction="Verify the sources indicate that credits could be redeemed via the myVerizon app.",
        field_label_for_value_check="compensation redemption method"
    )
    final_sources = _pick_sources(extracted.compensation_method, global_sources)
    await _add_sources_and_run_verifications(
        evaluator, item_node,
        node_id="Compensation_Method",
        value_str=value_str,
        final_sources=final_sources,
        expected_desc="Report correctly identifies the myVerizon app as the redemption channel for credits",
        expected_criterion_text=GROUND_TRUTH["compensation_method"],
        url_claim_text_builder=lambda v: f"Customers could redeem the compensation via {v}.",
        match_instruction=(
            "Judge whether the stated method matches 'redeemable via the myVerizon app'. "
            "Allow small phrasing differences referring to the same app."
        ),
        url_instruction="Confirm that sources specify the myVerizon app as the redemption channel."
    )

    # 9) Restoration Guidance
    item_node, value_str = await _verify_single_item(
        evaluator, main_node,
        node_id="Restoration_Guidance",
        expected_desc="Report correctly notes that Verizon advised customers to restart their devices to reconnect to the network",
        extracted_item=extracted.restoration_guidance,
        expected_criterion_text=GROUND_TRUTH["restoration_guidance"],
        url_claim_text_builder=lambda v: f"Verizon advised customers to {v}.",
        match_instruction=(
            "Judge whether the stated guidance satisfies 'advised customers to restart their devices to reconnect'. "
            "Accept equivalent phrasing like 'reboot phones' or 'power cycle devices' if clearly the same guidance."
        ),
        url_instruction="Verify the sources indicate Verizon advised customers to restart their devices to reconnect after restoration.",
        field_label_for_value_check="restoration guidance"
    )
    final_sources = _pick_sources(extracted.restoration_guidance, global_sources)
    await _add_sources_and_run_verifications(
        evaluator, item_node,
        node_id="Restoration_Guidance",
        value_str=value_str,
        final_sources=final_sources,
        expected_desc="Report correctly notes that Verizon advised customers to restart their devices to reconnect to the network",
        expected_criterion_text=GROUND_TRUTH["restoration_guidance"],
        url_claim_text_builder=lambda v: f"Verizon advised customers to {v}.",
        match_instruction=(
            "Judge whether the stated guidance matches 'restart their devices to reconnect'. "
            "Allow synonymous guidance like 'reboot device' if clearly equivalent."
        ),
        url_instruction="Confirm the sources state that Verizon advised customers to restart devices to reconnect."
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
    Evaluate an answer for the Verizon outage report task and return a structured result.
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
        default_model=model
    )

    # Extraction
    extracted = await evaluator.extract(
        prompt=prompt_extract_outage_report(),
        template_class=OutageReportExtraction,
        extraction_name="outage_report_extraction"
    )

    # Ground truth info for transparency
    evaluator.add_ground_truth({
        "expected": GROUND_TRUTH
    }, gt_type="ground_truth_outage_report")

    # Build verification tree
    await build_outage_verification_tree(evaluator, root, extracted)

    # Return aggregated summary
    return evaluator.get_summary()