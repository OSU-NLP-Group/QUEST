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
TASK_ID = "apple_watch_uwb_price_2025"
TASK_DESCRIPTION = """Apple announced a new generation of its item-tracking AirTag accessory in January 2026. This updated AirTag uses an upgraded Ultra Wideband chip for improved Precision Finding capabilities. At Apple's September 2025 product announcement event, the company introduced multiple Apple Watch models that use the same generation of Ultra Wideband chip as this January 2026 AirTag.

What is the starting price (in USD) of the most affordable Apple Watch model announced at the September 2025 event that uses the same Ultra Wideband chip generation as the AirTag announced in January 2026?

Your answer must include:
1. The specific AirTag product name and its announcement date
2. The Ultra Wideband chip generation it uses
3. The specific Apple Watch model name from the September 2025 event that uses the same chip
4. Verification that this Apple Watch model is the most affordable among all September 2025 Apple Watch models with that chip
5. The starting price in USD for that Apple Watch model
6. Reference URLs for all key claims
"""

EVENT_DATE_STR = "September 9, 2025"
AIRTAG_ANNOUNCEMENT_DATE_STR = "January 26, 2026"
TARGET_UWB_GEN = "U2"  # second-generation Ultra Wideband


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AirTagInfo(BaseModel):
    product_name: Optional[str] = None
    announcement_date: Optional[str] = None
    uwb_generation: Optional[str] = None
    urls_name_date: List[str] = Field(default_factory=list)
    urls_chip: List[str] = Field(default_factory=list)


class WatchInfo(BaseModel):
    model_name: Optional[str] = None
    event_date: Optional[str] = None
    uwb_generation: Optional[str] = None
    starting_price_usd: Optional[str] = None
    urls_event: List[str] = Field(default_factory=list)
    urls_chip: List[str] = Field(default_factory=list)
    urls_price: List[str] = Field(default_factory=list)


class TaskExtraction(BaseModel):
    airtag: Optional[AirTagInfo] = None
    watch: Optional[WatchInfo] = None
    affordability_comparison_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_task() -> str:
    return f"""
Extract the required structured information from the answer for this research task. Return JSON strictly following the schema below.

Schema:
- airtag:
  - product_name: The exact AirTag product name mentioned in the answer (e.g., "AirTag (2nd generation)" or "AirTag 2").
  - announcement_date: The announcement date string as stated in the answer (e.g., "{AIRTAG_ANNOUNCEMENT_DATE_STR}").
  - uwb_generation: The Ultra Wideband chip generation the answer claims for that AirTag (e.g., "U2", "second-generation Ultra Wideband", "2nd-generation UWB").
  - urls_name_date: Array of URL(s) explicitly cited in the answer that support the AirTag product name and its announcement date.
  - urls_chip: Array of URL(s) explicitly cited in the answer that support the AirTag UWB chip generation.

- watch:
  - model_name: The exact Apple Watch model name selected in the answer (e.g., "Apple Watch SE (2025)" or "Apple Watch Series 11").
  - event_date: The event date string the answer associates with the model announcement (e.g., "{EVENT_DATE_STR}").
  - uwb_generation: The UWB chip generation the answer claims for that Apple Watch model (e.g., "U2", "second-generation Ultra Wideband").
  - starting_price_usd: The starting price value as shown in the answer for this watch (e.g., "$249" or "249 USD" or "USD 249").
  - urls_event: Array of URL(s) cited in the answer that support that this model was announced at the {EVENT_DATE_STR} Apple event (official newsroom or reputable outlets).
  - urls_chip: Array of URL(s) cited in the answer that support that this watch model uses the identified UWB generation.
  - urls_price: Array of URL(s) cited in the answer that support the starting price value for this model.

- affordability_comparison_urls: Array of URL(s) cited in the answer that support the claim that the selected watch is the most affordable among all Apple Watch models announced at the {EVENT_DATE_STR} event that include the same UWB generation. These URLs can be Apple newsroom summaries or reputable tech/news roundups that list the lineup and starting prices.

Rules:
- Only extract URLs explicitly present in the answer. Accept plain URLs or markdown links; output the actual URLs.
- If a field is missing in the answer, set it to null (for strings) or [] (for URL lists).
- Do not invent data. Do not combine information from memory. Only extract what's in the answer.
- Preserve text exactly as it appears (e.g., "$249" vs "USD 249").

Return fields:
{{
  "airtag": {{
    "product_name": ...,
    "announcement_date": ...,
    "uwb_generation": ...,
    "urls_name_date": [...],
    "urls_chip": [...]
  }},
  "watch": {{
    "model_name": ...,
    "event_date": ...,
    "uwb_generation": ...,
    "starting_price_usd": ...,
    "urls_event": [...],
    "urls_chip": [...],
    "urls_price": [...]
  }},
  "affordability_comparison_urls": [...]
}}
    """.strip()


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe(s: Optional[str]) -> str:
    return s or ""


def _combine_urls(*lists: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for lst in lists:
        for u in lst or []:
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root,
    extracted: TaskExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and launch verifications.
    """

    # Create top-level critical sequential node (reflecting the rubric's root)
    task_node = evaluator.add_sequential(
        id="Price_Research_Task",
        desc="Find the starting price (USD) of the most affordable Apple Watch model announced at the Sep 9, 2025 event that uses the same UWB chip generation as the AirTag announced in Jan 2026, and provide required supporting details and URLs.",
        parent=root,
        critical=True,
    )

    # -------------------- 1) AirTag Details -------------------------------
    airtag_node = evaluator.add_parallel(
        id="AirTag_Details",
        desc="Identify the relevant January 2026 AirTag and its UWB chip generation.",
        parent=task_node,
        critical=True,
    )

    airtag = extracted.airtag or AirTagInfo()
    airtag_name = _safe(airtag.product_name)
    airtag_date = _safe(airtag.announcement_date)
    airtag_uwb = _safe(airtag.uwb_generation)

    # Leaf: AirTag_Name_And_Announcement_Date
    leaf_airtag_name_date = evaluator.add_leaf(
        id="AirTag_Name_And_Announcement_Date",
        desc="State the specific AirTag product name and its announcement date (must be Jan 26, 2026).",
        parent=airtag_node,
        critical=True,
    )
    claim_airtag_name_date = (
        f"The product named '{airtag_name}' was announced on {AIRTAG_ANNOUNCEMENT_DATE_STR}."
        if airtag_name
        else f"The updated AirTag was announced on {AIRTAG_ANNOUNCEMENT_DATE_STR}."
    )
    add_ins_airtag_name_date = (
        "Use the provided URLs to verify the product name and announcement date. "
        f"The date must match {AIRTAG_ANNOUNCEMENT_DATE_STR} (allowing minor formatting variants like 'Jan. 26, 2026'). "
        "If the answer does not specify a clear product name (e.g., 'AirTag (2nd generation)'), mark as Incorrect. "
        "Prefer Apple's official newsroom or press materials; reputable outlets are acceptable. "
        "If no valid URLs are provided, mark as Incorrect."
    )
    await evaluator.verify(
        claim=claim_airtag_name_date,
        node=leaf_airtag_name_date,
        sources=airtag.urls_name_date,
        additional_instruction=add_ins_airtag_name_date,
    )

    # Leaf: AirTag_UWB_Chip_Generation
    leaf_airtag_chip = evaluator.add_leaf(
        id="AirTag_UWB_Chip_Generation",
        desc="Identify the Ultra Wideband chip generation used in that AirTag (must be U2 / second-generation UWB).",
        parent=airtag_node,
        critical=True,
    )
    claim_airtag_chip = (
        f"The January 2026 AirTag uses the {TARGET_UWB_GEN} (second-generation) Ultra Wideband chip."
    )
    add_ins_airtag_chip = (
        "Accept equivalent phrasings such as 'U2', '2nd‑generation UWB', or 'second-generation Ultra Wideband'. "
        "Verify using the provided URLs only; do not rely on unstated knowledge. "
        "If URLs are missing or do not support U2 explicitly (or equivalent phrasing), mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_airtag_chip,
        node=leaf_airtag_chip,
        sources=airtag.urls_chip,
        additional_instruction=add_ins_airtag_chip,
    )

    # -------------------- 2) Apple Watch Model Selection ------------------
    watch_node = evaluator.add_parallel(
        id="Apple_Watch_Model_Selection",
        desc="Identify the Apple Watch model from Sep 9, 2025 that uses the same UWB chip generation and is the cheapest among those that do.",
        parent=task_node,
        critical=True,
    )

    watch = extracted.watch or WatchInfo()
    watch_name = _safe(watch.model_name)
    watch_event_date = _safe(watch.event_date)
    watch_uwb = _safe(watch.uwb_generation)

    # Leaf: Watch_Model_Name_And_Event
    leaf_watch_event = evaluator.add_leaf(
        id="Watch_Model_Name_And_Event",
        desc="Provide the specific Apple Watch model name and confirm it was announced at the Sep 9, 2025 Apple event.",
        parent=watch_node,
        critical=True,
    )
    claim_watch_event = (
        f"The Apple Watch model '{watch_name}' was announced at Apple's event on {EVENT_DATE_STR}."
        if watch_name
        else f"An Apple Watch model was announced at Apple's event on {EVENT_DATE_STR}."
    )
    add_ins_watch_event = (
        "Verify that the specific model name appears on an official Apple newsroom page for the event "
        f"or credible coverage of the {EVENT_DATE_STR} Apple event. "
        "If the model name is missing in the answer, mark Incorrect. "
        "Accept minor name variants (e.g., with or without generation/year tokens) if they clearly refer to the same model. "
        "If no valid URLs are provided, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_watch_event,
        node=leaf_watch_event,
        sources=watch.urls_event,
        additional_instruction=add_ins_watch_event,
    )

    # Leaf: Watch_UWB_Chip_Match
    leaf_watch_chip = evaluator.add_leaf(
        id="Watch_UWB_Chip_Match",
        desc="Verify that the identified Apple Watch model uses the same UWB chip generation as the Jan 2026 AirTag (U2 / second-generation UWB).",
        parent=watch_node,
        critical=True,
    )
    claim_watch_chip = (
        f"The Apple Watch model '{watch_name}' uses the {TARGET_UWB_GEN} (second-generation) Ultra Wideband chip."
        if watch_name
        else f"This Apple Watch model uses the {TARGET_UWB_GEN} (second-generation) Ultra Wideband chip."
    )
    add_ins_watch_chip = (
        "Accept equivalent phrasings such as 'U2', '2nd‑generation UWB', or 'second-generation Ultra Wideband'. "
        "Verify using the provided URLs only (product pages, tech specs, newsroom, or reputable reviews). "
        "If URLs are missing or do not explicitly support U2 (or equivalent), mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_watch_chip,
        node=leaf_watch_chip,
        sources=watch.urls_chip,
        additional_instruction=add_ins_watch_chip,
    )

    # Leaf: Most_Affordable_Among_U2_Watches
    leaf_most_affordable = evaluator.add_leaf(
        id="Most_Affordable_Among_U2_Watches",
        desc="Verify that this model has the lowest starting price among all Apple Watch models announced at the Sep 9, 2025 event that have the U2 (second-generation UWB) chip.",
        parent=watch_node,
        critical=True,
    )
    # Combine comparison sources with event/price specifics to maximize support options
    comparison_sources = _combine_urls(
        extracted.affordability_comparison_urls,
        watch.urls_price,
        watch.urls_event,
        watch.urls_chip,
    )
    claim_most_affordable = (
        f"Among the Apple Watch models announced at Apple's {EVENT_DATE_STR} event that include the {TARGET_UWB_GEN} Ultra Wideband chip, "
        f"the {watch_name} has the lowest starting price."
        if watch_name
        else f"Among the Apple Watch models announced at Apple's {EVENT_DATE_STR} event that include the {TARGET_UWB_GEN} Ultra Wideband chip, "
             f"the selected model has the lowest starting price."
    )
    add_ins_most_affordable = (
        "Use the provided URLs to determine the lineup and their starting prices. "
        "A single credible roundup explicitly stating that this model is the most affordable/entry-level in the 2025 lineup is sufficient. "
        "If an article lists starting prices for multiple models and shows this model has the lowest price, that is also sufficient. "
        "If no valid comparison URLs are provided, or if the pages do not support that it is the cheapest among U2-equipped models, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_most_affordable,
        node=leaf_most_affordable,
        sources=comparison_sources,
        additional_instruction=add_ins_most_affordable,
    )

    # -------------------- 3) Starting Price -------------------------------
    price_node = evaluator.add_parallel(
        id="Starting_Price",
        desc="Provide the starting price in USD for the identified Apple Watch model.",
        parent=task_node,
        critical=True,
    )

    leaf_price_value = evaluator.add_leaf(
        id="Starting_Price_USD_Value",
        desc="State the starting price as a USD dollar amount for the identified Apple Watch model.",
        parent=price_node,
        critical=True,
    )
    price_str = _safe(watch.starting_price_usd)
    claim_price_value = (
        f"The starting price for the Apple Watch model '{watch_name}' is {price_str} (USD)."
        if watch_name and price_str
        else f"The starting price for the selected Apple Watch model is {price_str} (USD)."
    )
    add_ins_price_value = (
        "Verify that the page states the starting price (phrases like 'starts at', 'from', or 'starting at' are acceptable). "
        "Accept formatting variants like '$249', 'USD 249', or 'US$249' as equivalent. "
        "If the answer omits the model name or the price, or no valid URLs are provided, mark Incorrect."
    )
    await evaluator.verify(
        claim=claim_price_value,
        node=leaf_price_value,
        sources=watch.urls_price,
        additional_instruction=add_ins_price_value,
    )

    # -------------------- 4) References -----------------------------------
    refs_node = evaluator.add_parallel(
        id="References",
        desc="Provide reference URLs for all key claims, using official Apple sources or reputable news outlets (per constraints).",
        parent=task_node,
        critical=True,
    )

    # AirTag name/date refs
    ref_airtag_name_date = evaluator.add_leaf(
        id="AirTag_Name_And_Date_URLs",
        desc=f"Provide URL(s) supporting the AirTag product name and its announcement date ({AIRTAG_ANNOUNCEMENT_DATE_STR}).",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim_airtag_name_date,
        node=ref_airtag_name_date,
        sources=airtag.urls_name_date,
        additional_instruction="Re-verify that the provided URLs support both the AirTag product name and the exact announcement date. If URLs missing or not supportive, mark Incorrect.",
    )

    # AirTag chip refs
    ref_airtag_chip = evaluator.add_leaf(
        id="AirTag_UWB_Chip_URLs",
        desc="Provide URL(s) supporting that the January 2026 AirTag uses U2 / second-generation UWB.",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim_airtag_chip,
        node=ref_airtag_chip,
        sources=airtag.urls_chip,
        additional_instruction="Re-verify that the provided URLs explicitly state U2 or second‑generation UWB for the AirTag. If URLs missing or not supportive, mark Incorrect.",
    )

    # Watch event refs
    ref_watch_event = evaluator.add_leaf(
        id="Watch_Announcement_Event_URLs",
        desc=f"Provide URL(s) supporting that the identified Apple Watch model was announced at the {EVENT_DATE_STR} event.",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim_watch_event,
        node=ref_watch_event,
        sources=watch.urls_event,
        additional_instruction="Re-verify model name presence and the specific event date on the provided URLs. If URLs missing or not supportive, mark Incorrect.",
    )

    # Watch chip refs
    ref_watch_chip = evaluator.add_leaf(
        id="Watch_UWB_Chip_URLs",
        desc="Provide URL(s) supporting that the identified Apple Watch model uses U2 / second-generation UWB (same as the Jan 2026 AirTag).",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim_watch_chip,
        node=ref_watch_chip,
        sources=watch.urls_chip,
        additional_instruction="Re-verify that the provided URLs explicitly confirm U2/second‑generation UWB for the watch model. If URLs missing or not supportive, mark Incorrect.",
    )

    # Watch price refs
    ref_watch_price = evaluator.add_leaf(
        id="Watch_Starting_Price_URLs",
        desc="Provide URL(s) supporting the stated starting price (USD) for the identified Apple Watch model.",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim_price_value,
        node=ref_watch_price,
        sources=watch.urls_price,
        additional_instruction="Re-verify starting price phrasing (e.g., 'starts at', 'from'). If URLs missing or not supportive, mark Incorrect.",
    )

    # Affordability comparison refs
    ref_affordability = evaluator.add_leaf(
        id="Affordability_Comparison_URLs",
        desc="Provide URL(s) supporting that the identified model is the lowest-priced among Sep 9, 2025 Apple Watch models that have U2 / second-generation UWB.",
        parent=refs_node,
        critical=True,
    )
    await evaluator.verify(
        claim=claim_most_affordable,
        node=ref_affordability,
        sources=extracted.affordability_comparison_urls,
        additional_instruction="Re-verify cheapest/entry-level claim with the provided URLs. If URLs missing or not supportive, mark Incorrect.",
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
    Evaluate an answer for the Apple Watch UWB price research task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
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
        default_model=model,
    )

    # Extract structured information from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_task(),
        template_class=TaskExtraction,
        extraction_name="airtag_watch_extraction",
    )

    # Add small contextual info for transparency
    evaluator.add_custom_info(
        info={
            "target_airtag_announcement_date": AIRTAG_ANNOUNCEMENT_DATE_STR,
            "target_event_date": EVENT_DATE_STR,
            "target_uwb_generation": TARGET_UWB_GEN,
        },
        info_type="task_context",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return summarized evaluation results
    return evaluator.get_summary()