import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "feb_2026_tech_announcements"
TASK_DESCRIPTION = (
    "Identify two major technology announcements that occurred in February 2026:\n\n"
    "1. An artificial intelligence model that was officially released in February 2026 and achieved a score above 75% on the ARC-AGI-2 benchmark. "
    "Provide the model's name and version, the exact release date, the ARC-AGI-2 score, and a reference URL from an official source.\n\n"
    "2. A major product launch event that was held in San Francisco in February 2026, where a flagship smartphone powered by the Snapdragon 8 Elite Gen 5 "
    "processor was announced. Provide the official event name, the exact event date, confirmation of the San Francisco location, the specific model name of "
    "the announced smartphone, confirmation of its Snapdragon 8 Elite Gen 5 processor, and a reference URL from an official source."
)


# -----------------------------------------------------------------------------
# Data Models for Extraction
# -----------------------------------------------------------------------------
class AIModelCandidate(BaseModel):
    name: Optional[str] = None
    version: Optional[str] = None
    # If the answer provides a combined "name + version" string, capture it here
    model_full_name: Optional[str] = None
    release_date: Optional[str] = None
    arc_agi2_score: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class AIModelsExtraction(BaseModel):
    items: List[AIModelCandidate] = Field(default_factory=list)


class ProductEventCandidate(BaseModel):
    event_name: Optional[str] = None
    event_date: Optional[str] = None
    event_location: Optional[str] = None
    product_name: Optional[str] = None
    processor: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)


class ProductEventsExtraction(BaseModel):
    items: List[ProductEventCandidate] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Extraction Prompts
# -----------------------------------------------------------------------------
def prompt_extract_ai_models() -> str:
    return """
    Extract up to two candidate AI model announcements described in the answer that could match:
    - An AI model officially released in February 2026,
    - That achieved a score on ARC-AGI-2 (must be above 75%).
    For each AI model candidate, extract the following fields exactly as they appear in the answer:
    - name: model name (e.g., GPT-4.1, Gemini 2.0, o4-mini)
    - version: version string if provided separately (e.g., "v2", "2.0", "Pro"); if not explicitly provided separately, set to null
    - model_full_name: if the answer provides a combined name+version (e.g., "Gemini 2.0 Pro"), put the full combined string here; otherwise null
    - release_date: the exact release date string as stated in the answer text (e.g., "February 12, 2026")
    - arc_agi2_score: the ARC-AGI-2 score string exactly as stated (e.g., "76.3%", "78%", "0.78")
    - reference_urls: an array of the URL(s) cited for this model. Only include URLs explicitly present in the answer.
    Return a JSON object with an 'items' array of such candidate objects. If some fields are missing in the answer, set them to null or empty array as applicable.
    """


def prompt_extract_product_events() -> str:
    return """
    Extract up to two candidate product launch events described in the answer that could match:
    - A major launch event held in San Francisco in February 2026,
    - Where a flagship smartphone with a Snapdragon 8 Elite Gen 5 processor was announced.
    For each event candidate, extract the following fields exactly as they appear in the answer:
    - event_name: the official event name
    - event_date: the exact event date (e.g., "February 20, 2026")
    - event_location: the event location string (should indicate San Francisco if correct)
    - product_name: the specific model name of the announced smartphone
    - processor: the processor identified for the smartphone (e.g., "Snapdragon 8 Elite Gen 5")
    - reference_urls: an array of the URL(s) cited for this event. Only include URLs explicitly present in the answer.
    Return a JSON object with an 'items' array of such candidate objects. If some fields are missing in the answer, set them to null or empty array as applicable.
    """


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def compose_model_full_name(ai: AIModelCandidate) -> str:
    if ai.model_full_name and ai.model_full_name.strip():
        return ai.model_full_name.strip()
    parts: List[str] = []
    if ai.name and ai.name.strip():
        parts.append(ai.name.strip())
    if ai.version and ai.version.strip():
        # Avoid duplicating if version already embedded in name
        if ai.version.strip() not in (ai.name or ""):
            parts.append(ai.version.strip())
    return " ".join(parts).strip()


def parse_arc_agi2_score_to_percent(score_text: Optional[str]) -> Optional[float]:
    if not score_text:
        return None
    s = score_text.strip()
    # Find the first floating number
    match = re.search(r"(\d+(\.\d+)?)", s)
    if not match:
        return None
    val = float(match.group(1))
    # If <= 1.0, interpret as ratio and convert to percentage
    if val <= 1.0:
        # Heuristic: treat 0.78 as 78%
        val = val * 100.0
    return val


def first_or_empty_ai(items: List[AIModelCandidate]) -> AIModelCandidate:
    return items[0] if items else AIModelCandidate()


def first_or_empty_event(items: List[ProductEventCandidate]) -> ProductEventCandidate:
    return items[0] if items else ProductEventCandidate()


# -----------------------------------------------------------------------------
# Verification Builders
# -----------------------------------------------------------------------------
async def verify_ai_model_item(
    evaluator: Evaluator,
    parent_node,
    ai_model: AIModelCandidate,
) -> None:
    """
    Build verification sub-tree for the AI model item.
    """
    ai_node = evaluator.add_parallel(
        id="ai_model_item",
        desc="AI model released in February 2026 with ARC-AGI-2 score > 75%",
        parent=parent_node,
        critical=False
    )

    # Critical existence of required fields (name, release_date, score, at least one URL)
    required_fields_ok = (
        (bool(ai_model.name) or bool(ai_model.model_full_name)) and
        bool(ai_model.release_date) and
        bool(ai_model.arc_agi2_score) and
        bool(ai_model.reference_urls)
    )
    evaluator.add_custom_node(
        result=required_fields_ok,
        id="ai_required_fields",
        desc="AI model: required fields provided (name, release date, ARC-AGI-2 score, and at least one reference URL)",
        parent=ai_node,
        critical=True
    )

    # model_name: Verify model name + version against source(s)
    model_name_leaf = evaluator.add_leaf(
        id="model_name",
        desc="Model name and version are correct per the official source",
        parent=ai_node,
        critical=True
    )
    full_name = compose_model_full_name(ai_model)
    name_claim = (
        f"The official source mentions the AI model as '{full_name}'. "
        f"If the model has an official version (e.g., numerals like 2.0, 4.1, etc.), this full name must reflect it."
    )
    await evaluator.verify(
        claim=name_claim,
        node=model_name_leaf,
        sources=ai_model.reference_urls,
        additional_instruction=(
            "Confirm that the page clearly names the model exactly or equivalently to the provided full name. "
            "Allow minor formatting variations (hyphens/spaces), but if the page includes a version while the answer omits it, mark as not supported."
        ),
    )

    # release_date: Verify exact date and ensure it's in February 2026
    release_date_leaf = evaluator.add_leaf(
        id="release_date",
        desc="Release date is correct and in February 2026",
        parent=ai_node,
        critical=True
    )
    release_claim = f"The model was officially released on {ai_model.release_date}."
    await evaluator.verify(
        claim=release_claim,
        node=release_date_leaf,
        sources=ai_model.reference_urls,
        additional_instruction=(
            "Confirm the page supports this exact release date and that the date falls in February 2026. "
            "If the page shows a different month/year or only an announcement without an official release, mark as not supported."
        ),
    )

    # Benchmark score subnode (critical)
    bench_node = evaluator.add_parallel(
        id="benchmark_main",
        desc="ARC-AGI-2 benchmark verification",
        parent=ai_node,
        critical=True
    )

    # benchmark_score value supported by source
    bench_value_leaf = evaluator.add_leaf(
        id="benchmark_score_value",
        desc="ARC-AGI-2 score value is correctly supported by source",
        parent=bench_node,
        critical=True
    )
    score_text = ai_model.arc_agi2_score or ""
    bench_claim = f"The model achieved an ARC-AGI-2 score of {score_text}."
    await evaluator.verify(
        claim=bench_claim,
        node=bench_value_leaf,
        sources=ai_model.reference_urls,
        additional_instruction=(
            "Verify specifically that the metric is ARC-AGI-2 (not ARC original), allowing reasonable naming variants like 'ARC-AGI 2'. "
            "If the page does not show the same score value, mark as not supported."
        ),
    )

    # benchmark_score threshold > 75% (custom check, binary)
    score_val = parse_arc_agi2_score_to_percent(ai_model.arc_agi2_score)
    threshold_ok = (score_val is not None) and (score_val > 75.0)
    evaluator.add_custom_node(
        result=threshold_ok,
        id="benchmark_score_above_75",
        desc=f"ARC-AGI-2 score > 75% (parsed: {score_val if score_val is not None else 'None'})",
        parent=bench_node,
        critical=True
    )

    # reference_url: verify that at least one provided URL is an official source documenting release and benchmark
    ref_leaf = evaluator.add_leaf(
        id="ai_reference_url_official",
        desc="Reference URL is an official source documenting the model release and benchmark",
        parent=ai_node,
        critical=True
    )
    ref_claim = (
        "This page is an official source (e.g., company website, official blog/press release, "
        "research organization page, or the benchmark authority) documenting the model release and benchmark."
    )
    await evaluator.verify(
        claim=ref_claim,
        node=ref_leaf,
        sources=ai_model.reference_urls,
        additional_instruction=(
            "Judge whether the domain and content indicate an official source (e.g., the developer's site/blog, "
            "official press release, or the benchmark maintainer). If the page looks like a third-party news site, "
            "aggregator, or personal blog, mark as not supported."
        ),
    )


async def verify_product_event_item(
    evaluator: Evaluator,
    parent_node,
    event: ProductEventCandidate,
) -> None:
    """
    Build verification sub-tree for the product event item.
    """
    event_node = evaluator.add_parallel(
        id="product_event_item",
        desc="Product launch event in San Francisco (February 2026) announcing a flagship smartphone with Snapdragon 8 Elite Gen 5",
        parent=parent_node,
        critical=False
    )

    # Critical existence of required fields (event_name, event_date, event_location, product_name, processor, at least one URL)
    event_required_ok = (
        bool(event.event_name) and
        bool(event.event_date) and
        bool(event.event_location) and
        bool(event.product_name) and
        bool(event.processor) and
        bool(event.reference_urls)
    )
    evaluator.add_custom_node(
        result=event_required_ok,
        id="event_required_fields",
        desc="Event: required fields provided (event name, date, location, product name, processor, at least one reference URL)",
        parent=event_node,
        critical=True
    )

    # event_name verification
    event_name_leaf = evaluator.add_leaf(
        id="event_name",
        desc="Official event name is correct",
        parent=event_node,
        critical=True
    )
    event_name_claim = f"The official event name is '{event.event_name}'."
    await evaluator.verify(
        claim=event_name_claim,
        node=event_name_leaf,
        sources=event.reference_urls,
        additional_instruction="Confirm the page explicitly presents this as the official event name."
    )

    # event_date verification (ensure February 2026)
    event_date_leaf = evaluator.add_leaf(
        id="event_date",
        desc="Event date is correct and in February 2026",
        parent=event_node,
        critical=True
    )
    event_date_claim = f"The event took place on {event.event_date}."
    await evaluator.verify(
        claim=event_date_claim,
        node=event_date_leaf,
        sources=event.reference_urls,
        additional_instruction="Confirm the date matches exactly and falls in February 2026."
    )

    # event_location verification (San Francisco)
    event_loc_leaf = evaluator.add_leaf(
        id="event_location",
        desc="Event location is San Francisco",
        parent=event_node,
        critical=True
    )
    event_loc_claim = "The event was held in San Francisco."
    await evaluator.verify(
        claim=event_loc_claim,
        node=event_loc_leaf,
        sources=event.reference_urls,
        additional_instruction="Accept variants like 'San Francisco, CA' or venue addresses that clearly indicate San Francisco."
    )

    # Announced product subnode (critical)
    product_node = evaluator.add_parallel(
        id="announced_product",
        desc="Announced flagship smartphone details",
        parent=event_node,
        critical=True
    )

    # product_name verification (announced at the event)
    prod_name_leaf = evaluator.add_leaf(
        id="product_name",
        desc="Specific smartphone model announced at the event is correct",
        parent=product_node,
        critical=True
    )
    prod_name_claim = f"At this event, the flagship smartphone '{event.product_name}' was announced."
    await evaluator.verify(
        claim=prod_name_claim,
        node=prod_name_leaf,
        sources=event.reference_urls,
        additional_instruction="Verify that the page ties this smartphone announcement to the event itself."
    )

    # processor verification (Snapdragon 8 Elite Gen 5)
    processor_leaf = evaluator.add_leaf(
        id="processor",
        desc="Smartphone uses Snapdragon 8 Elite Gen 5 processor",
        parent=product_node,
        critical=True
    )
    processor_claim = (
        f"The smartphone '{event.product_name}' is powered by the Snapdragon 8 Elite Gen 5 processor."
    )
    await evaluator.verify(
        claim=processor_claim,
        node=processor_leaf,
        sources=event.reference_urls,
        additional_instruction="Confirm that the processor is explicitly 'Snapdragon 8 Elite Gen 5' (allow 'Qualcomm Snapdragon 8 Elite Gen 5')."
    )

    # reference_url verification (official)
    event_ref_leaf = evaluator.add_leaf(
        id="event_reference_url_official",
        desc="Reference URL is an official source documenting the event",
        parent=event_node,
        critical=True
    )
    event_ref_claim = (
        "This page is an official source (e.g., event organizer, manufacturer, or company official site/press release) documenting the event details."
    )
    await evaluator.verify(
        claim=event_ref_claim,
        node=event_ref_leaf,
        sources=event.reference_urls,
        additional_instruction=(
            "Judge official status based on domain/branding and content. If third-party media or blogs without official affiliation, mark as not supported."
        ),
    )


# -----------------------------------------------------------------------------
# Main Evaluation Entry Point
# -----------------------------------------------------------------------------
async def evaluate_answer(
    client: LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict[str, Any]:
    """
    Entry point for evaluating the answer against the February 2026 technology announcements rubric.
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

    # Extraction
    ai_models_extraction = await evaluator.extract(
        prompt=prompt_extract_ai_models(),
        template_class=AIModelsExtraction,
        extraction_name="ai_models_extraction"
    )
    product_events_extraction = await evaluator.extract(
        prompt=prompt_extract_product_events(),
        template_class=ProductEventsExtraction,
        extraction_name="product_events_extraction"
    )

    # Select first candidate for each category as per rubric (filter to first item)
    ai_model_item = first_or_empty_ai(ai_models_extraction.items)
    product_event_item = first_or_empty_event(product_events_extraction.items)

    # Build verification subtrees
    await verify_ai_model_item(evaluator, root, ai_model_item)
    await verify_product_event_item(evaluator, root, product_event_item)

    # Return the structured result
    return evaluator.get_summary()