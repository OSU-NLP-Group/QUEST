import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator, AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "yellowstone_winter_first_2026_2027"
TASK_DESCRIPTION = """
Which lodge in Yellowstone National Park opens first for the winter 2026-2027 season?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class SelectedLodge(BaseModel):
    """The lodge the answer claims opens first for winter 2026–2027."""
    name: Optional[str] = None
    opening_date: Optional[str] = None  # Keep as free-form string; do not enforce a date format
    urls: List[str] = Field(default_factory=list)  # URLs directly supporting this lodge's opening info


class YellowstoneWinterFirstExtraction(BaseModel):
    """Structured extraction from the answer."""
    selected_lodge: Optional[SelectedLodge] = None
    schedule_urls: List[str] = Field(
        default_factory=list,
        description="URLs that list winter 2026–2027 opening dates for multiple Yellowstone lodges (e.g., official schedule pages)."
    )


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_yellowstone_winter_first() -> str:
    return """
    From the provided answer, extract the single specific lodge that the answer claims "opens first" for Yellowstone National Park's winter 2026–2027 season.

    Return a JSON object with:
    - selected_lodge:
        - name: The exact lodge name as stated in the answer (e.g., "Old Faithful Snow Lodge", "Mammoth Hot Springs Hotel"). If multiple are mentioned, choose the one explicitly claimed as the FIRST to open. If none is clearly identified, set to null.
        - opening_date: The opening date for that lodge for the winter 2026–2027 season as stated in the answer (string). If the answer provides a specific date or approximate like "mid-December 2026", copy it verbatim; otherwise null.
        - urls: All URLs cited in the answer that directly support this lodge's winter opening information for 2026–2027 (e.g., official Yellowstone National Park Lodges/Xanterra pages, NPS pages, or other credible sources). Include every relevant URL mentioned.
    - schedule_urls: Any URL(s) in the answer that list opening/closing dates for multiple Yellowstone winter lodges for 2026–2027 (e.g., a consolidated “Winter 2026–2027 opening dates” page). Include all such URLs. If none are provided, return an empty list.

    IMPORTANT:
    - Extract only what is explicitly present in the answer. Do not invent URLs or dates.
    - For URLs, capture full valid URLs. If the URL appears without a protocol, prepend "http://".
    - If the answer mentions sources like "according to Yellowstone National Park Lodges" but gives no URL, do NOT fabricate one; leave urls empty for that field.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(urls: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for u in urls:
        if not u:
            continue
        v = u.strip()
        if not v:
            continue
        if v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


def _gather_sources(extracted: YellowstoneWinterFirstExtraction) -> List[str]:
    urls: List[str] = []
    if extracted and extracted.selected_lodge and extracted.selected_lodge.urls:
        urls.extend(extracted.selected_lodge.urls)
    if extracted and extracted.schedule_urls:
        urls.extend(extracted.schedule_urls)
    return _dedup_urls(urls)


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: YellowstoneWinterFirstExtraction,
) -> None:
    """
    Build the verification tree per rubric and run verifications.
    """
    # Create the main rubric node (critical, parallel aggregation)
    main_node = evaluator.add_parallel(
        id="Yellowstone_Winter_Lodge_Identification",
        desc="Identify which lodge opens first for the winter 2026–2027 season (as constrained).",
        parent=parent_node,
        critical=True
    )

    # Convenience vars
    sel = extracted.selected_lodge if extracted else None
    lodge_name = (sel.name.strip() if sel and sel.name else "") or ""
    opening_date = (sel.opening_date.strip() if sel and sel.opening_date else "") or ""
    combined_sources = _gather_sources(extracted)

    # 1) Identifies_A_Specific_Lodge (custom existence check; about answer content)
    evaluator.add_custom_node(
        result=bool(lodge_name),
        id="Identifies_A_Specific_Lodge",
        desc="The answer names a specific lodge as the one that opens first.",
        parent=main_node,
        critical=True
    )

    # 2) Lodge_Is_In_Yellowstone_NP (ground with provided URLs)
    node_in_ynp = evaluator.add_leaf(
        id="Lodge_Is_In_Yellowstone_NP",
        desc="The named lodge is located in Yellowstone National Park.",
        parent=main_node,
        critical=True
    )
    claim_in_ynp = f"The lodge named '{lodge_name}' is located within Yellowstone National Park (USA)."
    await evaluator.verify(
        claim=claim_in_ynp,
        node=node_in_ynp,
        sources=combined_sources,
        additional_instruction=(
            "Verify that the named property is a lodge/hotel that is actually within Yellowstone National Park. "
            "Use the provided official or credible sources (e.g., yellowstonenationalparklodges.com or nps.gov). "
            "Minor naming variants (e.g., 'Hotel' vs. 'Lodge') should be treated as the same place."
        ),
    )

    # 3) Lodge_Is_Open_Winter_2026_2027
    node_open_winter = evaluator.add_leaf(
        id="Lodge_Is_Open_Winter_2026_2027",
        desc="The named lodge operates/is open during the winter 2026–2027 season.",
        parent=main_node,
        critical=True
    )
    claim_open_winter = (
        f"For the winter 2026–2027 season, the lodge '{lodge_name}' operates (is open to guests)."
    )
    await evaluator.verify(
        claim=claim_open_winter,
        node=node_open_winter,
        sources=combined_sources,
        additional_instruction=(
            "Check if the provided page(s) specify that this lodge operates during the winter 2026–2027 season. "
            "This can be indicated by explicit winter season dates, opening/closing dates in late 2026 to early 2027, "
            "or a winter operations schedule for 2026–2027."
        ),
    )

    # 4) Earliest_Opening_Date_Among_Winter_Lodges
    node_earliest = evaluator.add_leaf(
        id="Earliest_Opening_Date_Among_Winter_Lodges",
        desc="The named lodge’s opening date for winter 2026–2027 is earlier than every other lodge that operates in winter 2026–2027 (per the provided winter-lodge set/opening dates).",
        parent=main_node,
        critical=True
    )
    human_date = opening_date if opening_date else "its stated opening date"
    claim_earliest = (
        f"For Yellowstone National Park's 2026–2027 winter season, the first lodge to open is '{lodge_name}' "
        f"(opening on {human_date}), and no other winter-operating lodge opens earlier."
    )
    await evaluator.verify(
        claim=claim_earliest,
        node=node_earliest,
        sources=combined_sources,
        additional_instruction=(
            "Use any consolidated winter schedule page(s) or official lodge pages to confirm opening dates across all winter-operating lodges. "
            "Judge the claim strictly: if another lodge opens on the same day (a tie) or earlier, then this claim is incorrect. "
            "If a single official schedule page lists all lodges and dates, rely on that."
        ),
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
) -> Dict[str, Any]:
    """
    Evaluate an answer for: Which lodge in Yellowstone National Park opens first for the winter 2026–2027 season?
    Returns a structured summary dict with the verification tree and score.
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
        default_model=model,
    )

    # Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_yellowstone_winter_first(),
        template_class=YellowstoneWinterFirstExtraction,
        extraction_name="yellowstone_winter_first_extraction",
    )

    # Optionally record key extracted info for transparency
    evaluator.add_custom_info(
        info={
            "selected_lodge": extracted.selected_lodge.dict() if extracted and extracted.selected_lodge else None,
            "schedule_urls": extracted.schedule_urls if extracted else [],
        },
        info_type="extraction_summary",
        info_name="extracted_selection"
    )

    # Build tree and run verifications
    await build_and_verify_tree(evaluator, root, extracted)

    # Return final structured result
    return evaluator.get_summary()