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
TASK_ID = "christmas_day_2024_retail_access"
TASK_DESCRIPTION = (
    "For Christmas Day 2024 in the United States, provide the following information: "
    "(1) Name a national pharmacy chain that remains open on Christmas Day, "
    "(2) Name a convenience store chain that operates 24/7 on Christmas Day, "
    "(3) Name a grocery store chain that is open on Christmas Day, "
    "(4) Indicate whether Walmart, Target, and Costco are open or closed on Christmas Day, "
    "and (5) Name a fast-food chain that has locations potentially open on Christmas Day."
)

CHRISTMAS_DAY_DATE_TEXT = "December 25, 2024"
COUNTRY_TEXT = "United States"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class NamedEntity(BaseModel):
    name: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RetailerStatus(BaseModel):
    status: Optional[str] = None  # Expect normalized 'open' or 'closed'
    sources: List[str] = Field(default_factory=list)


class MajorRetailers(BaseModel):
    walmart: Optional[RetailerStatus] = None
    target: Optional[RetailerStatus] = None
    costco: Optional[RetailerStatus] = None


class AnswerExtraction(BaseModel):
    pharmacy: Optional[NamedEntity] = None
    convenience: Optional[NamedEntity] = None
    grocery: Optional[NamedEntity] = None
    major: Optional[MajorRetailers] = None
    fast_food: Optional[NamedEntity] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_main() -> str:
    return f"""
Extract the information the answer provides for Christmas Day 2024 in the United States. Return a single JSON with the following structure:

{{
  "pharmacy": {{"name": string|null, "sources": string[]}},
  "convenience": {{"name": string|null, "sources": string[]}},
  "grocery": {{"name": string|null, "sources": string[]}},
  "major": {{
    "walmart": {{"status": "open"|"closed"|null, "sources": string[]}},
    "target":  {{"status": "open"|"closed"|null, "sources": string[]}},
    "costco":  {{"status": "open"|"closed"|null, "sources": string[]}}
  }},
  "fast_food": {{"name": string|null, "sources": string[]}}
}}

Extraction rules:
- "pharmacy": Extract the first national pharmacy chain explicitly named as open on Christmas Day, and all URLs the answer cites that support that claim. If multiple are listed, choose the first. If none, set name to null and sources to [].
- "convenience": Extract the first convenience store chain explicitly named as operating 24/7 on Christmas Day, and all supporting URLs. If none, set null.
- "grocery": Extract the first grocery store chain explicitly named as open on Christmas Day, and all supporting URLs. If none, set null.
- "major": For Walmart, Target, and Costco, extract the explicit open/closed status the answer states for Christmas Day 2024, normalized to exactly "open" or "closed". If the answer uses phrases like "closed for the holiday" or "not open", normalize to "closed". If "open with limited hours", normalize to "open". If the answer does not clearly indicate open/closed for a retailer, set status to null. Extract all URLs the answer cites that directly support each retailer's status.
- "fast_food": Extract the first fast-food chain the answer claims has at least some locations potentially open on Christmas Day (hours may vary), and all supporting URLs. If none, set null.

Important:
- Only extract URLs explicitly present in the answer text (including markdown links). Do not invent URLs.
- Do not include general or unrelated URLs; include only those that support the specific claim.
- If the answer provides a sources section at the end, allocate each URL to the specific claim(s) it supports where possible.
- If a required field is missing in the answer, return null for the field (or [] for sources).
"""


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def normalize_status(status: Optional[str]) -> Optional[str]:
    if not status:
        return None
    s = status.strip().lower()
    # Check for "closed" semantics first to avoid 'open' inside 'not open'
    closed_markers = [
        "closed", "not open", "will be closed", "closed on", "holiday closure",
        "closed for christmas", "closed for the holiday"
    ]
    for m in closed_markers:
        if m in s:
            return "closed"
    open_markers = [
        "open", "will be open", "open on", "open with limited hours", "reduced hours"
    ]
    for m in open_markers:
        if m in s:
            return "open"
    return None


def has_nonempty_sources(sources: Optional[List[str]]) -> bool:
    return bool(sources) and any((u or "").strip() for u in sources or [])


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_named_chain_open(
    evaluator: Evaluator,
    parent,
    node_id_base: str,
    node_desc: str,
    entity: Optional[NamedEntity],
    claim_text: str,
    add_ins: str,
) -> None:
    node = evaluator.add_parallel(
        id=node_id_base,
        desc=node_desc,
        parent=parent,
        critical=False
    )
    exists = bool(entity and entity.name and str(entity.name).strip()) and has_nonempty_sources(entity.sources if entity else [])
    evaluator.add_custom_node(
        result=exists,
        id=f"{node_id_base}_exists",
        desc=f"{node_desc} — entity identified with supporting sources",
        parent=node,
        critical=True
    )

    support_leaf = evaluator.add_leaf(
        id=f"{node_id_base}_source_support",
        desc=f"Sources support: {claim_text.format(name=(entity.name if entity and entity.name else ''))}",
        parent=node,
        critical=True
    )
    sources_list = (entity.sources if entity else []) if entity else []
    chain_name = entity.name if entity and entity.name else ""
    claim = claim_text.format(name=chain_name)
    await evaluator.verify(
        claim=claim,
        node=support_leaf,
        sources=sources_list,
        additional_instruction=add_ins
    )


async def verify_retailer_status(
    evaluator: Evaluator,
    parent,
    retailer_name: str,
    status_obj: Optional[RetailerStatus],
    parent_id_base: str
) -> None:
    retailer_node = evaluator.add_parallel(
        id=f"{parent_id_base}_{retailer_name.lower()}",
        desc=f"{retailer_name} status on Christmas Day 2024",
        parent=parent,
        critical=False
    )

    norm_status = normalize_status(status_obj.status if status_obj else None)
    exists = bool(norm_status) and has_nonempty_sources(status_obj.sources if status_obj else [])
    evaluator.add_custom_node(
        result=exists,
        id=f"{parent_id_base}_{retailer_name.lower()}_exists",
        desc=f"{retailer_name}: status provided (open/closed) with sources",
        parent=retailer_node,
        critical=True
    )

    status_leaf = evaluator.add_leaf(
        id=f"{parent_id_base}_{retailer_name.lower()}_status_supported",
        desc=f"Sources support that {retailer_name} is {norm_status or 'UNKNOWN'} on Christmas Day 2024",
        parent=retailer_node,
        critical=True
    )

    status_text = norm_status or "unknown"
    if status_text == "unknown":
        # Still call verify to let the framework record failure cleanly with whatever sources were given.
        claim = f"On {CHRISTMAS_DAY_DATE_TEXT}, {retailer_name} is {status_text} in the {COUNTRY_TEXT}."
        sources = status_obj.sources if status_obj else []
        await evaluator.verify(
            claim=claim,
            node=status_leaf,
            sources=sources,
            additional_instruction=(
                "The provided status could not be normalized to open/closed. If sources don't clearly state open or closed, "
                "mark as not supported."
            )
        )
    else:
        claim = f"On {CHRISTMAS_DAY_DATE_TEXT}, {retailer_name} is {status_text} in the {COUNTRY_TEXT}."
        sources = status_obj.sources if status_obj else []
        await evaluator.verify(
            claim=claim,
            node=status_leaf,
            sources=sources,
            additional_instruction=(
                "Verify the specific open/closed status for Christmas Day 2024 using the provided URLs. "
                "Accept phrases like 'closed for Christmas' as 'closed' and 'open with limited hours' as 'open'. "
                "Focus on U.S. locations. If conflicting info across sources, prefer official retailer statements or "
                "reputable news; otherwise, mark as not supported."
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
    # Initialize evaluator with PARALLEL root since categories are independent
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

    # Extract structured info
    extraction = await evaluator.extract(
        prompt=prompt_extract_main(),
        template_class=AnswerExtraction,
        extraction_name="christmas_2024_retail_extraction",
    )

    # Build tree categories

    # 1) Pharmacy open on Christmas Day
    await verify_named_chain_open(
        evaluator=evaluator,
        parent=root,
        node_id_base="OpenPharmacyChain",
        node_desc="Identify at least one national pharmacy chain that is open on Christmas Day 2024",
        entity=extraction.pharmacy,
        claim_text=f"The chain {{name}} has at least some U.S. locations open on {CHRISTMAS_DAY_DATE_TEXT}. Hours may vary by location.",
        add_ins=(
            "Confirm that the cited source(s) explicitly indicate that the chain is open on Christmas Day 2024 in the U.S. "
            "It's acceptable if only some/selected locations are open or hours are reduced. If sources only refer to other years "
            "or vague holidays without Christmas Day, mark as not supported."
        ),
    )

    # 2) Convenience store operating 24/7 on Christmas Day
    await verify_named_chain_open(
        evaluator=evaluator,
        parent=root,
        node_id_base="TwentyFourSevenConvenienceStore",
        node_desc="Identify at least one convenience store chain that operates 24/7 on Christmas Day 2024",
        entity=extraction.convenience,
        claim_text=f"The chain {{name}} operates 24/7 and remains open on {CHRISTMAS_DAY_DATE_TEXT} in the {COUNTRY_TEXT}.",
        add_ins=(
            "Verify that the source(s) indicate 24/7 operation that includes Christmas Day 2024. "
            "Accept statements like 'open 365 days a year' if they reasonably imply being open on Christmas Day."
        ),
    )

    # 3) Grocery store open on Christmas Day
    await verify_named_chain_open(
        evaluator=evaluator,
        parent=root,
        node_id_base="OpenGroceryStore",
        node_desc="Identify at least one grocery store chain that is open on Christmas Day 2024",
        entity=extraction.grocery,
        claim_text=f"The grocery chain {{name}} has at least some U.S. locations open on {CHRISTMAS_DAY_DATE_TEXT}. Hours may vary by location.",
        add_ins=(
            "Verify that at least some locations are open on Christmas Day 2024. "
            "Sources may be official store hours pages or reputable news. If only curbside or pharmacy is open, that still counts as open."
        ),
    )

    # 4) Major retailers Walmart, Target, Costco open/closed verification
    major_node = evaluator.add_parallel(
        id="MajorRetailersClosed",
        desc="Verify the answer's Walmart, Target, and Costco statuses for Christmas Day 2024",
        parent=root,
        critical=False
    )
    major = extraction.major or MajorRetailers()

    await verify_retailer_status(
        evaluator=evaluator,
        parent=major_node,
        retailer_name="Walmart",
        status_obj=major.walmart if major else None,
        parent_id_base="major"
    )
    await verify_retailer_status(
        evaluator=evaluator,
        parent=major_node,
        retailer_name="Target",
        status_obj=major.target if major else None,
        parent_id_base="major"
    )
    await verify_retailer_status(
        evaluator=evaluator,
        parent=major_node,
        retailer_name="Costco",
        status_obj=major.costco if major else None,
        parent_id_base="major"
    )

    # 5) Fast-food chain with at least some locations potentially open
    await verify_named_chain_open(
        evaluator=evaluator,
        parent=root,
        node_id_base="FastFoodAvailability",
        node_desc="Identify at least one fast-food chain that has locations potentially open on Christmas Day 2024",
        entity=extraction.fast_food,
        claim_text=f"The fast-food chain {{name}} has at least some locations potentially open on {CHRISTMAS_DAY_DATE_TEXT} in the {COUNTRY_TEXT}. Hours may vary by location.",
        add_ins=(
            "Confirm that at least some locations may be open on Christmas Day 2024. "
            "Sources should state 'select locations open', 'hours vary by location', or similar language indicating potential availability."
        ),
    )

    # Optional: add custom info to summary
    evaluator.add_custom_info(
        info={"holiday": "Christmas Day", "date": CHRISTMAS_DAY_DATE_TEXT, "country": COUNTRY_TEXT},
        info_type="context",
        info_name="holiday_context"
    )

    return evaluator.get_summary()