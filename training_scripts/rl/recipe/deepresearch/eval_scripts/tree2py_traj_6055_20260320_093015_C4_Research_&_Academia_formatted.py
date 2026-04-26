import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy, VerificationNode
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task metadata                                                               #
# --------------------------------------------------------------------------- #
TASK_ID = "summer_2026_astro_confs"
TASK_DESCRIPTION = """I am an astrophysics researcher planning my conference attendance for the 2026 summer season. I want to identify major international conferences where I can present my research and network with colleagues from around the world.

Identify at least four international astronomy or astrophysics conferences that meet all of the following criteria:

1. The conference must be held between May 1, 2026, and September 30, 2026 (inclusive).
2. The conference must be an in-person event held at a physical location (not virtual or online-only).
3. The conference must be an international conference, symposium, congress, or assembly (as indicated by its name, scope, or organizing body).
4. The conference must be held in Europe, Asia, or North America.

For each qualifying conference, provide:
- The official name of the conference
- The exact dates (start date and end date) in 2026
- The city and country where it will be held
- A brief description (1-2 sentences) of the conference's main focus or topic areas
- A reference URL to the official conference website or a reputable astronomy conference listing (such as AAS calendar, CADC International Astronomy Meetings database, or IAU listings)
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class ConferenceItem(BaseModel):
    name: Optional[str] = None
    start_date: Optional[str] = None  # Keep as string for robustness (e.g., "June 10, 2026" or "2026-06-10")
    end_date: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    description: Optional[str] = None
    format: Optional[str] = None  # e.g., "in-person", "hybrid", "virtual"
    scope: Optional[str] = None   # e.g., "international", "global", "IAU", etc.
    reference_urls: List[str] = Field(default_factory=list)


class ConferencesExtraction(BaseModel):
    conferences: List[ConferenceItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_conferences() -> str:
    return """
    Extract the astronomy/astrophysics conference entries mentioned in the answer.
    For each conference, extract the following fields exactly as stated in the answer:
    - name: The official conference name
    - start_date: The start date in 2026 (string as provided; keep the original formatting)
    - end_date: The end date in 2026 (string as provided; keep the original formatting)
    - city: The host city
    - country: The host country
    - description: A brief 1–2 sentence description provided in the answer (if present)
    - format: A short phrase indicating the event format if present (e.g., "in-person", "onsite", "hybrid", "virtual")
    - scope: A short phrase about scope if present (e.g., contains "International", "IAU", "global", "worldwide")
    - reference_urls: All URLs explicitly cited in the answer that point to the official site or reputable listings (e.g., AAS calendar, CADC, IAU). Extract the actual URLs only.
    
    Rules:
    - Do not invent any information. If a field is missing, set it to null (or empty list for URLs).
    - Only include conferences explicitly mentioned in the answer.
    - Preserve the order from the answer. If more than 5 are listed, include them all; evaluators may consider the first 5.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def safe_str(x: Optional[str]) -> str:
    return (x or "").strip()

def safe_urls(urls: Optional[List[str]]) -> List[str]:
    return [u for u in (urls or []) if isinstance(u, str) and u.strip()]

def first_n(items: List[Any], n: int) -> List[Any]:
    return items[:n] if len(items) > n else items

def pad_to_length(items: List[Any], length: int, pad_item: Any) -> List[Any]:
    return items + [pad_item for _ in range(max(0, length - len(items)))]


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def verify_single_conference(
    evaluator: Evaluator,
    parent_node: VerificationNode,
    conf: ConferenceItem,
    idx: int,
) -> VerificationNode:
    """
    Build verification subtree for a single conference and launch verifications.
    """
    conf_idx = idx + 1
    name = safe_str(conf.name)
    start = safe_str(conf.start_date)
    end = safe_str(conf.end_date)
    city = safe_str(conf.city)
    country = safe_str(conf.country)
    desc = safe_str(conf.description)
    srcs = safe_urls(conf.reference_urls)

    # Parent node for this conference (parallel aggregation, non-critical)
    conf_node = evaluator.add_parallel(
        id=f"conference_{conf_idx}",
        desc=f"Evaluates all required attributes for conference #{conf_idx}.",
        parent=parent_node,
        critical=False,
    )

    # Create leaves (all critical as per rubric)
    name_dates_node = evaluator.add_leaf(
        id=f"conference_{conf_idx}_name_dates",
        desc="Conference has an official name and 2026 dates (May 1–Sep 30 inclusive) with exact start and end dates.",
        parent=conf_node,
        critical=True,
    )

    in_person_node = evaluator.add_leaf(
        id=f"conference_{conf_idx}_in_person",
        desc="Conference is an in-person event at a physical location (not virtual-only).",
        parent=conf_node,
        critical=True,
    )

    international_node = evaluator.add_leaf(
        id=f"conference_{conf_idx}_international",
        desc="Conference is an international conference/symposium/congress/assembly (as indicated by name/scope/organizer).",
        parent=conf_node,
        critical=True,
    )

    geo_node = evaluator.add_leaf(
        id=f"conference_{conf_idx}_geo",
        desc="Conference is held in Europe, Asia, or North America, with city and country specified.",
        parent=conf_node,
        critical=True,
    )

    # Description presence check as a critical custom node (existence as required by rubric)
    desc_present = evaluator.add_custom_node(
        result=(len(desc) > 0),
        id=f"conference_{conf_idx}_description",
        desc="A brief description (1–2 sentences) of main focus/topic areas is provided.",
        parent=conf_node,
        critical=True,
    )

    ref_url_node = evaluator.add_leaf(
        id=f"conference_{conf_idx}_reference_url",
        desc="A reference URL points to the official site or a reputable astronomy conference listing for this event.",
        parent=conf_node,
        critical=True,
    )

    # Build claims and dispatch verifications (prefer parallel batch)
    claims_and_sources: List[tuple[str, List[str] | None, VerificationNode, Optional[str]]] = []

    # 1) Name and dates within window (verify via cited URLs)
    name_dates_claim = (
        f"The official conference name is '{name}', and it takes place from {start} to {end} in 2026. "
        f"These dates fall between May 1 and September 30, 2026 (inclusive)."
    )
    name_dates_ins = (
        "Verify the page explicitly shows the conference name and the 2026 dates. "
        "Confirm that the start/end dates fall within May 1–September 30, 2026 (inclusive). "
        "If the page shows a different year or dates outside this range, mark unsupported."
    )
    claims_and_sources.append((name_dates_claim, srcs, name_dates_node, name_dates_ins))

    # 2) In-person format (onsite/hybrid acceptable, not virtual-only)
    in_person_claim = (
        f"The 2026 {name} is an in-person (onsite) event held at a physical location; "
        f"hybrid formats with an onsite component are acceptable. It is not virtual-only."
    )
    in_person_ins = (
        "Look for indicators like 'in-person', 'onsite', venue/address, or a specified location. "
        "If the page indicates 'virtual only' or lacks any physical venue/location, mark unsupported."
    )
    claims_and_sources.append((in_person_claim, srcs, in_person_node, in_person_ins))

    # 3) International scope
    international_claim = (
        f"The 2026 {name} is an international conference/symposium/congress/assembly (international in scope)."
    )
    international_ins = (
        "Accept if 'International' appears in the title, if the organizer is an international body (e.g., IAU, SPIE, COSPAR), "
        "or if the page clearly frames the event as international/global. National-only meetings without international framing do not count."
    )
    claims_and_sources.append((international_claim, srcs, international_node, international_ins))

    # 4) Geographic location (Europe, Asia, or North America) with specified city and country
    geo_claim = (
        f"The 2026 {name} will be held in {city}, {country}, which is located in Europe, Asia, or North America."
    )
    geo_ins = (
        "Verify the city and country on the page. Then judge whether the country is in Europe, Asia, or North America. "
        "If the location is outside these continents or not specified, mark unsupported."
    )
    claims_and_sources.append((geo_claim, srcs, geo_node, geo_ins))

    # 5) Reference URL is about this event and is official or reputable listing
    ref_claim = (
        f"This webpage is the official site or a reputable astronomy conference listing page for the 2026 {name}."
    )
    ref_ins = (
        "The page must be the official conference website or a reputable astronomy conference listing (e.g., AAS calendar, CADC International Astronomy Meetings, IAU listings). "
        "If the link is a random/unrelated page, mark unsupported."
    )
    claims_and_sources.append((ref_claim, srcs, ref_url_node, ref_ins))

    # Run the verifications in parallel for this conference
    await evaluator.batch_verify(claims_and_sources)

    return conf_node


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
) -> Dict:
    """
    Evaluate an answer for the 2026 summer astronomy/astrophysics international conferences task.
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

    # Top-level collection node (parallel aggregation)
    collection_node = evaluator.add_parallel(
        id="conference_collection",
        desc="Evaluates whether the solution provides at least four qualifying international astronomy/astrophysics conferences for Summer 2026.",
        parent=root,
        critical=False,
    )

    # Extract conferences from the answer
    extracted: ConferencesExtraction = await evaluator.extract(
        prompt=prompt_extract_conferences(),
        template_class=ConferencesExtraction,
        extraction_name="conferences_extraction",
    )

    # Keep only the first 5 for evaluation; pad if fewer
    confs = first_n(extracted.conferences, 5)
    confs = pad_to_length(confs, 5, ConferenceItem())

    # Build verification subtrees for up to 5 conferences
    conf_nodes: List[VerificationNode] = []
    for i, conf in enumerate(confs):
        node = await verify_single_conference(evaluator, collection_node, conf, i)
        conf_nodes.append(node)

    # Compute how many conferences fully qualify (i.e., their node aggregated score equals 1.0)
    qualified_count = 0
    for node in conf_nodes:
        try:
            score = node.aggregated_score  # triggers computation with mutation
            if score == 1.0:
                qualified_count += 1
        except Exception:
            pass

    evaluator.add_custom_info(
        info={"num_extracted": len(extracted.conferences), "num_evaluated": len(confs), "qualified_count": qualified_count},
        info_type="stats",
        info_name="extraction_and_qualification_stats",
    )

    # Add a critical count check: at least 4 fully qualified
    evaluator.add_custom_node(
        result=(qualified_count >= 4),
        id="at_least_four_qualified",
        desc="At least four conferences meet all required criteria.",
        parent=collection_node,
        critical=True,
    )

    # Return structured summary
    return evaluator.get_summary()