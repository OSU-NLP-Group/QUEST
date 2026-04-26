import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
# from obj_task_eval.llm_client.base_client import LLMClient  # Optional typing


TASK_ID = "li_state_park_all_criteria"
TASK_DESCRIPTION = """
Identify a New York State Park located on Long Island that meets ALL of the following criteria: (1) Offers overnight camping facilities for visitors, (2) Provides beach access designated for swimming, (3) Has shower facilities available for campers and visitors, (4) Includes playground facilities, (5) Offers hiking trails or nature trails, (6) Provides fishing access or opportunities (shore, pier, etc.), (7) Has designated picnic areas equipped with picnic tables, (8) Has public restroom facilities, (9) Accommodates recreational vehicles (RVs) with dump station facilities, (10) Has a campground with substantial capacity (at least 100 campsites), (11) Provides organized parking fields or multiple designated recreation areas, and (12) All facilities must be available within the same state park property. Provide the name of the state park and reference URLs supporting that it meets each of these criteria.
"""


# ----------------------------- Data Models --------------------------------- #
class ParkEvidence(BaseModel):
    park_name: Optional[str] = None
    # General park URLs cited in the answer (e.g., the park's official page, overview page, etc.)
    park_urls: List[str] = Field(default_factory=list)

    # Criterion-specific URLs as cited or implied by the answer
    long_island_state_park_urls: List[str] = Field(default_factory=list)          # C1
    overnight_camping_urls: List[str] = Field(default_factory=list)               # C2
    swimming_beach_urls: List[str] = Field(default_factory=list)                  # C3
    showers_urls: List[str] = Field(default_factory=list)                         # C4
    playground_urls: List[str] = Field(default_factory=list)                      # C5
    hiking_trails_urls: List[str] = Field(default_factory=list)                   # C6
    fishing_urls: List[str] = Field(default_factory=list)                         # C7
    picnic_areas_urls: List[str] = Field(default_factory=list)                    # C8
    restrooms_urls: List[str] = Field(default_factory=list)                       # C9
    rv_dump_station_urls: List[str] = Field(default_factory=list)                 # C10
    campsite_capacity_urls: List[str] = Field(default_factory=list)               # C11
    parking_fields_or_multi_areas_urls: List[str] = Field(default_factory=list)   # C12


# --------------------------- Extraction Prompt ----------------------------- #
def prompt_extract_park_evidence() -> str:
    return """
    Extract the single New York State Park selected in the answer and all cited URLs that support each specified criterion.
    If multiple parks are mentioned, extract only the FIRST park that the answer claims meets all criteria.

    Return:
    - park_name: The exact park name as stated in the answer.
    - park_urls: All general URLs cited about the park (e.g., official park page or overview page).
    - long_island_state_park_urls: URLs cited to support that the park is a New York State Park on Long Island (or in the Long Island Region, or in Nassau/Suffolk County).
    - overnight_camping_urls: URLs cited to support overnight camping availability.
    - swimming_beach_urls: URLs cited to support designated swimming beach access.
    - showers_urls: URLs cited to support shower facilities.
    - playground_urls: URLs cited to support presence of playground(s).
    - hiking_trails_urls: URLs cited to support hiking/nature trails.
    - fishing_urls: URLs cited to support fishing access/opportunities.
    - picnic_areas_urls: URLs cited to support picnic areas with tables.
    - restrooms_urls: URLs cited to support public restrooms/comfort stations.
    - rv_dump_station_urls: URLs cited to support RV accommodations and a dump station.
    - campsite_capacity_urls: URLs cited to support that the campground has at least 100 campsites.
    - parking_fields_or_multi_areas_urls: URLs cited to support organized parking fields or multiple designated recreation areas.

    Extraction rules:
    - Only extract URLs explicitly present in the answer text. Do not invent URLs.
    - If the answer provides a general park URL that plausibly supports multiple criteria, include it in 'park_urls'. If the answer text ties a URL to a particular criterion, include it in that criterion's URL list.
    - If a criterion has no specifically tied URLs in the answer, leave that list empty.
    - Accept URLs written as plain links or markdown links. Return the actual URLs.
    """


# ----------------------------- Helper Utils -------------------------------- #
def _dedup_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _choose_sources(ev: ParkEvidence, attr: str) -> List[str]:
    """Prefer criterion-specific URLs; if empty, fall back to general park_urls."""
    specific = getattr(ev, attr, []) or []
    if specific:
        return _dedup_preserve_order(specific)
    return _dedup_preserve_order(ev.park_urls or [])


def _claim_texts(park_name: str) -> Dict[str, str]:
    """Build claims for each criterion."""
    pn = park_name or "the identified park"
    return {
        "criterion_1": f"'{pn}' is a New York State Park and is located on Long Island (or in the Long Island Region, i.e., Nassau or Suffolk County) in New York.",
        "criterion_2": f"'{pn}' offers overnight camping facilities.",
        "criterion_3": f"'{pn}' provides beach access designated for swimming.",
        "criterion_4": f"'{pn}' has shower facilities available for campers or visitors.",
        "criterion_5": f"'{pn}' includes playground facilities.",
        "criterion_6": f"'{pn}' offers hiking trails or nature trails.",
        "criterion_7": f"'{pn}' provides fishing access or fishing opportunities (shoreline, pier, surf, etc.).",
        "criterion_8": f"'{pn}' has designated picnic areas with picnic tables.",
        "criterion_9": f"'{pn}' has public restroom facilities (e.g., restrooms, comfort stations, bathrooms).",
        "criterion_10": f"'{pn}' accommodates recreational vehicles (RVs) and has a dump station within the park property.",
        "criterion_11": f"'{pn}' has a campground with at least 100 campsites (100 or more).",
        "criterion_12": f"'{pn}' provides organized parking fields or has multiple designated recreation/day-use areas within the park.",
    }


def _additional_instruction_for(criterion_id: str, park_name: str) -> str:
    pn = park_name or "the park"
    common = (
        f"All facilities must be available within the same state park property named '{pn}'. "
        "Do not count amenities from nearby but different parks, concessions, marinas, or private facilities unless the page clearly states they are part of the same park property. "
        "Only judge based on the provided webpage evidence."
    )
    specifics = {
        "criterion_1": (
            "Verify the page indicates the site is a New York State Park AND that it is on Long Island. "
            "Accept explicit references to 'Long Island', 'Long Island Region', or location within Nassau/Suffolk County as sufficient. "
            "Do not rely on your own knowledge beyond what the page states."
        ),
        "criterion_2": (
            "Look for 'overnight camping', 'campground', 'campsites', or similar. Day-use only does not qualify."
        ),
        "criterion_3": (
            "Look for explicit 'swimming' at a beach or designated swim area. A shoreline without designated swimming does not qualify."
        ),
        "criterion_4": (
            "Accept 'showers', 'bathhouse with showers', or 'comfort stations with showers'."
        ),
        "criterion_5": (
            "Accept 'playground' or 'children's play area' within the park."
        ),
        "criterion_6": (
            "Accept 'hiking', 'trails', 'nature trail', or similar wording."
        ),
        "criterion_7": (
            "Accept 'fishing', 'surf fishing', 'pier fishing', or other explicit fishing opportunities within the park."
        ),
        "criterion_8": (
            "Accept 'picnic area(s)' and/or mention of 'picnic tables'."
        ),
        "criterion_9": (
            "Accept 'restrooms', 'bathrooms', or 'comfort stations'."
        ),
        "criterion_10": (
            "Look for 'RV' references and an on-site 'dump station' (also called 'sanitary dump'). RVs permitted plus dump station must both be available on the property."
        ),
        "criterion_11": (
            "Verify that the number of campsites is at least 100. If multiple loops or sections are listed with counts, the page should make total count >= 100 clear."
        ),
        "criterion_12": (
            "Accept explicit 'Parking Field' designations (e.g., Parking Field 1, 2, etc.) or statements about multiple designated recreation/day-use areas."
        ),
    }
    return f"{common} {specifics.get(criterion_id, '')}".strip()


# --------------------------- Verification Builder -------------------------- #
async def build_and_verify_tree(evaluator: Evaluator, ev: ParkEvidence) -> None:
    root = evaluator.root  # Initialized already

    park_name = ev.park_name or "the identified park"

    # Create all 12 critical leaf nodes under root (parallel aggregation on root)
    descriptions = {
        "criterion_1": "Specified park is a New York State Park located on Long Island",
        "criterion_2": "Park offers overnight camping facilities",
        "criterion_3": "Park provides beach access for swimming",
        "criterion_4": "Park has shower facilities available",
        "criterion_5": "Park includes playground facilities",
        "criterion_6": "Park offers hiking trails or nature trails",
        "criterion_7": "Park provides fishing access or opportunities",
        "criterion_8": "Park has designated picnic areas with tables",
        "criterion_9": "Park has public restroom facilities",
        "criterion_10": "Park accommodates RVs with dump station facilities",
        "criterion_11": "Park has campground with substantial capacity (at least 100 campsites)",
        "criterion_12": "Park provides organized parking fields or multiple recreation areas",
    }

    # Mapping from criterion to evidence attribute name
    attr_map = {
        "criterion_1": "long_island_state_park_urls",
        "criterion_2": "overnight_camping_urls",
        "criterion_3": "swimming_beach_urls",
        "criterion_4": "showers_urls",
        "criterion_5": "playground_urls",
        "criterion_6": "hiking_trails_urls",
        "criterion_7": "fishing_urls",
        "criterion_8": "picnic_areas_urls",
        "criterion_9": "restrooms_urls",
        "criterion_10": "rv_dump_station_urls",
        "criterion_11": "campsite_capacity_urls",
        "criterion_12": "parking_fields_or_multi_areas_urls",
    }

    claims = _claim_texts(park_name)

    # Prepare nodes and batch verifications
    batch_items = []
    created_nodes = {}

    for crit_id, desc in descriptions.items():
        node = evaluator.add_leaf(
            id=crit_id,
            desc=desc,
            parent=root,
            critical=True,
        )
        created_nodes[crit_id] = node

        urls = _choose_sources(ev, attr_map[crit_id])

        # Enforce source-grounding: if no URLs at all, fail this leaf directly.
        if not urls:
            node.score = 0.0
            node.status = "failed"
            continue

        # Build claim and additional instruction
        claim = claims[crit_id]
        add_ins = _additional_instruction_for(crit_id, park_name)

        # Queue for batch verification (multi-URL verification)
        batch_items.append((claim, urls, node, add_ins))

    # Run batched verifications (parallelized)
    if batch_items:
        await evaluator.batch_verify(batch_items)

    # Optionally record a summary of URLs used
    url_summary: Dict[str, int] = {}
    for crit_id, attr in attr_map.items():
        url_summary[crit_id] = len(getattr(ev, attr, []) or [])
    evaluator.add_custom_info({"park_name": park_name, "url_counts_by_criterion": url_summary}, "extraction_summary")


# --------------------------- Main Entry Function --------------------------- #
async def evaluate_answer(
    client: Any,  # LLMClient,
    answer: str,
    agent_name: str,
    answer_name: str,
    cache: CacheFileSys,
    semaphore: asyncio.Semaphore,
    logger: logging.Logger,
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Long Island NY State Park criteria task.
    """
    # Initialize evaluator with a parallel root (12 independent critical checks)
    evaluator = Evaluator()
    evaluator.initialize(
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

    # Extract structured evidence from the answer
    ev: ParkEvidence = await evaluator.extract(
        prompt=prompt_extract_park_evidence(),
        template_class=ParkEvidence,
        extraction_name="park_evidence",
    )

    # Build verification leaves and verify against cited URLs
    await build_and_verify_tree(evaluator, ev)

    # Return summarized evaluation result
    return evaluator.get_summary()