import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.llm_client.base_client import LLMClient
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nasa_astronauts_500days_isscmd_eva"
TASK_DESCRIPTION = (
    "Identify at least three NASA astronauts who have each achieved all of the following career milestones: "
    "(1) accumulated at least 500 cumulative days in space across all missions, "
    "(2) served as International Space Station (ISS) commander on at least one expedition, and "
    "(3) completed at least one spacewalk (EVA). For each astronaut identified, provide their full name, "
    "their total cumulative days in space (rounded to the nearest whole day), "
    "the expedition number(s) during which they served as ISS commander, "
    "their total number of spacewalks and cumulative EVA time, "
    "and a reference URL that verifies these achievements."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class AstronautEntry(BaseModel):
    name: Optional[str] = None
    total_days_in_space: Optional[str] = None  # Keep as string to allow variants like "675" or "675 days"
    iss_commander_expeditions: List[str] = Field(default_factory=list)  # e.g., ["Expedition 16", "Expedition 51"]
    eva_count: Optional[str] = None  # Keep string to allow "10", "10 EVAs", etc.
    eva_time: Optional[str] = None   # e.g., "60 hours 21 minutes"
    reference_urls: List[str] = Field(default_factory=list)


class AstronautsExtraction(BaseModel):
    astronauts: List[AstronautEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_astronauts() -> str:
    return """
    Extract up to five astronaut entries from the answer, in the order they appear. For each entry, extract:
    - name: the person's full name as given in the answer.
    - total_days_in_space: the cumulative total days in space as stated in the answer (use the number as written; if a phrase like "675 days 3 hours" is given, include that string).
    - iss_commander_expeditions: a list of expedition labels (e.g., "Expedition 16", "Expedition 39/40", "Expedition 39 and 40") during which the person served as ISS commander, exactly as written in the answer.
    - eva_count: the total number of spacewalks/EVAs as written (e.g., "10", "10 EVAs").
    - eva_time: the cumulative EVA time as written (e.g., "60 hours 21 minutes").
    - reference_urls: all URLs provided in the answer that are meant to verify this astronaut’s NASA status and the listed milestones/stats.
    
    Rules:
    1) Do not invent or infer data; extract only what is explicitly present in the answer.
    2) If any field is missing for an astronaut, set it to null (or [] for lists).
    3) Only include up to five entries at most (ignore extras beyond five).
    4) URLs can appear in plain text or markdown links; extract actual URLs.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _is_valid_url(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.strip()
    return u.startswith("http://") or u.startswith("https://")


def _normalize_name(name: Optional[str]) -> str:
    if not name:
        return ""
    # Lowercase, strip, and remove periods/commas for looser matching
    cleaned = name.strip().lower().replace(".", "").replace(",", "")
    return " ".join(cleaned.split())


def _expeditions_str(exps: List[str]) -> str:
    if not exps:
        return ""
    return "; ".join(exps)


# --------------------------------------------------------------------------- #
# Verification per astronaut                                                  #
# --------------------------------------------------------------------------- #
async def verify_astronaut_entry(
    evaluator: Evaluator,
    parent_node,
    entry: AstronautEntry,
    idx_one_based: int,
) -> Any:
    """
    Build and verify the tree for a single astronaut entry.
    Returns the aggregator node for this astronaut.
    """
    # Aggregator node for this astronaut (parallel; per-leaf critical checks inside)
    a_node = evaluator.add_parallel(
        id=f"astronaut_{idx_one_based}",
        desc=f"Astronaut entry #{idx_one_based} meets all per-astronaut requirements and includes all required fields.",
        parent=parent_node,
        critical=False
    )

    name = entry.name or ""
    urls = [u for u in (entry.reference_urls or []) if _is_valid_url(u)]
    days_str = entry.total_days_in_space or ""
    expeditions_list = entry.iss_commander_expeditions or []
    expeditions_text = _expeditions_str(expeditions_list)
    eva_count = entry.eva_count or ""
    eva_time = entry.eva_time or ""

    # 1) Name provided (existence)
    evaluator.add_custom_node(
        result=bool(name.strip()),
        id=f"a{idx_one_based}_name",
        desc="Full name is provided.",
        parent=a_node,
        critical=True
    )

    # 2) Reference URL provided (existence + basic validity)
    evaluator.add_custom_node(
        result=len(urls) >= 1,
        id=f"a{idx_one_based}_reference_url",
        desc="At least one valid reference URL is provided that can verify the astronaut’s NASA status and the listed milestones/stats.",
        parent=a_node,
        critical=True
    )

    # Prepare verification leaves (all critical)
    # 3) NASA astronaut status (grounded by reference URLs)
    nasa_status_node = evaluator.add_leaf(
        id=f"a{idx_one_based}_nasa_status",
        desc="Person is a NASA astronaut (not an astronaut from another agency).",
        parent=a_node,
        critical=True
    )
    nasa_claim = f"{name} is a NASA astronaut."
    nasa_ins = (
        "Verify that the person is a NASA astronaut (current or former). "
        "Accept official NASA profiles, NASA releases, or other credible sources. "
        "If the person is primarily affiliated with another space agency (e.g., Roscosmos, ESA, JAXA), mark as incorrect."
    )

    # 4) Total cumulative days in space: provided, accurate (rounded), and ≥ 500 (grounded by reference URLs)
    days_node = evaluator.add_leaf(
        id=f"a{idx_one_based}_days_in_space",
        desc="Total cumulative days in space is provided, is accurate when rounded to the nearest whole day, and is ≥ 500.",
        parent=a_node,
        critical=True
    )
    days_claim = (
        f"{name} has a cumulative time in space of approximately {days_str} days (rounded to the nearest whole day), "
        f"which is at least 500 days in total."
    )
    days_ins = (
        "Use the provided sources to verify the astronaut’s cumulative time in space and confirm it is ≥ 500 days. "
        "If the source lists detailed durations (e.g., days, hours, minutes), allow rounding to the nearest whole day. "
        "Small rounding differences (±1 day) are acceptable if clearly due to rounding. "
        "If the answer's provided number is clearly inconsistent with authoritative totals, mark as incorrect."
    )

    # 5) ISS commander expedition(s): provided, accurate, at least one (grounded by reference URLs)
    iss_cmd_node = evaluator.add_leaf(
        id=f"a{idx_one_based}_iss_commander",
        desc="ISS expedition number(s) during which they served as ISS commander are provided, accurate, and include at least one expedition.",
        parent=a_node,
        critical=True
    )
    iss_claim = (
        f"{name} served as commander of the International Space Station during the following expedition(s): "
        f"{expeditions_text}. There is at least one such commanded expedition."
    )
    iss_ins = (
        "Verify that the person served as ISS commander for at least one expedition. "
        "Allow reasonable formatting variants (e.g., 'Expedition 39/40' vs 'Expedition 39 and 40'). "
        "If no expedition is listed in the answer, consider the claim incorrect."
    )

    # 6) EVA stats: provided (count and time), accurate; count ≥ 1 (grounded by reference URLs)
    eva_node = evaluator.add_leaf(
        id=f"a{idx_one_based}_eva_stats",
        desc="Total number of EVAs and cumulative EVA time are provided and accurate; EVA count is ≥ 1.",
        parent=a_node,
        critical=True
    )
    eva_claim = (
        f"{name} has performed {eva_count} spacewalk(s) (EVAs) with a cumulative EVA time of {eva_time}. "
        f"This includes at least one EVA."
    )
    eva_ins = (
        "Verify both the EVA count and the total EVA time from the provided sources. "
        "Allow minor format differences in time reporting (e.g., 'hours:minutes' vs 'hrs mins'). "
        "If the count is missing, zero, or contradicted by sources, mark as incorrect."
    )

    # Perform verifications; leverage multi-URL verification
    claims_and_sources = [
        (nasa_claim, urls, nasa_status_node, nasa_ins),
        (days_claim, urls, days_node, days_ins),
        (iss_claim, urls, iss_cmd_node, iss_ins),
        (eva_claim, urls, eva_node, eva_ins),
    ]
    await evaluator.batch_verify(claims_and_sources)

    return a_node


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
    model: str = "o4-mini"
) -> Dict:
    """
    Evaluate an answer for the NASA astronauts with ≥500 days, ISS commander, and ≥1 EVA task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Use parallel at root to avoid unintended short-circuiting
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

    # Extract astronaut entries
    extracted = await evaluator.extract(
        prompt=prompt_extract_astronauts(),
        template_class=AstronautsExtraction,
        extraction_name="astronauts_extraction"
    )

    # Add a container node for astronaut entries
    entries_node = evaluator.add_parallel(
        id="astronaut_entries",
        desc="Evaluate provided astronaut entries (score up to 5 entries; any additional entries beyond 5 are not considered for scoring).",
        parent=root,
        critical=False
    )

    # Use only the first 5 entries; pad with placeholders if fewer provided
    entries = (extracted.astronauts or [])[:5]
    while len(entries) < 5:
        entries.append(AstronautEntry())

    astronaut_nodes = []
    for i in range(5):
        node = await verify_astronaut_entry(evaluator, entries_node, entries[i], i + 1)
        astronaut_nodes.append(node)

    # Compute how many DISTINCT astronauts fully satisfy all per-astronaut critical checks
    # A per-astronaut node passes only if all its critical children pass (aggregated_score == 1.0)
    qualifying_names: List[str] = []
    seen_norm: set = set()
    for i, a_node in enumerate(astronaut_nodes):
        try:
            score = a_node.aggregated_score  # triggers computation if needed
        except Exception:
            score = 0.0
        if score == 1.0:
            nm = entries[i].name or ""
            nm_norm = _normalize_name(nm)
            if nm and nm_norm and nm_norm not in seen_norm:
                seen_norm.add(nm_norm)
                qualifying_names.append(nm)

    # Final critical requirement: at least three DISTINCT qualifying astronauts
    min_required = 3
    result_three = len(qualifying_names) >= min_required
    evaluator.add_custom_node(
        result=result_three,
        id="at_least_three_distinct_qualifying_astronauts",
        desc="Among the evaluated entries (astronaut_1..astronaut_5 that are present), at least three DISTINCT individuals fully satisfy all per-astronaut critical checks.",
        parent=root,
        critical=True
    )

    # Record helpful info
    evaluator.add_custom_info(
        info={
            "distinct_qualified_count": len(qualifying_names),
            "qualified_names": qualifying_names
        },
        info_type="summary",
        info_name="qualified_astronauts_summary"
    )

    return evaluator.get_summary()