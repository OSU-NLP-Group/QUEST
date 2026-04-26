import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "olympics_xc_2026_6gold_tds5"
TASK_DESCRIPTION = """
Who is the Norwegian cross-country skier who became the first Winter Olympian to win 6 gold medals in a single Olympic Games at the 2026 Milan Cortina Olympics, achieving a perfect 6-for-6 record by winning all events entered, and who also holds the record for the most Tour de Ski overall titles with 5 victories?
"""


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class AthleteExtraction(BaseModel):
    """Structured extraction of the athlete answer and source URLs."""
    name: Optional[str] = None
    nationality: Optional[str] = None
    sport: Optional[str] = None

    # All URLs mentioned in the answer (generic sources section or inline)
    sources: List[str] = Field(default_factory=list)

    # URL groups (if the answer attributes per-claim sources)
    urls_olympics2026: List[str] = Field(default_factory=list)          # Pages about 2026 results/records
    urls_tour_de_ski: List[str] = Field(default_factory=list)           # Pages about Tour de Ski titles/records
    urls_olympic_total: List[str] = Field(default_factory=list)         # Pages about total Olympic gold count
    urls_world_cup: List[str] = Field(default_factory=list)             # Pages about total World Cup victories


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_athlete() -> str:
    return """
    Extract the following information from the answer text about the athlete that matches the described achievements.

    Required fields:
    - name: The athlete's full name as written in the answer.
    - nationality: The nationality stated in the answer (e.g., "Norway", "Norwegian"). If unspecified, return null.
    - sport: The sport/discipline stated in the answer (e.g., "cross-country skiing"). If unspecified, return null.

    URL fields:
    - sources: All URLs mentioned anywhere in the answer (including plain links and markdown links).
    - urls_olympics2026: URLs that specifically support the athlete winning 6 gold medals at the 2026 Milan Cortina Olympics and/or being the first to reach 6 golds at a single Winter Olympics, or the perfect 6-for-6 record.
    - urls_tour_de_ski: URLs that specifically support the athlete having 5 Tour de Ski overall titles (the all-time record).
    - urls_olympic_total: URLs that specifically support the athlete's career total Olympic gold medals (e.g., 11 in total, if claimed).
    - urls_world_cup: URLs that specifically support the athlete having over 100 World Cup victories (if claimed).

    Rules:
    - Return only URLs explicitly present in the answer text. Do not add or infer any URL.
    - If a specific URL group is not present in the answer, return an empty array for that group.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
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


def _merge_sources(*groups: List[str]) -> List[str]:
    combined: List[str] = []
    for g in groups:
        combined.extend(g or [])
    return _dedup_preserve_order(combined)


def _fallback_sources(primary: List[str], fallback: List[str]) -> List[str]:
    primary = _dedup_preserve_order(primary or [])
    if primary:
        return primary
    return _dedup_preserve_order(fallback or [])


# --------------------------------------------------------------------------- #
# Build verification nodes and batch verify                                   #
# --------------------------------------------------------------------------- #
async def build_and_verify_athlete_nodes(evaluator: Evaluator, parent_node, data: AthleteExtraction) -> None:
    """
    Build the verification tree under 'athlete_identification' and run evidence-based checks.
    """
    athlete_node = evaluator.add_parallel(
        id="athlete_identification",
        desc="Correctly identifies the athlete meeting all specified criteria",
        parent=parent_node,
        critical=False
    )

    # Prepare common fields
    name = data.name or "the athlete"
    all_sources = _merge_sources(
        data.sources,
        data.urls_olympics2026,
        data.urls_tour_de_ski,
        data.urls_olympic_total,
        data.urls_world_cup,
    )

    # Create leaf nodes (critical and non-critical as per rubric)
    node_nat = evaluator.add_leaf(
        id="nationality_verification",
        desc="The athlete is from Norway",
        parent=athlete_node,
        critical=True
    )

    node_sport = evaluator.add_leaf(
        id="sport_discipline_verification",
        desc="The athlete competes in cross-country skiing",
        parent=athlete_node,
        critical=True
    )

    node_6gold_2026 = evaluator.add_leaf(
        id="single_olympics_gold_count",
        desc="The athlete won 6 gold medals at the 2026 Milan Cortina Winter Olympics",
        parent=athlete_node,
        critical=True
    )

    node_first_ever_6 = evaluator.add_leaf(
        id="historic_first_achievement",
        desc="The athlete was the first Winter Olympian to win 6 gold medals in a single Olympic Games",
        parent=athlete_node,
        critical=True
    )

    node_perfect_6for6 = evaluator.add_leaf(
        id="perfect_event_record",
        desc="The athlete won gold in all 6 cross-country skiing events entered at the 2026 Olympics",
        parent=athlete_node,
        critical=True
    )

    node_tds_record = evaluator.add_leaf(
        id="tour_de_ski_record",
        desc="The athlete has won the Tour de Ski overall title 5 times, holding the record",
        parent=athlete_node,
        critical=True
    )

    node_total_olympic = evaluator.add_leaf(
        id="career_olympic_gold_total",
        desc="The athlete has won 11 total Olympic gold medals across their career",
        parent=athlete_node,
        critical=False
    )

    node_world_cup_100 = evaluator.add_leaf(
        id="world_cup_victories",
        desc="The athlete has achieved over 100 World Cup victories in their career",
        parent=athlete_node,
        critical=False
    )

    # Build claims and sources
    claims_and_sources = []

    claims_and_sources.append((
        f"{name} is from Norway.",
        _fallback_sources(all_sources, all_sources),
        node_nat,
        "Confirm the athlete's nationality. Treat 'Norwegian' as equivalent to 'from Norway'."
    ))

    claims_and_sources.append((
        f"{name} competes in cross-country skiing.",
        _fallback_sources(all_sources, all_sources),
        node_sport,
        "Allow reasonable synonyms such as 'cross country skiing', 'XC skiing', or 'Nordic cross-country'."
    ))

    claims_and_sources.append((
        f"{name} won 6 gold medals at the 2026 Milan Cortina Winter Olympics.",
        _fallback_sources(data.urls_olympics2026, all_sources),
        node_6gold_2026,
        "Verify that the 2026 Milan Cortina results show exactly six gold medals for this athlete."
    ))

    claims_and_sources.append((
        f"{name} was the first Winter Olympian to win 6 gold medals in a single Olympic Games.",
        _fallback_sources(data.urls_olympics2026, all_sources),
        node_first_ever_6,
        "Check explicit wording like 'first ever', 'first Winter Olympian', or equivalent statements tied to six golds at one Winter Olympics."
    ))

    claims_and_sources.append((
        f"{name} won gold in all 6 cross-country skiing events they entered at the 2026 Olympics, achieving a perfect 6-for-6 record.",
        _fallback_sources(data.urls_olympics2026, all_sources),
        node_perfect_6for6,
        "Look for evidence that every entered cross-country event in 2026 was won (6/6)."
    ))

    claims_and_sources.append((
        f"{name} has won the Tour de Ski overall title 5 times, holding the all-time record.",
        _fallback_sources(data.urls_tour_de_ski, all_sources),
        node_tds_record,
        "Confirm both (a) five overall Tour de Ski titles and (b) that this is the record/highest number."
    ))

    claims_and_sources.append((
        f"{name} has won 11 total Olympic gold medals across their career.",
        _fallback_sources(data.urls_olympic_total, all_sources),
        node_total_olympic,
        "Verify the total Olympic gold count across all editions sums to 11."
    ))

    claims_and_sources.append((
        f"{name} has over 100 World Cup victories in their career.",
        _fallback_sources(data.urls_world_cup, all_sources),
        node_world_cup_100,
        "This refers to FIS World Cup event wins across the athlete's career. 'Over 100' includes any number >= 101."
    ))

    # Run all verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


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
    Evaluate an answer for the 2026 6-gold cross-country skier identification task.
    """
    # Initialize evaluator (root node is non-critical by default)
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

    # Extract athlete and sources
    athlete_info = await evaluator.extract(
        prompt=prompt_extract_athlete(),
        template_class=AthleteExtraction,
        extraction_name="athlete_extraction"
    )

    # Build verification subtree and run checks
    await build_and_verify_athlete_nodes(evaluator, root, athlete_info)

    # Return structured summary
    return evaluator.get_summary()