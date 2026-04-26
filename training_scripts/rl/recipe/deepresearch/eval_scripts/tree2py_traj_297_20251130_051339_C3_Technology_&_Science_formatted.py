import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


TASK_ID = "apple_ipad_mini_m_series_progression"
TASK_DESCRIPTION = (
    "In October 2024, Apple announced a new generation of iPad mini that uses a chip originally introduced in Apple's "
    "iPhone Pro lineup. Identify this iPad mini model and provide the exact name of its chip. Then, find the Apple "
    "M-series chip that was announced exactly one year later on the same calendar date in 2025. Determine which "
    "generation this M-series chip represents in the Apple silicon M-series progression (where M1 is the first "
    "generation). List one of the products that was announced with this M-series chip at its launch event. Finally, "
    "return to the October 2024 iPad mini's chip and state how many performance CPU cores it has."
)


class PhaseAExtraction(BaseModel):
    ipad_mini_model: Optional[str] = None
    ipad_mini_announcement_date: Optional[str] = None
    ipad_mini_model_sources: List[str] = Field(default_factory=list)

    chip_name: Optional[str] = None
    chip_sources: List[str] = Field(default_factory=list)

    performance_core_count: Optional[str] = None
    performance_core_sources: List[str] = Field(default_factory=list)

    iphone_pro_origin_sources: List[str] = Field(default_factory=list)


class PhaseBExtraction(BaseModel):
    m_series_chip_name: Optional[str] = None
    m_series_announcement_date: Optional[str] = None
    m_series_generation_number: Optional[str] = None
    m_series_chip_sources: List[str] = Field(default_factory=list)

    m_series_launch_product: Optional[str] = None
    m_series_launch_sources: List[str] = Field(default_factory=list)


def prompt_extract_phase_a() -> str:
    return (
        "From the answer, extract details about the iPad mini announcement in October 2024 and its chip info:\n"
        "Required fields:\n"
        "1) ipad_mini_model: The exact iPad mini model name associated with the October 2024 announcement.\n"
        "2) ipad_mini_announcement_date: The calendar date of the announcement in the answer (e.g., 'October 15, 2024').\n"
        "3) ipad_mini_model_sources: All URLs cited that support the model identification and announcement details.\n"
        "4) chip_name: The exact official Apple designation of the chip used in that iPad mini.\n"
        "5) chip_sources: All URLs cited that specifically support the chip name used in the iPad mini.\n"
        "6) performance_core_count: The number of performance CPU cores in that chip (performance cores only).\n"
        "7) performance_core_sources: All URLs cited that support the performance core count.\n"
        "8) iphone_pro_origin_sources: URLs cited that support the statement that the chip was originally introduced in Apple's iPhone Pro lineup.\n\n"
        "Rules:\n"
        "- Extract only information explicitly present in the answer.\n"
        "- If any item is missing, set it to null (or empty list for URLs).\n"
        "- For URLs, extract actual URLs shown in the answer (including markdown link targets)."
    )


def prompt_extract_phase_b() -> str:
    return (
        "From the answer, extract details about the Apple M-series chip announced exactly one year later on the same "
        "calendar date in 2025 (relative to the October 2024 iPad mini announcement):\n"
        "Required fields:\n"
        "1) m_series_chip_name: The M-series chip name (e.g., 'M3', 'M4 Pro').\n"
        "2) m_series_announcement_date: The calendar date of the M-series chip announcement in 2025.\n"
        "3) m_series_generation_number: The generation index counting M1 as 1st, M2 as 2nd, etc. (e.g., '3' for M3).\n"
        "4) m_series_chip_sources: URLs cited that support the chip identification and announcement.\n"
        "5) m_series_launch_product: One product announced with this M-series chip at its launch event.\n"
        "6) m_series_launch_sources: URLs cited that support the launch product association with the chip.\n\n"
        "Rules:\n"
        "- Extract only information explicitly present in the answer.\n"
        "- If any item is missing, set it to null (or empty list for URLs).\n"
        "- For URLs, extract actual URLs shown in the answer (including markdown link targets)."
    )


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not isinstance(u, str):
            continue
        su = u.strip()
        if su and su not in seen:
            seen.add(su)
            out.append(su)
    return out


def _merge_sources(*lists: List[str]) -> Optional[List[str]]:
    merged: List[str] = []
    for lst in lists:
        merged.extend(lst or [])
    merged = _dedupe_urls(merged)
    return merged if merged else None


async def _build_phase_a(
    evaluator: Evaluator,
    root_node,
    a: PhaseAExtraction,
) -> None:
    phase_a_node = evaluator.add_sequential(
        id="Phase_A_Oct_2024_iPad_mini",
        desc="Identify the October 2024 iPad mini and provide required chip details.",
        parent=root_node,
        critical=True,
    )

    # A1: iPad mini model identification (critical leaf)
    a1_node = evaluator.add_leaf(
        id="A1_iPad_mini_Model",
        desc="Identifies the iPad mini model that was announced in October 2024 (the model must be the one from that announcement).",
        parent=phase_a_node,
        critical=True,
    )
    a1_claim_parts = []
    if a.ipad_mini_model:
        a1_claim_parts.append(f"The iPad mini model announced in October 2024 is '{a.ipad_mini_model}'.")
    else:
        a1_claim_parts.append("The answer claims to identify the iPad mini model announced in October 2024.")
    if a.ipad_mini_announcement_date:
        a1_claim_parts.append(f"The announcement date provided is '{a.ipad_mini_announcement_date}'.")
    a1_claim = " ".join(a1_claim_parts)
    a1_sources = _merge_sources(a.ipad_mini_model_sources, a.chip_sources)
    await evaluator.verify(
        claim=a1_claim,
        node=a1_node,
        sources=a1_sources,
        additional_instruction=(
            "Confirm that the specified iPad mini model is indeed associated with an Apple announcement in October 2024. "
            "Use official Apple newsroom or product pages if available. Allow minor naming variations like inclusion of '(7th generation)'."
        ),
    )

    # A2: Chip and CPU specs (parallel critical group)
    a2_node = evaluator.add_parallel(
        id="A2_iPad_mini_Chip_and_CPU_Specs",
        desc="Provides the required chip information for the identified October 2024 iPad mini.",
        parent=phase_a_node,
        critical=True,
    )

    # A2a: Chip name (official)
    a2a_node = evaluator.add_leaf(
        id="A2a_Chip_Name_Official",
        desc="Provides the exact official Apple designation of the chip used in the October 2024 iPad mini.",
        parent=a2_node,
        critical=True,
    )
    a2a_claim = (
        f"The exact official Apple designation of the chip used in the October 2024 iPad mini is '{a.chip_name}'."
        if a.chip_name else "The answer provides an official Apple chip name for the October 2024 iPad mini."
    )
    await evaluator.verify(
        claim=a2a_claim,
        node=a2a_node,
        sources=(a.chip_sources if a.chip_sources else None),
        additional_instruction=(
            "Verify the chip name against official Apple sources (newsroom or tech specs) or credible coverage. "
            "Accept minor formatting differences (e.g., 'A18 Pro' vs 'Apple A18 Pro')."
        ),
    )

    # A2b: Chip origin from iPhone Pro lineup
    a2b_node = evaluator.add_leaf(
        id="A2b_Chip_Origin_iPhone_Pro_Lineup",
        desc="The chip used in the October 2024 iPad mini is one originally introduced in Apple's iPhone Pro lineup.",
        parent=a2_node,
        critical=True,
    )
    chip_for_origin = a.chip_name or "the chip used in that iPad mini"
    a2b_claim = (
        f"The chip '{chip_for_origin}' was originally introduced in Apple's iPhone Pro lineup."
    )
    a2b_sources = (a.iphone_pro_origin_sources if a.iphone_pro_origin_sources else None)
    await evaluator.verify(
        claim=a2b_claim,
        node=a2b_node,
        sources=a2b_sources,
        additional_instruction=(
            "Check iPhone Pro announcement pages or official Apple newsroom articles to confirm this chip's debut was in an iPhone Pro lineup. "
            "Examples: 'A17 Pro' introduced with iPhone 15 Pro in 2023; 'A18 Pro' introduced with iPhone 16 Pro in 2024."
        ),
    )

    # A2c: Performance CPU core count
    a2c_node = evaluator.add_leaf(
        id="A2c_Performance_CPU_Core_Count",
        desc="States the number of performance CPU cores (performance cores only, not total CPU cores) in the October 2024 iPad mini's chip.",
        parent=a2_node,
        critical=True,
    )
    core_count_text = a.performance_core_count or "an unspecified number of"
    a2c_claim = (
        f"The chip '{chip_for_origin}' has {core_count_text} performance CPU cores (performance cores only)."
    )
    a2c_sources = (a.performance_core_sources if a.performance_core_sources else None)
    await evaluator.verify(
        claim=a2c_claim,
        node=a2c_node,
        sources=a2c_sources,
        additional_instruction=(
            "Only count performance cores (p-cores), not efficiency cores. If a page lists total cores and the split, use the split to confirm p-cores. "
            "Accept reasonable numeric formatting (e.g., '2', 'two')."
        ),
    )


async def _build_phase_b(
    evaluator: Evaluator,
    root_node,
    a: PhaseAExtraction,
    b: PhaseBExtraction,
) -> None:
    phase_b_node = evaluator.add_sequential(
        id="Phase_B_M_series_Chip_2025_Same_Date",
        desc="Identify the M-series chip announced exactly one year later on the same calendar date in 2025, then provide its generation number and a launch product.",
        parent=root_node,
        critical=True,
    )

    # B1: Identification of the M-series chip (exactly one year later, same calendar date)
    b1_node = evaluator.add_leaf(
        id="B1_M_series_Chip_Identification",
        desc="Identifies the Apple M-series chip that was announced exactly one year after the October 2024 iPad mini announcement on the same calendar date in 2025.",
        parent=phase_b_node,
        critical=True,
    )
    mini_date = a.ipad_mini_announcement_date or "the iPad mini's announcement date in October 2024"
    m_chip = b.m_series_chip_name or "the specified M-series chip"
    m_date = b.m_series_announcement_date or "the M-series chip's announcement date in 2025"
    b1_claim = (
        f"Apple announced {m_chip} on {m_date}. The iPad mini was announced on {mini_date} in October 2024. "
        f"These two announcements occurred on the same calendar date exactly one year apart."
    )
    b1_sources = _merge_sources(b.m_series_chip_sources, a.ipad_mini_model_sources, a.chip_sources)
    await evaluator.verify(
        claim=b1_claim,
        node=b1_node,
        sources=b1_sources,
        additional_instruction=(
            "Confirm that the M-series chip announcement date in 2025 matches the same month/day as the iPad mini "
            "announcement date in October 2024, one year later. Use official Apple newsroom pages or credible reports."
        ),
    )

    # B2: Attributes (parallel)
    b2_node = evaluator.add_parallel(
        id="B2_M_series_Chip_Attributes",
        desc="Provides required attributes about the identified M-series chip.",
        parent=phase_b_node,
        critical=True,
    )

    # B2a: Generation number
    b2a_node = evaluator.add_leaf(
        id="B2a_M_series_Generation_Number",
        desc="Correctly determines which generation the identified M-series chip is in the Apple silicon M-series progression, counting M1 as the 1st generation.",
        parent=b2_node,
        critical=True,
    )
    gen_text = b.m_series_generation_number or "an unspecified generation number"
    b2a_claim = (
        f"Counting M1 as the 1st generation, the chip '{m_chip}' represents generation {gen_text} in Apple's M-series progression. "
        "Sub-variants like Pro/Max/Ultra share the same base generation number as the underlying M-series (e.g., M3 Pro is 3rd generation)."
    )
    await evaluator.verify(
        claim=b2a_claim,
        node=b2a_node,
        sources=None,
        additional_instruction=(
            "Apply the naming rule: M1→1st, M2→2nd, M3→3rd, M4→4th, etc. Variants (Pro/Max/Ultra) keep the same generation index as Mx."
        ),
    )

    # B2b: Launch event product
    b2b_node = evaluator.add_leaf(
        id="B2b_Launch_Event_Product",
        desc="Lists one product that was officially announced with this M-series chip at its launch event.",
        parent=b2_node,
        critical=True,
    )
    launch_product_text = b.m_series_launch_product or "a product associated with this chip"
    b2b_claim = (
        f"One product announced with the {m_chip} at its launch event is '{launch_product_text}'."
    )
    await evaluator.verify(
        claim=b2b_claim,
        node=b2b_node,
        sources=(b.m_series_launch_sources if b.m_series_launch_sources else None),
        additional_instruction=(
            "Verify via Apple newsroom/event pages or official product pages that the specified product was introduced "
            "alongside the M-series chip at its launch event. Allow minor naming variations."
        ),
    )


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

    # Extract Phase A and Phase B information
    phase_a = await evaluator.extract(
        prompt=prompt_extract_phase_a(),
        template_class=PhaseAExtraction,
        extraction_name="phase_a_oct_2024_ipad_mini",
    )
    phase_b = await evaluator.extract(
        prompt=prompt_extract_phase_b(),
        template_class=PhaseBExtraction,
        extraction_name="phase_b_m_series_same_date_2025",
    )

    # Build verification tree according to rubric
    await _build_phase_a(evaluator, root, phase_a)
    await _build_phase_b(evaluator, root, phase_a, phase_b)

    return evaluator.get_summary()