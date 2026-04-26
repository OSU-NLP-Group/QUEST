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
TASK_ID = "morocco_travel_vaccines_and_weightloss_timeline"
TASK_DESCRIPTION = (
    "You are planning to travel to Morocco in mid-2026 and want to ensure you have all necessary vaccinations. "
    "Additionally, you are interested in starting treatment with a next-generation weight loss medication and want "
    "to know which one will be available (FDA-approved) soonest. Based on current CDC recommendations and regulatory timelines:\n\n"
    "1. Identify the CDC-recommended vaccines for travelers to Morocco, specifically addressing:\n"
    "   - Hepatitis A vaccination recommendation\n"
    "   - Typhoid vaccination recommendation\n"
    "   - Measles/MMR vaccination guidance\n\n"
    "2. Among the next-generation weight loss drugs currently in development (retatrutide by Eli Lilly, CagriSema by Novo Nordisk, "
    "and orforglipron by Eli Lilly), identify which one has the earliest expected FDA approval date based on available regulatory "
    "submission and approval timeline information, and provide that expected approval timeframe."
)

ROOT_NODE_DESC = (
    "Provides complete information about Morocco vaccination requirements and next-generation weight loss drug approval timelines"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VaccineInfo(BaseModel):
    statement: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class VaccinationExtraction(BaseModel):
    hepatitis_a: Optional[VaccineInfo] = None
    typhoid: Optional[VaccineInfo] = None
    mmr: Optional[VaccineInfo] = None


class DrugInfo(BaseModel):
    name: Optional[str] = None
    expected_approval_timeframe: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class WeightLossTimelineExtraction(BaseModel):
    earliest_drug: Optional[str] = None
    earliest_timeframe: Optional[str] = None
    earliest_sources: List[str] = Field(default_factory=list)
    retatrutide: Optional[DrugInfo] = None
    cagrisema: Optional[DrugInfo] = None
    orforglipron: Optional[DrugInfo] = None


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_vaccines() -> str:
    return (
        "Extract from the answer the CDC-recommended vaccine guidance for travelers to Morocco. "
        "For each of the following items, extract:\n"
        "1) statement: The exact text in the answer describing the recommendation/guidance.\n"
        "2) sources: All URLs cited in the answer that support the recommendation (e.g., CDC pages). Extract only URLs actually present in the answer.\n"
        "Items:\n"
        "- hepatitis_a (Hepatitis A vaccination recommendation for unvaccinated travelers aged 1 year or older going to Morocco)\n"
        "- typhoid (Typhoid vaccination recommendation for most travelers, especially those staying with friends/relatives or visiting smaller cities or rural areas)\n"
        "- mmr (Measles/MMR vaccination emphasis for all international travelers due to global rise in measles)\n"
        "Return a JSON object with keys: hepatitis_a, typhoid, mmr. Each value is an object with 'statement' and 'sources' (array of URLs). "
        "If a statement is missing, set it to null. If no sources are cited for an item, return an empty array for 'sources'."
    )


def prompt_extract_drug_timelines() -> str:
    return (
        "Extract from the answer the expected FDA approval timeline information regarding the following next-generation weight loss drugs:\n"
        "- retatrutide (Eli Lilly)\n"
        "- cagrisema (Novo Nordisk)\n"
        "- orforglipron (Eli Lilly)\n\n"
        "For each drug, extract:\n"
        "• name: The drug name as used in the answer\n"
        "• expected_approval_timeframe: The expected FDA approval timeframe mentioned in the answer (e.g., 'mid-2026', 'Q4 2025', 'H1 2026'). Use the exact phrasing from the answer.\n"
        "• sources: All URLs cited in the answer that support this timeline (e.g., press releases, pipeline updates, regulatory news). Extract only URLs present in the answer.\n\n"
        "Also extract the answer's identification of the single earliest expected FDA approval among the three, and its timeframe:\n"
        "• earliest_drug: Which of the three the answer states will be FDA-approved first (earliest)\n"
        "• earliest_timeframe: The expected approval timeframe for that earliest drug, as stated in the answer\n"
        "• earliest_sources: All URLs the answer cites specifically for the earliest drug/timeframe claim\n\n"
        "Return a JSON object with keys: earliest_drug, earliest_timeframe, earliest_sources, retatrutide, cagrisema, orforglipron. "
        "Each drug key maps to an object with fields name, expected_approval_timeframe, sources. "
        "If any field is missing, set it to null (or empty array for sources)."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_sources(info: Optional[VaccineInfo]) -> List[str]:
    return info.sources if info and info.sources else []


def _normalize_drug_name(name: Optional[str]) -> str:
    if not name:
        return ""
    return name.strip().lower()


def _get_drug_info_by_name(drug_name: Optional[str], timelines: WeightLossTimelineExtraction) -> Optional[DrugInfo]:
    name_norm = _normalize_drug_name(drug_name)
    if name_norm in ("retatrutide",):
        return timelines.retatrutide
    if name_norm in ("cagrisema", "cagrisema"):
        return timelines.cagrisema
    if name_norm in ("orforglipron",):
        return timelines.orforglipron
    return None


def _compose_earliest_comparison_instruction(timelines: WeightLossTimelineExtraction) -> str:
    ret = timelines.retatrutide.expected_approval_timeframe if timelines.retatrutide else None
    cag = timelines.cagrisema.expected_approval_timeframe if timelines.cagrisema else None
    orf = timelines.orforglipron.expected_approval_timeframe if timelines.orforglipron else None

    parts = []
    parts.append(f"Retatrutide timeframe: {ret or 'unknown'}")
    parts.append(f"CagriSema timeframe: {cag or 'unknown'}")
    parts.append(f"Orforglipron timeframe: {orf or 'unknown'}")
    return (
        "Using the answer's stated timelines, determine which is earliest. "
        "Allow quarter/half-year and 'early/mid/late' approximations. "
        "Treat year-only ranges as approximate. "
        "Here are the timelines extracted from the answer:\n"
        + "\n".join(parts)
    )


def _compose_timeline_support_instruction(drug: str) -> str:
    return (
        "Verify whether the cited sources explicitly support or clearly imply the expected FDA approval timeframe for "
        f"{drug}. Allow reasonable approximations (e.g., Q2 vs H1; 'mid' vs 'Q2'). "
        "If sources discuss regulatory submissions (e.g., NDA/BLA filing) and expected decision windows, treat those as supporting the timeframe."
    )


# --------------------------------------------------------------------------- #
# Verification functions                                                      #
# --------------------------------------------------------------------------- #
async def verify_vaccination_requirements(
    evaluator: Evaluator,
    parent_node,
    vaccines: VaccinationExtraction,
) -> None:
    vacc_node = evaluator.add_parallel(
        id="Morocco_Vaccination_Requirements",
        desc="Correctly identifies CDC-recommended vaccines for Morocco travel",
        parent=parent_node,
        critical=True
    )

    hepA_node = evaluator.add_leaf(
        id="Hepatitis_A_Recommendation",
        desc="States that Hepatitis A vaccine is recommended for unvaccinated travelers aged 1 year or older going to Morocco",
        parent=vacc_node,
        critical=True
    )
    typhoid_node = evaluator.add_leaf(
        id="Typhoid_Recommendation",
        desc="States that Typhoid vaccine is recommended for most travelers to Morocco, especially those staying with friends/relatives or visiting smaller cities or rural areas",
        parent=vacc_node,
        critical=True
    )
    mmr_node = evaluator.add_leaf(
        id="Measles_MMR_Special_Emphasis",
        desc="Notes that measles/MMR vaccination is specifically emphasized for all international travelers due to global rise in measles cases",
        parent=vacc_node,
        critical=True
    )

    hepA_claim = "The CDC recommends Hepatitis A vaccination for unvaccinated travelers aged 1 year or older going to Morocco."
    typhoid_claim = (
        "The CDC recommends Typhoid vaccination for most travelers to Morocco, especially those staying with friends or relatives "
        "or visiting smaller cities or rural areas."
    )
    mmr_claim = (
        "The CDC specifically emphasizes measles/MMR vaccination for all international travelers due to the global rise in measles cases."
    )

    claims_and_sources = [
        (
            hepA_claim,
            _safe_sources(vaccines.hepatitis_a),
            hepA_node,
            "Use CDC Traveler's Health page(s) for Morocco if provided; allow minor wording variations."
        ),
        (
            typhoid_claim,
            _safe_sources(vaccines.typhoid),
            typhoid_node,
            "Use CDC Morocco travel guidance sources if provided; allow minor paraphrasing."
        ),
        (
            mmr_claim,
            _safe_sources(vaccines.mmr),
            mmr_node,
            "Use CDC measles travel vaccination guidance if provided; the emphasis applies to all international travelers."
        ),
    ]

    await evaluator.batch_verify(claims_and_sources)


async def verify_drug_timeline(
    evaluator: Evaluator,
    parent_node,
    timelines: WeightLossTimelineExtraction,
) -> None:
    drug_node = evaluator.add_sequential(
        id="Next_Generation_Weight_Loss_Drug_Timeline",
        desc="Correctly identifies which next-generation weight loss drug has the earliest expected FDA approval and provides the expected approval timeframe",
        parent=parent_node,
        critical=True
    )

    # Leaf 1: Earliest drug identification
    earliest_id_node = evaluator.add_leaf(
        id="Earliest_Approval_Drug_Identification",
        desc="Identifies the specific next-generation weight loss drug (among retatrutide, CagriSema, and orforglipron) that has the earliest expected FDA approval date based on stated regulatory timelines",
        parent=drug_node,
        critical=True
    )

    earliest_drug_name = (timelines.earliest_drug or "").strip()
    earliest_claim = (
        "Based on the expected FDA approval timeframes provided in the answer, the earliest expected FDA approval among "
        "retatrutide (Eli Lilly), CagriSema (Novo Nordisk), and orforglipron (Eli Lilly) is "
        f"{earliest_drug_name if earliest_drug_name else 'unknown'}."
    )
    await evaluator.verify(
        claim=earliest_claim,
        node=earliest_id_node,
        additional_instruction=_compose_earliest_comparison_instruction(timelines)
    )

    # Leaf 2: Expected approval timeline for the identified drug
    timeline_node = evaluator.add_leaf(
        id="Expected_Approval_Timeline",
        desc="Provides the expected FDA approval timeframe for the identified drug based on the stated regulatory timeline information",
        parent=drug_node,
        critical=True
    )

    selected_drug_info = _get_drug_info_by_name(earliest_drug_name, timelines)
    selected_sources: List[str] = []
    if timelines.earliest_sources:
        selected_sources.extend(timelines.earliest_sources)
    if selected_drug_info and selected_drug_info.sources:
        selected_sources.extend(selected_drug_info.sources)
    # Deduplicate sources
    selected_sources = list(dict.fromkeys(selected_sources))

    earliest_timeframe = (timelines.earliest_timeframe or "").strip()
    timeline_claim = (
        f"The expected FDA approval timeframe for {earliest_drug_name if earliest_drug_name else 'the selected drug'} "
        f"is {earliest_timeframe if earliest_timeframe else 'unknown'}."
    )

    await evaluator.verify(
        claim=timeline_claim,
        node=timeline_node,
        sources=selected_sources if selected_sources else None,
        additional_instruction=_compose_timeline_support_instruction(earliest_drug_name or "the selected drug")
    )


# --------------------------------------------------------------------------- #
# Main evaluation function                                                    #
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
) -> Dict[str, Any]:
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

    # Extract both sections in parallel
    vaccines_extraction_task = evaluator.extract(
        prompt=prompt_extract_vaccines(),
        template_class=VaccinationExtraction,
        extraction_name="vaccination_guidance"
    )
    drug_timelines_task = evaluator.extract(
        prompt=prompt_extract_drug_timelines(),
        template_class=WeightLossTimelineExtraction,
        extraction_name="drug_timelines"
    )
    vaccines_extraction, drug_timelines = await asyncio.gather(vaccines_extraction_task, drug_timelines_task)

    # Add top-level aggregation node as per rubric
    planning_node = evaluator.add_parallel(
        id="Travel_and_Medication_Planning",
        desc=ROOT_NODE_DESC,
        parent=root,
        critical=False
    )

    # Verification subtasks
    await verify_vaccination_requirements(evaluator, planning_node, vaccines_extraction)
    await verify_drug_timeline(evaluator, planning_node, drug_timelines)

    # Record custom info for debugging/traceability
    evaluator.add_custom_info(
        info={
            "vaccines_extracted": vaccines_extraction.dict() if hasattr(vaccines_extraction, "dict") else str(vaccines_extraction),
            "drug_timelines_extracted": drug_timelines.dict() if hasattr(drug_timelines, "dict") else str(drug_timelines),
        },
        info_type="extraction_debug",
        info_name="extracted_structures_snapshot"
    )

    return evaluator.get_summary()