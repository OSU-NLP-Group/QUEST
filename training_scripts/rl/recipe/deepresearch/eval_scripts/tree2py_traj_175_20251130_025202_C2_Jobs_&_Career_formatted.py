import asyncio
import logging
from typing import Any, List, Dict, Optional

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "harvard_head_coach_info_2024"
TASK_DESCRIPTION = (
    "Harvard University appointed a new head football coach in February 2024, replacing the legendary Tim Murphy who retired after 30 years. "
    "For someone interested in understanding the career path to becoming an Ivy League head football coach, research and provide the following information about Harvard's current head coach: "
    "(1) their full name, (2) the university where they played college football as an undergraduate, and "
    "(3) the total number of years of coaching experience they had accumulated before taking the Harvard position. "
    "Please include reference URLs from Harvard Athletics or other credible sources to support your findings."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class CoachInfoExtraction(BaseModel):
    """
    Structured extraction of the requested information about Harvard's current head football coach.
    """
    full_name: Optional[str] = None
    undergraduate_university: Optional[str] = None
    coaching_experience_years_before_harvard: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt helpers                                                   #
# --------------------------------------------------------------------------- #
def prompt_extract_coach_info() -> str:
    return """
    Extract the requested information about Harvard University's current head football coach (appointed in February 2024 after Tim Murphy retired).
    You must extract ONLY what is explicitly present in the provided answer text, without adding or inferring.

    Required fields:
    1) full_name: The coach's full name.
    2) undergraduate_university: The university where the coach played college football as an undergraduate.
    3) coaching_experience_years_before_harvard: The total number of years of coaching experience the coach had BEFORE taking the Harvard job (free-form string is acceptable, e.g., "20", "over 20", "more than 15", "about 18").
    4) sources: An array of all URL(s) explicitly provided in the answer that support any of the above facts. Return only valid URLs. Include official Harvard Athletics or Harvard-affiliated pages if present (e.g., gocrimson.com, harvard.edu), or other credible sources (e.g., major news outlets, official athletics sites of prior schools, NCAA, Ivy League).

    Rules:
    - If any field is missing in the answer, set it to null (for strings) or [] for sources.
    - Do not invent or normalize values beyond what is stated (e.g., keep "20+" or "over 20" if that's how it's stated).
    - For sources, collect all unique URLs mentioned that are relevant to these facts. Keep them as-is if present.

    Return a single JSON object with keys:
    - full_name
    - undergraduate_university
    - coaching_experience_years_before_harvard
    - sources
    """


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def _dedupe_preserve_order(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


# --------------------------------------------------------------------------- #
# Verification building                                                       #
# --------------------------------------------------------------------------- #
async def build_and_verify(
    evaluator: Evaluator,
    task_node,
    extracted: CoachInfoExtraction,
) -> None:
    """
    Build verification leaves according to the rubric and verify claims against cited sources.
    The parent node (task_node) is critical; all children leaves here are marked critical.
    """

    # Prepare sources (deduplicated)
    sources_list = _dedupe_preserve_order(extracted.sources or [])

    # Add a small gating existence check for sources to ensure other checks can rely on URLs
    evaluator.add_custom_node(
        result=len(sources_list) > 0,
        id="supporting_reference_urls_provided",
        desc="At least one supporting reference URL is provided in the answer",
        parent=task_node,
        critical=True
    )

    # Leaf: Coach full name
    coach_name_node = evaluator.add_leaf(
        id="coach_full_name",
        desc="Provide the full name of Harvard's current head football coach",
        parent=task_node,
        critical=True,
    )
    name_val = extracted.full_name or ""
    name_claim = (
        f"As of February 2024 and onward, Harvard University's head football coach (hired to replace Tim Murphy) is {name_val}."
    )
    await evaluator.verify(
        claim=name_claim,
        node=coach_name_node,
        sources=sources_list,
        additional_instruction=(
            "Verify that the page identifies the current Harvard head football coach by this name. "
            "Allow minor formatting differences (e.g., middle initials). "
            "The key is that this person is the head coach named after Tim Murphy's retirement in 2024."
        )
    )

    # Leaf: Undergraduate football university
    undergrad_node = evaluator.add_leaf(
        id="undergraduate_football_university",
        desc="Provide the university where the coach played college football as an undergraduate",
        parent=task_node,
        critical=True,
    )
    undergrad_val = extracted.undergraduate_university or ""
    undergrad_claim = (
        f"{name_val} played college football as an undergraduate at {undergrad_val}."
    )
    await evaluator.verify(
        claim=undergrad_claim,
        node=undergrad_node,
        sources=sources_list,
        additional_instruction=(
            "Look for biographical sections (e.g., 'played at', 'lettered at', 'college playing career'). "
            "Ensure this refers to undergraduate football, not graduate school."
        )
    )

    # Leaf: Pre-Harvard coaching experience years
    years_node = evaluator.add_leaf(
        id="pre_harvard_coaching_experience_years",
        desc="Provide the total number of years of coaching experience the coach had accumulated before taking the Harvard position",
        parent=task_node,
        critical=True,
    )
    years_val = extracted.coaching_experience_years_before_harvard or ""
    years_claim = (
        f"Before being named Harvard head coach in 2024, {name_val} had {years_val} years of coaching experience."
    )
    await evaluator.verify(
        claim=years_claim,
        node=years_node,
        sources=sources_list,
        additional_instruction=(
            "Accept reasonable textual variants like 'over 20 years', '20+', or 'about 20'. "
            "If the total is not explicitly given, it may be inferred from the timeline on the page. "
            "Judge the claim correct if the total stated is consistent with the biography."
        )
    )

    # Leaf: Supporting reference URLs credibility/relevance
    ref_urls_node = evaluator.add_leaf(
        id="supporting_reference_urls",
        desc="Include reference URL(s) from Harvard Athletics or other credible sources that support the provided findings",
        parent=task_node,
        critical=True,
    )
    credibility_claim = (
        "This webpage is either an official Harvard Athletics or Harvard-affiliated page "
        "(e.g., gocrimson.com or harvard.edu) OR a comparably credible source (e.g., NCAA, Ivy League, major news outlet, "
        "or an official athletics site of the coach's prior school), and it provides biographical or appointment information "
        "about Harvard's head football coach that can support the facts above."
    )
    await evaluator.verify(
        claim=credibility_claim,
        node=ref_urls_node,
        sources=sources_list,
        additional_instruction=(
            "Pass if at least one of the provided URLs is clearly from Harvard Athletics/harvard.edu OR another established and credible outlet, "
            "and the page contains relevant information (appointment announcement or coach bio) that could support the facts provided."
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
    model: str = "o4-mini",
) -> Dict:
    """
    Evaluate an answer for the Harvard head football coach information task.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # top-level structure
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

    # Create a critical task node mirroring the rubric root (since evaluator root is non-critical by design)
    task_node = evaluator.add_parallel(
        id="task_root",
        desc="Provide the requested information about Harvard's current head football coach (appointed Feb 2024) with supporting references",
        parent=root,
        critical=True,
    )

    # Extract structured information from the answer
    extracted: CoachInfoExtraction = await evaluator.extract(
        prompt=prompt_extract_coach_info(),
        template_class=CoachInfoExtraction,
        extraction_name="coach_info_extraction",
    )

    # Optionally record some custom info for debugging
    evaluator.add_custom_info(
        info={
            "full_name": extracted.full_name,
            "undergraduate_university": extracted.undergraduate_university,
            "coaching_experience_years_before_harvard": extracted.coaching_experience_years_before_harvard,
            "source_url_count": len(extracted.sources or []),
        },
        info_type="extract_summary",
        info_name="extracted_fields_overview",
    )

    # Build verification leaves and verify claims
    await build_and_verify(evaluator, task_node, extracted)

    # Return summary
    return evaluator.get_summary()