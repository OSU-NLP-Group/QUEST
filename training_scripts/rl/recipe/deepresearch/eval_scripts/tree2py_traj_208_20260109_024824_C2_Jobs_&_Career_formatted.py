import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "mn_cpa_shorter_work_exp_2026"
TASK_DESCRIPTION = (
    "Among the two new CPA licensure pathways in Minnesota that became effective on January 1, 2026 "
    "(the Bachelor's degree pathway and the Master's degree pathway), which pathway requires a shorter "
    "duration of work experience? What is the specific duration of experience required for this pathway?"
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ShorterPathwayExtraction(BaseModel):
    """
    Extract the user’s stated shorter pathway and its required experience duration,
    along with any cited source URLs from the answer text.
    """
    shorter_pathway: Optional[str] = None
    duration: Optional[str] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_shorter_pathway() -> str:
    return """
    From the provided answer, extract the following information about Minnesota's two new CPA licensure pathways
    that became effective on January 1, 2026:

    1) shorter_pathway: Which pathway is stated to require the shorter duration of qualifying work experience?
       The two options are the "Bachelor's degree pathway" and the "Master's degree pathway".
       - If the answer uses synonyms (e.g., bachelor's, baccalaureate; master's, graduate), normalize to one of:
         "Bachelor's degree pathway" or "Master's degree pathway" when possible.
       - If the answer expresses uncertainty or does not specify, return null.

    2) duration: The specific duration of qualifying work experience the answer states for that shorter-experience pathway.
       - Preserve the unit exactly as stated (e.g., "1 year", "12 months", "2,000 hours").
       - If not stated, return null.

    3) source_urls: All URLs cited in the answer that support the claim.
       - Extract only actual URLs mentioned (including markdown links). If none are provided, return an empty list.

    Return a JSON object with fields: shorter_pathway, duration, source_urls.
    Do not infer or invent anything not explicitly supported by the provided answer text.
    """


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def normalize_pathway_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    low = label.lower()
    if "master" in low:
        return "Master's degree pathway"
    if "bachelor" in low or "baccalaureate" in low:
        return "Bachelor's degree pathway"
    # If it doesn't match well-known forms, return original
    return label.strip()


def format_duration(duration: Optional[str]) -> str:
    return duration.strip() if duration else "None"


# --------------------------------------------------------------------------- #
# Main verification logic                                                     #
# --------------------------------------------------------------------------- #
async def _verify_shorter_pathway_and_duration(
    evaluator: Evaluator,
    parent_node,
    extracted: ShorterPathwayExtraction
) -> None:
    """
    Build the rubric tree as specified and run verifications.
    """
    # Create the main (critical) node reflecting the rubric root
    optimal_node = evaluator.add_parallel(
        id="Optimal_Pathway_Determination",
        desc="Answer identifies which of the two new Minnesota CPA pathways has the shorter work-experience requirement and states the specific duration.",
        parent=parent_node,
        critical=True
    )

    # Normalize pathway label for clearer claims
    normalized_pathway = normalize_pathway_label(extracted.shorter_pathway)
    sources = extracted.source_urls if extracted.source_urls else None
    stated_duration = format_duration(extracted.duration)

    # Leaf 1: Pathway identified correctly (critical)
    pathway_leaf = evaluator.add_leaf(
        id="Pathway_Identified",
        desc="Answer correctly identifies which pathway (Bachelor's vs Master's) requires the shorter duration of qualifying work experience.",
        parent=optimal_node,
        critical=True
    )
    pathway_claim = (
        f"Among Minnesota's two new CPA licensure pathways effective on January 1, 2026 "
        f"(the Bachelor's degree pathway and the Master's degree pathway), "
        f"the {normalized_pathway if normalized_pathway else 'unspecified'} requires a shorter "
        f"duration of qualifying work experience than the other pathway."
    )
    await evaluator.verify(
        claim=pathway_claim,
        node=pathway_leaf,
        sources=sources,
        additional_instruction=(
            "Use the provided webpage(s) if available to determine which pathway truly requires a shorter duration "
            "of qualifying work experience. If a page lists both durations, compare them. "
            "Allow minor naming variations (e.g., 'bachelor', 'baccalaureate', or 'master'). "
            "Focus on the two new pathways effective January 1, 2026 in Minnesota."
        ),
    )

    # Leaf 2: Experience duration correct (critical)
    duration_leaf = evaluator.add_leaf(
        id="Experience_Duration",
        desc="Answer correctly states the specific duration of qualifying work experience required for the shorter-experience pathway.",
        parent=optimal_node,
        critical=True
    )
    duration_claim = (
        f"The qualifying work experience required for the "
        f"{normalized_pathway if normalized_pathway else 'shorter-experience pathway'} "
        f"is {stated_duration}."
    )
    # Make the duration verification depend on the pathway identification leaf
    await evaluator.verify(
        claim=duration_claim,
        node=duration_leaf,
        sources=sources,
        additional_instruction=(
            "Verify the exact duration required for the stated shorter-experience pathway. "
            "Accept equivalent expressions (e.g., '1 year' ≈ '12 months') and unit variations "
            "(e.g., hours vs months) if they are clearly equivalent per the source. "
            "Rely on the policy effective January 1, 2026 for Minnesota."
        ),
        extra_prerequisites=[pathway_leaf],
    )


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
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
    Evaluate an answer for the Minnesota CPA shorter work-experience pathway question.
    """
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root wrapper; rubric root added as a critical child
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
        prompt=prompt_extract_shorter_pathway(),
        template_class=ShorterPathwayExtraction,
        extraction_name="shorter_pathway_extraction",
    )

    # Optional: Record normalized info for debugging
    evaluator.add_custom_info(
        {
            "shorter_pathway_raw": extracted.shorter_pathway,
            "shorter_pathway_normalized": normalize_pathway_label(extracted.shorter_pathway),
            "duration_raw": extracted.duration,
            "source_urls_count": len(extracted.source_urls),
        },
        info_type="debug",
        info_name="normalized_extraction_overview",
    )

    # Build and run verification according to the rubric
    await _verify_shorter_pathway_and_duration(evaluator, root, extracted)

    return evaluator.get_summary()