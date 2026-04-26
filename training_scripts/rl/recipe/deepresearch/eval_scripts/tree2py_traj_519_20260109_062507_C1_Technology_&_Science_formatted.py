import asyncio
import logging
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "neuromorphic_breakthrough_2025"
TASK_DESCRIPTION = (
    "In October 2025, a breakthrough in neuromorphic computing involving diffusive "
    "memristor-based artificial neurons that use silver ions and require only a single "
    "transistor footprint per neuron was announced. Which university or research institution "
    "developed this technology, who was the lead researcher, and in which scientific journal "
    "was the research published?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class BreakthroughAttribution(BaseModel):
    """
    Structured extraction of attribution details from the agent's answer.
    """
    institution: Optional[str] = None
    lead_researcher: Optional[str] = None
    journal: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_attribution() -> str:
    return (
        "Extract the attribution details for the described October 2025 neuromorphic computing breakthrough: "
        "diffusive memristor-based artificial neurons that use silver ions and require only a single transistor "
        "footprint per neuron.\n\n"
        "Return a JSON object with the following fields:\n"
        "1. institution: The university or research institution explicitly stated in the answer as having "
        "   developed the described technology. If absent, return null.\n"
        "2. lead_researcher: The lead researcher explicitly stated in the answer for this work "
        "   (e.g., principal investigator, lead author, or project lead). If absent, return null.\n"
        "3. journal: The scientific journal explicitly stated in the answer where the research was published. "
        "   If absent, return null.\n"
        "4. sources: An array of all URLs cited in the answer that support the attribution (press releases, "
        "   official university pages, journal article pages, etc.). Only include actual URLs present in the answer. "
        "   If no URLs are present, return an empty list.\n\n"
        "Important:\n"
        "- Extract the fields exactly as they appear in the answer; do not infer or invent.\n"
        "- Include all URLs provided in any 'sources' section or embedded within text/markdown links."
    )


# --------------------------------------------------------------------------- #
# Common additional instruction for verifications                             #
# --------------------------------------------------------------------------- #
COMMON_VERIFICATION_ADDITIONAL_INSTRUCTION = (
    "You must verify the claim using the provided URL sources only. Do not rely on your own knowledge.\n"
    "The claim refers to a specific October 2025 neuromorphic computing breakthrough characterized by:\n"
    "• diffusive memristor-based artificial neurons,\n"
    "• use of silver ions,\n"
    "• only a single-transistor footprint per neuron.\n"
    "Accept minor variants in naming (e.g., 'Univ.' vs 'University', middle initials, accented characters) "
    "and reasonable journal name abbreviations. If the sources are irrelevant, inaccessible, or do not explicitly "
    "support the claim, conclude that the claim is not supported."
)


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def verify_institution(
    evaluator: Evaluator,
    parent_node,
    extracted: BreakthroughAttribution,
) -> None:
    """
    Build and execute verification for the Institution criterion.
    """
    inst_group = evaluator.add_sequential(
        id="Institution",
        desc=(
            "Provides the university or research institution that developed the described technology, "
            "and it is the correct institution for this specific breakthrough."
        ),
        parent=parent_node,
        critical=True,
    )

    # Existence + sources
    evaluator.add_custom_node(
        result=bool(extracted.institution and extracted.institution.strip()) and bool(extracted.sources),
        id="Institution_exists",
        desc="Institution is provided and at least one source URL is present in the answer",
        parent=inst_group,
        critical=True,
    )

    # Evidence-based verification against cited sources
    inst_verify_node = evaluator.add_leaf(
        id="Institution_supported",
        desc="Institution attribution is supported by the cited sources",
        parent=inst_group,
        critical=True,
    )

    claim = (
        f"The described October 2025 neuromorphic computing breakthrough (diffusive memristor-based artificial neuron "
        f"using silver ions with a single-transistor footprint per neuron) was developed by {extracted.institution}."
    )

    await evaluator.verify(
        claim=claim,
        node=inst_verify_node,
        sources=extracted.sources,
        additional_instruction=COMMON_VERIFICATION_ADDITIONAL_INSTRUCTION,
    )


async def verify_lead_researcher(
    evaluator: Evaluator,
    parent_node,
    extracted: BreakthroughAttribution,
) -> None:
    """
    Build and execute verification for the Lead Researcher criterion.
    """
    lead_group = evaluator.add_sequential(
        id="Lead_Researcher",
        desc=(
            "Provides the lead researcher for the described work, and it is the correct lead researcher associated "
            "with this breakthrough."
        ),
        parent=parent_node,
        critical=True,
    )

    # Existence + sources
    evaluator.add_custom_node(
        result=bool(extracted.lead_researcher and extracted.lead_researcher.strip()) and bool(extracted.sources),
        id="Lead_Researcher_exists",
        desc="Lead researcher is provided and at least one source URL is present in the answer",
        parent=lead_group,
        critical=True,
    )

    # Evidence-based verification against cited sources
    lead_verify_node = evaluator.add_leaf(
        id="Lead_Researcher_supported",
        desc="Lead researcher attribution is supported by the cited sources",
        parent=lead_group,
        critical=True,
    )

    claim = (
        f"The lead researcher of the described October 2025 neuromorphic computing breakthrough "
        f"(diffusive memristor-based artificial neuron using silver ions with a single-transistor footprint per neuron) "
        f"is {extracted.lead_researcher}."
    )

    add_ins = (
        COMMON_VERIFICATION_ADDITIONAL_INSTRUCTION
        + "\nAccept synonyms like 'principal investigator (PI)', 'project lead', 'lead author', or 'senior author' "
          "as equivalent indications of lead researcher if clearly tied to this breakthrough."
    )

    await evaluator.verify(
        claim=claim,
        node=lead_verify_node,
        sources=extracted.sources,
        additional_instruction=add_ins,
    )


async def verify_journal(
    evaluator: Evaluator,
    parent_node,
    extracted: BreakthroughAttribution,
) -> None:
    """
    Build and execute verification for the Journal criterion.
    """
    journal_group = evaluator.add_sequential(
        id="Journal",
        desc=(
            "Provides the scientific journal in which the described research was published, and it is the correct journal "
            "for this breakthrough."
        ),
        parent=parent_node,
        critical=True,
    )

    # Existence + sources
    evaluator.add_custom_node(
        result=bool(extracted.journal and extracted.journal.strip()) and bool(extracted.sources),
        id="Journal_exists",
        desc="Journal name is provided and at least one source URL is present in the answer",
        parent=journal_group,
        critical=True,
    )

    # Evidence-based verification against cited sources
    journal_verify_node = evaluator.add_leaf(
        id="Journal_supported",
        desc="Journal attribution is supported by the cited sources",
        parent=journal_group,
        critical=True,
    )

    claim = (
        f"The research for the described October 2025 neuromorphic computing breakthrough "
        f"(diffusive memristor-based artificial neuron using silver ions with a single-transistor footprint per neuron) "
        f"was published in {extracted.journal}."
    )

    add_ins = (
        COMMON_VERIFICATION_ADDITIONAL_INSTRUCTION
        + "\nAllow common journal name variants or abbreviations (e.g., 'Nat. Electronics' for 'Nature Electronics') "
          "if the page clearly indicates the same journal."
    )

    await evaluator.verify(
        claim=claim,
        node=journal_verify_node,
        sources=extracted.sources,
        additional_instruction=add_ins,
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
    Evaluate an agent's answer for the neuromorphic breakthrough attribution task.
    Returns a standardized summary dictionary produced by the obj_task_eval evaluator.
    """
    # Initialize evaluator with a parallel root (non-critical root; we'll add a critical child node)
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

    # Extract attribution details from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_attribution(),
        template_class=BreakthroughAttribution,
        extraction_name="breakthrough_attribution",
    )

    # Build the critical attribution node per rubric
    attribution_node = evaluator.add_parallel(
        id="Neuromorphic_Breakthrough_Attribution",
        desc=(
            "Identify the correct developing institution, lead researcher, and publication journal for the described "
            "October 2025 diffusive memristor-based artificial-neuron breakthrough."
        ),
        parent=root,
        critical=True,
    )

    # Add custom info for clarity (features defining the breakthrough)
    evaluator.add_custom_info(
        info={
            "key_features": [
                "diffusive memristor-based artificial neurons",
                "uses silver ions",
                "single-transistor footprint per neuron",
                "announcement/publication timeframe: October 2025",
            ]
        },
        info_type="context_features",
        info_name="breakthrough_features"
    )

    # Verify each critical sub-criterion
    await verify_institution(evaluator, attribution_node, extracted)
    await verify_lead_researcher(evaluator, attribution_node, extracted)
    await verify_journal(evaluator, attribution_node, extracted)

    # Return structured summary
    return evaluator.get_summary()