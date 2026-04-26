import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator, AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "vr_xr2_gen2_2023_under600"
TASK_DESCRIPTION = (
    "Which standalone VR headset uses the Qualcomm Snapdragon XR2 Gen 2 processor, was released in 2023, "
    "and has a starting retail price under $600 USD? Provide the model name and the processor specification."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class VRHeadsetSelection(BaseModel):
    """
    Extraction result for the primary headset the answer identifies as meeting the constraints.
    """
    model_name: Optional[str] = None
    processor: Optional[str] = None  # e.g., "Qualcomm Snapdragon XR2 Gen 2"
    release_year: Optional[str] = None  # Prefer a 4-digit year, e.g., "2023"
    starting_price: Optional[str] = None  # The starting retail price as written in the answer (text)
    sources: List[str] = Field(default_factory=list)  # All URLs cited for this headset in the answer


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_vr_headset_selection() -> str:
    return """
    Your task is to extract the SINGLE VR headset the answer presents as the correct result (the primary choice).
    If multiple headsets are mentioned, choose the one the answer identifies as the final/primary pick. If not explicitly indicated,
    choose the first one that appears to be the intended answer.

    Extract the following fields for that single headset:
    - model_name: The headset model name (e.g., "Meta Quest 3").
    - processor: The processor/chipset as stated (e.g., "Qualcomm Snapdragon XR2 Gen 2"). If unspecified, return null.
    - release_year: The headset's release year mentioned in the answer as a 4-digit string (e.g., "2023"). If unspecified, return null.
    - starting_price: The starting retail price as stated in the answer (include currency and amount as written). If unspecified, return null.
    - sources: An array of all URLs cited in the answer that are relevant to this headset and could support details like processor, release year, standalone nature, or price. If none are present, return an empty array.

    IMPORTANT:
    - Follow the "SPECIAL RULES FOR URL SOURCES EXTRACTION" closely: only extract URLs explicitly present in the answer (including markdown links), do not invent.
    - Do not normalize or infer missing values; if something is not stated in the answer, return null for that field.
    """


# --------------------------------------------------------------------------- #
# Verification sub-tree builder                                               #
# --------------------------------------------------------------------------- #
async def build_vr_headset_verification(
    evaluator: Evaluator,
    parent_node,
    info: VRHeadsetSelection
) -> None:
    """
    Build the verification tree based on the rubric and run verifications.
    """
    # Top-level critical node: "VR_Headset_Identification"
    vr_node = evaluator.add_parallel(
        id="VR_Headset_Identification",
        desc="Identify a standalone VR headset that meets all constraints and provide the model name and processor specification.",
        parent=parent_node,
        critical=True
    )

    # Critical checks that the answer provides model name and processor specification
    provides_model_node = evaluator.add_custom_node(
        result=bool(info.model_name and info.model_name.strip()),
        id="Provides_Model_Name",
        desc="The answer provides the headset model name.",
        parent=vr_node,
        critical=True
    )

    provides_processor_node = evaluator.add_custom_node(
        result=bool(info.processor and info.processor.strip()),
        id="Provides_Processor_Specification",
        desc="The answer provides the processor specification (i.e., identifies the processor used, consistent with the constraints).",
        parent=vr_node,
        critical=True
    )

    # Critical node grouping all constraints
    constraints_node = evaluator.add_parallel(
        id="Headset_Meets_All_Constraints",
        desc="The identified headset satisfies all stated constraints.",
        parent=vr_node,
        critical=True
    )

    # Prepare sources (may be empty)
    sources = info.sources if info and info.sources else None
    model_display = info.model_name or "the headset"

    # 1) Standalone check
    standalone_leaf = evaluator.add_leaf(
        id="Is_Standalone_VR_Headset",
        desc="The headset is standalone (does not require a gaming console or PC connection).",
        parent=constraints_node,
        critical=True
    )
    standalone_claim = (
        f"{model_display} is a standalone/all-in-one VR headset that can operate without a tethered PC or game console, "
        "with onboard processing and battery."
    )
    await evaluator.verify(
        claim=standalone_claim,
        node=standalone_leaf,
        sources=sources,
        additional_instruction=(
            "Accept synonymous phrases such as 'standalone', 'all-in-one', 'untethered', or 'no PC required'. "
            "If a device can be optionally connected to a PC but does not require it to operate, it still counts as standalone."
        )
    )

    # 2) Processor check (XR2 Gen 2)
    processor_leaf = evaluator.add_leaf(
        id="Uses_XR2_Gen_2",
        desc="The headset uses the Qualcomm Snapdragon XR2 Gen 2 chipset.",
        parent=constraints_node,
        critical=True
    )
    processor_claim = (
        f"{model_display} uses the Qualcomm Snapdragon XR2 Gen 2 chipset "
        "(also stylized as 'Snapdragon XR2 Gen 2' or 'XR2 Gen2' or 'XR2 Gen 2')."
    )
    await evaluator.verify(
        claim=processor_claim,
        node=processor_leaf,
        sources=sources,
        additional_instruction=(
            "Focus on whether the processor is explicitly the second-generation XR2. "
            "Do not accept XR2 Gen 1, XR2+ Gen 1, or other chipsets."
        )
    )

    # 3) Release year check: 2023
    released_leaf = evaluator.add_leaf(
        id="Released_In_2023",
        desc="The headset release year is 2023.",
        parent=constraints_node,
        critical=True
    )
    released_claim = f"{model_display} was released (launched/first made available) in the year 2023."
    await evaluator.verify(
        claim=released_claim,
        node=released_leaf,
        sources=sources,
        additional_instruction=(
            "Prefer the first availability/launch date. If only 'announced' is mentioned without release, do not count as released. "
            "Minor regional rollouts still count if initial retail availability occurred in 2023 somewhere."
        )
    )

    # 4) Price check: starting price under $600 USD
    price_leaf = evaluator.add_leaf(
        id="Starting_Price_Under_600",
        desc="The headset starting retail price is under $600 USD.",
        parent=constraints_node,
        critical=True
    )
    price_claim = (
        f"The starting retail price (base model at launch) for {model_display} was under $600 USD."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=sources,
        additional_instruction=(
            "Interpret 'starting price' as the launch/base SKU MSRP in USD where possible. "
            "A price of $599.99 qualifies (it is under $600.00), while $600.00 does not. "
            "If only non-USD currencies are present, determine whether the cited USD-equivalent or MSRP indicates under $600 explicitly."
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
    """
    Evaluate an answer for the VR headset identification task.
    """
    # Initialize evaluator (root is non-critical by framework design; we add a critical child node for the rubric root)
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

    # Extract the single, primary headset selection from the answer
    selection = await evaluator.extract(
        prompt=prompt_extract_vr_headset_selection(),
        template_class=VRHeadsetSelection,
        extraction_name="vr_headset_selection"
    )

    # Build the verification tree and run checks
    await build_vr_headset_verification(evaluator, root, selection)

    # Return the final structured summary
    return evaluator.get_summary()