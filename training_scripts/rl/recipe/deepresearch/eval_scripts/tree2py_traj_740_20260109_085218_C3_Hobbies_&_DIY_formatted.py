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
TASK_ID = "guitar_truss_rod_specs"
TASK_DESCRIPTION = (
    "For a DIY electric guitar build using a standard thin C-shape neck profile "
    "(0.800 inches at the 1st fret and 0.850 inches at the 12th fret), identify the "
    "recommended dual-action truss rod model, specify its weight for a 14-inch length, "
    "and provide the required installation channel dimensions including width and depth "
    "(measured from the bottom of the fretboard). Include reference URLs for all specifications."
)

# Ground-truth expectations (used only to shape claims/prompts; verification is still evidence-based)
EXPECTED_MODEL = "StewMac Hot Rod dual-action truss rod"
EXPECTED_WEIGHT_14IN = "103 grams"
EXPECTED_CHANNEL_WIDTH = '7/32"'  # Commonly represented as 7/32 inches
EXPECTED_CHANNEL_DEPTH = '7/16"'  # Commonly represented as 7/16 inches
EXPECTED_DEPTH_REFERENCE = "from the bottom of the fretboard"

# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class SpecWithSources(BaseModel):
    value: Optional[str] = None
    urls: List[str] = Field(default_factory=list)


class DepthSpecWithSources(BaseModel):
    value: Optional[str] = None
    measurement_reference: Optional[str] = None  # e.g., "from the bottom of the fretboard"
    urls: List[str] = Field(default_factory=list)


class TrussRodExtraction(BaseModel):
    truss_rod_type: Optional[SpecWithSources] = None  # Expect "dual-action", "two-way", "double-action", etc.
    model_name: Optional[SpecWithSources] = None      # Expect "StewMac Hot Rod ..."
    weight_14_in: Optional[SpecWithSources] = None    # Expect "103 g", "103 grams", etc.
    channel_width: Optional[SpecWithSources] = None   # Expect '7/32"', '7/32 inches', etc.
    channel_depth: Optional[DepthSpecWithSources] = None  # Expect '7/16"', '7/16 inches', etc.


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_truss_rod_specs() -> str:
    return """
Extract, from the answer text, the truss rod specifications and the exact URLs cited to support each spec.

Return a JSON object with these fields:
- truss_rod_type: { value: string or null, urls: string[] } 
  • value: as stated in the answer (e.g., "dual-action", "two-way", "double-action", etc.)
  • urls: all URLs in the answer that support this type (if any)
- model_name: { value: string or null, urls: string[] }
  • value: as stated in the answer (e.g., "StewMac Hot Rod dual-action truss rod")
  • urls: all URLs supporting this model identification (if any)
- weight_14_in: { value: string or null, urls: string[] }
  • value: the 14-inch length truss rod weight exactly as stated (e.g., "103 grams", "103 g", "0.103 kg")
  • urls: all URLs that support the 14-inch weight
- channel_width: { value: string or null, urls: string[] }
  • value: the required installation channel width as stated (e.g., '7/32"', '7/32 inches', '0.21875"', '5.56 mm')
  • urls: all URLs that support the channel width requirement
- channel_depth: { value: string or null, measurement_reference: string or null, urls: string[] }
  • value: the required installation channel depth as stated (e.g., '7/16"', '7/16 inches', '0.4375"', '11.11 mm')
  • measurement_reference: the reference for where depth is measured from if stated (e.g., "from the bottom of the fretboard")
  • urls: all URLs that support the channel depth requirement

IMPORTANT:
- Extract only what appears in the answer. Do not infer or invent.
- For URLs, extract only actual URLs present in the answer (plain links or markdown links).
- Normalize units in the 'value' field exactly as written in the answer (do not convert).
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _valid_urls(urls: Optional[List[str]]) -> List[str]:
    if not urls:
        return []
    # Basic filtering for non-empty strings that look like URLs
    cleaned = []
    for u in urls:
        if not isinstance(u, str):
            continue
        s = u.strip()
        if len(s) == 0:
            continue
        # Accept with or without protocol; Verifier handles normalization internally
        cleaned.append(s)
    return cleaned


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
async def _verify_url_support(
    evaluator: Evaluator,
    parent_node,
    leaf_id: str,
    desc: str,
    claim: str,
    urls: List[str],
    additional_instruction: str,
    critical: bool = True,
):
    """
    Add a URL-support verification leaf. If no URLs are provided, mark as failed via custom node.
    """
    urls = _valid_urls(urls)
    if not urls:
        evaluator.add_custom_node(
            result=False,
            id=leaf_id,
            desc=f"{desc} (No URLs provided in the answer to support this.)",
            parent=parent_node,
            critical=critical,
        )
        return

    leaf = evaluator.add_leaf(
        id=leaf_id,
        desc=desc,
        parent=parent_node,
        critical=critical,
    )
    await evaluator.verify(
        claim=claim,
        node=leaf,
        sources=urls,
        additional_instruction=additional_instruction,
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
    Evaluate an answer for the guitar truss rod specifications task.
    """
    # Initialize evaluator and root
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

    # Extraction
    extracted: TrussRodExtraction = await evaluator.extract(
        prompt=prompt_extract_truss_rod_specs(),
        template_class=TrussRodExtraction,
        extraction_name="truss_rod_specs_extraction",
    )

    # Add ground-truth info for transparency (not used as hard constraints)
    evaluator.add_ground_truth({
        "expected_model": EXPECTED_MODEL,
        "expected_weight_14in": EXPECTED_WEIGHT_14IN,
        "expected_channel_width": EXPECTED_CHANNEL_WIDTH,
        "expected_channel_depth": f'{EXPECTED_CHANNEL_DEPTH} ({EXPECTED_DEPTH_REFERENCE})'
    }, gt_type="expected_specs")

    # Build the rubric tree according to JSON
    # Top-level critical node
    complete_node = evaluator.add_parallel(
        id="Complete_Truss_Rod_Specification",
        desc="Verify the response identifies the required dual-action truss rod and provides all required specs with supporting reference URLs.",
        parent=root,
        critical=True
    )

    # --------------------- Truss_Rod_Type_And_Model --------------------- #
    type_model_node = evaluator.add_parallel(
        id="Truss_Rod_Type_And_Model",
        desc="Response identifies the required truss rod type and specific model.",
        parent=complete_node,
        critical=True
    )

    # Leaf: Type_Verification (answer states dual-action)
    type_value = extracted.truss_rod_type.value if extracted.truss_rod_type else None
    type_leaf = evaluator.add_leaf(
        id="Type_Verification",
        desc="The truss rod is identified as dual-action (two-way adjustable), allowing both forward and backward neck adjustment.",
        parent=type_model_node,
        critical=True
    )
    type_claim_text = (
        "The answer identifies the truss rod as dual-action (two-way adjustable). "
        f"Extracted type phrase from the answer: '{type_value}'."
    )
    await evaluator.verify(
        claim=type_claim_text,
        node=type_leaf,
        additional_instruction=(
            "Accept synonyms like 'dual-action', 'double action', 'two-way', 'two way', '2-way'. "
            "If the answer describes single-action or only one direction of adjustment, mark as incorrect."
        ),
    )

    # Leaf: Model_Name (answer names the StewMac Hot Rod dual-action truss rod)
    model_value = extracted.model_name.value if extracted.model_name else None
    model_leaf = evaluator.add_leaf(
        id="Model_Name",
        desc="The specific recommended model is identified as the StewMac Hot Rod dual-action truss rod.",
        parent=type_model_node,
        critical=True
    )
    model_claim_text = (
        f"The answer names the truss rod model as the StewMac Hot Rod dual-action truss rod. "
        f"Extracted model phrase from the answer: '{model_value}'."
    )
    await evaluator.verify(
        claim=model_claim_text,
        node=model_leaf,
        additional_instruction=(
            "Allow reasonable variants like 'StewMac Hot Rod', 'Hot Rod dual action truss rod', "
            "'Stewart-MacDonald Hot Rod two-way truss rod'. It must clearly be the StewMac Hot Rod dual-action product."
        ),
    )

    # ---------------------- Weight_14_Inch_With_URL ---------------------- #
    weight_group = evaluator.add_parallel(
        id="Weight_14_Inch_With_URL",
        desc="Response provides the required 14-inch truss rod weight and a supporting reference URL.",
        parent=complete_node,
        critical=True
    )
    weight_value_str = extracted.weight_14_in.value if extracted.weight_14_in else None
    weight_urls = extracted.weight_14_in.urls if extracted.weight_14_in else []

    # Leaf: Weight_Value (answer states 103 grams for 14-inch)
    weight_value_leaf = evaluator.add_leaf(
        id="Weight_Value",
        desc="Weight is specified as 103 grams for the 14-inch length model.",
        parent=weight_group,
        critical=True
    )
    weight_value_claim = (
        f"The answer states that the 14-inch StewMac Hot Rod truss rod weighs {EXPECTED_WEIGHT_14IN}. "
        f"Extracted weight phrase from the answer: '{weight_value_str}'."
    )
    await evaluator.verify(
        claim=weight_value_claim,
        node=weight_value_leaf,
        additional_instruction=(
            "Treat '103 g' and '103 grams' as equivalent. Also accept minor format variants like '0.103 kg' if clearly equivalent. "
            "The claim should clearly be for the 14-inch Hot Rod length."
        ),
    )

    # Leaf: Weight_Reference_URL (at least one URL supports 103g for 14-inch)
    await _verify_url_support(
        evaluator=evaluator,
        parent_node=weight_group,
        leaf_id="Weight_Reference_URL",
        desc="A reference URL is provided that supports the stated 14-inch weight.",
        claim=(
            f"The 14-inch StewMac Hot Rod truss rod weighs {EXPECTED_WEIGHT_14IN}."
        ),
        urls=weight_urls,
        additional_instruction=(
            "Verify the page explicitly or unambiguously supports the 14-inch Hot Rod weight as 103 grams "
            "(allow '103 g', or equivalent units like ~0.103 kg). If the page refers to a different length or a different model, "
            "do not accept."
        ),
        critical=True,
    )

    # --------------------- Channel_Width_With_URL ------------------------ #
    width_group = evaluator.add_parallel(
        id="Channel_Width_With_URL",
        desc="Response provides the required installation channel width and a supporting reference URL.",
        parent=complete_node,
        critical=True
    )
    width_value_str = extracted.channel_width.value if extracted.channel_width else None
    width_urls = extracted.channel_width.urls if extracted.channel_width else []

    # Leaf: Channel_Width_Value (answer states 7/32 inches)
    width_value_leaf = evaluator.add_leaf(
        id="Channel_Width_Value",
        desc="Installation channel width is specified as 7/32 inches.",
        parent=width_group,
        critical=True
    )
    width_claim = (
        f"The answer specifies the installation channel width for the Hot Rod truss rod as {EXPECTED_CHANNEL_WIDTH} (i.e., 7/32 inches). "
        f"Extracted width phrase from the answer: '{width_value_str}'."
    )
    await evaluator.verify(
        claim=width_claim,
        node=width_value_leaf,
        additional_instruction=(
            "Allow equivalent numeric formats such as '7/32\"', '0.21875\"', or ~5.56 mm. "
            "If the answer clearly states 7/32 inches (or an exact equivalent), mark as correct."
        ),
    )

    # Leaf: Channel_Width_Reference_URL (URL supports the width)
    await _verify_url_support(
        evaluator=evaluator,
        parent_node=width_group,
        leaf_id="Channel_Width_Reference_URL",
        desc="A reference URL is provided that supports the stated channel width.",
        claim=(
            "The required installation channel (slot) width for the StewMac Hot Rod dual-action truss rod is 7/32 inches "
            '(≈0.21875").'
        ),
        urls=width_urls,
        additional_instruction=(
            "Verify the page specifies the slot width as 7/32 inches (or clear equivalent in inches/mm). "
            "Ensure the spec is for the StewMac Hot Rod truss rod installation slot, not a different product."
        ),
        critical=True,
    )

    # --------------------- Channel_Depth_With_URL ------------------------ #
    depth_group = evaluator.add_parallel(
        id="Channel_Depth_With_URL",
        desc="Response provides the required installation channel depth (including measurement basis) and a supporting reference URL.",
        parent=complete_node,
        critical=True
    )
    depth_value_str = extracted.channel_depth.value if extracted.channel_depth else None
    depth_ref_str = extracted.channel_depth.measurement_reference if extracted.channel_depth else None
    depth_urls = extracted.channel_depth.urls if extracted.channel_depth else []

    # Leaf: Channel_Depth_Value (answer states 7/16 inches measured from bottom of fretboard)
    depth_value_leaf = evaluator.add_leaf(
        id="Channel_Depth_Value",
        desc="Installation channel depth is specified as 7/16 inches, measured from the bottom of the fretboard.",
        parent=depth_group,
        critical=True
    )
    depth_claim = (
        f"The answer specifies the installation channel depth as {EXPECTED_CHANNEL_DEPTH} (i.e., 7/16 inches), "
        f"measured from the bottom of the fretboard. Extracted depth phrase: '{depth_value_str}'. "
        f"Extracted measurement reference: '{depth_ref_str}'."
    )
    await evaluator.verify(
        claim=depth_claim,
        node=depth_value_leaf,
        additional_instruction=(
            "Accept equivalent numeric formats such as '7/16\"', '0.4375\"', or ~11.11 mm. "
            "Also verify that the measurement reference is 'from the bottom/underside of the fretboard' or an equivalent phrasing."
        ),
    )

    # Leaf: Channel_Depth_Reference_URL (URL supports depth and preferably measurement reference)
    await _verify_url_support(
        evaluator=evaluator,
        parent_node=depth_group,
        leaf_id="Channel_Depth_Reference_URL",
        desc="A reference URL is provided that supports the stated channel depth (including the measurement reference point if specified by the source).",
        claim=(
            "The required installation channel (slot) depth for the StewMac Hot Rod dual-action truss rod is 7/16 inches "
            '(≈0.4375"), measured from the bottom (underside) of the fretboard.'
        ),
        urls=depth_urls,
        additional_instruction=(
            "Verify the page supports 7/16\" depth for the Hot Rod truss rod. "
            "If the source explicitly states the measurement reference (from the bottom/underside of the fretboard), confirm that as well. "
            "If the page is ambiguous on the reference point but clearly supports the 7/16\" depth spec for the Hot Rod slot, "
            "it can still be accepted."
        ),
        critical=True,
    )

    # Return evaluator summary
    return evaluator.get_summary()