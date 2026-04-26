import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "perseverance_bright_angel_leopard_spots"
TASK_DESCRIPTION = """
In July 2024, NASA's Perseverance rover collected a rock core sample from a distinctive rock located in Jezero Crater's Bright Angel formation, within the ancient Neretva Vallis river channel region. The rock exhibited unusual surface features described as "leopard spots," and the sample has been associated with potential biosignatures that were reported in a peer-reviewed Nature paper published in September 2025. Based on this information, identify: (1) The name of the collected rock core sample, (2) The name of the source rock from which this sample was collected, (3) The two specific iron-bearing minerals that were detected in the leopard spot features (identify both the hydrated iron-phosphate mineral and the iron-sulfide mineral).
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class IdentificationsExtraction(BaseModel):
    core_sample_name: Optional[str] = None
    source_rock_name: Optional[str] = None
    hydrated_iron_phosphate_mineral: Optional[str] = None
    iron_sulfide_mineral: Optional[str] = None


class AnswerSources(BaseModel):
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_identifications() -> str:
    return """
    Extract the four requested identifications explicitly stated in the answer text. Return a JSON object with:
    - core_sample_name: The specific name of the collected rock core sample (e.g., the official sample/core name used by the Perseverance mission).
    - source_rock_name: The name of the source rock from which this core sample was collected (e.g., an informal rock target name).
    - hydrated_iron_phosphate_mineral: The name of the hydrated iron-phosphate mineral detected in the "leopard spots" features.
    - iron_sulfide_mineral: The name of the iron-sulfide mineral detected in the "leopard spots" features.

    Rules:
    1) Extract the names exactly as written in the answer. Do not invent or infer any names not present in the answer.
    2) If any of the four items is not mentioned or unclear, set that field to null.
    3) If multiple candidates are listed for a field, choose the most emphasized one or the first clearly identified one.
    """


def prompt_extract_sources() -> str:
    return """
    Extract all URLs explicitly presented in the answer (including plain URLs and markdown links). 
    Return a JSON object with:
    - urls: array of strings; each element is a full URL.

    Rules:
    1) Only include URLs that are explicitly present in the answer text.
    2) Normalize markdown links to plain URLs.
    3) Ignore obviously malformed URLs; ensure each starts with http:// or https:// (prepend http:// if missing).
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def is_nonempty(s: Optional[str]) -> bool:
    return bool(s) and s.strip() != ""


def filter_nature_urls(urls: List[str]) -> List[str]:
    lowered = [u.lower() for u in urls]
    selected = []
    for i, u in enumerate(lowered):
        if ("nature.com" in u) or ("doi.org/10.1038" in u):
            selected.append(urls[i])
    return selected


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_requested_identifications_tree(
    evaluator: Evaluator,
    parent_node,
    extracted: IdentificationsExtraction,
) -> None:
    """
    Build and verify the 'Requested_Identifications' subtree.

    We add four critical checks to ensure the answer provided each identification.
    """
    req_node = evaluator.add_parallel(
        id="Requested_Identifications",
        desc="Provide the four requested identifications.",
        parent=parent_node,
        critical=True
    )

    # 1) Core sample name
    core_exists = is_nonempty(extracted.core_sample_name)
    evaluator.add_custom_node(
        result=core_exists,
        id="Sample_Core_Name",
        desc="Provide the correct name of the collected rock core sample.",
        parent=req_node,
        critical=True
    )

    # 2) Source rock name
    source_exists = is_nonempty(extracted.source_rock_name)
    evaluator.add_custom_node(
        result=source_exists,
        id="Source_Rock_Name",
        desc="Provide the correct name of the source rock from which the sample core was collected.",
        parent=req_node,
        critical=True
    )

    # 3) Hydrated iron-phosphate mineral
    hip_exists = is_nonempty(extracted.hydrated_iron_phosphate_mineral)
    hip_leaf = evaluator.add_leaf(
        id="Hydrated_Iron_Phosphate_Mineral",
        desc="Identify the hydrated iron-phosphate mineral detected in the leopard spot features.",
        parent=req_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    hip_claim = (
        f"The answer identifies the hydrated iron-phosphate mineral as '{extracted.hydrated_iron_phosphate_mineral}'."
        if hip_exists else
        "The answer fails to identify a hydrated iron-phosphate mineral for the leopard spot features."
    )
    await evaluator.verify(
        claim=hip_claim,
        node=hip_leaf,
        additional_instruction="Judge solely based on the answer text. Pass if a specific mineral name is clearly provided for the hydrated iron-phosphate category."
    )

    # 4) Iron-sulfide mineral
    is_exists = is_nonempty(extracted.iron_sulfide_mineral)
    is_leaf = evaluator.add_leaf(
        id="Iron_Sulfide_Mineral",
        desc="Identify the iron-sulfide mineral detected in the leopard spot features.",
        parent=req_node,
        critical=True,
        status="initialized",
        score=0.0
    )
    is_claim = (
        f"The answer identifies the iron-sulfide mineral as '{extracted.iron_sulfide_mineral}'."
        if is_exists else
        "The answer fails to identify an iron-sulfide mineral for the leopard spot features."
    )
    await evaluator.verify(
        claim=is_claim,
        node=is_leaf,
        additional_instruction="Judge solely based on the answer text. Pass if a specific mineral name is clearly provided for the iron-sulfide category."
    )


async def build_context_constraints_tree(
    evaluator: Evaluator,
    parent_node,
    extracted_sources: AnswerSources,
) -> None:
    """
    Build and verify the 'Stated_Context_Constraints_Satisfied' subtree.

    Each leaf is a binary check that the answer asserts the corresponding constraint.
    """
    ctx_node = evaluator.add_parallel(
        id="Stated_Context_Constraints_Satisfied",
        desc="The identified sample/rock/minerals are consistent with all stated context constraints in the prompt/constraints.",
        parent=parent_node,
        critical=True
    )

    claims_and_sources: List[Tuple[str, List[str] | str | None, Any, Optional[str]]] = []

    # Mission and location
    mission_node = evaluator.add_leaf(
        id="Mission_And_Location",
        desc="The referenced rock/sample is from NASA's Perseverance rover in Jezero Crater on Mars.",
        parent=ctx_node,
        critical=True
    )
    mission_claim = ("The answer references NASA's Perseverance rover and Jezero Crater on Mars.")
    mission_ins = (
        "Verify strictly from the answer text. Pass if the answer clearly references both Perseverance (Mars 2020 rover) "
        "and Jezero Crater on Mars (phrasing variations allowed)."
    )
    claims_and_sources.append((mission_claim, None, mission_node, mission_ins))

    # Discovery date
    date_node = evaluator.add_leaf(
        id="Discovery_Date",
        desc="The collection/discovery timing is in July 2024.",
        parent=ctx_node,
        critical=True
    )
    date_claim = "The answer states the collection/discovery timing is in July 2024."
    date_ins = "Judge only by the answer text. Accept paraphrases like 'collected in July 2024' or 'July 2024 collection'."
    claims_and_sources.append((date_claim, None, date_node, date_ins))

    # Geologic unit
    unit_node = evaluator.add_leaf(
        id="Geologic_Unit",
        desc="The rock is located within the Bright Angel formation.",
        parent=ctx_node,
        critical=True
    )
    unit_claim = "The answer states the rock is located within the Bright Angel formation."
    unit_ins = "Judge only by the answer text. Accept paraphrases mentioning 'Bright Angel formation' specifically."
    claims_and_sources.append((unit_claim, None, unit_node, unit_ins))

    # Regional context
    regional_node = evaluator.add_leaf(
        id="Regional_Context",
        desc="The rock is in or near Neretva Vallis (ancient river channel region in Jezero Crater).",
        parent=ctx_node,
        critical=True
    )
    regional_claim = "The answer states the rock is in or near Neretva Vallis in Jezero Crater (ancient river channel region)."
    regional_ins = "Judge only by the answer text. Accept minor phrasing variations; the mention of Neretva Vallis should be clear."
    claims_and_sources.append((regional_claim, None, regional_node, regional_ins))

    # Surface features description
    features_node = evaluator.add_leaf(
        id="Surface_Features_Description",
        desc="The distinctive surface features are described as “leopard spots.”",
        parent=ctx_node,
        critical=True
    )
    features_claim = "The answer describes the distinctive surface features as 'leopard spots'."
    features_ins = "Judge only by the answer text. Accept quotation marks or paraphrases that explicitly use the phrase 'leopard spots'."
    claims_and_sources.append((features_claim, None, features_node, features_ins))

    # Biosignature association
    bio_node = evaluator.add_leaf(
        id="Biosignature_Association",
        desc="The sample is associated with potential biosignatures (as stated).",
        parent=ctx_node,
        critical=True
    )
    bio_claim = "The answer associates the sample with potential biosignatures."
    bio_ins = "Judge only by the answer text. Accept phrasing like 'potential biosignatures', 'possible biosignature indicators', etc."
    claims_and_sources.append((bio_claim, None, bio_node, bio_ins))

    # Peer-reviewed Nature report (September 2025)
    nature_node = evaluator.add_leaf(
        id="Peer_Reviewed_Nature_Report",
        desc="The findings are reported in a peer-reviewed Nature paper published in September 2025 (as stated).",
        parent=ctx_node,
        critical=True
    )
    nature_claim = (
        "The findings are reported in a peer-reviewed Nature paper published in September 2025."
    )
    nature_urls = filter_nature_urls(extracted_sources.urls)
    nature_ins = (
        "If URLs are provided and include a Nature page, verify the page indicates a Nature journal article "
        "with publication month/year September 2025. Otherwise, judge solely based on the answer text."
    )
    # Use Nature URLs if present; otherwise None (simple verification)
    nature_sources: List[str] | None = nature_urls if len(nature_urls) > 0 else None
    claims_and_sources.append((nature_claim, nature_sources, nature_node, nature_ins))

    # Execute verifications in parallel
    await evaluator.batch_verify(claims_and_sources)


# --------------------------------------------------------------------------- #
# Main evaluation entry                                                       #
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
    Evaluate an answer for the Bright Angel / leopard spots identification task.
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

    # Extract identifications and URLs from the answer
    identifications, sources = await asyncio.gather(
        evaluator.extract(
            prompt=prompt_extract_identifications(),
            template_class=IdentificationsExtraction,
            extraction_name="requested_identifications",
        ),
        evaluator.extract(
            prompt=prompt_extract_sources(),
            template_class=AnswerSources,
            extraction_name="answer_sources",
        )
    )

    # Build the main (critical) investigation node
    sample_investigation_node = evaluator.add_parallel(
        id="Sample_Investigation",
        desc="Evaluate whether the response identifies the required sample/rock/minerals and whether those identifications are consistent with all stated prompt constraints (mission, location, timing, features, and publication context).",
        parent=root,
        critical=True
    )

    # Build requested identifications subtree
    await build_requested_identifications_tree(
        evaluator=evaluator,
        parent_node=sample_investigation_node,
        extracted=identifications,
    )

    # Build context constraints subtree
    await build_context_constraints_tree(
        evaluator=evaluator,
        parent_node=sample_investigation_node,
        extracted_sources=sources,
    )

    # Return summary with verification tree
    return evaluator.get_summary()