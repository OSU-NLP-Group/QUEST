import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.llm_client.base_client import LLMClient


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nif_feb12_2024_yield"
TASK_DESCRIPTION = (
    "What was the fusion energy yield, in megajoules, achieved in the National Ignition Facility experiment conducted on February 12, 2024?"
)


# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class NIFExtraction(BaseModel):
    """
    Structured extraction from the agent's answer.
    """
    experiment_date: Optional[str] = None
    facility: Optional[str] = None
    fusion_yield_mj: Optional[str] = None
    laser_input_mj: Optional[str] = None
    yield_more_than_double: Optional[bool] = None
    source_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_nif_info() -> str:
    return """
    Extract the following fields from the answer text about the National Ignition Facility (NIF) experiment:
    - experiment_date: The experiment date string as explicitly stated (e.g., "February 12, 2024", "Feb 12, 2024", "2024-02-12", or "12 February 2024"). If not present, return null.
    - facility: The named facility where the experiment was conducted (e.g., "National Ignition Facility", "NIF", "NIF at LLNL"). If not present, return null.
    - fusion_yield_mj: The stated fusion energy yield value in megajoules as text from the answer (e.g., "5.2 MJ", "5.20 MJ", "~5.2 MJ"). Do not convert; extract exactly as stated. If not present, return null.
    - laser_input_mj: The stated laser input energy value in megajoules as text from the answer (e.g., "2.2 MJ"). Do not convert; extract exactly as stated. If not present, return null.
    - yield_more_than_double: A boolean. True if the answer explicitly describes that the yield is more than double the input energy (e.g., phrases like "more than double", "over twice", "greater than 2x", "gain > 2"), otherwise false. If unclear or not stated, set to false.
    - source_urls: List all URLs explicitly provided in the answer as sources or references. Include full URLs (prepend http:// if protocol is missing). If none, return an empty list.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _parse_first_float(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    # Find first floating-point number in the string
    m = re.search(r"([-+]?\d+(?:[.,]\d+)?)", text)
    if not m:
        return None
    num_str = m.group(1).replace(",", ".")
    try:
        return float(num_str)
    except Exception:
        return None


def _filter_official_urls(urls: List[str]) -> List[str]:
    """
    Keep only official LLNL/NIF URLs.
    For this task, we treat URLs containing 'llnl.gov' as official (e.g., llnl.gov, lasers.llnl.gov).
    """
    official = []
    for u in urls:
        if not isinstance(u, str):
            continue
        low = u.lower()
        if "llnl.gov" in low:
            official.append(u)
    return official


# --------------------------------------------------------------------------- #
# Verification tree construction and checks                                   #
# --------------------------------------------------------------------------- #
async def build_main_verification(
    evaluator: Evaluator,
    parent_node,
    extracted: NIFExtraction
) -> None:
    """
    Construct the verification nodes according to the rubric and issue verifications.
    """
    # Create the main critical (parallel) node per rubric
    main_node = evaluator.add_parallel(
        id="February_12_2024_NIF_Fusion_Yield",
        desc="Answer reports the fusion energy yield for the National Ignition Facility experiment conducted on February 12, 2024, meeting all stated constraints.",
        parent=parent_node,
        critical=True
    )

    # Prepare source URLs (prefer official LLNL/NIF URLs when verifying against webpages)
    all_urls = extracted.source_urls or []
    official_urls = _filter_official_urls(all_urls)
    urls_for_verification = official_urls if len(official_urls) > 0 else (all_urls if len(all_urls) > 0 else None)

    # 1) Experiment date is Feb 12, 2024
    date_node = evaluator.add_leaf(
        id="Experiment_Date_Is_Feb_12_2024",
        desc="States the experiment date as February 12, 2024.",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly states that the experiment was conducted on February 12, 2024.",
        node=date_node,
        additional_instruction="Accept common variants like 'Feb 12, 2024', '12 February 2024', or '2024-02-12'. The key is that the date corresponds to 2024-02-12."
    )

    # 2) Experiment conducted at NIF
    facility_node = evaluator.add_leaf(
        id="Experiment_Conducted_At_NIF",
        desc="Identifies the experiment as conducted at the National Ignition Facility (NIF).",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer identifies the experiment as conducted at the National Ignition Facility (NIF).",
        node=facility_node,
        additional_instruction="Allow equivalent phrasings like 'NIF at LLNL', 'NIF (National Ignition Facility)', or 'National Ignition Facility at Lawrence Livermore National Laboratory'."
    )

    # 3) Fusion energy yield is 5.2 MJ
    yield_node = evaluator.add_leaf(
        id="Fusion_Energy_Yield_Is_5_2_MJ",
        desc="Reports the fusion energy yield as 5.2 megajoules (MJ).",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The fusion energy yield is 5.2 megajoules (MJ).",
        node=yield_node,
        sources=urls_for_verification,  # Prefer verifying against cited source pages if available
        additional_instruction="Verify that the webpage (if provided) explicitly supports a 5.2 MJ fusion energy yield. Allow minor formatting like '5.20 MJ' or '≈5.2 MJ'."
    )

    # 4) Laser input energy is 2.2 MJ
    input_node = evaluator.add_leaf(
        id="Laser_Input_Energy_Is_2_2_MJ",
        desc="Reports the laser input energy as 2.2 megajoules (MJ).",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The laser input energy is 2.2 megajoules (MJ).",
        node=input_node,
        sources=urls_for_verification,  # Prefer verifying against cited source pages if available
        additional_instruction="Verify that the webpage (if provided) explicitly supports a 2.2 MJ laser input energy figure. Allow minor formatting like '2.20 MJ'."
    )

    # 5) Yield more than doubles input (i.e., yield > 2 × input)
    # The rubric says "Describes the yield as more than doubling the input".
    # We verify that the answer explicitly includes such phrasing (e.g., 'more than double', 'over twice', 'gain > 2', 'Q>2').
    double_desc_node = evaluator.add_leaf(
        id="Yield_More_Than_Doubles_Input",
        desc="Describes the yield as more than doubling the input energy (i.e., yield > 2 × input).",
        parent=main_node,
        critical=True
    )
    await evaluator.verify(
        claim="The answer explicitly describes that the fusion yield was more than double the input energy (e.g., 'more than double', 'over twice', 'greater than 2x', 'gain > 2', or 'Q>2').",
        node=double_desc_node,
        additional_instruction="Look for explicit descriptive language indicating >2×, not just the raw numbers."
    )

    # 6) Provides an official LLNL or NIF source URL
    has_official_source = len(official_urls) > 0
    evaluator.add_custom_node(
        result=has_official_source,
        id="Official_LLNL_or_NIF_Source_URL_Provided",
        desc="Provides a reference URL from an official LLNL or NIF source that supports the stated figures/claim.",
        parent=main_node,
        critical=True
    )


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
) -> Dict[str, Any]:
    """
    Evaluate an answer for the NIF Feb 12, 2024 fusion yield task.
    """
    # 1) Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # One main rubric section
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

    # 2) Extract structured info from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_nif_info(),
        template_class=NIFExtraction,
        extraction_name="nif_extraction"
    )

    # 3) Optional: add expected reference info (for transparency)
    evaluator.add_ground_truth({
        "expected_date": "February 12, 2024",
        "expected_facility": "National Ignition Facility (NIF)",
        "expected_fusion_yield_mj": "5.2 MJ",
        "expected_laser_input_mj": "2.2 MJ",
        "note": "Official LLNL/NIF sources should be provided (domain contains llnl.gov)."
    })

    # 4) Build verification nodes and run checks
    await build_main_verification(evaluator, root, extracted)

    # 5) Return evaluation summary
    return evaluator.get_summary()