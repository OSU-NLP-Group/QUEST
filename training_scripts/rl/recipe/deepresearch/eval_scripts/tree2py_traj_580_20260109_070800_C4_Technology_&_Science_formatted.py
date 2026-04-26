import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "highest_memory_chip_2024_2025"
TASK_DESCRIPTION = """
Identify the data center AI accelerator chip that was announced or released during 2024-2025 with the highest memory capacity. Provide the following information: the specific chip model name, the manufacturer, the memory capacity (with units), the memory type (HBM generation), the memory bandwidth (with units), the release or announcement timeframe, and a reference URL to the official product page or announcement.
"""

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ChipInfo(BaseModel):
    chip_model_name: Optional[str] = None
    manufacturer: Optional[str] = None
    memory_capacity: Optional[str] = None  # Keep as free-form string with units (prefer GB)
    memory_type: Optional[str] = None      # e.g., HBM3, HBM3E
    memory_bandwidth: Optional[str] = None # e.g., 8 TB/s
    timeframe: Optional[str] = None        # e.g., "Announced in March 2024", "Released Q4 2025"
    official_url: Optional[str] = None     # Manufacturer product page or official announcement
    support_urls: List[str] = Field(default_factory=list)  # Any additional URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chip_info() -> str:
    return """
    Extract the information for the single data center AI accelerator chip identified in the answer.
    Return a JSON object with these fields:
    - chip_model_name: The exact chip model name.
    - manufacturer: The company/manufacturer name.
    - memory_capacity: The memory capacity string including a value and unit (prefer GB if available, otherwise keep as stated).
    - memory_type: The HBM generation (e.g., HBM3, HBM3E) if stated; otherwise keep the memory type string as stated in the answer.
    - memory_bandwidth: The memory bandwidth string including value and units (e.g., TB/s), exactly as in the answer.
    - timeframe: The release or announcement timeframe/dates (e.g., "Announced March 2024", "Released in 2025").
    - official_url: The official manufacturer product page or official announcement URL cited in the answer.
    - support_urls: A list of any additional URLs (beyond the official_url) that the answer cites to support its claims (e.g., press coverage, spec pages).
    
    Important:
    - Only extract information that appears explicitly in the provided answer text.
    - If a field is missing, set it to null (for single-value fields) or [] (for lists).
    - For URLs, extract actual URLs (including protocol). If markdown links are used, extract the destination URL.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _normalize_sources(info: ChipInfo) -> List[str]:
    """Combine official_url and support_urls into a single list of sources."""
    sources: List[str] = []
    if info.official_url and info.official_url.strip():
        sources.append(info.official_url.strip())
    for u in info.support_urls:
        if isinstance(u, str) and u.strip():
            sources.append(u.strip())
    # Deduplicate while preserving order
    seen = set()
    unique_sources = []
    for u in sources:
        if u not in seen:
            unique_sources.append(u)
            seen.add(u)
    return unique_sources


def _has_gb_units(s: Optional[str]) -> bool:
    if not s:
        return False
    lower = s.lower()
    return "gb" in lower or "gib" in lower  # Accept GiB variant


def _has_tbps_units(s: Optional[str]) -> bool:
    if not s:
        return False
    lower = s.lower()
    return ("tb/s" in lower) or ("tbps" in lower) or ("terabytes per second" in lower)


def _looks_like_hbm(s: Optional[str]) -> bool:
    if not s:
        return False
    lower = s.lower()
    return "hbm" in lower  # e.g., hbm2e, hbm3, hbm3e


def _is_valid_url(s: Optional[str]) -> bool:
    if not s:
        return False
    s = s.strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def _has_timeframe_year_2024_2025(s: Optional[str]) -> bool:
    if not s:
        return False
    return ("2024" in s) or ("2025" in s)


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_selection_constraints_checks(
    evaluator: Evaluator,
    parent_node,
    info: ChipInfo
) -> None:
    """
    Build and execute verification nodes under 'chip_meets_selection_constraints' (critical parallel).
    """
    constraints_node = evaluator.add_parallel(
        id="chip_meets_selection_constraints",
        desc="The identified chip satisfies the selection constraints (eligibility and highest-memory criterion).",
        parent=parent_node,
        critical=True
    )

    # 1) Is data center AI accelerator designed for AI/ML workloads (Critical leaf)
    is_dc_ai_node = evaluator.add_leaf(
        id="is_data_center_ai_accelerator_for_ai_ml",
        desc="The chip is a data center AI accelerator designed for AI/ML workloads.",
        parent=constraints_node,
        critical=True
    )
    accel_claim = (
        f"The {info.manufacturer or ''} {info.chip_model_name or 'chip'} is a data center AI accelerator intended for AI/ML workloads."
    )
    await evaluator.verify(
        claim=accel_claim,
        node=is_dc_ai_node,
        sources=_normalize_sources(info),
        additional_instruction=(
            "Use only the provided webpages. Look for language such as 'data center', 'server', 'AI accelerator', "
            "'GPU for training/inference', 'HBM memory', 'NVLink', or similar. Consumer or edge-only products "
            "should not be considered data center accelerators."
        )
    )

    # 2) Timeframe provided and falls within 2024-2025
    #    Break into two steps: (a) field exists; (b) supported by sources that it was announced/released in 2024 or 2025.
    timeframe_main = evaluator.add_sequential(
        id="timeframe_provided_and_within_2024_2025",
        desc="A release or announcement timeframe is provided and it falls within 2024-2025.",
        parent=constraints_node,
        critical=True
    )

    timeframe_exists = evaluator.add_custom_node(
        result=(info.timeframe is not None and info.timeframe.strip() != ""),
        id="timeframe_field_provided",
        desc="Timeframe field is provided in the answer.",
        parent=timeframe_main,
        critical=True
    )

    timeframe_within_node = evaluator.add_leaf(
        id="timeframe_within_2024_2025_supported",
        desc="The chip was announced or released in 2024 or 2025 (supported by sources).",
        parent=timeframe_main,
        critical=True
    )
    timeframe_claim = (
        f"The {info.chip_model_name or 'chip'} was announced or released in 2024 or 2025."
    )
    await evaluator.verify(
        claim=timeframe_claim,
        node=timeframe_within_node,
        sources=_normalize_sources(info),
        additional_instruction=(
            "Verify any explicit dates or timeframe statements on the provided pages. "
            "Accept statements like 'Announced at [event] 2024' or 'Released in 2025'."
        )
    )

    # 3) Highest memory among eligible chips (Critical leaf)
    highest_mem_node = evaluator.add_leaf(
        id="highest_memory_among_eligible_2024_2025_chips",
        desc="Among data center AI accelerator chips announced/released in 2024-2025, the identified chip has the highest memory capacity.",
        parent=constraints_node,
        critical=True
    )
    highest_claim = (
        f"Among data center AI accelerator chips announced or released in 2024–2025, "
        f"the {info.chip_model_name or 'chip'} has the highest memory capacity."
    )
    await evaluator.verify(
        claim=highest_claim,
        node=highest_mem_node,
        sources=_normalize_sources(info),
        additional_instruction=(
            "Use only the provided webpages. Look for an explicit comparative statement indicating the chip has the "
            "highest memory capacity among its contemporaries (2024–2025). If no page clearly states this, the claim "
            "is not supported."
        )
    )


async def build_required_fields_checks(
    evaluator: Evaluator,
    parent_node,
    info: ChipInfo
) -> None:
    """
    Build and execute verification nodes under 'required_fields_provided' (critical parallel),
    primarily as existence/format checks (custom nodes).
    """
    req_node = evaluator.add_parallel(
        id="required_fields_provided",
        desc="All required fields are provided with appropriate units/format.",
        parent=parent_node,
        critical=True
    )

    # Manufacturer provided
    evaluator.add_custom_node(
        result=(info.manufacturer is not None and info.manufacturer.strip() != ""),
        id="manufacturer_provided",
        desc="The manufacturer name is provided.",
        parent=req_node,
        critical=True
    )

    # Memory capacity value and units in GB (or GiB)
    evaluator.add_custom_node(
        result=_has_gb_units(info.memory_capacity),
        id="memory_capacity_value_and_units_gb",
        desc="The memory capacity is stated with a value and units in GB.",
        parent=req_node,
        critical=True
    )

    # Memory type HBM generation specified
    evaluator.add_custom_node(
        result=_looks_like_hbm(info.memory_type),
        id="memory_type_hbm_generation_specified",
        desc="The memory type is specified as an HBM generation (e.g., HBM3, HBM3E).",
        parent=req_node,
        critical=True
    )

    # Memory bandwidth value and units in TB/s (or equivalent)
    evaluator.add_custom_node(
        result=_has_tbps_units(info.memory_bandwidth),
        id="memory_bandwidth_value_and_units",
        desc="The memory bandwidth is stated with a value and units in TB/s.",
        parent=req_node,
        critical=True
    )

    # Official reference URL provided
    evaluator.add_custom_node(
        result=_is_valid_url(info.official_url),
        id="official_reference_url_provided",
        desc="An official reference URL is provided (manufacturer product page or official announcement).",
        parent=req_node,
        critical=True
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
    Evaluate the answer for the 'highest_memory_chip_2024_2025' task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Root stays non-critical; main task node added under root
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

    # Extract chip info from the answer
    chip_info = await evaluator.extract(
        prompt=prompt_extract_chip_info(),
        template_class=ChipInfo,
        extraction_name="chip_info"
    )

    # Build main critical sequential node for the task
    main_node = evaluator.add_sequential(
        id="highest_memory_chip_2024_2025",
        desc="Identify the data center AI accelerator chip announced or released in 2024-2025 with the highest memory capacity, and provide the required specifications and an official reference URL.",
        parent=root,
        critical=True
    )

    # Step 1: Chip model name provided (Critical existence check)
    evaluator.add_custom_node(
        result=(chip_info.chip_model_name is not None and chip_info.chip_model_name.strip() != ""),
        id="chip_model_name_provided",
        desc="A specific chip model name is provided.",
        parent=main_node,
        critical=True
    )

    # Step 2: Selection constraints (Critical parallel checks)
    await build_selection_constraints_checks(evaluator, main_node, chip_info)

    # Step 3: Required fields provided (Critical parallel checks)
    await build_required_fields_checks(evaluator, main_node, chip_info)

    # Add small custom info for context (optional)
    evaluator.add_custom_info(
        info={
            "normalized_sources": _normalize_sources(chip_info),
            "timeframe_contains_2024_or_2025": _has_timeframe_year_2024_2025(chip_info.timeframe)
        },
        info_type="helper_signals",
        info_name="helper_signals"
    )

    # Return evaluation summary
    return evaluator.get_summary()