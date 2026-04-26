import asyncio
import logging
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "analog_ai_chip_eval"
TASK_DESCRIPTION = (
    "Identify an analog AI chip that delivers at least 25 TOPS of AI compute performance while consuming 10 watts "
    "or less of power, incorporates on-chip memory or in-memory computing capabilities without requiring external DRAM "
    "for weight parameter storage, is commercially available or officially announced with documented specifications, "
    "and is designed for AI inference workloads."
)

# --------------------------------------------------------------------------- #
# Data models for extracted information                                       #
# --------------------------------------------------------------------------- #
class ChipExtraction(BaseModel):
    """Structured information extracted from the agent's answer for a single analog AI chip."""
    vendor: Optional[str] = None
    chip_model: Optional[str] = None

    # Constraint-related claims as stated in the answer (free-form strings)
    analog_tech_desc: Optional[str] = None
    performance_tops: Optional[str] = None
    power_consumption_watts: Optional[str] = None
    memory_architecture_desc: Optional[str] = None
    availability_status: Optional[str] = None
    inference_workload_desc: Optional[str] = None

    # All source URLs explicitly mentioned in the answer
    sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_chip_info() -> str:
    return (
        "Extract exactly one specific analog AI chip referenced in the answer and the constraint-related details "
        "the answer claims for that chip. If multiple chips are mentioned, choose the primary chip the answer "
        "emphasizes (or the first concrete chip mentioned) to be evaluated. Return the following fields:\n"
        "1) vendor: The company or vendor name (e.g., 'CompanyName').\n"
        "2) chip_model: The specific chip's model/name (e.g., 'Chip X100').\n"
        "3) analog_tech_desc: Any phrasing that indicates analog computing, compute-in-memory, analog matrix processing, "
        "   or mixed-signal arrays, as stated in the answer.\n"
        "4) performance_tops: The stated AI compute performance figure(s) for the chip, extracted exactly as written "
        "   (e.g., '30 TOPS', '>= 25 TOPS').\n"
        "5) power_consumption_watts: The stated power consumption fact(s) for the chip, extracted exactly as written "
        "   (e.g., '5 W typical', '≤ 10 W').\n"
        "6) memory_architecture_desc: Any statement about on-chip memory or in-memory computing, especially indicating "
        "   that external DRAM is not required for weight storage.\n"
        "7) availability_status: Any statement about commercial availability or official announcement and the existence "
        "   of publicly documented specifications (e.g., a product page or press release).\n"
        "8) inference_workload_desc: Any statement indicating the chip is designed for or optimized for AI inference.\n"
        "9) sources: A list of ALL URLs explicitly mentioned in the answer (including markdown links). If a URL is missing "
        "   protocol, prepend http://. Only include valid URLs. If none are present, return an empty list.\n\n"
        "Return null for any field that is not present in the answer. Do not invent or infer missing information."
    )


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _chip_label(extracted: ChipExtraction) -> str:
    vendor = (extracted.vendor or "").strip()
    model = (extracted.chip_model or "").strip()
    if vendor and model:
        return f"{vendor} {model}"
    return vendor or model or "the identified chip"


# --------------------------------------------------------------------------- #
# Verification sub-tree construction                                          #
# --------------------------------------------------------------------------- #
async def build_verification_tree(
    evaluator: Evaluator,
    root: Any,
    extracted: ChipExtraction,
) -> None:
    """
    Build the verification tree according to the rubric and run the checks.
    """
    # 1) Top-level sequential critical node for identification and constraints
    analog_node = evaluator.add_sequential(
        id="analog_ai_chip_identification",
        desc="Identify a specific analog AI chip and verify it meets all stated performance, power, memory-architecture, availability, and inference-workload requirements.",
        parent=root,
        critical=True,
    )

    # 2) Check that the answer specifies a particular chip (vendor + model)
    identified = bool((extracted.vendor or "").strip()) and bool((extracted.chip_model or "").strip())
    evaluator.add_custom_node(
        result=identified,
        id="chip_is_identified",
        desc="The answer specifies a particular chip (at minimum: company/vendor and chip model/name) to be evaluated against the constraints.",
        parent=analog_node,
        critical=True,
    )

    # 3) Parallel critical constraints node
    constraints_node = evaluator.add_parallel(
        id="meets_all_constraints",
        desc="The identified chip satisfies all stated constraints in the question/constraints section.",
        parent=analog_node,
        critical=True,
    )

    # Prepare common info
    chip_name = _chip_label(extracted)
    sources = extracted.sources if extracted and extracted.sources else None

    # 3.1 Analog computing technology
    analog_leaf = evaluator.add_leaf(
        id="analog_computing_technology",
        desc="The chip uses analog computing technology (compute-in-memory or analog matrix processing) rather than a purely digital architecture.",
        parent=constraints_node,
        critical=True,
    )
    analog_claim = (
        f"The chip {chip_name} uses analog computing technology (e.g., compute-in-memory, analog matrix processing, "
        f"or mixed-signal arrays) rather than being purely digital."
    )
    await evaluator.verify(
        claim=analog_claim,
        node=analog_leaf,
        sources=sources,
        additional_instruction=(
            "Look for phrases such as 'analog', 'compute-in-memory', 'in-memory computing', 'analog matrix multiplication', "
            "'ReRAM/Flash CIM', or 'mixed-signal array'. The claim should be explicitly supported by the cited page(s). "
            "If the page describes the chip solely as a conventional digital NPU/GPU/CPU without analog CIM or analog MAC arrays, fail."
        ),
    )

    # 3.2 Performance requirement (>= 25 TOPS)
    perf_leaf = evaluator.add_leaf(
        id="performance_requirement",
        desc="The chip delivers at least 25 TOPS of AI compute performance.",
        parent=constraints_node,
        critical=True,
    )
    perf_claim = f"The chip {chip_name} delivers at least 25 TOPS of AI compute performance."
    await evaluator.verify(
        claim=perf_claim,
        node=perf_leaf,
        sources=sources,
        additional_instruction=(
            "Verify that the page explicitly states a performance at or above 25 TOPS (aka TeraOPS/s). "
            "Allow minor formatting variants (e.g., '25+ TOPS', '≥25 TOPS', '30 TOPS'). If only unrelated metrics are provided "
            "and no clear TOPS >= 25 is shown, mark as not supported."
        ),
    )

    # 3.3 Power consumption limit (<= 10 W)
    power_leaf = evaluator.add_leaf(
        id="power_consumption_limit",
        desc="The chip operates with a power consumption of 10 watts or less.",
        parent=constraints_node,
        critical=True,
    )
    power_claim = f"The chip {chip_name} operates at 10 watts or less of power."
    await evaluator.verify(
        claim=power_claim,
        node=power_leaf,
        sources=sources,
        additional_instruction=(
            "Look for explicit power figures that are ≤ 10 W (typical or max, clearly indicated). "
            "If the only stated figures exceed 10 W or are ambiguous without a ≤10 W operating mode, fail."
        ),
    )

    # 3.4 On-chip memory / in-memory computing (no external DRAM needed for weights)
    mem_leaf = evaluator.add_leaf(
        id="on_chip_memory_architecture",
        desc="The chip incorporates on-chip memory or in-memory computing capabilities such that external DRAM is not required for weight parameter storage.",
        parent=constraints_node,
        critical=True,
    )
    mem_claim = (
        f"The chip {chip_name} incorporates on-chip memory or in-memory computing for weights, "
        f"such that external DRAM is not required for weight parameter storage."
    )
    await evaluator.verify(
        claim=mem_claim,
        node=mem_leaf,
        sources=sources,
        additional_instruction=(
            "The page should explicitly indicate that model weights are stored on-chip (e.g., in Flash/ReRAM/SRAM arrays) "
            "or via compute-in-memory, and that no external DRAM is required for weights. "
            "If the page indicates reliance on external DRAM for weight storage, fail."
        ),
    )

    # 3.5 Commercial availability or official announcement with documented specs
    avail_leaf = evaluator.add_leaf(
        id="commercial_availability",
        desc="The chip is commercially available or officially announced, and its specifications are publicly documented.",
        parent=constraints_node,
        critical=True,
    )
    avail_claim = (
        f"The chip {chip_name} is commercially available or officially announced, and it has publicly documented specifications."
    )
    await evaluator.verify(
        claim=avail_claim,
        node=avail_leaf,
        sources=sources,
        additional_instruction=(
            "Accept an official product page or press release that clearly documents specifications or lists a spec table. "
            "Research-only or academic prototype pages without an official announcement/spec documentation should fail."
        ),
    )

    # 3.6 AI inference capability
    infer_leaf = evaluator.add_leaf(
        id="ai_inference_capability",
        desc="The chip is designed and optimized for AI inference workloads.",
        parent=constraints_node,
        critical=True,
    )
    infer_claim = f"The chip {chip_name} is designed and optimized for AI inference workloads."
    await evaluator.verify(
        claim=infer_claim,
        node=infer_leaf,
        sources=sources,
        additional_instruction=(
            "Confirm that the page explicitly positions the chip for inference (e.g., 'inference accelerator', "
            "'optimized for inference', or mentions accelerating inference tasks). "
            "If the page focuses solely on training without inference or does not address inference, fail."
        ),
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
    model: str = "o4-mini",
) -> Dict:
    """
    Entry point to evaluate an agent's answer against the analog AI chip rubric.
    """
    # Initialize evaluator with a sequential root to reflect staged validation
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

    # Extract chip info and constraint-related statements from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_chip_info(),
        template_class=ChipExtraction,
        extraction_name="analog_chip_extraction",
    )

    # Add custom info for transparency (optional)
    evaluator.add_custom_info(
        info={
            "thresholds": {"min_tops": "25 TOPS", "max_power": "10 W"},
            "chip_extracted": _chip_label(extracted),
            "sources_count": len(extracted.sources) if extracted and extracted.sources else 0,
        },
        info_type="thresholds_and_chip",
        info_name="evaluation_parameters",
    )

    # Build verification tree and run checks
    await build_verification_tree(evaluator, root, extracted)

    # Return structured summary
    return evaluator.get_summary()