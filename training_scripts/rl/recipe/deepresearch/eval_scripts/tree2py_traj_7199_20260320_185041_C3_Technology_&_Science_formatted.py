import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "nvidia_arch_successor_cpu"
TASK_DESCRIPTION = (
    "NVIDIA announced a new GPU architecture at its GTC 2024 conference on March 18, 2024. "
    "This architecture's GPUs contain exactly 208 billion transistors and are manufactured using TSMC's 4NP process. "
    "The first wafer of this architecture produced on U.S. soil was manufactured at TSMC's Arizona facility on October 17, 2025. "
    "At Computex 2024, NVIDIA announced that this architecture will be succeeded by a new GPU architecture following the company's one-year rhythm roadmap. "
    "What is the name of the ARM-based CPU that will pair with this successor GPU architecture?"
)

# Expected names per rubric
EXPECTED_BASE_ARCH = "Blackwell"
EXPECTED_SUCCESSOR_ARCH = "Rubin"
EXPECTED_CPU_NAME = "Vera"
EXPECTED_CPU_CORES = "88"
EXPECTED_CPU_THREADS = "176"


# --------------------------------------------------------------------------- #
# Extraction models                                                           #
# --------------------------------------------------------------------------- #
class BaseArchitectureExtraction(BaseModel):
    name: Optional[str] = None
    general_sources: List[str] = Field(default_factory=list)
    gtc_announcement_urls: List[str] = Field(default_factory=list)
    transistor_count_urls: List[str] = Field(default_factory=list)
    manufacturing_process_urls: List[str] = Field(default_factory=list)
    us_wafer_urls: List[str] = Field(default_factory=list)


class SuccessorArchitectureExtraction(BaseModel):
    name: Optional[str] = None
    successor_urls: List[str] = Field(default_factory=list)
    computex_urls: List[str] = Field(default_factory=list)


class CPUExtraction(BaseModel):
    name: Optional[str] = None
    pairing_urls: List[str] = Field(default_factory=list)
    spec_urls: List[str] = Field(default_factory=list)
    core_count_mentioned: Optional[str] = None
    thread_count_mentioned: Optional[str] = None


class CombinedExtraction(BaseModel):
    base_arch: Optional[BaseArchitectureExtraction] = None
    successor_arch: Optional[SuccessorArchitectureExtraction] = None
    cpu: Optional[CPUExtraction] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_combined() -> str:
    return """
Extract the specific entities and all explicit source URLs from the answer.

Return a JSON object with the following structure (use null for any missing scalar fields and [] for missing URL lists):
{
  "base_arch": {
    "name": string | null,                               // The GPU architecture name claimed as the one announced at GTC 2024 (e.g., "Blackwell")
    "general_sources": string[] ,                        // Any URLs cited about this base architecture in general
    "gtc_announcement_urls": string[],                   // URLs that support the announcement at GTC 2024 on March 18, 2024
    "transistor_count_urls": string[],                   // URLs that support the 208-billion-transistors fact
    "manufacturing_process_urls": string[],              // URLs that support use of TSMC's 4NP process
    "us_wafer_urls": string[]                            // URLs that support the first U.S. wafer at TSMC Arizona on Oct 17, 2025
  },
  "successor_arch": {
    "name": string | null,                               // The name of the successor GPU architecture (e.g., "Rubin")
    "successor_urls": string[],                          // URLs that state it succeeds the base architecture in NVIDIA's roadmap
    "computex_urls": string[]                            // URLs that specifically tie this announcement/roadmap to Computex 2024
  },
  "cpu": {
    "name": string | null,                               // The ARM-based CPU name that pairs with the successor GPU architecture (e.g., "Vera")
    "pairing_urls": string[],                            // URLs that explicitly state this CPU pairs with the successor GPU architecture
    "spec_urls": string[],                               // URLs that provide or confirm the CPU specs
    "core_count_mentioned": string | null,               // The core count mentioned in the answer for this CPU (extract as string exactly as written)
    "thread_count_mentioned": string | null              // The thread count mentioned in the answer for this CPU (extract as string exactly as written)
  }
}

Special rules:
- Extract only URLs explicitly present in the answer. Do not invent or infer any URLs.
- Accept plain URLs or markdown links; always return the actual URL.
- If a URL is missing a protocol, prepend http://
- Do not deduplicate; include each URL only once per list even if repeated in answer.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_list(lst: Optional[List[str]]) -> bool:
    return bool(lst) and len(lst) > 0


def _merge_urls(*url_lists: Optional[List[str]]) -> List[str]:
    merged: List[str] = []
    seen = set()
    for urls in url_lists:
        if not urls:
            continue
        for u in urls:
            if u and u not in seen:
                seen.add(u)
                merged.append(u)
    return merged


def _safe(obj: Optional[BaseModel]) -> BaseModel:
    return obj or BaseModel()  # type: ignore


# --------------------------------------------------------------------------- #
# Verification subroutines                                                    #
# --------------------------------------------------------------------------- #
async def build_and_verify_base_architecture(
    evaluator: Evaluator,
    parent_node,
    data: CombinedExtraction,
) -> None:
    """
    Construct the 'Identify_Base_Architecture' subtree and run verifications.
    All nodes under this subtree are critical, as required by the rubric.
    """
    base = data.base_arch or BaseArchitectureExtraction()

    # Parent node (critical, parallel as per rubric)
    base_node = evaluator.add_parallel(
        id="Identify_Base_Architecture",
        desc="Identify the GPU architecture announced at GTC 2024 on March 18, 2024, with 208B transistors and first U.S. wafer at TSMC Arizona on Oct 17, 2025",
        parent=parent_node,
        critical=True,
    )

    # 1) Provide_Architecture_Name (leaf)
    provide_name_leaf = evaluator.add_leaf(
        id="Provide_Architecture_Name",
        desc=f"Provide the name of the GPU architecture (must be {EXPECTED_BASE_ARCH})",
        parent=base_node,
        critical=True,
    )
    extracted_name = base.name or ""
    await evaluator.verify(
        claim=f"The identified base GPU architecture name '{extracted_name}' refers to the same architecture as '{EXPECTED_BASE_ARCH}'.",
        node=provide_name_leaf,
        additional_instruction="Treat minor variations (case, punctuation) as the same; if the answer did not provide a name or it differs, mark incorrect.",
    )

    # 2) Verify_Architecture_Facts (critical parallel group)
    facts_node = evaluator.add_parallel(
        id="Verify_Architecture_Facts",
        desc="Verify that the identified architecture matches all stated facts from the question",
        parent=base_node,
        critical=True,
    )

    # 2.a) GTC 2024 announcement date
    gtc_src_exist = evaluator.add_custom_node(
        result=_non_empty_list(base.gtc_announcement_urls),
        id="Confirm_GTC_Announcement_sources_present",
        desc="URL(s) provided to support GTC 2024 announcement fact",
        parent=facts_node,
        critical=True,
    )
    gtc_leaf = evaluator.add_leaf(
        id="Confirm_GTC_Announcement",
        desc="Confirm the architecture was announced at GTC 2024 on March 18, 2024, with supporting URL reference",
        parent=facts_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"NVIDIA announced the {EXPECTED_BASE_ARCH} GPU architecture at GTC 2024 on March 18, 2024.",
        node=gtc_leaf,
        sources=base.gtc_announcement_urls,
        additional_instruction="Verify the event (GTC 2024) and date (March 18, 2024) are explicitly supported by the page(s). If no valid URL is provided, this should not be supported.",
    )

    # 2.b) Transistor count = 208B
    trans_src_exist = evaluator.add_custom_node(
        result=_non_empty_list(base.transistor_count_urls),
        id="Confirm_Transistor_Count_sources_present",
        desc="URL(s) provided to support 208B transistor count",
        parent=facts_node,
        critical=True,
    )
    trans_leaf = evaluator.add_leaf(
        id="Confirm_Transistor_Count",
        desc="Confirm the architecture's GPUs contain 208 billion transistors, with supporting URL reference",
        parent=facts_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"A GPU in NVIDIA's {EXPECTED_BASE_ARCH} architecture has exactly 208 billion transistors.",
        node=trans_leaf,
        sources=base.transistor_count_urls,
        additional_instruction="Accept references to B200/Blackwell where the 208B figure is stated. If the page lists a close number but not 208B, consider not supported.",
    )

    # 2.c) Manufacturing process = TSMC 4NP
    proc_src_exist = evaluator.add_custom_node(
        result=_non_empty_list(base.manufacturing_process_urls),
        id="Confirm_Manufacturing_Process_sources_present",
        desc="URL(s) provided to support TSMC 4NP manufacturing process",
        parent=facts_node,
        critical=True,
    )
    proc_leaf = evaluator.add_leaf(
        id="Confirm_Manufacturing_Process",
        desc="Confirm the architecture uses TSMC's 4NP manufacturing process, with supporting URL reference",
        parent=facts_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The {EXPECTED_BASE_ARCH} architecture uses TSMC's 4NP manufacturing process.",
        node=proc_leaf,
        sources=base.manufacturing_process_urls,
        additional_instruction="Look for explicit mention of 'TSMC 4NP' in relation to Blackwell.",
    )

    # 2.d) First U.S. wafer at TSMC Arizona on Oct 17, 2025
    wafer_src_exist = evaluator.add_custom_node(
        result=_non_empty_list(base.us_wafer_urls),
        id="Confirm_US_Manufacturing_Date_sources_present",
        desc="URL(s) provided to support first U.S. wafer at TSMC Arizona on Oct 17, 2025",
        parent=facts_node,
        critical=True,
    )
    wafer_leaf = evaluator.add_leaf(
        id="Confirm_US_Manufacturing_Date",
        desc="Confirm the first U.S. wafer was produced at TSMC Arizona on October 17, 2025, with supporting URL reference",
        parent=facts_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The first wafer of NVIDIA's {EXPECTED_BASE_ARCH} architecture produced on U.S. soil was manufactured at TSMC's Arizona facility on October 17, 2025.",
        node=wafer_leaf,
        sources=base.us_wafer_urls,
        additional_instruction="This must be explicitly supported (date and location). If sources are missing or irrelevant, mark as not supported.",
    )


async def build_and_verify_successor_architecture(
    evaluator: Evaluator,
    parent_node,
    data: CombinedExtraction,
) -> None:
    """
    Construct the 'Identify_Successor_Architecture' subtree and run verifications.
    All nodes under this subtree are critical, as required by the rubric.
    """
    succ = data.successor_arch or SuccessorArchitectureExtraction()

    succ_node = evaluator.add_parallel(
        id="Identify_Successor_Architecture",
        desc="Identify the GPU architecture that succeeds the base architecture following NVIDIA's roadmap",
        parent=parent_node,
        critical=True,
    )

    # Provide_Successor_Name
    provide_succ_leaf = evaluator.add_leaf(
        id="Provide_Successor_Name",
        desc=f"Provide the name of the successor GPU architecture (must be {EXPECTED_SUCCESSOR_ARCH})",
        parent=succ_node,
        critical=True,
    )
    extracted_successor = succ.name or ""
    await evaluator.verify(
        claim=f"The identified successor GPU architecture name '{extracted_successor}' refers to the same architecture as '{EXPECTED_SUCCESSOR_ARCH}'.",
        node=provide_succ_leaf,
        additional_instruction="Treat minor variations (case, punctuation) as the same; if the name is missing or different, mark incorrect.",
    )

    # Verify_Succession_Relationship
    urls = _merge_urls(succ.successor_urls, succ.computex_urls)
    urls_exist = evaluator.add_custom_node(
        result=_non_empty_list(urls),
        id="Verify_Succession_Relationship_sources_present",
        desc="URL(s) provided that state the successor follows the base architecture (ideally Computex 2024)",
        parent=succ_node,
        critical=True,
    )
    verify_succ_leaf = evaluator.add_leaf(
        id="Verify_Succession_Relationship",
        desc="Confirm that the successor architecture was announced as following the base architecture, with supporting URL reference",
        parent=succ_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"NVIDIA announced that the {EXPECTED_SUCCESSOR_ARCH} GPU architecture would succeed {EXPECTED_BASE_ARCH} in its roadmap, associated with Computex 2024 and a one-year cadence.",
        node=verify_succ_leaf,
        sources=urls,
        additional_instruction="Prefer Computex 2024 sources; explicit language that Rubin succeeds Blackwell should be present.",
    )


async def build_and_verify_paired_cpu(
    evaluator: Evaluator,
    parent_node,
    data: CombinedExtraction,
) -> None:
    """
    Construct the 'Identify_Paired_CPU' subtree and run verifications.
    All nodes under this subtree are critical, as required by the rubric.
    """
    cpu = data.cpu or CPUExtraction()

    cpu_node = evaluator.add_parallel(
        id="Identify_Paired_CPU",
        desc="Identify the ARM-based CPU that pairs with the successor GPU architecture",
        parent=parent_node,
        critical=True,
    )

    # Provide_CPU_Name
    provide_cpu_leaf = evaluator.add_leaf(
        id="Provide_CPU_Name",
        desc=f"Provide the correct name of the ARM-based CPU (must be {EXPECTED_CPU_NAME})",
        parent=cpu_node,
        critical=True,
    )
    extracted_cpu_name = cpu.name or ""
    await evaluator.verify(
        claim=f"The identified ARM-based CPU name '{extracted_cpu_name}' refers to the same CPU as '{EXPECTED_CPU_NAME}'.",
        node=provide_cpu_leaf,
        additional_instruction="Treat minor variations (case, punctuation) as the same; if the name is missing or different, mark incorrect.",
    )

    # Verify_CPU_Pairing
    pair_urls_exist = evaluator.add_custom_node(
        result=_non_empty_list(cpu.pairing_urls),
        id="Verify_CPU_Pairing_sources_present",
        desc="URL(s) provided that the CPU pairs with the successor GPU architecture",
        parent=cpu_node,
        critical=True,
    )
    verify_pair_leaf = evaluator.add_leaf(
        id="Verify_CPU_Pairing",
        desc="Confirm that the CPU pairs with the successor GPU architecture, with supporting URL reference",
        parent=cpu_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"NVIDIA's ARM-based CPU '{EXPECTED_CPU_NAME}' is designed to pair with (or be the matching CPU for) the '{EXPECTED_SUCCESSOR_ARCH}' GPU architecture (the successor to {EXPECTED_BASE_ARCH}).",
        node=verify_pair_leaf,
        sources=cpu.pairing_urls,
        additional_instruction="Look for explicit wording that Vera is the matching/pairing CPU for Rubin or the platform succeeding Blackwell.",
    )

    # Verify_CPU_Specifications
    spec_urls_exist = evaluator.add_custom_node(
        result=_non_empty_list(cpu.spec_urls),
        id="Verify_CPU_Specifications_sources_present",
        desc="URL(s) provided that confirm CPU core/thread counts",
        parent=cpu_node,
        critical=True,
    )
    verify_specs_leaf = evaluator.add_leaf(
        id="Verify_CPU_Specifications",
        desc="Confirm the CPU has 88 custom ARM cores and 176 threads, with supporting URL reference",
        parent=cpu_node,
        critical=True,
    )
    await evaluator.verify(
        claim=f"The '{EXPECTED_CPU_NAME}' CPU has {EXPECTED_CPU_CORES} custom ARM cores and {EXPECTED_CPU_THREADS} threads.",
        node=verify_specs_leaf,
        sources=cpu.spec_urls,
        additional_instruction="The page should explicitly list both the core count (88) and thread count (176).",
    )


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
    Evaluate an answer for the NVIDIA architecture successor CPU task.
    """
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

    # Extract structured data
    extracted = await evaluator.extract(
        prompt=prompt_extract_combined(),
        template_class=CombinedExtraction,
        extraction_name="structured_extraction",
    )

    # Add ground truth (expected values per rubric)
    evaluator.add_ground_truth(
        {
            "expected_base_architecture": EXPECTED_BASE_ARCH,
            "expected_successor_architecture": EXPECTED_SUCCESSOR_ARCH,
            "expected_cpu_name": EXPECTED_CPU_NAME,
            "expected_cpu_specs": {
                "cores": EXPECTED_CPU_CORES,
                "threads": EXPECTED_CPU_THREADS,
            },
            "key_facts": {
                "gtc_announcement_date": "March 18, 2024",
                "transistor_count": "208 billion",
                "manufacturing_process": "TSMC 4NP",
                "first_us_wafer_date_place": "October 17, 2025 @ TSMC Arizona",
                "computex_successor_context": "Computex 2024 one-year cadence",
            },
        },
        gt_type="expected_facts",
    )

    # Build the full rubric tree
    top_node = evaluator.add_sequential(
        id="Identify_CPU_for_Architecture_Successor",
        desc="Identify the ARM-based CPU that pairs with the successor to the GPU architecture described in the question",
        parent=root,
        critical=True,
    )

    # 1) Identify_Base_Architecture (critical)
    await build_and_verify_base_architecture(evaluator, top_node, extracted)

    # 2) Identify_Successor_Architecture (critical)
    await build_and_verify_successor_architecture(evaluator, top_node, extracted)

    # 3) Identify_Paired_CPU (critical)
    await build_and_verify_paired_cpu(evaluator, top_node, extracted)

    # Return standard summary
    return evaluator.get_summary()