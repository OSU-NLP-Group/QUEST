import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "console_specs_2025_us"
TASK_DESCRIPTION = """
A gamer in the United States is looking to purchase a major gaming console in late 2025 that meets specific technical requirements for their gaming setup. They need a console that satisfies ALL of the following criteria:

1. Has at least 16 GB of system RAM/memory
2. Supports backwards compatibility with games from the previous console generation
3. Has at least 800 GB of internal storage capacity
4. Uses an AMD-based processor (CPU) architecture

Identify which major gaming console currently available in the United States meets all these requirements, and provide the following technical specifications for that console:
- The number of CPU cores
- The GPU computing performance measured in TFLOPS
- An official reference URL that confirms these technical specifications
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ConsoleSpecExtraction(BaseModel):
    """
    Flattened extraction of the selected console and its key claims/specifications
    exactly as presented in the answer text.
    """
    selected_console: Optional[str] = None

    # Availability claim (as stated in the answer)
    us_availability_mentioned: Optional[bool] = None

    # Eligibility requirement claims (free text or summarized booleans)
    ram: Optional[str] = None                      # e.g., "16 GB GDDR6", "16GB unified memory"
    storage: Optional[str] = None                  # e.g., "825 GB SSD", "1 TB internal storage"
    backward_compatibility_prev_gen: Optional[bool] = None
    cpu_architecture: Optional[str] = None         # e.g., "AMD Zen 2", "Custom AMD SoC"

    # Requested final specs
    cpu_core_count: Optional[str] = None           # keep as string to allow formats like "8 cores/16 threads"
    gpu_tflops: Optional[str] = None               # keep as string (e.g., "12", "10.28", "12.0")

    # URLs provided by the answer
    official_spec_url: Optional[str] = None        # Manufacturer or official spec/support page if given
    additional_urls: List[str] = Field(default_factory=list)  # Any other URLs cited in the answer


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_console_spec() -> str:
    return """
    Extract the console selection and the requested technical details exactly as stated in the answer text.

    Return a JSON object with the following fields:
    - selected_console: The explicit console model the answer recommends/chooses to meet all requirements.
      If multiple consoles are mentioned, pick the one the answer claims meets the requirements or the one most clearly recommended.
      If unclear, pick the first major console explicitly proposed as the solution. Otherwise null.
    - us_availability_mentioned: true if the answer explicitly states or clearly implies that the selected console is available
      in the United States in late 2025; false if it explicitly says it is not available; null if the answer does not address availability.
    - ram: The RAM/memory spec text for the selected console if provided (e.g., "16GB GDDR6", "16GB unified memory"), else null.
    - storage: The internal storage capacity text as provided (e.g., "825GB SSD", "1TB"), else null.
    - backward_compatibility_prev_gen: true if the answer claims support for backwards compatibility with games from the previous console generation; 
      false if it claims not supported; null if not addressed.
    - cpu_architecture: The CPU/processor architecture text (e.g., "AMD Zen 2", "Custom AMD SoC"), else null.
    - cpu_core_count: The number of CPU cores as stated (prefer a concise numeric string like "8", or "8 cores/16 threads"), else null.
    - gpu_tflops: The GPU computing performance in TFLOPS as stated (numeric string like "10.28", "12", "12.0"), else null.
    - official_spec_url: A single manufacturer or official URL from the answer that lists specifications or official technical details for the selected console
      (e.g., domains like playstation.com, xbox.com, nintendo.com). If none is present, set to null.
    - additional_urls: All other URLs in the answer that relate to the selected console specs or backward compatibility. If none, return an empty array.

    IMPORTANT:
    - Only extract information explicitly present in the answer. Do not infer or invent.
    - For urls, extract the actual URLs as they appear. If a URL is present in markdown format, extract the URL target.
    - Keep numeric-like fields as strings (e.g., "10.28") to allow flexible formats present in the answer.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return s is not None and str(s).strip() != ""


def _combine_sources(extracted: ConsoleSpecExtraction) -> List[str]:
    urls: List[str] = []
    if _nonempty(extracted.official_spec_url):
        urls.append(extracted.official_spec_url.strip())  # type: ignore
    for u in extracted.additional_urls or []:
        if _nonempty(u):
            urls.append(u.strip())
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


# --------------------------------------------------------------------------- #
# Verification construction                                                   #
# --------------------------------------------------------------------------- #
async def build_verification_tree(evaluator: Evaluator, extracted: ConsoleSpecExtraction) -> None:
    """
    Build the verification tree per rubric.
    """

    # Top-level: A critical parallel node that represents the whole task requirements
    main = evaluator.add_parallel(
        id="Console_Selection_and_Specs",
        desc="Select an eligible major gaming console available in the United States in late 2025 and provide the requested specs with an official reference URL.",
        parent=evaluator.root,
        critical=True,
    )

    # 1) Console_Identified (Existence)
    console_identified = evaluator.add_custom_node(
        result=_nonempty(extracted.selected_console),
        id="Console_Identified",
        desc="The answer explicitly names the selected console model.",
        parent=main,
        critical=True,
    )

    # 2) US_Availability_Late_2025 (as claimed/supported by the answer text)
    us_avail_leaf = evaluator.add_leaf(
        id="US_Availability_Late_2025",
        desc="The selected console is currently available in the United States in late 2025 (as claimed/supported by the answer).",
        parent=main,
        critical=True,
    )
    selected_console_name = extracted.selected_console or "the selected console"
    availability_claim = f"According to the answer, {selected_console_name} is currently available in the United States in late 2025."
    await evaluator.verify(
        claim=availability_claim,
        node=us_avail_leaf,
        additional_instruction="Judge only based on the answer content and task context; do not rely on outside knowledge.",
    )

    # 3) Meets_Technical_Requirements (critical parallel with 4 checks)
    tech_meets = evaluator.add_parallel(
        id="Meets_Technical_Requirements",
        desc="The selected console satisfies all stated technical eligibility requirements.",
        parent=main,
        critical=True,
    )
    sources_all = _combine_sources(extracted)

    # 3.1 RAM >= 16GB
    ram_leaf = evaluator.add_leaf(
        id="RAM_At_Least_16GB",
        desc="The selected console has at least 16 GB of system RAM/memory.",
        parent=tech_meets,
        critical=True,
    )
    claim_ram = f"{selected_console_name} has at least 16 GB of system memory (RAM)."
    await evaluator.verify(
        claim=claim_ram,
        node=ram_leaf,
        sources=sources_all if sources_all else None,
        additional_instruction="Accept phrasing like '16GB unified memory', 'GDDR6', or similar. If it says 16GB or higher, it qualifies.",
    )

    # 3.2 Backward Compatibility
    bc_leaf = evaluator.add_leaf(
        id="Backward_Compatibility_Previous_Gen",
        desc="The selected console supports backwards compatibility with games from the previous console generation.",
        parent=tech_meets,
        critical=True,
    )
    claim_bc = f"{selected_console_name} supports backwards compatibility with games from its previous console generation."
    await evaluator.verify(
        claim=claim_bc,
        node=bc_leaf,
        sources=sources_all if sources_all else None,
        additional_instruction="The page(s) should indicate that games from the immediate prior generation are playable or supported.",
    )

    # 3.3 Storage >= 800GB
    storage_leaf = evaluator.add_leaf(
        id="Storage_At_Least_800GB",
        desc="The selected console has at least 800 GB of internal storage capacity.",
        parent=tech_meets,
        critical=True,
    )
    claim_storage = f"{selected_console_name} provides at least 800 GB of internal storage capacity (e.g., 825 GB or 1 TB qualifies)."
    await evaluator.verify(
        claim=claim_storage,
        node=storage_leaf,
        sources=sources_all if sources_all else None,
        additional_instruction="Look for internal storage size. If it's 800GB or more (including 825GB or 1TB), consider it satisfied.",
    )

    # 3.4 AMD-based CPU
    amd_leaf = evaluator.add_leaf(
        id="AMD_Based_CPU",
        desc="The selected console uses an AMD-based processor (CPU) architecture.",
        parent=tech_meets,
        critical=True,
    )
    claim_amd = f"{selected_console_name} uses an AMD-based CPU architecture (e.g., a custom AMD SoC or AMD Zen family)."
    await evaluator.verify(
        claim=claim_amd,
        node=amd_leaf,
        sources=sources_all if sources_all else None,
        additional_instruction="The source should indicate the CPU is AMD-based (e.g., AMD Zen, custom AMD SoC).",
    )

    # 4) Requested_Specs_and_Citation_Provided (critical parallel)
    req_specs = evaluator.add_parallel(
        id="Requested_Specs_and_Citation_Provided",
        desc="The answer provides the required technical specifications and an official reference URL confirming them.",
        parent=main,
        critical=True,
    )

    # 4.1 CPU core count provided (existence)
    cpu_cores_present = evaluator.add_custom_node(
        result=_nonempty(extracted.cpu_core_count),
        id="CPU_Core_Count_Provided",
        desc="The answer provides the number of CPU cores for the selected console.",
        parent=req_specs,
        critical=True,
    )

    # 4.2 GPU TFLOPS provided (existence)
    gpu_tflops_present = evaluator.add_custom_node(
        result=_nonempty(extracted.gpu_tflops),
        id="GPU_TFLOPS_Provided",
        desc="The answer provides the GPU computing performance measured in TFLOPS for the selected console.",
        parent=req_specs,
        critical=True,
    )

    # 4.3 Official reference URL provided (existence)
    official_url_present = evaluator.add_custom_node(
        result=_nonempty(extracted.official_spec_url),
        id="Official_Reference_URL_Provided",
        desc="The answer includes an official reference URL that confirms the console’s technical specifications.",
        parent=req_specs,
        critical=True,
    )

    # Additional explicit confirmations to ensure the official URL actually confirms the provided specs
    # 4.4 CPU core count matches/confirmed by official URL
    cpu_cores_confirm = evaluator.add_leaf(
        id="CPU_Core_Count_Officially_Confirmed",
        desc="The provided official reference URL confirms the stated CPU core count.",
        parent=req_specs,
        critical=True,
    )
    cpu_core_val = extracted.cpu_core_count or ""
    claim_cpu_cores = f"{selected_console_name} has a CPU with {cpu_core_val}."
    await evaluator.verify(
        claim=claim_cpu_cores,
        node=cpu_cores_confirm,
        sources=extracted.official_spec_url if _nonempty(extracted.official_spec_url) else None,
        additional_instruction="Verify against the official manufacturer page(s). Allow minor formatting like '8 cores/16 threads'; focus on core count.",
    )

    # 4.5 GPU TFLOPS matches/confirmed by official URL
    gpu_tflops_confirm = evaluator.add_leaf(
        id="GPU_TFLOPS_Officially_Confirmed",
        desc="The provided official reference URL confirms the stated GPU TFLOPS.",
        parent=req_specs,
        critical=True,
    )
    gpu_tf_val = extracted.gpu_tflops or ""
    claim_gpu_tflops = f"{selected_console_name} has GPU computing performance of {gpu_tf_val} TFLOPS (approximately)."
    await evaluator.verify(
        claim=claim_gpu_tflops,
        node=gpu_tflops_confirm,
        sources=extracted.official_spec_url if _nonempty(extracted.official_spec_url) else None,
        additional_instruction="Check the official manufacturer page(s). Allow minor rounding differences (e.g., 10.28 ~ 10.3).",
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
    Entry point to evaluate an answer for the 2025 console selection/specs task.
    """
    # Initialize evaluator with a parallel root
    evaluator = Evaluator()
    evaluator.initialize(
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

    # 1) Extract structured data from the answer
    extracted = await evaluator.extract(
        prompt=prompt_extract_console_spec(),
        template_class=ConsoleSpecExtraction,
        extraction_name="console_spec_extraction",
    )

    # 2) Build verification tree according to rubric and run verifications
    await build_verification_tree(evaluator, extracted)

    # 3) Return evaluation summary
    return evaluator.get_summary()