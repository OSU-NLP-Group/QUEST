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
TASK_ID = "laptop_2024_specs"
TASK_DESCRIPTION = (
    "Identify a laptop model that was released or announced in 2024 and is available for purchase in the United States, "
    "meeting ALL of the following technical specifications:\n\n"
    "1. Processor: Must have a 2024-generation processor (Apple M4, Intel Core Ultra 200 series, or AMD Ryzen AI 9 HX series) with at least 10 CPU cores\n"
    "2. Memory: At least 16GB of RAM\n"
    "3. Storage: At least 512GB of internal storage\n"
    "4. Display: Screen size must be between 13 and 16 inches (inclusive)\n"
    "5. Battery Life: Must provide at least 15 hours of battery life based on standard testing or manufacturer specifications\n"
    "6. Release Date: Must have been officially released or announced in 2024\n\n"
    "Provide the specific manufacturer name and complete model name, along with valid URL references that verify each of the technical specifications listed above."
)

ROOT_DESC = "Identify and validate a laptop model released or announced in 2024, available for purchase in the United States, meeting all specified technical requirements, with URL evidence for each requirement"


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class LaptopIdentification(BaseModel):
    manufacturer: Optional[str] = None
    model_name: Optional[str] = None


class LaptopSpecs(BaseModel):
    processor: Optional[str] = None
    cpu_core_count: Optional[str] = None
    ram: Optional[str] = None
    storage: Optional[str] = None
    display_size: Optional[str] = None
    battery_life: Optional[str] = None
    release_info: Optional[str] = None


class LaptopURLs(BaseModel):
    general_urls: List[str] = Field(default_factory=list)
    availability_urls: List[str] = Field(default_factory=list)
    processor_urls: List[str] = Field(default_factory=list)
    ram_urls: List[str] = Field(default_factory=list)
    storage_urls: List[str] = Field(default_factory=list)
    display_urls: List[str] = Field(default_factory=list)
    battery_urls: List[str] = Field(default_factory=list)
    release_urls: List[str] = Field(default_factory=list)


class LaptopExtraction(BaseModel):
    identification: Optional[LaptopIdentification] = None
    specs: Optional[LaptopSpecs] = None
    urls: Optional[LaptopURLs] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_laptop() -> str:
    return """
    From the provided answer text, extract the following structured information about a single laptop model that the answer proposes:

    identification:
      - manufacturer: the brand/manufacturer name (e.g., Apple, Lenovo, Dell)
      - model_name: the full and specific model name as given in the answer

    specs:
      - processor: the processor/model as named (e.g., Apple M4, Intel Core Ultra 7 255, AMD Ryzen AI 9 HX 370)
      - cpu_core_count: the stated or implied total CPU core count if mentioned (as text)
      - ram: the RAM capacity as text (e.g., "16GB", "32 GB", "up to 64GB")
      - storage: the internal storage as text (e.g., "512GB SSD", "1TB")
      - display_size: the screen size as text (e.g., "14-inch", "15.6\"")
      - battery_life: the battery life claim as text if given (e.g., "up to 18 hours")
      - release_info: the release/announcement information as text if given (e.g., "announced in 2024", "released October 2024")

    urls:
      - general_urls: any general product pages or reviews cited that pertain to the model overall
      - availability_urls: URLs that specifically show the laptop is available for purchase in the United States (official store or authorized US retailers)
      - processor_urls: URLs that support the processor model/generation and core count for this laptop
      - ram_urls: URLs that support the RAM specification
      - storage_urls: URLs that support the internal storage specification
      - display_urls: URLs that support the display size specification
      - battery_urls: URLs that support the battery life claim
      - release_urls: URLs that support that the laptop was announced/released in 2024

    Rules:
    - Extract ONLY what is explicitly given in the answer.
    - For any unknown field, return null (for strings) or [] (for URL lists).
    - For URL fields, return all URLs mentioned for that field. Use full URLs; if malformed or clearly invalid, omit them.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _safe_str(x: Optional[str]) -> str:
    return (x or "").strip()


def _unique_urls(urls: List[str]) -> List[str]:
    seen = set()
    result = []
    for u in urls:
        u = (u or "").strip()
        if not u:
            continue
        if u not in seen:
            seen.add(u)
            result.append(u)
    return result


def _urls_for(primary: Optional[List[str]], fallback: Optional[List[str]]) -> List[str]:
    primary = primary or []
    fallback = fallback or []
    return _unique_urls(primary + fallback)


def _model_label(extracted: LaptopExtraction) -> str:
    m = _safe_str(extracted.identification.manufacturer if extracted.identification else None)
    mod = _safe_str(extracted.identification.model_name if extracted.identification else None)
    if m and mod:
        return f"{m} {mod}"
    if mod:
        return mod
    if m:
        return m
    return "the laptop model"


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def verify_laptop_identification(evaluator: Evaluator, parent_node, extracted: LaptopExtraction) -> None:
    manu = _safe_str(extracted.identification.manufacturer if extracted.identification else None)
    model = _safe_str(extracted.identification.model_name if extracted.identification else None)

    evaluator.add_custom_node(
        result=bool(manu) and bool(model),
        id="laptop_identification",
        desc="Provide the specific manufacturer name and complete laptop model name",
        parent=parent_node,
        critical=True
    )


async def verify_us_availability(evaluator: Evaluator, parent_node, extracted: LaptopExtraction) -> None:
    node = evaluator.add_parallel(
        id="us_availability_verification",
        desc="Verify the laptop is available for purchase in the United States",
        parent=parent_node,
        critical=True
    )

    urls = _urls_for(
        extracted.urls.availability_urls if extracted.urls else [],
        extracted.urls.general_urls if extracted.urls else []
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="us_availability_source",
        desc="Provide a valid URL reference showing US purchase availability",
        parent=node,
        critical=True
    )

    avail_leaf = evaluator.add_leaf(
        id="us_availability",
        desc="Laptop must be available for purchase in the United States",
        parent=node,
        critical=True
    )
    claim = f"{_model_label(extracted)} is available for purchase in the United States (US)."
    await evaluator.verify(
        claim=claim,
        node=avail_leaf,
        sources=urls,
        additional_instruction=(
            "Verify regional availability for the US. Accept official US store pages or authorized US retailers that sell this model, "
            "or product pages that explicitly indicate United States availability, US pricing (USD $), or delivery/shipping to the United States."
        )
    )


async def verify_processor(evaluator: Evaluator, parent_node, extracted: LaptopExtraction) -> None:
    node = evaluator.add_parallel(
        id="processor_verification",
        desc="Verify processor meets the 2024-generation and core-count requirements",
        parent=parent_node,
        critical=True
    )

    urls = _urls_for(
        extracted.urls.processor_urls if extracted.urls else [],
        extracted.urls.general_urls if extracted.urls else []
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="processor_source",
        desc="Provide a valid URL reference that supports the processor generation/model and CPU core count",
        parent=node,
        critical=True
    )

    # Generation check
    gen_leaf = evaluator.add_leaf(
        id="processor_generation",
        desc="Processor must be from a 2024-generation architecture: Apple M4, Intel Core Ultra 200 series, or AMD Ryzen AI 9 HX series",
        parent=node,
        critical=True
    )
    processor_name = _safe_str(extracted.specs.processor if extracted.specs else None)
    gen_claim = (
        f"The {_model_label(extracted)} uses a 2024-generation processor: Apple M4, Intel Core Ultra 200 series "
        f"(e.g., Core Ultra 7/9 2xx), or AMD Ryzen AI 9 HX series (e.g., HX 3xx)."
    )
    await evaluator.verify(
        claim=gen_claim,
        node=gen_leaf,
        sources=urls,
        additional_instruction=(
            "Confirm the processor belongs to one of: Apple M4 family, Intel Core Ultra 200 (2xx) series, or AMD Ryzen AI 9 HX series. "
            f"If the answer mentions a specific CPU (e.g., '{processor_name}'), ensure it maps to one of these families."
        )
    )

    # Core count check
    cores_leaf = evaluator.add_leaf(
        id="core_count",
        desc="Processor must have at least 10 CPU cores",
        parent=node,
        critical=True
    )
    cores_claim = (
        f"The processor used by {_model_label(extracted)} has at least 10 CPU cores in total "
        "(count all performance/efficiency cores)."
    )
    await evaluator.verify(
        claim=cores_claim,
        node=cores_leaf,
        sources=urls,
        additional_instruction="If the page lists separate core types (P/E cores), sum them to get the total core count."
    )


async def verify_memory(evaluator: Evaluator, parent_node, extracted: LaptopExtraction) -> None:
    node = evaluator.add_parallel(
        id="memory_verification",
        desc="Verify RAM meets the minimum requirement",
        parent=parent_node,
        critical=True
    )

    urls = _urls_for(
        extracted.urls.ram_urls if extracted.urls else [],
        extracted.urls.general_urls if extracted.urls else []
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="ram_source",
        desc="Provide a valid URL reference for the RAM specification",
        parent=node,
        critical=True
    )

    ram_leaf = evaluator.add_leaf(
        id="ram_capacity",
        desc="Laptop must have at least 16GB of RAM",
        parent=node,
        critical=True
    )
    ram_claim = f"{_model_label(extracted)} has at least 16GB of RAM."
    await evaluator.verify(
        claim=ram_claim,
        node=ram_leaf,
        sources=urls,
        additional_instruction="If multiple SKUs are shown, it's acceptable if at least one purchasing configuration meets or exceeds 16GB RAM."
    )


async def verify_storage(evaluator: Evaluator, parent_node, extracted: LaptopExtraction) -> None:
    node = evaluator.add_parallel(
        id="storage_verification",
        desc="Verify internal storage meets the minimum requirement",
        parent=parent_node,
        critical=True
    )

    urls = _urls_for(
        extracted.urls.storage_urls if extracted.urls else [],
        extracted.urls.general_urls if extracted.urls else []
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="storage_source",
        desc="Provide a valid URL reference for the storage specification",
        parent=node,
        critical=True
    )

    storage_leaf = evaluator.add_leaf(
        id="storage_capacity",
        desc="Laptop must have at least 512GB of internal storage",
        parent=node,
        critical=True
    )
    storage_claim = f"{_model_label(extracted)} has at least 512GB of internal storage."
    await evaluator.verify(
        claim=storage_claim,
        node=storage_leaf,
        sources=urls,
        additional_instruction="Storage can be SSD; if multiple SKUs exist, at least one purchasable configuration should have ≥ 512GB."
    )


async def verify_display(evaluator: Evaluator, parent_node, extracted: LaptopExtraction) -> None:
    node = evaluator.add_parallel(
        id="display_verification",
        desc="Verify display size is within the required range",
        parent=parent_node,
        critical=True
    )

    urls = _urls_for(
        extracted.urls.display_urls if extracted.urls else [],
        extracted.urls.general_urls if extracted.urls else []
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="display_source",
        desc="Provide a valid URL reference for the display size specification",
        parent=node,
        critical=True
    )

    display_leaf = evaluator.add_leaf(
        id="display_size",
        desc="Display size must be between 13 and 16 inches (inclusive)",
        parent=node,
        critical=True
    )
    display_claim = f"The screen size of {_model_label(extracted)} is between 13 and 16 inches inclusive."
    await evaluator.verify(
        claim=display_claim,
        node=display_leaf,
        sources=urls,
        additional_instruction="Allow decimals like 13.3, 14.5, 15.6. The value must fall within [13, 16] inches."
    )


async def verify_battery(evaluator: Evaluator, parent_node, extracted: LaptopExtraction) -> None:
    node = evaluator.add_parallel(
        id="battery_verification",
        desc="Verify battery life meets the minimum requirement",
        parent=parent_node,
        critical=True
    )

    urls = _urls_for(
        extracted.urls.battery_urls if extracted.urls else [],
        extracted.urls.general_urls if extracted.urls else []
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="battery_source",
        desc="Provide a valid URL reference for the battery life (test result or manufacturer specification)",
        parent=node,
        critical=True
    )

    battery_leaf = evaluator.add_leaf(
        id="battery_life",
        desc="Laptop must provide at least 15 hours of battery life based on standard testing or manufacturer specifications",
        parent=node,
        critical=True
    )
    battery_claim = f"{_model_label(extracted)} provides at least 15 hours of battery life (per standard testing or manufacturer specifications)."
    await evaluator.verify(
        claim=battery_claim,
        node=battery_leaf,
        sources=urls,
        additional_instruction="Accept 'up to' values if they clearly state ≥ 15 hours under a standard or manufacturer-documented test methodology."
    )


async def verify_release(evaluator: Evaluator, parent_node, extracted: LaptopExtraction) -> None:
    node = evaluator.add_parallel(
        id="release_verification",
        desc="Verify the laptop was officially released or announced in 2024",
        parent=parent_node,
        critical=True
    )

    urls = _urls_for(
        extracted.urls.release_urls if extracted.urls else [],
        extracted.urls.general_urls if extracted.urls else []
    )

    evaluator.add_custom_node(
        result=bool(urls),
        id="release_source",
        desc="Provide a valid URL reference confirming the 2024 release or announcement",
        parent=node,
        critical=True
    )

    release_leaf = evaluator.add_leaf(
        id="release_year",
        desc="Laptop must have been officially released or announced in 2024",
        parent=node,
        critical=True
    )
    release_claim = f"{_model_label(extracted)} was officially released or announced in 2024."
    await evaluator.verify(
        claim=release_claim,
        node=release_leaf,
        sources=urls,
        additional_instruction="Look for explicit date mentions in 2024 for release or announcement from the manufacturer or reputable publications."
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
    Evaluate an answer for the 2024 laptop specification task.
    """
    # 1) Initialize evaluator
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

    # 2) Extract structured information from the answer
    extracted: LaptopExtraction = await evaluator.extract(
        prompt=prompt_extract_laptop(),
        template_class=LaptopExtraction,
        extraction_name="laptop_extraction"
    )

    # Ensure sub-objects exist to avoid None checks later
    if extracted.identification is None:
        extracted.identification = LaptopIdentification()
    if extracted.specs is None:
        extracted.specs = LaptopSpecs()
    if extracted.urls is None:
        extracted.urls = LaptopURLs()

    # Add a constraints summary as custom info for transparency
    evaluator.add_custom_info(
        info={
            "requirements": {
                "processor_2024_gen": ["Apple M4", "Intel Core Ultra 200 series", "AMD Ryzen AI 9 HX series"],
                "min_cpu_cores": 10,
                "min_ram_gb": 16,
                "min_storage_gb": 512,
                "display_range_inches": [13, 16],
                "min_battery_hours": 15,
                "release_year": 2024,
                "us_availability": True
            }
        },
        info_type="constraints_summary"
    )

    # 3) Build verification tree following the rubric (put everything under a critical main node)
    task_root = evaluator.add_parallel(
        id="task_main",
        desc=ROOT_DESC,
        parent=root,
        critical=True
    )

    # 3.1 Identification (critical leaf)
    await verify_laptop_identification(evaluator, task_root, extracted)

    # 3.2 US availability subtree (critical)
    await verify_us_availability(evaluator, task_root, extracted)

    # 3.3 Processor subtree (critical)
    await verify_processor(evaluator, task_root, extracted)

    # 3.4 Memory subtree (critical)
    await verify_memory(evaluator, task_root, extracted)

    # 3.5 Storage subtree (critical)
    await verify_storage(evaluator, task_root, extracted)

    # 3.6 Display subtree (critical)
    await verify_display(evaluator, task_root, extracted)

    # 3.7 Battery subtree (critical)
    await verify_battery(evaluator, task_root, extracted)

    # 3.8 Release/announcement year subtree (critical)
    await verify_release(evaluator, task_root, extracted)

    # 4) Return structured summary
    return evaluator.get_summary()