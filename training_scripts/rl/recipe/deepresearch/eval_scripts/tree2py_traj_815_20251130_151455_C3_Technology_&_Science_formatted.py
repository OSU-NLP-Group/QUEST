import asyncio
import logging
import re
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "console_high_performance_chicago"
TASK_DESCRIPTION = (
    "Identify a gaming console that meets the following technical specifications for high-performance gaming: "
    "GPU performance of at least 10 TFLOPs, system memory (RAM) of at least 16GB, and a CPU with at least 8 cores. "
    "For the console you identify, provide:\n\n"
    "1. The console model name\n"
    "2. Verification of its GPU performance (in TFLOPs) with supporting reference URL\n"
    "3. Verification of its system memory capacity (in GB) with supporting reference URL\n"
    "4. Verification of its CPU core count with supporting reference URL\n"
    "5. The name and complete street address of a physical retail store in Chicago, Illinois where this console is currently available for purchase, with supporting reference URL\n"
    "6. The current retail price at major retailers with supporting reference URL"
)

MIN_GPU_TFLOPS = 10.0
MIN_RAM_GB = 16.0
MIN_CPU_CORES = 8


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class PriceEntry(BaseModel):
    retailer: Optional[str] = None
    price_text: Optional[str] = None
    price_url: Optional[str] = None


class AnswerExtraction(BaseModel):
    # Console + Specs
    model_name: Optional[str] = None

    gpu_tflops: Optional[str] = None
    gpu_sources: List[str] = Field(default_factory=list)

    ram_gb: Optional[str] = None
    ram_sources: List[str] = Field(default_factory=list)

    cpu_cores: Optional[str] = None
    cpu_sources: List[str] = Field(default_factory=list)

    # Chicago store
    store_name: Optional[str] = None
    street_address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    store_url: Optional[str] = None
    availability_urls: List[str] = Field(default_factory=list)

    # Major retailer price info
    prices: List[PriceEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_all() -> str:
    return """
Extract the following information exactly as presented in the answer text. Do not infer or invent anything. If an item is missing, return null (for a single value) or an empty array (for a list).

CONSOLE + SPECS (single console only):
- model_name: The exact console model name the answer commits to (e.g., "PlayStation 5", "Xbox Series X").
- gpu_tflops: The GPU performance value as stated (e.g., "10.28 TFLOPs").
- gpu_sources: All URLs cited in the answer that support the GPU performance.
- ram_gb: The system memory value as stated (e.g., "16 GB GDDR6").
- ram_sources: All URLs cited in the answer that support the RAM specification.
- cpu_cores: The CPU core count as stated (e.g., "8 cores").
- cpu_sources: All URLs cited in the answer that support the CPU core count.

CHICAGO STORE (physical retail store in Chicago, IL):
- store_name: Store name as stated (e.g., "Best Buy Chicago North Avenue").
- street_address: Full street address line (e.g., "1000 W North Ave").
- city: The city (should be "Chicago" if provided).
- state: The state (e.g., "IL" or "Illinois").
- zip_code: The ZIP code if present (e.g., "60642").
- store_url: A primary store or store-location/product page URL cited that shows this store and address.
- availability_urls: Any additional URLs cited that show availability status for the console at this Chicago store.

CURRENT RETAIL PRICE AT MAJOR RETAILERS:
- prices: An array where each element includes:
  - retailer: The retailer name as stated (e.g., "Best Buy", "Target", "Walmart", "GameStop", "Amazon", "Microsoft Store", "PlayStation Direct").
  - price_text: The price as stated (e.g., "$499.99", "USD 499.99").
  - price_url: The URL cited that shows this price for the console.

Return a JSON object with these fields:
{
  "model_name": ...,
  "gpu_tflops": ...,
  "gpu_sources": [...],
  "ram_gb": ...,
  "ram_sources": [...],
  "cpu_cores": ...,
  "cpu_sources": [...],
  "store_name": ...,
  "street_address": ...,
  "city": ...,
  "state": ...,
  "zip_code": ...,
  "store_url": ...,
  "availability_urls": [...],
  "prices": [
    {"retailer": ..., "price_text": ..., "price_url": ...},
    ...
  ]
}

Special URL rules:
- Extract only URLs explicitly present in the answer (plain or markdown). Do not invent or crawl.
- If a URL is missing protocol, prepend http://.
"""


# --------------------------------------------------------------------------- #
# Helpers: numeric parsing                                                    #
# --------------------------------------------------------------------------- #
_num_pattern = re.compile(r"\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> List[float]:
    return [float(x) for x in _num_pattern.findall(text or "")]


def parse_tflops(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    s = text.lower()
    nums = _extract_numbers(s)
    if not nums:
        return None
    # Prefer interpreting numbers alongside units where possible
    if "tflop" in s or "teraflop" in s:
        # Take the largest number that appears with this unit context (fallback to max)
        return max(nums)
    if "gflop" in s or "gigaflop" in s:
        return max(nums) / 1000.0
    # If the string explicitly says TF (rare) and implies TFLOPs
    if "tf" in s and "flop" in s:
        return max(nums)
    # Fallback: if the string looks like a TFLOPs value (e.g., "10.28"), treat the max numeric as TFLOPs
    return max(nums)


def parse_gb(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    s = text.lower()
    nums = _extract_numbers(s)
    if not nums:
        return None
    # Unit-aware conversion
    if "tb" in s or "terabyte" in s:
        return max(nums) * 1024.0
    if "mb" in s or "megabyte" in s:
        return max(nums) / 1024.0
    # Default GB
    return max(nums)


def parse_cores(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    s = text.lower()
    nums = _extract_numbers(s)
    if not nums:
        return None
    # Heuristic: choose an integer <= 256 that is most likely the core count (max reasonable)
    candidates = [int(round(n)) for n in nums if n <= 256]
    return max(candidates) if candidates else None


def is_chicago_location(city: Optional[str], state: Optional[str], street_address: Optional[str], zip_code: Optional[str]) -> bool:
    def _norm(x: Optional[str]) -> str:
        return (x or "").strip().lower()

    c = _norm(city)
    st = _norm(state)
    addr = _norm(street_address)
    z = _norm(zip_code)

    chicago_by_fields = (c == "chicago") and (st in {"il", "illinois"})
    chicago_in_addr = "chicago" in addr and (" il" in addr or " illinois" in addr)
    chicago_zip = bool(z) and z.startswith("606")  # Common Chicago ZIP prefix

    return chicago_by_fields or chicago_in_addr or chicago_zip


def pick_first_price_with_url(prices: List[PriceEntry]) -> Optional[PriceEntry]:
    for p in prices:
        if p and p.price_text and p.price_url:
            return p
    return None


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_console_and_specs_nodes(evaluator: Evaluator, parent_node, data: AnswerExtraction) -> None:
    # Parent container (critical, parallel): "console_and_specs"
    console_node = evaluator.add_parallel(
        id="console_and_specs",
        desc="Provide console model and verify it meets all technical minimums with supporting URLs.",
        parent=parent_node,
        critical=True
    )

    # 1) Console model name provided (critical existence check)
    model_exists = bool((data.model_name or "").strip())
    evaluator.add_custom_node(
        result=model_exists,
        id="console_model_name",
        desc="Provide the console model name.",
        parent=console_node,
        critical=True
    )

    # 2) GPU ≥ 10 TFLOPs with citation
    gpu_group = evaluator.add_parallel(
        id="gpu_meets_threshold_with_citation",
        desc="Provide the console's GPU performance in TFLOPs, confirm it is ≥ 10 TFLOPs, and include a supporting reference URL.",
        parent=console_node,
        critical=True
    )
    gpu_value_provided = bool((data.gpu_tflops or "").strip()) and len(data.gpu_sources) > 0
    evaluator.add_custom_node(
        result=gpu_value_provided,
        id="gpu_value_provided",
        desc="GPU TFLOPs value is provided and at least one supporting URL is cited.",
        parent=gpu_group,
        critical=True
    )
    parsed_tflops = parse_tflops(data.gpu_tflops)
    evaluator.add_custom_node(
        result=(parsed_tflops is not None and parsed_tflops >= MIN_GPU_TFLOPS),
        id="gpu_meets_threshold",
        desc=f"GPU TFLOPs value meets or exceeds {MIN_GPU_TFLOPS}.",
        parent=gpu_group,
        critical=True
    )
    gpu_support_leaf = evaluator.add_leaf(
        id="gpu_value_supported_by_sources",
        desc="Cited source(s) explicitly support the stated GPU TFLOPs for this console.",
        parent=gpu_group,
        critical=True
    )
    gpu_claim = (
        f"The console '{data.model_name or 'the console'}' has a GPU performance of {data.gpu_tflops or 'UNKNOWN'} "
        f"(expressed in TFLOPs or equivalent)."
    )
    await evaluator.verify(
        claim=gpu_claim,
        node=gpu_support_leaf,
        sources=data.gpu_sources,
        additional_instruction=(
            "Verify that the cited page(s) explicitly mention the GPU performance in TFLOPs (or clearly equivalent). "
            "Minor formatting differences are fine. If TFLOPs is not present or clearly inferable from the page, "
            "mark as not supported."
        )
    )

    # 3) RAM ≥ 16 GB with citation
    ram_group = evaluator.add_parallel(
        id="ram_meets_threshold_with_citation",
        desc="Provide the console's system memory (RAM) in GB, confirm it is ≥ 16GB, and include a supporting reference URL.",
        parent=console_node,
        critical=True
    )
    ram_value_provided = bool((data.ram_gb or "").strip()) and len(data.ram_sources) > 0
    evaluator.add_custom_node(
        result=ram_value_provided,
        id="ram_value_provided",
        desc="System memory (RAM) value is provided and at least one supporting URL is cited.",
        parent=ram_group,
        critical=True
    )
    parsed_ram = parse_gb(data.ram_gb)
    evaluator.add_custom_node(
        result=(parsed_ram is not None and parsed_ram >= MIN_RAM_GB),
        id="ram_meets_threshold",
        desc=f"System memory (RAM) meets or exceeds {MIN_RAM_GB} GB.",
        parent=ram_group,
        critical=True
    )
    ram_support_leaf = evaluator.add_leaf(
        id="ram_value_supported_by_sources",
        desc="Cited source(s) explicitly support the stated RAM capacity for this console.",
        parent=ram_group,
        critical=True
    )
    ram_claim = (
        f"The console '{data.model_name or 'the console'}' has {data.ram_gb or 'UNKNOWN'} of system memory (RAM)."
    )
    await evaluator.verify(
        claim=ram_claim,
        node=ram_support_leaf,
        sources=data.ram_sources,
        additional_instruction=(
            "Verify that the page(s) explicitly state the system memory capacity (e.g., 16 GB). "
            "Allow minor format differences, but the numeric capacity must be present on the page."
        )
    )

    # 4) CPU ≥ 8 cores with citation
    cpu_group = evaluator.add_parallel(
        id="cpu_meets_threshold_with_citation",
        desc="Provide the console's CPU core count, confirm it is ≥ 8 cores, and include a supporting reference URL.",
        parent=console_node,
        critical=True
    )
    cpu_value_provided = bool((data.cpu_cores or "").strip()) and len(data.cpu_sources) > 0
    evaluator.add_custom_node(
        result=cpu_value_provided,
        id="cpu_value_provided",
        desc="CPU core count is provided and at least one supporting URL is cited.",
        parent=cpu_group,
        critical=True
    )
    parsed_cores = parse_cores(data.cpu_cores)
    evaluator.add_custom_node(
        result=(parsed_cores is not None and parsed_cores >= MIN_CPU_CORES),
        id="cpu_meets_threshold",
        desc=f"CPU core count meets or exceeds {MIN_CPU_CORES} cores.",
        parent=cpu_group,
        critical=True
    )
    cpu_support_leaf = evaluator.add_leaf(
        id="cpu_value_supported_by_sources",
        desc="Cited source(s) explicitly support the stated CPU core count for this console.",
        parent=cpu_group,
        critical=True
    )
    cpu_claim = (
        f"The console '{data.model_name or 'the console'}' has a CPU with {data.cpu_cores or 'UNKNOWN'} cores."
    )
    await evaluator.verify(
        claim=cpu_claim,
        node=cpu_support_leaf,
        sources=data.cpu_sources,
        additional_instruction=(
            "Verify that the page(s) explicitly mention the CPU core count. "
            "Minor wording differences (e.g., '8-core CPU') are acceptable."
        )
    )


async def build_chicago_store_and_price_nodes(evaluator: Evaluator, parent_node, data: AnswerExtraction) -> None:
    # Parent container (critical, parallel): "chicago_store_and_price"
    chi_node = evaluator.add_parallel(
        id="chicago_store_and_price",
        desc="Provide a Chicago, Illinois physical retail store offering the console and provide current retail price info, with supporting URLs.",
        parent=parent_node,
        critical=True
    )

    # 1) Store name + complete address in Chicago with citation
    store_group = evaluator.add_parallel(
        id="store_name_and_complete_address_in_chicago_with_citation",
        desc="Provide the physical retail store name and complete street address located in Chicago, Illinois, and include a supporting reference URL.",
        parent=chi_node,
        critical=True
    )
    store_fields_provided = bool((data.store_name or "").strip()) and bool((data.street_address or "").strip()) and bool((data.store_url or "").strip())
    evaluator.add_custom_node(
        result=store_fields_provided,
        id="store_name_and_complete_address_in_chicago_exists",
        desc="Store name, complete street address, and a supporting URL are provided.",
        parent=store_group,
        critical=True
    )
    evaluator.add_custom_node(
        result=is_chicago_location(data.city, data.state, data.street_address, data.zip_code),
        id="store_is_in_chicago",
        desc="The provided address is in Chicago, Illinois (city/state or address indicates Chicago, IL).",
        parent=store_group,
        critical=True
    )
    store_support_leaf = evaluator.add_leaf(
        id="store_name_and_address_supported_by_source",
        desc="Cited store URL shows the store name and complete street address in Chicago, Illinois.",
        parent=store_group,
        critical=True
    )
    store_claim = (
        f"The page shows the store '{data.store_name or 'UNKNOWN STORE'}' located at "
        f"'{data.street_address or 'UNKNOWN ADDRESS'}' in Chicago, Illinois."
    )
    await evaluator.verify(
        claim=store_claim,
        node=store_support_leaf,
        sources=data.store_url,
        additional_instruction=(
            "Confirm that the page contains both the store name and the full street address, and that the location is Chicago, Illinois. "
            "Minor formatting differences are acceptable."
        )
    )

    # 2) Evidence that the console is currently available for purchase at the identified Chicago store
    availability_group = evaluator.add_parallel(
        id="evidence_currently_available_for_purchase_with_citation",
        desc="Provide evidence via a supporting reference URL that the console is currently available for purchase at the identified Chicago store (e.g., active listing/availability status).",
        parent=chi_node,
        critical=True
    )
    availability_sources: List[str] = data.availability_urls[:] if data.availability_urls else []
    if not availability_sources and data.store_url:
        availability_sources = [data.store_url]
    evaluator.add_custom_node(
        result=len(availability_sources) > 0,
        id="availability_source_provided",
        desc="At least one URL is provided to show current availability status for the console at this Chicago store.",
        parent=availability_group,
        critical=True
    )
    availability_leaf = evaluator.add_leaf(
        id="availability_supported",
        desc="Cited source(s) show that the console is currently available for purchase at the identified Chicago store.",
        parent=availability_group,
        critical=True
    )
    availability_claim = (
        f"The console '{data.model_name or 'the console'}' is currently available for purchase at the specified Chicago store."
    )
    await evaluator.verify(
        claim=availability_claim,
        node=availability_leaf,
        sources=availability_sources,
        additional_instruction=(
            "Treat 'currently available' as the page indicating in-stock/available status (e.g., 'In stock', "
            "'Available now', 'Pick up today', active 'Add to cart' for the Chicago location). "
            "If the page shows 'Out of stock', 'Sold out', or similar, treat as not available."
        )
    )

    # 3) Current retail price at major retailers with citation
    price_group = evaluator.add_parallel(
        id="current_retail_price_at_major_retailers_with_citation",
        desc="Provide current retail price information at major retailer(s) and include supporting reference URL(s).",
        parent=chi_node,
        critical=True
    )
    selected_price = pick_first_price_with_url(data.prices)
    price_info_provided = bool(selected_price and (selected_price.price_text or "").strip() and (selected_price.price_url or "").strip())
    evaluator.add_custom_node(
        result=price_info_provided,
        id="retail_price_info_provided",
        desc="A price text and a retailer price URL are provided.",
        parent=price_group,
        critical=True
    )
    price_leaf = evaluator.add_leaf(
        id="retail_price_supported_by_source",
        desc="Cited price URL shows the stated current retail price for the console.",
        parent=price_group,
        critical=True
    )
    price_claim = (
        f"The current retail price for the console '{data.model_name or 'the console'}' at "
        f"'{(selected_price.retailer if selected_price else 'UNKNOWN RETAILER')}' is "
        f"'{(selected_price.price_text if selected_price else 'UNKNOWN PRICE')}'."
    )
    await evaluator.verify(
        claim=price_claim,
        node=price_leaf,
        sources=(selected_price.price_url if selected_price else None),
        additional_instruction=(
            "Confirm that the page explicitly shows the quoted price for the console model. "
            "Minor currency/formatting differences are acceptable, but the numeric price must be present."
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
    Evaluate an answer for the 'console_high_performance_chicago' task using obj_task_eval evaluator.
    """
    # Initialize evaluator with a sequential root (non-critical by default)
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

    # Add threshold info as custom info for transparency
    evaluator.add_custom_info(
        {
            "min_gpu_tflops": MIN_GPU_TFLOPS,
            "min_ram_gb": MIN_RAM_GB,
            "min_cpu_cores": MIN_CPU_CORES
        },
        info_type="constraints",
        info_name="technical_thresholds"
    )

    # Extract all required fields from the answer
    extraction = await evaluator.extract(
        prompt=prompt_extract_all(),
        template_class=AnswerExtraction,
        extraction_name="extracted_console_store_price"
    )

    # Build a critical sequential top-level node to strictly enforce ordering:
    # 1) Console + specs must pass before moving to 2) Chicago store + price.
    task_main = evaluator.add_sequential(
        id="task_root",
        desc="Identify a compliant console (GPU ≥10 TFLOPs, RAM ≥16GB, CPU ≥8 cores) and provide Chicago store + current price info with citations.",
        parent=root,
        critical=True
    )

    # Subtree 1: Console + specs
    await build_console_and_specs_nodes(evaluator, task_main, extraction)

    # Subtree 2: Chicago store + price (will be skipped if Subtree 1 fails due to sequential gating)
    await build_chicago_store_and_price_nodes(evaluator, task_main, extraction)

    # Return structured evaluation summary
    return evaluator.get_summary()