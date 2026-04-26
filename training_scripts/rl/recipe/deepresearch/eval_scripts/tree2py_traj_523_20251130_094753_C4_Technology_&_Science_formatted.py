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
TASK_ID = "desktop_n3e_nov2024"
TASK_DESCRIPTION = (
    "Identify the specific desktop computer product that meets all of the following criteria: "
    "(1) Released (became available to customers) in November 2024; "
    "(2) Features a processor manufactured using TSMC's N3E (3nm Enhanced) process technology; "
    "(3) Has a starting price of $599 USD; "
    "(4) Base configuration includes 16GB of unified memory; "
    "(5) Base configuration includes 256GB of storage; "
    "(6) Equipped with a 10-core CPU consisting of 4 performance cores and 6 efficiency cores; "
    "(7) Base configuration includes a 10-core GPU; "
    "(8) Became officially available for shipping on November 8, 2024. "
    "Provide the product name and include reference URLs that verify each of these specifications."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ProductSpecSources(BaseModel):
    product_name: Optional[str] = None

    # URL groups per required specification
    desktop_proof_urls: List[str] = Field(default_factory=list)
    release_month_year_urls: List[str] = Field(default_factory=list)
    processor_n3e_urls: List[str] = Field(default_factory=list)
    starting_price_599_urls: List[str] = Field(default_factory=list)
    base_memory_16_unified_urls: List[str] = Field(default_factory=list)
    base_storage_256_urls: List[str] = Field(default_factory=list)
    cpu_10core_4p6e_urls: List[str] = Field(default_factory=list)
    base_gpu_10core_urls: List[str] = Field(default_factory=list)
    preorder_oct29_2024_urls: List[str] = Field(default_factory=list)
    shipping_nov8_2024_urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_product_and_sources() -> str:
    return """
    From the answer, extract:
    - product_name: the specific desktop computer model identified.
    - desktop_proof_urls: URLs that support the claim that this product is a desktop computer (not a laptop/tablet/phone).
    - release_month_year_urls: URLs that support the claim the product became available to customers in November 2024.
    - processor_n3e_urls: URLs that support the claim the processor is manufactured using TSMC N3E (3nm Enhanced) process technology.
    - starting_price_599_urls: URLs that support the claim the product's starting price is $599 USD (phrases like "starting at $599" or "from $599" count).
    - base_memory_16_unified_urls: URLs that support the claim the base configuration includes 16GB of unified memory.
    - base_storage_256_urls: URLs that support the claim the base configuration includes 256GB of storage.
    - cpu_10core_4p6e_urls: URLs that support the claim the CPU has 10 cores comprising 4 performance cores + 6 efficiency cores (4P+6E).
    - base_gpu_10core_urls: URLs that support the claim the base configuration includes a 10-core GPU.
    - preorder_oct29_2024_urls: URLs that support the claim preorders began on October 29, 2024.
    - shipping_nov8_2024_urls: URLs that support the claim shipping began on November 8, 2024.

    IMPORTANT:
    - Extract only URLs explicitly present in the answer. Do not invent or infer URLs.
    - Include full URLs. Markdown links are acceptable; extract the actual URL.
    - If the answer provides no URL for a field, return an empty array for that field.
    - Do not deduplicate; include all URLs cited for a given claim exactly as presented.
    - If the product_name is missing, set it to null.
    """


# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #
def _has_sources(urls: List[str]) -> bool:
    return bool(urls) and len(urls) > 0


def _fail_node_due_to_missing_sources(node):
    node.score = 0.0
    node.status = "failed"


def _safe_product_name(name: Optional[str]) -> str:
    return name.strip() if (name and isinstance(name, str)) else "the identified product"


# --------------------------------------------------------------------------- #
# Verification builder                                                        #
# --------------------------------------------------------------------------- #
async def build_and_verify_nodes(evaluator: Evaluator, parent_node, info: ProductSpecSources) -> None:
    """
    Build leaf nodes per rubric and perform verification against cited URLs.
    All leaf nodes are critical under the root (as required).
    """

    # 1) Product name provided (custom check)
    evaluator.add_custom_node(
        result=bool(info.product_name) and len(info.product_name.strip()) > 0,
        id="product_name_provided",
        desc="The answer provides the product name (specific desktop computer model).",
        parent=parent_node,
        critical=True
    )

    pname = _safe_product_name(info.product_name)

    # 2) Is desktop computer (with citation)
    node_desktop = evaluator.add_leaf(
        id="is_desktop_computer",
        desc="The identified product is a desktop computer (not a laptop/tablet/phone), supported by at least one reference URL.",
        parent=parent_node,
        critical=True
    )
    if _has_sources(info.desktop_proof_urls):
        claim = f"{pname} is a desktop computer (not a laptop/tablet/phone)."
        await evaluator.verify(
            claim=claim,
            node=node_desktop,
            sources=info.desktop_proof_urls,
            additional_instruction=(
                "Accept pages that explicitly categorize it as a 'desktop', 'mini PC', 'tower', or 'all-in-one'. "
                "Pages describing laptops/tablets/phones should not count. "
                "If the page clearly shows it is a desktop product line, that is sufficient."
            ),
        )
    else:
        _fail_node_due_to_missing_sources(node_desktop)

    # 3) Released in November 2024 (with citation)
    node_release = evaluator.add_leaf(
        id="release_month_year_with_citation",
        desc="The identified device was released (became available to customers) in November 2024, supported by at least one reference URL.",
        parent=parent_node,
        critical=True
    )
    if _has_sources(info.release_month_year_urls):
        claim = f"{pname} became available to customers in November 2024."
        await evaluator.verify(
            claim=claim,
            node=node_release,
            sources=info.release_month_year_urls,
            additional_instruction=(
                "Look for wording such as 'available in November 2024', 'on sale in Nov 2024', "
                "'available to order in Nov 2024', or a shipping start date in November 2024."
            ),
        )
    else:
        _fail_node_due_to_missing_sources(node_release)

    # 4) Processor manufactured using TSMC N3E (3nm Enhanced) (with citation)
    node_n3e = evaluator.add_leaf(
        id="processor_n3e_with_citation",
        desc="The device’s processor is manufactured using TSMC N3E (3nm Enhanced) process technology, supported by at least one reference URL.",
        parent=parent_node,
        critical=True
    )
    if _has_sources(info.processor_n3e_urls):
        claim = f"The processor used in {pname} is manufactured using TSMC N3E (3nm Enhanced) process technology."
        await evaluator.verify(
            claim=claim,
            node=node_n3e,
            sources=info.processor_n3e_urls,
            additional_instruction=(
                "Accept synonyms like 'TSMC N3E', '3nm (N3E)', 'TSMC's N3E 3nm node'. "
                "The statement must clearly tie the product's CPU/SoC to TSMC N3E."
            ),
        )
    else:
        _fail_node_due_to_missing_sources(node_n3e)

    # 5) Starting price $599 (with citation)
    node_price = evaluator.add_leaf(
        id="starting_price_599_with_citation",
        desc="The device has a starting price of $599 USD, supported by at least one reference URL.",
        parent=parent_node,
        critical=True
    )
    if _has_sources(info.starting_price_599_urls):
        claim = f"The starting price of {pname} is $599 USD."
        await evaluator.verify(
            claim=claim,
            node=node_price,
            sources=info.starting_price_599_urls,
            additional_instruction="Wording such as 'starting at $599' or 'from $599' counts as supporting evidence.",
        )
    else:
        _fail_node_due_to_missing_sources(node_price)

    # 6) Base memory 16GB unified (with citation)
    node_mem = evaluator.add_leaf(
        id="base_memory_16gb_unified_with_citation",
        desc="The base configuration includes 16GB of unified memory, supported by at least one reference URL.",
        parent=parent_node,
        critical=True
    )
    if _has_sources(info.base_memory_16_unified_urls):
        claim = f"The base configuration of {pname} includes 16GB of unified memory."
        await evaluator.verify(
            claim=claim,
            node=node_mem,
            sources=info.base_memory_16_unified_urls,
            additional_instruction="Look for 'unified memory' or equivalent phrasing; base configuration must list 16GB.",
        )
    else:
        _fail_node_due_to_missing_sources(node_mem)

    # 7) Base storage 256GB (with citation)
    node_storage = evaluator.add_leaf(
        id="base_storage_256gb_with_citation",
        desc="The base configuration includes 256GB of storage, supported by at least one reference URL.",
        parent=parent_node,
        critical=True
    )
    if _has_sources(info.base_storage_256_urls):
        claim = f"The base configuration of {pname} includes 256GB of storage."
        await evaluator.verify(
            claim=claim,
            node=node_storage,
            sources=info.base_storage_256_urls,
            additional_instruction="Base SKU storage must be 256GB; SSD wording is fine.",
        )
    else:
        _fail_node_due_to_missing_sources(node_storage)

    # 8) CPU 10-core (4 performance + 6 efficiency) (with citation)
    node_cpu = evaluator.add_leaf(
        id="cpu_10core_4p6e_with_citation",
        desc="The device features a 10-core CPU consisting of 4 performance cores and 6 efficiency cores, supported by at least one reference URL.",
        parent=parent_node,
        critical=True
    )
    if _has_sources(info.cpu_10core_4p6e_urls):
        claim = f"The CPU in {pname} has 10 cores configured as 4 performance cores and 6 efficiency cores."
        await evaluator.verify(
            claim=claim,
            node=node_cpu,
            sources=info.cpu_10core_4p6e_urls,
            additional_instruction="Accept '4P+6E', '4 performance + 6 efficiency', or equivalent phrasing.",
        )
    else:
        _fail_node_due_to_missing_sources(node_cpu)

    # 9) Base GPU 10-core (with citation)
    node_gpu = evaluator.add_leaf(
        id="base_gpu_10core_with_citation",
        desc="The base configuration includes a 10-core GPU, supported by at least one reference URL.",
        parent=parent_node,
        critical=True
    )
    if _has_sources(info.base_gpu_10core_urls):
        claim = f"The base configuration of {pname} includes a 10-core GPU."
        await evaluator.verify(
            claim=claim,
            node=node_gpu,
            sources=info.base_gpu_10core_urls,
            additional_instruction="The base SKU must list a GPU with 10 cores; minor wording variations are acceptable.",
        )
    else:
        _fail_node_due_to_missing_sources(node_gpu)

    # 10) Preorder Oct 29, 2024 (with citation)
    node_preorder = evaluator.add_leaf(
        id="preorder_oct29_2024_with_citation",
        desc="The device was available for pre-order starting October 29, 2024, supported by at least one reference URL.",
        parent=parent_node,
        critical=True
    )
    if _has_sources(info.preorder_oct29_2024_urls):
        claim = f"Pre-orders for {pname} began on October 29, 2024."
        await evaluator.verify(
            claim=claim,
            node=node_preorder,
            sources=info.preorder_oct29_2024_urls,
            additional_instruction="Look for explicit pre-order start date wording like 'preorder begins Oct 29, 2024'.",
        )
    else:
        _fail_node_due_to_missing_sources(node_preorder)

    # 11) Shipping Nov 8, 2024 (with citation)
    node_shipping = evaluator.add_leaf(
        id="shipping_nov8_2024_with_citation",
        desc="The device became officially available for shipping on November 8, 2024, supported by at least one reference URL.",
        parent=parent_node,
        critical=True
    )
    if _has_sources(info.shipping_nov8_2024_urls):
        claim = f"Shipping for {pname} began on November 8, 2024."
        await evaluator.verify(
            claim=claim,
            node=node_shipping,
            sources=info.shipping_nov8_2024_urls,
            additional_instruction="Accept phrasing like 'ships starting Nov 8, 2024' or 'available for delivery on Nov 8, 2024'.",
        )
    else:
        _fail_node_due_to_missing_sources(node_shipping)

    # Optional: record counts for transparency
    evaluator.add_custom_info(
        info={
            "product_name": info.product_name,
            "source_counts": {
                "desktop_proof_urls": len(info.desktop_proof_urls),
                "release_month_year_urls": len(info.release_month_year_urls),
                "processor_n3e_urls": len(info.processor_n3e_urls),
                "starting_price_599_urls": len(info.starting_price_599_urls),
                "base_memory_16_unified_urls": len(info.base_memory_16_unified_urls),
                "base_storage_256_urls": len(info.base_storage_256_urls),
                "cpu_10core_4p6e_urls": len(info.cpu_10core_4p6e_urls),
                "base_gpu_10core_urls": len(info.base_gpu_10core_urls),
                "preorder_oct29_2024_urls": len(info.preorder_oct29_2024_urls),
                "shipping_nov8_2024_urls": len(info.shipping_nov8_2024_urls),
            }
        },
        info_type="source_statistics",
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
    Evaluate an answer for the desktop computer identification task with N3E and November 2024 constraints.
    """
    # Initialize evaluator (root node is created internally with non-critical default)
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Independent checks; all are critical children
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

    # Extract product name and per-spec URLs from the answer text
    extracted_info = await evaluator.extract(
        prompt=prompt_extract_product_and_sources(),
        template_class=ProductSpecSources,
        extraction_name="product_spec_sources"
    )

    # Build leaf nodes and run verifications
    await build_and_verify_nodes(evaluator, root, extracted_info)

    # Return structured summary
    return evaluator.get_summary()