import asyncio
import logging
from typing import Any, List, Optional, Dict

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy
from obj_task_eval.utils.cache_filesys import CacheFileSys


TASK_ID = "switch2_phoenix_retail_and_specs"
TASK_DESCRIPTION = (
    "A gamer living in Phoenix, Arizona wants to purchase the Nintendo Switch 2 console. "
    "Identify a major electronics retail chain that has a physical store location in Phoenix, and verify the following specifications of the Nintendo Switch 2: "
    "its official release date, manufacturer's suggested retail price, available storage capacity options, backward compatibility rate with original Nintendo Switch games, "
    "memory (RAM) specification, and memory bandwidth in performance mode."
)


class RetailerExtraction(BaseModel):
    retailer_name: Optional[str] = None
    phoenix_store_address: Optional[str] = None
    location_urls: List[str] = Field(default_factory=list)
    type_urls: List[str] = Field(default_factory=list)


class SwitchSpecSources(BaseModel):
    release_date_urls: List[str] = Field(default_factory=list)
    msrp_urls: List[str] = Field(default_factory=list)
    storage_urls: List[str] = Field(default_factory=list)
    bc_urls: List[str] = Field(default_factory=list)
    ram_urls: List[str] = Field(default_factory=list)
    bandwidth_urls: List[str] = Field(default_factory=list)


def prompt_extract_retailer() -> str:
    return """
    From the answer, extract information about the recommended retailer for buying the Nintendo Switch 2 in Phoenix, Arizona.

    Return the following fields:
    - retailer_name: The retailer/chain name stated in the answer (e.g., Best Buy, GameStop, Target, Walmart, etc.).
    - phoenix_store_address: The street address for a Phoenix, AZ store if the answer explicitly provides it; otherwise null.
    - location_urls: All URLs mentioned in the answer that specifically show a Phoenix, Arizona store location or store locator page for this retailer (e.g., a store detail page with city/state and street address).
    - type_urls: All URLs mentioned that demonstrate the retailer is a major national electronics retail chain (e.g., Wikipedia page, official corporate site, or press articles indicating nationwide presence).

    Rules:
    - Only include URLs explicitly present in the answer (plain URLs or markdown links). Do not invent or search for new URLs.
    - Deduplicate URLs and ensure they are complete and valid (prepend http:// if protocol missing).
    - If a single URL supports both the Phoenix location and chain type, include it in both lists.
    """


def prompt_extract_switch2_sources() -> str:
    return """
    Extract all source URLs cited in the answer for each Nintendo Switch 2 specification category below.

    Return:
    - release_date_urls: URLs that the answer uses to support the official release date.
    - msrp_urls: URLs that support the MSRP/manufacturer suggested retail price.
    - storage_urls: URLs that support the available storage options (e.g., 128GB and 512GB).
    - bc_urls: URLs that support backward compatibility rate with original Nintendo Switch games.
    - ram_urls: URLs that support the memory/RAM specification.
    - bandwidth_urls: URLs that support memory bandwidth in performance mode.

    Rules:
    - Only include URLs explicitly present in the answer (plain URLs or markdown links). Do not invent or search for new URLs.
    - Deduplicate URLs and ensure they are complete and valid (prepend http:// if protocol missing).
    - If the answer cites one URL for multiple specs, include that URL in each relevant list.
    - If no URL is provided for a spec, return an empty list for that spec.
    """


async def verify_retailer_nodes(evaluator: Evaluator, root, retailer: RetailerExtraction) -> None:
    # Retailer_Location (Critical)
    location_node = evaluator.add_leaf(
        id="Retailer_Location",
        desc="The retailer has at least one physical store location in Phoenix, Arizona with a verifiable address",
        parent=root,
        critical=True,
    )
    retailer_name = retailer.retailer_name or "the retailer"
    claim_location = (
        f"{retailer_name} has at least one physical store location in Phoenix, Arizona with a verifiable street address."
    )
    await evaluator.verify(
        claim=claim_location,
        node=location_node,
        sources=retailer.location_urls,
        additional_instruction=(
            "Confirm the page shows a store in Phoenix, AZ and includes a clear address (street + city/state). "
            "Accept any Phoenix location for the chain. Ignore non‑Phoenix locations."
        ),
    )

    # Retailer_Type (Critical)
    type_node = evaluator.add_leaf(
        id="Retailer_Type",
        desc="The retailer is a major national electronics retail chain (Best Buy, GameStop, Target, Walmart, or similar established chain)",
        parent=root,
        critical=True,
    )
    claim_type = (
        f"{retailer_name} is a major national electronics retail chain with widespread presence in the United States."
    )
    await evaluator.verify(
        claim=claim_type,
        node=type_node,
        sources=retailer.type_urls,
        additional_instruction=(
            "Verify that the retailer is a widely recognized national chain (e.g., Best Buy, GameStop, Target, Walmart, "
            "or similar) with many stores nationwide; Wikipedia or corporate pages indicating nationwide presence are acceptable."
        ),
    )


async def verify_switch2_specs(evaluator: Evaluator, root, sources: SwitchSpecSources) -> None:
    # Release_Date (Critical)
    release_node = evaluator.add_leaf(
        id="Release_Date",
        desc="The Nintendo Switch 2's official release date is verified as June 5, 2025",
        parent=root,
        critical=True,
    )
    claim_release = "The official release date of the Nintendo Switch 2 is June 5, 2025."
    await evaluator.verify(
        claim=claim_release,
        node=release_node,
        sources=sources.release_date_urls,
        additional_instruction=(
            "Confirm the page explicitly states the Nintendo Switch 2 release/launch date as June 5, 2025. "
            "Allow minor formatting variations (e.g., 'June 5th, 2025')."
        ),
    )

    # Retail_Price (Non-Critical)
    price_node = evaluator.add_leaf(
        id="Retail_Price",
        desc="The Nintendo Switch 2's manufacturer suggested retail price is verified as $449.99 USD",
        parent=root,
        critical=False,
    )
    claim_price = "The manufacturer suggested retail price (MSRP) of the Nintendo Switch 2 is $449.99 USD."
    await evaluator.verify(
        claim=claim_price,
        node=price_node,
        sources=sources.msrp_urls,
        additional_instruction=(
            "Confirm the MSRP listed is $449.99 (USD). Accept equivalent notation like 'US$449.99'."
        ),
    )

    # Storage_Options (Non-Critical)
    storage_node = evaluator.add_leaf(
        id="Storage_Options",
        desc="The Nintendo Switch 2 is confirmed to be available in both 128GB and 512GB storage configurations",
        parent=root,
        critical=False,
    )
    claim_storage = "The Nintendo Switch 2 is available in both 128GB and 512GB storage configurations."
    await evaluator.verify(
        claim=claim_storage,
        node=storage_node,
        sources=sources.storage_urls,
        additional_instruction=(
            "Confirm the page explicitly mentions both 128GB and 512GB storage configurations for Nintendo Switch 2."
        ),
    )

    # Backward_Compatibility (Non-Critical)
    bc_node = evaluator.add_leaf(
        id="Backward_Compatibility",
        desc="The Nintendo Switch 2's backward compatibility rate with original Nintendo Switch games is at least 99%",
        parent=root,
        critical=False,
    )
    claim_bc = (
        "The Nintendo Switch 2 has backward compatibility with original Nintendo Switch games at a rate of at least 99%."
    )
    await evaluator.verify(
        claim=claim_bc,
        node=bc_node,
        sources=sources.bc_urls,
        additional_instruction=(
            "Confirm that the page states a compatibility rate of 99% or higher with the original Nintendo Switch game library."
        ),
    )

    # Memory_Specification (Non-Critical)
    ram_node = evaluator.add_leaf(
        id="Memory_Specification",
        desc="The Nintendo Switch 2 is confirmed to have 12GB LPDDR5X RAM",
        parent=root,
        critical=False,
    )
    claim_ram = "The Nintendo Switch 2 has 12GB of LPDDR5X RAM."
    await evaluator.verify(
        claim=claim_ram,
        node=ram_node,
        sources=sources.ram_urls,
        additional_instruction=(
            "Confirm the RAM specification is explicitly listed as 12GB LPDDR5X."
        ),
    )

    # Memory_Bandwidth (Non-Critical)
    bandwidth_node = evaluator.add_leaf(
        id="Memory_Bandwidth",
        desc="The Nintendo Switch 2's memory bandwidth in performance mode is verified as 102GB/s",
        parent=root,
        critical=False,
    )
    claim_bw = "The Nintendo Switch 2 memory bandwidth in performance mode is 102 GB/s."
    await evaluator.verify(
        claim=claim_bw,
        node=bandwidth_node,
        sources=sources.bandwidth_urls,
        additional_instruction=(
            "Confirm the page specifies memory bandwidth in performance mode as 102 GB/s. "
            "Allow minor spacing variations (e.g., '102GB/s', '102 GB/s')."
        ),
    )


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

    retailer_info = await evaluator.extract(
        prompt=prompt_extract_retailer(),
        template_class=RetailerExtraction,
        extraction_name="retailer_info",
    )

    spec_sources = await evaluator.extract(
        prompt=prompt_extract_switch2_sources(),
        template_class=SwitchSpecSources,
        extraction_name="switch2_spec_sources",
    )

    evaluator.add_ground_truth(
        {
            "expected_specs": {
                "release_date": "June 5, 2025",
                "msrp_usd": "$449.99",
                "storage_options": ["128GB", "512GB"],
                "backward_compatibility_min_rate": ">=99%",
                "ram_spec": "12GB LPDDR5X",
                "memory_bandwidth_performance_mode": "102 GB/s",
            }
        },
        gt_type="ground_truth",
    )

    await verify_retailer_nodes(evaluator, root, retailer_info)
    await verify_switch2_specs(evaluator, root, spec_sources)

    return evaluator.get_summary()