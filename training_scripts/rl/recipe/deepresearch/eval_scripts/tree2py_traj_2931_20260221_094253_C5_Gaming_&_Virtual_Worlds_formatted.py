import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.llm_client.base_client import LLMClient

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "gaming_infra_2026_dallas"
TASK_DESCRIPTION = (
    "A competitive gaming organization is hosting an esports tournament in Dallas, Texas in March 2026 "
    "with matches featuring 40 concurrent players per game instance. Research and provide a complete "
    "infrastructure solution including: (1) Identify a suitable data center facility located in the Dallas "
    "metropolitan area, (2) Specify the minimum server hardware requirements according to documented "
    "industry standards for hosting 40-player matches: RAM capacity (in GB), CPU specifications (vCPU count "
    "and clock speed in GHz), Storage capacity and type (in GB, specify SSD/NVMe), and (3) State the network "
    "infrastructure requirements for competitive gaming (bandwidth and latency). All components must include "
    "reference URLs from credible industry sources documenting the specifications."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class DataCenterInfo(BaseModel):
    name: Optional[str] = None
    location_text: Optional[str] = None
    url: Optional[str] = None


class RAMSpec(BaseModel):
    ram_gb: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CPUSpec(BaseModel):
    vcpu_count: Optional[str] = None
    clock_speed_ghz: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class StorageSpec(BaseModel):
    capacity_gb: Optional[str] = None
    storage_type: Optional[str] = None  # e.g., SSD, NVMe
    sources: List[str] = Field(default_factory=list)


class NetworkSpec(BaseModel):
    bandwidth_mbps: Optional[str] = None
    latency_ms: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class InfrastructureExtraction(BaseModel):
    data_center: Optional[DataCenterInfo] = None
    ram: Optional[RAMSpec] = None
    cpu: Optional[CPUSpec] = None
    storage: Optional[StorageSpec] = None
    network: Optional[NetworkSpec] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_infrastructure() -> str:
    return """
    Extract the complete infrastructure solution details exactly as provided in the answer for the Dallas esports tournament (40 concurrent players per game instance). Return a single JSON object containing the following fields:

    data_center:
      - name: Specific data center facility name (e.g., "Equinix DA11", "Colo Provider XYZ Dallas")
      - location_text: The location/address/city text as stated (e.g., "Dallas, TX", "Plano, TX", "DFW area")
      - url: A single URL documenting the data center facility (official provider page preferred)

    ram:
      - ram_gb: RAM capacity in GB (string; keep units or ranges if present, e.g., "32 GB", "64–128 GB")
      - sources: Array of URLs from the answer explicitly supporting RAM requirements for multiplayer or game servers sized for about 40 players

    cpu:
      - vcpu_count: Number of vCPU/cores specified (string; e.g., "8 vCPUs", "16 cores")
      - clock_speed_ghz: CPU clock speed in GHz (string; e.g., "3.5 GHz", ">= 3.0 GHz")
      - sources: Array of URLs supporting CPU requirements for hosting competitive/multiplayer servers around 40 players

    storage:
      - capacity_gb: Storage capacity in GB (string; e.g., "500 GB", "1 TB")
      - storage_type: Storage type (string; e.g., "SSD", "NVMe SSD")
      - sources: Array of URLs supporting storage specifications/requirements for multiplayer servers (prefer guidance mentioning SSD/NVMe)

    network:
      - bandwidth_mbps: Bandwidth requirement in Mbps (string; e.g., "100 Mbps", "1 Gbps")
      - latency_ms: Latency requirement in milliseconds (string; e.g., "< 20 ms", "≤ 50 ms")
      - sources: Array of URLs supporting competitive gaming network requirements (bandwidth/latency)

    IMPORTANT:
    - Extract ONLY what the answer explicitly states. Do not invent values or URLs.
    - For each 'sources' array, include all URLs in the answer that support that specific component (credible industry sources, vendor docs, engine docs, cloud provider docs, well-known platform guidance). If no URLs are present for a component, return an empty array.
    - If any field is not present in the answer, return null for that field.
    - Normalize obvious malformed URLs only by adding http:// if the protocol is missing; otherwise return them as-is.
    """


# --------------------------------------------------------------------------- #
# Verification helpers                                                        #
# --------------------------------------------------------------------------- #
def _has_non_empty(val: Optional[str]) -> bool:
    return bool(val) and bool(str(val).strip())


def _has_any_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


# --------------------------------------------------------------------------- #
# Verification functions for each branch                                      #
# --------------------------------------------------------------------------- #
async def verify_data_center(
    evaluator: Evaluator,
    parent_node,
    dc: Optional[DataCenterInfo],
) -> None:
    node = evaluator.add_parallel(
        id="Data_Center_Facility",
        desc="Appropriate data center identified in Dallas area with documentation",
        parent=parent_node,
        critical=True,  # Make the facility branch mandatory for complete solution
    )

    # Facility_Named (existence)
    evaluator.add_custom_node(
        result=(dc is not None and _has_non_empty(dc.name)),
        id="Facility_Named",
        desc="Specific data center facility name provided",
        parent=node,
        critical=True,
    )

    # Located_In_Dallas_Area (verify with URL)
    located_leaf = evaluator.add_leaf(
        id="Located_In_Dallas_Area",
        desc="Confirmed location is in Dallas metropolitan area, Texas",
        parent=node,
        critical=True,
    )
    facility_name = dc.name if dc and dc.name else ""
    facility_url = dc.url if dc and dc.url else None
    claim_loc = (
        f"The data center facility '{facility_name}' is located in the Dallas metropolitan area in the state of Texas."
    )
    await evaluator.verify(
        claim=claim_loc,
        node=located_leaf,
        sources=facility_url,
        additional_instruction=(
            "Check the facility page for city/address. Consider Dallas metro/DFW area acceptable (e.g., Dallas, "
            "Plano, Irving, Richardson, Addison, Carrollton, Garland, or similar in the Dallas-Fort Worth area). "
            "If the page shows any of these localities or clearly states Dallas/DFW, treat as supported."
        ),
    )

    # Reference_URL (verify the URL documents the facility)
    ref_leaf = evaluator.add_leaf(
        id="Data_Center_Reference_URL",
        desc="Valid reference URL provided documenting the data center facility",
        parent=node,
        critical=True,
    )
    claim_ref = (
        f"This webpage is an official or authoritative documentation page for the data center facility '{facility_name}'."
    )
    await evaluator.verify(
        claim=claim_ref,
        node=ref_leaf,
        sources=facility_url,
        additional_instruction=(
            "The page should clearly document the facility (provider site, official listing, credible colocation "
            "profile). Generic or unrelated pages should not be considered valid."
        ),
    )


async def verify_ram(
    evaluator: Evaluator,
    parent_node,
    ram: Optional[RAMSpec],
) -> None:
    node = evaluator.add_parallel(
        id="RAM_Specifications",
        desc="RAM requirements specified according to industry standards for 40-player capacity with proper documentation",
        parent=parent_node,
        critical=True,  # Mandatory component
    )

    # RAM_Amount_Specified (existence)
    evaluator.add_custom_node(
        result=(ram is not None and _has_non_empty(ram.ram_gb)),
        id="RAM_Amount_Specified",
        desc="Specific RAM amount in GB provided",
        parent=node,
        critical=True,
    )

    # RAM_Meets_Industry_Standards (verify support in sources)
    meets_leaf = evaluator.add_leaf(
        id="RAM_Meets_Industry_Standards",
        desc="RAM amount is appropriate for 40 concurrent players based on documented industry standards for multiplayer servers",
        parent=node,
        critical=True,
    )
    ram_val = ram.ram_gb if ram and ram.ram_gb else ""
    ram_sources = ram.sources if ram else []
    claim_ram = (
        f"According to the cited source(s), hosting around 40 concurrent players per server instance requires at least {ram_val} of RAM, "
        "or the recommended RAM equals/exceeds this value."
    )
    await evaluator.verify(
        claim=claim_ram,
        node=meets_leaf,
        sources=ram_sources,
        additional_instruction=(
            "Look for pages that document multiplayer/dedicated server memory requirements or instance sizing guidelines. "
            "If a source provides a recommendation or minimum that matches or exceeds the provided RAM value for ~40 players, treat as supported. "
            "Minor unit formatting differences (e.g., GB vs GiB) are acceptable."
        ),
    )

    # Reference_URL (verify credibility/relevance of sources)
    ref_leaf = evaluator.add_leaf(
        id="RAM_Reference_URL",
        desc="Valid reference URL provided supporting RAM specification from credible industry source",
        parent=node,
        critical=True,
    )
    claim_ram_ref = (
        "The cited URL(s) are credible industry sources (e.g., vendor docs, official engine documentation, cloud provider guidance, or well-known platform pages) "
        "that explicitly document RAM requirements/recommendations for multiplayer/dedicated gaming servers."
    )
    await evaluator.verify(
        claim=claim_ram_ref,
        node=ref_leaf,
        sources=ram_sources,
        additional_instruction=(
            "Assess credibility and relevance: official vendor/cloud docs or recognized platforms are credible; "
            "ensure the page discusses RAM requirements for game servers or high-concurrency workloads."
        ),
    )


async def verify_cpu(
    evaluator: Evaluator,
    parent_node,
    cpu: Optional[CPUSpec],
) -> None:
    node = evaluator.add_parallel(
        id="CPU_Specifications",
        desc="CPU requirements specified according to industry standards with proper documentation",
        parent=parent_node,
        critical=True,  # Mandatory component
    )

    # vCPU_Count_Specified (existence)
    evaluator.add_custom_node(
        result=(cpu is not None and _has_non_empty(cpu.vcpu_count)),
        id="vCPU_Count_Specified",
        desc="Number of vCPU cores specified",
        parent=node,
        critical=True,
    )

    # Clock_Speed_Specified (existence)
    evaluator.add_custom_node(
        result=(cpu is not None and _has_non_empty(cpu.clock_speed_ghz)),
        id="Clock_Speed_Specified",
        desc="CPU clock speed in GHz specified",
        parent=node,
        critical=True,
    )

    # CPU_Meets_Industry_Standards (verify both vCPU count and clock speed for ~40 players)
    meets_leaf = evaluator.add_leaf(
        id="CPU_Meets_Industry_Standards",
        desc="CPU specifications meet documented industry standards for competitive gaming servers supporting 40 players",
        parent=node,
        critical=True,
    )
    vcpu_val = cpu.vcpu_count if cpu and cpu.vcpu_count else ""
    clock_val = cpu.clock_speed_ghz if cpu and cpu.clock_speed_ghz else ""
    cpu_sources = cpu.sources if cpu else []
    claim_cpu = (
        f"According to the cited source(s), to host approximately 40 concurrent players, a server CPU with {vcpu_val} "
        f"and a clock speed around {clock_val} (or higher) is required or recommended for competitive/multiplayer servers."
    )
    await evaluator.verify(
        claim=claim_cpu,
        node=meets_leaf,
        sources=cpu_sources,
        additional_instruction=(
            "Focus on documented recommendations for CPU cores/vCPU and clock speed for multiplayer/dedicated servers. "
            "Allow reasonable equivalents (e.g., per-core GHz vs boost frequencies). If the source recommends equal or higher specs, treat as supported."
        ),
    )

    # Reference_URL (verify credibility/relevance)
    ref_leaf = evaluator.add_leaf(
        id="CPU_Reference_URL",
        desc="Valid reference URL provided supporting CPU specifications from credible industry source",
        parent=node,
        critical=True,
    )
    claim_cpu_ref = (
        "The cited URL(s) are credible industry sources that explicitly document CPU requirements or sizing guidance for multiplayer/dedicated gaming servers."
    )
    await evaluator.verify(
        claim=claim_cpu_ref,
        node=ref_leaf,
        sources=cpu_sources,
        additional_instruction=(
            "Evaluate whether the page is an official vendor/cloud doc, engine/platform guidance, or similarly credible source relevant to CPU sizing for game servers."
        ),
    )


async def verify_storage(
    evaluator: Evaluator,
    parent_node,
    storage: Optional[StorageSpec],
) -> None:
    node = evaluator.add_parallel(
        id="Storage_Specifications",
        desc="Storage requirements specified according to industry standards with proper documentation",
        parent=parent_node,
        critical=True,  # Mandatory component
    )

    # Storage_Capacity_Specified (existence)
    evaluator.add_custom_node(
        result=(storage is not None and _has_non_empty(storage.capacity_gb)),
        id="Storage_Capacity_Specified",
        desc="Storage capacity in GB specified",
        parent=node,
        critical=True,
    )

    # Storage_Type_Specified (existence)
    evaluator.add_custom_node(
        result=(storage is not None and _has_non_empty(storage.storage_type)),
        id="Storage_Type_Specified",
        desc="Storage type specified (SSD, NVMe, or equivalent high-performance storage)",
        parent=node,
        critical=True,
    )

    # Storage_Meets_Industry_Standards (verify)
    meets_leaf = evaluator.add_leaf(
        id="Storage_Meets_Industry_Standards",
        desc="Storage specifications meet documented industry minimum for multiplayer servers",
        parent=node,
        critical=True,
    )
    cap_val = storage.capacity_gb if storage and storage.capacity_gb else ""
    type_val = storage.storage_type if storage and storage.storage_type else ""
    storage_sources = storage.sources if storage else []
    claim_storage = (
        f"According to the cited source(s), for multiplayer/competitive game servers, at least {cap_val} of storage "
        f"using {type_val} (or equivalent high-performance storage) is required or recommended."
    )
    await evaluator.verify(
        claim=claim_storage,
        node=meets_leaf,
        sources=storage_sources,
        additional_instruction=(
            "Look for guidance that mentions SSD/NVMe for game servers or high-concurrency workloads and capacity recommendations. "
            "If the source recommends equal or higher performance/type and similar capacity, treat as supported."
        ),
    )

    # Reference_URL (verify credibility/relevance)
    ref_leaf = evaluator.add_leaf(
        id="Storage_Reference_URL",
        desc="Valid reference URL provided supporting storage specifications from credible industry source",
        parent=node,
        critical=True,
    )
    claim_storage_ref = (
        "The cited URL(s) are credible industry sources that explicitly document storage type/capacity recommendations for multiplayer/dedicated game servers."
    )
    await evaluator.verify(
        claim=claim_storage_ref,
        node=ref_leaf,
        sources=storage_sources,
        additional_instruction=(
            "Official vendor/cloud docs, platform/engine guidance, or recognized performance recommendations are considered credible."
        ),
    )


async def verify_network(
    evaluator: Evaluator,
    parent_node,
    net: Optional[NetworkSpec],
) -> None:
    node = evaluator.add_parallel(
        id="Network_Requirements",
        desc="Network infrastructure requirements for competitive gaming specified with proper documentation",
        parent=parent_node,
        critical=True,  # Mandatory component
    )

    # Bandwidth_Specified (existence)
    evaluator.add_custom_node(
        result=(net is not None and _has_non_empty(net.bandwidth_mbps)),
        id="Bandwidth_Specified",
        desc="Minimum bandwidth requirement specified in Mbps",
        parent=node,
        critical=True,
    )

    # Latency_Specified (existence)
    evaluator.add_custom_node(
        result=(net is not None and _has_non_empty(net.latency_ms)),
        id="Latency_Specified",
        desc="Latency requirement specified in milliseconds",
        parent=node,
        critical=True,
    )

    # Network_Meets_Gaming_Standards (verify)
    meets_leaf = evaluator.add_leaf(
        id="Network_Meets_Gaming_Standards",
        desc="Network requirements meet documented competitive gaming standards",
        parent=node,
        critical=True,
    )
    bw_val = net.bandwidth_mbps if net and net.bandwidth_mbps else ""
    lat_val = net.latency_ms if net and net.latency_ms else ""
    net_sources = net.sources if net else []
    claim_net = (
        f"Competitive gaming standards (as documented by the cited source(s)) require at least {bw_val} of bandwidth and "
        f"a latency of {lat_val} or lower for reliable gameplay."
    )
    await evaluator.verify(
        claim=claim_net,
        node=meets_leaf,
        sources=net_sources,
        additional_instruction=(
            "Look for guidance that explicitly mentions bandwidth (Mbps/Gbps) and latency (ms) thresholds for competitive gaming. "
            "If the source recommends equal or stricter thresholds than those provided, treat as supported."
        ),
    )

    # Reference_URL (verify credibility/relevance)
    ref_leaf = evaluator.add_leaf(
        id="Network_Reference_URL",
        desc="Valid reference URL provided supporting network requirements from credible industry source",
        parent=node,
        critical=True,
    )
    claim_net_ref = (
        "The cited URL(s) are credible industry sources clearly documenting competitive gaming network requirements (bandwidth and latency)."
    )
    await evaluator.verify(
        claim=claim_net_ref,
        node=ref_leaf,
        sources=net_sources,
        additional_instruction=(
            "Examples: official platform/gaming provider docs, ISP/network vendor guidance for gaming, or recognized industry publications. "
            "Ensure the page discusses latency/bandwidth norms for competitive gaming."
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
) -> Dict[str, Any]:
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Components evaluated independently
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

    # IMPORTANT: Root must be non-critical to satisfy framework constraints (critical parent requires all children critical).
    # We enforce "complete solution" by marking each major branch as critical instead.

    # Extract structured info
    extracted = await evaluator.extract(
        prompt=prompt_extract_infrastructure(),
        template_class=InfrastructureExtraction,
        extraction_name="infrastructure_extraction",
    )

    # Build verification subtrees
    await verify_data_center(evaluator, root, extracted.data_center)
    await verify_ram(evaluator, root, extracted.ram)
    await verify_cpu(evaluator, root, extracted.cpu)
    await verify_storage(evaluator, root, extracted.storage)
    await verify_network(evaluator, root, extracted.network)

    # Optional: record custom info summary
    summary_info = {
        "data_center_name": extracted.data_center.name if extracted.data_center else None,
        "data_center_url": extracted.data_center.url if extracted.data_center else None,
        "ram": extracted.ram.ram_gb if extracted.ram else None,
        "cpu_vcpu": extracted.cpu.vcpu_count if extracted.cpu else None,
        "cpu_clock": extracted.cpu.clock_speed_ghz if extracted.cpu else None,
        "storage_capacity": extracted.storage.capacity_gb if extracted.storage else None,
        "storage_type": extracted.storage.storage_type if extracted.storage else None,
        "network_bandwidth": extracted.network.bandwidth_mbps if extracted.network else None,
        "network_latency": extracted.network.latency_ms if extracted.network else None,
        "source_counts": {
            "ram_sources": len(extracted.ram.sources) if extracted.ram else 0,
            "cpu_sources": len(extracted.cpu.sources) if extracted.cpu else 0,
            "storage_sources": len(extracted.storage.sources) if extracted.storage else 0,
            "network_sources": len(extracted.network.sources) if extracted.network else 0,
        },
    }
    evaluator.add_custom_info(summary_info, info_type="extraction_summary")

    return evaluator.get_summary()