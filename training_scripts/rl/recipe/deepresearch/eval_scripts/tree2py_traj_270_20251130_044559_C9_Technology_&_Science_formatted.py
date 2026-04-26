import asyncio
import logging
from typing import Optional, List, Dict, Any, Set

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "dgx_h100_4node_spec"
TASK_DESCRIPTION = """
Design a comprehensive technical specification document for deploying a 4-system AI training cluster using NVIDIA DGX H100 systems in a major United States data center market. Your specification must address the following thirteen categories with detailed technical requirements and supporting URL references from authoritative sources:

1. System Hardware Selection: Identify the system model, specify the 8U rackmount form factor, confirm 8 H100 GPUs per system, and provide detailed GPU specifications including TDP (700W), memory (80GB HBM3), and bandwidth (3 TB/s). Calculate total GPU power draw per system.

2. Power Supply Configuration: Specify the six power supplies (3300W each @ 200-240V, 16A, 50-60Hz), minimum operational requirements (4 of 6 supplies energized, 3 circuits), maximum system power (10.2 kW), and preferred configuration (415 VAC, 32A, three-phase, N+1).

3. Cooling Infrastructure: Define minimum airflow (300 L/s per GPU, 2400 L/s total), specify hot/cold aisle containment architecture, and establish operating temperature range (5-30°C).

4. Rack Infrastructure: Specify rack weight capacity (350 lbs per U minimum) and power density (60+ kW per rack), with explanation of why traditional 5-10 kW racks are insufficient.

5. Memory Specifications: Calculate total memory per system (640GB) and total cluster memory (2.56TB).

6. Network Infrastructure: Specify minimum bandwidth (100+ Gbps) and optimal bandwidth for distributed training (800 Gbps - 1.6 Tbps between nodes).

7. Data Center Tier Requirements: Specify minimum Tier III certification, define concurrent maintainability and redundant distribution path requirements, and optionally note Tier IV fault tolerance features.

8. Energy Efficiency: Specify target PUE (≤1.2) and acceptable range (1.2-1.8).

9. Deployment Scale: Confirm 4-system minimum for multi-node training, calculate total IT power (40.8 kW), estimate facility power with PUE, and calculate rack space (32U minimum).

10. Environmental Requirements: Specify temperature control (5-30°C) and humidity control measures.

11. Physical Space Requirements: Calculate number of racks needed based on power density and estimate floor space.

12. Redundancy and Reliability: Specify power redundancy (N+1 or better), cooling redundancy, and network redundancy configurations.

13. Geographic Location: Identify a specific major US data center market (e.g., Northern Virginia, Chicago, Phoenix, Dallas, etc.) with MW-scale power capacity and supporting infrastructure.

All technical specifications must be supported by URL references to manufacturer datasheets, industry standards documentation, data center provider specifications, or authoritative technical resources.
"""


# --------------------------------------------------------------------------- #
# Data models                                                                 #
# --------------------------------------------------------------------------- #
class HardwareSelection(BaseModel):
    system_model: Optional[str] = None
    form_factor: Optional[str] = None
    gpu_count: Optional[str] = None
    gpu_tdp_watts: Optional[str] = None
    gpu_memory: Optional[str] = None
    gpu_memory_bandwidth: Optional[str] = None
    total_gpu_power_draw: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PowerSupplyConfig(BaseModel):
    psu_count: Optional[str] = None
    psu_rating: Optional[str] = None
    min_operational_psus: Optional[str] = None
    min_circuits: Optional[str] = None
    max_system_power: Optional[str] = None
    preferred_power_feed: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class CoolingInfrastructure(BaseModel):
    airflow_per_gpu: Optional[str] = None
    airflow_total: Optional[str] = None
    hot_cold_aisle_containment: Optional[str] = None
    operating_temp_range: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RackInfrastructure(BaseModel):
    rack_weight_capacity: Optional[str] = None
    rack_power_density: Optional[str] = None
    traditional_rack_insufficient_explanation: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MemorySpecifications(BaseModel):
    system_memory_total: Optional[str] = None
    cluster_memory_total: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class NetworkInfrastructure(BaseModel):
    min_bandwidth: Optional[str] = None
    optimal_bandwidth_range: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DataCenterTierRequirements(BaseModel):
    min_tier: Optional[str] = None
    tier_concurrent_maintainability: Optional[str] = None
    tier_redundant_distribution_paths: Optional[str] = None
    tier_iv_optional_note: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EnergyEfficiency(BaseModel):
    target_pue: Optional[str] = None
    acceptable_pue_range: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class DeploymentScale(BaseModel):
    min_system_count: Optional[str] = None
    total_it_power: Optional[str] = None
    facility_power_estimate_with_pue: Optional[str] = None
    rack_space_u: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class EnvironmentalRequirements(BaseModel):
    temperature_range: Optional[str] = None
    humidity_control_measures: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class PhysicalSpaceRequirements(BaseModel):
    rack_count_estimate: Optional[str] = None
    floor_space_estimate: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class RedundancyReliability(BaseModel):
    power_redundancy: Optional[str] = None
    cooling_redundancy: Optional[str] = None
    network_redundancy: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class GeographicLocation(BaseModel):
    market_name: Optional[str] = None
    supports_mw_scale_power: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class ReferencesExtraction(BaseModel):
    urls: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompts                                                          #
# --------------------------------------------------------------------------- #
def prompt_extract_hardware() -> str:
    return """
Extract the system hardware selection details stated in the answer. Return:
- system_model: model name (e.g., "NVIDIA DGX H100")
- form_factor: form factor as written (e.g., "8U rackmount")
- gpu_count: number of H100 GPUs per system (as written)
- gpu_tdp_watts: H100 GPU TDP value as written (e.g., "700W")
- gpu_memory: H100 GPU memory as written (e.g., "80GB HBM3")
- gpu_memory_bandwidth: H100 memory bandwidth as written (e.g., "3 TB/s")
- total_gpu_power_draw: total GPU power per system as written (e.g., "5.6 kW (8 × 700W)")
- sources: list of URLs in the answer that support these hardware details. Use only URLs explicitly present in the answer.
If any field is missing, set it to null; if no sources, return an empty list.
"""


def prompt_extract_power() -> str:
    return """
Extract the power supply and electrical requirements stated in the answer. Return:
- psu_count: number of power supplies per system (as written, e.g., "6")
- psu_rating: rating text as written (e.g., "3300W @ 200–240V, 16A, 50–60Hz")
- min_operational_psus: minimum operational requirement text (e.g., "4 of 6 PSUs energized")
- min_circuits: minimum number of power circuits per system (as written, e.g., "3 circuits")
- max_system_power: maximum system power consumption (as written, e.g., "10.2 kW")
- preferred_power_feed: preferred feed text (e.g., "415 VAC, 32A, three-phase")
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_cooling() -> str:
    return """
Extract the cooling infrastructure details stated in the answer. Return:
- airflow_per_gpu: minimum airflow per GPU (e.g., "300 L/s per GPU")
- airflow_total: total airflow for 8 GPUs (e.g., "2400 L/s")
- hot_cold_aisle_containment: statement about hot/cold aisle containment architecture
- operating_temp_range: operating temperature range (e.g., "5–30°C (41–86°F)")
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_rack() -> str:
    return """
Extract the rack infrastructure details stated in the answer. Return:
- rack_weight_capacity: rack weight capacity statement (e.g., "≥350 lbs per U")
- rack_power_density: rack power density statement (e.g., "≥60 kW per rack")
- traditional_rack_insufficient_explanation: brief text explaining why 5–10 kW racks are insufficient
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_memory() -> str:
    return """
Extract the memory calculations stated in the answer. Return:
- system_memory_total: total memory per system as written (e.g., "640GB (8 × 80GB)")
- cluster_memory_total: total cluster memory as written (e.g., "2.56TB (4 × 640GB)")
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_network() -> str:
    return """
Extract the network infrastructure requirements stated in the answer. Return:
- min_bandwidth: minimum bandwidth (e.g., ">=100 Gbps")
- optimal_bandwidth_range: optimal inter-node bandwidth range (e.g., "800 Gbps – 1.6 Tbps")
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_tier() -> str:
    return """
Extract the data center tier requirements stated in the answer. Return:
- min_tier: minimum tier required (e.g., "Tier III")
- tier_concurrent_maintainability: statement that Tier III requires concurrent maintainability
- tier_redundant_distribution_paths: statement that Tier III requires redundant distribution paths
- tier_iv_optional_note: optional Tier IV fault tolerance note if present; else null
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_pue() -> str:
    return """
Extract the energy efficiency requirements stated in the answer. Return:
- target_pue: target PUE (e.g., "≤1.2")
- acceptable_pue_range: acceptable PUE range (e.g., "1.2–1.8")
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_deployment_scale() -> str:
    return """
Extract the deployment scale items stated in the answer. Return:
- min_system_count: minimum system count (e.g., "4 systems")
- total_it_power: total IT power (e.g., "40.8 kW (4 × 10.2 kW)")
- facility_power_estimate_with_pue: facility power estimate and/or method using PUE as written
- rack_space_u: rack space required in U (e.g., "32U (4 × 8U)")
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_environmental() -> str:
    return """
Extract the environmental requirements stated in the answer. Return:
- temperature_range: operating temperature range (e.g., "5–30°C")
- humidity_control_measures: text describing humidity control measures
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_physical_space() -> str:
    return """
Extract the physical space requirements stated in the answer. Return:
- rack_count_estimate: number of racks needed based on power density and/or system count (as written)
- floor_space_estimate: floor space estimate for the deployment (as written)
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_redundancy() -> str:
    return """
Extract the redundancy and reliability requirements stated in the answer. Return:
- power_redundancy: power redundancy configuration (e.g., "N+1" or better)
- cooling_redundancy: cooling redundancy configuration (as written)
- network_redundancy: network redundancy configuration (as written)
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_location() -> str:
    return """
Extract the geographic location information stated in the answer. Return:
- market_name: the specific major US data center market identified (e.g., "Northern Virginia")
- supports_mw_scale_power: statement that this market supports MW-scale power delivery and relevant infrastructure
- sources: URLs that support these statements (from the answer text)
"""


def prompt_extract_references() -> str:
    return """
Extract all authoritative supporting URL references listed anywhere in the answer. Return:
- urls: an array of all URLs explicitly present in the answer that support the specification. Include manufacturer datasheets, industry standards, and provider specifications.
"""


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _dedup_urls(*lists: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for lst in lists:
        for url in lst or []:
            if not isinstance(url, str):
                continue
            u = url.strip()
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


def _nz(val: Optional[str], fallback: str = "unspecified") -> str:
    if val is None or str(val).strip() == "":
        return fallback
    return str(val).strip()


# --------------------------------------------------------------------------- #
# Verification builders                                                       #
# --------------------------------------------------------------------------- #
async def build_hardware_nodes(evaluator: Evaluator, parent, hw: HardwareSelection):
    cat = evaluator.add_parallel(
        id="system_hardware_selection",
        desc="System hardware selection requirements are satisfied",
        parent=parent,
        critical=True,
    )

    # system_model_identified
    n1 = evaluator.add_leaf(
        id="system_model_identified",
        desc="Identifies the system model as NVIDIA DGX H100",
        parent=cat,
        critical=True,
    )
    claim = f"The specification identifies the system model as 'NVIDIA DGX H100' (extracted value: '{_nz(hw.system_model)}')."
    await evaluator.verify(
        claim=claim,
        node=n1,
        sources=hw.sources,
        additional_instruction="Confirm that at least one provided authoritative source page is for NVIDIA DGX H100 systems."
    )

    # form_factor_8u
    n2 = evaluator.add_leaf(
        id="form_factor_8u",
        desc="Specifies the system form factor as 8U rackmount",
        parent=cat,
        critical=True,
    )
    claim = f"The DGX H100 system form factor is 8U rackmount (extracted value: '{_nz(hw.form_factor)}')."
    await evaluator.verify(
        claim=claim,
        node=n2,
        sources=hw.sources,
        additional_instruction="Look for '8U' or equivalent form factor language on the cited DGX H100 datasheet or product page."
    )

    # gpu_count_8
    n3 = evaluator.add_leaf(
        id="gpu_count_8",
        desc="Specifies exactly 8 NVIDIA H100 GPUs per system",
        parent=cat,
        critical=True,
    )
    claim = f"Each DGX H100 system includes exactly 8 NVIDIA H100 GPUs (extracted value: '{_nz(hw.gpu_count)}')."
    await evaluator.verify(
        claim=claim,
        node=n3,
        sources=hw.sources,
        additional_instruction="Confirm the cited source states 8 GPUs per DGX H100 system."
    )

    # gpu_tdp_700w
    n4 = evaluator.add_leaf(
        id="gpu_tdp_700w",
        desc="Specifies H100 GPU TDP as 700W",
        parent=cat,
        critical=True,
    )
    claim = f"The NVIDIA H100 GPU TDP is 700 W (extracted value: '{_nz(hw.gpu_tdp_watts)}')."
    await evaluator.verify(
        claim=claim,
        node=n4,
        sources=hw.sources,
        additional_instruction="Verify that an authoritative NVIDIA source indicates 700W TDP for H100 in DGX configurations."
    )

    # gpu_memory_80gb_hbm3
    n5 = evaluator.add_leaf(
        id="gpu_memory_80gb_hbm3",
        desc="Specifies H100 GPU memory as 80GB HBM3",
        parent=cat,
        critical=True,
    )
    claim = f"Each H100 GPU has 80 GB HBM3 memory (extracted value: '{_nz(hw.gpu_memory)}')."
    await evaluator.verify(
        claim=claim,
        node=n5,
        sources=hw.sources,
        additional_instruction="Confirm the H100 memory capacity per GPU is 80GB HBM3 on a cited NVIDIA document."
    )

    # gpu_memory_bandwidth_3tbs
    n6 = evaluator.add_leaf(
        id="gpu_memory_bandwidth_3tbs",
        desc="Specifies H100 GPU memory bandwidth as 3 TB/s",
        parent=cat,
        critical=True,
    )
    claim = f"The H100 GPU memory bandwidth is approximately 3 TB/s (extracted value: '{_nz(hw.gpu_memory_bandwidth)}')."
    await evaluator.verify(
        claim=claim,
        node=n6,
        sources=hw.sources,
        additional_instruction="Allow small rounding; confirm bandwidth ≈3 TB/s per NVIDIA documentation."
    )

    # total_gpu_power_draw_calculated
    n7 = evaluator.add_leaf(
        id="total_gpu_power_draw_calculated",
        desc="Calculates total GPU power draw per system as 5.6 kW (8 × 700W)",
        parent=cat,
        critical=True,
    )
    claim = (
        "Given 8 GPUs each at 700 W TDP, the total GPU power draw per system is 5.6 kW (8 × 700 W). "
        f"(extracted statement: '{_nz(hw.total_gpu_power_draw)}')"
    )
    await evaluator.verify(
        claim=claim,
        node=n7,
        sources=hw.sources,
        additional_instruction="Confirm that sources support 700W per H100 GPU and infer 5.6 kW by simple multiplication."
    )


async def build_power_nodes(evaluator: Evaluator, parent, pwr: PowerSupplyConfig):
    cat = evaluator.add_parallel(
        id="power_supply_configuration",
        desc="Power supply and electrical requirements are satisfied",
        parent=parent,
        critical=True,
    )

    n1 = evaluator.add_leaf(
        id="psu_count_6",
        desc="Specifies six power supplies per system",
        parent=cat,
        critical=True,
    )
    claim = f"The DGX H100 system uses six power supplies (extracted value: '{_nz(pwr.psu_count)}')."
    await evaluator.verify(claim=claim, node=n1, sources=pwr.sources)

    n2 = evaluator.add_leaf(
        id="psu_rating",
        desc="Specifies each PSU rating as 3300W @ 200–240V, 16A, 50–60Hz",
        parent=cat,
        critical=True,
    )
    claim = f"Each PSU is rated 3300 W at 200–240 V, 16 A, 50–60 Hz (extracted rating: '{_nz(pwr.psu_rating)}')."
    await evaluator.verify(claim=claim, node=n2, sources=pwr.sources)

    n3 = evaluator.add_leaf(
        id="min_operational_psus_4_of_6",
        desc="Specifies minimum operational requirement: at least 4 of 6 PSUs energized",
        parent=cat,
        critical=True,
    )
    claim = f"The system can operate with at least 4 of 6 PSUs energized (extracted: '{_nz(pwr.min_operational_psus)}')."
    await evaluator.verify(claim=claim, node=n3, sources=pwr.sources)

    n4 = evaluator.add_leaf(
        id="min_circuits_3",
        desc="Specifies minimum of 3 power circuits per system",
        parent=cat,
        critical=True,
    )
    claim = f"A minimum of 3 separate power circuits per system is required (extracted: '{_nz(pwr.min_circuits)}')."
    await evaluator.verify(claim=claim, node=n4, sources=pwr.sources)

    n5 = evaluator.add_leaf(
        id="max_system_power_10_2kw",
        desc="Specifies maximum system power consumption as 10.2 kW",
        parent=cat,
        critical=True,
    )
    claim = f"The maximum DGX H100 system power is 10.2 kW (extracted: '{_nz(pwr.max_system_power)}')."
    await evaluator.verify(claim=claim, node=n5, sources=pwr.sources)

    n6 = evaluator.add_leaf(
        id="preferred_power_feed_config",
        desc="Specifies preferred power feed configuration as 415 VAC, 32A, three-phase (excluding redundancy, which is checked under redundancy/reliability)",
        parent=cat,
        critical=True,
    )
    claim = f"The preferred power feed is 415 VAC, 32 A, three-phase (extracted: '{_nz(pwr.preferred_power_feed)}')."
    await evaluator.verify(claim=claim, node=n6, sources=pwr.sources)


async def build_cooling_nodes(evaluator: Evaluator, parent, clg: CoolingInfrastructure):
    cat = evaluator.add_parallel(
        id="cooling_infrastructure",
        desc="Cooling infrastructure requirements are satisfied",
        parent=parent,
        critical=True,
    )

    n1 = evaluator.add_leaf(
        id="airflow_per_gpu_300_ls",
        desc="Specifies minimum airflow as 300 L/s per GPU",
        parent=cat,
        critical=True,
    )
    claim = f"Minimum airflow is 300 L/s per GPU (extracted: '{_nz(clg.airflow_per_gpu)}')."
    await evaluator.verify(claim=claim, node=n1, sources=clg.sources)

    n2 = evaluator.add_leaf(
        id="airflow_total_2400_ls",
        desc="Computes total airflow as 2400 L/s for 8 GPUs (8 × 300 L/s)",
        parent=cat,
        critical=True,
    )
    claim = f"Total airflow for 8 GPUs is 2400 L/s (8 × 300 L/s) (extracted: '{_nz(clg.airflow_total)}')."
    await evaluator.verify(
        claim=claim, node=n2, sources=clg.sources,
        additional_instruction="Allow simple derivation from per-GPU airflow × 8 GPUs."
    )

    n3 = evaluator.add_leaf(
        id="hot_cold_aisle_containment",
        desc="Specifies hot/cold aisle containment architecture",
        parent=cat,
        critical=True,
    )
    claim = f"The specification calls for hot/cold aisle containment (extracted: '{_nz(clg.hot_cold_aisle_containment)}')."
    await evaluator.verify(claim=claim, node=n3, sources=clg.sources)

    n4 = evaluator.add_leaf(
        id="operating_temp_range_5_30c",
        desc="Specifies operating temperature range as 5–30°C (41–86°F)",
        parent=cat,
        critical=True,
    )
    claim = f"The operating temperature range is 5–30°C (41–86°F) (extracted: '{_nz(clg.operating_temp_range)}')."
    await evaluator.verify(claim=claim, node=n4, sources=clg.sources)


async def build_rack_nodes(evaluator: Evaluator, parent, rack: RackInfrastructure):
    cat = evaluator.add_parallel(
        id="rack_infrastructure",
        desc="Rack infrastructure requirements are satisfied",
        parent=parent,
        critical=True,
    )

    n1 = evaluator.add_leaf(
        id="rack_weight_capacity_350_lbs_per_u",
        desc="Specifies rack weight capacity of at least ~350 lbs per U",
        parent=cat,
        critical=True,
    )
    claim = f"The rack weight capacity is at least ~350 lbs per U (extracted: '{_nz(rack.rack_weight_capacity)}')."
    await evaluator.verify(claim=claim, node=n1, sources=rack.sources)

    n2 = evaluator.add_leaf(
        id="rack_power_density_60kw",
        desc="Specifies rack power density of at least 60 kW per rack",
        parent=cat,
        critical=True,
    )
    claim = f"Rack power density is at least 60 kW per rack (extracted: '{_nz(rack.rack_power_density)}')."
    await evaluator.verify(claim=claim, node=n2, sources=rack.sources)

    n3 = evaluator.add_leaf(
        id="traditional_rack_insufficient_explained",
        desc="Explains why traditional 5–10 kW racks are insufficient for this AI deployment",
        parent=cat,
        critical=True,
    )
    claim = (
        "The specification explains why traditional 5–10 kW racks are insufficient for this AI deployment "
        f"(extracted explanation: '{_nz(rack.traditional_rack_insufficient_explanation)}')."
    )
    await evaluator.verify(claim=claim, node=n3, sources=rack.sources)


async def build_memory_nodes(evaluator: Evaluator, parent, mem: MemorySpecifications):
    cat = evaluator.add_parallel(
        id="memory_specifications",
        desc="Memory specification calculations are satisfied",
        parent=parent,
        critical=True,
    )

    n1 = evaluator.add_leaf(
        id="system_memory_640gb",
        desc="Calculates total memory per system as 640GB (8 × 80GB)",
        parent=cat,
        critical=True,
    )
    claim = f"Total HBM memory per system is 640 GB (8 × 80 GB) (extracted: '{_nz(mem.system_memory_total)}')."
    await evaluator.verify(claim=claim, node=n1, sources=mem.sources)

    n2 = evaluator.add_leaf(
        id="cluster_memory_2_56tb",
        desc="Calculates total cluster memory as 2.56TB (4 × 640GB)",
        parent=cat,
        critical=True,
    )
    claim = f"Total cluster HBM memory (4 systems) is 2.56 TB (4 × 640 GB) (extracted: '{_nz(mem.cluster_memory_total)}')."
    await evaluator.verify(claim=claim, node=n2, sources=mem.sources)


async def build_network_nodes(evaluator: Evaluator, parent, net: NetworkInfrastructure):
    cat = evaluator.add_parallel(
        id="network_infrastructure",
        desc="Network infrastructure requirements are satisfied",
        parent=parent,
        critical=True,
    )

    n1 = evaluator.add_leaf(
        id="min_bandwidth_100gbps",
        desc="Specifies minimum network bandwidth as 100+ Gbps",
        parent=cat,
        critical=True,
    )
    claim = f"The minimum network bandwidth is at least 100 Gbps (extracted: '{_nz(net.min_bandwidth)}')."
    await evaluator.verify(claim=claim, node=n1, sources=net.sources)

    n2 = evaluator.add_leaf(
        id="optimal_bandwidth_800g_to_1_6t",
        desc="Specifies optimal inter-node bandwidth as 800 Gbps to 1.6 Tbps",
        parent=cat,
        critical=True,
    )
    claim = f"Optimal inter-node bandwidth for distributed training is between 800 Gbps and 1.6 Tbps (extracted: '{_nz(net.optimal_bandwidth_range)}')."
    await evaluator.verify(claim=claim, node=n2, sources=net.sources)


async def build_tier_nodes(evaluator: Evaluator, parent, tier: DataCenterTierRequirements):
    # NOTE: Set parent non-critical to allow an optional non-critical child under the category.
    cat = evaluator.add_parallel(
        id="data_center_tier_requirements",
        desc="Data center tier requirements are satisfied",
        parent=parent,
        critical=False,
    )

    n1 = evaluator.add_leaf(
        id="min_tier_iii",
        desc="Specifies minimum Tier III certification",
        parent=cat,
        critical=True,
    )
    claim = f"The minimum data center tier is Tier III (extracted: '{_nz(tier.min_tier)}')."
    await evaluator.verify(
        claim=claim,
        node=n1,
        sources=tier.sources,
        additional_instruction="Cross-check with Uptime Institute Tier III definition."
    )

    n2 = evaluator.add_leaf(
        id="tier_iii_concurrently_maintainable",
        desc="Defines Tier III concurrent maintainability requirement",
        parent=cat,
        critical=True,
    )
    claim = f"Tier III includes concurrent maintainability (extracted: '{_nz(tier.tier_concurrent_maintainability)}')."
    await evaluator.verify(claim=claim, node=n2, sources=tier.sources)

    n3 = evaluator.add_leaf(
        id="tier_iii_redundant_distribution_paths",
        desc="Defines Tier III redundant distribution path requirement",
        parent=cat,
        critical=True,
    )
    claim = f"Tier III requires redundant distribution paths (extracted: '{_nz(tier.tier_redundant_distribution_paths)}')."
    await evaluator.verify(claim=claim, node=n3, sources=tier.sources)

    n4 = evaluator.add_leaf(
        id="tier_iv_optional",
        desc="Optionally notes Tier IV fault tolerance features",
        parent=cat,
        critical=False,
    )
    claim = f"The specification optionally notes Tier IV fault tolerance features (extracted: '{_nz(tier.tier_iv_optional_note)}')."
    await evaluator.verify(claim=claim, node=n4, sources=tier.sources)


async def build_pue_nodes(evaluator: Evaluator, parent, pue: EnergyEfficiency):
    cat = evaluator.add_parallel(
        id="energy_efficiency",
        desc="Energy efficiency (PUE) requirements are satisfied",
        parent=parent,
        critical=True,
    )

    n1 = evaluator.add_leaf(
        id="target_pue_le_1_2",
        desc="Specifies target PUE as ≤ 1.2",
        parent=cat,
        critical=True,
    )
    claim = f"The target PUE is 1.2 or below (extracted: '{_nz(pue.target_pue)}')."
    await evaluator.verify(claim=claim, node=n1, sources=pue.sources)

    n2 = evaluator.add_leaf(
        id="acceptable_pue_1_2_to_1_8",
        desc="Specifies acceptable PUE range as 1.2–1.8",
        parent=cat,
        critical=True,
    )
    claim = f"The acceptable PUE range is 1.2–1.8 (extracted: '{_nz(pue.acceptable_pue_range)}')."
    await evaluator.verify(claim=claim, node=n2, sources=pue.sources)


async def build_deployment_nodes(evaluator: Evaluator, parent, dep: DeploymentScale):
    cat = evaluator.add_parallel(
        id="deployment_scale",
        desc="Deployment scale requirements and calculations are satisfied",
        parent=parent,
        critical=True,
    )

    n1 = evaluator.add_leaf(
        id="min_system_count_4",
        desc="Confirms minimum cluster size of 4 DGX H100 systems for multi-node training",
        parent=cat,
        critical=True,
    )
    claim = f"The minimum cluster size is 4 DGX H100 systems (extracted: '{_nz(dep.min_system_count)}')."
    await evaluator.verify(claim=claim, node=n1, sources=dep.sources)

    n2 = evaluator.add_leaf(
        id="total_it_power_40_8kw",
        desc="Calculates total IT power as 40.8 kW (4 × 10.2 kW)",
        parent=cat,
        critical=True,
    )
    claim = f"Total IT power is 40.8 kW (4 × 10.2 kW) (extracted: '{_nz(dep.total_it_power)}')."
    await evaluator.verify(
        claim=claim, node=n2, sources=dep.sources,
        additional_instruction="Support comes from per-system max power 10.2 kW multiplied by 4 systems."
    )

    n3 = evaluator.add_leaf(
        id="facility_power_estimate_with_pue",
        desc="Estimates facility power using PUE and total IT power",
        parent=cat,
        critical=True,
    )
    claim = (
        "Facility power is estimated by multiplying IT power by PUE (extracted statement: "
        f"'{_nz(dep.facility_power_estimate_with_pue)}')."
    )
    await evaluator.verify(
        claim=claim, node=n3, sources=dep.sources,
        additional_instruction="Confirm that the method uses Facility Power = PUE × IT Power, consistent with standard PUE definition."
    )

    n4 = evaluator.add_leaf(
        id="rack_space_32u",
        desc="Calculates rack space as 32U minimum (4 × 8U)",
        parent=cat,
        critical=True,
    )
    claim = f"Rack space required is at least 32U (4 × 8U) (extracted: '{_nz(dep.rack_space_u)}')."
    await evaluator.verify(claim=claim, node=n4, sources=dep.sources)


async def build_environmental_nodes(evaluator: Evaluator, parent, envr: EnvironmentalRequirements):
    cat = evaluator.add_parallel(
        id="environmental_requirements",
        desc="Environmental requirements are satisfied",
        parent=parent,
        critical=True,
    )

    # Although the rubric focuses humidity here, we also verify temperature range if present as context
    n1 = evaluator.add_leaf(
        id="humidity_control_measures",
        desc="Describes humidity control measures",
        parent=cat,
        critical=True,
    )
    claim = f"The specification describes humidity control measures (extracted: '{_nz(envr.humidity_control_measures)}')."
    await evaluator.verify(
        claim=claim, node=n1, sources=envr.sources,
        additional_instruction="Look for humidity setpoints, ranges, control methods, or standards references."
    )


async def build_physical_space_nodes(evaluator: Evaluator, parent, phys: PhysicalSpaceRequirements):
    cat = evaluator.add_parallel(
        id="physical_space_requirements",
        desc="Physical space requirements are stated",
        parent=parent,
        critical=True,
    )

    n1 = evaluator.add_leaf(
        id="rack_count_based_on_power_density",
        desc="Calculates number of racks needed based on system count and rack power density",
        parent=cat,
        critical=True,
    )
    claim = (
        "The number of racks is calculated based on system count and assumed rack power density (e.g., ≥60 kW/rack) "
        f"(extracted rack count estimate: '{_nz(phys.rack_count_estimate)}')."
    )
    await evaluator.verify(claim=claim, node=n1, sources=phys.sources)

    n2 = evaluator.add_leaf(
        id="floor_space_estimate",
        desc="Estimates floor space requirement",
        parent=cat,
        critical=True,
    )
    claim = f"The specification provides a floor space estimate (extracted: '{_nz(phys.floor_space_estimate)}')."
    await evaluator.verify(claim=claim, node=n2, sources=phys.sources)


async def build_redundancy_nodes(evaluator: Evaluator, parent, red: RedundancyReliability):
    cat = evaluator.add_parallel(
        id="redundancy_and_reliability",
        desc="Redundancy and reliability requirements are specified",
        parent=parent,
        critical=True,
    )

    n1 = evaluator.add_leaf(
        id="power_redundancy_n_plus_1_or_better",
        desc="Specifies power redundancy as N+1 or better",
        parent=cat,
        critical=True,
    )
    claim = f"Power redundancy is N+1 or better (extracted: '{_nz(red.power_redundancy)}')."
    await evaluator.verify(claim=claim, node=n1, sources=red.sources)

    n2 = evaluator.add_leaf(
        id="cooling_redundancy_specified",
        desc="Specifies cooling redundancy configuration",
        parent=cat,
        critical=True,
    )
    claim = f"The specification defines cooling redundancy (extracted: '{_nz(red.cooling_redundancy)}')."
    await evaluator.verify(claim=claim, node=n2, sources=red.sources)

    n3 = evaluator.add_leaf(
        id="network_redundancy_specified",
        desc="Specifies network redundancy configuration",
        parent=cat,
        critical=True,
    )
    claim = f"The specification defines network redundancy (extracted: '{_nz(red.network_redundancy)}')."
    await evaluator.verify(claim=claim, node=n3, sources=red.sources)


async def build_location_nodes(evaluator: Evaluator, parent, loc: GeographicLocation):
    cat = evaluator.add_parallel(
        id="geographic_location",
        desc="Geographic location requirements are satisfied",
        parent=parent,
        critical=True,
    )

    n1 = evaluator.add_leaf(
        id="major_us_market_identified",
        desc="Identifies a specific major US data center market",
        parent=cat,
        critical=True,
    )
    claim = f"A specific major US data center market is identified (extracted: '{_nz(loc.market_name)}')."
    await evaluator.verify(
        claim=claim, node=n1, sources=loc.sources,
        additional_instruction="Look for well-known markets such as Northern Virginia, Phoenix, Dallas, Chicago, etc."
    )

    n2 = evaluator.add_leaf(
        id="mw_scale_power_capacity_supported",
        desc="States that the chosen market supports MW-scale power delivery capability",
        parent=cat,
        critical=True,
    )
    claim = (
        "The chosen market supports MW-scale power delivery capability and supporting infrastructure "
        f"(extracted: '{_nz(loc.supports_mw_scale_power)}')."
    )
    await evaluator.verify(claim=claim, node=n2, sources=loc.sources)


async def build_references_nodes(evaluator: Evaluator, parent, all_urls: List[str]):
    cat = evaluator.add_parallel(
        id="authoritative_url_references",
        desc="Authoritative URL references are provided to support the stated technical requirements",
        parent=parent,
        critical=True,
    )

    n1 = evaluator.add_leaf(
        id="urls_support_requirements",
        desc="Provides supporting URL reference(s) from authoritative sources (e.g., manufacturer datasheets, industry standards, or data center provider specifications) for the technical specifications",
        parent=cat,
        critical=True,
    )
    claim = (
        "The provided URLs are authoritative (e.g., manufacturer datasheets, industry standards, or provider specs) "
        "and they support the technical requirements stated in the specification."
    )
    await evaluator.verify(
        claim=claim,
        node=n1,
        sources=all_urls,
        additional_instruction="Accept pass if at least some URLs are clearly authoritative and substantively support the specification claims."
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
    Evaluate an answer for the DGX H100 4-system cluster specification task.
    """
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

    # Run all extractions in parallel
    (
        hw,
        pwr,
        clg,
        rack,
        mem,
        net,
        tier,
        pue,
        dep,
        envr,
        phys,
        red,
        loc,
        refs,
    ) = await asyncio.gather(
        evaluator.extract(prompt=prompt_extract_hardware(), template_class=HardwareSelection, extraction_name="hardware_selection"),
        evaluator.extract(prompt=prompt_extract_power(), template_class=PowerSupplyConfig, extraction_name="power_supply"),
        evaluator.extract(prompt=prompt_extract_cooling(), template_class=CoolingInfrastructure, extraction_name="cooling_infrastructure"),
        evaluator.extract(prompt=prompt_extract_rack(), template_class=RackInfrastructure, extraction_name="rack_infrastructure"),
        evaluator.extract(prompt=prompt_extract_memory(), template_class=MemorySpecifications, extraction_name="memory_specifications"),
        evaluator.extract(prompt=prompt_extract_network(), template_class=NetworkInfrastructure, extraction_name="network_infrastructure"),
        evaluator.extract(prompt=prompt_extract_tier(), template_class=DataCenterTierRequirements, extraction_name="dc_tier_requirements"),
        evaluator.extract(prompt=prompt_extract_pue(), template_class=EnergyEfficiency, extraction_name="energy_efficiency"),
        evaluator.extract(prompt=prompt_extract_deployment_scale(), template_class=DeploymentScale, extraction_name="deployment_scale"),
        evaluator.extract(prompt=prompt_extract_environmental(), template_class=EnvironmentalRequirements, extraction_name="environmental_requirements"),
        evaluator.extract(prompt=prompt_extract_physical_space(), template_class=PhysicalSpaceRequirements, extraction_name="physical_space_requirements"),
        evaluator.extract(prompt=prompt_extract_redundancy(), template_class=RedundancyReliability, extraction_name="redundancy_reliability"),
        evaluator.extract(prompt=prompt_extract_location(), template_class=GeographicLocation, extraction_name="geographic_location"),
        evaluator.extract(prompt=prompt_extract_references(), template_class=ReferencesExtraction, extraction_name="all_references"),
    )

    # Aggregate all URLs (deduplicated) for the final "authoritative references" check
    all_urls = _dedup_urls(
        getattr(hw, "sources", []),
        getattr(pwr, "sources", []),
        getattr(clg, "sources", []),
        getattr(rack, "sources", []),
        getattr(mem, "sources", []),
        getattr(net, "sources", []),
        getattr(tier, "sources", []),
        getattr(pue, "sources", []),
        getattr(dep, "sources", []),
        getattr(envr, "sources", []),
        getattr(phys, "sources", []),
        getattr(red, "sources", []),
        getattr(loc, "sources", []),
        getattr(refs, "urls", []),
    )

    # Record some custom info
    evaluator.add_custom_info(
        info={
            "total_supporting_urls": len(all_urls),
            "example_urls": all_urls[:5]
        },
        info_type="urls_summary",
        info_name="urls_overview"
    )

    # Build verification subtrees for each rubric category
    await build_hardware_nodes(evaluator, root, hw)
    await build_power_nodes(evaluator, root, pwr)
    await build_cooling_nodes(evaluator, root, clg)
    await build_rack_nodes(evaluator, root, rack)
    await build_memory_nodes(evaluator, root, mem)
    await build_network_nodes(evaluator, root, net)
    await build_tier_nodes(evaluator, root, tier)  # parent set to non-critical to allow optional child
    await build_pue_nodes(evaluator, root, pue)
    await build_deployment_nodes(evaluator, root, dep)
    await build_environmental_nodes(evaluator, root, envr)
    await build_physical_space_nodes(evaluator, root, phys)
    await build_redundancy_nodes(evaluator, root, red)
    await build_location_nodes(evaluator, root, loc)
    await build_references_nodes(evaluator, root, all_urls)

    # Return the final summary
    return evaluator.get_summary()