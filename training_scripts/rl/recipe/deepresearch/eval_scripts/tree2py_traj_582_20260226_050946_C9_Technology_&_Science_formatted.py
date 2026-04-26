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
TASK_ID = "advanced_hw_solution_multi_paradigm"
TASK_DESCRIPTION = """
A leading artificial intelligence research institute in California is planning to establish a next-generation heterogeneous computing testbed facility to evaluate emerging computing paradigms for AI workload optimization. The facility director has tasked you with identifying one hardware component from each of five distinct computing technology domains that meet specific technical requirements.

For each of the five computing paradigms listed below, identify a specific commercially available or announced hardware component (including manufacturer name and product model) that satisfies all stated technical constraints:

1. Neuromorphic Computing Processor:
- Must consume less than 5 watts of power under typical operating conditions
- Must support at least 500,000 neurons per chip
- Must be from a manufacturer with publicly announced commercial availability or development partnership programs as of February 2026

2. Photonic Computing Interconnect:
- Must provide at least 50 Tbps of total optical bandwidth
- Must demonstrate at least 10x power efficiency improvement compared to traditional 28nm CMOS implementations
- Must be from a manufacturer with publicly announced commercial availability or development partnership programs as of February 2026

3. Chiplet-Based Processor:
- Must utilize the UCIe (Universal Chiplet Interconnect Express) standard for die-to-die communication
- Must support chiplet interconnect data rates of at least 32 GT/s per pin
- Must be from a manufacturer with publicly announced commercial availability or development partnership programs as of February 2026

4. Quantum Computing Processor:
- Must have at least 100 qubits
- Must demonstrate below-threshold error correction capabilities
- Must be from a manufacturer with publicly announced commercial availability or development partnership programs as of February 2026

5. Edge AI Accelerator:
- Must deliver at least 20 TOPS (tera-operations per second) of AI performance
- Must operate with a thermal design power (TDP) of 10 watts or less
- Must be from a manufacturer with publicly announced commercial availability or development partnership programs as of February 2026

For each component, provide:
- The manufacturer name and specific product model/name
- Verification that all technical specifications are met
- Reference URL(s) supporting your identification and specifications
"""


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class ComponentBase(BaseModel):
    manufacturer: Optional[str] = None
    product_model: Optional[str] = None
    manufacturer_urls: List[str] = Field(default_factory=list)
    status_urls: List[str] = Field(default_factory=list)


class NeuromorphicComponent(ComponentBase):
    power_urls: List[str] = Field(default_factory=list)
    neuron_capacity_urls: List[str] = Field(default_factory=list)


class PhotonicComponent(ComponentBase):
    bandwidth_urls: List[str] = Field(default_factory=list)
    efficiency_urls: List[str] = Field(default_factory=list)


class ChipletComponent(ComponentBase):
    ucie_urls: List[str] = Field(default_factory=list)
    data_rate_urls: List[str] = Field(default_factory=list)


class QuantumComponent(ComponentBase):
    qubit_urls: List[str] = Field(default_factory=list)
    error_correction_urls: List[str] = Field(default_factory=list)


class EdgeAIComponent(ComponentBase):
    tops_urls: List[str] = Field(default_factory=list)
    tdp_urls: List[str] = Field(default_factory=list)


class HardwareSolutionExtraction(BaseModel):
    neuromorphic: Optional[NeuromorphicComponent] = None
    photonic: Optional[PhotonicComponent] = None
    chiplet: Optional[ChipletComponent] = None
    quantum: Optional[QuantumComponent] = None
    edge_ai: Optional[EdgeAIComponent] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_hardware_solution() -> str:
    return """
    Extract exactly one component per paradigm from the answer text. If multiple components are mentioned for a paradigm, select the first one mentioned. For each paradigm, extract:
    Common fields:
    - manufacturer: The company/manufacturer name.
    - product_model: The specific product model/name.
    - manufacturer_urls: All URLs explicitly provided in the answer that identify or describe the product and manufacturer (e.g., product page, datasheet).
    - status_urls: All URLs explicitly provided that indicate commercial availability or development partnership programs (as of February 2026).

    Neuromorphic:
    - power_urls: URLs supporting typical power consumption < 5W.
    - neuron_capacity_urls: URLs supporting ≥ 500,000 neurons per chip.

    Photonic:
    - bandwidth_urls: URLs supporting ≥ 50 Tbps total optical bandwidth.
    - efficiency_urls: URLs supporting ≥ 10x power efficiency vs traditional 28nm CMOS.

    Chiplet:
    - ucie_urls: URLs confirming UCIe standard for die-to-die communication.
    - data_rate_urls: URLs supporting ≥ 32 GT/s per pin data rate.

    Quantum:
    - qubit_urls: URLs supporting ≥ 100 qubits.
    - error_correction_urls: URLs supporting below-threshold error correction capabilities.

    Edge AI:
    - tops_urls: URLs supporting ≥ 20 TOPS performance.
    - tdp_urls: URLs supporting TDP ≤ 10W.

    IMPORTANT:
    - Only extract URLs explicitly present in the answer text. Include full URLs.
    - If any requested data is missing, set it to null (for strings) or an empty list (for URLs).
    - Do not invent any information.
    - Return a single JSON object with keys: neuromorphic, photonic, chiplet, quantum, edge_ai, each following the specified schema.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _non_empty_str(s: Optional[str]) -> bool:
    return bool(s) and bool(str(s).strip())


def _has_urls(urls: Optional[List[str]]) -> bool:
    return bool(urls) and len(urls) > 0


def _component_label(comp: ComponentBase, fallback: str) -> str:
    m = comp.manufacturer if comp else None
    p = comp.product_model if comp else None
    if _non_empty_str(m) and _non_empty_str(p):
        return f"{m} {p}"
    if _non_empty_str(m):
        return str(m)
    if _non_empty_str(p):
        return str(p)
    return fallback


# --------------------------------------------------------------------------- #
# Domain verification builders                                                #
# --------------------------------------------------------------------------- #
async def build_neuromorphic_checks(evaluator: Evaluator, parent_node, comp: Optional[NeuromorphicComponent]) -> None:
    node = evaluator.add_sequential(
        id="Neuromorphic_Computing_Component",
        desc="Evaluation of the neuromorphic computing processor selection and specifications",
        parent=parent_node,
        critical=False
    )

    comp_obj = comp or NeuromorphicComponent()

    # Identification
    ident = evaluator.add_parallel(
        id="Neuromorphic_Component_Identification",
        desc="Identification and validation of a specific neuromorphic chip manufacturer and product",
        parent=node,
        critical=True
    )

    # Manufacturer provided
    evaluator.add_custom_node(
        result=_non_empty_str(comp_obj.manufacturer) and _non_empty_str(comp_obj.product_model),
        id="Neuromorphic_Manufacturer",
        desc="A specific neuromorphic chip manufacturer and product name are provided",
        parent=ident,
        critical=True
    )

    # Manufacturer URL exists
    url_exist_man = evaluator.add_custom_node(
        result=_has_urls(comp_obj.manufacturer_urls),
        id="Neuromorphic_Manufacturer_URL",
        desc="Valid URL reference supporting the manufacturer and product identification",
        parent=ident,
        critical=True
    )

    # Verify identification against manufacturer URLs
    manu_verify = evaluator.add_leaf(
        id="Neuromorphic_Manufacturer_Verify",
        desc="Manufacturer and product identification is supported by cited manufacturer/product URLs",
        parent=ident,
        critical=True
    )
    manu_claim = f"The selected neuromorphic processor is {_component_label(comp_obj, 'the specified product')} (manufacturer and product identification are correct)."
    await evaluator.verify(
        claim=manu_claim,
        node=manu_verify,
        sources=comp_obj.manufacturer_urls,
        additional_instruction="Confirm the page(s) clearly identify the exact product model and manufacturer of a neuromorphic computing processor."
    )

    # Commercial status URL exists
    status_url_exist = evaluator.add_custom_node(
        result=_has_urls(comp_obj.status_urls),
        id="Neuromorphic_Status_URL",
        desc="URL reference confirming commercial availability status",
        parent=ident,
        critical=True
    )

    # Verify commercial status
    status_leaf = evaluator.add_leaf(
        id="Neuromorphic_Commercial_Status",
        desc="The manufacturer has announced commercial availability or development partnership programs as of February 2026",
        parent=ident,
        critical=True
    )
    status_claim = f"As of February 2026, the manufacturer {_component_label(comp_obj, 'the manufacturer')} has publicly announced commercial availability or a development partnership program relevant to this neuromorphic product."
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=comp_obj.status_urls,
        additional_instruction="Check whether the provided page(s) indicate commercial availability or formal partnership programs; accept reasonable equivalents such as 'available now', 'developer program', 'commercial program'."
    )

    # Technical specifications
    specs = evaluator.add_parallel(
        id="Neuromorphic_Technical_Specifications",
        desc="Verification of neuromorphic chip technical specifications against requirements",
        parent=node,
        critical=True
    )

    # Power consumption < 5W
    evaluator.add_custom_node(
        result=_has_urls(comp_obj.power_urls),
        id="Neuromorphic_Power_URL",
        desc="URL reference confirming power consumption specification",
        parent=specs,
        critical=True
    )
    power_leaf = evaluator.add_leaf(
        id="Neuromorphic_Power_Consumption",
        desc="The neuromorphic processor consumes less than 5 watts under typical operating conditions",
        parent=specs,
        critical=True
    )
    power_claim = f"The neuromorphic processor {_component_label(comp_obj, 'the product')} consumes less than 5 watts under typical operating conditions."
    await evaluator.verify(
        claim=power_claim,
        node=power_leaf,
        sources=comp_obj.power_urls,
        additional_instruction="Verify typical/nominal operating power; accept phrasing like '<5W', '≈4W', or typical values under 5W."
    )

    # Neuron capacity >= 500,000 per chip
    evaluator.add_custom_node(
        result=_has_urls(comp_obj.neuron_capacity_urls),
        id="Neuromorphic_Capacity_URL",
        desc="URL reference confirming neuron capacity specification",
        parent=specs,
        critical=True
    )
    neurons_leaf = evaluator.add_leaf(
        id="Neuromorphic_Neuron_Capacity",
        desc="The neuromorphic chip supports at least 500,000 neurons per chip",
        parent=specs,
        critical=True
    )
    neurons_claim = f"The neuromorphic chip {_component_label(comp_obj, 'the product')} supports at least 500,000 neurons per chip."
    await evaluator.verify(
        claim=neurons_claim,
        node=neurons_leaf,
        sources=comp_obj.neuron_capacity_urls,
        additional_instruction="Confirm neuron capacity per chip ≥ 500,000; allow reasonable formatting or unit equivalents."
    )


async def build_photonic_checks(evaluator: Evaluator, parent_node, comp: Optional[PhotonicComponent]) -> None:
    node = evaluator.add_sequential(
        id="Photonic_Computing_Component",
        desc="Evaluation of the photonic computing interconnect solution and specifications",
        parent=parent_node,
        critical=False
    )

    comp_obj = comp or PhotonicComponent()

    ident = evaluator.add_parallel(
        id="Photonic_Component_Identification",
        desc="Identification and validation of a specific photonic chip or interconnect manufacturer and product",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(comp_obj.manufacturer) and _non_empty_str(comp_obj.product_model),
        id="Photonic_Manufacturer",
        desc="A specific photonic chip or interconnect manufacturer and product name are provided",
        parent=ident,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.manufacturer_urls),
        id="Photonic_Manufacturer_URL",
        desc="Valid URL reference supporting the manufacturer and product identification",
        parent=ident,
        critical=True
    )

    manu_verify = evaluator.add_leaf(
        id="Photonic_Manufacturer_Verify",
        desc="Manufacturer and product identification is supported by cited manufacturer/product URLs",
        parent=ident,
        critical=True
    )
    manu_claim = f"The selected photonic computing interconnect is {_component_label(comp_obj, 'the specified product')} (manufacturer and product identification are correct)."
    await evaluator.verify(
        claim=manu_claim,
        node=manu_verify,
        sources=comp_obj.manufacturer_urls,
        additional_instruction="Confirm the page(s) clearly identify the exact product model and manufacturer of a photonic interconnect."
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.status_urls),
        id="Photonic_Status_URL",
        desc="URL reference confirming commercial availability status",
        parent=ident,
        critical=True
    )

    status_leaf = evaluator.add_leaf(
        id="Photonic_Commercial_Status",
        desc="The manufacturer has announced commercial availability or development partnership programs as of February 2026",
        parent=ident,
        critical=True
    )
    status_claim = f"As of February 2026, the manufacturer {_component_label(comp_obj, 'the manufacturer')} has publicly announced commercial availability or a development partnership program relevant to this photonic product."
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=comp_obj.status_urls,
        additional_instruction="Check whether the provided page(s) indicate commercial availability or formal partnership programs."
    )

    specs = evaluator.add_parallel(
        id="Photonic_Technical_Specifications",
        desc="Verification of photonic solution technical specifications against requirements",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.bandwidth_urls),
        id="Photonic_Bandwidth_URL",
        desc="URL reference confirming bandwidth specification",
        parent=specs,
        critical=True
    )
    bw_leaf = evaluator.add_leaf(
        id="Photonic_Bandwidth",
        desc="The photonic interconnect provides at least 50 Tbps of total optical bandwidth",
        parent=specs,
        critical=True
    )
    bw_claim = f"The photonic interconnect {_component_label(comp_obj, 'the product')} provides at least 50 Tbps of total optical bandwidth."
    await evaluator.verify(
        claim=bw_claim,
        node=bw_leaf,
        sources=comp_obj.bandwidth_urls,
        additional_instruction="Confirm aggregate/total optical bandwidth ≥ 50 Tbps."
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.efficiency_urls),
        id="Photonic_Efficiency_URL",
        desc="URL reference confirming power efficiency specification",
        parent=specs,
        critical=True
    )
    eff_leaf = evaluator.add_leaf(
        id="Photonic_Power_Efficiency",
        desc="The photonic solution demonstrates at least 10x power efficiency improvement compared to traditional 28nm CMOS implementations",
        parent=specs,
        critical=True
    )
    eff_claim = f"The photonic solution {_component_label(comp_obj, 'the product')} demonstrates at least a 10x power efficiency improvement compared to traditional 28nm CMOS implementations."
    await evaluator.verify(
        claim=eff_claim,
        node=eff_leaf,
        sources=comp_obj.efficiency_urls,
        additional_instruction="Look for explicit statements of ≥10x efficiency improvement relative to 28nm CMOS."
    )


async def build_chiplet_checks(evaluator: Evaluator, parent_node, comp: Optional[ChipletComponent]) -> None:
    node = evaluator.add_sequential(
        id="Chiplet_Architecture_Component",
        desc="Evaluation of the chiplet-based processor selection and specifications",
        parent=parent_node,
        critical=False
    )

    comp_obj = comp or ChipletComponent()

    ident = evaluator.add_parallel(
        id="Chiplet_Component_Identification",
        desc="Identification and validation of a specific chiplet-based processor manufacturer and product",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(comp_obj.manufacturer) and _non_empty_str(comp_obj.product_model),
        id="Chiplet_Manufacturer",
        desc="A specific chiplet-based processor manufacturer and product name are provided",
        parent=ident,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.manufacturer_urls),
        id="Chiplet_Manufacturer_URL",
        desc="Valid URL reference supporting the manufacturer and product identification",
        parent=ident,
        critical=True
    )

    manu_verify = evaluator.add_leaf(
        id="Chiplet_Manufacturer_Verify",
        desc="Manufacturer and product identification is supported by cited manufacturer/product URLs",
        parent=ident,
        critical=True
    )
    manu_claim = f"The selected chiplet-based processor is {_component_label(comp_obj, 'the specified product')} (manufacturer and product identification are correct)."
    await evaluator.verify(
        claim=manu_claim,
        node=manu_verify,
        sources=comp_obj.manufacturer_urls,
        additional_instruction="Confirm the page(s) clearly identify the exact product model and manufacturer of a chiplet-based processor."
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.status_urls),
        id="Chiplet_Status_URL",
        desc="URL reference confirming commercial availability status",
        parent=ident,
        critical=True
    )

    status_leaf = evaluator.add_leaf(
        id="Chiplet_Commercial_Status",
        desc="The manufacturer has announced commercial availability or development partnership programs as of February 2026",
        parent=ident,
        critical=True
    )
    status_claim = f"As of February 2026, the manufacturer {_component_label(comp_obj, 'the manufacturer')} has publicly announced commercial availability or a development partnership program relevant to this chiplet product."
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=comp_obj.status_urls,
        additional_instruction="Check whether the page(s) indicate commercial availability or formal partnership programs."
    )

    specs = evaluator.add_parallel(
        id="Chiplet_Technical_Specifications",
        desc="Verification of chiplet processor technical specifications against requirements",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.ucie_urls),
        id="Chiplet_Standard_URL",
        desc="URL reference confirming UCIe standard compliance",
        parent=specs,
        critical=True
    )
    ucie_leaf = evaluator.add_leaf(
        id="Chiplet_Interconnect_Standard",
        desc="The chiplet architecture utilizes the UCIe (Universal Chiplet Interconnect Express) standard for die-to-die communication",
        parent=specs,
        critical=True
    )
    ucie_claim = f"The chiplet-based processor {_component_label(comp_obj, 'the product')} utilizes the UCIe (Universal Chiplet Interconnect Express) standard for die-to-die communication."
    await evaluator.verify(
        claim=ucie_claim,
        node=ucie_leaf,
        sources=comp_obj.ucie_urls,
        additional_instruction="Confirm explicit mention of UCIe standard for die-to-die communication."
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.data_rate_urls),
        id="Chiplet_Rate_URL",
        desc="URL reference confirming data rate specification",
        parent=specs,
        critical=True
    )
    rate_leaf = evaluator.add_leaf(
        id="Chiplet_Data_Rate",
        desc="The chiplet interconnect supports data rates of at least 32 GT/s per pin",
        parent=specs,
        critical=True
    )
    rate_claim = f"The chiplet interconnect for {_component_label(comp_obj, 'the product')} supports data rates of at least 32 GT/s per pin."
    await evaluator.verify(
        claim=rate_claim,
        node=rate_leaf,
        sources=comp_obj.data_rate_urls,
        additional_instruction="Confirm per-pin data rate ≥ 32 GT/s; accept equivalent phrasing such as 32 Giga-transfers per second."
    )


async def build_quantum_checks(evaluator: Evaluator, parent_node, comp: Optional[QuantumComponent]) -> None:
    node = evaluator.add_sequential(
        id="Quantum_Computing_Component",
        desc="Evaluation of the quantum processor selection and specifications",
        parent=parent_node,
        critical=False
    )

    comp_obj = comp or QuantumComponent()

    ident = evaluator.add_parallel(
        id="Quantum_Component_Identification",
        desc="Identification and validation of a specific quantum processor manufacturer and system",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(comp_obj.manufacturer) and _non_empty_str(comp_obj.product_model),
        id="Quantum_Manufacturer",
        desc="A specific quantum processor manufacturer and system name are provided",
        parent=ident,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.manufacturer_urls),
        id="Quantum_Manufacturer_URL",
        desc="Valid URL reference supporting the manufacturer and system identification",
        parent=ident,
        critical=True
    )

    manu_verify = evaluator.add_leaf(
        id="Quantum_Manufacturer_Verify",
        desc="Manufacturer and system identification is supported by cited manufacturer/product URLs",
        parent=ident,
        critical=True
    )
    manu_claim = f"The selected quantum computing processor/system is {_component_label(comp_obj, 'the specified system')} (manufacturer and system identification are correct)."
    await evaluator.verify(
        claim=manu_claim,
        node=manu_verify,
        sources=comp_obj.manufacturer_urls,
        additional_instruction="Confirm the page(s) clearly identify the exact system and manufacturer for a quantum processor."
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.status_urls),
        id="Quantum_Status_URL",
        desc="URL reference confirming commercial availability status",
        parent=ident,
        critical=True
    )

    status_leaf = evaluator.add_leaf(
        id="Quantum_Commercial_Status",
        desc="The manufacturer has announced commercial availability or development partnership programs as of February 2026",
        parent=ident,
        critical=True
    )
    status_claim = f"As of February 2026, the manufacturer {_component_label(comp_obj, 'the manufacturer')} has publicly announced commercial availability or a development partnership program relevant to this quantum processor/system."
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=comp_obj.status_urls,
        additional_instruction="Check for clear statements indicating availability or partnership programs."
    )

    specs = evaluator.add_parallel(
        id="Quantum_Technical_Specifications",
        desc="Verification of quantum processor technical specifications against requirements",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.qubit_urls),
        id="Quantum_Qubit_URL",
        desc="URL reference confirming qubit count specification",
        parent=specs,
        critical=True
    )
    qubit_leaf = evaluator.add_leaf(
        id="Quantum_Qubit_Count",
        desc="The quantum processor has at least 100 qubits",
        parent=specs,
        critical=True
    )
    qubit_claim = f"The quantum processor/system {_component_label(comp_obj, 'the system')} has at least 100 qubits."
    await evaluator.verify(
        claim=qubit_claim,
        node=qubit_leaf,
        sources=comp_obj.qubit_urls,
        additional_instruction="Confirm that the qubit count is ≥ 100."
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.error_correction_urls),
        id="Quantum_Error_URL",
        desc="URL reference confirming error correction capabilities",
        parent=specs,
        critical=True
    )
    ec_leaf = evaluator.add_leaf(
        id="Quantum_Error_Correction",
        desc="The quantum system demonstrates below-threshold error correction capabilities",
        parent=specs,
        critical=True
    )
    ec_claim = f"The quantum system {_component_label(comp_obj, 'the system')} demonstrates below-threshold error correction capabilities (e.g., error rates below fault-tolerance threshold or experimental demonstrations meeting threshold criteria)."
    await evaluator.verify(
        claim=ec_claim,
        node=ec_leaf,
        sources=comp_obj.error_correction_urls,
        additional_instruction="Look for evidence of below-threshold error correction: e.g., logical error rates below threshold, fault-tolerant demonstrations, or explicit mentions of meeting/being below threshold."
    )


async def build_edge_ai_checks(evaluator: Evaluator, parent_node, comp: Optional[EdgeAIComponent]) -> None:
    node = evaluator.add_sequential(
        id="Edge_AI_Component",
        desc="Evaluation of the edge AI accelerator selection and specifications",
        parent=parent_node,
        critical=False
    )

    comp_obj = comp or EdgeAIComponent()

    ident = evaluator.add_parallel(
        id="Edge_AI_Component_Identification",
        desc="Identification and validation of a specific edge AI accelerator manufacturer and product",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_non_empty_str(comp_obj.manufacturer) and _non_empty_str(comp_obj.product_model),
        id="Edge_AI_Manufacturer",
        desc="A specific edge AI accelerator manufacturer and product name are provided",
        parent=ident,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.manufacturer_urls),
        id="Edge_AI_Manufacturer_URL",
        desc="Valid URL reference supporting the manufacturer and product identification",
        parent=ident,
        critical=True
    )

    manu_verify = evaluator.add_leaf(
        id="Edge_AI_Manufacturer_Verify",
        desc="Manufacturer and product identification is supported by cited manufacturer/product URLs",
        parent=ident,
        critical=True
    )
    manu_claim = f"The selected edge AI accelerator is {_component_label(comp_obj, 'the specified product')} (manufacturer and product identification are correct)."
    await evaluator.verify(
        claim=manu_claim,
        node=manu_verify,
        sources=comp_obj.manufacturer_urls,
        additional_instruction="Confirm the page(s) clearly identify the exact product model and manufacturer for an edge AI accelerator."
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.status_urls),
        id="Edge_AI_Status_URL",
        desc="URL reference confirming commercial availability status",
        parent=ident,
        critical=True
    )

    status_leaf = evaluator.add_leaf(
        id="Edge_AI_Commercial_Status",
        desc="The manufacturer has announced commercial availability or development partnership programs as of February 2026",
        parent=ident,
        critical=True
    )
    status_claim = f"As of February 2026, the manufacturer {_component_label(comp_obj, 'the manufacturer')} has publicly announced commercial availability or a development partnership program relevant to this edge AI accelerator."
    await evaluator.verify(
        claim=status_claim,
        node=status_leaf,
        sources=comp_obj.status_urls,
        additional_instruction="Check for clear statements indicating availability or partnership programs."
    )

    specs = evaluator.add_parallel(
        id="Edge_AI_Technical_Specifications",
        desc="Verification of edge AI accelerator technical specifications against requirements",
        parent=node,
        critical=True
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.tops_urls),
        id="Edge_AI_Performance_URL",
        desc="URL reference confirming TOPS performance specification",
        parent=specs,
        critical=True
    )
    perf_leaf = evaluator.add_leaf(
        id="Edge_AI_Performance",
        desc="The edge AI accelerator delivers at least 20 TOPS (tera-operations per second)",
        parent=specs,
        critical=True
    )
    perf_claim = f"The edge AI accelerator {_component_label(comp_obj, 'the product')} delivers at least 20 TOPS (tera-operations per second)."
    await evaluator.verify(
        claim=perf_claim,
        node=perf_leaf,
        sources=comp_obj.tops_urls,
        additional_instruction="Confirm AI performance metric (TOPS) ≥ 20."
    )

    evaluator.add_custom_node(
        result=_has_urls(comp_obj.tdp_urls),
        id="Edge_AI_Power_URL",
        desc="URL reference confirming power specification",
        parent=specs,
        critical=True
    )
    power_leaf = evaluator.add_leaf(
        id="Edge_AI_Power",
        desc="The edge AI processor operates with a thermal design power (TDP) of 10 watts or less",
        parent=specs,
        critical=True
    )
    power_claim = f"The edge AI accelerator {_component_label(comp_obj, 'the product')} operates with a thermal design power (TDP) of 10 watts or less."
    await evaluator.verify(
        claim=power_claim,
        node=power_leaf,
        sources=comp_obj.tdp_urls,
        additional_instruction="Confirm TDP (or typical power) ≤ 10W; accept equivalent phrasing indicating TDP under or equal to 10W."
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
    Evaluate an answer for the multi-paradigm advanced hardware solution task.
    """
    # Initialize evaluator
    evaluator = Evaluator()
    root = evaluator.initialize(
        task_id=TASK_ID,
        strategy=AggregationStrategy.PARALLEL,  # Parallel across domains
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

    # Extract the structured hardware solution
    extraction = await evaluator.extract(
        prompt=prompt_extract_hardware_solution(),
        template_class=HardwareSolutionExtraction,
        extraction_name="hardware_solution_extraction",
    )

    # Build verification subtrees for each domain
    await build_neuromorphic_checks(evaluator, root, extraction.neuromorphic)
    await build_photonic_checks(evaluator, root, extraction.photonic)
    await build_chiplet_checks(evaluator, root, extraction.chiplet)
    await build_quantum_checks(evaluator, root, extraction.quantum)
    await build_edge_ai_checks(evaluator, root, extraction.edge_ai)

    # Optional: record a compact summary of selected components
    summary_info = {
        "neuromorphic": {
            "manufacturer": extraction.neuromorphic.manufacturer if extraction.neuromorphic else None,
            "product_model": extraction.neuromorphic.product_model if extraction.neuromorphic else None,
        },
        "photonic": {
            "manufacturer": extraction.photonic.manufacturer if extraction.photonic else None,
            "product_model": extraction.photonic.product_model if extraction.photonic else None,
        },
        "chiplet": {
            "manufacturer": extraction.chiplet.manufacturer if extraction.chiplet else None,
            "product_model": extraction.chiplet.product_model if extraction.chiplet else None,
        },
        "quantum": {
            "manufacturer": extraction.quantum.manufacturer if extraction.quantum else None,
            "product_model": extraction.quantum.product_model if extraction.quantum else None,
        },
        "edge_ai": {
            "manufacturer": extraction.edge_ai.manufacturer if extraction.edge_ai else None,
            "product_model": extraction.edge_ai.product_model if extraction.edge_ai else None,
        },
    }
    evaluator.add_custom_info(summary_info, info_type="component_summary")

    # Return structured result
    return evaluator.get_summary()