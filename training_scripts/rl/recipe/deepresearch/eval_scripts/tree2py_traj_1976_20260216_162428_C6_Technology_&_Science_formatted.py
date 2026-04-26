import asyncio
import logging
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy


# --------------------------------------------------------------------------- #
# Task constants                                                              #
# --------------------------------------------------------------------------- #
TASK_ID = "hybrid_quantum_ai_facility_spec"
TASK_DESCRIPTION = (
    "A technology research consortium is planning to establish a hybrid quantum-AI research facility in the United States "
    "to host IBM's next-generation fault-tolerant quantum computer systems (based on IBM's 2029 Starling roadmap). "
    "Document the comprehensive technical infrastructure specifications this facility must meet, including: "
    "(1) cryogenic infrastructure requirements for quantum computing systems, "
    "(2) data center energy efficiency standards to achieve industry-leading performance, "
    "(3) green building certification requirements and point thresholds, and "
    "(4) AI accelerator infrastructure specifications for co-located classical computing systems. "
    "Each specification must be supported by official documentation or industry standards."
)


# --------------------------------------------------------------------------- #
# Data models for extraction                                                  #
# --------------------------------------------------------------------------- #
class StarlingSpecs(BaseModel):
    starling_refs: List[str] = Field(default_factory=list)
    logical_qubit_capacity: Optional[str] = None
    gate_count_capability: Optional[str] = None
    target_timeline: Optional[str] = None
    error_correction_code_type: Optional[str] = None
    physical_to_logical_ratio: Optional[str] = None
    error_correction_refs: List[str] = Field(default_factory=list)


class CryogenicSpecs(BaseModel):
    cryo_refs: List[str] = Field(default_factory=list)
    base_temperature_range: Optional[str] = None
    cooling_power_requirements: Optional[str] = None
    fridge_specs: Optional[str] = None
    fridge_refs: List[str] = Field(default_factory=list)


class PhysicalFacilitySpecs(BaseModel):
    physical_refs: List[str] = Field(default_factory=list)
    space_requirements: Optional[str] = None
    environmental_controls: Optional[str] = None


class EnergyEfficiencySpecs(BaseModel):
    pue_refs: List[str] = Field(default_factory=list)
    industry_leading_pue_value: Optional[str] = None
    hyperscale_standard_pue: Optional[str] = None
    pue_methodology: Optional[str] = None
    power_delivery_specs: Optional[str] = None
    backup_power_systems: Optional[str] = None
    power_infra_refs: List[str] = Field(default_factory=list)


class GreenBuildingSpecs(BaseModel):
    leed_refs: List[str] = Field(default_factory=list)
    certified_threshold: Optional[str] = None
    silver_threshold: Optional[str] = None
    gold_threshold: Optional[str] = None
    platinum_threshold: Optional[str] = None
    rating_system_selection: Optional[str] = None
    gross_floor_area_requirement: Optional[str] = None
    energy_atmosphere_criteria: Optional[str] = None
    data_center_specific_refs: List[str] = Field(default_factory=list)
    alt_cert_refs: List[str] = Field(default_factory=list)
    green_globes: Optional[str] = None
    iso_standards: Optional[str] = None


class AIInfraSpecs(BaseModel):
    ai_accel_refs: List[str] = Field(default_factory=list)
    gpu_specs: Optional[str] = None
    tpu_specs: Optional[str] = None
    custom_asic_options: Optional[str] = None
    processing_throughput: Optional[str] = None
    memory_specifications: Optional[str] = None
    power_delivery_for_ai: Optional[str] = None
    cooling_capacity: Optional[str] = None
    integration_considerations: Optional[str] = None
    network_internal_connectivity: Optional[str] = None
    network_external_bandwidth: Optional[str] = None
    network_latency_requirements: Optional[str] = None
    network_refs: List[str] = Field(default_factory=list)


class FacilitySpecExtraction(BaseModel):
    starling: Optional[StarlingSpecs] = None
    cryogenic: Optional[CryogenicSpecs] = None
    physical_facility: Optional[PhysicalFacilitySpecs] = None
    energy_efficiency: Optional[EnergyEfficiencySpecs] = None
    green_building: Optional[GreenBuildingSpecs] = None
    ai_infrastructure: Optional[AIInfraSpecs] = None


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_facility_specs() -> str:
    return """
    Extract the comprehensive technical infrastructure specifications from the answer for a hybrid quantum-AI research facility intended to host IBM's next-generation fault-tolerant quantum computers (IBM's 2029 Starling roadmap). 
    For each section, extract both the specific values/descriptions stated in the answer and all supporting URLs explicitly cited in the answer text. 
    Return the following JSON structure. If a field is not present in the answer, set it to null (for single value fields) or an empty array (for URL arrays).

    {
      "starling": {
        "starling_refs": [list of URLs that reference IBM Starling roadmap or official announcements],
        "logical_qubit_capacity": "text for target logical qubit count (e.g., '200 logical qubits')",
        "gate_count_capability": "text for target gate count capability (e.g., '100 million gates')",
        "target_timeline": "text for target deployment timeline (e.g., '2029')",
        "error_correction_code_type": "text describing error correction code type (e.g., 'bivariate bicycle codes', 'qLDPC')",
        "physical_to_logical_ratio": "text for physical-to-logical qubit encoding ratio/parameters (e.g., 'X physical per 1 logical')",
        "error_correction_refs": [list of URLs that support error correction architecture statements]
      },
      "cryogenic": {
        "cryo_refs": [list of URLs for superconducting qubit cryogenic requirements],
        "base_temperature_range": "text for base operating temperature range (e.g., '6–50 mK', '10 mK')",
        "cooling_power_requirements": "text for cooling power specs (e.g., 'X μW at 10 mK')",
        "fridge_specs": "text for dilution refrigerator technical specs",
        "fridge_refs": [list of URLs referencing dilution refrigerator specifications]
      },
      "physical_facility": {
        "physical_refs": [list of URLs for facility physical requirements],
        "space_requirements": "text describing physical space/footprint needs",
        "environmental_controls": "text describing vibration isolation, EMI shielding, etc."
      },
      "energy_efficiency": {
        "pue_refs": [list of URLs for PUE benchmarks/standards],
        "industry_leading_pue_value": "text for industry-leading PUE value or range (e.g., '1.09–1.2')",
        "hyperscale_standard_pue": "text for hyperscale DC PUE standards/targets",
        "pue_methodology": "text for PUE measurement methodology (e.g., 'TTM')",
        "power_delivery_specs": "text for power delivery capacity/redundancy",
        "backup_power_systems": "text for backup power and UPS requirements",
        "power_infra_refs": [list of URLs supporting power infrastructure specs]
      },
      "green_building": {
        "leed_refs": [list of URLs for official LEED requirements],
        "certified_threshold": "text for LEED Certified (e.g., '40–49 points')",
        "silver_threshold": "text for LEED Silver (e.g., '50–59 points')",
        "gold_threshold": "text for LEED Gold (e.g., '60–79 points')",
        "platinum_threshold": "text for LEED Platinum (e.g., '80+ points')",
        "rating_system_selection": "text for appropriate rating system (e.g., 'LEED BD+C: Data Centers')",
        "gross_floor_area_requirement": "text for gross floor area completion requirement (e.g., '60% complete')",
        "energy_atmosphere_criteria": "text mentioning Energy and Atmosphere category criteria",
        "data_center_specific_refs": [list of URLs specifically about LEED for data centers],
        "alt_cert_refs": [list of URLs for alternative certifications like Green Globes, ISO standards]",
        "green_globes": "text mentioning Green Globes for data centers (if any)",
        "iso_standards": "text mentioning relevant ISO standards (e.g., 'ISO 50001')"
      },
      "ai_infrastructure": {
        "ai_accel_refs": [list of URLs for AI accelerator hardware specs (GPU/TPU/ASIC)],
        "gpu_specs": "text for GPU options/specifications",
        "tpu_specs": "text for TPU architecture/specifications",
        "custom_asic_options": "text for custom ASIC accelerator options",
        "processing_throughput": "text for throughput requirements/benchmarks",
        "memory_specifications": "text for memory capacity/bandwidth",
        "power_delivery_for_ai": "text for power delivery requirements for high-density AI compute",
        "cooling_capacity": "text for cooling capacity requirements for AI",
        "integration_considerations": "text about integration with quantum infrastructure",
        "network_internal_connectivity": "text for internal network connectivity",
        "network_external_bandwidth": "text for external network bandwidth/connectivity",
        "network_latency_requirements": "text for network latency requirements",
        "network_refs": [list of URLs for AI network requirements/specs]
      }
    }

    Special rules:
    - Only extract URLs explicitly present in the answer (plain or markdown). Keep full URLs with protocol.
    - Do not invent values or sources. If not present, use null (value fields) or empty lists (URL arrays).
    - Preserve units and ranges exactly as written in the answer text.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _combine_sources(*lists: List[str]) -> List[str]:
    seen = set()
    combined: List[str] = []
    for lst in lists:
        for url in lst or []:
            if url and url not in seen:
                seen.add(url)
                combined.append(url)
    return combined


def _is_nonempty_text(s: Optional[str]) -> bool:
    return bool(s and s.strip())


async def _verify_or_fail_due_to_missing_sources(
    evaluator: Evaluator,
    leaf_id: str,
    desc: str,
    parent,
    critical: bool,
    claim: str,
    sources: List[str],
    add_ins: str
):
    node = evaluator.add_leaf(id=leaf_id, desc=desc, parent=parent, critical=critical)
    if sources and len(sources) > 0:
        await evaluator.verify(
            claim=claim,
            node=node,
            sources=sources,
            additional_instruction=add_ins
        )
    else:
        node.score = 0.0
        node.status = "failed"
    return node


# --------------------------------------------------------------------------- #
# Verification subtrees                                                       #
# --------------------------------------------------------------------------- #
async def build_quantum_section(evaluator: Evaluator, parent_node, spec: FacilitySpecExtraction):
    quantum_node = evaluator.add_parallel(
        id="Quantum_Computing_Infrastructure_Specifications",
        desc="Complete quantum computing infrastructure requirements are specified",
        parent=parent_node,
        critical=False
    )

    # IBM Starling system specifications (sequential)
    starling_node = evaluator.add_sequential(
        id="IBM_Starling_System_Specifications",
        desc="IBM Starling quantum computer system specifications from 2029 roadmap are documented",
        parent=quantum_node,
        critical=False
    )
    starling = spec.starling or StarlingSpecs()

    # Critical existence: Starling official references present
    evaluator.add_custom_node(
        result=len(starling.starling_refs) > 0,
        id="Starling_Reference_Documentation",
        desc="Official IBM documentation or announcement for Starling system is referenced with URL",
        parent=starling_node,
        critical=True
    )

    # Performance targets (parallel)
    perf_node = evaluator.add_parallel(
        id="Quantum_Performance_Targets",
        desc="Key performance targets from IBM roadmap are specified",
        parent=starling_node,
        critical=False
    )
    # Logical qubit capacity (critical)
    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Logical_Qubit_Capacity",
        desc="Target logical qubit count for Starling system (200 logical qubits) is specified",
        parent=perf_node,
        critical=True,
        claim=f"The IBM Starling system target logical qubit capacity is {starling.logical_qubit_capacity or '[unspecified]'} (expected approximately 200 logical qubits).",
        sources=starling.starling_refs,
        add_ins="Use IBM official Starling roadmap/announcement pages. Treat minor phrasing variations as equivalent. Mark incorrect if the sources do not support the stated capacity or if sources are missing."
    )
    # Gate count capability (critical)
    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Gate_Count_Capability",
        desc="Target quantum gate count capability (100 million gates) is specified",
        parent=perf_node,
        critical=True,
        claim=f"The IBM Starling system target gate count capability is {starling.gate_count_capability or '[unspecified]'} (expected around 100 million gates).",
        sources=starling.starling_refs,
        add_ins="Verify the gate count capability against IBM sources. Allow approximate phrasing (e.g., '100M'). Mark incorrect if unsupported or missing sources."
    )
    # Target timeline (critical)
    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Target_Timeline",
        desc="Target deployment timeline (2029) is specified",
        parent=perf_node,
        critical=True,
        claim=f"The target deployment timeline for IBM Starling is {starling.target_timeline or '[unspecified]'} (expected 2029).",
        sources=starling.starling_refs,
        add_ins="Confirm the deployment year using IBM's roadmap page. Allow minor wording differences. Mark incorrect if sources don't support the stated year or if sources are missing."
    )

    # Error correction architecture (parallel, non-critical)
    ec_node = evaluator.add_parallel(
        id="Error_Correction_Architecture",
        desc="Quantum error correction architecture specifications are provided",
        parent=starling_node,
        critical=False
    )
    ec_sources = _combine_sources(starling.starling_refs, starling.error_correction_refs)

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Error_Correction_Code_Type",
        desc="Type of error correction code (bivariate bicycle codes or qLDPC) is specified",
        parent=ec_node,
        critical=False,
        claim=f"The error correction code type is {starling.error_correction_code_type or '[unspecified]'} (e.g., bivariate bicycle codes or qLDPC).",
        sources=ec_sources,
        add_ins="Check IBM or referenced technical documents for the specified code type. Accept equivalent terminology (e.g., qLDPC variants)."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Physical_to_Logical_Qubit_Encoding",
        desc="Physical to logical qubit encoding ratio or parameters are provided",
        parent=ec_node,
        critical=False,
        claim=f"The physical-to-logical qubit encoding ratio/parameters are {starling.physical_to_logical_ratio or '[unspecified]'} according to the references.",
        sources=ec_sources,
        add_ins="Verify that the encoding ratio/parameters are consistent with the referenced IBM or peer-reviewed materials."
    )

    # Cryogenic infrastructure requirements (sequential)
    cryo_node = evaluator.add_sequential(
        id="Cryogenic_Infrastructure_Requirements",
        desc="Cryogenic system requirements for quantum computing are specified",
        parent=quantum_node,
        critical=False
    )
    cryo = spec.cryogenic or CryogenicSpecs()

    # Critical existence: Cryogenic references present
    evaluator.add_custom_node(
        result=len(cryo.cryo_refs) > 0,
        id="Cryogenic_Reference_Documentation",
        desc="Technical documentation for quantum computing cryogenic requirements is referenced with URL",
        parent=cryo_node,
        critical=True
    )

    # Operating temperature specifications (parallel)
    temp_node = evaluator.add_parallel(
        id="Operating_Temperature_Specifications",
        desc="Operating temperature specifications for superconducting qubits are provided",
        parent=cryo_node,
        critical=False
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Base_Temperature_Range",
        desc="Base operating temperature in millikelvin range (approximately 6-50 mK) is specified",
        parent=temp_node,
        critical=True,
        claim=f"The base operating temperature for superconducting qubits is {cryo.base_temperature_range or '[unspecified]'}, which should be approximately within 6–50 mK.",
        sources=cryo.cryo_refs,
        add_ins="Confirm base temperature range using cryogenic references (e.g., dilution refrigerator specs). Allow reasonable approximations. Mark incorrect if unsupported."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Cooling_Power_Requirements",
        desc="Cooling power specifications for dilution refrigerator are provided",
        parent=temp_node,
        critical=False,
        claim=f"The cooling power specifications for the dilution refrigerator are {cryo.cooling_power_requirements or '[unspecified]'} as per the references.",
        sources=_combine_sources(cryo.cryo_refs, cryo.fridge_refs),
        add_ins="Verify cooling power data (e.g., μW at specific mK) against technical documentation."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Dilution_Refrigerator_Specifications",
        desc="Dilution refrigerator technical specifications or requirements are documented",
        parent=cryo_node,
        critical=False,
        claim=f"Dilution refrigerator technical specifications include {cryo.fridge_specs or '[unspecified]'} according to the references.",
        sources=_combine_sources(cryo.cryo_refs, cryo.fridge_refs),
        add_ins="Check that the referenced documents explicitly state the specified refrigerator technical parameters."
    )

    # Physical facility requirements (parallel, non-critical)
    phys_node = evaluator.add_parallel(
        id="Physical_Facility_Requirements",
        desc="Physical infrastructure requirements for quantum computing facility are specified",
        parent=quantum_node,
        critical=False
    )
    phys = spec.physical_facility or PhysicalFacilitySpecs()

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Space_Requirements",
        desc="Physical space requirements or footprint specifications are provided",
        parent=phys_node,
        critical=False,
        claim=f"Physical space/footprint requirements are {phys.space_requirements or '[unspecified]'} according to the references.",
        sources=phys.physical_refs,
        add_ins="Verify space/footprint details from the cited facility planning documents or vendor specifications."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Environmental_Controls",
        desc="Environmental control requirements (vibration isolation, EMI shielding, etc.) are specified",
        parent=phys_node,
        critical=False,
        claim=f"Environmental control requirements (e.g., vibration isolation, EMI shielding) include {phys.environmental_controls or '[unspecified]'} according to the references.",
        sources=phys.physical_refs,
        add_ins="Confirm environmental control requirements against facility design or vendor references."
    )


async def build_energy_section(evaluator: Evaluator, parent_node, spec: FacilitySpecExtraction):
    energy_node = evaluator.add_parallel(
        id="Energy_Efficiency_Standards",
        desc="Data center energy efficiency standards and requirements are documented",
        parent=parent_node,
        critical=False
    )
    energy = spec.energy_efficiency or EnergyEfficiencySpecs()

    # PUE requirements (sequential)
    pue_node = evaluator.add_sequential(
        id="PUE_Requirements",
        desc="Power Usage Effectiveness (PUE) requirements and benchmarks are specified",
        parent=energy_node,
        critical=False
    )

    # Critical existence: PUE references present
    evaluator.add_custom_node(
        result=len(energy.pue_refs) > 0,
        id="PUE_Reference_Documentation",
        desc="Industry standards or benchmarks for data center PUE are referenced with URL",
        parent=pue_node,
        critical=True
    )

    # Industry benchmark identification (parallel)
    bench_node = evaluator.add_parallel(
        id="Industry_Benchmark_Identification",
        desc="Industry-leading PUE benchmarks are identified",
        parent=pue_node,
        critical=False
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Industry_Leading_PUE_Value",
        desc="Industry-leading PUE value (approximately 1.09-1.2 range) is specified",
        parent=bench_node,
        critical=True,
        claim=f"The industry-leading PUE value/range is {energy.industry_leading_pue_value or '[unspecified]'}, expected approximately 1.09–1.2.",
        sources=energy.pue_refs,
        add_ins="Verify PUE benchmarks from industry-standard sources (e.g., hyperscale operator sustainability reports). Allow close values (e.g., 1.1)."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Hyperscale_Standard_PUE",
        desc="Hyperscale data center PUE standards or targets are provided",
        parent=bench_node,
        critical=False,
        claim=f"Hyperscale data center PUE standards/targets are {energy.hyperscale_standard_pue or '[unspecified]'} according to the references.",
        sources=energy.pue_refs,
        add_ins="Confirm hyperscale target/standard values using referenced hyperscale operator documentation."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="PUE_Measurement_Methodology",
        desc="PUE measurement methodology (e.g., trailing twelve-month) is specified",
        parent=bench_node,
        critical=False,
        claim=f"PUE measurement methodology is {energy.pue_methodology or '[unspecified]'} (e.g., trailing twelve-month).",
        sources=energy.pue_refs,
        add_ins="Check that references explicitly state the PUE measurement methodology."
    )

    # Power infrastructure requirements (parallel, non-critical)
    power_node = evaluator.add_parallel(
        id="Power_Infrastructure_Requirements",
        desc="Electrical power infrastructure requirements are specified",
        parent=energy_node,
        critical=False
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Power_Delivery_Specifications",
        desc="Power delivery capacity and redundancy specifications are provided",
        parent=power_node,
        critical=False,
        claim=f"Power delivery capacity and redundancy specifications are {energy.power_delivery_specs or '[unspecified]'} according to the references.",
        sources=_combine_sources(energy.power_infra_refs, energy.pue_refs),
        add_ins="Verify power delivery and redundancy specs (e.g., MW capacity, N+1) against referenced standards or operator documents."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Backup_Power_Systems",
        desc="Backup power and UPS requirements are specified",
        parent=power_node,
        critical=False,
        claim=f"Backup power and UPS requirements are {energy.backup_power_systems or '[unspecified]'} according to the references.",
        sources=_combine_sources(energy.power_infra_refs, energy.pue_refs),
        add_ins="Confirm UPS/generator/energy storage requirements from the referenced standards or operator materials."
    )


async def build_green_section(evaluator: Evaluator, parent_node, spec: FacilitySpecExtraction):
    green_node = evaluator.add_sequential(
        id="Green_Building_Certification_Requirements",
        desc="Green building certification requirements and standards are documented",
        parent=parent_node,
        critical=False
    )
    gb = spec.green_building or GreenBuildingSpecs()

    # LEED certification requirements (sequential)
    leed_node = evaluator.add_sequential(
        id="LEED_Certification_Requirements",
        desc="LEED certification requirements for data centers are specified",
        parent=green_node,
        critical=False
    )

    # Critical existence: LEED references present
    evaluator.add_custom_node(
        result=len(gb.leed_refs) > 0,
        id="LEED_Reference_Documentation",
        desc="Official LEED certification standards and requirements are referenced with URL",
        parent=leed_node,
        critical=True
    )

    # Certification levels and thresholds (parallel, non-critical children)
    thresh_node = evaluator.add_parallel(
        id="Certification_Levels_and_Thresholds",
        desc="LEED certification levels and point thresholds are documented",
        parent=leed_node,
        critical=False
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Certified_Level_Threshold",
        desc="LEED Certified level point threshold (40-49 points) is specified",
        parent=thresh_node,
        critical=False,
        claim=f"LEED Certified level point threshold is {gb.certified_threshold or '[unspecified]'} (expected 40–49 points).",
        sources=gb.leed_refs,
        add_ins="Verify point thresholds on official LEED documentation. Accept minor formatting differences."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Silver_Level_Threshold",
        desc="LEED Silver level point threshold (50-59 points) is specified",
        parent=thresh_node,
        critical=False,
        claim=f"LEED Silver level point threshold is {gb.silver_threshold or '[unspecified]'} (expected 50–59 points).",
        sources=gb.leed_refs,
        add_ins="Confirm Silver threshold from official LEED sources."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Gold_Level_Threshold",
        desc="LEED Gold level point threshold (60-79 points) is specified",
        parent=thresh_node,
        critical=False,
        claim=f"LEED Gold level point threshold is {gb.gold_threshold or '[unspecified]'} (expected 60–79 points).",
        sources=gb.leed_refs,
        add_ins="Confirm Gold threshold from official LEED sources."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Platinum_Level_Threshold",
        desc="LEED Platinum level point threshold (80+ points) is specified",
        parent=thresh_node,
        critical=False,
        claim=f"LEED Platinum level point threshold is {gb.platinum_threshold or '[unspecified]'} (expected 80+ points).",
        sources=gb.leed_refs,
        add_ins="Confirm Platinum threshold from official LEED sources."
    )

    # Data center specific LEED requirements (parallel)
    dc_node = evaluator.add_parallel(
        id="Data_Center_Specific_Requirements",
        desc="Data center specific LEED requirements are documented",
        parent=leed_node,
        critical=False
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Rating_System_Selection",
        desc="Appropriate LEED rating system for data centers (BD+C: Data Centers) is identified",
        parent=dc_node,
        critical=True,
        claim=f"The appropriate LEED rating system is {gb.rating_system_selection or '[unspecified]'} (expected 'BD+C: Data Centers').",
        sources=_combine_sources(gb.leed_refs, gb.data_center_specific_refs),
        add_ins="Verify that the LEED rating system for data centers is correctly identified as BD+C: Data Centers."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Gross_Floor_Area_Requirement",
        desc="LEED requirement for gross floor area completion (60%) is specified",
        parent=dc_node,
        critical=False,
        claim=f"LEED gross floor area completion requirement is {gb.gross_floor_area_requirement or '[unspecified]'} (e.g., 60%).",
        sources=_combine_sources(gb.leed_refs, gb.data_center_specific_refs),
        add_ins="Verify gross floor area completion requirement from LEED sources."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Energy_and_Atmosphere_Criteria",
        desc="Energy and Atmosphere category requirements or criteria are mentioned",
        parent=dc_node,
        critical=False,
        claim=f"Energy and Atmosphere category requirements/criteria are {gb.energy_atmosphere_criteria or '[unspecified]'} as documented.",
        sources=_combine_sources(gb.leed_refs, gb.data_center_specific_refs),
        add_ins="Confirm EA criteria references in LEED documentation."
    )

    # Alternative certification options (parallel, non-critical)
    alt_node = evaluator.add_parallel(
        id="Alternative_Certification_Options",
        desc="Alternative green building certification options are identified",
        parent=green_node,
        critical=False
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Green_Globes_Certification",
        desc="Green Globes certification option for data centers is mentioned",
        parent=alt_node,
        critical=False,
        claim=f"Green Globes certification option is mentioned as {gb.green_globes or '[unspecified]'} for data centers.",
        sources=gb.alt_cert_refs,
        add_ins="Verify references mentioning Green Globes for data centers."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="ISO_Standards",
        desc="Relevant ISO standards (e.g., ISO 50001 for energy management) are identified",
        parent=alt_node,
        critical=False,
        claim=f"Relevant ISO standards are identified as {gb.iso_standards or '[unspecified]'} (e.g., ISO 50001 for energy management).",
        sources=gb.alt_cert_refs,
        add_ins="Confirm ISO standards references (e.g., 50001) supporting energy management certification."
    )


async def build_ai_section(evaluator: Evaluator, parent_node, spec: FacilitySpecExtraction):
    ai_node = evaluator.add_parallel(
        id="AI_Infrastructure_Specifications",
        desc="AI accelerator and supporting infrastructure specifications are documented",
        parent=parent_node,
        critical=False
    )
    ai = spec.ai_infrastructure or AIInfraSpecs()

    # AI accelerator requirements (sequential)
    accel_node = evaluator.add_sequential(
        id="AI_Accelerator_Requirements",
        desc="AI accelerator hardware requirements are specified",
        parent=ai_node,
        critical=False
    )

    # Critical existence: AI accelerator references present
    evaluator.add_custom_node(
        result=len(ai.ai_accel_refs) > 0,
        id="AI_Accelerator_Reference_Documentation",
        desc="Technical documentation for AI accelerators (GPU/TPU/ASIC) is referenced with URL",
        parent=accel_node,
        critical=True
    )

    # Processor architecture options (parallel)
    arch_node = evaluator.add_parallel(
        id="Processor_Architecture_Options",
        desc="AI accelerator processor architecture options are identified",
        parent=accel_node,
        critical=False
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="GPU_Specifications",
        desc="GPU options or specifications for AI workloads are provided",
        parent=arch_node,
        critical=False,
        claim=f"GPU options/specifications for AI workloads are {ai.gpu_specs or '[unspecified]'} according to the references.",
        sources=ai.ai_accel_refs,
        add_ins="Verify GPU specifications (e.g., HBM capacity, FLOPS) using vendor/official documents."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="TPU_Specifications",
        desc="TPU architecture or specifications are provided",
        parent=arch_node,
        critical=False,
        claim=f"TPU architecture/specifications are {ai.tpu_specs or '[unspecified]'} according to the references.",
        sources=ai.ai_accel_refs,
        add_ins="Confirm TPU specifications using official documentation."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Custom_ASIC_Options",
        desc="Custom ASIC accelerator options are mentioned",
        parent=arch_node,
        critical=False,
        claim=f"Custom ASIC accelerator options are {ai.custom_asic_options or '[unspecified]'} as referenced.",
        sources=ai.ai_accel_refs,
        add_ins="Verify mention of custom ASIC options using cited documents."
    )

    # Performance metrics (parallel)
    perf_node = evaluator.add_parallel(
        id="Performance_Metrics",
        desc="AI accelerator performance metrics or requirements are specified",
        parent=accel_node,
        critical=False
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Processing_Throughput",
        desc="Processing throughput requirements or benchmarks are provided",
        parent=perf_node,
        critical=False,
        claim=f"Processing throughput requirements/benchmarks are {ai.processing_throughput or '[unspecified]'} according to the references.",
        sources=ai.ai_accel_refs,
        add_ins="Verify throughput benchmarks (e.g., TFLOPS, tokens/sec) using vendor/benchmark documents."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Memory_Specifications",
        desc="Memory capacity and bandwidth specifications are provided",
        parent=perf_node,
        critical=False,
        claim=f"Memory capacity/bandwidth specifications are {ai.memory_specifications or '[unspecified]'} according to the references.",
        sources=ai.ai_accel_refs,
        add_ins="Verify HBM capacity, bandwidth, or equivalent memory specs in official sources."
    )

    # Power and cooling infrastructure (parallel)
    pc_node = evaluator.add_parallel(
        id="Power_and_Cooling_Infrastructure",
        desc="Power and cooling infrastructure for AI systems are specified",
        parent=ai_node,
        critical=False
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Power_Delivery_for_AI",
        desc="Power delivery requirements for high-density AI compute are specified",
        parent=pc_node,
        critical=False,
        claim=f"Power delivery requirements for high-density AI compute are {ai.power_delivery_for_ai or '[unspecified]'} according to the references.",
        sources=ai.ai_accel_refs,
        add_ins="Confirm power delivery requirements (e.g., kW/rack, busway specs)."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Cooling_Capacity",
        desc="Cooling capacity requirements for AI infrastructure are provided",
        parent=pc_node,
        critical=False,
        claim=f"Cooling capacity requirements for AI infrastructure are {ai.cooling_capacity or '[unspecified]'} according to the references.",
        sources=ai.ai_accel_refs,
        add_ins="Verify cooling capacity and approaches (e.g., liquid cooling) using vendor/industry documents."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Integration_Considerations",
        desc="Integration considerations with quantum computing infrastructure are mentioned",
        parent=pc_node,
        critical=False,
        claim=f"Integration considerations with quantum computing infrastructure include {ai.integration_considerations or '[unspecified]'} according to the references.",
        sources=ai.ai_accel_refs,
        add_ins="Confirm integration considerations from referenced facility planning or vendor documents."
    )

    # Network infrastructure (parallel)
    net_node = evaluator.add_parallel(
        id="Network_Infrastructure",
        desc="Network infrastructure requirements are specified",
        parent=ai_node,
        critical=False
    )
    net_sources = _combine_sources(ai.ai_accel_refs, ai.network_refs)

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Internal_Connectivity",
        desc="Internal network connectivity requirements are provided",
        parent=net_node,
        critical=False,
        claim=f"Internal network connectivity requirements are {ai.network_internal_connectivity or '[unspecified]'} according to the references.",
        sources=net_sources,
        add_ins="Verify internal fabric requirements (e.g., Infiniband, Ethernet) and topology using references."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="External_Network_Specifications",
        desc="External network bandwidth and connectivity specifications are provided",
        parent=net_node,
        critical=False,
        claim=f"External network bandwidth/connectivity specifications are {ai.network_external_bandwidth or '[unspecified]'} according to the references.",
        sources=net_sources,
        add_ins="Confirm WAN/peering/bandwidth requirements using the cited documents."
    )

    await _verify_or_fail_due_to_missing_sources(
        evaluator,
        leaf_id="Latency_Requirements",
        desc="Network latency requirements for AI workloads are specified",
        parent=net_node,
        critical=False,
        claim=f"Network latency requirements for AI workloads are {ai.network_latency_requirements or '[unspecified]'} according to the references.",
        sources=net_sources,
        add_ins="Verify latency requirements or SLAs from referenced materials."
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
    model: str = "o4-mini"
) -> Dict:
    # Initialize evaluator (root as non-critical parallel to allow partial credit and avoid critical constraints)
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
        default_model=model
    )

    # Extract structured specifications from answer
    extracted_specs = await evaluator.extract(
        prompt=prompt_extract_facility_specs(),
        template_class=FacilitySpecExtraction,
        extraction_name="facility_specs_extraction"
    )

    # Add ground truth info for reference expectations (not used for scoring directly)
    evaluator.add_ground_truth({
        "ibm_starling_expected": {
            "logical_qubits": "≈200",
            "gate_count": "≈100 million",
            "timeline": "2029"
        },
        "cryogenic_expected": {
            "base_temperature_range_mK": "≈6–50 mK"
        },
        "energy_efficiency_expected": {
            "industry_leading_pue_range": "≈1.09–1.2"
        },
        "leed_expected_thresholds": {
            "Certified": "40–49",
            "Silver": "50–59",
            "Gold": "60–79",
            "Platinum": "80+",
            "Rating_System": "BD+C: Data Centers"
        }
    }, gt_type="expected_norms")

    # Build sections under root per rubric
    await build_quantum_section(evaluator, root, extracted_specs)
    await build_energy_section(evaluator, root, extracted_specs)
    await build_green_section(evaluator, root, extracted_specs)
    await build_ai_section(evaluator, root, extracted_specs)

    # Return summary
    return evaluator.get_summary()