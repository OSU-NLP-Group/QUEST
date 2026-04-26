import asyncio
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from obj_task_eval.evaluator import Evaluator
from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.verification_tree import AggregationStrategy


# -----------------------------------------------------------------------------
# Task-specific constants
# -----------------------------------------------------------------------------
TASK_ID = "breakthrough_2025_specs"
TASK_DESCRIPTION = """
A technology research firm is preparing a comprehensive report on breakthrough technologies that achieved major production or commercial milestones in 2025. They need you to compile specific technical specifications for their semiconductor, quantum computing, AI processing, energy storage, and wireless communication sections.

Provide the following information:

1. The name of TSMC's most advanced semiconductor process node that entered volume production in Q4 2025
2. The type of transistor architecture employed in this TSMC process node
3. The model name of IBM's quantum processor that has over 1,000 qubits
4. The exact qubit count of this IBM quantum processor
5. The model name of Qualcomm's laptop processor that features 80 TOPS NPU performance
6. The energy density value (in Wh/kg) achieved by Mercedes-Benz's solid-state battery technology
7. The upper frequency limit (in GHz) of 5G millimeter wave (mmWave) Frequency Range 2 (FR2) bands

Each specification must be accurate and verifiable through official sources or credible technology publications.
""".strip()


# -----------------------------------------------------------------------------
# Data models for extraction
# -----------------------------------------------------------------------------
class TSMCSpec(BaseModel):
    process_node_name: Optional[str] = None
    transistor_architecture: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class IBMQuantumSpec(BaseModel):
    model_name: Optional[str] = None
    qubit_count: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class QualcommSpec(BaseModel):
    laptop_processor_model: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class MercedesSSBSpec(BaseModel):
    energy_density_wh_per_kg: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class FR2Spec(BaseModel):
    upper_frequency_limit_ghz: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class AllSpecs(BaseModel):
    tsmc: Optional[TSMCSpec] = None
    ibm: Optional[IBMQuantumSpec] = None
    qualcomm: Optional[QualcommSpec] = None
    mercedes: Optional[MercedesSSBSpec] = None
    fr2: Optional[FR2Spec] = None


# -----------------------------------------------------------------------------
# Extraction prompt
# -----------------------------------------------------------------------------
def prompt_extract_specs() -> str:
    return """
    Extract the requested 2025 breakthrough-technology specifications exactly as stated in the answer text. For each category below, extract both the value(s) and all supporting source URLs that the answer explicitly provides.

    Return a JSON with the following structure:
    {
      "tsmc": {
        "process_node_name": string | null,
        "transistor_architecture": string | null,
        "sources": string[]     // all URLs the answer provides that support the TSMC items
      },
      "ibm": {
        "model_name": string | null,
        "qubit_count": string | null,
        "sources": string[]     // URLs supporting the IBM items
      },
      "qualcomm": {
        "laptop_processor_model": string | null,
        "sources": string[]     // URLs supporting the Qualcomm item
      },
      "mercedes": {
        "energy_density_wh_per_kg": string | null,
        "sources": string[]     // URLs supporting the Mercedes solid-state battery item
      },
      "fr2": {
        "upper_frequency_limit_ghz": string | null,
        "sources": string[]     // URLs supporting the FR2 upper frequency limit
      }
    }

    Important instructions:
    - Only extract information explicitly present in the answer.
    - For URLs, extract the actual links (including protocol). Accept URLs shown as markdown links.
    - If a field is missing in the answer, set it to null (or [] for arrays).
    - Keep numbers as strings exactly as written in the answer (e.g., "1,121" or "52.6").
    - Do not infer or invent any values or URLs.
    """.strip()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _has_text(value: Optional[str]) -> bool:
    return value is not None and isinstance(value, str) and value.strip() != ""


def _has_sources(urls: Optional[List[str]]) -> bool:
    return bool(urls) and isinstance(urls, list) and len(urls) > 0


# -----------------------------------------------------------------------------
# Verification subtrees
# -----------------------------------------------------------------------------
async def build_tsmc_section(evaluator: Evaluator, parent) -> None:
    """
    TSMC Q4 2025 volume-production process node specifications
    """
    # Section node (critical)
    tsmc_node = evaluator.add_parallel(
        id="tsmc_section",
        desc="TSMC Q4 2025 volume-production process node specifications",
        parent=parent,
        critical=True
    )

    # Load extracted
    extracted: AllSpecs = evaluator._extraction_results[-1]["result"] if evaluator._extraction_results else {}
    # The extractor records a dict with all fields; we can reconstruct Pydantic models, but we only need dict access
    # Safely map to structure
    tsmc = extracted.get("tsmc") if isinstance(extracted, dict) else None
    if isinstance(tsmc, dict):
        tsmc_name = tsmc.get("process_node_name")
        tsmc_arch = tsmc.get("transistor_architecture")
        tsmc_sources = tsmc.get("sources", [])
    else:
        tsmc_name = None
        tsmc_arch = None
        tsmc_sources = []

    # Existence gates (critical siblings)
    evaluator.add_custom_node(
        result=_has_text(tsmc_name),
        id="tsmc_process_node_name_exists",
        desc="TSMC process node name is provided in the answer",
        parent=tsmc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(tsmc_arch),
        id="tsmc_transistor_architecture_exists",
        desc="TSMC transistor architecture is provided in the answer",
        parent=tsmc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(tsmc_sources),
        id="tsmc_sources_present",
        desc="TSMC specifications include at least one supporting source URL",
        parent=tsmc_node,
        critical=True
    )

    # Leaf: process node name (volume production in Q4 2025)
    tsmc_name_leaf = evaluator.add_leaf(
        id="tsmc_process_node_name",
        desc="Provide the name of TSMC's most advanced process node that entered volume production in Q4 2025",
        parent=tsmc_node,
        critical=True
    )
    claim_name = f"TSMC began volume (mass) production for its {tsmc_name or ''} semiconductor process node during Q4 2025 (October–December 2025)."
    await evaluator.verify(
        claim=claim_name,
        node=tsmc_name_leaf,
        sources=tsmc_sources,
        additional_instruction="Focus on confirming that this named process node entered volume/mass production in Q4 2025 (Oct–Dec 2025). It's acceptable if the source uses synonyms like 'mass production' or 'HVM'. The phrase 'most advanced' does not need to be explicitly present if the timeline clearly identifies the node."
    )

    # Leaf: transistor architecture
    tsmc_arch_leaf = evaluator.add_leaf(
        id="tsmc_transistor_architecture",
        desc="Provide the transistor architecture used in that TSMC process node",
        parent=tsmc_node,
        critical=True
    )
    claim_arch = f"The transistor architecture used in TSMC's {tsmc_name or ''} process node is {tsmc_arch or ''}."
    await evaluator.verify(
        claim=claim_arch,
        node=tsmc_arch_leaf,
        sources=tsmc_sources,
        additional_instruction="Accept standard naming variants for transistor architectures, e.g., 'GAAFET', 'gate-all-around', 'nanosheet', or 'FinFET'. Minor wording differences are acceptable as long as the meaning is the same."
    )


async def build_ibm_section(evaluator: Evaluator, parent) -> None:
    """
    IBM quantum processor (>1,000 qubits) specifications
    """
    ibm_node = evaluator.add_parallel(
        id="ibm_quantum_section",
        desc="IBM quantum processor (>1,000 qubits) specifications",
        parent=parent,
        critical=True
    )

    extracted: AllSpecs = evaluator._extraction_results[-1]["result"] if evaluator._extraction_results else {}
    ibm = extracted.get("ibm") if isinstance(extracted, dict) else None
    if isinstance(ibm, dict):
        model = ibm.get("model_name")
        qubits = ibm.get("qubit_count")
        sources = ibm.get("sources", [])
    else:
        model, qubits, sources = None, None, []

    evaluator.add_custom_node(
        result=_has_text(model),
        id="ibm_model_exists",
        desc="IBM quantum processor model name is provided in the answer",
        parent=ibm_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_text(qubits),
        id="ibm_qubits_exists",
        desc="IBM quantum processor qubit count is provided in the answer",
        parent=ibm_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="ibm_sources_present",
        desc="IBM quantum specification includes at least one supporting source URL",
        parent=ibm_node,
        critical=True
    )

    # Leaf: model (>1,000 qubits)
    model_leaf = evaluator.add_leaf(
        id="ibm_quantum_processor_model",
        desc="Provide the model name of IBM's quantum processor that has over 1,000 qubits",
        parent=ibm_node,
        critical=True
    )
    claim_model = f"IBM has a quantum processor model named {model or ''} that has over 1,000 qubits."
    await evaluator.verify(
        claim=claim_model,
        node=model_leaf,
        sources=sources,
        additional_instruction="Confirm that this specific IBM quantum processor model name exists and is associated with a qubit count greater than 1,000."
    )

    # Leaf: exact qubit count
    qubits_leaf = evaluator.add_leaf(
        id="ibm_quantum_processor_qubit_count",
        desc="Provide the exact qubit count of the IBM processor identified as having over 1,000 qubits",
        parent=ibm_node,
        critical=True
    )
    claim_qubits = f"IBM's {model or ''} quantum processor has exactly {qubits or ''} qubits."
    await evaluator.verify(
        claim=claim_qubits,
        node=qubits_leaf,
        sources=sources,
        additional_instruction="Allow numeric formatting variants (e.g., with or without commas). The value should match exactly aside from trivial formatting."
    )


async def build_qualcomm_section(evaluator: Evaluator, parent) -> None:
    """
    Qualcomm laptop processor with 80 TOPS NPU specification
    """
    qc_node = evaluator.add_parallel(
        id="qualcomm_section",
        desc="Qualcomm laptop processor with 80 TOPS NPU specification",
        parent=parent,
        critical=True
    )

    extracted: AllSpecs = evaluator._extraction_results[-1]["result"] if evaluator._extraction_results else {}
    qualcomm = extracted.get("qualcomm") if isinstance(extracted, dict) else None
    if isinstance(qualcomm, dict):
        model = qualcomm.get("laptop_processor_model")
        sources = qualcomm.get("sources", [])
    else:
        model, sources = None, []

    evaluator.add_custom_node(
        result=_has_text(model),
        id="qualcomm_model_exists",
        desc="Qualcomm laptop processor model name is provided in the answer",
        parent=qc_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="qualcomm_sources_present",
        desc="Qualcomm specification includes at least one supporting source URL",
        parent=qc_node,
        critical=True
    )

    # Leaf: model + 80 TOPS NPU
    qc_leaf = evaluator.add_leaf(
        id="qualcomm_laptop_processor_model",
        desc="Provide the model name of Qualcomm's laptop processor that features 80 TOPS NPU performance",
        parent=qc_node,
        critical=True
    )
    claim_qc = f"Qualcomm's laptop processor {model or ''} features an NPU with 80 TOPS performance."
    await evaluator.verify(
        claim=claim_qc,
        node=qc_leaf,
        sources=sources,
        additional_instruction="Confirm that the NPU performance stated is 80 TOPS (allow phrasing like 'up to 80 TOPS'). Verify specifically the NPU metric rather than combined system TOPS."
    )


async def build_mercedes_section(evaluator: Evaluator, parent) -> None:
    """
    Mercedes-Benz solid-state battery energy density specification
    """
    mb_node = evaluator.add_parallel(
        id="mercedes_section",
        desc="Mercedes-Benz solid-state battery energy density specification",
        parent=parent,
        critical=True
    )

    extracted: AllSpecs = evaluator._extraction_results[-1]["result"] if evaluator._extraction_results else {}
    mercedes = extracted.get("mercedes") if isinstance(extracted, dict) else None
    if isinstance(mercedes, dict):
        density = mercedes.get("energy_density_wh_per_kg")
        sources = mercedes.get("sources", [])
    else:
        density, sources = None, []

    evaluator.add_custom_node(
        result=_has_text(density),
        id="mercedes_energy_density_exists",
        desc="Mercedes solid-state battery energy density value is provided in the answer",
        parent=mb_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="mercedes_sources_present",
        desc="Mercedes solid-state battery item includes at least one supporting source URL",
        parent=mb_node,
        critical=True
    )

    # Leaf: energy density Wh/kg
    mb_leaf = evaluator.add_leaf(
        id="mercedes_solid_state_energy_density",
        desc="Provide the energy density value (Wh/kg) achieved by Mercedes-Benz's solid-state battery technology",
        parent=mb_node,
        critical=True
    )
    claim_mb = f"Mercedes-Benz's solid-state battery technology achieved an energy density of {density or ''} Wh/kg."
    await evaluator.verify(
        claim=claim_mb,
        node=mb_leaf,
        sources=sources,
        additional_instruction="Confirm that the stated Wh/kg is presented as an achieved or demonstrated value for Mercedes-Benz's solid-state battery technology. Allow minor rounding differences."
    )


async def build_fr2_section(evaluator: Evaluator, parent) -> None:
    """
    5G mmWave FR2 upper frequency limit specification
    """
    fr2_node = evaluator.add_parallel(
        id="fr2_section",
        desc="5G mmWave FR2 upper frequency limit specification",
        parent=parent,
        critical=True
    )

    extracted: AllSpecs = evaluator._extraction_results[-1]["result"] if evaluator._extraction_results else {}
    fr2 = extracted.get("fr2") if isinstance(extracted, dict) else None
    if isinstance(fr2, dict):
        upper = fr2.get("upper_frequency_limit_ghz")
        sources = fr2.get("sources", [])
    else:
        upper, sources = None, []

    evaluator.add_custom_node(
        result=_has_text(upper),
        id="fr2_upper_limit_exists",
        desc="FR2 upper frequency limit value is provided in the answer",
        parent=fr2_node,
        critical=True
    )
    evaluator.add_custom_node(
        result=_has_sources(sources),
        id="fr2_sources_present",
        desc="FR2 item includes at least one supporting source URL",
        parent=fr2_node,
        critical=True
    )

    fr2_leaf = evaluator.add_leaf(
        id="fr2_upper_frequency_limit",
        desc="Provide the upper frequency limit (in GHz) of 5G millimeter wave (mmWave) Frequency Range 2 (FR2) bands",
        parent=fr2_node,
        critical=True
    )
    claim_fr2 = f"The upper frequency limit of 5G FR2 (mmWave) bands is {upper or ''} GHz."
    await evaluator.verify(
        claim=claim_fr2,
        node=fr2_leaf,
        sources=sources,
        additional_instruction="Confirm the FR2 (Frequency Range 2) upper bound in GHz from credible sources (e.g., 3GPP specs or authoritative technical references). Commonly cited FR2 range is approximately 24.25–52.6 GHz."
    )


async def build_sources_overview_section(evaluator: Evaluator, parent) -> None:
    """
    Each specification is supported with an official source or credible technology publication reference.
    Implemented as individual critical checks for source presence (non-empty URL lists) per item.
    """
    overview_node = evaluator.add_parallel(
        id="sources_and_verifiability",
        desc="Each specification is supported with an official source or a credible technology publication reference (URL/citation)",
        parent=parent,
        critical=True
    )

    extracted: AllSpecs = evaluator._extraction_results[-1]["result"] if evaluator._extraction_results else {}

    def _present(urls: Optional[List[str]]) -> bool:
        return _has_sources(urls)

    # TSMC
    tsmc = extracted.get("tsmc") if isinstance(extracted, dict) else None
    evaluator.add_custom_node(
        result=_present(tsmc.get("sources", []) if isinstance(tsmc, dict) else []),
        id="sources_tsmc_present",
        desc="TSMC specification includes at least one official/credible source URL",
        parent=overview_node,
        critical=True
    )

    # IBM
    ibm = extracted.get("ibm") if isinstance(extracted, dict) else None
    evaluator.add_custom_node(
        result=_present(ibm.get("sources", []) if isinstance(ibm, dict) else []),
        id="sources_ibm_present",
        desc="IBM quantum specification includes at least one official/credible source URL",
        parent=overview_node,
        critical=True
    )

    # Qualcomm
    qualcomm = extracted.get("qualcomm") if isinstance(extracted, dict) else None
    evaluator.add_custom_node(
        result=_present(qualcomm.get("sources", []) if isinstance(qualcomm, dict) else []),
        id="sources_qualcomm_present",
        desc="Qualcomm specification includes at least one official/credible source URL",
        parent=overview_node,
        critical=True
    )

    # Mercedes
    mercedes = extracted.get("mercedes") if isinstance(extracted, dict) else None
    evaluator.add_custom_node(
        result=_present(mercedes.get("sources", []) if isinstance(mercedes, dict) else []),
        id="sources_mercedes_present",
        desc="Mercedes solid-state battery specification includes at least one official/credible source URL",
        parent=overview_node,
        critical=True
    )

    # FR2
    fr2 = extracted.get("fr2") if isinstance(extracted, dict) else None
    evaluator.add_custom_node(
        result=_present(fr2.get("sources", []) if isinstance(fr2, dict) else []),
        id="sources_fr2_present",
        desc="FR2 upper frequency limit specification includes at least one official/credible source URL",
        parent=overview_node,
        critical=True
    )


# -----------------------------------------------------------------------------
# Main evaluation entry point
# -----------------------------------------------------------------------------
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
    Evaluate an answer for 2025 breakthrough-technology specifications.
    """
    # Initialize evaluator (root is non-critical by design, but we'll add critical children to gate overall score)
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

    # Extraction step
    extracted_specs = await evaluator.extract(
        prompt=prompt_extract_specs(),
        template_class=AllSpecs,
        extraction_name="extracted_specs"
    )

    # Build verification tree according to rubric
    # Top-level sections are critical to make the overall evaluation strict
    await build_tsmc_section(evaluator, root)
    await build_ibm_section(evaluator, root)
    await build_qualcomm_section(evaluator, root)
    await build_mercedes_section(evaluator, root)
    await build_fr2_section(evaluator, root)
    await build_sources_overview_section(evaluator, root)

    # Return evaluation summary
    return evaluator.get_summary()