import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from obj_task_eval.utils.cache_filesys import CacheFileSys
from obj_task_eval.evaluator import Evaluator
from obj_task_eval.verification_tree import AggregationStrategy

# --------------------------------------------------------------------------- #
# Task-specific constants                                                     #
# --------------------------------------------------------------------------- #
TASK_ID = "four_paradigms_2024_2026"
TASK_DESCRIPTION = (
    "Between April 2024 and January 2026, four major computing hardware systems representing different computing "
    "paradigms were announced or launched: one neuromorphic computing system, one quantum computing chip, one photonic "
    "processor, and one AI accelerator platform. For each of these four systems, identify: (1) the name of the system or "
    "platform, (2) the organization that developed it, (3) one key performance or efficiency metric with its specific "
    "numerical value, and (4) the deployment status or availability timeline. Additionally, identify which of these four "
    "systems claims the highest energy efficiency improvement factor compared to traditional computing technology."
)

TIME_WINDOW_START = "2024-04-01"
TIME_WINDOW_END = "2026-01-31"
TIME_WINDOW_HUMAN = "between April 1, 2024 and January 31, 2026 (inclusive)"


# --------------------------------------------------------------------------- #
# Data models for structured extraction                                       #
# --------------------------------------------------------------------------- #
class SystemItem(BaseModel):
    name: Optional[str] = None
    organization: Optional[str] = None
    paradigm: Optional[str] = None  # e.g., "neuromorphic", "quantum", "photonic", "ai accelerator"
    announcement_or_launch_date: Optional[str] = None  # any format as in answer, e.g., "Nov 2025"
    key_metric_name: Optional[str] = None
    key_metric_value: Optional[str] = None  # must include a numeric value like "10x", "5 W", "120 TOPS"
    energy_efficiency_improvement_factor: Optional[str] = None  # e.g., "100x vs CPUs", "20x improvement"
    deployment_status_or_timeline: Optional[str] = None
    sources: List[str] = Field(default_factory=list)


class SystemsExtraction(BaseModel):
    neuromorphic: Optional[SystemItem] = None
    quantum: Optional[SystemItem] = None
    photonic: Optional[SystemItem] = None
    ai_accelerator: Optional[SystemItem] = None

    # Energy-efficiency comparison summary provided in the answer
    highest_efficiency_system_name: Optional[str] = None
    highest_efficiency_system_paradigm: Optional[str] = None
    highest_efficiency_factor_value: Optional[str] = None
    highest_efficiency_sources: List[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Extraction prompt                                                           #
# --------------------------------------------------------------------------- #
def prompt_extract_systems() -> str:
    return """
    Extract details for exactly four distinct computing hardware systems mentioned in the answer, one for each paradigm:
    – neuromorphic computing system (1 item)
    – quantum computing chip (1 item)
    – photonic processor (1 item)
    – AI accelerator platform (1 item)

    For each of the four items, extract the following fields:
    1) name: System or platform name (string)
    2) organization: Developing organization/company/institution (string)
    3) paradigm: One of: "neuromorphic", "quantum", "photonic", "ai accelerator" (string)
    4) announcement_or_launch_date: The stated announcement or launch date as given in the answer (string). If multiple dates are given, prefer the primary announcement/launch date.
    5) key_metric_name: One key performance/efficiency metric mentioned (string)
    6) key_metric_value: The specific metric value, which must include numeric characters such as digits or 'x' multipliers (string), e.g., "120 TOPS", "30x", "5 W"
    7) energy_efficiency_improvement_factor: If the answer states an energy-efficiency improvement factor relative to traditional or conventional computing technology (e.g., CPUs/GPUs), extract it (string, e.g., "50x"). If absent, set to null.
    8) deployment_status_or_timeline: Deployment status or availability timeline (string). If absent, set to null.
    9) sources: All URL(s) provided in the answer that support this system (array of strings). Extract only valid URLs.

    Also extract an overall comparison summary as stated in the answer:
    – highest_efficiency_system_name: The system the answer claims has the highest energy-efficiency improvement factor among the four (string)
    – highest_efficiency_system_paradigm: Which of the four paradigms it belongs to (string)
    – highest_efficiency_factor_value: The numeric factor as stated (string, e.g., "100x"). If not clearly stated, set to null.
    – highest_efficiency_sources: URL(s) the answer cites to support this highest-factor claim (array of strings).

    IMPORTANT RULES:
    – Do not invent information. If a field is missing in the answer, set it to null (or empty list for URLs).
    – When multiple candidates are mentioned for a paradigm, pick the first reasonable one clearly associated with that paradigm.
    – For URLs, extract explicit URLs only (including those inside markdown links).
    – Keep field values exactly as they appear in the answer, without reformatting.
    """


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _nonempty(s: Optional[str]) -> bool:
    return bool(s and str(s).strip())


def _has_digits(s: Optional[str]) -> bool:
    if not s:
        return False
    return any(ch.isdigit() for ch in s)


def _safe_sources(item: Optional[SystemItem]) -> List[str]:
    return [] if item is None else (item.sources or [])


def _system_label(kind: str) -> str:
    mapping = {
        "neuromorphic": "neuromorphic computing system",
        "quantum": "quantum computing chip",
        "photonic": "photonic processor",
        "ai_accelerator": "AI accelerator platform",
    }
    return mapping.get(kind, kind)


# --------------------------------------------------------------------------- #
# Verification for a single system                                            #
# --------------------------------------------------------------------------- #
async def verify_system(
    evaluator: Evaluator,
    parent_node,
    item: Optional[SystemItem],
    kind: str,
) -> None:
    """
    Build verification subtree for one system type under the critical 'systems_extraction' node.
    All leaves under this system are critical checks, per rubric.
    """
    sys_prefix = {
        "neuromorphic": "neuromorphic",
        "quantum": "quantum",
        "photonic": "photonic",
        "ai_accelerator": "ai"
    }[kind]

    sys_node = evaluator.add_parallel(
        id=f"{kind}_system",
        desc=f"{_system_label(kind).capitalize()} (one item) satisfies all required fields and constraints",
        parent=parent_node,
        critical=True  # Parent is critical; children must be critical per framework constraints
    )

    name_val = item.name if item else None
    org_val = item.organization if item else None
    date_val = item.announcement_or_launch_date if item else None
    metric_name = item.key_metric_name if item else None
    metric_value = item.key_metric_value if item else None
    deploy_val = item.deployment_status_or_timeline if item else None
    sources_list = _safe_sources(item)

    # 1) Name provided (critical)
    evaluator.add_custom_node(
        result=_nonempty(name_val),
        id=f"{sys_prefix}_name_provided",
        desc=f"Provides the {_system_label(kind)} name",
        parent=sys_node,
        critical=True
    )

    # 2) Organization provided (critical)
    evaluator.add_custom_node(
        result=_nonempty(org_val),
        id=f"{sys_prefix}_org_provided",
        desc=f"Provides the developing organization for the {_system_label(kind)}",
        parent=sys_node,
        critical=True
    )

    # 3) Date in range (critical) – verify against sources if available
    date_node = evaluator.add_leaf(
        id=f"{sys_prefix}_date_in_range",
        desc=f"Provides an announcement/launch date for the {_system_label(kind)} and it falls between April 2024 and January 2026 (inclusive)",
        parent=sys_node,
        critical=True
    )
    date_claim = (
        f"The {_system_label(kind)} '{name_val or '[name missing]'}' was announced or launched on "
        f"'{date_val or '[no date provided]'}', and this date falls {TIME_WINDOW_HUMAN}."
    )
    await evaluator.verify(
        claim=date_claim,
        node=date_node,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            f"Judge Incorrect if the answer does not explicitly state a date, or if the stated date is outside the "
            f"time window {TIME_WINDOW_HUMAN}. If multiple dates appear on a page, use the primary announcement/launch date."
        )
    )

    # 4) Paradigm match (critical) – verify against sources
    paradigm_node = evaluator.add_leaf(
        id=f"{sys_prefix}_paradigm_match",
        desc=f"The identified system is correctly characterized as {_system_label(kind)}",
        parent=sys_node,
        critical=True
    )
    paradigm_claim = f"The system '{name_val or '[name missing]'}' is a {_system_label(kind)}."
    await evaluator.verify(
        claim=paradigm_claim,
        node=paradigm_node,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Confirm that the page explicitly characterizes the hardware as the specified paradigm. "
            "Allow closely related synonyms (e.g., neuromorphic/spiking neural network hardware; "
            "quantum computing chip; photonic/optical processor; AI accelerator platform)."
        )
    )

    # 5) Metric with numeric value (critical) – verify value exists and is stated on a source
    metric_node = evaluator.add_leaf(
        id=f"{sys_prefix}_metric_numeric",
        desc=f"Provides one key performance or efficiency metric for the {_system_label(kind)} with a specific numerical value (publicly disclosed)",
        parent=sys_node,
        critical=True
    )
    metric_claim = (
        f"The answer provides a key metric for '{name_val or '[name missing]'}' — "
        f"'{metric_name or '[metric name missing]'}' with the specific value '{metric_value or '[value missing]'}', "
        "and the cited source explicitly states this metric value."
    )
    await evaluator.verify(
        claim=metric_claim,
        node=metric_node,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Judge Incorrect if no numeric characters appear in the provided value, or if the value cannot be found on the page. "
            "Numbers with units (e.g., 120 TOPS, 5 W, 20x) are acceptable."
        )
    )

    # 6) Deployment timeline/status provided (critical) – presence check
    evaluator.add_custom_node(
        result=_nonempty(deploy_val),
        id=f"{sys_prefix}_deployment_timeline",
        desc=f"Provides deployment status or an availability timeline for the {_system_label(kind)}",
        parent=sys_node,
        critical=True
    )

    # 7) Product or facility constraint (critical) – verify against sources
    product_node = evaluator.add_leaf(
        id=f"{sys_prefix}_product_or_facility_constraint",
        desc="Indicates the system is a commercial product, in production, or deployed at a recognized research facility",
        parent=sys_node,
        critical=True
    )
    product_claim = (
        f"The '{name_val or '[name missing]'}' system is either a commercial product or in production, "
        "or it is deployed at a recognized research facility (e.g., national lab, major university/institute)."
    )
    await evaluator.verify(
        claim=product_claim,
        node=product_node,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "Look for explicit language such as product availability, production status, or evidence of deployment "
            "at a notable/recognized research facility. If none is present, judge Incorrect."
        )
    )

    # 8) Authoritative sources included (critical) – verify by URLs where possible
    sources_node = evaluator.add_leaf(
        id=f"{sys_prefix}_sources_authoritative",
        desc="Includes traceable citation(s) to official company announcement(s) or otherwise authoritative source(s) supporting the system claims",
        parent=sys_node,
        critical=True
    )
    sources_claim = (
        f"The answer includes at least one URL source for '{name_val or '[name missing]'}' that is either an official company "
        "announcement/newsroom/blog page or a recognized authoritative source (e.g., major research institution or reputable trade/tech outlet) "
        "and that supports the system details."
    )
    # If no URLs, still call verify (simple) so the node gets explicit failed status
    await evaluator.verify(
        claim=sources_claim,
        node=sources_node,
        sources=sources_list if sources_list else None,
        additional_instruction=(
            "If no URL is provided, judge Incorrect. If URL(s) exist, verify that at least one is an official domain (company, "
            "newsroom, press release, official blog) or a recognized authoritative source and that it supports the system details."
        )
    )


# --------------------------------------------------------------------------- #
# Energy-efficiency comparison verification                                   #
# --------------------------------------------------------------------------- #
async def verify_energy_efficiency_comparison(
    evaluator: Evaluator,
    parent_node,
    extraction: SystemsExtraction
) -> None:
    cmp_node = evaluator.add_parallel(
        id="energy_efficiency_comparison",
        desc="Correctly identifies which of the four systems claims the highest energy-efficiency improvement factor vs traditional computing technology",
        parent=parent_node,
        critical=False
    )

    # Build convenience variables
    neu = extraction.neuromorphic
    qua = extraction.quantum
    pho = extraction.photonic
    ai = extraction.ai_accelerator

    # 1) The comparison uses numeric improvement factors and attributes them to systems
    uses_factors_leaf = evaluator.add_leaf(
        id="comparison_uses_improvement_factors",
        desc="States the energy-efficiency improvement factor(s) used for comparison as numeric factors relative to traditional computing technology and attributes them to the relevant systems",
        parent=cmp_node,
        critical=False
    )

    factors_summary = (
        f"neuromorphic: '{neu.energy_efficiency_improvement_factor}' | "
        f"quantum: '{qua.energy_efficiency_improvement_factor}' | "
        f"photonic: '{pho.energy_efficiency_improvement_factor}' | "
        f"ai accelerator: '{ai.energy_efficiency_improvement_factor}'"
        if neu and qua and pho and ai else
        "One or more systems missing factor values."
    )
    uses_claim = (
        "The answer states energy-efficiency improvement factor(s) as numeric values (e.g., 'x' multipliers or percentages) "
        "relative to traditional/conventional computing technology (e.g., CPUs/GPUs), and attributes them to the relevant systems. "
        f"Extracted summary: {factors_summary}"
    )
    await evaluator.verify(
        claim=uses_claim,
        node=uses_factors_leaf,
        additional_instruction=(
            "Judge Correct if for each of the four systems there is a numeric improvement factor explicitly tied to that system "
            "and it is stated relative to traditional/conventional computing. If any system lacks such a factor, judge Incorrect."
        )
    )

    # 2) Comparison restricted to the four identified systems
    restricted_leaf = evaluator.add_leaf(
        id="comparison_restricted_to_four",
        desc="The comparison is made only among the four systems identified in the response",
        parent=cmp_node,
        critical=False
    )
    four_names = [
        neu.name if neu else None,
        qua.name if qua else None,
        pho.name if pho else None,
        ai.name if ai else None
    ]
    restricted_claim = (
        f"The energy-efficiency comparison in the answer considers only these four systems and no other systems: {four_names}."
    )
    await evaluator.verify(
        claim=restricted_claim,
        node=restricted_leaf,
        additional_instruction=(
            "Use only the answer text to judge whether the comparison is restricted to the four identified systems."
        )
    )

    # 3) Highest factor correctly selected
    highest_leaf = evaluator.add_leaf(
        id="highest_factor_correctly_selected",
        desc="Selects the system with the highest claimed energy-efficiency improvement factor consistent with the provided numeric factors and cited sources",
        parent=cmp_node,
        critical=False
    )
    highest_name = extraction.highest_efficiency_system_name or "[missing]"
    highest_paradigm = extraction.highest_efficiency_system_paradigm or "[missing]"
    highest_value = extraction.highest_efficiency_factor_value or "[missing]"

    # Build a comparative summary for the claim
    summary_list = []
    if neu:
        summary_list.append(f"Neuromorphic '{neu.name}': {neu.energy_efficiency_improvement_factor}")
    if qua:
        summary_list.append(f"Quantum '{qua.name}': {qua.energy_efficiency_improvement_factor}")
    if pho:
        summary_list.append(f"Photonic '{pho.name}': {pho.energy_efficiency_improvement_factor}")
    if ai:
        summary_list.append(f"AI accelerator '{ai.name}': {ai.energy_efficiency_improvement_factor}")
    comparison_summary = "; ".join([s for s in summary_list if s])

    highest_claim = (
        f"Given the factors stated in the answer — {comparison_summary} — the system with the highest claimed "
        f"energy-efficiency improvement factor is '{highest_name}' (paradigm: {highest_paradigm}) with value '{highest_value}'."
    )

    all_cmp_urls: List[str] = []
    if extraction.highest_efficiency_sources:
        all_cmp_urls.extend(extraction.highest_efficiency_sources)
    # Also include all systems' sources in case the 'highest' claim references them implicitly
    for sys_item in [neu, qua, pho, ai]:
        if sys_item and sys_item.sources:
            all_cmp_urls.extend([u for u in sys_item.sources if u])

    await evaluator.verify(
        claim=highest_claim,
        node=highest_leaf,
        sources=all_cmp_urls if all_cmp_urls else None,
        additional_instruction=(
            "Use the numeric values as stated in the answer to determine the maximum. If the 'highest' label does not match the "
            "largest numeric factor among the four systems, judge Incorrect. If evidence is ambiguous or missing, judge Incorrect."
        )
    )


# --------------------------------------------------------------------------- #
# Systems distinctness verification                                           #
# --------------------------------------------------------------------------- #
async def verify_systems_distinct(evaluator: Evaluator, parent_node, extraction: SystemsExtraction) -> None:
    node = evaluator.add_leaf(
        id="systems_are_distinct",
        desc="The four identified systems are distinct items (not the same system repeated under multiple paradigms)",
        parent=parent_node,
        critical=True
    )
    names = [
        extraction.neuromorphic.name if extraction.neuromorphic else None,
        extraction.quantum.name if extraction.quantum else None,
        extraction.photonic.name if extraction.photonic else None,
        extraction.ai_accelerator.name if extraction.ai_accelerator else None
    ]
    orgs = [
        extraction.neuromorphic.organization if extraction.neuromorphic else None,
        extraction.quantum.organization if extraction.quantum else None,
        extraction.photonic.organization if extraction.photonic else None,
        extraction.ai_accelerator.organization if extraction.ai_accelerator else None
    ]
    claim = (
        f"The four systems listed are distinct items (not duplicates or the same system under multiple paradigms). "
        f"Names: {names}; Organizations: {orgs}."
    )
    await evaluator.verify(
        claim=claim,
        node=node,
        additional_instruction="If any two items appear to refer to the same system, judge Incorrect."
    )


# --------------------------------------------------------------------------- #
# Main evaluation                                                             #
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
    # Initialize evaluator with root sequential aggregation
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
        default_model=model
    )

    # Record ground-truth constraints (time window)
    evaluator.add_ground_truth({
        "time_window_start": TIME_WINDOW_START,
        "time_window_end": TIME_WINDOW_END,
        "time_window_human": TIME_WINDOW_HUMAN
    }, gt_type="constraints")

    # Extract structured information once
    systems_extraction = await evaluator.extract(
        prompt=prompt_extract_systems(),
        template_class=SystemsExtraction,
        extraction_name="systems_extraction"
    )

    # Build 'systems_extraction' critical node
    systems_node = evaluator.add_parallel(
        id="systems_extraction",
        desc="Extract and substantiate four distinct systems (neuromorphic, quantum chip, photonic processor, AI accelerator) with required attributes and constraints",
        parent=root,
        critical=True  # Critical parent; all direct children must be critical per framework constraints
    )

    # For each required paradigm, add its verification tree (mark each node as critical since parent is critical)
    await verify_system(evaluator, systems_node, systems_extraction.neuromorphic, "neuromorphic")
    await verify_system(evaluator, systems_node, systems_extraction.quantum, "quantum")
    await verify_system(evaluator, systems_node, systems_extraction.photonic, "photonic")
    await verify_system(evaluator, systems_node, systems_extraction.ai_accelerator, "ai_accelerator")

    # Distinctness check (critical)
    await verify_systems_distinct(evaluator, systems_node, systems_extraction)

    # Energy-efficiency comparison (second step under root; will be skipped automatically if systems_extraction fails)
    await verify_energy_efficiency_comparison(evaluator, root, systems_extraction)

    # Return evaluator summary
    return evaluator.get_summary()